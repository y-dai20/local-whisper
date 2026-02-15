use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;

use futures_util::{SinkExt, StreamExt};
use once_cell::sync::{Lazy, OnceCell};
use parking_lot::Mutex;
use tauri::AppHandle;
use tokio::runtime::{Builder, Runtime};
use tokio::sync::mpsc as tokio_mpsc;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

use super::TranscriptionSource;
use crate::audio::processing::save_audio_session_to_wav;
use crate::audio::state::try_recording_state;
use crate::emit_transcription_segment;

#[derive(Debug, Clone)]
pub struct WebSocketTranscriptEvent {
    pub session_id: Option<String>,
    pub chunk_index: Option<u64>,
    pub start_seconds: Option<f64>,
    pub end_seconds: Option<f64>,
    pub text: Option<String>,
    pub is_final: Option<bool>,
}

struct SourceConnection {
    outbound_tx: tokio_mpsc::UnboundedSender<Message>,
    transcript_rx: Receiver<WebSocketTranscriptEvent>,
    unhealthy: Arc<AtomicBool>,
    buffered_audio: Vec<f32>,
    buffered_start_sample: u64,
    total_input_samples: u64,
}

struct WebSocketTranscriptionClient {
    runtime: Runtime,
    connections: HashMap<TranscriptionSource, SourceConnection>,
}

static CLIENT: OnceCell<Mutex<WebSocketTranscriptionClient>> = OnceCell::new();
static REMOTE_SESSION_ID_MAP: Lazy<Mutex<HashMap<String, u64>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static NEXT_REMOTE_SESSION_ID: Lazy<Mutex<u64>> = Lazy::new(|| Mutex::new(2_000_000));
const SAMPLE_RATE: u64 = 16_000;

pub fn stream_audio_chunk_and_emit(
    audio_16k_f32: &[f32],
    _language: &str,
    source: TranscriptionSource,
    session_id_counter: u64,
    is_final: bool,
    app_handle: &AppHandle,
) -> Result<(), String> {
    let events = send_audio_to_websocket(audio_16k_f32, source, is_final)?;

    for (event, event_audio) in events {
        let Some(text) = event.text.as_ref().map(|t| t.trim()).filter(|t| !t.is_empty()) else {
            continue;
        };

        let message_id = event.chunk_index.unwrap_or(0);
        let event_is_final = event.is_final.unwrap_or(is_final);
        let mapped_session_id =
            resolve_remote_session_id(event.session_id.as_deref(), session_id_counter);

        emit_transcription_segment(
            app_handle,
            text.to_string(),
            event_audio.clone(),
            mapped_session_id,
            message_id,
            event_is_final,
            source.event_source().to_string(),
        )?;

        if event_is_final {
            if let Some(audio) = event_audio.as_deref() {
                persist_api_audio_if_needed(audio, mapped_session_id, message_id, source);
            }
        }
    }

    Ok(())
}

pub fn reset_all_connections() {
    if let Some(client) = CLIENT.get() {
        let mut guard = client.lock();
        log::info!(
            "WebSocket: resetting {} active source connection(s)",
            guard.connections.len()
        );
        guard.connections.clear();
    }
    REMOTE_SESSION_ID_MAP.lock().clear();
}

fn send_audio_to_websocket(
    audio_16k_f32: &[f32],
    source: TranscriptionSource,
    is_final: bool,
) -> Result<Vec<(WebSocketTranscriptEvent, Option<Vec<f32>>)>, String> {
    if audio_16k_f32.is_empty() {
        return Ok(Vec::new());
    }

    let client = CLIENT.get_or_try_init(|| {
        let runtime = Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("Failed to create websocket runtime: {}", e))?;
        Ok::<_, String>(Mutex::new(WebSocketTranscriptionClient {
            runtime,
            connections: HashMap::new(),
        }))
    })?;

    let mut guard = client.lock();
    if !guard.connections.contains_key(&source) {
        log::info!(
            "WebSocket({}): no existing connection, creating one",
            source.event_source()
        );
        let conn = guard.runtime.block_on(create_connection(source))?;
        guard.connections.insert(source, conn);
        log::info!(
            "WebSocket({}): transport connected, waiting for session_started",
            source.event_source()
        );
    }

    let conn = guard
        .connections
        .get_mut(&source)
        .ok_or_else(|| "WebSocket connection not found".to_string())?;

    if conn.unhealthy.load(Ordering::Relaxed) {
        log::warn!(
            "WebSocket({}): connection unhealthy, dropping and requiring reconnect",
            source.event_source()
        );
        guard.connections.remove(&source);
        return Err("WebSocket connection became unhealthy".to_string());
    }

    let pcm = f32_to_i16(audio_16k_f32);
    conn.outbound_tx
        .send(Message::Binary(i16_to_le_bytes(&pcm).into()))
        .map_err(|e| format!("Failed to queue WS PCM binary: {}", e))?;

    if is_final {
        conn.outbound_tx
            .send(Message::Text("flush".to_string().into()))
            .map_err(|e| format!("Failed to queue WS flush command: {}", e))?;
    }

    conn.buffered_audio.extend_from_slice(audio_16k_f32);
    conn.total_input_samples = conn
        .total_input_samples
        .wrapping_add(audio_16k_f32.len() as u64);

    drain_events(conn)
}

async fn create_connection(source: TranscriptionSource) -> Result<SourceConnection, String> {
    let url = websocket_url();
    log::info!(
        "WebSocket({}): connecting to {}",
        source.event_source(),
        url
    );
    let (ws, _) = connect_async(&url)
        .await
        .map_err(|e| format!("Failed to connect websocket {}: {}", url, e))?;
    log::info!(
        "WebSocket({}): transport connected to {}",
        source.event_source(),
        url
    );

    let (mut write, mut read) = ws.split();
    let (outbound_tx, mut outbound_rx) = tokio_mpsc::unbounded_channel::<Message>();
    let (tx, rx): (Sender<WebSocketTranscriptEvent>, Receiver<WebSocketTranscriptEvent>) =
        mpsc::channel();
    let unhealthy = Arc::new(AtomicBool::new(false));
    let session_id_holder = Arc::new(Mutex::new(None::<String>));

    let unhealthy_write = unhealthy.clone();
    let source_name_write = source.event_source().to_string();
    tokio::spawn(async move {
        while let Some(msg) = outbound_rx.recv().await {
            if let Err(err) = write.send(msg).await {
                log::warn!("WebSocket({}): writer failed: {}", source_name_write, err);
                unhealthy_write.store(true, Ordering::Relaxed);
                break;
            }
        }
    });

    let unhealthy_read = unhealthy.clone();
    let session_id_read = session_id_holder.clone();
    let source_name_read = source.event_source().to_string();
    tokio::spawn(async move {
        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    handle_text_message(&text, &tx, &session_id_read, &source_name_read);
                }
                Ok(Message::Binary(bin)) => {
                    if let Ok(text) = String::from_utf8(bin.to_vec()) {
                        handle_text_message(&text, &tx, &session_id_read, &source_name_read);
                    }
                }
                Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => {}
                Ok(Message::Close(_)) => {
                    log::warn!("WebSocket({}): received close frame", source_name_read);
                    unhealthy_read.store(true, Ordering::Relaxed);
                    break;
                }
                Err(err) => {
                    log::warn!("WebSocket({}): reader failed: {}", source_name_read, err);
                    unhealthy_read.store(true, Ordering::Relaxed);
                    break;
                }
                _ => {}
            }
        }
    });

    Ok(SourceConnection {
        outbound_tx,
        transcript_rx: rx,
        unhealthy,
        buffered_audio: Vec::new(),
        buffered_start_sample: 0,
        total_input_samples: 0,
    })
}

fn handle_text_message(
    text: &str,
    tx: &Sender<WebSocketTranscriptEvent>,
    session_id_holder: &Arc<Mutex<Option<String>>>,
    source_name: &str,
) {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(text) else {
        return;
    };
    let Some(kind) = value.get("type").and_then(|v| v.as_str()) else {
        return;
    };

    match kind {
        "session_started" => {
            if let Some(session_id) = value.get("session_id").and_then(|v| v.as_str()) {
                *session_id_holder.lock() = Some(session_id.to_string());
                log::info!(
                    "WebSocket({}): session started (session_id={})",
                    source_name,
                    session_id
                );
            }
        }
        "transcript" => {
            if let Some(event) = transcript_event_from_value(&value, &session_id_holder.lock()) {
                let _ = tx.send(event);
            }
        }
        _ => {}
    }
}

fn websocket_url() -> String {
    if let Ok(url) = std::env::var("LOCAL_WHISPER_WS_URL") {
        let trimmed = url.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }

    let base = std::env::var("LOCAL_WHISPER_API_BASE_URL")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "http://127.0.0.1:8000".to_string());
    let base = if let Some(rest) = base.strip_prefix("https://") {
        format!("wss://{}", rest)
    } else if let Some(rest) = base.strip_prefix("http://") {
        format!("ws://{}", rest)
    } else {
        base
    };
    let path = std::env::var("LOCAL_WHISPER_WS_PATH")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "/api/ws/audio".to_string());
    format!("{}{}", base.trim_end_matches('/'), path)
}

fn drain_events(
    conn: &mut SourceConnection,
) -> Result<Vec<(WebSocketTranscriptEvent, Option<Vec<f32>>)>, String> {
    let mut events = Vec::new();
    loop {
        match conn.transcript_rx.try_recv() {
            Ok(event) => {
                let event_audio = if event.is_final.unwrap_or(false) {
                    extract_audio_from_time_range(conn, event.start_seconds, event.end_seconds)
                } else {
                    None
                };
                events.push((event, event_audio));
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                return Err("WebSocket transcript channel disconnected".to_string());
            }
        }
    }
    Ok(events)
}

fn transcript_event_from_value(
    value: &serde_json::Value,
    default_session_id: &Option<String>,
) -> Option<WebSocketTranscriptEvent> {
    let text = value.get("text").and_then(|v| v.as_str()).map(|s| s.to_string())?;
    if text.trim().is_empty() {
        return None;
    }

    Some(WebSocketTranscriptEvent {
        session_id: value
            .get("session_id")
            .or_else(|| value.get("sessionId"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| default_session_id.clone()),
        chunk_index: value
            .get("chunk_index")
            .or_else(|| value.get("chunkIndex"))
            .and_then(|v| v.as_u64()),
        start_seconds: value
            .get("start_time_sec")
            .and_then(json_number_to_f64),
        end_seconds: value
            .get("end_time_sec")
            .and_then(json_number_to_f64),
        text: Some(text),
        is_final: value
            .get("is_final")
            .or_else(|| value.get("isFinal"))
            .and_then(|v| v.as_bool()),
    })
}

fn json_number_to_f64(value: &serde_json::Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_i64().map(|v| v as f64))
        .or_else(|| value.as_u64().map(|v| v as f64))
}

fn extract_audio_from_time_range(
    conn: &mut SourceConnection,
    start_seconds: Option<f64>,
    end_seconds: Option<f64>,
) -> Option<Vec<f32>> {
    let (start_seconds, end_seconds) = match (start_seconds, end_seconds) {
        (Some(start), Some(end)) if end > start && start >= 0.0 => (start, end),
        _ => return None,
    };

    let start_sample = (start_seconds * SAMPLE_RATE as f64).floor() as u64;
    let end_sample = (end_seconds * SAMPLE_RATE as f64).ceil() as u64;
    if end_sample <= start_sample {
        return None;
    }

    let buffer_end_sample = conn
        .buffered_start_sample
        .wrapping_add(conn.buffered_audio.len() as u64);

    if start_sample < conn.buffered_start_sample || end_sample > buffer_end_sample {
        log::warn!(
            "WebSocket: requested sample range [{}..{}) is outside buffered range [{}..{})",
            start_sample,
            end_sample,
            conn.buffered_start_sample,
            buffer_end_sample
        );
        return None;
    }

    let from = (start_sample - conn.buffered_start_sample) as usize;
    let to = (end_sample - conn.buffered_start_sample) as usize;
    if to <= from {
        return None;
    }

    let extracted = conn.buffered_audio[from..to].to_vec();
    prune_buffer_before_sample(conn, end_sample);
    Some(extracted)
}

fn prune_buffer_before_sample(conn: &mut SourceConnection, sample: u64) {
    if sample <= conn.buffered_start_sample {
        return;
    }

    let buffer_end_sample = conn
        .buffered_start_sample
        .wrapping_add(conn.buffered_audio.len() as u64);
    let prune_to = sample.min(buffer_end_sample);
    if prune_to <= conn.buffered_start_sample {
        return;
    }

    let drop_count = (prune_to - conn.buffered_start_sample) as usize;
    conn.buffered_audio.drain(0..drop_count);
    conn.buffered_start_sample = prune_to;
}

fn f32_to_i16(audio: &[f32]) -> Vec<i16> {
    audio
        .iter()
        .map(|&sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect()
}

fn i16_to_le_bytes(samples: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

fn resolve_remote_session_id(remote_id: Option<&str>, fallback: u64) -> u64 {
    let Some(remote) = remote_id.map(str::trim).filter(|v| !v.is_empty()) else {
        return fallback;
    };
    {
        let map = REMOTE_SESSION_ID_MAP.lock();
        if let Some(id) = map.get(remote) {
            return *id;
        }
    }

    let mut map = REMOTE_SESSION_ID_MAP.lock();
    if let Some(id) = map.get(remote) {
        return *id;
    }
    let mut next = NEXT_REMOTE_SESSION_ID.lock();
    *next = next.wrapping_add(1);
    let assigned = *next;
    map.insert(remote.to_string(), assigned);
    assigned
}

fn persist_api_audio_if_needed(
    audio_16k_f32: &[f32],
    mapped_session_id: u64,
    message_id: u64,
    source: TranscriptionSource,
) {
    let Some(state) = try_recording_state() else {
        return;
    };
    let guard = state.lock();
    if !guard.recording_save_enabled || !guard.is_recording {
        return;
    }
    let Some(recording_dir) = guard.current_recording_dir.clone() else {
        return;
    };
    drop(guard);

    let source_tag = match source {
        TranscriptionSource::Mic => "mic",
        TranscriptionSource::System => "system",
    };
    let synthetic_session_id = mapped_session_id
        .wrapping_mul(1_000_000)
        .wrapping_add(message_id);
    let audio = audio_16k_f32.to_vec();

    std::thread::spawn(move || {
        if let Err(err) =
            save_audio_session_to_wav(&audio, synthetic_session_id, &recording_dir, source_tag)
        {
            log::error!(
                "Failed to save API {} audio for session={}, message={}: {}",
                source_tag,
                mapped_session_id,
                message_id,
                err
            );
        }
    });
}
