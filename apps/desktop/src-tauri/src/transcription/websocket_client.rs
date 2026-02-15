use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;

use futures_util::{SinkExt, StreamExt};
use once_cell::sync::{Lazy, OnceCell};
use parking_lot::Mutex;
use serde::Deserialize;
use tauri::AppHandle;
use tokio::runtime::{Builder, Runtime};
use tokio::sync::mpsc as tokio_mpsc;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

use super::TranscriptionSource;
use crate::emit_transcription_segment;

#[derive(Debug, Deserialize, Clone)]
pub struct WebSocketTranscriptEvent {
    pub session_id: Option<String>,
    pub chunk_index: Option<u64>,
    pub text: Option<String>,
    pub is_final: Option<bool>,
}

struct SourceConnection {
    outbound_tx: tokio_mpsc::UnboundedSender<Message>,
    transcript_rx: Receiver<WebSocketTranscriptEvent>,
    unhealthy: Arc<AtomicBool>,
}

struct WebSocketTranscriptionClient {
    runtime: Runtime,
    connections: HashMap<TranscriptionSource, SourceConnection>,
}

static CLIENT: OnceCell<Mutex<WebSocketTranscriptionClient>> = OnceCell::new();
static REMOTE_SESSION_ID_MAP: Lazy<Mutex<HashMap<String, u64>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static NEXT_REMOTE_SESSION_ID: Lazy<Mutex<u64>> = Lazy::new(|| Mutex::new(2_000_000));

pub fn stream_audio_chunk_and_emit(
    audio_16k_f32: &[f32],
    _language: &str,
    source: TranscriptionSource,
    session_id_counter: u64,
    is_final: bool,
    app_handle: &AppHandle,
) -> Result<(), String> {
    let events = send_audio_to_websocket(audio_16k_f32, source, is_final)?;

    for event in events {
        let Some(text) = event.text.as_ref().map(|t| t.trim()).filter(|t| !t.is_empty()) else {
            continue;
        };
        let message_id = event.chunk_index.unwrap_or(0);
        let event_is_final = event.is_final.unwrap_or(is_final);
        let mapped_session_id = resolve_remote_session_id(event.session_id.as_deref(), session_id_counter);

        emit_transcription_segment(
            app_handle,
            text.to_string(),
            None,
            mapped_session_id,
            message_id,
            event_is_final,
            source.event_source().to_string(),
        )?;
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
) -> Result<Vec<WebSocketTranscriptEvent>, String> {
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
                    handle_text_message(
                        &text,
                        &tx,
                        &session_id_read,
                        &source_name_read,
                    );
                }
                Ok(Message::Binary(bin)) => {
                    if let Ok(text) = String::from_utf8(bin.to_vec()) {
                        handle_text_message(
                            &text,
                            &tx,
                            &session_id_read,
                            &source_name_read,
                        );
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

fn drain_events(conn: &mut SourceConnection) -> Result<Vec<WebSocketTranscriptEvent>, String> {
    let mut events = Vec::new();
    loop {
        match conn.transcript_rx.try_recv() {
            Ok(event) => events.push(event),
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
        text: Some(text),
        is_final: value
            .get("is_final")
            .or_else(|| value.get("isFinal"))
            .and_then(|v| v.as_bool()),
    })
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
