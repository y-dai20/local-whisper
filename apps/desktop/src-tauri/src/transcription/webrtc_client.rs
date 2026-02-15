use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use once_cell::sync::{Lazy, OnceCell};
use opus::{Application as OpusApplication, Channels as OpusChannels, Encoder as OpusEncoder};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::runtime::{Builder, Runtime};
use tauri::AppHandle;
use webrtc::api::interceptor_registry::register_default_interceptors;
use webrtc::api::media_engine::{MediaEngine, MIME_TYPE_OPUS};
use webrtc::api::setting_engine::SettingEngine;
use webrtc::api::APIBuilder;
use webrtc::data_channel::data_channel_message::DataChannelMessage;
use webrtc::data_channel::RTCDataChannel;
use webrtc::ice::network_type::NetworkType;
use webrtc::ice_transport::ice_connection_state::RTCIceConnectionState;
use webrtc::ice_transport::ice_server::RTCIceServer;
use webrtc::interceptor::registry::Registry;
use webrtc::media::Sample;
use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::peer_connection_state::RTCPeerConnectionState;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::peer_connection::RTCPeerConnection;
use webrtc::rtp_transceiver::rtp_codec::RTCRtpCodecCapability;
use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;

use super::TranscriptionSource;
use crate::emit_transcription_segment;

#[derive(Debug, Deserialize, Clone)]
pub struct WebRtcTranscriptEvent {
    #[serde(alias = "sessionId")]
    pub session_id: Option<String>,
    #[serde(alias = "chunkIndex")]
    pub chunk_index: Option<u64>,
    pub text: Option<String>,
    #[serde(alias = "isFinal")]
    pub is_final: Option<bool>,
}

#[derive(Debug, Serialize)]
struct OfferPayload {
    sdp: String,
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Debug, Deserialize)]
struct AnswerPayload {
    sdp: String,
    #[serde(rename = "type")]
    kind: String,
}

struct SourceConnection {
    peer: Arc<RTCPeerConnection>,
    track: Arc<TrackLocalStaticSample>,
    transcript_channel: Option<Arc<RTCDataChannel>>,
    transcript_rx: Receiver<WebRtcTranscriptEvent>,
    pcm_16k_pending: Vec<i16>,
    opus_encoder: OpusEncoder,
    unhealthy: Arc<AtomicBool>,
    connecting: Arc<AtomicBool>,
}

struct WebRtcTranscriptionClient {
    runtime: Runtime,
    connections: HashMap<TranscriptionSource, SourceConnection>,
}

#[derive(Debug, Clone)]
struct RetryCircuitState {
    failure_rounds: usize,
    blocked_until: Option<Instant>,
}

static CLIENT: OnceCell<Mutex<WebRtcTranscriptionClient>> = OnceCell::new();
static REMOTE_SESSION_ID_MAP: Lazy<Mutex<HashMap<String, u64>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static NEXT_REMOTE_SESSION_ID: Lazy<Mutex<u64>> = Lazy::new(|| Mutex::new(1_000_000));
static RETRY_CIRCUIT_MAP: Lazy<Mutex<HashMap<TranscriptionSource, RetryCircuitState>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub fn send_audio_to_webrtc(
    audio_16k_f32: &[f32],
    _language: &str,
    source: TranscriptionSource,
    _session_id: u64,
    _is_final: bool,
) -> Result<Vec<WebRtcTranscriptEvent>, String> {
    if audio_16k_f32.is_empty() {
        return Ok(Vec::new());
    }

    let client = CLIENT.get_or_try_init(|| {
        let runtime = Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("Failed to create WebRTC runtime: {}", e))?;
        Ok::<_, String>(Mutex::new(WebRtcTranscriptionClient {
            runtime,
            connections: HashMap::new(),
        }))
    })?;

    let mut guard = client.lock();
    let handle = guard.runtime.handle().clone();
    let max_attempts = webrtc_max_attempts();
    let mut last_error: Option<String> = None;

    if let Some(remaining) = retry_circuit_remaining(source) {
        log::warn!(
            "WebRTC retry cooldown active for source={:?} ({}ms remaining, active_connections={})",
            source,
            remaining.as_millis(),
            active_connections_summary(&guard.connections)
        );
        return Err(format!(
            "WebRTC retry cooldown active ({}ms remaining)",
            remaining.as_millis()
        ));
    }

    for attempt in 0..max_attempts {
        if guard
            .connections
            .get(&source)
            .map(|conn| conn.unhealthy.load(Ordering::Relaxed))
            .unwrap_or(false)
        {
            log::warn!(
                "Detected unhealthy WebRTC connection for source={:?}, resetting (active_connections={})",
                source,
                active_connections_summary(&guard.connections)
            );
            reset_source_connection(&mut guard, source);
        }

        if !guard.connections.contains_key(&source) {
            let conn = guard.runtime.block_on(create_connection(source));
            match conn {
                Ok(conn) => {
                    guard.connections.insert(source, conn);
                    log::info!(
                        "WebRTC connection established for source={:?} (attempt={}/{}, active_connections={})",
                        source,
                        attempt + 1,
                        max_attempts,
                        active_connections_summary(&guard.connections)
                    );
                }
                Err(err) => {
                    log::warn!(
                        "WebRTC connection attempt failed for source={:?} (attempt={}/{}, active_connections={}): {}",
                        source,
                        attempt + 1,
                        max_attempts,
                        active_connections_summary(&guard.connections),
                        err
                    );
                    last_error = Some(err);
                    if attempt + 1 < max_attempts {
                        thread::sleep(retry_backoff(attempt));
                        continue;
                    }
                    break;
                }
            }
        }

        let result = {
            let conn = guard
                .connections
                .get_mut(&source)
                .ok_or_else(|| "WebRTC connection not found".to_string())?;

            let mut pcm_16k = f32_to_i16(audio_16k_f32);
            conn.pcm_16k_pending.append(&mut pcm_16k);

            if conn.connecting.load(Ordering::Relaxed) {
                if conn.pcm_16k_pending.len() > 16_000 * 5 {
                    let drain_len = conn.pcm_16k_pending.len().saturating_sub(16_000 * 5);
                    conn.pcm_16k_pending.drain(..drain_len);
                }
                return drain_transcript_events(conn);
            }

            let mut packets = Vec::new();
            while conn.pcm_16k_pending.len() >= 320 {
                let frame: Vec<i16> = conn.pcm_16k_pending.drain(..320).collect();
                let mut packet_buf = vec![0u8; 1500];
                let packet_len = conn
                    .opus_encoder
                    .encode(&frame, &mut packet_buf)
                    .map_err(|e| format!("Failed to encode Opus frame: {}", e))?;
                packet_buf.truncate(packet_len);
                packets.push(packet_buf);
            }

            let track = conn.track.clone();
            handle.block_on(async move {
                for packet in packets {
                    track
                        .write_sample(&Sample {
                            data: packet.into(),
                            duration: Duration::from_millis(20),
                            ..Default::default()
                        })
                        .await
                        .map_err(|e| format!("Failed to write WebRTC audio sample: {}", e))?;
                }
                Ok::<(), String>(())
            })
        };

        match result {
            Ok(()) => {
                let mut events = Vec::new();
                loop {
                    let conn = guard
                        .connections
                        .get_mut(&source)
                        .ok_or_else(|| "WebRTC connection not found".to_string())?;
                    match conn.transcript_rx.try_recv() {
                        Ok(event) => events.push(event),
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => {
                            reset_source_connection(&mut guard, source);
                            return Err("WebRTC transcript channel disconnected".to_string());
                        }
                    }
                }
                reset_retry_circuit(source);
                return Ok(events);
            }
            Err(err) => {
                log::warn!(
                    "WebRTC send failed for source={:?} (attempt={}/{}, active_connections={}): {}",
                    source,
                    attempt + 1,
                    max_attempts,
                    active_connections_summary(&guard.connections),
                    err
                );
                last_error = Some(err);
                reset_source_connection(&mut guard, source);
                if attempt + 1 < max_attempts {
                    thread::sleep(retry_backoff(attempt));
                    continue;
                }
                break;
            }
        }
    }

    mark_retry_circuit_failure(source);
    Err(format!(
        "WebRTC send failed after {} attempts: {}",
        max_attempts,
        last_error.unwrap_or_else(|| "unknown error".to_string())
    ))
}

pub fn stream_audio_chunk_and_emit(
    audio_16k_f32: &[f32],
    language: &str,
    source: TranscriptionSource,
    session_id_counter: u64,
    is_final: bool,
    app_handle: &AppHandle,
) -> Result<(), String> {
    let events = send_audio_to_webrtc(
        audio_16k_f32,
        language,
        source,
        session_id_counter,
        is_final,
    )?;

    for event in events {
        let Some(text) = event.text.as_ref().map(|t| t.trim()).filter(|t| !t.is_empty()) else {
            continue;
        };
        let mapped_session_id = resolve_remote_session_id(
            event.session_id.as_deref(),
            session_id_counter,
        );
        let message_id = event.chunk_index.unwrap_or(0);
        let event_is_final = event.is_final.unwrap_or(is_final);

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

async fn create_connection(source: TranscriptionSource) -> Result<SourceConnection, String> {
    let mut media_engine = MediaEngine::default();
    media_engine
        .register_default_codecs()
        .map_err(|e| format!("Failed to register codecs: {}", e))?;

    let mut registry = Registry::new();
    registry = register_default_interceptors(registry, &mut media_engine)
        .map_err(|e| format!("Failed to register interceptors: {}", e))?;

    let mut setting_engine = SettingEngine::default();
    setting_engine.set_network_types(vec![NetworkType::Udp4]);

    let api = APIBuilder::new()
        .with_media_engine(media_engine)
        .with_interceptor_registry(registry)
        .with_setting_engine(setting_engine)
        .build();

    let config = RTCConfiguration {
        ice_servers: vec![RTCIceServer {
            urls: vec!["stun:stun.l.google.com:19302".to_string()],
            ..Default::default()
        }],
        ..Default::default()
    };

    let peer = Arc::new(
        api.new_peer_connection(config)
            .await
            .map_err(|e| format!("Failed to create peer connection: {}", e))?,
    );
    let unhealthy = Arc::new(AtomicBool::new(false));
    let connecting = Arc::new(AtomicBool::new(true));
    {
        let unhealthy_flag = unhealthy.clone();
        let connecting_flag = connecting.clone();
        peer.on_ice_connection_state_change(Box::new(move |state: RTCIceConnectionState| {
            if matches!(
                state,
                RTCIceConnectionState::Failed | RTCIceConnectionState::Closed
            ) {
                unhealthy_flag.store(true, Ordering::Relaxed);
                connecting_flag.store(false, Ordering::Relaxed);
            } else if matches!(state, RTCIceConnectionState::Connected) {
                connecting_flag.store(false, Ordering::Relaxed);
            }
            Box::pin(async {})
        }));
    }
    {
        let unhealthy_flag = unhealthy.clone();
        let connecting_flag = connecting.clone();
        peer.on_peer_connection_state_change(Box::new(move |state: RTCPeerConnectionState| {
            match state {
                RTCPeerConnectionState::Connected => {
                    connecting_flag.store(false, Ordering::Relaxed);
                }
                RTCPeerConnectionState::Connecting => {
                    connecting_flag.store(true, Ordering::Relaxed);
                }
                RTCPeerConnectionState::Failed | RTCPeerConnectionState::Closed => {
                    unhealthy_flag.store(true, Ordering::Relaxed);
                    connecting_flag.store(false, Ordering::Relaxed);
                }
                _ => {}
            }
            Box::pin(async {})
        }));
    }

    let track = Arc::new(TrackLocalStaticSample::new(
        RTCRtpCodecCapability {
            mime_type: MIME_TYPE_OPUS.to_string(),
            clock_rate: 48_000,
            channels: 2,
            ..Default::default()
        },
        "audio".to_string(),
        format!("local-whisper-{}", source.event_source()),
    ));

    let rtp_sender = peer
        .add_track(track.clone())
        .await
        .map_err(|e| format!("Failed to add WebRTC track: {}", e))?;

    tokio::spawn(async move {
        let mut rtcp_buf = vec![0u8; 1500];
        while rtp_sender.read(&mut rtcp_buf).await.is_ok() {}
    });

    let (tx, rx): (Sender<WebRtcTranscriptEvent>, Receiver<WebRtcTranscriptEvent>) = mpsc::channel();
    let tx_on_data = tx.clone();
    peer.on_data_channel(Box::new(move |dc: Arc<RTCDataChannel>| {
        let tx_inner = tx_on_data.clone();
        Box::pin(async move {
            dc.on_message(Box::new(move |msg: DataChannelMessage| {
                let tx_msg = tx_inner.clone();
                Box::pin(async move {
                    if let Ok(text) = std::str::from_utf8(&msg.data) {
                        log::debug!("Raw datachannel payload: {}", text);
                        if let Some(event) = parse_transcript_event(text) {
                            log::debug!(
                                "Received datachannel event: session_id={:?} chunk_index={:?} is_final={:?} text_len={}",
                                event.session_id,
                                event.chunk_index,
                                event.is_final,
                                event.text.as_ref().map(|t| t.len()).unwrap_or(0)
                            );
                            let _ = tx_msg.send(event);
                        } else {
                            log::debug!("Ignored unsupported datachannel payload: {}", text);
                        }
                    }
                })
            }));
        })
    }));

    // Keep label compatible with the previous frontend implementation.
    let transcript_channel = peer
        .create_data_channel("transcript", None)
        .await
        .ok();

    if let Some(dc) = transcript_channel.as_ref() {
        dc.on_open(Box::new(move || {
            Box::pin(async move {
                log::info!("Transcript data channel opened");
            })
        }));
        let tx_inner = tx.clone();
        dc.on_message(Box::new(move |msg: DataChannelMessage| {
            let tx_msg = tx_inner.clone();
            Box::pin(async move {
                if let Ok(text) = std::str::from_utf8(&msg.data) {
                    log::debug!("Raw transcript payload: {}", text);
                    if let Some(event) = parse_transcript_event(text) {
                        log::debug!(
                            "Received transcript event: session_id={:?} chunk_index={:?} is_final={:?} text_len={}",
                            event.session_id,
                            event.chunk_index,
                            event.is_final,
                            event.text.as_ref().map(|t| t.len()).unwrap_or(0)
                        );
                        let _ = tx_msg.send(event);
                    } else {
                        log::debug!("Ignored unsupported transcript payload: {}", text);
                    }
                }
            })
        }));
    }

    let offer = peer
        .create_offer(None)
        .await
        .map_err(|e| format!("Failed to create WebRTC offer: {}", e))?;
    peer.set_local_description(offer)
        .await
        .map_err(|e| format!("Failed to set local description: {}", e))?;
    // Do not wait for full ICE gathering. Send offer immediately to reduce setup latency.

    let local_desc = peer
        .local_description()
        .await
        .ok_or_else(|| "Missing local description".to_string())?;

    let base_url = std::env::var("LOCAL_WHISPER_API_BASE_URL")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "http://127.0.0.1:8000".to_string());
    let endpoint = std::env::var("LOCAL_WHISPER_WEBRTC_OFFER_PATH")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "/api/webrtc/offer".to_string());
    let url = format!("{}{}", base_url.trim_end_matches('/'), endpoint);

    let offer_payload = OfferPayload {
        sdp: local_desc.sdp,
        kind: local_desc.sdp_type.to_string(),
    };

    let answer = reqwest::Client::new()
        .post(&url)
        .json(&offer_payload)
        .send()
        .await
        .map_err(|e| format!("Failed to call WebRTC offer endpoint: {}", e))?;

    if !answer.status().is_success() {
        return Err(format!("WebRTC offer failed with {}", answer.status()));
    }

    let answer_payload: AnswerPayload = answer
        .json()
        .await
        .map_err(|e| format!("Invalid WebRTC answer payload: {}", e))?;

    let remote = RTCSessionDescription::answer(answer_payload.sdp)
        .map_err(|e| format!("Invalid answer SDP: {}", e))?;
    if answer_payload.kind.to_lowercase() != "answer" {
        return Err(format!(
            "Unexpected SDP type from server: {}",
            answer_payload.kind
        ));
    }

    peer.set_remote_description(remote)
        .await
        .map_err(|e| format!("Failed to set remote description: {}", e))?;

    let opus_encoder = OpusEncoder::new(16_000, OpusChannels::Mono, OpusApplication::Voip)
        .map_err(|e| format!("Failed to create Opus encoder: {}", e))?;

    Ok(SourceConnection {
        peer,
        track,
        transcript_channel,
        transcript_rx: rx,
        pcm_16k_pending: Vec::new(),
        opus_encoder,
        unhealthy,
        connecting,
    })
}

fn parse_transcript_event(raw: &str) -> Option<WebRtcTranscriptEvent> {
    if let Ok(event) = serde_json::from_str::<WebRtcTranscriptEvent>(raw) {
        return Some(event);
    }

    let value: Value = serde_json::from_str(raw).ok()?;
    parse_transcript_event_from_value(&value)
}

fn parse_transcript_event_from_value(value: &Value) -> Option<WebRtcTranscriptEvent> {
    if let Ok(event) = serde_json::from_value::<WebRtcTranscriptEvent>(value.clone()) {
        if event.text.as_ref().map(|t| !t.trim().is_empty()).unwrap_or(false) {
            return Some(event);
        }
    }

    let obj = value.as_object()?;

    for key in ["data", "payload", "transcript", "result", "event"] {
        if let Some(v) = obj.get(key) {
            if let Some(event) = parse_transcript_event_from_value(v) {
                return Some(event);
            }
        }
    }

    let text = extract_text(obj)?;
    let session_id = extract_string(obj, &["session_id", "sessionId", "sid"]);
    let chunk_index = extract_u64(obj, &["chunk_index", "chunkIndex", "index"]);
    let is_final = extract_bool(obj, &["is_final", "isFinal", "final"]);

    Some(WebRtcTranscriptEvent {
        session_id,
        chunk_index,
        text: Some(text),
        is_final,
    })
}

fn extract_text(obj: &Map<String, Value>) -> Option<String> {
    for key in ["text", "message", "content", "transcript", "result"] {
        if let Some(v) = obj.get(key) {
            match v {
                Value::String(s) if !s.trim().is_empty() => return Some(s.clone()),
                Value::Array(items) => {
                    let merged = items
                        .iter()
                        .filter_map(|item| item.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    if !merged.trim().is_empty() {
                        return Some(merged);
                    }
                }
                _ => {}
            }
        }
    }
    None
}

fn extract_string(obj: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|k| obj.get(*k))
        .find_map(|v| v.as_str().map(|s| s.to_string()))
}

fn extract_u64(obj: &Map<String, Value>, keys: &[&str]) -> Option<u64> {
    keys.iter().filter_map(|k| obj.get(*k)).find_map(|v| match v {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => s.parse::<u64>().ok(),
        _ => None,
    })
}

fn extract_bool(obj: &Map<String, Value>, keys: &[&str]) -> Option<bool> {
    keys.iter().filter_map(|k| obj.get(*k)).find_map(|v| match v {
        Value::Bool(b) => Some(*b),
        Value::String(s) => match s.to_lowercase().as_str() {
            "true" | "1" => Some(true),
            "false" | "0" => Some(false),
            _ => None,
        },
        _ => None,
    })
}

fn f32_to_i16(audio: &[f32]) -> Vec<i16> {
    audio
        .iter()
        .map(|&sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect()
}

pub fn close_all() {
    if let Some(client) = CLIENT.get() {
        let mut guard = client.lock();
        log::info!(
            "Closing all WebRTC connections (active_connections={})",
            active_connections_summary(&guard.connections)
        );
        let handle = guard.runtime.handle().clone();
        let peers: Vec<_> = guard
            .connections
            .drain()
            .map(|(_, conn)| conn.peer.clone())
            .collect();
        for peer in peers {
            handle.block_on(async move {
                let _ = peer.close().await;
            });
        }
    }
}

fn reset_source_connection(
    guard: &mut parking_lot::MutexGuard<'_, WebRtcTranscriptionClient>,
    source: TranscriptionSource,
) {
    let handle = guard.runtime.handle().clone();
    if let Some(old) = guard.connections.remove(&source) {
        handle.block_on(async move {
            let _ = old.peer.close().await;
        });
        log::info!(
            "Closed WebRTC connection for source={:?} (active_connections={})",
            source,
            active_connections_summary(&guard.connections)
        );
    }
}

pub fn reset_all_connections() {
    close_all();
    REMOTE_SESSION_ID_MAP.lock().clear();
    RETRY_CIRCUIT_MAP.lock().clear();
}

fn env_u64_or(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn env_usize_or(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn webrtc_max_attempts() -> usize {
    env_usize_or("LOCAL_WHISPER_WEBRTC_MAX_ATTEMPTS", 10)
        .max(1)
        .min(10)
}

fn retry_backoff(attempt: usize) -> Duration {
    let base_ms = env_u64_or("LOCAL_WHISPER_WEBRTC_RETRY_BACKOFF_MS", 200);
    let backoff_ms = base_ms.saturating_mul((attempt as u64) + 1);
    Duration::from_millis(backoff_ms)
}

fn retry_circuit_remaining(source: TranscriptionSource) -> Option<Duration> {
    let map = RETRY_CIRCUIT_MAP.lock();
    let state = map.get(&source)?;
    let blocked_until = state.blocked_until?;
    let now = Instant::now();
    if blocked_until > now {
        Some(blocked_until.duration_since(now))
    } else {
        None
    }
}

fn mark_retry_circuit_failure(source: TranscriptionSource) {
    let mut map = RETRY_CIRCUIT_MAP.lock();
    let state = map.entry(source).or_insert(RetryCircuitState {
        failure_rounds: 0,
        blocked_until: None,
    });
    state.failure_rounds = state.failure_rounds.saturating_add(1);
    let cooldown = retry_circuit_cooldown(state.failure_rounds);
    state.blocked_until = Some(Instant::now() + cooldown);
}

fn retry_circuit_cooldown(failure_rounds: usize) -> Duration {
    let base_ms = env_u64_or("LOCAL_WHISPER_WEBRTC_COOLDOWN_BASE_MS", 1_000);
    let max_ms = env_u64_or("LOCAL_WHISPER_WEBRTC_COOLDOWN_MAX_MS", 30_000);
    let factor = failure_rounds.max(1) as u64;
    Duration::from_millis((base_ms.saturating_mul(factor)).min(max_ms))
}

fn reset_retry_circuit(source: TranscriptionSource) {
    RETRY_CIRCUIT_MAP.lock().remove(&source);
}

fn active_connections_summary(
    connections: &HashMap<TranscriptionSource, SourceConnection>,
) -> String {
    let mut labels: Vec<&str> = connections
        .keys()
        .map(|source| match source {
            TranscriptionSource::Mic => "mic",
            TranscriptionSource::System => "system",
        })
        .collect();
    labels.sort_unstable();
    format!("{} [{}]", connections.len(), labels.join(","))
}

fn drain_transcript_events(
    conn: &mut SourceConnection,
) -> Result<Vec<WebRtcTranscriptEvent>, String> {
    let mut events = Vec::new();
    loop {
        match conn.transcript_rx.try_recv() {
            Ok(event) => events.push(event),
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                return Err("WebRTC transcript channel disconnected".to_string());
            }
        }
    }
    Ok(events)
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
