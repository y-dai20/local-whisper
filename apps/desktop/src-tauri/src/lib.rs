use asr_core::{WhisperContext, WhisperParams};
use chrono;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{WavSpec, WavWriter};
use once_cell::sync::OnceCell;
use parking_lot::Mutex as ParkingMutex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use tauri::{AppHandle, Emitter};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use voice_activity_detector::VoiceActivityDetector;

mod screen_recording;
mod system_audio;

const REMOTE_MODELS: &[RemoteModel] = &[
    RemoteModel {
        id: "base",
        name: "Whisper Base",
        filename: "ggml-base.bin",
        size: 74438528,
        description: "英語・多言語兼用 / 約 74 MB",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
    },
    RemoteModel {
        id: "small",
        name: "Whisper Small",
        filename: "ggml-small.bin",
        size: 244452544,
        description: "中規模モデル / 約 244 MB",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
    },
    RemoteModel {
        id: "medium",
        name: "Whisper Medium",
        filename: "ggml-medium.bin",
        size: 769073152,
        description: "高精度モデル / 約 769 MB",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
    },
    RemoteModel {
        id: "large-v3-turbo",
        name: "Whisper Large v3 Turbo",
        filename: "ggml-large-v3-turbo.bin",
        size: 3085627392,
        description: "最新 Large モデル / 約 3.1 GB",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
    },
];

static WHISPER_CTX: OnceCell<Arc<Mutex<Option<WhisperContext>>>> = OnceCell::new();
static WHISPER_PARAMS: OnceCell<Arc<ParkingMutex<WhisperParams>>> = OnceCell::new();
static RECORDING_STATE: OnceCell<Arc<ParkingMutex<RecordingState>>> = OnceCell::new();
static RECORDING_SAVE_PATH: OnceCell<Arc<ParkingMutex<Option<String>>>> = OnceCell::new();

const VAD_SAMPLE_RATE: u32 = 16_000;
const VAD_CHUNK_SIZE: usize = 512;
const DEFAULT_VAD_THRESHOLD: f32 = 0.1;
const DEFAULT_PARTIAL_TRANSCRIPT_INTERVAL_SAMPLES: usize = 4 * VAD_SAMPLE_RATE as usize;
const SESSION_MAX_SAMPLES: usize = 30 * VAD_SAMPLE_RATE as usize;
const SILENCE_TIMEOUT_SAMPLES: usize = 1 * VAD_SAMPLE_RATE as usize;
const VAD_PRE_BUFFER_MS: usize = 200;
const VAD_POST_BUFFER_MS: usize = 200;
const VAD_PRE_BUFFER_SAMPLES: usize = (VAD_SAMPLE_RATE as usize * VAD_PRE_BUFFER_MS) / 1000;
const VAD_POST_BUFFER_SAMPLES: usize = (VAD_SAMPLE_RATE as usize * VAD_POST_BUFFER_MS) / 1000;

struct SileroVadState {
    vad: VoiceActivityDetector,
    pending: Vec<f32>,
    threshold: f32,
    pre_buffer: Vec<f32>,
    post_buffer_remaining: usize,
    is_voice_active: bool,
}

struct RecordingState {
    is_recording: bool,
    is_muted: bool,
    mic_stream_id: u64,
    audio_buffer: Vec<f32>,
    session_audio: Vec<f32>,
    sample_rate: u32,
    selected_device_name: Option<String>,
    vad_state: Option<SileroVadState>,
    session_samples: usize,
    last_voice_sample: Option<usize>,
    last_partial_emit_samples: usize,
    session_id_counter: u64,
    active_session_id: Option<u64>,
    transcription_tx: Option<mpsc::Sender<TranscriptionCommand>>,
    transcription_handle: Option<JoinHandle<()>>,
    language: Option<String>,
    vad_threshold: f32,
    partial_transcript_interval_samples: usize,
    system_audio_enabled: bool,
    recording_save_enabled: bool,
    screen_recording_enabled: bool,
    screen_recording_active: bool,
    current_recording_dir: Option<String>,
    last_vad_event_time: std::time::Instant,
}

fn default_recording_state() -> RecordingState {
    RecordingState {
        is_recording: false,
        is_muted: true,
        mic_stream_id: 0,
        audio_buffer: Vec::new(),
        session_audio: Vec::new(),
        sample_rate: VAD_SAMPLE_RATE,
        selected_device_name: None,
        vad_state: None,
        session_samples: 0,
        last_voice_sample: None,
        last_partial_emit_samples: 0,
        session_id_counter: 0,
        active_session_id: None,
        transcription_tx: None,
        transcription_handle: None,
        language: None,
        vad_threshold: DEFAULT_VAD_THRESHOLD,
        partial_transcript_interval_samples: DEFAULT_PARTIAL_TRANSCRIPT_INTERVAL_SAMPLES,
        system_audio_enabled: false,
        recording_save_enabled: false,
        screen_recording_enabled: false,
        screen_recording_active: false,
        current_recording_dir: None,
        last_vad_event_time: std::time::Instant::now(),
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct TranscriptionSegment {
    text: String,
    timestamp: u64,
    #[serde(rename = "audioData")]
    audio_data: Option<Vec<f32>>,
    #[serde(rename = "sessionId")]
    session_id: String,
    #[serde(rename = "isFinal")]
    is_final: bool,
    source: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct VoiceActivityEvent {
    source: String,
    #[serde(rename = "isActive")]
    is_active: bool,
    timestamp: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct StreamingConfig {
    #[serde(rename = "vadThreshold")]
    vad_threshold: f32,
    #[serde(rename = "partialIntervalSeconds")]
    partial_interval_seconds: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
struct WhisperParamsConfig {
    #[serde(rename = "audioCtx")]
    audio_ctx: i32,
    temperature: f32,
}

impl From<WhisperParams> for WhisperParamsConfig {
    fn from(params: WhisperParams) -> Self {
        Self {
            audio_ctx: params.audio_ctx,
            temperature: params.temperature,
        }
    }
}

impl From<WhisperParamsConfig> for WhisperParams {
    fn from(config: WhisperParamsConfig) -> Self {
        WhisperParams {
            audio_ctx: config.audio_ctx,
            temperature: config.temperature,
        }
        .clamped()
    }
}

pub(crate) fn emit_transcription_segment(
    app_handle: &AppHandle,
    text: String,
    audio_data: Option<Vec<f32>>,
    session_id: String,
    is_final: bool,
    source: String,
) -> Result<(), String> {
    if text.trim().is_empty() {
        return Ok(());
    }

    let segment = TranscriptionSegment {
        text: text.clone(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        audio_data,
        session_id: session_id.clone(),
        is_final,
        source: source.clone(),
    };

    // 録画録音中の場合、txtファイルに保存
    let state =
        RECORDING_STATE.get_or_init(|| Arc::new(ParkingMutex::new(default_recording_state())));
    let state_guard = state.lock();
    let recording_dir = if state_guard.is_recording {
        state_guard.current_recording_dir.clone()
    } else {
        None
    };
    drop(state_guard);

    if is_final {
        if let Some(recording_dir) = recording_dir {
            save_transcription_to_txt(&recording_dir, &text, &session_id, &source)?;
        }
    }

    app_handle
        .emit("transcription-segment", &segment)
        .map_err(|e| e.to_string())
}

enum TranscriptionCommand {
    Run {
        audio: Vec<f32>,
        language: Option<String>,
        session_id: String,
        is_final: bool,
    },
    Stop,
}

fn spawn_transcription_worker(
    app_handle: AppHandle,
) -> (mpsc::Sender<TranscriptionCommand>, JoinHandle<()>) {
    let (tx, rx) = mpsc::channel::<TranscriptionCommand>();
    let handle = thread::spawn(move || {
        use std::collections::{HashMap, HashSet};

        while let Ok(command) = rx.recv() {
            match command {
                TranscriptionCommand::Run {
                    audio,
                    language,
                    session_id,
                    is_final,
                } => {
                    // キューにある全てのコマンドを収集
                    let mut all_commands = vec![(audio, language, session_id, is_final)];
                    while let Ok(next_command) = rx.try_recv() {
                        match next_command {
                            TranscriptionCommand::Run {
                                audio: a,
                                language: l,
                                session_id: s,
                                is_final: f,
                            } => {
                                all_commands.push((a, l, s, f));
                            }
                            TranscriptionCommand::Stop => return,
                        }
                    }

                    // セッションごとに最新のリクエストのみを保持
                    let mut latest_requests: HashMap<String, (Vec<f32>, Option<String>, bool)> =
                        HashMap::new();
                    let mut final_requests = Vec::new();
                    let mut sessions_with_final = HashSet::new();

                    for (audio, language, session_id, is_final) in all_commands {
                        if is_final {
                            // finalリクエストは必ず処理
                            sessions_with_final.insert(session_id.clone());
                            final_requests.push((audio, language, session_id, is_final));
                        } else {
                            // 非finalリクエストは最新のみ保持
                            latest_requests.insert(session_id, (audio, language, is_final));
                        }
                    }

                    // 非finalリクエストを処理
                    for (session_id, (audio, language, is_final)) in latest_requests {
                        if sessions_with_final.contains(&session_id) {
                            continue;
                        }
                        if let Err(err) = transcribe_and_emit(
                            &audio,
                            language.clone(),
                            session_id.clone(),
                            is_final,
                            &app_handle,
                        ) {
                            eprintln!("Transcription worker error: {}", err);
                        }
                    }

                    // finalリクエストを処理
                    for (audio, language, session_id, is_final) in final_requests {
                        if let Err(err) = transcribe_and_emit(
                            &audio,
                            language.clone(),
                            session_id,
                            is_final,
                            &app_handle,
                        ) {
                            eprintln!("Transcription worker error: {}", err);
                        }
                    }
                }
                TranscriptionCommand::Stop => break,
            }
        }
    });
    (tx, handle)
}

fn transcribe_and_emit(
    audio_data: &[f32],
    language: Option<String>,
    session_id: String,
    is_final: bool,
    app_handle: &AppHandle,
) -> Result<(), String> {
    let ctx_lock = WHISPER_CTX
        .get()
        .ok_or_else(|| "Whisper not initialized".to_string())?
        .clone();
    let ctx_guard = ctx_lock.lock().unwrap();
    let ctx = ctx_guard
        .as_ref()
        .ok_or_else(|| "Whisper context not available".to_string())?;

    let lang = language.as_deref().unwrap_or("ja");
    let text = ctx
        .transcribe_with_language(audio_data, lang)
        .map_err(|e| e.to_string())?;
    emit_transcription_segment(
        app_handle,
        text,
        if is_final {
            Some(audio_data.to_vec())
        } else {
            None
        },
        session_id,
        is_final,
        "user".to_string(),
    )
}

fn save_transcription_to_txt(
    recording_dir: &str,
    text: &str,
    session_id: &str,
    source: &str,
) -> Result<(), String> {
    use std::fs::OpenOptions;
    use std::io::Write;

    let txt_path = std::path::Path::new(recording_dir).join("transcription.txt");

    let timestamp = chrono::Local::now().format("%H:%M:%S");
    println!(
        "[{}] Saving transcription to: {} (source: {}, session: {})",
        timestamp,
        txt_path.display(),
        source,
        session_id,
    );

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&txt_path)
        .map_err(|e| format!("Failed to open transcription file: {}", e))?;

    let line = format!("[{}] [{}] {}\n", timestamp, source, text);

    file.write_all(line.as_bytes())
        .map_err(|e| format!("Failed to write transcription: {}", e))?;

    println!("[{}] Successfully wrote transcription to file", timestamp);

    Ok(())
}

fn setup_recording_directory(state: &mut RecordingState, base_path: &str) -> Result<(), String> {
    let now = chrono::Local::now();
    let date_dir = now.format("%Y-%m-%d").to_string();
    let time_dir = now.format("%H%M%S").to_string();
    let mut recording_dir = PathBuf::from(base_path);
    recording_dir.push(&date_dir);
    recording_dir.push(&time_dir);

    if !recording_dir.exists() {
        std::fs::create_dir_all(&recording_dir)
            .map_err(|e| format!("Failed to create recording directory: {}", e))?;
    }

    state.current_recording_dir = Some(recording_dir.to_string_lossy().to_string());
    println!(
        "[{}] Recording directory: {}",
        now.format("%H:%M:%S"),
        recording_dir.display()
    );

    Ok(())
}

fn ensure_active_session(state: &mut RecordingState) {
    if state.active_session_id.is_some() {
        return;
    }
    state.session_id_counter = state.session_id_counter.wrapping_add(1);
    state.active_session_id = Some(state.session_id_counter);
    state.session_audio.clear();
    state.session_samples = 0;
    state.last_partial_emit_samples = 0;
    state.last_voice_sample = None;
    let now = chrono::Local::now();
    println!(
        "[{}] Starting session #{}",
        now.format("%H:%M:%S"),
        state.session_id_counter
    );
}

fn queue_transcription(state: &RecordingState, is_final: bool) {
    if state.session_audio.is_empty() {
        return;
    }
    let Some(session_id) = state.active_session_id else {
        return;
    };
    let Some(tx) = &state.transcription_tx else {
        return;
    };
    let audio = state.session_audio.clone();
    let language = state.language.clone();
    let session_id_str = format!("mic_{}", session_id);
    if tx
        .send(TranscriptionCommand::Run {
            audio,
            language,
            session_id: session_id_str,
            is_final,
        })
        .is_err()
    {
        eprintln!("Failed to send transcription command");
    }
}

fn save_mic_session_audio_to_wav(
    audio_data: &[f32],
    session_id: u64,
    recording_dir: &str,
) -> Result<(), String> {
    let start_time = std::time::Instant::now();
    let now = chrono::Local::now();
    let timestamp = now.format("%H%M%S");
    let filename = format!("mic_audio_session_{}_{}.wav", session_id, timestamp);

    let mut path = PathBuf::from(recording_dir);

    println!(
        "[{}] Saving mic audio to recording directory: {}",
        now.format("%H:%M:%S"),
        path.display()
    );

    if !path.exists() {
        let now = chrono::Local::now();
        println!(
            "[{}] Creating directory: {}",
            now.format("%H:%M:%S"),
            path.display()
        );
        std::fs::create_dir_all(&path).map_err(|e| format!("Failed to create directory: {}", e))?;
    }

    path.push(filename);

    let spec = WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let now = chrono::Local::now();
    println!(
        "[{}] Creating WAV file: {}",
        now.format("%H:%M:%S"),
        path.display()
    );

    let mut writer =
        WavWriter::create(&path, spec).map_err(|e| format!("Failed to create WAV file: {}", e))?;

    for &sample in audio_data {
        let sample_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer
            .write_sample(sample_i16)
            .map_err(|e| format!("Failed to write sample: {}", e))?;
    }

    writer
        .finalize()
        .map_err(|e| format!("Failed to finalize WAV file: {}", e))?;

    let elapsed = start_time.elapsed();
    let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

    let now = chrono::Local::now();
    println!(
        "[{}] Successfully saved microphone audio session #{} to: {} ({} bytes, took {:.2}ms)",
        now.format("%H:%M:%S"),
        session_id,
        path.display(),
        file_size,
        elapsed.as_secs_f64() * 1000.0
    );

    Ok(())
}

fn finalize_active_session(state: &mut RecordingState, reason: &str) {
    if state.active_session_id.is_none() || state.session_audio.is_empty() {
        state.active_session_id = None;
        state.session_audio.clear();
        state.session_samples = 0;
        state.last_partial_emit_samples = 0;
        state.last_voice_sample = None;
        return;
    }

    let now = chrono::Local::now();
    println!(
        "[{}] Finalizing session #{} ({})",
        now.format("%H:%M:%S"),
        state.active_session_id.unwrap(),
        reason,
    );
    queue_transcription(state, true);

    // 録音セッション中のみ音声を保存
    if state.recording_save_enabled && state.is_recording {
        if let Some(recording_dir) = &state.current_recording_dir {
            let audio_clone = state.session_audio.clone();
            let session_id = state.active_session_id.unwrap();
            let audio_len = audio_clone.len();
            let duration = audio_len as f32 / 16000.0;
            let dir = recording_dir.clone();

            let now = chrono::Local::now();
            println!(
                "[{}] Saving microphone audio session #{}: {} samples ({:.2}s) to {}",
                now.format("%H:%M:%S"),
                session_id,
                audio_len,
                duration,
                dir
            );

            std::thread::spawn(move || {
                if let Err(e) = save_mic_session_audio_to_wav(&audio_clone, session_id, &dir) {
                    eprintln!("Failed to save microphone audio session: {}", e);
                }
            });
        } else {
            let now = chrono::Local::now();
            println!(
                "[{}] Microphone audio save enabled but no recording directory set",
                now.format("%H:%M:%S")
            );
        }
    }

    state.active_session_id = None;
    state.session_audio.clear();
    state.session_samples = 0;
    state.last_partial_emit_samples = 0;
    state.last_voice_sample = None;
}

fn stop_transcription_worker(state: &mut RecordingState) {
    if let Some(tx) = state.transcription_tx.take() {
        let _ = tx.send(TranscriptionCommand::Stop);
    }
    if let Some(handle) = state.transcription_handle.take() {
        let _ = handle.join();
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct TranscriptionResult {
    success: bool,
    text: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ModelInfo {
    name: String,
    path: String,
    size: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct RemoteModelStatus {
    id: String,
    name: String,
    filename: String,
    size: u64,
    description: String,
    installed: bool,
    path: Option<String>,
}

struct RemoteModel {
    id: &'static str,
    name: &'static str,
    filename: &'static str,
    size: u64,
    description: &'static str,
    url: &'static str,
}

#[derive(Debug, Serialize, Deserialize)]
struct AudioDevice {
    name: String,
    is_default: bool,
}

fn model_directory() -> Result<PathBuf, String> {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .map(|p| p.join("vendor/whisper.cpp/models"))
        .ok_or_else(|| "Failed to resolve model directory".to_string())
}

fn read_installed_models() -> Result<Vec<ModelInfo>, String> {
    let mut models = Vec::new();
    let model_dir = model_directory()?;

    if !model_dir.exists() {
        return Ok(models);
    }

    let entries = std::fs::read_dir(&model_dir)
        .map_err(|e| format!("Failed to read model directory: {}", e))?;

    for entry in entries {
        if let Ok(entry) = entry {
            let path = entry.path();
            if let Some(filename) = path.file_name() {
                let filename_str = filename.to_string_lossy();
                if filename_str.starts_with("ggml-") && filename_str.ends_with(".bin") {
                    if let Ok(metadata) = entry.metadata() {
                        models.push(ModelInfo {
                            name: filename_str.to_string(),
                            path: path.to_string_lossy().to_string(),
                            size: metadata.len(),
                        });
                    }
                }
            }
        }
    }

    models.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(models)
}

#[tauri::command]
async fn scan_models() -> Result<Vec<ModelInfo>, String> {
    read_installed_models()
}

#[tauri::command]
async fn initialize_whisper(model_path: String) -> Result<String, String> {
    let params_state =
        WHISPER_PARAMS.get_or_init(|| Arc::new(ParkingMutex::new(WhisperParams::default())));

    let mut ctx = WhisperContext::new(&model_path).map_err(|e| e.to_string())?;
    {
        let params_guard = params_state.lock();
        ctx.set_params(*params_guard);
    }

    WHISPER_CTX
        .get_or_init(|| Arc::new(Mutex::new(None)))
        .lock()
        .unwrap()
        .replace(ctx);

    Ok("Whisper initialized successfully".to_string())
}

fn whisper_params_state() -> Arc<ParkingMutex<WhisperParams>> {
    WHISPER_PARAMS
        .get_or_init(|| Arc::new(ParkingMutex::new(WhisperParams::default())))
        .clone()
}

#[tauri::command]
async fn get_whisper_params() -> Result<WhisperParamsConfig, String> {
    let state = whisper_params_state();
    let guard = state.lock();
    Ok(WhisperParamsConfig::from(*guard))
}

#[tauri::command]
async fn set_whisper_params(config: WhisperParamsConfig) -> Result<(), String> {
    let params: WhisperParams = config.into();
    let state = whisper_params_state();
    {
        let mut guard = state.lock();
        *guard = params;
    }

    if let Some(ctx_lock) = WHISPER_CTX.get() {
        let mut ctx_guard = ctx_lock.lock().unwrap();
        if let Some(ctx) = ctx_guard.as_mut() {
            ctx.set_params(params);
        }
    }

    let now = chrono::Local::now();
    println!(
        "[{}] Updated Whisper params: audio_ctx {}, temperature {:.2}",
        now.format("%H:%M:%S"),
        params.audio_ctx,
        params.temperature
    );

    Ok(())
}

#[tauri::command]
async fn list_remote_models() -> Result<Vec<RemoteModelStatus>, String> {
    let installed = read_installed_models()?;
    let mut statuses = Vec::new();

    for remote in REMOTE_MODELS {
        let installed_entry = installed.iter().find(|m| {
            Path::new(&m.path)
                .file_name()
                .map(|n| n == remote.filename)
                .unwrap_or(false)
        });

        statuses.push(RemoteModelStatus {
            id: remote.id.to_string(),
            name: remote.name.to_string(),
            filename: remote.filename.to_string(),
            size: remote.size,
            description: remote.description.to_string(),
            installed: installed_entry.is_some(),
            path: installed_entry.map(|m| m.path.clone()),
        });
    }

    Ok(statuses)
}

#[tauri::command]
async fn install_model(model_id: String) -> Result<ModelInfo, String> {
    let model = REMOTE_MODELS
        .iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| "Unknown model id".to_string())?;

    let dir = model_directory()?;
    if !dir.exists() {
        std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create model dir: {}", e))?;
    }

    let target_path = dir.join(model.filename);
    if target_path.exists() {
        let metadata = std::fs::metadata(&target_path)
            .map_err(|e| format!("Failed to read existing model metadata: {}", e))?;
        return Ok(ModelInfo {
            name: model.filename.to_string(),
            path: target_path.to_string_lossy().to_string(),
            size: metadata.len(),
        });
    }

    let tmp_path = target_path.with_extension("download");
    let client = Client::new();
    let mut response = client
        .get(model.url)
        .send()
        .await
        .map_err(|e| format!("Failed to download model: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Download failed with status {}", response.status()));
    }

    let mut file = fs::File::create(&tmp_path)
        .await
        .map_err(|e| format!("Failed to create temp file: {}", e))?;

    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| format!("Failed to read download chunk: {}", e))?
    {
        file.write_all(chunk.as_ref())
            .await
            .map_err(|e| format!("Failed to write chunk: {}", e))?;
    }
    file.flush()
        .await
        .map_err(|e| format!("Failed to flush download: {}", e))?;

    fs::rename(&tmp_path, &target_path)
        .await
        .map_err(|e| format!("Failed to move downloaded model: {}", e))?;

    let metadata = std::fs::metadata(&target_path)
        .map_err(|e| format!("Failed to read model metadata: {}", e))?;

    Ok(ModelInfo {
        name: model.filename.to_string(),
        path: target_path.to_string_lossy().to_string(),
        size: metadata.len(),
    })
}

#[tauri::command]
async fn delete_model(model_path: String) -> Result<(), String> {
    let dir = model_directory()?;
    let canonical_dir =
        std::fs::canonicalize(&dir).map_err(|e| format!("Failed to resolve model dir: {}", e))?;

    let target_path = PathBuf::from(&model_path);
    if !target_path.exists() {
        return Ok(());
    }

    let canonical_target = std::fs::canonicalize(&target_path)
        .map_err(|e| format!("Failed to resolve target path: {}", e))?;

    if !canonical_target.starts_with(&canonical_dir) {
        return Err("Invalid model path".to_string());
    }

    fs::remove_file(canonical_target)
        .await
        .map_err(|e| format!("Failed to delete model: {}", e))
}

async fn start_mic_stream(app_handle: AppHandle, language: Option<String>) -> Result<(), String> {
    let state =
        RECORDING_STATE.get_or_init(|| Arc::new(ParkingMutex::new(default_recording_state())));

    let (selected_device_name, current_mic_stream_id, configured_vad_threshold) = {
        let mut state_guard = state.lock();

        stop_transcription_worker(&mut state_guard);
        let (tx, handle) = spawn_transcription_worker(app_handle.clone());
        state_guard.transcription_tx = Some(tx);
        state_guard.transcription_handle = Some(handle);
        state_guard.language = language.clone();

        state_guard.mic_stream_id = state_guard.mic_stream_id.wrapping_add(1);
        let current_mic_stream_id = state_guard.mic_stream_id;

        let selected_device_name = state_guard.selected_device_name.clone();
        let configured_vad_threshold = state_guard.vad_threshold;

        state_guard.audio_buffer.clear();
        state_guard.session_audio.clear();
        state_guard.session_samples = 0;
        state_guard.last_voice_sample = None;
        state_guard.last_partial_emit_samples = 0;
        state_guard.active_session_id = None;
        state_guard.sample_rate = VAD_SAMPLE_RATE;

        let now = chrono::Local::now();
        println!(
            "[{}] Starting mic stream #{}",
            now.format("%H:%M:%S"),
            current_mic_stream_id
        );

        (
            selected_device_name,
            current_mic_stream_id,
            configured_vad_threshold,
        )
    };

    let host = cpal::default_host();

    // Use selected device or default
    let device = if let Some(device_name) = &selected_device_name {
        let now = chrono::Local::now();
        println!(
            "[{}] Looking for device: {}",
            now.format("%H:%M:%S"),
            device_name
        );

        host.input_devices()
            .map_err(|e| format!("Failed to enumerate devices: {}", e))?
            .find(|d| {
                if let Ok(name) = d.name() {
                    name == *device_name
                } else {
                    false
                }
            })
            .ok_or_else(|| format!("Selected device '{}' not found", device_name))?
    } else {
        host.default_input_device()
            .ok_or("No input device available")?
    };
    let device_name = device
        .name()
        .unwrap_or_else(|_| "Unknown device".to_string());
    let now = chrono::Local::now();
    println!(
        "[{}] Using input device: {}{}",
        now.format("%H:%M:%S"),
        device_name,
        selected_device_name
            .as_ref()
            .map(|_| "")
            .unwrap_or(" (default)")
    );
    let config = device
        .default_input_config()
        .map_err(|e| format!("Failed to get default input config: {}", e))?;

    let device_sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    let now = chrono::Local::now();
    println!(
        "[{}] Recording config - Device sample rate: {}, Channels: {}, Format: {:?}",
        now.format("%H:%M:%S"),
        device_sample_rate,
        channels,
        config.sample_format()
    );

    let vad_state = match VoiceActivityDetector::builder()
        .sample_rate(VAD_SAMPLE_RATE as i32)
        .chunk_size(VAD_CHUNK_SIZE)
        .build()
    {
        Ok(vad) => {
            let now = chrono::Local::now();
            println!(
                "[{}] Voice Activity Detector initialized",
                now.format("%H:%M:%S")
            );
            Some(SileroVadState {
                vad,
                pending: Vec::new(),
                threshold: configured_vad_threshold,
                pre_buffer: Vec::new(),
                post_buffer_remaining: 0,
                is_voice_active: false,
            })
        }
        Err(err) => {
            let now = chrono::Local::now();
            println!(
                "[{}] Failed to initialize VAD: {err:?}. Falling back to raw audio.",
                now.format("%H:%M:%S"),
            );
            None
        }
    };

    {
        let mut state_guard = state.lock();
        state_guard.vad_state = vad_state;
    }

    let now = chrono::Local::now();
    println!(
        "[{}] Building audio stream for format {:?}",
        now.format("%H:%M:%S"),
        config.sample_format()
    );

    // Build stream with recording ID check to prevent old callbacks from writing
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => {
            let state_clone = state.clone();
            let callback_count = Arc::new(ParkingMutex::new(0u64));
            let zero_chunk_count = Arc::new(ParkingMutex::new(0u64));
            let logged_non_zero = Arc::new(ParkingMutex::new(false));

            let app_handle_clone = app_handle.clone();
            device.build_input_stream(
                &config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mut state = state_clone.lock();
                    if !state.is_muted && state.mic_stream_id == current_mic_stream_id {
                        ensure_active_session(&mut state);
                        let mono_samples: Vec<f32> = data.iter().step_by(channels).copied().collect();
                        let processed_samples = if device_sample_rate == VAD_SAMPLE_RATE {
                            mono_samples
                        } else {
                            resample_audio(&mono_samples, device_sample_rate, VAD_SAMPLE_RATE)
                        };
                        for sample in processed_samples {
                            push_sample_with_optional_vad(&mut state, sample, &app_handle_clone);
                        }

                        if state.session_samples >= SESSION_MAX_SAMPLES {
                            finalize_active_session(&mut state, "session_max_duration");
                            ensure_active_session(&mut state);
                        }
                        let chunk_max = data.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                        if chunk_max == 0.0 {
                            let mut zero_count = zero_chunk_count.lock();
                            *zero_count += 1;
                            if *zero_count <= 5 {
                                let now = chrono::Local::now();
                                println!(
                                    "[{}] Audio callback chunk all zeros (count #{}, {} samples)",
                                    now.format("%H:%M:%S"),
                                    *zero_count,
                                    data.len()
                                );
                            }
                        } else {
                            let mut logged_non_zero_guard = logged_non_zero.lock();
                            if !*logged_non_zero_guard {
                                *logged_non_zero_guard = true;
                                let now = chrono::Local::now();
                                let preview: Vec<String> = data.iter().take(10).map(|v| format!("{:.4}", v)).collect();
                                println!(
                                    "[{}] First non-zero chunk detected: max {:.4}, preview [{}]",
                                    now.format("%H:%M:%S"),
                                    chunk_max,
                                    preview.join(" ")
                                );
                            }
                        }
                        let mut count = callback_count.lock();
                        *count += 1;
                        if *count % 100 == 0 {
                            let now = chrono::Local::now();
                            println!("[{}] Audio callback #{}: received {} samples, buffer size: {} samples ({:.2}s)",
                                     now.format("%H:%M:%S"), *count, data.len(),
                                     state.audio_buffer.len(), state.audio_buffer.len() as f32 / VAD_SAMPLE_RATE as f32);
                        }
                    }
                },
                |err| eprintln!("Error in audio stream: {}", err),
                None,
            )
        },
        cpal::SampleFormat::I16 => {
            let state_clone = state.clone();
            let callback_count = Arc::new(ParkingMutex::new(0u64));
            let app_handle_clone = app_handle.clone();

            device.build_input_stream(
                &config.into(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let mut state = state_clone.lock();
                    if !state.is_muted && state.mic_stream_id == current_mic_stream_id {
                        ensure_active_session(&mut state);
                        let mono_samples: Vec<f32> = data
                            .iter()
                            .step_by(channels)
                            .map(|&sample| sample as f32 / i16::MAX as f32)
                            .collect();
                        let processed_samples = if device_sample_rate == VAD_SAMPLE_RATE {
                            mono_samples
                        } else {
                            resample_audio(&mono_samples, device_sample_rate, VAD_SAMPLE_RATE)
                        };
                        for sample in processed_samples {
                            push_sample_with_optional_vad(&mut state, sample, &app_handle_clone);
                        }

                        if state.session_samples >= SESSION_MAX_SAMPLES {
                            finalize_active_session(&mut state, "session_max_duration");
                            ensure_active_session(&mut state);
                        }
                        let mut count = callback_count.lock();
                        *count += 1;
                        if *count % 100 == 0 {
                            let now = chrono::Local::now();
                            println!("[{}] Audio callback #{}: received {} samples, buffer size: {} samples ({:.2}s)",
                                     now.format("%H:%M:%S"), *count, data.len(),
                                     state.audio_buffer.len(), state.audio_buffer.len() as f32 / VAD_SAMPLE_RATE as f32);
                        }
                    }
                },
                |err| eprintln!("Error in audio stream: {}", err),
                None,
            )
        },
        _ => {
            return Err("Unsupported sample format".to_string());
        }
    }.map_err(|e| {
        let now = chrono::Local::now();
        println!(
            "[{}] Failed to build audio stream: {}",
            now.format("%H:%M:%S"),
            e
        );
        format!("Failed to build stream: {}", e)
    })?;

    let now = chrono::Local::now();
    println!("[{}] Starting audio stream…", now.format("%H:%M:%S"));

    stream.play().map_err(|e| {
        let now = chrono::Local::now();
        println!(
            "[{}] Failed to start audio stream: {}",
            now.format("%H:%M:%S"),
            e
        );
        format!("Failed to start stream: {}", e)
    })?;

    let now = chrono::Local::now();
    println!(
        "[{}] Audio stream started successfully",
        now.format("%H:%M:%S")
    );

    {
        let mut state_guard = state.lock();
        state_guard.audio_buffer.clear();
        state_guard.sample_rate = VAD_SAMPLE_RATE;
    }

    // Leak the stream to keep it alive - it will be invalidated by mic_stream_id on next stream
    // This prevents Send trait issues while ensuring old callbacks can't write to new streams
    std::mem::forget(stream);

    let now = chrono::Local::now();
    println!(
        "[{}] Mic stream started successfully",
        now.format("%H:%M:%S")
    );

    Ok(())
}

#[tauri::command]
async fn start_recording(_app_handle: AppHandle, language: Option<String>) -> Result<(), String> {
    let state = RECORDING_STATE.get().ok_or("Recording not initialized")?;

    let mut state_guard = state.lock();
    if state_guard.is_recording {
        return Err("Already recording".to_string());
    }

    let now = chrono::Local::now();
    println!("[{}] Starting recording session...", now.format("%H:%M:%S"));

    state_guard.is_recording = true;
    state_guard.language = language.clone();

    if state_guard.recording_save_enabled {
        let save_path = RECORDING_SAVE_PATH.get_or_init(|| Arc::new(ParkingMutex::new(None)));
        let path_guard = save_path.lock();
        if let Some(base_path) = path_guard.as_ref() {
            setup_recording_directory(&mut state_guard, base_path)?;
        }
    }

    let now = chrono::Local::now();
    println!("[{}] Recording session started", now.format("%H:%M:%S"));

    Ok(())
}

#[tauri::command]
async fn update_language(language: Option<String>) -> Result<(), String> {
    let state = RECORDING_STATE.get().ok_or("Recording not initialized")?;

    let mut state_guard = state.lock();
    state_guard.language = language.clone();
    drop(state_guard);

    let now = chrono::Local::now();
    let lang_str = language.as_deref().unwrap_or("auto");
    println!("[{}] Language updated to: {}", now.format("%H:%M:%S"), lang_str);

    system_audio::update_language(language);

    Ok(())
}

#[tauri::command]
async fn stop_recording() -> Result<(), String> {
    let state = RECORDING_STATE.get().ok_or("Recording not initialized")?;

    let mut state_guard = state.lock();
    if !state_guard.is_recording {
        return Err("Not recording".to_string());
    }

    let now = chrono::Local::now();
    println!("[{}] Stopping recording session...", now.format("%H:%M:%S"));

    state_guard.is_recording = false;
    state_guard.current_recording_dir = None;

    let now = chrono::Local::now();
    println!("[{}] Recording session stopped", now.format("%H:%M:%S"));

    Ok(())
}

#[tauri::command]
async fn start_mic(app_handle: AppHandle, language: Option<String>) -> Result<(), String> {
    let state =
        RECORDING_STATE.get_or_init(|| Arc::new(ParkingMutex::new(default_recording_state())));

    {
        let mut state_guard = state.lock();
        if !state_guard.is_muted {
            return Ok(());
        }
        state_guard.is_muted = false;
    }

    let now = chrono::Local::now();
    println!("[{}] Microphone unmuted", now.format("%H:%M:%S"));

    start_mic_stream(app_handle, language).await?;

    Ok(())
}

#[tauri::command]
async fn stop_mic() -> Result<(), String> {
    let state =
        RECORDING_STATE.get_or_init(|| Arc::new(ParkingMutex::new(default_recording_state())));

    let mut state_guard = state.lock();
    if state_guard.is_muted {
        return Ok(());
    }

    state_guard.is_muted = true;

    finalize_active_session(&mut state_guard, "mic_stopped");
    stop_transcription_worker(&mut state_guard);
    flush_vad_pending(&mut state_guard);
    state_guard.vad_state = None;

    let now = chrono::Local::now();
    println!("[{}] Microphone muted", now.format("%H:%M:%S"));

    Ok(())
}

#[tauri::command]
async fn get_mic_status() -> Result<bool, String> {
    let state =
        RECORDING_STATE.get_or_init(|| Arc::new(ParkingMutex::new(default_recording_state())));

    let state_guard = state.lock();
    Ok(!state_guard.is_muted)
}

fn process_vad_chunk_only(vad_state: &mut SileroVadState, chunk: &[f32]) -> Result<f32, String> {
    let chunk_i16: Vec<i16> = chunk
        .iter()
        .map(|&sample| (sample * i16::MAX as f32) as i16)
        .collect();
    let probability = vad_state.vad.predict(chunk_i16);
    Ok(probability)
}

pub(crate) fn emit_voice_activity_event(app_handle: &AppHandle, source: &str, is_active: bool) {
    let event = VoiceActivityEvent {
        source: source.to_string(),
        is_active,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    if let Err(e) = app_handle.emit("voice-activity", &event) {
        eprintln!("Failed to emit voice activity event: {}", e);
    }
}

fn push_sample_with_optional_vad(state: &mut RecordingState, sample: f32, app_handle: &AppHandle) {
    state.session_samples += 1;

    if state.vad_state.is_none() {
        state.audio_buffer.push(sample);
        state.session_audio.push(sample);
        state.last_voice_sample = Some(state.session_samples);
    } else {
        let mut disable_vad = false;
        let mut voice_detected = false;

        if let Some(vad_state) = state.vad_state.as_mut() {
            vad_state.pending.push(sample);

            while vad_state.pending.len() >= VAD_CHUNK_SIZE {
                let chunk: Vec<f32> = vad_state.pending.drain(..VAD_CHUNK_SIZE).collect();
                match process_vad_chunk_only(vad_state, &chunk) {
                    Ok(prob) => {
                        let voice_in_chunk = prob > vad_state.threshold;

                        if voice_in_chunk {
                            // 音声検出: プレバッファの内容を先に追加
                            if !vad_state.is_voice_active && !vad_state.pre_buffer.is_empty() {
                                state.audio_buffer.extend_from_slice(&vad_state.pre_buffer);
                                state.session_audio.extend_from_slice(&vad_state.pre_buffer);
                                vad_state.pre_buffer.clear();
                            }

                            // 現在のチャンクを追加
                            state.audio_buffer.extend_from_slice(&chunk);
                            state.session_audio.extend_from_slice(&chunk);
                            voice_detected = true;
                            vad_state.is_voice_active = true;
                            vad_state.post_buffer_remaining = VAD_POST_BUFFER_SAMPLES;
                        } else {
                            // 音声未検出
                            if vad_state.is_voice_active && vad_state.post_buffer_remaining > 0 {
                                // ポストバッファ期間中: 音声として追加
                                state.audio_buffer.extend_from_slice(&chunk);
                                state.session_audio.extend_from_slice(&chunk);
                                vad_state.post_buffer_remaining =
                                    vad_state.post_buffer_remaining.saturating_sub(chunk.len());

                                if vad_state.post_buffer_remaining == 0 {
                                    vad_state.is_voice_active = false;
                                }
                            } else {
                                // プレバッファに保存（最大200ms分）
                                vad_state.pre_buffer.extend_from_slice(&chunk);
                                if vad_state.pre_buffer.len() > VAD_PRE_BUFFER_SAMPLES {
                                    let excess =
                                        vad_state.pre_buffer.len() - VAD_PRE_BUFFER_SAMPLES;
                                    vad_state.pre_buffer.drain(0..excess);
                                }
                                vad_state.is_voice_active = false;
                            }
                        }
                    }
                    Err(err) => {
                        eprintln!("Silero VAD failed, disabling VAD: {}", err);
                        state.audio_buffer.extend_from_slice(&chunk);
                        state.session_audio.extend_from_slice(&chunk);
                        disable_vad = true;
                        break;
                    }
                }
            }
        }

        if disable_vad {
            state.vad_state = None;
        }

        if voice_detected {
            state.last_voice_sample = Some(state.session_samples);
        }
    }

    // Emit periodic VAD voice activity events every 500ms
    let now = std::time::Instant::now();
    if now.duration_since(state.last_vad_event_time).as_millis() >= 500 {
        let is_voice_active = state
            .vad_state
            .as_ref()
            .map_or(false, |v| v.is_voice_active);
        emit_voice_activity_event(app_handle, "user", is_voice_active);
        state.last_vad_event_time = now;
    }

    if let Some(last_voice) = state.last_voice_sample {
        if state.session_samples - last_voice >= SILENCE_TIMEOUT_SAMPLES {
            finalize_active_session(state, "silence_timeout");
            ensure_active_session(state);
        }
    }

    if state.session_samples - state.last_partial_emit_samples
        >= state.partial_transcript_interval_samples
    {
        queue_transcription(state, false);
        state.last_partial_emit_samples = state.session_samples;
    }
}

fn flush_vad_pending(state: &mut RecordingState) {
    if state.vad_state.is_none() {
        return;
    }

    let mut disable_vad = false;
    if let Some(vad_state) = state.vad_state.as_mut() {
        if vad_state.pending.is_empty() {
            return;
        }

        let mut chunk = vad_state.pending.clone();
        let original_len = chunk.len();
        vad_state.pending.clear();
        if chunk.len() < VAD_CHUNK_SIZE {
            chunk.resize(VAD_CHUNK_SIZE, 0.0);
        }

        match process_vad_chunk_only(vad_state, &chunk) {
            Ok(prob) => {
                if prob >= vad_state.threshold {
                    state.audio_buffer.extend_from_slice(&chunk[..original_len]);
                    state
                        .session_audio
                        .extend_from_slice(&chunk[..original_len]);
                }
            }
            Err(err) => {
                eprintln!("Silero VAD failed during flush: {}", err);
                state.audio_buffer.extend_from_slice(&chunk[..original_len]);
                state
                    .session_audio
                    .extend_from_slice(&chunk[..original_len]);
                disable_vad = true;
            }
        }
    }

    if disable_vad {
        state.vad_state = None;
    }
}

#[tauri::command]
async fn list_audio_devices() -> Result<Vec<AudioDevice>, String> {
    let host = cpal::default_host();
    let default_device_name = host.default_input_device().and_then(|d| d.name().ok());

    let devices: Vec<AudioDevice> = host
        .input_devices()
        .map_err(|e| format!("Failed to enumerate devices: {}", e))?
        .filter_map(|device| {
            device.name().ok().map(|name| {
                let is_default = default_device_name.as_ref() == Some(&name);
                AudioDevice { name, is_default }
            })
        })
        .collect();

    let now = chrono::Local::now();
    println!(
        "[{}] Detected {} audio input device(s)",
        now.format("%H:%M:%S"),
        devices.len()
    );
    for device in &devices {
        println!(
            "  - {}{}",
            device.name,
            if device.is_default { " (default)" } else { "" }
        );
    }

    Ok(devices)
}

#[tauri::command]
async fn select_audio_device(device_name: String) -> Result<(), String> {
    let state =
        RECORDING_STATE.get_or_init(|| Arc::new(ParkingMutex::new(default_recording_state())));

    let mut state_guard = state.lock();
    state_guard.selected_device_name = Some(device_name.clone());

    let now = chrono::Local::now();
    println!(
        "[{}] Selected audio device: {}",
        now.format("%H:%M:%S"),
        device_name
    );

    Ok(())
}

#[tauri::command]
async fn get_streaming_config() -> Result<StreamingConfig, String> {
    let state =
        RECORDING_STATE.get_or_init(|| Arc::new(ParkingMutex::new(default_recording_state())));

    let state_guard = state.lock();
    Ok(StreamingConfig {
        vad_threshold: state_guard.vad_threshold,
        partial_interval_seconds: state_guard.partial_transcript_interval_samples as f32
            / VAD_SAMPLE_RATE as f32,
    })
}

#[tauri::command]
async fn set_streaming_config(config: StreamingConfig) -> Result<(), String> {
    let state =
        RECORDING_STATE.get_or_init(|| Arc::new(ParkingMutex::new(default_recording_state())));

    let mut state_guard = state.lock();
    let clamped_threshold = config.vad_threshold.clamp(0.01, 0.99);
    let clamped_interval_seconds = config.partial_interval_seconds.clamp(0.5, 30.0);
    let samples = ((clamped_interval_seconds * VAD_SAMPLE_RATE as f32).round() as usize).max(1);

    state_guard.vad_threshold = clamped_threshold;
    state_guard.partial_transcript_interval_samples = samples;
    state_guard.last_partial_emit_samples = 0;

    if let Some(vad_state) = state_guard.vad_state.as_mut() {
        vad_state.threshold = clamped_threshold;
    }

    let now = chrono::Local::now();
    println!(
        "[{}] Updated streaming config: threshold {:.4}, partial interval {:.2}s ({} samples)",
        now.format("%H:%M:%S"),
        clamped_threshold,
        clamped_interval_seconds,
        samples
    );

    Ok(())
}

#[tauri::command]
async fn check_microphone_permission() -> Result<bool, String> {
    // On macOS, try to access the default input device
    // If permission is denied, this will fail
    let host = cpal::default_host();
    let permission = host
        .default_input_device()
        .map(|device| device.default_input_config().is_ok())
        .unwrap_or(false);

    let now = chrono::Local::now();
    println!(
        "[{}] Microphone permission check: {}",
        now.format("%H:%M:%S"),
        if permission { "granted" } else { "denied" }
    );

    Ok(permission)
}

#[tauri::command]
async fn start_system_audio(app_handle: AppHandle) -> Result<(), String> {
    let state = RECORDING_STATE.get().ok_or("Recording not initialized")?;
    system_audio::start_system_audio_capture(state.clone(), app_handle)
}

#[tauri::command]
async fn stop_system_audio() -> Result<(), String> {
    let state = RECORDING_STATE.get().ok_or("Recording not initialized")?;
    system_audio::stop_system_audio_capture(state.clone())
}

#[tauri::command]
async fn get_system_audio_status() -> Result<bool, String> {
    let state = RECORDING_STATE.get().ok_or("Recording not initialized")?;
    let state_guard = state.lock();
    Ok(state_guard.system_audio_enabled)
}

#[tauri::command]
async fn set_recording_save_config(enabled: bool, path: Option<String>) -> Result<(), String> {
    let state =
        RECORDING_STATE.get_or_init(|| Arc::new(ParkingMutex::new(default_recording_state())));

    let mut state_guard = state.lock();
    state_guard.recording_save_enabled = enabled;
    drop(state_guard);

    let save_path = RECORDING_SAVE_PATH.get_or_init(|| Arc::new(ParkingMutex::new(None)));
    let mut path_guard = save_path.lock();
    *path_guard = path.clone();
    drop(path_guard);

    let now = chrono::Local::now();
    if enabled {
        if let Some(p) = &path {
            println!("[{}] Recording save enabled: {}", now.format("%H:%M:%S"), p);
        }
    } else {
        println!("[{}] Recording save disabled", now.format("%H:%M:%S"));
    }

    Ok(())
}

#[tauri::command]
async fn get_recording_save_config() -> Result<(bool, Option<String>), String> {
    let state =
        RECORDING_STATE.get_or_init(|| Arc::new(ParkingMutex::new(default_recording_state())));

    let state_guard = state.lock();
    let enabled = state_guard.recording_save_enabled;
    drop(state_guard);

    let save_path = RECORDING_SAVE_PATH.get_or_init(|| Arc::new(ParkingMutex::new(None)));
    let path_guard = save_path.lock();
    let path = path_guard.clone();

    Ok((enabled, path))
}

#[tauri::command]
async fn set_screen_recording_config(enabled: bool) -> Result<(), String> {
    let state =
        RECORDING_STATE.get_or_init(|| Arc::new(ParkingMutex::new(default_recording_state())));

    let mut state_guard = state.lock();
    state_guard.screen_recording_enabled = enabled;

    let now = chrono::Local::now();
    if enabled {
        println!("[{}] Screen recording enabled", now.format("%H:%M:%S"));
    } else {
        println!("[{}] Screen recording disabled", now.format("%H:%M:%S"));
    }

    Ok(())
}

#[tauri::command]
async fn get_screen_recording_config() -> Result<bool, String> {
    let state =
        RECORDING_STATE.get_or_init(|| Arc::new(ParkingMutex::new(default_recording_state())));

    let state_guard = state.lock();
    Ok(state_guard.screen_recording_enabled)
}

#[tauri::command]
async fn start_screen_recording() -> Result<(), String> {
    let state =
        RECORDING_STATE.get_or_init(|| Arc::new(ParkingMutex::new(default_recording_state())));

    let mut state_guard = state.lock();

    if state_guard.recording_save_enabled {
        let save_path = RECORDING_SAVE_PATH.get_or_init(|| Arc::new(ParkingMutex::new(None)));
        let path_guard = save_path.lock();
        if let Some(base_path) = path_guard.as_ref() {
            setup_recording_directory(&mut state_guard, base_path)?;
        }
    }

    let recording_dir = state_guard.current_recording_dir.clone();
    state_guard.screen_recording_active = true;
    drop(state_guard);

    if let Some(dir) = recording_dir {
        let now = chrono::Local::now();
        let timestamp = now.format("%Y%m%d_%H%M%S");
        let filename = format!("screen_recording_{}.mp4", timestamp);
        let full_path = std::path::Path::new(&dir).join(filename);

        println!(
            "[{}] Starting screen recording to: {}",
            now.format("%H:%M:%S"),
            full_path.display()
        );

        screen_recording::start_screen_recording(full_path.to_str().unwrap())
    } else {
        Err("Recording directory not set".to_string())
    }
}

#[tauri::command]
async fn stop_screen_recording() -> Result<(), String> {
    let state =
        RECORDING_STATE.get_or_init(|| Arc::new(ParkingMutex::new(default_recording_state())));

    let mut state_guard = state.lock();
    state_guard.screen_recording_active = false;
    drop(state_guard);

    screen_recording::stop_screen_recording()
}

#[tauri::command]
async fn get_screen_recording_status() -> Result<bool, String> {
    let state =
        RECORDING_STATE.get_or_init(|| Arc::new(ParkingMutex::new(default_recording_state())));

    let state_guard = state.lock();
    Ok(state_guard.screen_recording_active)
}

#[tauri::command]
async fn get_supported_languages() -> Result<Vec<(String, String)>, String> {
    Ok(vec![
        ("auto".to_string(), "自動検出".to_string()),
        ("ja".to_string(), "日本語".to_string()),
        ("en".to_string(), "English".to_string()),
        ("zh".to_string(), "中文".to_string()),
        ("ko".to_string(), "한국어".to_string()),
        ("es".to_string(), "Español".to_string()),
        ("fr".to_string(), "Français".to_string()),
        ("de".to_string(), "Deutsch".to_string()),
        ("it".to_string(), "Italiano".to_string()),
        ("pt".to_string(), "Português".to_string()),
        ("ru".to_string(), "Русский".to_string()),
    ])
}

// Linear resampling without aggressive filtering to preserve amplitude
fn resample_audio(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return input.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = ((input.len() as f64) / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos.floor() as usize;
        let frac = src_pos - idx as f64;

        if idx + 1 < input.len() {
            let sample = input[idx] as f64 * (1.0 - frac) + input[idx + 1] as f64 * frac;
            output.push(sample as f32);
        } else if idx < input.len() {
            output.push(input[idx]);
        }
    }

    output
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            scan_models,
            initialize_whisper,
            start_recording,
            stop_recording,
            update_language,
            start_mic,
            stop_mic,
            get_mic_status,
            get_supported_languages,
            list_remote_models,
            install_model,
            delete_model,
            list_audio_devices,
            select_audio_device,
            get_streaming_config,
            set_streaming_config,
            get_whisper_params,
            set_whisper_params,
            set_recording_save_config,
            get_recording_save_config,
            set_screen_recording_config,
            get_screen_recording_config,
            start_screen_recording,
            stop_screen_recording,
            get_screen_recording_status,
            check_microphone_permission,
            start_system_audio,
            stop_system_audio,
            get_system_audio_status,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
