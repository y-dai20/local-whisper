use chrono;
use cpal::traits::{DeviceTrait, HostTrait};
use env_logger::Env;
use log::{error, info};
use once_cell::sync::OnceCell;
use parking_lot::Mutex as ParkingMutex;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tauri::{AppHandle, Emitter};

mod audio;
mod commands;
mod mic;
mod screen_recording;
mod system_audio;
mod transcription;
mod whisper;

pub use whisper::{
    delete_model_impl, get_whisper_params_impl, initialize_whisper_impl, install_model_impl,
    list_remote_models_impl, scan_models_impl, set_whisper_params_impl, ModelInfo,
    RemoteModelStatus, WhisperParamsConfig,
};

use audio::constants::VAD_SAMPLE_RATE;
use audio::processing::finalize_active_session;
use audio::state::{recording_state, try_recording_state, RecordingState};
use mic::{start_mic_stream};
use transcription::{transcribe_and_emit_common, TranscriptionSegment};
use transcription::worker::stop_transcription_worker;

static RECORDING_SAVE_PATH: OnceCell<Arc<ParkingMutex<Option<String>>>> = OnceCell::new();
static APP_HANDLE: OnceCell<AppHandle> = OnceCell::new();

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

    info!(
        "[emit_transcription_segment] Emitting segment #{}: {} | is_final: {}",
        session_id, text, is_final
    );

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
    let state = recording_state();
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
    info!(
        "Saving transcription to: {} (source: {}, session: {})",
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

    info!("Successfully wrote transcription to file");

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
    info!("Recording directory prepared: {}", recording_dir.display());

    Ok(())
}



#[derive(Debug, Serialize, Deserialize, Clone)]
struct VoiceActivityEvent {
    source: String,
    #[serde(rename = "isActive")]
    is_active: bool,
    timestamp: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StreamingConfig {
    #[serde(rename = "vadThreshold")]
    pub vad_threshold: f32,
    #[serde(rename = "partialIntervalSeconds")]
    pub partial_interval_seconds: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AudioDevice {
    pub name: String,
    pub is_default: bool,
}



pub(crate) async fn start_recording_impl(language: Option<String>) -> Result<(), String> {
    let state = try_recording_state().ok_or("Recording not initialized")?;

    let mut state_guard = state.lock();
    if state_guard.is_recording {
        return Err("Already recording".to_string());
    }

    info!("Starting recording session...");

    state_guard.is_recording = true;
    state_guard.language = language.clone();

    if state_guard.recording_save_enabled {
        let save_path = RECORDING_SAVE_PATH.get_or_init(|| Arc::new(ParkingMutex::new(None)));
        let path_guard = save_path.lock();
        if let Some(base_path) = path_guard.as_ref() {
            setup_recording_directory(&mut state_guard, base_path)?;
        }
    }

    info!("Recording session started");

    Ok(())
}

pub(crate) async fn update_language_impl(language: Option<String>) -> Result<(), String> {
    let state = try_recording_state().ok_or("Recording not initialized")?;

    let mut state_guard = state.lock();
    state_guard.language = language.clone();
    drop(state_guard);

    let lang_str = language.as_deref().unwrap_or("auto");
    info!("Language updated to: {}", lang_str);

    let _ = system_audio::update_language(language);

    Ok(())
}

pub(crate) async fn stop_recording_impl() -> Result<(), String> {
    let state = try_recording_state().ok_or("Recording not initialized")?;

    let mut state_guard = state.lock();
    if !state_guard.is_recording {
        return Err("Not recording".to_string());
    }

    info!("Stopping recording session...");

    state_guard.is_recording = false;
    state_guard.current_recording_dir = None;

    info!("Recording session stopped");

    Ok(())
}

pub(crate) async fn start_mic_impl(language: Option<String>) -> Result<(), String> {
    let state = recording_state();

    {
        let mut state_guard = state.lock();
        if !state_guard.is_muted {
            return Ok(());
        }
        state_guard.is_muted = false;
    }

    info!("Microphone unmuted");

    let app_handle = APP_HANDLE.get().ok_or("App not initialized")?.clone();
    start_mic_stream(app_handle, language).await?;

    Ok(())
}

pub(crate) async fn stop_mic_impl() -> Result<(), String> {
    let state = recording_state();

    let mut state_guard = state.lock();
    if state_guard.is_muted {
        return Ok(());
    }

    state_guard.is_muted = true;

    finalize_active_session(&mut state_guard, "mic_stopped");
    stop_transcription_worker(&mut state_guard);
    state_guard.vad_state = None;

    info!("Microphone muted");

    Ok(())
}

pub(crate) async fn get_mic_status_impl() -> Result<bool, String> {
    let state = recording_state();

    let state_guard = state.lock();
    Ok(!state_guard.is_muted)
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
        error!("Failed to emit voice activity event: {}", e);
    }
}

pub(crate) async fn list_audio_devices_impl() -> Result<Vec<AudioDevice>, String> {
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

    info!("Detected {} audio input device(s)", devices.len());
    for device in &devices {
        info!(
            "  - {}{}",
            device.name,
            if device.is_default { " (default)" } else { "" }
        );
    }

    Ok(devices)
}

pub(crate) async fn select_audio_device_impl(device_name: String) -> Result<(), String> {
    let state = recording_state();

    let mut state_guard = state.lock();
    state_guard.selected_device_name = Some(device_name.clone());

    info!("Selected audio device: {}", device_name);

    Ok(())
}

pub(crate) async fn get_streaming_config_impl() -> Result<StreamingConfig, String> {
    let state = recording_state();

    let state_guard = state.lock();
    Ok(StreamingConfig {
        vad_threshold: state_guard.vad_threshold,
        partial_interval_seconds: state_guard.partial_transcript_interval_samples as f32
            / VAD_SAMPLE_RATE as f32,
    })
}

pub(crate) async fn set_streaming_config_impl(config: StreamingConfig) -> Result<(), String> {
    let state = recording_state();

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

    info!(
        "Updated streaming config: threshold {:.4}, partial interval {:.2}s ({} samples)",
        clamped_threshold, clamped_interval_seconds, samples
    );

    Ok(())
}

pub(crate) async fn check_microphone_permission_impl() -> Result<bool, String> {
    // On macOS, try to access the default input device
    // If permission is denied, this will fail
    let host = cpal::default_host();
    let permission = host
        .default_input_device()
        .map(|device| device.default_input_config().is_ok())
        .unwrap_or(false);

    info!(
        "Microphone permission check: {}",
        if permission { "granted" } else { "denied" }
    );

    Ok(permission)
}

pub(crate) async fn start_system_audio_impl() -> Result<(), String> {
    let state = try_recording_state().ok_or("Recording not initialized")?;
    let app_handle = APP_HANDLE.get().ok_or("App not initialized")?.clone();
    system_audio::start_system_audio_capture(state.clone(), app_handle)
}

pub(crate) async fn stop_system_audio_impl() -> Result<(), String> {
    let state = try_recording_state().ok_or("Recording not initialized")?;
    system_audio::stop_system_audio_capture(state.clone())
}

pub(crate) async fn get_system_audio_status_impl() -> Result<bool, String> {
    let state = try_recording_state().ok_or("Recording not initialized")?;
    let state_guard = state.lock();
    Ok(state_guard.system_audio_enabled)
}

pub(crate) async fn set_recording_save_config_impl(
    enabled: bool,
    path: Option<String>,
) -> Result<(), String> {
    let state = recording_state();

    let mut state_guard = state.lock();
    state_guard.recording_save_enabled = enabled;
    drop(state_guard);

    let save_path = RECORDING_SAVE_PATH.get_or_init(|| Arc::new(ParkingMutex::new(None)));
    let mut path_guard = save_path.lock();
    *path_guard = path.clone();
    drop(path_guard);

    if enabled {
        if let Some(p) = &path {
            info!("Recording save enabled: {}", p);
        }
    } else {
        info!("Recording save disabled");
    }

    Ok(())
}

pub(crate) async fn get_recording_save_config_impl() -> Result<(bool, Option<String>), String> {
    let state = recording_state();

    let state_guard = state.lock();
    let enabled = state_guard.recording_save_enabled;
    drop(state_guard);

    let save_path = RECORDING_SAVE_PATH.get_or_init(|| Arc::new(ParkingMutex::new(None)));
    let path_guard = save_path.lock();
    let path = path_guard.clone();

    Ok((enabled, path))
}

pub(crate) async fn set_screen_recording_config_impl(enabled: bool) -> Result<(), String> {
    let state = recording_state();

    let mut state_guard = state.lock();
    state_guard.screen_recording_enabled = enabled;

    if enabled {
        info!("Screen recording enabled");
    } else {
        info!("Screen recording disabled");
    }

    Ok(())
}

pub(crate) async fn get_screen_recording_config_impl() -> Result<bool, String> {
    let state = recording_state();

    let state_guard = state.lock();
    Ok(state_guard.screen_recording_enabled)
}

pub(crate) async fn start_screen_recording_impl() -> Result<(), String> {
    let state = recording_state();

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

        info!("Starting screen recording to: {}", full_path.display());

        screen_recording::start_screen_recording(full_path.to_str().unwrap())
    } else {
        Err("Recording directory not set".to_string())
    }
}

pub(crate) async fn stop_screen_recording_impl() -> Result<(), String> {
    let state = recording_state();

    let mut state_guard = state.lock();
    state_guard.screen_recording_active = false;
    drop(state_guard);

    screen_recording::stop_screen_recording()
}

pub(crate) async fn get_screen_recording_status_impl() -> Result<bool, String> {
    let state = recording_state();

    let state_guard = state.lock();
    Ok(state_guard.screen_recording_active)
}

pub(crate) async fn get_supported_languages_impl() -> Result<Vec<(String, String)>, String> {
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _ = env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .try_init();

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let _ = APP_HANDLE.set(app.handle().clone());
            Ok(())
        });

    let app = commands::register(app);

    app.run(tauri::generate_context!())
        .expect("error while running tauri application");
}
