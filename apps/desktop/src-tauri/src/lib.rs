use asr_core::WhisperContext;
use chrono;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use once_cell::sync::OnceCell;
use parking_lot::Mutex as ParkingMutex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};
use tokio::fs;
use tokio::io::AsyncWriteExt;

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
static RECORDING_STATE: OnceCell<Arc<ParkingMutex<RecordingState>>> = OnceCell::new();

struct RecordingState {
    is_recording: bool,
    audio_buffer: Vec<f32>,
    sample_rate: u32,
    selected_device_name: Option<String>,
    recording_id: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct TranscriptionSegment {
    text: String,
    timestamp: u64,
    #[serde(rename = "audioData")]
    audio_data: Option<Vec<f32>>,
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
    let ctx = WhisperContext::new(&model_path).map_err(|e| e.to_string())?;

    WHISPER_CTX
        .get_or_init(|| Arc::new(Mutex::new(None)))
        .lock()
        .unwrap()
        .replace(ctx);

    Ok("Whisper initialized successfully".to_string())
}

#[tauri::command]
async fn transcribe_audio(
    audio_data: Vec<f32>,
    language: Option<String>,
    app_handle: AppHandle,
) -> Result<TranscriptionResult, String> {
    let ctx_lock = WHISPER_CTX
        .get()
        .ok_or("Whisper not initialized")?
        .clone();

    let ctx_guard = ctx_lock.lock().unwrap();
    let ctx = ctx_guard.as_ref().ok_or("Whisper context not available")?;

    let lang = language.as_deref().unwrap_or("ja");

    match ctx.transcribe_with_language(&audio_data, lang) {
        Ok(text) => {
            let segment = TranscriptionSegment {
                text: text.clone(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
                audio_data: Some(audio_data.clone()),
            };

            let _ = app_handle.emit("transcription-segment", &segment);

            Ok(TranscriptionResult {
                success: true,
                text: Some(text),
                error: None,
            })
        }
        Err(e) => Ok(TranscriptionResult {
            success: false,
            text: None,
            error: Some(e.to_string()),
        }),
    }
}

#[tauri::command]
async fn list_remote_models() -> Result<Vec<RemoteModelStatus>, String> {
    let installed = read_installed_models()?;
    let mut statuses = Vec::new();

    for remote in REMOTE_MODELS {
        let installed_entry = installed
            .iter()
            .find(|m| Path::new(&m.path).file_name().map(|n| n == remote.filename).unwrap_or(false));

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

#[tauri::command]
async fn transcribe_audio_stream(
    audio_data: Vec<i16>,
    app_handle: AppHandle,
) -> Result<(), String> {
    let ctx_lock = WHISPER_CTX
        .get()
        .ok_or("Whisper not initialized")?
        .clone();

    let audio_f32 = asr_core::convert_pcm_to_f32(&audio_data);

    let ctx_guard = ctx_lock.lock().unwrap();
    let ctx = ctx_guard.as_ref().ok_or("Whisper context not available")?;

    ctx.transcribe_with_callback(&audio_f32, |text| {
        let segment = TranscriptionSegment {
            text: text.to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            audio_data: Some(audio_f32.clone()),
        };
        let _ = app_handle.emit("transcription-segment", &segment);
    })
    .map_err(|e| e.to_string())?;

    Ok(())
}

#[tauri::command]
async fn start_recording(_app_handle: AppHandle) -> Result<(), String> {
    let state = RECORDING_STATE.get_or_init(|| {
        Arc::new(ParkingMutex::new(RecordingState {
            is_recording: false,
            audio_buffer: Vec::new(),
            sample_rate: 16000,
            selected_device_name: None,
            recording_id: 0,
        }))
    });

    let (selected_device_name, current_recording_id);
    {
        let mut state_guard = state.lock();
        if state_guard.is_recording {
            return Err("Already recording".to_string());
        }

        // Increment recording ID to invalidate old callbacks
        state_guard.recording_id = state_guard.recording_id.wrapping_add(1);
        current_recording_id = state_guard.recording_id;

        selected_device_name = state_guard.selected_device_name.clone();

        let now = chrono::Local::now();
        println!("[{}] Starting recording #{}", now.format("%H:%M:%S"), current_recording_id);
    }

    let host = cpal::default_host();

    // Use selected device or default
    let device = if let Some(device_name) = &selected_device_name {
        let now = chrono::Local::now();
        println!("[{}] Looking for device: {}", now.format("%H:%M:%S"), device_name);

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
    let device_name = device.name().unwrap_or_else(|_| "Unknown device".to_string());
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
    let config = device.default_input_config()
        .map_err(|e| format!("Failed to get default input config: {}", e))?;

    let device_sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    let now = chrono::Local::now();
    println!("[{}] Recording config - Device sample rate: {}, Channels: {}, Format: {:?}",
             now.format("%H:%M:%S"), device_sample_rate, channels, config.sample_format());

    // Build stream with recording ID check to prevent old callbacks from writing
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => {
            let state_clone = state.clone();
            let callback_count = Arc::new(ParkingMutex::new(0u64));
            let zero_chunk_count = Arc::new(ParkingMutex::new(0u64));
            let logged_non_zero = Arc::new(ParkingMutex::new(false));

            device.build_input_stream(
                &config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mut state = state_clone.lock();
                    if state.is_recording && state.recording_id == current_recording_id {
                        for &sample in data.iter().step_by(channels) {
                            state.audio_buffer.push(sample);
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
                                     state.audio_buffer.len(), state.audio_buffer.len() as f32 / device_sample_rate as f32);
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

            device.build_input_stream(
                &config.into(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let mut state = state_clone.lock();
                    if state.is_recording && state.recording_id == current_recording_id {
                        for &sample in data.iter().step_by(channels) {
                            state.audio_buffer.push(sample as f32 / 32768.0);
                        }
                        let mut count = callback_count.lock();
                        *count += 1;
                        if *count % 100 == 0 {
                            let now = chrono::Local::now();
                            println!("[{}] Audio callback #{}: received {} samples, buffer size: {} samples ({:.2}s)",
                                     now.format("%H:%M:%S"), *count, data.len(),
                                     state.audio_buffer.len(), state.audio_buffer.len() as f32 / device_sample_rate as f32);
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
    }.map_err(|e| format!("Failed to build stream: {}", e))?;

    stream.play().map_err(|e| format!("Failed to start stream: {}", e))?;

    {
        let mut state_guard = state.lock();
        state_guard.is_recording = true;
        state_guard.audio_buffer.clear();
        state_guard.sample_rate = device_sample_rate;
    }

    // Leak the stream to keep it alive - it will be invalidated by recording_id on next recording
    // This prevents Send trait issues while ensuring old callbacks can't write to new recordings
    std::mem::forget(stream);

    let now = chrono::Local::now();
    println!("[{}] Recording started successfully", now.format("%H:%M:%S"));

    Ok(())
}

#[tauri::command]
async fn stop_recording(_app_handle: AppHandle) -> Result<Vec<f32>, String> {
    let state = RECORDING_STATE.get().ok_or("Recording not initialized")?;

    let mut state_guard = state.lock();
    if !state_guard.is_recording {
        return Err("Not recording".to_string());
    }

    state_guard.is_recording = false;
    let device_sample_rate = state_guard.sample_rate;

    let now = chrono::Local::now();
    println!("[{}] Stopping recording, waiting for stream...", now.format("%H:%M:%S"));

    // Wait for stream thread to finish and drop the stream
    std::thread::sleep(std::time::Duration::from_millis(200));

    let audio_data = std::mem::take(&mut state_guard.audio_buffer);
    drop(state_guard);
    let sample_count = audio_data.len();

    let now = chrono::Local::now();
    println!("[{}] Recording stopped - captured {} samples ({:.2} seconds at {}Hz)",
             now.format("%H:%M:%S"), sample_count, sample_count as f32 / device_sample_rate as f32, device_sample_rate);

    if sample_count == 0 {
        return Err("No audio data captured".to_string());
    }

    // Check audio data statistics
    let non_zero_count = audio_data.iter().filter(|&&s| s.abs() > 0.0001).count();
    let max_amplitude = audio_data.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);
    let avg_amplitude = audio_data.iter().map(|&s| s.abs()).sum::<f32>() / audio_data.len() as f32;

    let now = chrono::Local::now();
    println!("[{}] Audio stats - Non-zero samples: {}/{} ({:.1}%), Max amplitude: {:.4}, Avg amplitude: {:.4}",
             now.format("%H:%M:%S"), non_zero_count, sample_count,
             (non_zero_count as f32 / sample_count as f32) * 100.0,
             max_amplitude, avg_amplitude);

    // Show first 10 samples
    let now = chrono::Local::now();
    print!("[{}] First 10 samples: ", now.format("%H:%M:%S"));
    for i in 0..10.min(audio_data.len()) {
        print!("{:.4} ", audio_data[i]);
    }
    println!();

    // Resample to 16kHz if needed
    let resampled_data = if device_sample_rate != 16000 {
        let now = chrono::Local::now();
        println!("[{}] Resampling from {}Hz to 16000Hz...", now.format("%H:%M:%S"), device_sample_rate);
        let resampled = resample_audio(&audio_data, device_sample_rate, 16000);

        // Check resampled data
        let max_resampled = resampled.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);
        let avg_resampled = resampled.iter().map(|&s| s.abs()).sum::<f32>() / resampled.len() as f32;
        let now = chrono::Local::now();
        println!("[{}] Resampled stats - Max: {:.4}, Avg: {:.4}",
                 now.format("%H:%M:%S"), max_resampled, avg_resampled);

        resampled
    } else {
        audio_data
    };

    let now = chrono::Local::now();
    println!("[{}] Final audio: {} samples ({:.2} seconds at 16kHz)",
             now.format("%H:%M:%S"), resampled_data.len(), resampled_data.len() as f32 / 16000.0);

    Ok(resampled_data)
}

#[tauri::command]
async fn list_audio_devices() -> Result<Vec<AudioDevice>, String> {
    let host = cpal::default_host();
    let default_device_name = host.default_input_device()
        .and_then(|d| d.name().ok());

    let devices: Vec<AudioDevice> = host.input_devices()
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
    let state = RECORDING_STATE.get_or_init(|| {
        Arc::new(ParkingMutex::new(RecordingState {
            is_recording: false,
            audio_buffer: Vec::new(),
            sample_rate: 16000,
            selected_device_name: None,
            recording_id: 0,
        }))
    });

    let mut state_guard = state.lock();
    state_guard.selected_device_name = Some(device_name.clone());

    let now = chrono::Local::now();
    println!("[{}] Selected audio device: {}", now.format("%H:%M:%S"), device_name);

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

// Linear resampling with simple low-pass pre-filter to avoid aliasing
fn resample_audio(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return input.to_vec();
    }

    // Simple moving-average low-pass filter when downsampling
    let filtered: Vec<f32> = if from_rate > to_rate {
        let window = (from_rate / to_rate).max(2) as usize;
        let mut acc = 0.0f32;
        let mut filtered = Vec::with_capacity(input.len());
        for (i, &sample) in input.iter().enumerate() {
            acc += sample;
            if i >= window {
                acc -= input[i - window];
            }
            filtered.push(acc / window as f32);
        }
        filtered
    } else {
        input.to_vec()
    };

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = ((filtered.len() as f64) / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos.floor() as usize;
        let frac = src_pos - idx as f64;

        if idx + 1 < filtered.len() {
            let sample = filtered[idx] as f64 * (1.0 - frac)
                + filtered[idx + 1] as f64 * frac;
            output.push(sample as f32);
        } else if idx < filtered.len() {
            output.push(filtered[idx]);
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
            transcribe_audio,
            transcribe_audio_stream,
            start_recording,
            stop_recording,
            get_supported_languages,
            list_audio_devices,
            select_audio_device,
            check_microphone_permission,
            list_remote_models,
            install_model,
            delete_model
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
