use asr_core::{WhisperContext, WhisperParams};
use log::info;
use once_cell::sync::OnceCell;
use parking_lot::Mutex as ParkingMutex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::fs;
use tokio::io::AsyncWriteExt;

use crate::audio::constants::{self, VAD_SAMPLE_RATE};
use crate::audio::state::try_recording_state;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub path: String,
    pub size: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RemoteModelStatus {
    pub id: String,
    pub name: String,
    pub filename: String,
    pub size: u64,
    pub description: String,
    pub installed: bool,
    pub path: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct WhisperParamsConfig {
    #[serde(rename = "audioCtx")]
    pub audio_ctx: i32,
    pub temperature: f32,
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

#[derive(Debug)]
pub struct RemoteModel {
    pub id: &'static str,
    pub name: &'static str,
    pub filename: &'static str,
    pub size: u64,
    pub description: &'static str,
    pub url: &'static str,
}

pub static WHISPER_CTX: OnceCell<Arc<Mutex<Option<WhisperContext>>>> = OnceCell::new();
pub static WHISPER_PARAMS: OnceCell<Arc<ParkingMutex<WhisperParams>>> = OnceCell::new();

pub const REMOTE_MODELS: &[RemoteModel] = &[
    RemoteModel {
        id: "base",
        name: "Base",
        filename: "ggml-base.bin",
        size: 74_438_528,
        description: "英語・多言語兼用 / 約 74 MB",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
    },
    RemoteModel {
        id: "small",
        name: "Small",
        filename: "ggml-small.bin",
        size: 244_452_544,
        description: "中規模モデル / 約 244 MB",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
    },
    RemoteModel {
        id: "medium",
        name: "Medium",
        filename: "ggml-medium.bin",
        size: 769_073_152,
        description: "高精度モデル / 約 769 MB",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
    },
    RemoteModel {
        id: "large-v3-turbo",
        name: "Large v3 Turbo",
        filename: "ggml-large-v3-turbo.bin",
        size: 3_085_627_392,
        description: "最新 Large モデル / 約 3.1 GB",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
    },
];

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

fn whisper_params_state() -> Arc<ParkingMutex<WhisperParams>> {
    WHISPER_PARAMS
        .get_or_init(|| Arc::new(ParkingMutex::new(WhisperParams::default())))
        .clone()
}

pub async fn scan_models_impl() -> Result<Vec<ModelInfo>, String> {
    read_installed_models()
}

pub async fn initialize_whisper_impl(model_path: String) -> Result<String, String> {
    let params_state = whisper_params_state();

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

pub async fn get_whisper_params_impl() -> Result<WhisperParamsConfig, String> {
    let state = whisper_params_state();
    let guard = state.lock();
    Ok(WhisperParamsConfig::from(*guard))
}

pub async fn set_whisper_params_impl(config: WhisperParamsConfig) -> Result<(), String> {
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

    let new_max_samples = constants::calculate_session_max_samples(params.audio_ctx);
    if let Some(recording_state) = try_recording_state() {
        let mut state_guard = recording_state.lock();
        state_guard.session_max_samples = new_max_samples;
    }

    info!(
        "Updated Whisper params: audio_ctx {}, temperature {:.2}, max session duration {:.1}s",
        params.audio_ctx,
        params.temperature,
        new_max_samples as f32 / VAD_SAMPLE_RATE as f32
    );

    Ok(())
}

pub async fn list_remote_models_impl() -> Result<Vec<RemoteModelStatus>, String> {
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

pub async fn install_model_impl(model_id: String) -> Result<ModelInfo, String> {
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

pub async fn delete_model_impl(model_path: String) -> Result<(), String> {
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
