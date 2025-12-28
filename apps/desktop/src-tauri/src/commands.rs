use crate::*;
use tauri::{Builder, Runtime};

#[tauri::command]
async fn scan_models() -> Result<Vec<ModelInfo>, String> {
    crate::scan_models_impl().await
}

#[tauri::command]
async fn initialize_whisper(model_path: String) -> Result<String, String> {
    crate::initialize_whisper_impl(model_path).await
}

#[tauri::command]
async fn get_whisper_params() -> Result<WhisperParamsConfig, String> {
    crate::get_whisper_params_impl().await
}

#[tauri::command]
async fn set_whisper_params(config: WhisperParamsConfig) -> Result<(), String> {
    crate::set_whisper_params_impl(config).await
}

#[tauri::command]
async fn list_remote_models() -> Result<Vec<RemoteModelStatus>, String> {
    crate::list_remote_models_impl().await
}

#[tauri::command]
async fn install_model(model_id: String) -> Result<ModelInfo, String> {
    crate::install_model_impl(model_id).await
}

#[tauri::command]
async fn delete_model(model_path: String) -> Result<(), String> {
    crate::delete_model_impl(model_path).await
}

#[tauri::command]
async fn start_recording(language: Option<String>) -> Result<(), String> {
    crate::start_recording_impl(language).await
}

#[tauri::command]
async fn update_language(language: Option<String>) -> Result<(), String> {
    crate::update_language_impl(language).await
}

#[tauri::command]
async fn stop_recording() -> Result<(), String> {
    crate::stop_recording_impl().await
}

#[tauri::command]
async fn start_mic(language: Option<String>) -> Result<(), String> {
    crate::start_mic_impl(language).await
}

#[tauri::command]
async fn stop_mic() -> Result<(), String> {
    crate::stop_mic_impl().await
}

#[tauri::command]
async fn get_mic_status() -> Result<bool, String> {
    crate::get_mic_status_impl().await
}

#[tauri::command]
async fn list_audio_devices() -> Result<Vec<AudioDevice>, String> {
    crate::list_audio_devices_impl().await
}

#[tauri::command]
async fn select_audio_device(device_name: String) -> Result<(), String> {
    crate::select_audio_device_impl(device_name).await
}

#[tauri::command]
async fn get_streaming_config() -> Result<StreamingConfig, String> {
    crate::get_streaming_config_impl().await
}

#[tauri::command]
async fn set_streaming_config(config: StreamingConfig) -> Result<(), String> {
    crate::set_streaming_config_impl(config).await
}

#[tauri::command]
async fn check_microphone_permission() -> Result<bool, String> {
    crate::check_microphone_permission_impl().await
}

#[tauri::command]
async fn start_system_audio() -> Result<(), String> {
    crate::start_system_audio_impl().await
}

#[tauri::command]
async fn stop_system_audio() -> Result<(), String> {
    crate::stop_system_audio_impl().await
}

#[tauri::command]
async fn get_system_audio_status() -> Result<bool, String> {
    crate::get_system_audio_status_impl().await
}

#[tauri::command]
async fn set_recording_save_config(enabled: bool, path: Option<String>) -> Result<(), String> {
    crate::set_recording_save_config_impl(enabled, path).await
}

#[tauri::command]
async fn get_recording_save_config() -> Result<(bool, Option<String>), String> {
    crate::get_recording_save_config_impl().await
}

#[tauri::command]
async fn set_screen_recording_config(enabled: bool) -> Result<(), String> {
    crate::set_screen_recording_config_impl(enabled).await
}

#[tauri::command]
async fn get_screen_recording_config() -> Result<bool, String> {
    crate::get_screen_recording_config_impl().await
}

#[tauri::command]
async fn start_screen_recording() -> Result<(), String> {
    crate::start_screen_recording_impl().await
}

#[tauri::command]
async fn stop_screen_recording() -> Result<(), String> {
    crate::stop_screen_recording_impl().await
}

#[tauri::command]
async fn get_screen_recording_status() -> Result<bool, String> {
    crate::get_screen_recording_status_impl().await
}

#[tauri::command]
async fn get_supported_languages() -> Result<Vec<(String, String)>, String> {
    crate::get_supported_languages_impl().await
}

pub fn register<R: Runtime>(builder: Builder<R>) -> Builder<R> {
    builder.invoke_handler(tauri::generate_handler![
        scan_models,
        initialize_whisper,
        get_whisper_params,
        set_whisper_params,
        list_remote_models,
        install_model,
        delete_model,
        start_recording,
        update_language,
        stop_recording,
        start_mic,
        stop_mic,
        get_mic_status,
        list_audio_devices,
        select_audio_device,
        get_streaming_config,
        set_streaming_config,
        check_microphone_permission,
        start_system_audio,
        stop_system_audio,
        get_system_audio_status,
        set_recording_save_config,
        get_recording_save_config,
        set_screen_recording_config,
        get_screen_recording_config,
        start_screen_recording,
        stop_screen_recording,
        get_screen_recording_status,
        get_supported_languages,
    ])
}
