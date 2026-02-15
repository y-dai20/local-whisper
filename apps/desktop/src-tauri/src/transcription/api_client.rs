use tauri::AppHandle;

use super::websocket_client;
use super::TranscriptionSource;

pub fn stream_audio_chunk_and_emit(
    audio_16k_f32: &[f32],
    language: &str,
    source: TranscriptionSource,
    session_id_counter: u64,
    is_final: bool,
    app_handle: &AppHandle,
) -> Result<(), String> {
    websocket_client::stream_audio_chunk_and_emit(
        audio_16k_f32,
        language,
        source,
        session_id_counter,
        is_final,
        app_handle,
    )
}

pub fn reset_all_connections(app_handle: Option<&AppHandle>) {
    websocket_client::reset_all_connections(app_handle);
}
