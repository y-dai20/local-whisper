use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use log::{error, info};
use parking_lot::Mutex as ParkingMutex;
use std::sync::Arc;
use tauri::AppHandle;
use voice_activity_detector::VoiceActivityDetector;

use crate::audio::constants::{VAD_CHUNK_SIZE, VAD_SAMPLE_RATE};
use crate::audio::processing::finalize_active_session;
use crate::audio::state::{recording_state, RecordingState, SileroVadState};
use crate::audio::utils::resample_audio;
use crate::transcription::{spawn_transcription_worker, TranscriptionCommand};

pub fn stop_transcription_worker(state: &mut RecordingState) {
    if let Some(tx) = state.transcription_tx.take() {
        let _ = tx.send(TranscriptionCommand::Stop);
    }
    if let Some(handle) = state.transcription_handle.take() {
        let _ = handle.join();
    }
}

pub async fn start_mic_stream(app_handle: AppHandle, language: Option<String>) -> Result<(), String> {
    let state = recording_state();

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
        state_guard.sample_rate = VAD_SAMPLE_RATE;

        info!("Starting mic stream #{}", current_mic_stream_id);

        (
            selected_device_name,
            current_mic_stream_id,
            configured_vad_threshold,
        )
    };

    let host = cpal::default_host();

    let device = if let Some(device_name) = &selected_device_name {
        info!("Looking for device: {}", device_name);

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
    info!(
        "Using input device: {}{}",
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

    info!(
        "Recording config - Device sample rate: {}, Channels: {}, Format: {:?}",
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
            info!("Voice Activity Detector initialized");
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
            info!("Failed to initialize VAD: {err:?}. Falling back to raw audio.",);
            None
        }
    };

    {
        let mut state_guard = state.lock();
        state_guard.vad_state = vad_state;
    }

    info!(
        "Building audio stream for format {:?}",
        config.sample_format()
    );

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => {
            let state_clone = state.clone();
            let callback_count = Arc::new(ParkingMutex::new(0u64));

            let app_handle_clone = app_handle.clone();
            device.build_input_stream(
                &config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mut state = state_clone.lock();
                    if !state.is_muted && state.mic_stream_id == current_mic_stream_id {
                        let mono_samples: Vec<f32> = data.iter().step_by(channels).copied().collect();
                        let processed_samples = if device_sample_rate == VAD_SAMPLE_RATE {
                            mono_samples
                        } else {
                            resample_audio(&mono_samples, device_sample_rate, VAD_SAMPLE_RATE)
                        };
                        for sample in processed_samples {
                            crate::audio::processing::push_sample_with_optional_vad(&mut state, sample, &app_handle_clone);
                        }

                        if state.session_samples >= state.session_max_samples {
                            finalize_active_session(&mut state, "session_max_duration");
                        }
                        let mut count = callback_count.lock();
                        *count += 1;
                        if *count % 100 == 0 {
                            info!("Audio callback #{}: received {} samples, buffer size: {} samples ({:.2}s)",
                                     *count, data.len(),
                                     state.audio_buffer.len(), state.audio_buffer.len() as f32 / VAD_SAMPLE_RATE as f32);
                        }
                    }
                },
                |err| error!("Error in audio stream: {}", err),
                None,
            )
        },
        _ => {
            return Err("Unsupported sample format".to_string());
        }
    }.map_err(|e| {
        info!(
            "Failed to build audio stream: {}",
            e
        );
        format!("Failed to build stream: {}", e)
    })?;

    info!("Starting audio stream…");

    stream.play().map_err(|e| {
        info!("Failed to start audio stream: {}", e);
        format!("Failed to start stream: {}", e)
    })?;

    info!("Audio stream started successfully",);

    {
        let mut state_guard = state.lock();
        state_guard.audio_buffer.clear();
        state_guard.sample_rate = VAD_SAMPLE_RATE;
    }

    std::mem::forget(stream);

    info!("Mic stream started successfully",);

    Ok(())
}
