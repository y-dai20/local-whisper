use std::path::PathBuf;

use chrono::Local;
use hound::{SampleFormat, WavSpec, WavWriter};
use log::{debug, error, info};

use super::constants::{
    VAD_CHUNK_SIZE, VAD_POST_BUFFER_SAMPLES, VAD_PRE_BUFFER_SAMPLES, VAD_SAMPLE_RATE,
};
use crate::audio::state::{RecordingState, SileroVadState};
use crate::emit_voice_activity_event;
use crate::transcription::{worker::queue_transcription, TranscriptionSource};

pub fn finalize_session_common(
    audio: &[f32],
    session_id_counter: u64,
    reason: &str,
    source: &str,
    recording_save_enabled: bool,
    is_recording: bool,
    recording_dir: Option<&str>,
) {
    if audio.is_empty() {
        return;
    }

    info!(
        "Finalizing {} session #{} ({})",
        source, session_id_counter, reason,
    );

    if recording_save_enabled && is_recording {
        if let Some(dir) = recording_dir {
            let audio_clone = audio.to_vec();
            let audio_len = audio_clone.len();
            let duration = audio_len as f32 / VAD_SAMPLE_RATE as f32;
            let dir = dir.to_string();
            let source = source.to_string();

            info!(
                "Saving {} audio session #{}: {} samples ({:.2}s) to {}",
                source, session_id_counter, audio_len, duration, dir
            );

            std::thread::spawn(move || {
                if let Err(e) =
                    save_audio_session_to_wav(&audio_clone, session_id_counter, &dir, &source)
                {
                    error!("Failed to save {} audio session: {}", source, e);
                }
            });
        } else {
            info!("Recording save enabled but no recording directory set");
        }
    }
}

pub fn finalize_active_session(state: &mut RecordingState, reason: &str) {
    if state.session_audio.is_empty() {
        state.session_audio.clear();
        state.session_samples = 0;
        state.last_partial_emit_samples = 0;
        state.last_voice_sample = None;
        return;
    }

    let session_id_counter = state
        .transcription_state(TranscriptionSource::Mic)
        .session_id_counter;

    finalize_session_common(
        &state.session_audio,
        session_id_counter,
        reason,
        "mic",
        state.recording_save_enabled,
        state.is_recording,
        state.current_recording_dir.as_deref(),
    );

    if !state.suppress_transcription {
        queue_transcription(state, true);
    }

    state.session_audio.clear();
    state.session_samples = 0;
    state.last_partial_emit_samples = 0;
    state.last_voice_sample = None;
}

fn process_vad_chunk_only(
    vad_state: &mut SileroVadState,
    chunk: &[f32],
) -> Result<f32, String> {
    let chunk_i16: Vec<i16> = chunk
        .iter()
        .map(|&sample| (sample * i16::MAX as f32) as i16)
        .collect();
    let probability = vad_state.vad.predict(chunk_i16);
    Ok(probability)
}

pub fn push_sample_with_optional_vad(
    state: &mut RecordingState,
    sample: f32,
    app_handle: &tauri::AppHandle,
) {
    if state.suppress_transcription {
        return;
    }

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
                            if !vad_state.is_voice_active && !vad_state.pre_buffer.is_empty() {
                                state.audio_buffer.extend_from_slice(&vad_state.pre_buffer);
                                state.session_audio.extend_from_slice(&vad_state.pre_buffer);
                                vad_state.pre_buffer.clear();
                            }

                            state.audio_buffer.extend_from_slice(&chunk);
                            state.session_audio.extend_from_slice(&chunk);
                            voice_detected = true;
                            vad_state.is_voice_active = true;
                            vad_state.post_buffer_remaining = VAD_POST_BUFFER_SAMPLES;
                        } else {
                            if vad_state.is_voice_active && vad_state.post_buffer_remaining > 0 {
                                state.audio_buffer.extend_from_slice(&chunk);
                                state.session_audio.extend_from_slice(&chunk);
                                vad_state.post_buffer_remaining =
                                    vad_state.post_buffer_remaining.saturating_sub(chunk.len());

                                if vad_state.post_buffer_remaining == 0 {
                                    vad_state.is_voice_active = false;
                                }
                            } else {
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
                        error!("Silero VAD failed, disabling VAD: {}", err);
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

    let now = std::time::Instant::now();
    if now.duration_since(state.last_vad_event_time).as_millis() >= 500 {
        let is_voice_active = state
            .vad_state
            .as_ref()
            .map_or(false, |v| v.is_voice_active);
        let session_id = state
            .transcription_state(TranscriptionSource::Mic)
            .session_id_counter;
        emit_voice_activity_event(app_handle, "user", is_voice_active, session_id);
        state.last_vad_event_time = now;
    }

    if let Some(last_voice) = state.last_voice_sample {
        if state.session_samples - last_voice >= super::constants::SILENCE_TIMEOUT_SAMPLES {
            finalize_active_session(state, "silence_timeout");
        }
    }

    if state.session_samples - state.last_partial_emit_samples
        >= state.partial_transcript_interval_samples
    {
        if !state.suppress_transcription {
            queue_transcription(state, false);
        }
        state.last_partial_emit_samples = state.session_samples;
    }
}

pub fn save_audio_session_to_wav(
    audio_data: &[f32],
    session_id: u64,
    recording_dir: &str,
    source: &str,
) -> Result<(), String> {
    let start_time = std::time::Instant::now();
    let now = Local::now();
    let timestamp = now.format("%H%M%S");
    let filename = format!("{}_audio_session_{}_{}.wav", source, session_id, timestamp);

    let mut path = PathBuf::from(recording_dir);

    info!(
        "Saving {} audio to recording directory: {}",
        source,
        path.display()
    );

    if !path.exists() {
        debug!("Creating directory: {}", path.display());
        std::fs::create_dir_all(&path).map_err(|e| format!("Failed to create directory: {}", e))?;
    }

    path.push(filename);

    let spec = WavSpec {
        channels: 1,
        sample_rate: VAD_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    debug!("Creating WAV file: {}", path.display());

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

    info!(
        "Successfully saved {} audio session #{} to: {} ({} bytes, took {:.2}ms)",
        source,
        session_id,
        path.display(),
        file_size,
        elapsed.as_secs_f64() * 1000.0
    );

    Ok(())
}
