use hound::{WavSpec, WavWriter};
use parking_lot::Mutex as ParkingMutex;
use std::os::raw::c_int;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::AppHandle;
use voice_activity_detector::VoiceActivityDetector;

extern "C" {
    fn system_audio_start(callback: extern "C" fn(*const f32, c_int)) -> c_int;
    fn system_audio_stop() -> c_int;
}

struct SystemAudioSession {
    session_audio: Vec<f32>,
    vad: VoiceActivityDetector,
    vad_pending: Vec<f32>,
    vad_threshold: f32,
    session_samples: usize,
    last_voice_sample: Option<usize>,
    last_partial_emit_samples: usize,
    session_id_counter: u64,
    partial_transcript_interval_samples: usize,
    last_vad_event_time: std::time::Instant,
    is_voice_active: bool,
}

static mut SYSTEM_AUDIO_SESSION: Option<SystemAudioSession> = None;
static mut APP_HANDLE: Option<AppHandle> = None;
static mut RECORDING_STATE: Option<Arc<ParkingMutex<super::RecordingState>>> = None;
static SAMPLE_COUNTER: AtomicU64 = AtomicU64::new(0);
static SAMPLE_LOG_COUNT: AtomicU64 = AtomicU64::new(0);
static LAST_LOG_INSTANT: Mutex<Option<Instant>> = Mutex::new(None);
const LOG_INTERVAL: Duration = Duration::from_secs(2);
const VAD_CHUNK_SIZE: usize = 512;
const SILENCE_TIMEOUT_SAMPLES: usize = 1 * 16000;
const SESSION_MAX_SAMPLES: usize = 30 * 16000;

extern "C" fn audio_callback(samples: *const f32, count: c_int) {
    unsafe {
        let Some(session) = SYSTEM_AUDIO_SESSION.as_mut() else {
            return;
        };
        let Some(state_arc) = &RECORDING_STATE else {
            return;
        };
        let state = state_arc.lock();

        if !state.is_recording || !state.system_audio_enabled {
            return;
        }

        let app_handle = APP_HANDLE.as_ref();
        let language = state.language.clone();
        drop(state);

        let slice = std::slice::from_raw_parts(samples, count as usize);

        ensure_system_audio_session(session);

        let mut sum_squares = 0.0_f32;
        let mut max_sample = 0.0_f32;
        let mut non_zero_count = 0;
        for &sample in slice {
            sum_squares += sample * sample;
            max_sample = max_sample.max(sample.abs());
            if sample.abs() > 0.0001 {
                non_zero_count += 1;
            }
            process_system_audio_sample(session, sample, app_handle, language.as_deref());
        }

        let log_count = SAMPLE_LOG_COUNT.fetch_add(1, Ordering::Relaxed);
        if log_count < 5 {
            let preview: Vec<String> = slice.iter().take(10).map(|v| format!("{:.6}", v)).collect();
            info!(
                "System audio sample details: count={}, max={:.6}, non_zero={}/{}, preview=[{}]",
                count,
                max_sample,
                non_zero_count,
                count,
                preview.join(" ")
            );
        }

        let rms = if count > 0 {
            (sum_squares / count as f32).sqrt()
        } else {
            0.0
        };

        let total_samples =
            SAMPLE_COUNTER.fetch_add(count as u64, Ordering::Relaxed) + count as u64;
        let mut last_log = LAST_LOG_INSTANT.lock().unwrap();
        let now = Instant::now();
        let should_log = last_log
            .map(|ts| now.duration_since(ts) >= LOG_INTERVAL)
            .unwrap_or(true);

        if should_log {
            let total_seconds = total_samples as f32 / 16000.0;
            info!(
                "System audio: received {} samples (total {:.2}s, RMS {:.4})",
                count,
                total_seconds,
                rms
            );
            *last_log = Some(now);
        }
    }
}

fn ensure_system_audio_session(session: &mut SystemAudioSession) {
    if session.session_id_counter > 0 && !session.session_audio.is_empty() {
        return;
    }
    session.session_id_counter = session.session_id_counter.wrapping_add(1);
    session.session_audio.clear();
    session.session_samples = 0;
    session.last_partial_emit_samples = 0;
    session.last_voice_sample = None;
    info!("Starting SystemAudio session #{}", session.session_id_counter);
}

fn process_system_audio_sample(
    session: &mut SystemAudioSession,
    sample: f32,
    app_handle: Option<&AppHandle>,
    language: Option<&str>,
) {
    session.session_samples += 1;
    session.vad_pending.push(sample);

    let mut voice_detected = false;

    while session.vad_pending.len() >= VAD_CHUNK_SIZE {
        let chunk: Vec<f32> = session.vad_pending.drain(..VAD_CHUNK_SIZE).collect();
        let chunk_i16: Vec<i16> = chunk
            .iter()
            .map(|&s| (s * i16::MAX as f32) as i16)
            .collect();

        let probability = session.vad.predict(chunk_i16);

        if probability > session.vad_threshold {
            session.session_audio.extend_from_slice(&chunk);
            session.last_voice_sample = Some(session.session_samples);
            voice_detected = true;
            session.is_voice_active = true;
        } else {
            session.is_voice_active = false;
        }
    }

    // Emit periodic VAD voice activity events every 500ms
    if let Some(app) = app_handle {
        let now = std::time::Instant::now();
        if now.duration_since(session.last_vad_event_time).as_millis() >= 500 {
            super::emit_voice_activity_event(app, "system", session.is_voice_active);
            session.last_vad_event_time = now;
        }
    }

    if let Some(last_voice) = session.last_voice_sample {
        if session.session_samples - last_voice >= SILENCE_TIMEOUT_SAMPLES {
            finalize_system_audio_session(session, "silence_timeout", app_handle, language);
            ensure_system_audio_session(session);
        }
    }

    if session.session_samples >= SESSION_MAX_SAMPLES {
        finalize_system_audio_session(session, "session_max_duration", app_handle, language);
        ensure_system_audio_session(session);
    }

    if session.session_samples - session.last_partial_emit_samples
        >= session.partial_transcript_interval_samples
    {
        queue_system_audio_transcription(session, false, app_handle, language);
        session.last_partial_emit_samples = session.session_samples;
    }
}

fn queue_system_audio_transcription(
    session: &SystemAudioSession,
    is_final: bool,
    app_handle: Option<&AppHandle>,
    language: Option<&str>,
) {
    if session.session_audio.is_empty() {
        return;
    }
    if session.session_id_counter == 0 {
        return;
    }
    let Some(app) = app_handle else {
        return;
    };

    let audio = session.session_audio.clone();
    let lang = language.unwrap_or("ja").to_string();
    let app_clone = app.clone();
    let session_id = session.session_id_counter;

    std::thread::spawn(move || {
        if let Err(e) = transcribe_system_audio(&audio, &lang, session_id, is_final, &app_clone) {
            error!("System audio transcription error: {}", e);
        }
    });
}

fn finalize_system_audio_session(
    session: &mut SystemAudioSession,
    reason: &str,
    app_handle: Option<&AppHandle>,
    language: Option<&str>,
) {
    if session.session_id_counter == 0 || session.session_audio.is_empty() {
        session.session_audio.clear();
        session.session_samples = 0;
        session.last_partial_emit_samples = 0;
        session.last_voice_sample = None;
        return;
    }

    info!(
        "Finalizing SystemAudio session #{} ({})",
        session.session_id_counter,
        reason,
    );

    queue_system_audio_transcription(session, true, app_handle, language);

    unsafe {
        if let Some(state_arc) = &RECORDING_STATE {
            let state = state_arc.lock();
            let save_enabled = state.recording_save_enabled;
            let recording_dir = state.current_recording_dir.clone();
            let is_recording = state.is_recording;
            drop(state);

            // 録音セッション中のみ音声を保存
            if save_enabled && is_recording {
                if let Some(dir) = recording_dir {
                    let audio_clone = session.session_audio.clone();
                    let session_id = session.session_id_counter;
                    let audio_len = audio_clone.len();
                    let duration = audio_len as f32 / 16000.0;

                    info!(
                        "Saving system audio session #{}: {} samples ({:.2}s) to {}",
                        session_id,
                        audio_len,
                        duration,
                        dir
                    );

                    std::thread::spawn(move || {
                        if let Err(e) = save_session_audio_to_wav(&audio_clone, session_id, &dir) {
                            error!("Failed to save system audio session: {}", e);
                        }
                    });
                } else {
                    info!("Recording save enabled but no recording directory set");
                }
            }
        }
    }

    session.session_audio.clear();
    session.session_samples = 0;
    session.last_partial_emit_samples = 0;
    session.last_voice_sample = None;
}

fn transcribe_system_audio(
    audio_data: &[f32],
    language: &str,
    session_id: u64,
    is_final: bool,
    app_handle: &AppHandle,
) -> Result<(), String> {
    use super::WHISPER_CTX;

    let ctx_lock = WHISPER_CTX
        .get()
        .ok_or_else(|| "Whisper not initialized".to_string())?
        .clone();
    let ctx_guard = ctx_lock.lock().unwrap();
    let ctx = ctx_guard
        .as_ref()
        .ok_or_else(|| "Whisper context not available".to_string())?;

    let text = ctx
        .transcribe_with_language(audio_data, language)
        .map_err(|e| e.to_string())?;

    if text.trim().is_empty() {
        return Ok(());
    }

    let session_id_str = format!("system_{}", session_id);

    // Use emit_transcription_segment to save to txt file
    super::emit_transcription_segment(
        app_handle,
        text,
        if is_final {
            Some(audio_data.to_vec())
        } else {
            None
        },
        session_id_str,
        is_final,
        "system".to_string(),
    )
}

pub fn start_system_audio_capture(
    state: Arc<ParkingMutex<super::RecordingState>>,
    app_handle: AppHandle,
) -> Result<(), String> {
    unsafe {
        let state_guard = state.lock();
        let vad_threshold = state_guard.vad_threshold;
        let partial_interval = state_guard.partial_transcript_interval_samples;
        drop(state_guard);

        let vad = VoiceActivityDetector::builder()
            .sample_rate(16000)
            .chunk_size(VAD_CHUNK_SIZE)
            .build()
            .map_err(|e| format!("Failed to initialize VAD: {:?}", e))?;

        SYSTEM_AUDIO_SESSION = Some(SystemAudioSession {
            session_audio: Vec::new(),
            vad,
            vad_pending: Vec::new(),
            vad_threshold,
            session_samples: 0,
            last_voice_sample: None,
            last_partial_emit_samples: 0,
            session_id_counter: 0,
            partial_transcript_interval_samples: partial_interval,
            last_vad_event_time: std::time::Instant::now(),
            is_voice_active: false,
        });

        APP_HANDLE = Some(app_handle);
        RECORDING_STATE = Some(state.clone());
        SAMPLE_COUNTER.store(0, Ordering::Relaxed);

        let result = system_audio_start(audio_callback);

        if result == 0 {
            let mut state_guard = state.lock();
            state_guard.system_audio_enabled = true;

            info!("System audio capture started");

            Ok(())
        } else if result == -2 {
            Err("System audio capture requires macOS 12.3+".to_string())
        } else {
            Err("Failed to start system audio capture".to_string())
        }
    }
}

pub fn stop_system_audio_capture(
    state: Arc<ParkingMutex<super::RecordingState>>,
) -> Result<(), String> {
    unsafe {
        let result = system_audio_stop();

        let mut state_guard = state.lock();
        state_guard.system_audio_enabled = false;

        SYSTEM_AUDIO_SESSION = None;
        APP_HANDLE = None;
        RECORDING_STATE = None;

        info!("System audio capture stopped");

        if result == 0 {
            Ok(())
        } else {
            Err("Failed to stop system audio capture".to_string())
        }
    }
}

pub fn update_language(
    language: Option<String>,
) -> Result<(), String> {
    unsafe {
        if let Some(state_arc) = &RECORDING_STATE {
            let mut state_guard = state_arc.lock();
            state_guard.language = language.clone();
            drop(state_guard);

            let lang_str = language.as_deref().unwrap_or("auto");
            info!("System Language updated to: {}", lang_str);
        }
    }

    Ok(())
}

fn save_session_audio_to_wav(
    audio_data: &[f32],
    session_id: u64,
    recording_dir: &str,
) -> Result<(), String> {
    let start_time = std::time::Instant::now();
    let now = chrono::Local::now();
    let timestamp = now.format("%H%M%S");
    let filename = format!("system_audio_session_{}_{}.wav", session_id, timestamp);

    let mut path = PathBuf::from(recording_dir);

    info!(
        "Saving system audio to recording directory: {}",
        path.display()
    );

    if !path.exists() {
        info!(
            "Creating directory: {}",
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

    info!(
        "Creating WAV file: {}",
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

    info!(
        "Successfully saved system audio session #{} to: {} ({} bytes, took {:.2}ms)",
        session_id,
        path.display(),
        file_size,
        elapsed.as_secs_f64() * 1000.0
    );

    Ok(())
}
