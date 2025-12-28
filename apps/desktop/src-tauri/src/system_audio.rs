use crate::audio::{
    save_audio_session_to_wav, try_recording_state, RecordingState, SILENCE_TIMEOUT_SAMPLES,
    VAD_CHUNK_SIZE, VAD_SAMPLE_RATE,
};
use log::{error, info};
use parking_lot::Mutex as ParkingMutex;
use std::os::raw::c_int;
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
static mut RECORDING_STATE: Option<Arc<ParkingMutex<RecordingState>>> = None;
static SAMPLE_COUNTER: AtomicU64 = AtomicU64::new(0);
static SAMPLE_LOG_COUNT: AtomicU64 = AtomicU64::new(0);
static LAST_LOG_INSTANT: Mutex<Option<Instant>> = Mutex::new(None);
const LOG_INTERVAL: Duration = Duration::from_secs(2);
fn current_session_max_samples() -> usize {
    try_recording_state()
        .map(|state| state.lock().session_max_samples)
        .unwrap_or(30 * VAD_SAMPLE_RATE as usize)
}

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
                count, total_seconds, rms
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
    info!(
        "Starting SystemAudio session #{}",
        session.session_id_counter
    );
}

fn process_system_audio_sample(
    session: &mut SystemAudioSession,
    sample: f32,
    app_handle: Option<&AppHandle>,
    language: Option<&str>,
) {
    session.session_samples += 1;
    session.vad_pending.push(sample);

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

    if session.session_samples >= current_session_max_samples() {
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

    queue_system_audio_transcription(session, true, app_handle, language);

    unsafe {
        if let Some(state_arc) = &RECORDING_STATE {
            let state = state_arc.lock();
            let save_enabled = state.recording_save_enabled;
            let recording_dir = state.current_recording_dir.clone();
            let is_recording = state.is_recording;
            drop(state);

            crate::audio::finalize_session_common(
                &session.session_audio,
                session.session_id_counter,
                reason,
                "system",
                save_enabled,
                is_recording,
                recording_dir.as_deref(),
            );
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
    super::transcribe_and_emit_common(
        audio_data,
        language,
        "system",
        session_id,
        is_final,
        app_handle,
        "system",
        Some(&|new_counter| unsafe {
            if let Some(session) = SYSTEM_AUDIO_SESSION.as_mut() {
                session.session_id_counter = new_counter;
            }
        }),
    )?;

    Ok(())
}

pub fn start_system_audio_capture(
    state: Arc<ParkingMutex<RecordingState>>,
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

pub fn stop_system_audio_capture(state: Arc<ParkingMutex<RecordingState>>) -> Result<(), String> {
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

pub fn update_language(language: Option<String>) -> Result<(), String> {
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

