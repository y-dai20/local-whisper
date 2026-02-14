use std::sync::{mpsc, Arc};
use std::thread::JoinHandle;
use std::time::Instant;

use asr_core::WhisperParams;
use once_cell::sync::OnceCell;
use parking_lot::Mutex as ParkingMutex;
use voice_activity_detector::VoiceActivityDetector;

use crate::transcription::{TranscriptionCommand, TranscriptionSource};

use super::constants::{
    calculate_session_max_samples, DEFAULT_PARTIAL_TRANSCRIPT_INTERVAL_SAMPLES,
    DEFAULT_VAD_THRESHOLD, VAD_SAMPLE_RATE,
};

#[derive(Debug)]
pub struct SileroVadState {
    pub vad: VoiceActivityDetector,
    pub pending: Vec<f32>,
    pub threshold: f32,
    pub pre_buffer: Vec<f32>,
    pub post_buffer_remaining: usize,
    pub is_voice_active: bool,
}

#[derive(Debug, Default, Clone)]
pub struct SourceTranscriptionState {
    pub session_id_counter: u64,
    pub transcribed_samples: usize,
    pub message_id_counter: u64,
}

#[derive(Debug)]
pub struct RecordingState {
    pub is_recording: bool,
    pub is_muted: bool,
    pub mic_stream_id: u64,
    pub audio_buffer: Vec<f32>,
    pub session_audio: Vec<f32>,
    pub sample_rate: u32,
    pub selected_device_name: Option<String>,
    pub vad_state: Option<SileroVadState>,
    pub session_samples: usize,
    pub last_voice_sample: Option<usize>,
    pub last_partial_emit_samples: usize,
    pub transcription_tx: Option<mpsc::Sender<TranscriptionCommand>>,
    pub transcription_handle: Option<JoinHandle<()>>,
    pub language: Option<String>,
    pub vad_threshold: f32,
    pub partial_transcript_interval_samples: usize,
    pub system_audio_enabled: bool,
    pub transcription_mode: String,
    pub recording_save_enabled: bool,
    pub screen_recording_enabled: bool,
    pub screen_recording_active: bool,
    pub suppress_transcription: bool,
    pub current_recording_dir: Option<String>,
    pub last_vad_event_time: Instant,
    pub session_max_samples: usize,
    pub mic_transcription: SourceTranscriptionState,
    pub system_transcription: SourceTranscriptionState,
}

pub fn default_recording_state() -> RecordingState {
    let default_params = WhisperParams::default();
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
        transcription_tx: None,
        transcription_handle: None,
        language: None,
        vad_threshold: DEFAULT_VAD_THRESHOLD,
        partial_transcript_interval_samples: DEFAULT_PARTIAL_TRANSCRIPT_INTERVAL_SAMPLES,
        system_audio_enabled: false,
        transcription_mode: "local".to_string(),
        recording_save_enabled: false,
        screen_recording_enabled: false,
        screen_recording_active: false,
        suppress_transcription: false,
        current_recording_dir: None,
        last_vad_event_time: Instant::now(),
        session_max_samples: calculate_session_max_samples(default_params.audio_ctx),
        mic_transcription: SourceTranscriptionState::default(),
        system_transcription: SourceTranscriptionState::default(),
    }
}

impl RecordingState {
    pub fn transcription_state(
        &self,
        source: TranscriptionSource,
    ) -> &SourceTranscriptionState {
        match source {
            TranscriptionSource::Mic => &self.mic_transcription,
            TranscriptionSource::System => &self.system_transcription,
        }
    }

    pub fn transcription_state_mut(
        &mut self,
        source: TranscriptionSource,
    ) -> &mut SourceTranscriptionState {
        match source {
            TranscriptionSource::Mic => &mut self.mic_transcription,
            TranscriptionSource::System => &mut self.system_transcription,
        }
    }
}

static RECORDING_STATE: OnceCell<Arc<ParkingMutex<RecordingState>>> = OnceCell::new();

pub fn recording_state() -> Arc<ParkingMutex<RecordingState>> {
    RECORDING_STATE
        .get_or_init(|| Arc::new(ParkingMutex::new(default_recording_state())))
        .clone()
}

pub fn try_recording_state() -> Option<Arc<ParkingMutex<RecordingState>>> {
    RECORDING_STATE.get().cloned()
}
