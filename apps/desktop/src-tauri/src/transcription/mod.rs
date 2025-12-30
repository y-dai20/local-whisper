use serde::{Deserialize, Serialize};

pub mod worker;

pub use worker::{spawn_transcription_worker, transcribe_and_emit_common};

#[derive(Debug, Clone)]
pub enum TranscriptionCommand {
    Run {
        audio: Vec<f32>,
        language: Option<String>,
        session_id: String,
        is_final: bool,
    },
    Stop,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TranscriptionSegment {
    pub text: String,
    pub timestamp: u64,
    #[serde(rename = "audioData")]
    pub audio_data: Option<Vec<f32>>,
    #[serde(rename = "sessionId")]
    pub session_id: String,
    #[serde(rename = "isFinal")]
    pub is_final: bool,
    pub source: String,
}
