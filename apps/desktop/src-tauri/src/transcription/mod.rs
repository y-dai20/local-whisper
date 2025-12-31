use serde::{Deserialize, Serialize};

pub mod worker;

pub use worker::spawn_transcription_worker;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TranscriptionSource {
    Mic,
    System,
}

impl TranscriptionSource {
    pub fn session_prefix(self) -> &'static str {
        match self {
            TranscriptionSource::Mic => "mic",
            TranscriptionSource::System => "system",
        }
    }

    pub fn event_source(self) -> &'static str {
        match self {
            TranscriptionSource::Mic => "user",
            TranscriptionSource::System => "system",
        }
    }
}

#[derive(Debug, Clone)]
pub enum TranscriptionCommand {
    Run {
        audio: Vec<f32>,
        language: Option<String>,
        source: TranscriptionSource,
        session_id_counter: u64,
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
