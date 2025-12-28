use serde::{Deserialize, Serialize};

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

#[derive(Debug)]
pub enum TranscriptionCommand {
    Run {
        audio: Vec<f32>,
        language: Option<String>,
        session_id: String,
        is_final: bool,
    },
    Stop,
}
