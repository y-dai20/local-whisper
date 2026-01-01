pub const VAD_SAMPLE_RATE: u32 = 16_000;
pub const VAD_CHUNK_SIZE: usize = 500;
pub const DEFAULT_VAD_THRESHOLD: f32 = 0.1;
pub const DEFAULT_PARTIAL_TRANSCRIPT_INTERVAL_SAMPLES: usize = 2 * VAD_SAMPLE_RATE as usize;
pub const SILENCE_TIMEOUT_SAMPLES: usize = 1 * VAD_SAMPLE_RATE as usize;
pub const VAD_PRE_BUFFER_MS: usize = 200;
pub const VAD_POST_BUFFER_MS: usize = 200;
pub const VAD_PRE_BUFFER_SAMPLES: usize = (VAD_SAMPLE_RATE as usize * VAD_PRE_BUFFER_MS) / 1000;
pub const VAD_POST_BUFFER_SAMPLES: usize = (VAD_SAMPLE_RATE as usize * VAD_POST_BUFFER_MS) / 1000;

pub fn calculate_session_max_samples(audio_ctx: i32) -> usize {
    let max_seconds = (audio_ctx as f32 / 1500.0 * 30.0).max(1.0);
    (max_seconds * VAD_SAMPLE_RATE as f32) as usize
}
