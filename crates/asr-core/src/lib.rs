use std::path::Path;
use std::time::Instant;
use thiserror::Error;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext as WhisperRsContext, WhisperContextParameters,
};

#[derive(Error, Debug)]
pub enum WhisperError {
    #[error("Failed to initialize context: {0}")]
    InitializationFailed(String),
    #[error("Failed to process audio: {0}")]
    ProcessingFailed(String),
    #[error("Invalid model path")]
    InvalidModelPath,
}

pub struct WhisperContext {
    ctx: WhisperRsContext,
}

impl WhisperContext {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, WhisperError> {
        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.use_gpu(true).flash_attn(true);

        let ctx = WhisperRsContext::new_with_params(
            model_path.as_ref().to_str().ok_or(WhisperError::InvalidModelPath)?,
            ctx_params,
        )
            .map_err(|e| WhisperError::InitializationFailed(e.to_string()))?;

        Ok(Self { ctx })
    }

    pub fn transcribe(&self, audio_data: &[f32]) -> Result<String, WhisperError> {
        self.transcribe_with_language(audio_data, "ja")
    }

    pub fn transcribe_with_language(&self, audio_data: &[f32], language: &str) -> Result<String, WhisperError> {
        let mut state = self.ctx.create_state()
            .map_err(|e| WhisperError::ProcessingFailed(e.to_string()))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_translate(false);
        params.set_language(Some(language));
        params.set_n_threads(num_cpus());
        params.set_single_segment(false);

        let inference_start = Instant::now();
        state.full(params, audio_data)
            .map_err(|e| WhisperError::ProcessingFailed(e.to_string()))?;
        let inference_elapsed = inference_start.elapsed();
        println!(
            "[Whisper] Inference (streaming) completed in {:.2?} for {} samples",
            inference_elapsed,
            audio_data.len()
        );

        let num_segments = state.full_n_segments()
            .map_err(|e| WhisperError::ProcessingFailed(e.to_string()))?;

        let mut full_text = String::new();
        for i in 0..num_segments {
            let segment = state.full_get_segment_text(i)
                .map_err(|e| WhisperError::ProcessingFailed(e.to_string()))?;
            full_text.push_str(&segment);
        }

        Ok(full_text)
    }

    pub fn transcribe_with_callback<F>(
        &self,
        audio_data: &[f32],
        callback: F,
    ) -> Result<(), WhisperError>
    where
        F: FnMut(&str),
    {
        self.transcribe_with_callback_and_language(audio_data, "ja", callback)
    }

    pub fn transcribe_with_callback_and_language<F>(
        &self,
        audio_data: &[f32],
        language: &str,
        mut callback: F,
    ) -> Result<(), WhisperError>
    where
        F: FnMut(&str),
    {
        let mut state = self.ctx.create_state()
            .map_err(|e| WhisperError::ProcessingFailed(e.to_string()))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_translate(false);
        params.set_language(Some(language));
        params.set_n_threads(num_cpus());
        params.set_single_segment(false);

        let inference_start = Instant::now();
        state.full(params, audio_data)
            .map_err(|e| WhisperError::ProcessingFailed(e.to_string()))?;
        let inference_elapsed = inference_start.elapsed();
        println!(
            "[Whisper] Inference (streaming) completed in {:.2?} for {} samples",
            inference_elapsed,
            audio_data.len()
        );

        let num_segments = state.full_n_segments()
            .map_err(|e| WhisperError::ProcessingFailed(e.to_string()))?;

        for i in 0..num_segments {
            if let Ok(segment) = state.full_get_segment_text(i) {
                callback(&segment);
            }
        }

        Ok(())
    }
}

fn num_cpus() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
}

pub fn convert_pcm_to_f32(pcm_data: &[i16]) -> Vec<f32> {
    pcm_data
        .iter()
        .map(|&sample| sample as f32 / 32768.0)
        .collect()
}
