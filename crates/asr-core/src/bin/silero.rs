use anyhow::{Context, Result};
use hound::WavReader;
use voice_activity_detector::VoiceActivityDetector;

const SAMPLE_RATE: u32 = 16_000;
const CHUNK_SIZE: usize = 512;
const THRESHOLD: f32 = 0.5;
const AUDIO_PATH: &str = "/Users/yamada/Documents/Assets/test16.wav";

fn main() -> Result<()> {
    println!("Initializing Voice Activity Detector");
    let mut vad = VoiceActivityDetector::builder()
        .sample_rate(SAMPLE_RATE as i32)
        .chunk_size(CHUNK_SIZE)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build VAD: {:?}", e))?;

    println!("Reading audio from {}", AUDIO_PATH);
    let audio = read_mono_audio(AUDIO_PATH, SAMPLE_RATE)?;

    let mut segments = Vec::new();
    let mut current_start = None;
    let mut last_active_time = 0.0f32;
    let chunk_duration = CHUNK_SIZE as f32 / SAMPLE_RATE as f32;
    let mut current_time = 0.0f32;

    for chunk_slice in audio.chunks(CHUNK_SIZE) {
        if chunk_slice.len() < CHUNK_SIZE {
            break;
        }
        let chunk_i16: Vec<i16> = chunk_slice
            .iter()
            .map(|&sample| (sample * i16::MAX as f32) as i16)
            .collect();
        let prob = vad.predict(chunk_i16);
        println!("probability: {}", prob);
        if prob >= THRESHOLD {
            if current_start.is_none() {
                current_start = Some(current_time);
            }
            last_active_time = current_time + chunk_duration;
        } else if let Some(start) = current_start {
            segments.push((start, last_active_time));
            current_start = None;
        }

        current_time += chunk_duration;
    }

    if let Some(start) = current_start {
        segments.push((start, last_active_time));
    }

    if segments.is_empty() {
        println!("No speech segments detected.");
    } else {
        println!("Detected segments:");
        for (i, (start, end)) in segments.iter().enumerate() {
            println!("  #{i}: {:.2} - {:.2} sec", start, end);
        }
    }

    Ok(())
}

fn read_mono_audio(path: &str, expected_rate: u32) -> Result<Vec<f32>> {
    let mut reader = WavReader::open(path).context("Failed to open WAV file")?;
    let spec = reader.spec();
    if spec.channels != 1 {
        anyhow::bail!("expected mono audio, got {} channels", spec.channels);
    }
    if spec.sample_rate != expected_rate {
        anyhow::bail!(
            "expected {expected_rate} Hz audio, got {} Hz",
            spec.sample_rate
        );
    }

    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s: Result<f32, _>| s.map_err(anyhow::Error::from))
            .collect::<Result<Vec<_>>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s: Result<i16, _>| Ok(s? as f32 / i16::MAX as f32))
            .collect::<Result<Vec<_>>>()?,
    };
    Ok(samples)
}
