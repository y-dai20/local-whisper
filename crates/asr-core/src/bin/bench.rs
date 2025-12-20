use std::{env, error::Error, path::Path, time::Instant};

use asr_core::WhisperContext;
use hound::{SampleFormat, WavReader};

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!(
            "Usage: cargo run -p asr-core --bin bench -- <MODEL_PATH> <AUDIO_PATH> [LANGUAGE]"
        );
        return Err("missing arguments".into());
    }

    let model_path = &args[1];
    let audio_path = &args[2];
    let language = args.get(3).map(String::as_str).unwrap_or("ja");

    println!("Model:   {}", model_path);
    println!("Audio:   {}", audio_path);
    println!("Language: {}", language);

    let model_load_start = Instant::now();
    let ctx = WhisperContext::new(model_path)?;
    println!(
        "Model loaded in {:.2?}",
        model_load_start.elapsed()
    );

    let (audio, sample_rate) = load_wav(Path::new(audio_path))?;
    println!(
        "Audio stats: {} samples ({:.2}s at {} Hz)",
        audio.len(),
        audio.len() as f32 / sample_rate as f32,
        sample_rate
    );

    let transcription_start = Instant::now();
    let text = ctx.transcribe_with_language(&audio, language)?;
    println!(
        "Transcription completed in {:.2?}",
        transcription_start.elapsed()
    );
    println!("\n=== Result ===\n{}", text.trim());

    Ok(())
}

fn load_wav(path: &Path) -> Result<(Vec<f32>, u32), Box<dyn Error>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    if spec.sample_format != SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err(format!(
            "unsupported WAV format: expected 16-bit PCM, got {:?} ({} bits)",
            spec.sample_format, spec.bits_per_sample
        )
        .into());
    }

    let channels = spec.channels as usize;
    if channels == 0 {
        return Err("WAV file has zero channels".into());
    }

    let mut mono = Vec::new();

    if channels == 1 {
        for sample in reader.samples::<i16>() {
            mono.push(sample? as f32 / 32768.0);
        }
    } else {
        let mut interleaved = Vec::with_capacity(channels);
        for sample in reader.samples::<i16>() {
            interleaved.push(sample?);
            if interleaved.len() == channels {
                let sum: i32 = interleaved.iter().map(|&v| v as i32).sum();
                mono.push(sum as f32 / channels as f32 / 32768.0);
                interleaved.clear();
            }
        }
    }

    Ok((mono, spec.sample_rate))
}
