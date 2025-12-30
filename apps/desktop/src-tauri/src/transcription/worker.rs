use asr_core::TranscribedSegment;
use log::{debug, error, info};
use serde::{Deserialize, Serialize};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use tauri::AppHandle;

use crate::audio::constants::VAD_SAMPLE_RATE;
use crate::audio::processing::trim_session_audio_samples;
use crate::audio::state::try_recording_state;
use crate::emit_transcription_segment;
use crate::whisper::WHISPER_CTX;

use super::TranscriptionCommand;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedSegment {
    pub text: String,
    pub start_ms: i64,
    pub end_ms: i64,
}

fn slice_audio_segment(audio_data: &[f32], start_ms: i64, end_ms: i64) -> Vec<f32> {
    if start_ms >= end_ms {
        return Vec::new();
    }

    let start_sample =
        ((start_ms * VAD_SAMPLE_RATE as i64) / 1000).clamp(0, audio_data.len() as i64) as usize;
    let end_sample =
        ((end_ms * VAD_SAMPLE_RATE as i64) / 1000).clamp(0, audio_data.len() as i64) as usize;

    if start_sample >= end_sample || start_sample >= audio_data.len() {
        return Vec::new();
    }
    audio_data[start_sample..end_sample.min(audio_data.len())].to_vec()
}

pub fn split_segments_on_punctuation(segments: &[TranscribedSegment]) -> Vec<PreparedSegment> {
    const TERMINATORS: &[char] = &['。', '！', '？', '．', '!', '?', '.', ',', '、'];
    const FINALIZE_THRESHOLD_MS: i64 = 1_500;

    let mut prepared = Vec::new();
    let mut current_text = String::new();
    let mut current_start: Option<i64> = None;
    let mut current_end: i64 = 0;

    let last_segment_end_ms = segments.last().map(|s| s.end_ms).unwrap_or(0);

    debug!(
        "[split_segments] Processing {} segments, last_segment_end_ms={}",
        segments.len(),
        last_segment_end_ms
    );
    for (seg_idx, segment) in segments.iter().enumerate() {
        debug!(
            "[split_segments] segment #{}: {}ms-{}ms, text='{}'",
            seg_idx, segment.start_ms, segment.end_ms, segment.text
        );

        if current_start.is_none() {
            current_start = Some(segment.start_ms);
        }

        current_text.push_str(&segment.text);
        current_end = segment.end_ms;

        let has_terminator = segment
            .text
            .chars()
            .rev()
            .find(|c| !c.is_whitespace())
            .map(|c| TERMINATORS.contains(&c))
            .unwrap_or(false);

        let is_far_enough_from_end =
            (last_segment_end_ms - segment.end_ms) >= FINALIZE_THRESHOLD_MS;
        let should_split = has_terminator && is_far_enough_from_end;

        if should_split {
            let text = current_text.clone();
            let has_content = text.chars().any(|c| !c.is_whitespace());
            debug!(
                "[split_segments]   segment #{} triggers split (end_ms={}, last_end_ms={}, diff={}ms), accumulated text='{}'",
                seg_idx, segment.end_ms, last_segment_end_ms, last_segment_end_ms - segment.end_ms,
                text.replace('\n', " ")
            );
            if has_content {
                prepared.push(PreparedSegment {
                    text,
                    start_ms: current_start.unwrap_or(segment.start_ms),
                    end_ms: current_end,
                });
            }
            current_text.clear();
            current_start = None;
        }
    }

    if current_text.chars().any(|c| !c.is_whitespace()) {
        debug!(
            "[split_segments]   final accumulated text='{}'",
            current_text.replace('\n', " ")
        );
        prepared.push(PreparedSegment {
            text: current_text.clone(),
            start_ms: current_start.unwrap_or(0),
            end_ms: current_end,
        });
    }

    if prepared.len() > 2 {
        let mut combined = prepared.first().cloned().unwrap();
        for segment in prepared.iter().take(prepared.len() - 1).skip(1) {
            if !combined.text.trim_end().is_empty() {
                combined.text.push('\n');
            }
            combined.text.push_str(segment.text.trim_start());
            combined.end_ms = segment.end_ms;
        }
        let last = prepared.last().cloned().unwrap();
        prepared = vec![combined, last];
    }

    debug!(
        "[split_segments] Result: {} prepared segments",
        prepared.len()
    );
    prepared
}

pub fn spawn_transcription_worker(
    app_handle: AppHandle,
) -> (mpsc::Sender<TranscriptionCommand>, JoinHandle<()>) {
    let (tx, rx) = mpsc::channel::<TranscriptionCommand>();
    let handle = thread::spawn(move || {
        use std::collections::{HashMap, HashSet};

        while let Ok(command) = rx.recv() {
            match command {
                TranscriptionCommand::Run {
                    audio,
                    language,
                    session_id,
                    is_final,
                } => {
                    let mut all_commands = vec![(audio, language, session_id, is_final)];
                    while let Ok(next_command) = rx.try_recv() {
                        match next_command {
                            TranscriptionCommand::Run {
                                audio: a,
                                language: l,
                                session_id: s,
                                is_final: f,
                            } => {
                                all_commands.push((a, l, s, f));
                            }
                            TranscriptionCommand::Stop => return,
                        }
                    }

                    let mut latest_requests: HashMap<String, (Vec<f32>, Option<String>, bool)> =
                        HashMap::new();
                    let mut final_requests = Vec::new();
                    let mut sessions_with_final = HashSet::new();

                    for (audio, language, session_id, is_final) in all_commands {
                        if is_final {
                            sessions_with_final.insert(session_id.clone());
                            final_requests.push((audio, language, session_id, is_final));
                        } else {
                            latest_requests.insert(session_id, (audio, language, is_final));
                        }
                    }

                    for (session_id, (audio, language, is_final)) in latest_requests {
                        if sessions_with_final.contains(&session_id) {
                            continue;
                        }
                        if let Err(err) =
                            transcribe_and_emit(&audio, language.clone(), is_final, &app_handle)
                        {
                            error!("Transcription worker error: {}", err);
                        }
                    }

                    for (audio, language, _session_id, is_final) in final_requests {
                        if let Err(err) =
                            transcribe_and_emit(&audio, language.clone(), is_final, &app_handle)
                        {
                            error!("Transcription worker error: {}", err);
                        }
                    }
                }
                TranscriptionCommand::Stop => break,
            }
        }
    });
    (tx, handle)
}

pub fn transcribe_and_emit_common(
    audio_data: &[f32],
    language: &str,
    session_id_prefix: &str,
    session_id_counter: u64,
    is_final: bool,
    app_handle: &AppHandle,
    source: &str,
    on_session_rotate: Option<&dyn Fn(u64)>,
) -> Result<Option<i64>, String> {
    let ctx_lock = WHISPER_CTX
        .get()
        .ok_or_else(|| "Whisper not initialized".to_string())?
        .clone();
    let ctx_guard = ctx_lock.lock().unwrap();
    let ctx = ctx_guard
        .as_ref()
        .ok_or_else(|| "Whisper context not available".to_string())?;

    let session_id = format!("{}_{}", session_id_prefix, session_id_counter);

    if is_final {
        let text = ctx
            .transcribe_with_language(audio_data, language)
            .map_err(|e| e.to_string())?;

        if let Some(callback) = on_session_rotate {
            callback(session_id_counter + 1);
        }

        emit_transcription_segment(
            app_handle,
            text,
            Some(audio_data.to_vec()),
            session_id,
            is_final,
            source.to_string(),
        )?;

        return Ok(None);
    }

    let segments = ctx
        .transcribe_segments_with_language(audio_data, language)
        .map_err(|e| e.to_string())?;

    let prepared_segments = split_segments_on_punctuation(&segments);
    info!(
        "[transcribe_and_emit_common] Got {} prepared segments from {} segments",
        prepared_segments.len(),
        segments.len()
    );

    let mut finalized_cutoff_ms: Option<i64> = None;
    let total_segments = prepared_segments.len();
    let mut current_session_id = session_id.clone();
    let mut current_counter = session_id_counter;

    for (idx, segment) in prepared_segments.iter().enumerate() {
        debug!(
            "[transcribe_and_emit_common] Emitting segment #{}: {}ms - {}ms, text=\"{}\"",
            idx,
            segment.start_ms,
            segment.end_ms,
            segment.text.replace('\n', " ")
        );
        let segment_audio = slice_audio_segment(audio_data, segment.start_ms, segment.end_ms);
        let segment_is_final = total_segments > 0 && idx + 1 != total_segments;

        if let Err(err) = emit_transcription_segment(
            app_handle,
            segment.text.clone(),
            if segment_audio.is_empty() {
                None
            } else {
                Some(segment_audio)
            },
            current_session_id.clone(),
            segment_is_final,
            source.to_string(),
        ) {
            error!("Failed to emit transcription segment: {}", err);
        }

        if segment_is_final {
            finalized_cutoff_ms = Some(
                finalized_cutoff_ms.map_or(segment.end_ms, |cutoff| cutoff.max(segment.end_ms)),
            );

            current_counter += 1;
            if let Some(callback) = on_session_rotate {
                callback(current_counter);
            }
            current_session_id = format!("{}_{}", session_id_prefix, current_counter);
            debug!(
                "[transcribe_and_emit_common] Rotated session_id to: {}",
                current_session_id
            );
        }
    }

    Ok(finalized_cutoff_ms)
}

fn transcribe_and_emit(
    audio_data: &[f32],
    language: Option<String>,
    is_final: bool,
    app_handle: &AppHandle,
) -> Result<(), String> {
    let state =
        try_recording_state().ok_or_else(|| "Recording state not initialized".to_string())?;

    let (session_id_counter, lang) = {
        let state_guard = state.lock();
        (
            state_guard.session_id_counter,
            language.as_deref().unwrap_or("ja").to_string(),
        )
    };

    let finalized_cutoff_ms = transcribe_and_emit_common(
        audio_data,
        &lang,
        "mic",
        session_id_counter,
        is_final,
        app_handle,
        "user",
        Some(&|new_counter| {
            if let Some(state) = try_recording_state() {
                let mut state_guard = state.lock();
                state_guard.session_id_counter = new_counter;
            }
        }),
    )?;

    if let Some(cutoff_ms) = finalized_cutoff_ms {
        let cutoff_samples = ((cutoff_ms.max(0) * VAD_SAMPLE_RATE as i64) / 1000) as usize;
        let cutoff_samples = cutoff_samples.min(audio_data.len());
        if cutoff_samples > 0 {
            trim_session_audio_samples(cutoff_samples);
        }
    }

    Ok(())
}
