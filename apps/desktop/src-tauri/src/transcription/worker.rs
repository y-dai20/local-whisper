use asr_core::TranscribedSegment;
use log::{debug, error, info};
use serde::{Deserialize, Serialize};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use tauri::AppHandle;
use std::sync::mpsc::Sender;

use crate::audio::constants::VAD_SAMPLE_RATE;
use crate::audio::state::{try_recording_state, RecordingState};
use crate::emit_transcription_segment;
use crate::whisper::WHISPER_CTX;

use super::{TranscriptionCommand, TranscriptionSource};

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

fn split_segments_on_punctuation(segments: &[TranscribedSegment]) -> Vec<PreparedSegment> {
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
                    source,
                    session_id_counter,
                    is_final,
                } => {
                    let mut all_commands =
                        vec![(audio, language, source, session_id_counter, is_final)];
                    while let Ok(next_command) = rx.try_recv() {
                        match next_command {
                            TranscriptionCommand::Run {
                                audio: a,
                                language: l,
                                source: src,
                                session_id_counter: sid,
                                is_final: f,
                            } => {
                                all_commands.push((a, l, src, sid, f));
                            }
                            TranscriptionCommand::Stop => return,
                        }
                    }

                    let mut latest_requests: HashMap<
                        (TranscriptionSource, u64),
                        (Vec<f32>, Option<String>, bool),
                    > = HashMap::new();
                    let mut final_requests = Vec::new();
                    let mut sessions_with_final = HashSet::new();

                    for (audio, language, source, session_id_counter, is_final) in all_commands {
                        let key = (source, session_id_counter);
                        if is_final {
                            sessions_with_final.insert(key);
                            final_requests.push((audio, language, source, session_id_counter));
                        } else {
                            latest_requests.insert(key, (audio, language, is_final));
                        }
                    }

                    for (audio, language, source, session_id_counter) in final_requests {
                        if let Err(err) = transcribe_and_emit(
                            &audio,
                            language.clone(),
                            source,
                            session_id_counter,
                            true,
                            &app_handle,
                        ) {
                            error!("Transcription worker error: {}", err);
                        }
                    }

                    for (key, (audio, language, is_final)) in latest_requests {
                        if sessions_with_final.contains(&key) {
                            continue;
                        }
                        let (source, session_id_counter) = key;
                        if let Err(err) = transcribe_and_emit(
                            &audio,
                            language.clone(),
                            source,
                            session_id_counter,
                            is_final,
                            &app_handle,
                        ) {
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
    session_id_counter: u64,
    is_final: bool,
    app_handle: &AppHandle,
    source: &str,
    message_id_counter: u64,
    transcribed_samples: usize,
) -> Result<(usize, u64), String> {
    let ctx_lock = WHISPER_CTX
        .get()
        .ok_or_else(|| "Whisper not initialized".to_string())?
        .clone();
    let ctx_guard = ctx_lock.lock().unwrap();
    let ctx = ctx_guard
        .as_ref()
        .ok_or_else(|| "Whisper context not available".to_string())?;

    let mut next_message_id = message_id_counter;

    // Skip already transcribed samples
    if transcribed_samples >= audio_data.len() {
        debug!(
            "[transcribe_and_emit_common] Skipping transcription: already transcribed {} samples, audio length {}",
            transcribed_samples,
            audio_data.len()
        );
        return Ok((transcribed_samples, next_message_id));
    }

    let audio_to_transcribe = &audio_data[transcribed_samples..];
    debug!(
        "[transcribe_and_emit_common] Transcribing from sample {} to {}, total {} samples",
        transcribed_samples,
        audio_data.len(),
        audio_to_transcribe.len()
    );

    if is_final {
        let text = ctx
            .transcribe_with_language(audio_to_transcribe, language)
            .map_err(|e| e.to_string())?;

        emit_transcription_segment(
            app_handle,
            text,
            Some(audio_to_transcribe.to_vec()),
            session_id_counter,
            next_message_id,
            is_final,
            source.to_string(),
        )?;

        next_message_id = next_message_id.wrapping_add(1);

        return Ok((audio_data.len(), next_message_id));
    }

    let segments = ctx
        .transcribe_segments_with_language(audio_to_transcribe, language)
        .map_err(|e| e.to_string())?;

    let prepared_segments = split_segments_on_punctuation(&segments);
    info!(
        "[transcribe_and_emit_common] Got {} prepared segments from {} segments",
        prepared_segments.len(),
        segments.len()
    );

    let total_segments = prepared_segments.len();
    let mut new_transcribed_samples = transcribed_samples;

    for (idx, segment) in prepared_segments.iter().enumerate() {
        let segment_is_final = total_segments > 0 && idx + 1 != total_segments;

        // Calculate absolute sample position in the original audio
        let segment_start_samples = transcribed_samples + ((segment.start_ms * VAD_SAMPLE_RATE as i64) / 1000) as usize;
        let segment_end_samples = transcribed_samples + ((segment.end_ms * VAD_SAMPLE_RATE as i64) / 1000) as usize;

        debug!(
            "[transcribe_and_emit_common] Emitting segment #{}: {}ms - {}ms (samples {} - {}), text=\"{}\", is_final={}",
            idx,
            segment.start_ms,
            segment.end_ms,
            segment_start_samples,
            segment_end_samples,
            segment.text.replace('\n', " "),
            segment_is_final
        );

        let segment_audio = slice_audio_segment(audio_to_transcribe, segment.start_ms, segment.end_ms);

        if let Err(err) = emit_transcription_segment(
            app_handle,
            segment.text.clone(),
            if segment_audio.is_empty() {
                None
            } else {
                Some(segment_audio)
            },
            session_id_counter,
            next_message_id,
            segment_is_final,
            source.to_string(),
        ) {
            error!("Failed to emit transcription segment: {}", err);
        }

        if segment_is_final {
            // Update transcribed samples to the end of this finalized segment
            new_transcribed_samples = segment_end_samples;
            next_message_id = next_message_id.wrapping_add(1);
            debug!(
                "[transcribe_and_emit_common] Updated transcribed_samples to {}",
                new_transcribed_samples
            );
        }
    }

    Ok((new_transcribed_samples, next_message_id))
}

fn transcribe_and_emit(
    audio_data: &[f32],
    language: Option<String>,
    source: TranscriptionSource,
    session_id_counter: u64,
    is_final: bool,
    app_handle: &AppHandle,
) -> Result<(), String> {
    let state =
        try_recording_state().ok_or_else(|| "Recording state not initialized".to_string())?;

    let (lang, source_state) = {
        let state_guard = state.lock();
        let lang = language
            .clone()
            .or_else(|| state_guard.language.clone())
            .unwrap_or_else(|| "ja".to_string());
        let source_state = state_guard.transcription_state(source).clone();
        (lang, source_state)
    };

    let (new_transcribed_samples, next_message_id) = transcribe_and_emit_common(
        audio_data,
        &lang,
        source_state.session_id_counter,
        is_final,
        app_handle,
        source.event_source(),
        source_state.message_id_counter,
        source_state.transcribed_samples,
    )?;

    {
        let mut state_guard = state.lock();
        let source_state_mut = state_guard.transcription_state_mut(source);

        if is_final {
            source_state_mut.session_id_counter =
                source_state_mut.session_id_counter.wrapping_add(1);
            source_state_mut.message_id_counter = 0;
            source_state_mut.transcribed_samples = 0;
        } else {
            source_state_mut.transcribed_samples = new_transcribed_samples;
            source_state_mut.message_id_counter = next_message_id;
        }
    }

    Ok(())
}

pub fn queue_transcription_with_source(
    audio: Vec<f32>,
    language: Option<String>,
    session_id_counter: u64,
    source: TranscriptionSource,
    is_final: bool,
    tx: &Sender<TranscriptionCommand>,
) {
    if audio.is_empty() {
        return;
    }

    if tx
        .send(TranscriptionCommand::Run {
            audio,
            language,
            source,
            session_id_counter,
            is_final,
        })
        .is_err()
    {
        log::error!("Failed to send transcription command");
    }
}

pub fn queue_transcription(state: &RecordingState, is_final: bool) {
    if state.session_audio.is_empty() {
        return;
    }
    let Some(tx) = &state.transcription_tx else {
        return;
    };

    let session_id_counter = state
        .transcription_state(TranscriptionSource::Mic)
        .session_id_counter;

    queue_transcription_with_source(
        state.session_audio.clone(),
        state.language.clone(),
        session_id_counter,
        TranscriptionSource::Mic,
        is_final,
        tx,
    );
}

pub fn stop_transcription_worker(state: &mut RecordingState) {
    if let Some(tx) = state.transcription_tx.take() {
        let _ = tx.send(TranscriptionCommand::Stop);
    }
    if let Some(handle) = state.transcription_handle.take() {
        let _ = handle.join();
    }
}
