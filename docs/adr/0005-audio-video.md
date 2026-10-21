# ADR 0005 — Audio and video: faster-whisper + pyannote directly, sentence-level lines

Date: 2026-10-21 · Status: accepted

## faster-whisper + pyannote.audio 4 instead of WhisperX
WhisperX pins older torch versions than the CUDA 12.8 torch already used by Docling and the
reranker. Its useful extra (assigning words to speakers) is ~20 lines (`assign_speakers`), so
we call faster-whisper (large-v3-turbo, fp16) and pyannote `speaker-diarization-community-1`
directly. On the fixtures: 3 min audio transcribed in 8 s, diarized in 4 s.

Windows notes:
- `import torch` before faster-whisper: CTranslate2 needs torch's bundled cuBLAS
  (`cublas64_12.dll`), otherwise transcription fails.
- pyannote gets the waveform in memory; its own decoding (torchcodec) needs FFmpeg DLLs.

## Stages
`parse` = model outputs only (`audio.wav`, `asr.json`, `diarization.json`, video `frames.json`),
each cached, run as batches Whisper → pyannote → VLM. Turns are built from them in `chunk`, so
changing line/turn rules needs only `tkb reindex <file>`, no GPU.

## One transcript line per sentence
Milestone 4 requires quotes within ±5 s. With whole speaker turns as lines, a 2-minute
monologue gave every quote the timestamp 0:00; with 8 s lines the "$249" quote was still
pinned 10 s early. Lines now break at every sentence end (`audio.max_line_seconds: 0`); each line
is `[m:ss] Speaker: text`, and `answer.generate.pinpoint()` maps an answer sentence to the
best-matching line (and the nearest keyframe for video).

## Speakers
Diarization labels are mapped to people manually (`tkb speakers label`), with self-introductions
("My name is …") offered as suggestions. Whisper writes "Will I.M." for Will.i.am, so names carry
aliases. Relabelling re-chunks the recording so chunk text carries the person's name.

## Keyframes
Scene changes (ffmpeg `scene > 0.3`) plus one frame per 30 s, ≥ 2 s apart, captioned by the VLM.
On the 2-min demo: 15 frames, mostly near-identical "person in a kitchen" captions; the useful
one shows the product box. Captions add little when the speaker says everything; they matter
for screen recordings and product demos without narration.

## Open
- The support judge flags summary sentences that combine facts from several passages, and the
  answer LLM sometimes cites the wrong passage for a fact (cmp-02).
- No speaker auto-matching across recordings (voice embeddings) yet.
