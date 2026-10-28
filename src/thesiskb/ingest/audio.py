"""Audio: 16 kHz WAV → faster-whisper words → pyannote speakers → speaker turns.

Stage outputs in data/processed/<sha>/ (each cached separately, so a failed diarization does
not redo transcription):
  audio.wav        16 kHz mono PCM (ffmpeg)
  asr.json         language + words with timestamps
  diarization.json speaker turns from pyannote
  transcript.json  words assigned to speakers, merged into turns (the parse output)

Models are loaded once per batch (`Transcriber`, `Diarizer`) and freed with close(), so
Whisper, pyannote and Ollama models are never resident together.
"""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import re
import subprocess
import wave
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from thesiskb.ingest.chunking import Block

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class Word(BaseModel):
    start: float
    end: float
    text: str
    prob: float | None = None
    speaker: str | None = None


class SpeakerTurn(BaseModel):
    start: float
    end: float
    speaker: str | None
    text: str = ""


class Transcript(BaseModel):
    language: str | None = None
    duration: float = 0.0
    words: list[Word] = Field(default_factory=list)
    turns: list[SpeakerTurn] = Field(default_factory=list)
    speakers: list[str] = Field(default_factory=list)


def extract_wav(src: Path, wav: Path, ffmpeg: str = "ffmpeg") -> Path:
    wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(wav),
        ],
        check=True,
    )
    return wav


def load_wav(wav: Path):
    import numpy as np

    with wave.open(str(wav)) as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, np.int16).astype(np.float32) / 32768.0


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class Transcriber:
    def __init__(self, model: str = "large-v3-turbo", compute_type: str = "float16") -> None:
        # torch first: its wheel ships the CUDA 12 cuBLAS/cuDNN DLLs and adds them to the DLL
        # search path. CTranslate2 (faster-whisper) needs them too but doesn't bundle them on
        # Windows ("cublas64_12.dll is not found" otherwise).
        with contextlib.suppress(ImportError):
            import torch  # noqa: F401
        from faster_whisper import WhisperModel

        self.model_name = model
        try:
            self._model = WhisperModel(model, device="cuda", compute_type=compute_type)
        except (RuntimeError, ValueError) as e:  # no CUDA: slow but works (CI, laptops)
            log.warning("faster-whisper on CPU (%s)", e)
            self._model = WhisperModel(model, device="cpu", compute_type="int8")

    def transcribe(self, wav: Path, out_json: Path) -> dict[str, Any]:
        audio = load_wav(wav)
        segments, info = self._model.transcribe(audio, word_timestamps=True, vad_filter=True)
        words = [
            {
                "start": round(w.start, 2),
                "end": round(w.end, 2),
                "text": w.word,
                "prob": round(w.probability, 3),
            }
            for seg in segments
            for w in (seg.words or [])
        ]
        data = {
            "model": self.model_name,  # which speech model made this transcript (Models tab)
            "language": info.language,
            "duration": round(len(audio) / SAMPLE_RATE, 2),
            "words": words,
        }
        out_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    def close(self) -> None:
        del self._model
        _free_gpu()


class Diarizer:
    def __init__(self, model: str, hf_token: str | None) -> None:
        import torch
        from pyannote.audio import Pipeline

        if not hf_token:
            raise RuntimeError(
                "Diarization needs TKB_HF_TOKEN in .env and accepted terms for " + model
            )
        self._pipe = Pipeline.from_pretrained(model, token=hf_token)
        if torch.cuda.is_available():
            self._pipe.to(torch.device("cuda"))

    def diarize(self, wav: Path, out_json: Path) -> list[dict[str, Any]]:
        import torch

        audio = load_wav(wav)
        # pass the waveform in memory: pyannote's own file decoding needs FFmpeg DLLs on Windows
        result = self._pipe({"waveform": torch.from_numpy(audio)[None], "sample_rate": SAMPLE_RATE})
        ann = getattr(result, "speaker_diarization", result)
        turns = [
            {"start": round(seg.start, 2), "end": round(seg.end, 2), "speaker": spk}
            for seg, _, spk in ann.itertracks(yield_label=True)
        ]
        out_json.write_text(json.dumps(turns), encoding="utf-8")
        return turns

    def close(self) -> None:
        del self._pipe
        _free_gpu()


def assign_speakers(
    words: list[Word], diarization: list[dict[str, Any]], max_gap: float = 1.0
) -> list[Word]:
    """Give each word the speaker whose turn overlaps it most (or the nearest within max_gap)."""
    for w in words:
        best, best_overlap, best_gap = None, 0.0, max_gap
        for t in diarization:
            overlap = min(w.end, t["end"]) - max(w.start, t["start"])
            if overlap > best_overlap:
                best, best_overlap = t["speaker"], overlap
            elif best_overlap == 0.0:
                # 0 for a zero-length word inside the turn (Whisper sometimes gives start == end)
                gap = max(0.0, t["start"] - w.end, w.start - t["end"])
                if gap < best_gap:
                    best, best_gap = t["speaker"], gap
        w.speaker = best
    return words


def merge_turns(
    words: list[Word], max_pause: float = 2.0, max_turn: float = 20.0
) -> list[SpeakerTurn]:
    """Consecutive words of one speaker form a turn. A long pause starts a new one, and so does
    a sentence end once a turn is longer than max_turn seconds — a 2-minute monologue as one
    turn would leave every quote from it with the same timestamp."""
    turns: list[SpeakerTurn] = []
    for w in words:
        last = turns[-1] if turns else None
        same = last and last.speaker == w.speaker and w.start - last.end <= max_pause
        if same and last.end - last.start >= max_turn and re.search(r"[.!?]\s*$", last.text):
            same = False
        if same:
            last.end = w.end
            last.text += w.text
        else:
            turns.append(SpeakerTurn(start=w.start, end=w.end, speaker=w.speaker, text=w.text))
    for t in turns:
        t.text = t.text.strip()
    return [t for t in turns if t.text]


def build_transcript(
    asr: dict[str, Any], diarization: list[dict[str, Any]], max_turn: float = 0.0
) -> Transcript:
    words = assign_speakers([Word(**w) for w in asr["words"]], diarization)
    turns = merge_turns(words, max_turn=max_turn)
    return Transcript(
        language=asr.get("language"),
        duration=asr.get("duration", 0.0),
        words=words,
        turns=turns,
        speakers=sorted({t.speaker for t in turns if t.speaker}),
    )


def ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m // 60}:{m % 60:02d}:{s:02d}" if m >= 60 else f"{m}:{s:02d}"


def time_windows(
    items: list[tuple[float, float]], min_len: float = 60, max_len: float = 120
) -> list[tuple[float, float]]:
    """Group consecutive (start, end) items into windows of ~min_len..max_len seconds."""
    windows: list[tuple[float, float]] = []
    start = end = None
    for a, b in items:
        if start is None:
            start, end = a, b
        elif b - start > max_len and end - start >= min_len / 2:
            windows.append((start, end))
            start, end = a, b
        else:
            end = max(end, b)
    if start is not None:
        windows.append((start, end))
    return windows


def transcript_blocks(
    transcript: Transcript,
    names: dict[str, str],
    windows_s: tuple[float, float] = (60, 120),
    frames: list[Block] | None = None,
) -> list[Block]:
    """Turn blocks ("[0:23] Will.i.am: ...") plus optional keyframe blocks, interleaved by time
    and grouped into time-window sections so chunks never span more than ~2 minutes."""
    items: list[Block] = []
    for t in transcript.turns:
        who = names.get(t.speaker or "", t.speaker or "Unknown speaker")
        items.append(
            Block(
                text=f"[{ts(t.start)}] {who}: {t.text}",
                kind="turn",
                locator={"t_start": t.start, "t_end": t.end, "speaker": t.speaker},
            )
        )
    items.extend(frames or [])
    items.sort(key=lambda b: b.locator["t_start"])
    windows = time_windows(
        [(b.locator["t_start"], b.locator.get("t_end") or b.locator["t_start"]) for b in items],
        *windows_s,
    )
    for b in items:
        start = b.locator["t_start"]
        a, z = next(((a, z) for a, z in windows if a <= start <= z), windows[-1])
        b.section = [f"{ts(a)}–{ts(z)}"]
    return items


# phrase case-insensitive, but the name itself must be capitalised ("I'm a musician" is no name)
_SELF_INTRO = re.compile(
    r"\b(?i:my name is|i am|i'm|ich bin|mein name ist|ich heiße)\s+"
    r"((?:[A-Z][\w.'-]*\.?\s?){1,3})",
)


_NOT_NAME_TAIL = {
    "I",
    "I'm",
    "I’m",
    "And",
    "Und",
    "So",
    "We",
    "Wir",
    "Ich",
    "The",
    "Das",
    "Der",
    "Die",
}


def suggest_names(transcript: Transcript) -> dict[str, str]:
    """'My name is X' in a speaker's own turn → suggested name for that label (not applied)."""
    out: dict[str, str] = {}
    for t in transcript.turns:
        if not t.speaker or t.speaker in out:
            continue
        m = _SELF_INTRO.search(t.text)
        if m:
            tokens = m.group(1).split()
            # "My name is Will I.M. I make music": the next sentence's "I" is not the name
            while tokens and tokens[-1].rstrip(".") in _NOT_NAME_TAIL:
                tokens.pop()
            name = " ".join(tokens).rstrip(".")
            if len(name) > 1 and name.lower() not in {"a", "the", "so", "here", "not"}:
                out[t.speaker] = name
    return out
