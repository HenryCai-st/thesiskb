"""Video: audio track → audio.py; keyframes → VLM captions, aligned to the transcript timeline.

Keyframes: scene changes (ffmpeg `select='gt(scene,T)'`) plus one frame every N seconds as a
fallback, de-duplicated when closer than `min_gap` seconds. Outputs in data/processed/<sha>/:
  frames/<ms>.jpg   keyframes named by timestamp in milliseconds
  frames.json       [{t, path, caption}]  (parse output for the visual track)
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

from pydantic import BaseModel

from thesiskb.ingest.audio import ts
from thesiskb.ingest.chunking import Block
from thesiskb.llm.gateway import LLMGateway, image_message

log = logging.getLogger(__name__)

FRAME_PROMPT = (
    "This is a frame from a video at {t}. Describe in 1-3 sentences what is on screen (people, "
    "objects, product, user interface, slide). Then copy any visible text, names, numbers and "
    "specifications exactly. Output plain text, no preamble."
)


class Keyframe(BaseModel):
    t: float
    path: str
    caption: str = ""


def _probe_duration(video: Path, ffprobe: str = "ffprobe") -> float:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return float(out.strip() or 0)


def scene_times(video: Path, threshold: float, ffmpeg: str = "ffmpeg") -> list[float]:
    """Timestamps of scene changes, read from ffmpeg's showinfo output."""
    proc = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(video),
            "-an",
            "-vf",
            f"select='gt(scene,{threshold})',showinfo",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return [float(m) for m in re.findall(r"pts_time:([\d.]+)", proc.stderr)]


def pick_times(
    scenes: list[float], duration: float, every: float, min_gap: float = 2.0
) -> list[float]:
    fallback = [i * every for i in range(int(duration // every) + 1)] if every > 0 else []
    picked: list[float] = []
    for t in sorted({round(x, 2) for x in [*scenes, *fallback] if x < duration}):
        if not picked or t - picked[-1] >= min_gap:
            picked.append(t)
    return picked


def extract_frames(video: Path, times: list[float], out_dir: Path, ffmpeg: str = "ffmpeg"):
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in times:
        path = out_dir / f"{int(t * 1000):08d}.jpg"
        if not path.exists():
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{t:.2f}",
                    "-i",
                    str(video),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale='min(1280,iw)':-2",
                    "-q:v",
                    "3",
                    str(path),
                ],
                check=True,
            )
        frames.append(Keyframe(t=t, path=path.as_posix()))
    return frames


def caption_frames(gateway: LLMGateway, frames: list[Keyframe]) -> list[Keyframe]:
    for f in frames:
        jpeg = Path(f.path).read_bytes()
        f.caption = gateway.chat(
            image_message(FRAME_PROMPT.format(t=ts(f.t)), jpeg), role="vision", max_tokens=300
        ).strip()
    return frames


def read_keyframes(
    gateway: LLMGateway,
    video: Path,
    out_dir: Path,
    scene_threshold: float = 0.3,
    every: float = 30,
    max_frames: int = 60,
) -> list[Keyframe]:
    duration = _probe_duration(video)
    times = pick_times(scene_times(video, scene_threshold), duration, every)
    if len(times) > max_frames:  # very cutty video: keep an even spread
        step = len(times) / max_frames
        times = [times[int(i * step)] for i in range(max_frames)]
    frames = caption_frames(gateway, extract_frames(video, times, out_dir / "frames"))
    (out_dir / "frames.json").write_text(
        json.dumps([f.model_dump() for f in frames], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return frames


def frame_blocks(frames: list[Keyframe]) -> list[Block]:
    return [
        Block(
            text=f"[{ts(f.t)}] (on screen) {f.caption}",
            kind="frame",
            locator={"t_start": f.t, "t_end": f.t, "frame": f.path},
        )
        for f in frames
        if f.caption
    ]
