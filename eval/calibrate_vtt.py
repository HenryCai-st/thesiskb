"""Compare Whisper word timestamps with a reference subtitle file (WebVTT).

Usage:
    uv run python eval/calibrate_vtt.py <recording file> <reference.vtt> [--find "phrase" ...]

VTT cues are coarse (a line of text per 2-8 s, overlapping for YouTube auto-captions), so each
VTT word gets a time by linear interpolation inside its cue. The VTT words are aligned to the
Whisper words (difflib on normalised tokens) and the offsets of matched words are summarised.
`--find` prints the VTT time of a phrase, to use as a gold timestamp in eval/questions.yaml.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
import sys
from pathlib import Path

from thesiskb.core.config import get_pipeline_config
from thesiskb.ingest.detect import sha256_file

_CUE = re.compile(r"(\d+):(\d{2}):(\d{2})\.(\d{3})\s+-->\s+(\d+):(\d{2}):(\d{2})\.(\d{3})")


def _secs(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def norm(word: str) -> str:
    return re.sub(r"[^\w$]", "", word.lower())


def parse_vtt(path: Path) -> list[tuple[float, str]]:
    """(time, word) for every word, interpolated within its cue.

    Rolling auto-captions repeat text across overlapping cues; a cue's span is taken as
    [its start, next cue's start] so overlapping text is not double-timed.
    """
    cues: list[tuple[float, float, str]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        m = _CUE.search(lines[i])
        if m:
            start, end = _secs(*m.groups()[:4]), _secs(*m.groups()[4:])
            text = []
            i += 1
            while i < len(lines) and lines[i].strip():
                text.append(lines[i].strip())
                i += 1
            cues.append((start, end, " ".join(text)))
        i += 1
    words: list[tuple[float, str]] = []
    for k, (start, end, text) in enumerate(cues):
        span_end = min(end, cues[k + 1][0]) if k + 1 < len(cues) else end
        toks = [t for t in re.sub(r"\[[^\]]*\]", " ", text).split() if norm(t)]
        for j, tok in enumerate(toks):
            words.append((start + (span_end - start) * j / max(1, len(toks)), tok))
    return words


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", type=Path)
    ap.add_argument("vtt", type=Path)
    ap.add_argument("--find", action="append", default=[])
    args = ap.parse_args()

    asr_path = Path(get_pipeline_config().processed_dir) / sha256_file(args.recording) / "asr.json"
    asr = json.loads(asr_path.read_text(encoding="utf-8"))["words"]
    ref = parse_vtt(args.vtt)

    a = [norm(w["text"]) for w in asr]
    b = [norm(w) for _, w in ref]
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    offsets = []
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            offsets.append(ref[block.b + k][0] - asr[block.a + k]["start"])
    print(
        f"{args.recording.name}: {len(asr)} Whisper words, {len(ref)} VTT words, "
        f"{len(offsets)} matched ({len(offsets) / max(1, len(b)):.0%} of VTT)"
    )
    if offsets:
        abs_off = sorted(abs(o) for o in offsets)
        print(
            f"  offset VTT - Whisper: median {statistics.median(offsets):+.2f} s, "
            f"mean {statistics.mean(offsets):+.2f} s"
        )
        print(
            f"  |offset|: median {statistics.median(abs_off):.2f} s, "
            f"p90 {abs_off[int(0.9 * (len(abs_off) - 1))]:.2f} s, max {abs_off[-1]:.2f} s"
        )
    for phrase in args.find:
        target = [norm(t) for t in phrase.split()]
        for k in range(len(b) - len(target) + 1):
            if b[k : k + len(target)] == target:
                print(f"  '{phrase}': VTT {ref[k][0]:.1f} s")
                break
        else:
            print(f"  '{phrase}': not found in VTT")
        for k in range(len(a) - len(target) + 1):
            if a[k : k + len(target)] == target:
                print(f"  '{phrase}': Whisper {asr[k]['start']:.1f} s")
                break


if __name__ == "__main__":
    main()
