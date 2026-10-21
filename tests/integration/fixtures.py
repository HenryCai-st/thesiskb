"""Synthetic fixtures, generated at test time (no real data in the repo, BUILD_GUIDE §8)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PDF_LINES = [
    "Denoising Diffusion Models for Thesis Testing",
    "1 Introduction",
    "Diffusion models add Gaussian noise to data and learn to reverse the process.",
    "Score-based models estimate the gradient of the log density of the data.",
    "2 Sampling",
    "DDIM reduces the number of sampling steps compared with DDPM.",
]

SPEECH = (
    "Hello, this is a short test recording. The supervisor prefers small diffusion models. "
    "The proposal is due in October."
)


def make_pdf(path: Path, lines: list[str] = PDF_LINES) -> Path:
    """A valid one-page PDF with real text (Helvetica), written by hand: no PDF library."""

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    ops = ["BT", "/F1 18 Tf", "72 740 Td", f"({esc(lines[0])}) Tj", "/F1 11 Tf"]
    for line in lines[1:]:
        size = 14 if line[:1].isdigit() else 11
        ops += [f"/F1 {size} Tf", "0 -28 Td", f"({esc(line)}) Tj"]
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    path.write_bytes(bytes(out))
    return path


def _windows_tts() -> bool:
    return sys.platform == "win32" and bool(shutil.which("powershell"))


def tts_available() -> bool:
    return bool((shutil.which("espeak-ng") or _windows_tts()) and shutil.which("ffmpeg"))


def make_speech(path: Path, text: str = SPEECH) -> Path:
    """~10 s of synthetic English speech (espeak-ng; the built-in voice on Windows), as
    16 kHz mono WAV."""
    raw = path.with_suffix(".raw.wav")
    if shutil.which("espeak-ng"):
        subprocess.run(["espeak-ng", "-s", "150", "-w", str(raw), text], check=True)
    else:
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.SetOutputToWaveFile('{raw}'); $s.Speak('{text}'); $s.Dispose()"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(raw),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(path),
        ],
        check=True,
    )
    raw.unlink()
    return path
