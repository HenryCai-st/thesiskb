"""Photos, whiteboards, post-its, handwriting: orientation fix, OCR + VLM reading.

Per image (GPU stage `parse`, output data/processed/<sha>/image.json + upright.jpg):
  1. EXIF transpose, then find the upright rotation: transcribe a short prefix at 0/90/180/270°
     and keep the rotation with the highest mean token logprob. (Asking the VLM which way the
     text is rotated gave the same answer for every rotation; confidence was 4/4 correct.)
  2. VLM transcription of the upright image (best for handwriting).
  3. VLM structured pass: kind, short description, and one entry per post-it note.
  4. Tesseract OCR (deu+eng) of the upright image, kept alongside for printed text.
The reading that gets indexed is chosen by `choose_text()`; both are stored.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import re
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from thesiskb.ingest.chunking import Block
from thesiskb.llm.gateway import LLMGateway, image_message

log = logging.getLogger(__name__)

ROTATIONS = (0, 90, 180, 270)  # counter-clockwise, as PIL's Image.rotate
MAX_SIDE = 1024  # Qwen2.5-VL reads this image size well; larger is slower, not better

ImageKind = Literal[
    "handwritten_note",
    "whiteboard",
    "post_its",
    "printed_document",
    "screenshot",
    "diagram",
    "photo",
]

TRANSCRIBE_PROMPT = (
    "Transcribe all handwritten or printed text in this image exactly, in reading order. "
    "Text that belongs together (e.g. a date and the label written next to or under it) goes on "
    "one line, separated by ' - '. Copy dates, numbers and names exactly as written. "
    "Output only the text."
)

DESCRIBE_PROMPT = """The text in this image was transcribed as:
---
{transcription}
---
Describe the image. Return JSON with these keys:
"kind": one of handwritten_note, whiteboard, post_its, printed_document, screenshot, diagram, photo
"description": 1-3 sentences on what it shows and how it is laid out (e.g. a timeline with arrows \
from one step to the next, a table, a mind map).
"notes": only if there are separate sticky notes / post-its: a list of objects \
{{"text": "...", "color": "...", "position": "top-left|top|top-right|left|center|right|bottom-left|\
bottom|bottom-right"}}; otherwise an empty list."""


class Note(BaseModel):
    text: str
    color: str | None = None
    position: str | None = None


class VisionDescription(BaseModel):
    kind: ImageKind = "photo"
    description: str = ""
    notes: list[Note] = Field(default_factory=list)


class OcrResult(BaseModel):
    text: str = ""
    mean_conf: float = 0.0  # Tesseract word confidence 0-100
    n_words: int = 0


class ImageReading(BaseModel):
    rotation: int  # degrees counter-clockwise applied after EXIF transpose
    orientation_scores: dict[int, float]
    size: tuple[int, int]  # upright image size in pixels
    transcription: str
    transcription_logprob: float
    vision: VisionDescription
    ocr: OcrResult
    chosen: Literal["vlm", "ocr"]


def _jpeg(im, max_side: int = MAX_SIDE) -> bytes:
    im = im.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else -math.inf


def find_rotation(gateway: LLMGateway, im, role: str = "vision") -> tuple[int, dict[int, float]]:
    """Rotation (ccw degrees) whose short transcription the VLM is most confident about."""
    scores: dict[int, float] = {}
    for angle in ROTATIONS:
        rotated = im.rotate(angle, expand=True) if angle else im
        _, lps = gateway.chat_with_logprobs(
            image_message(TRANSCRIBE_PROMPT, _jpeg(rotated)), role=role, max_tokens=48
        )
        scores[angle] = round(_mean(lps), 4)
    if all(math.isinf(s) for s in scores.values()):
        return 0, scores  # provider gave no logprobs: keep EXIF orientation
    return max(scores, key=scores.get), scores


def run_ocr(
    image_path: Path, languages: list[str], cmd: str = "tesseract", tessdata: str | None = None
) -> OcrResult:
    # "-c tessedit_create_tsv=1" rather than the "tsv" config name: the config file lives in
    # the install's tessdata/configs, which a custom --tessdata-dir doesn't have
    args = [cmd, str(image_path), "-", "-l", "+".join(languages), "-c", "tessedit_create_tsv=1"]
    if tessdata:
        args[1:1] = ["--tessdata-dir", tessdata]
    try:
        out = subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8", timeout=120, check=True
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        log.warning("tesseract failed: %s", e)
        return OcrResult()
    lines: dict[tuple, list[str]] = {}
    confs: list[float] = []
    for row in csv.DictReader(io.StringIO(out), delimiter="\t", quoting=csv.QUOTE_NONE):
        word = (row.get("text") or "").strip()
        conf = float(row.get("conf") or -1)
        if not word or conf < 0:
            continue
        key = (row["block_num"], row["par_num"], row["line_num"])
        lines.setdefault(key, []).append(word)
        confs.append(conf)
    return OcrResult(
        text="\n".join(" ".join(ws) for ws in lines.values()),
        mean_conf=round(sum(confs) / len(confs), 1) if confs else 0.0,
        n_words=len(confs),
    )


def choose_text(kind: str, ocr: OcrResult, min_conf: float = 80, min_words: int = 20) -> str:
    """VLM for handwriting/whiteboards/notes; Tesseract only for clearly printed text."""
    if kind in ("printed_document", "screenshot") and (
        ocr.mean_conf >= min_conf and ocr.n_words >= min_words
    ):
        return "ocr"
    return "vlm"


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, re.S)
    return json.loads(m.group(0) if m else raw)


def read_image(
    gateway: LLMGateway,
    src: Path,
    out_dir: Path,
    languages: list[str],
    tesseract_cmd: str = "tesseract",
    tessdata: str | None = None,
    role: str = "vision",
) -> ImageReading:
    from PIL import Image, ImageOps

    with Image.open(src) as raw:
        im = ImageOps.exif_transpose(raw).convert("RGB")
    rotation, scores = find_rotation(gateway, im, role)
    upright = im.rotate(rotation, expand=True) if rotation else im
    out_dir.mkdir(parents=True, exist_ok=True)
    upright_path = out_dir / "upright.jpg"
    upright.save(upright_path, "JPEG", quality=92)

    transcription, lps = gateway.chat_with_logprobs(
        image_message(TRANSCRIBE_PROMPT, _jpeg(upright)), role=role, max_tokens=1024
    )
    transcription = transcription.strip()

    raw = gateway.chat(
        image_message(DESCRIBE_PROMPT.format(transcription=transcription), _jpeg(upright)),
        role=role,
        response_format={"type": "json_object"},
        max_tokens=1024,
    )
    try:
        vision = VisionDescription.model_validate(_parse_json(raw))
    except (ValidationError, ValueError) as e:
        log.warning("vision description unparsable for %s: %s", src.name, e)
        vision = VisionDescription(description=raw.strip()[:500])

    ocr = run_ocr(upright_path, languages, tesseract_cmd, tessdata)
    reading = ImageReading(
        rotation=rotation,
        orientation_scores=scores,
        size=upright.size,
        transcription=transcription,
        transcription_logprob=round(_mean(lps), 4),
        vision=vision,
        ocr=ocr,
        chosen=choose_text(vision.kind, ocr),
    )
    (out_dir / "image.json").write_text(reading.model_dump_json(indent=2), encoding="utf-8")
    return reading


_POSITIONS = {
    "top-left": (0, 0, 0.5, 0.5),
    "top": (0, 0, 1, 0.5),
    "top-right": (0.5, 0, 1, 0.5),
    "left": (0, 0, 0.5, 1),
    "center": (0.25, 0.25, 0.75, 0.75),
    "right": (0.5, 0, 1, 1),
    "bottom-left": (0, 0.5, 0.5, 1),
    "bottom": (0, 0.5, 1, 1),
    "bottom-right": (0.5, 0.5, 1, 1),
}


def image_blocks(reading: ImageReading, image_path: str) -> list[Block]:
    """Blocks for chunking. Locators point at the original file plus the rotation to show it
    upright; post-it notes get an approximate region (fractions of the upright image)."""
    base = {"image": image_path, "rotation": reading.rotation}
    text = reading.transcription if reading.chosen == "vlm" else reading.ocr.text
    kind_label = reading.vision.kind.replace("_", " ")
    # No "in order" line: qwen2.5vl:7b lists a circular arrow timeline in page-position order
    # (March before January), and a confidently wrong order is worse than none. Dated items
    # can be ordered downstream (answer LLM now, tasks table in Milestone 5).
    caption = f"{kind_label.capitalize()}: {reading.vision.description}".strip()
    blocks = [Block(text=caption, kind="caption", section=["Image"], locator=dict(base))]
    if reading.vision.notes:
        for note in reading.vision.notes:
            region = _POSITIONS.get((note.position or "").lower())
            colour = f" ({note.color} note)" if note.color else ""
            blocks.append(
                Block(
                    text=f"{note.text}{colour}",
                    kind="note",
                    section=["Image"],
                    locator=base | ({"region": list(region)} if region else {}),
                )
            )
    elif text:
        blocks.append(Block(text=text, kind="text", section=["Image"], locator=dict(base)))
    return blocks
