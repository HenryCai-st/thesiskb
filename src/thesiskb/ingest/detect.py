"""Hashing, modality detection, dedupe and registration of `sources` rows.

Original files are never renamed or copied: the `sources` table is the index from content hash
(sha256) to the file's current path. Moving or renaming a file keeps its sha256, so re-ingesting
only updates `path`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from thesiskb.core.models import Source

MODALITY_BY_SUFFIX = {
    ".pdf": "pdf",
    ".epub": "epub",
    ".docx": "document",
    ".pptx": "document",
    ".md": "document",
    ".html": "document",
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".heic": "image",
    ".webp": "image",
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
    ".flac": "audio",
    ".ogg": "audio",
    ".mp4": "video",
    ".mkv": "video",
    ".mov": "video",
    ".webm": "video",
}

DOCUMENT_MODALITIES = {"pdf", "epub", "document"}


class DetectedFile(BaseModel):
    path: Path
    sha256: str
    modality: str


def sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def detect_modality(path: Path) -> str | None:
    return MODALITY_BY_SUFFIX.get(path.suffix.lower())


def iter_input_files(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*") if f.is_file() and detect_modality(f))
        elif p.is_file() and detect_modality(p):
            yield p


def detect(paths: Iterable[Path]) -> list[DetectedFile]:
    """Hash and classify files, dropping exact duplicates (same content, different path)."""
    seen: dict[str, DetectedFile] = {}
    for path in iter_input_files(paths):
        sha = sha256_file(path)
        if sha not in seen:
            seen[sha] = DetectedFile(
                path=path.resolve(), sha256=sha, modality=detect_modality(path)
            )
    return list(seen.values())


def default_source_type(f: DetectedFile) -> str | None:
    if f.modality in DOCUMENT_MODALITIES:
        return "paper"
    return {"image": "note", "audio": "meeting", "video": "demo"}.get(f.modality)


def register(session: Session, f: DetectedFile, source_type: str | None = None) -> Source:
    """Insert or refresh the `sources` row for a file. Returns the (possibly existing) row."""
    src = session.scalar(select(Source).where(Source.sha256 == f.sha256))
    path = f.path.as_posix()
    if src is None:
        src = Source(
            sha256=f.sha256,
            path=path,
            modality=f.modality,
            source_type=source_type or default_source_type(f),
            title=f.path.stem,
        )
        session.add(src)
        session.flush()
    else:
        if src.path != path:
            src.path = path  # file was moved or renamed; content is unchanged
        if source_type:
            src.source_type = source_type
    return src
