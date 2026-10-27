"""Hashing, modality detection, dedupe and registration of `sources` rows.

Original files are never renamed or copied: the `sources` table is the index from content hash
(sha256) to the file's current path. Moving or renaming a file keeps its sha256, so re-ingesting
only updates `path`. A file that changed at the same path is a new source that `replaces` the
old one; the old one is retired once the new one is ingested (pipeline.retire_replaced).

Meeting exports in CSV (Granola) are split into one unit per row: each row has its own sha256
and the path "<file>.csv#<row id>", so a grown CSV only ingests its new rows.
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
    ".htm": "document",
    ".xhtml": "document",
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
    ".csv": "meeting_csv",
}

DOCUMENT_MODALITIES = {"pdf", "epub", "document"}


class DetectedFile(BaseModel):
    path: Path
    sha256: str
    modality: str
    unit: str | None = None  # row id inside a multi-unit file (meeting CSV)

    @property
    def source_path(self) -> str:
        """`sources.path`: the file, plus "#<unit>" for one row of a CSV."""
        return self.path.as_posix() + (f"#{self.unit}" if self.unit else "")

    @property
    def name(self) -> str:
        return self.path.name + (f"#{self.unit[:8]}" if self.unit else "")


def sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def detect_modality(path: Path) -> str | None:
    return MODALITY_BY_SUFFIX.get(path.suffix.lower())


def _saved_page_asset(f: Path) -> bool:
    """A file inside "<name>_files/" next to "<name>.html" (a browser's "Save page as")."""
    for d in f.parents:
        if d.name.endswith("_files"):
            stem = d.name.removesuffix("_files")
            if any((d.parent / (stem + ext)).exists() for ext in (".html", ".htm")):
                return True
    return False


def iter_input_files(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_dir():
            yield from sorted(
                f
                for f in p.rglob("*")
                if f.is_file() and detect_modality(f) and not _saved_page_asset(f)
            )
        elif p.is_file() and detect_modality(p):
            yield p


def detect(paths: Iterable[Path]) -> list[DetectedFile]:
    """Hash and classify files, dropping exact duplicates (same content, different path)."""
    seen: dict[str, DetectedFile] = {}
    for path in iter_input_files(paths):
        if detect_modality(path) == "meeting_csv":
            from thesiskb.ingest.meetings_csv import csv_units

            for unit, sha in csv_units(path):  # [] if the CSV is not a known meeting export
                seen.setdefault(
                    sha,
                    DetectedFile(
                        path=path.resolve(), sha256=sha, modality="meeting_csv", unit=unit
                    ),
                )
            continue
        sha = sha256_file(path)
        if sha not in seen:
            seen[sha] = DetectedFile(
                path=path.resolve(), sha256=sha, modality=detect_modality(path)
            )
    return list(seen.values())


def default_source_type(f: DetectedFile) -> str | None:
    if f.modality == "epub":
        return "book"
    if f.path.suffix.lower() in (".html", ".htm", ".xhtml"):
        return "web"
    if f.modality in DOCUMENT_MODALITIES:
        return "paper"
    return {"image": "note", "audio": "meeting", "video": "demo", "meeting_csv": "meeting"}.get(
        f.modality
    )


def register(
    session: Session, f: DetectedFile, source_type: str | None = None, local_only: bool = False
) -> Source:
    """Insert or refresh the `sources` row for a file. Returns the (possibly existing) row."""
    src = session.scalar(select(Source).where(Source.sha256 == f.sha256))
    path = f.source_path
    if src is None:
        # same path, other content: the file changed — the new source replaces the old one
        previous = session.scalar(
            select(Source)
            .where(Source.path == path)
            .order_by(Source.ingested_at.desc().nullslast())
        )
        src = Source(
            sha256=f.sha256,
            path=path,
            modality=f.modality,
            source_type=source_type
            or (previous.source_type if previous else None)
            or default_source_type(f),
            title=f.path.stem,
            local_only=local_only or bool(previous and previous.local_only),
            verification_status=previous.verification_status if previous else "unverified",
            replaces=previous.id if previous else None,
        )
        session.add(src)
        session.flush()
    else:
        if src.path != path:
            src.path = path  # file was moved or renamed; content is unchanged
        if source_type:
            src.source_type = source_type
        if local_only:
            src.local_only = True  # only ever tightened here; `tkb local-only --off` relaxes
    return src
