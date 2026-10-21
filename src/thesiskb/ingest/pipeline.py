"""Stage runner for ingestion. Idempotent: each (source, stage) has one row in `jobs`.

Stages, run as batches so only one heavy GPU model is loaded at a time:
  parse  documents: Docling → data/processed/<sha>/docling.json   (GPU, Ollama unloaded first)
         images:    VLM + OCR → data/processed/<sha>/image.json   (Ollama vision model only)
  chunk  stage output → segments + chunks rows                     (CPU)
  embed  bge-m3 embeddings + tsvector for chunks                   (Ollama embedding model)

A parse output already on disk counts as done, so re-ingesting after a DB reset needs no GPU.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.orm import Session

from thesiskb.core.config import PipelineConfig, get_settings
from thesiskb.core.models import Chunk, Job, Segment, Source
from thesiskb.ingest.chunking import ChunkDraft, chunk_blocks, get_token_counter
from thesiskb.ingest.detect import DOCUMENT_MODALITIES, DetectedFile, register
from thesiskb.ingest.documents import FTS_CONFIG, detect_language, extract_blocks
from thesiskb.ingest.metadata import arxiv_id_from_name, lookup_arxiv
from thesiskb.llm.gateway import LLMGateway

log = logging.getLogger(__name__)

STAGES = ("parse", "chunk", "embed")
SUPPORTED_MODALITIES = DOCUMENT_MODALITIES | {"image"}
PARSE_OUTPUT = {"image": "image.json"}  # everything else: docling.json
EMBED_BATCH = 16


def processed_dir(cfg: PipelineConfig, sha256: str) -> Path:
    return Path(cfg.processed_dir) / sha256


def parse_output(cfg: PipelineConfig, src: Source) -> Path:
    return processed_dir(cfg, src.sha256) / PARSE_OUTPUT.get(src.modality, "docling.json")


def job_done(session: Session, source: Source, stage: str) -> bool:
    status = session.scalar(
        select(Job.status).where(Job.source_id == source.id, Job.stage == stage)
    )
    return status == "done"


def run_stage(
    session: Session, source: Source, stage: str, fn: Callable[[], None], force: bool = False
) -> bool:
    """Run `fn` unless the stage is already done. Returns True if work was done."""
    job = session.scalar(select(Job).where(Job.source_id == source.id, Job.stage == stage))
    if job and job.status == "done" and not force:
        return False
    if job is None:
        job = Job(source_id=source.id, stage=stage)
        session.add(job)
    job.status, job.error, job.started_at, job.finished_at = "running", None, _now(), None
    session.commit()
    try:
        fn()
    except Exception as e:
        session.rollback()
        job = session.merge(job)
        job.status, job.error, job.finished_at = "failed", f"{type(e).__name__}: {e}", _now()
        session.commit()
        raise
    job.status, job.finished_at = "done", _now()
    session.commit()
    return True


def _now() -> datetime:
    return datetime.now(UTC)


class IngestReport(dict):
    """stage → number of sources where work was actually done; 'failed' → [(file, error)]."""


def ingest(
    session: Session,
    files: list[DetectedFile],
    cfg: PipelineConfig,
    gateway: LLMGateway,
    source_type: str | None = None,
    force: bool = False,
    metadata_lookup: bool = True,
) -> IngestReport:
    report = IngestReport({s: 0 for s in STAGES})
    report["failed"] = []
    sources = [
        (register(session, f, source_type), f) for f in files if f.modality in SUPPORTED_MODALITIES
    ]
    session.commit()

    # --- parse: one GPU batch per modality ---
    todo: dict[str, list[tuple[Source, DetectedFile]]] = {"document": [], "image": []}
    for s, f in sources:
        if force:
            pass
        elif job_done(session, s, "parse"):
            continue
        elif parse_output(cfg, s).exists():
            run_stage(session, s, "parse", lambda: None)  # output cached on disk; no GPU work
            continue
        todo["image" if s.modality == "image" else "document"].append((s, f))

    if todo["document"]:
        _parse_documents(session, todo["document"], cfg, report)
    if todo["image"]:
        _parse_images(session, todo["image"], cfg, gateway, report)

    # --- chunk (CPU) ---
    count = get_token_counter()
    for src, f in sources:
        if not job_done(session, src, "parse"):
            continue
        chunker = _chunk_image if src.modality == "image" else _chunk_document
        work = partial(chunker, session, src, f, cfg, count, metadata_lookup)
        if _safe(report, session, src, f, "chunk", work, force):
            report["chunk"] += 1
            # new chunks invalidate old embeddings
            session.execute(delete(Job).where(Job.source_id == src.id, Job.stage == "embed"))
            session.commit()

    # --- embed (Ollama bge-m3) ---
    for src, f in sources:
        if not job_done(session, src, "chunk"):
            continue
        work = partial(_embed_source, session, src, gateway)
        if _safe(report, session, src, f, "embed", work, force):
            report["embed"] += 1
    return report


def _parse_documents(session, todo, cfg: PipelineConfig, report: IngestReport) -> None:
    from thesiskb.ingest.documents import DoclingConverter
    from thesiskb.llm.ollama import unload_all

    unload_all(get_settings().ollama_base_url)
    converter = DoclingConverter(do_ocr=bool((cfg.model_extra or {}).get("pdf_ocr", False)))
    try:
        for src, f in todo:
            log.info("parse %s", f.path.name)
            work = partial(converter.convert, f.path, parse_output(cfg, src))
            if _safe(report, session, src, f, "parse", work, force=True):
                report["parse"] += 1
    finally:
        converter.close()


def _parse_images(session, todo, cfg, gateway: LLMGateway, report: IngestReport) -> None:
    from thesiskb.ingest.images import read_image
    from thesiskb.llm.ollama import unload_all

    settings = get_settings()
    unload_all(settings.ollama_base_url)  # free VRAM for the vision model
    langs = (cfg.model_extra or {}).get("ocr", {}).get("languages", ["eng"])
    try:
        for src, f in todo:
            log.info("read image %s", f.path.name)
            work = partial(
                read_image,
                gateway,
                f.path,
                processed_dir(cfg, src.sha256),
                langs,
                settings.tesseract_cmd,
                settings.tessdata_dir,
            )
            if _safe(report, session, src, f, "parse", work, force=True):
                report["parse"] += 1
    finally:
        unload_all(settings.ollama_base_url)  # don't leave the VLM next to the answer LLM


def _safe(
    report: IngestReport,
    session: Session,
    src: Source,
    f: DetectedFile,
    stage: str,
    fn: Callable[[], object],
    force: bool,
) -> bool:
    """run_stage, but a failure is recorded (job row + report) instead of aborting the batch."""
    try:
        return run_stage(session, src, stage, fn, force)
    except Exception as e:  # noqa: BLE001
        log.exception("%s failed for %s", stage, f.path.name)
        report["failed"].append((f.path.name, f"{stage}: {type(e).__name__}: {e}"))
        return False


def _replace_rows(session: Session, src: Source) -> None:
    session.execute(delete(Chunk).where(Chunk.source_id == src.id))
    session.execute(delete(Segment).where(Segment.source_id == src.id))


def _add_chunks(
    session: Session, src: Source, drafts: list[ChunkDraft], fts: str, seg_ids_for
) -> None:
    for d in drafts:
        session.add(
            Chunk(
                source_id=src.id,
                segment_ids=seg_ids_for(d),
                text=d.text,
                context_header=d.context_header,
                fts_config=fts,
                locator=d.locator,
            )
        )
    src.ingested_at = _now()


def _chunk_document(
    session: Session,
    src: Source,
    f: DetectedFile,
    cfg: PipelineConfig,
    count,
    metadata_lookup: bool,
) -> None:
    out_dir = processed_dir(cfg, src.sha256)
    parsed = extract_blocks(out_dir / "docling.json")
    sample = " ".join(b.text for b in parsed.blocks[:80])
    lang = detect_language(sample)

    meta = None
    if metadata_lookup and (aid := arxiv_id_from_name(f.path.name)):
        meta = lookup_arxiv(aid)
    src.title = (meta.title if meta else None) or parsed.title or src.title
    if meta:
        src.authors = meta.authors or src.authors
        src.published = meta.published or src.published
        src.origin = meta.origin or src.origin
    src.language = lang
    src.pipeline_version = cfg.pipeline_version

    _replace_rows(session, src)
    page_segments: dict[int, Segment] = {}
    for page, page_text in sorted(parsed.page_texts.items()):
        seg = Segment(
            source_id=src.id,
            kind="page",
            text=page_text,
            locator={"page": page, "size": list(parsed.page_sizes.get(page, ()))},
        )
        session.add(seg)
        page_segments[page] = seg
    session.flush()

    def seg_ids(d: ChunkDraft):
        first, last = d.locator.get("page"), d.locator.get("page_end")
        pages = range(first or 0, (last or first or 0) + 1)
        return [page_segments[p].id for p in pages if p in page_segments]

    drafts = chunk_blocks(parsed.blocks, cfg.chunking, _short_title(src), count)
    _add_chunks(session, src, drafts, FTS_CONFIG.get(lang, "simple"), seg_ids)
    (out_dir / "blocks.json").write_text(
        json.dumps([b.model_dump() for b in parsed.blocks], ensure_ascii=False), encoding="utf-8"
    )


def _chunk_image(
    session: Session,
    src: Source,
    f: DetectedFile,
    cfg: PipelineConfig,
    count,
    metadata_lookup: bool,
) -> None:
    from thesiskb.ingest.images import ImageReading, image_blocks

    reading = ImageReading.model_validate_json(
        (processed_dir(cfg, src.sha256) / "image.json").read_text(encoding="utf-8")
    )
    blocks = image_blocks(reading, src.path)
    lang = detect_language(" ".join(b.text for b in blocks))
    src.language = lang
    src.pipeline_version = cfg.pipeline_version

    _replace_rows(session, src)
    segments = [
        Segment(source_id=src.id, kind="image", text=b.text, locator=b.locator) for b in blocks
    ]
    session.add_all(segments)
    session.flush()
    all_ids = [s.id for s in segments]

    drafts = chunk_blocks(blocks, cfg.chunking, _short_title(src), count)
    _add_chunks(session, src, drafts, FTS_CONFIG.get(lang, "simple"), lambda d: all_ids)


def _short_title(src: Source) -> str:
    """'Paper: <title> (2022)' style prefix for context headers."""
    kind = (src.source_type or "document").capitalize()
    year = f" ({src.published.year})" if src.published else ""
    title = src.title if src.modality != "image" else src.path.rsplit("/", 1)[-1]
    return f"{kind}: {title}{year}"


def _embed_source(session: Session, src: Source, gateway: LLMGateway) -> None:
    chunks = session.scalars(
        select(Chunk).where(Chunk.source_id == src.id).order_by(Chunk.id)
    ).all()
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        vectors = gateway.embed([embedding_text(c) for c in batch])
        for c, v in zip(batch, vectors, strict=True):
            c.embedding = v
    session.flush()
    session.execute(
        update(Chunk)
        .where(Chunk.source_id == src.id)
        .values(
            tsv=func.to_tsvector(
                text("fts_config::regconfig"),
                func.coalesce(Chunk.context_header, "") + " " + Chunk.text,
            )
        )
    )


def embedding_text(c: Chunk) -> str:
    return f"{c.context_header}\n\n{c.text}" if c.context_header else c.text
