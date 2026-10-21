"""Stage runner for ingestion. Idempotent: each (source, stage) has one row in `jobs`.

Stages, run as batches so only one heavy GPU model is loaded at a time:
  parse  documents: Docling → data/processed/<sha>/docling.json   (GPU, Ollama unloaded first)
         images:    VLM + OCR → data/processed/<sha>/image.json   (Ollama vision model only)
         audio:     Whisper batch, then pyannote batch → transcript.json
         video:     as audio, then keyframe captions (VLM batch) → frames.json
  chunk  stage output → segments + chunks rows                     (CPU)
  embed  bge-m3 embeddings + tsvector for chunks                   (Ollama embedding model)
  extract tasks / statements / entities per chunk → extract.json   (answer-size LLM)

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

from thesiskb.core.config import PipelineConfig, get_models_config, get_settings
from thesiskb.core.models import Chunk, Job, Person, Segment, Source, Speaker
from thesiskb.ingest.chunking import ChunkDraft, chunk_blocks, get_token_counter
from thesiskb.ingest.detect import DOCUMENT_MODALITIES, DetectedFile, register
from thesiskb.ingest.documents import FTS_CONFIG, detect_language, extract_blocks
from thesiskb.ingest.metadata import arxiv_id_from_name, lookup_arxiv
from thesiskb.llm.gateway import LLMGateway

log = logging.getLogger(__name__)

STAGES = ("parse", "chunk", "embed", "extract")
SUPPORTED_MODALITIES = DOCUMENT_MODALITIES | {"image", "audio", "video"}
# parse = model outputs only; turns are built from them in the (cheap) chunk stage.
# video: frames.json is written last, after diarization.json
PARSE_OUTPUT = {"image": "image.json", "audio": "diarization.json", "video": "frames.json"}
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
    extract: bool = True,
) -> IngestReport:
    report = IngestReport({s: 0 for s in STAGES})
    report["failed"] = []
    sources = [
        (register(session, f, source_type), f) for f in files if f.modality in SUPPORTED_MODALITIES
    ]
    session.commit()

    # --- parse: one GPU batch per modality ---
    todo: dict[str, list[tuple[Source, DetectedFile]]] = {"document": [], "image": [], "av": []}
    for s, f in sources:
        if force:
            pass
        elif job_done(session, s, "parse"):
            continue
        elif parse_output(cfg, s).exists():
            run_stage(session, s, "parse", lambda: None)  # output cached on disk; no GPU work
            continue
        group = {"image": "image", "audio": "av", "video": "av"}.get(s.modality, "document")
        todo[group].append((s, f))

    if todo["document"]:
        _parse_documents(session, todo["document"], cfg, report)
    if todo["image"]:
        _parse_images(session, todo["image"], cfg, gateway, report)
    if todo["av"]:
        _parse_av(session, todo["av"], cfg, gateway, report, force)

    # --- chunk (CPU) ---
    count = get_token_counter()
    for src, f in sources:
        if not job_done(session, src, "parse"):
            continue
        chunker = {"image": _chunk_image, "audio": _chunk_av, "video": _chunk_av}.get(
            src.modality, _chunk_document
        )
        work = partial(chunker, session, src, f, cfg, count, metadata_lookup)
        if _safe(report, session, src, f, "chunk", work, force):
            report["chunk"] += 1
            # new chunks invalidate old embeddings and extractions
            session.execute(
                delete(Job).where(Job.source_id == src.id, Job.stage.in_(("embed", "extract")))
            )
            session.commit()

    # --- embed (Ollama bge-m3) ---
    for src, f in sources:
        if not job_done(session, src, "chunk"):
            continue
        work = partial(_embed_source, session, src, gateway)
        if _safe(report, session, src, f, "embed", work, force):
            report["embed"] += 1

    # --- extract (answer-size LLM; cached per chunk text) ---
    if extract:
        for src, f in sources:
            if not job_done(session, src, "embed"):
                continue
            work = partial(_extract_source, session, src, cfg, gateway)
            if _safe(report, session, src, f, "extract", work, force):
                report["extract"] += 1
    return report


def _extract_source(session: Session, src: Source, cfg: PipelineConfig, gateway) -> None:
    from thesiskb.ingest.extract import extract_source

    rep = extract_source(session, src, gateway, processed_dir(cfg, src.sha256) / "extract.json")
    log.info(
        "extract %s: %d chunks (%d LLM calls) → %d tasks, %d statements, %d entity mentions, "
        "%d relations; %d items dropped without verbatim evidence",
        src.path.rsplit("/", 1)[-1],
        rep.chunks,
        rep.llm_calls,
        rep.tasks,
        rep.statements,
        rep.entities,
        rep.relations,
        rep.dropped_no_evidence,
    )


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


def _parse_av(session, todo, cfg, gateway: LLMGateway, report: IngestReport, force: bool):
    """Audio/video parse as model batches: Whisper → pyannote → (video) VLM keyframes.

    Each step's output is cached in data/processed/<sha>/, so a re-run resumes where it failed.
    """
    from thesiskb.ingest import audio, video
    from thesiskb.llm.ollama import unload_all

    settings = get_settings()
    models = get_models_config()
    extra = cfg.model_extra or {}
    failed: dict[str, str] = {}
    dirs = {src.sha256: processed_dir(cfg, src.sha256) for src, _ in todo}

    def step(name: str, items, fn) -> None:
        for src, f in items:
            if src.sha256 in failed:
                continue
            try:
                fn(src, f, dirs[src.sha256])
            except Exception as e:  # noqa: BLE001 — recorded, batch continues
                log.exception("%s failed for %s", name, f.path.name)
                failed[src.sha256] = f"{name}: {type(e).__name__}: {e}"

    def need(filename: str):
        return [(s, f) for s, f in todo if force or not (dirs[s.sha256] / filename).exists()]

    step(
        "extract audio",
        need("audio.wav"),
        lambda s, f, d: audio.extract_wav(f.path, d / "audio.wav"),
    )

    unload_all(settings.ollama_base_url)
    if items := need("asr.json"):
        asr_cfg = models.role("asr")
        log.info("transcribe %d file(s) with %s", len(items), asr_cfg.model)
        tr = audio.Transcriber(
            asr_cfg.model, (asr_cfg.model_extra or {}).get("compute_type", "float16")
        )
        try:
            step(
                "transcribe", items, lambda s, f, d: tr.transcribe(d / "audio.wav", d / "asr.json")
            )
        finally:
            tr.close()
    if items := need("diarization.json"):
        log.info("diarize %d file(s)", len(items))
        try:
            dia = audio.Diarizer(models.role("diarization").model, settings.hf_token)
        except Exception as e:  # noqa: BLE001 — e.g. missing token: every file fails the same
            for s, _ in items:
                failed.setdefault(s.sha256, f"diarize: {type(e).__name__}: {e}")
        else:
            try:
                step(
                    "diarize",
                    items,
                    lambda s, f, d: dia.diarize(d / "audio.wav", d / "diarization.json"),
                )
            finally:
                dia.close()

    videos = [(s, f) for s, f in need("frames.json") if s.modality == "video"]
    if videos:
        vcfg = extra.get("video", {})
        try:
            step(
                "keyframes",
                videos,
                lambda s, f, d: video.read_keyframes(
                    gateway,
                    f.path,
                    d,
                    scene_threshold=vcfg.get("scene_threshold", 0.3),
                    every=vcfg.get("keyframe_fallback_seconds", 30),
                ),
            )
        finally:
            unload_all(settings.ollama_base_url)

    for src, f in todo:
        err = failed.get(src.sha256)

        def done(err=err):
            if err:
                raise RuntimeError(err)

        if _safe(report, session, src, f, "parse", done, force=True):
            report["parse"] += 1


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


def speaker_names(session: Session, src: Source) -> dict[str, str]:
    """diarization label → person name, for labels mapped with `tkb speakers label`."""
    rows = session.execute(
        select(Speaker.diarization_label, Person.name)
        .join(Person, Person.id == Speaker.person_id)
        .where(Speaker.source_id == src.id)
    ).all()
    return dict(rows)


def _chunk_av(
    session: Session,
    src: Source,
    f: DetectedFile,
    cfg: PipelineConfig,
    count,
    metadata_lookup: bool,
) -> None:
    from thesiskb.ingest.audio import build_transcript, transcript_blocks
    from thesiskb.ingest.video import Keyframe, frame_blocks

    d = processed_dir(cfg, src.sha256)
    audio_cfg = (cfg.model_extra or {}).get("audio", {})
    transcript = build_transcript(
        json.loads((d / "asr.json").read_text("utf-8")),
        json.loads((d / "diarization.json").read_text("utf-8")),
        max_turn=audio_cfg.get("max_line_seconds", 0.0),
    )
    (d / "transcript.json").write_text(transcript.model_dump_json(), encoding="utf-8")
    frames = []
    if (d / "frames.json").exists():
        frames = [Keyframe(**k) for k in json.loads((d / "frames.json").read_text("utf-8"))]
    src.language = transcript.language
    src.pipeline_version = cfg.pipeline_version

    speakers = {
        s.diarization_label: s
        for s in session.scalars(select(Speaker).where(Speaker.source_id == src.id))
    }
    for label in transcript.speakers:
        if label not in speakers:
            speakers[label] = Speaker(source_id=src.id, diarization_label=label)
            session.add(speakers[label])
    session.flush()

    window = tuple((cfg.model_extra or {}).get("audio", {}).get("turn_window_seconds", [60, 120]))
    blocks = transcript_blocks(
        transcript, speaker_names(session, src), window, frame_blocks(frames)
    )
    _replace_rows(session, src)
    segments = []
    for b in blocks:
        label = b.locator.get("speaker")
        segments.append(
            Segment(
                source_id=src.id,
                kind=b.kind,
                text=b.text,
                speaker_id=speakers[label].id if label in speakers else None,
                locator=b.locator,
            )
        )
    session.add_all(segments)
    session.flush()

    drafts = chunk_blocks(blocks, cfg.chunking, _short_title(src), count)
    fts = FTS_CONFIG.get((transcript.language or "en")[:2], "simple")
    for dft in drafts:
        a, z = dft.locator.get("t_start", 0), dft.locator.get("t_end") or 0
        labels = dft.locator.get("speakers", [])
        session.add(
            Chunk(
                source_id=src.id,
                segment_ids=[s.id for s in segments if a <= s.locator["t_start"] <= z],
                text=dft.text,
                context_header=dft.context_header,
                fts_config=fts,
                speaker_id=speakers[labels[0]].id if len(labels) == 1 else None,
                locator=dft.locator,
            )
        )
    src.ingested_at = _now()


def rechunk(session: Session, src: Source, cfg: PipelineConfig, gateway: LLMGateway) -> None:
    """Redo chunk + embed for one source (e.g. after speaker labels changed)."""
    f = DetectedFile(path=Path(src.path), sha256=src.sha256, modality=src.modality)
    chunker = {"image": _chunk_image, "audio": _chunk_av, "video": _chunk_av}.get(
        src.modality, _chunk_document
    )
    run_stage(
        session,
        src,
        "chunk",
        partial(chunker, session, src, f, cfg, get_token_counter(), False),
        force=True,
    )
    run_stage(session, src, "embed", partial(_embed_source, session, src, gateway), force=True)
    run_stage(
        session, src, "extract", partial(_extract_source, session, src, cfg, gateway), force=True
    )


def _short_title(src: Source) -> str:
    """'Paper: <title> (2022)' style prefix for context headers."""
    kind = (src.source_type or "document").capitalize()
    year = f" ({src.published.year})" if src.published else ""
    title = src.title if src.modality not in ("image",) else src.path.rsplit("/", 1)[-1]
    if src.modality in ("audio", "video") and title:
        title = title.replace("_", " ").strip()
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
