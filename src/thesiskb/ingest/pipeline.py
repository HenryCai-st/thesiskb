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
from thesiskb.ingest.parsers import (
    ParseContext,
    Parser,
    load_plugins,
    parser_for,
    register_parser,
)
from thesiskb.ingest.parsers import supported_modalities as registered_modalities
from thesiskb.llm.gateway import LLMGateway

log = logging.getLogger(__name__)

STAGES = ("parse", "chunk", "embed", "extract")
EMBED_BATCH = 16


def processed_dir(cfg: PipelineConfig, sha256: str) -> Path:
    return Path(cfg.processed_dir) / sha256


def parse_output(cfg: PipelineConfig, src: Source) -> Path:
    parser = parser_for(src.modality)
    return processed_dir(cfg, src.sha256) / (parser.output_for(src) if parser else "")


def supported_modalities(cfg: PipelineConfig | None = None) -> set[str]:
    """Modalities some parser handles, after loading plug-ins listed in pipeline.yaml."""
    load_plugins(((cfg.model_extra or {}).get("plugins") if cfg else None) or [])
    return registered_modalities()


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
    local_only: bool = False,
    progress: Callable[..., None] | None = None,
) -> IngestReport:
    """`progress(stage, filename)` is called once per file and stage (also when skipped or
    failed, so a bar always completes); `progress("extract", filename, i, n)` per chunk."""
    step = progress or (lambda *a: None)
    report = IngestReport({s: 0 for s in STAGES})
    report["failed"] = []
    modalities = supported_modalities(cfg)
    sources = [
        (register(session, f, source_type, local_only), f)
        for f in files
        if f.modality in modalities
    ]
    session.commit()
    ctx = _context(session, cfg, gateway, report, force, metadata_lookup, step)

    # --- parse: one batch per parser, so each GPU model family is loaded once ---
    todo: dict[str, list[tuple[Source, DetectedFile]]] = {}
    parsers: dict[str, Parser] = {}
    for s, f in sources:
        if force:
            pass
        elif job_done(session, s, "parse"):
            step("parse", f.path.name)
            continue
        elif parse_output(cfg, s).exists():
            run_stage(session, s, "parse", lambda: None)  # output cached on disk; no GPU work
            step("parse", f.path.name)
            continue
        parser = parser_for(s.modality)
        parsers[parser.name] = parser
        todo.setdefault(parser.name, []).append((s, f))
    for name, items in todo.items():
        parsers[name].parse(ctx, items)

    # --- chunk (CPU) ---
    count = get_token_counter()
    for src, f in sources:
        step("chunk", f.path.name)
        if not job_done(session, src, "parse"):
            continue
        work = partial(parser_for(src.modality).chunk, ctx, src, f, count)
        if _safe(report, session, src, f, "chunk", work, force):
            report["chunk"] += 1
            # new chunks invalidate old embeddings and extractions
            session.execute(
                delete(Job).where(Job.source_id == src.id, Job.stage.in_(("embed", "extract")))
            )
            session.commit()

    # --- embed (Ollama bge-m3) ---
    for src, f in sources:
        step("embed", f.path.name)
        if not job_done(session, src, "chunk"):
            continue
        work = partial(_embed_source, session, src, gateway)
        if _safe(report, session, src, f, "embed", work, force):
            report["embed"] += 1

    # --- extract (answer-size LLM; cached per chunk text) ---
    if extract:
        for src, f in sources:
            if job_done(session, src, "embed"):
                on_chunk = partial(step, "extract", f.path.name)
                work = partial(_extract_source, session, src, cfg, gateway, on_chunk)
                if _safe(report, session, src, f, "extract", work, force):
                    report["extract"] += 1
            step("extract", f.path.name)
    return report


def _context(
    session, cfg, gateway, report, force=False, metadata_lookup=True, step=lambda *a: None
) -> ParseContext:
    """ParseContext whose mark() records each file's parse stage in jobs (and the report)."""

    def mark(src, f, fn) -> bool:
        ok = _safe(report, session, src, f, "parse", fn, force=True)
        if ok:
            report["parse"] = report.get("parse", 0) + 1
        step("parse", f.path.name)
        return ok

    return ParseContext(
        session=session,
        cfg=cfg,
        gateway=gateway,
        force=force,
        metadata_lookup=metadata_lookup,
        mark=mark,
    )


def _extract_source(session: Session, src: Source, cfg: PipelineConfig, gateway, on_chunk=None):
    from thesiskb.ingest.extract import extract_source

    out = processed_dir(cfg, src.sha256) / "extract.json"
    rep = extract_source(session, src, gateway, out, on_chunk=on_chunk)
    log.info(
        "extract %s: %d chunks (%d LLM calls) → %d tasks, %d statements, %d entity mentions, "
        "%d relations; %d items dropped without verbatim evidence; %d chunks unparsable "
        "(retried next run)",
        src.path.rsplit("/", 1)[-1],
        rep.chunks,
        rep.llm_calls,
        rep.tasks,
        rep.statements,
        rep.entities,
        rep.relations,
        rep.dropped_no_evidence,
        rep.failed_chunks,
    )


def _parse_documents(ctx: ParseContext, todo) -> None:
    from thesiskb.ingest.documents import DoclingConverter
    from thesiskb.llm.ollama import unload_all

    unload_all(get_settings().ollama_base_url)
    converter = DoclingConverter(do_ocr=bool(ctx.options.get("pdf_ocr", False)))
    try:
        for src, f in todo:
            log.info("parse %s", f.path.name)
            ctx.mark(src, f, partial(converter.convert, f.path, parse_output(ctx.cfg, src)))
    finally:
        converter.close()


def _parse_images(ctx: ParseContext, todo) -> None:
    from thesiskb.ingest.images import read_image
    from thesiskb.llm.ollama import unload_all

    settings = get_settings()
    unload_all(settings.ollama_base_url)  # free VRAM for the vision model
    langs = ctx.options.get("ocr", {}).get("languages", ["eng"])
    try:
        for src, f in todo:
            log.info("read image %s", f.path.name)
            work = partial(
                read_image,
                ctx.gateway,
                f.path,
                ctx.out_dir(src),
                langs,
                settings.tesseract_cmd,
                settings.tessdata_dir,
                ctx.gateway.role_for("vision", src.local_only),
            )
            ctx.mark(src, f, work)
    finally:
        unload_all(settings.ollama_base_url)  # don't leave the VLM next to the answer LLM


def _parse_av(ctx: ParseContext, todo) -> None:
    """Audio/video parse as model batches: Whisper → pyannote → (video) VLM keyframes.

    Each step's output is cached in data/processed/<sha>/, so a re-run resumes where it failed.
    """
    from thesiskb.ingest import audio, video
    from thesiskb.llm.backends import create_backend
    from thesiskb.llm.ollama import unload_all

    settings = get_settings()
    models = get_models_config()
    extra, force, gateway = ctx.options, ctx.force, ctx.gateway
    failed: dict[str, str] = {}
    dirs = {src.sha256: ctx.out_dir(src) for src, _ in todo}

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
        tr = create_backend("asr", asr_cfg, settings)
        try:
            step(
                "transcribe", items, lambda s, f, d: tr.transcribe(d / "audio.wav", d / "asr.json")
            )
        finally:
            tr.close()
    if items := need("diarization.json"):
        log.info("diarize %d file(s)", len(items))
        try:
            dia = create_backend("diarization", models.role("diarization"), settings)
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
                    role=gateway.role_for("vision", s.local_only),
                ),
            )
        finally:
            unload_all(settings.ollama_base_url)

    for src, f in todo:
        err = failed.get(src.sha256)

        def done(err=err):
            if err:
                raise RuntimeError(err)

        ctx.mark(src, f, done)


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
        log.exception(
            "%s failed for %s", stage, f.path.name, extra={"sha256": f.sha256, "stage": stage}
        )
        msg = f"{stage}: {type(e).__name__}: {e}"
        if "APIConnectionError" in type(e).__name__ or "Connection refused" in str(e):
            msg += (
                f" — Ollama not reachable at {get_settings().ollama_base_url}; start the "
                "Ollama app, then re-run (finished stages are skipped)"
            )
        report["failed"].append((f.path.name, msg))
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


def _chunk_document(ctx: ParseContext, src: Source, f: DetectedFile, count) -> None:
    session, cfg = ctx.session, ctx.cfg
    out_dir = processed_dir(cfg, src.sha256)
    parsed = extract_blocks(out_dir / "docling.json")
    sample = " ".join(b.text for b in parsed.blocks[:80])
    lang = detect_language(sample)

    meta = None
    if ctx.metadata_lookup and (aid := arxiv_id_from_name(f.path.name)):
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


def _chunk_image(ctx: ParseContext, src: Source, f: DetectedFile, count) -> None:
    session, cfg = ctx.session, ctx.cfg
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


def write_blocks(ctx: ParseContext, src: Source, blocks, count, language: str | None = None):
    """Generic chunk stage for plug-in parsers: one segment per block, section-aware chunks.

    `blocks` are `ingest.chunking.Block`s with a locator each (e.g. {"line": 12}). Returns the
    number of chunks written."""
    lang = language or detect_language(" ".join(b.text for b in blocks[:80]))
    src.language = lang
    src.pipeline_version = ctx.cfg.pipeline_version
    _replace_rows(ctx.session, src)
    segments = [
        Segment(source_id=src.id, kind=b.kind, text=b.text, locator=b.locator) for b in blocks
    ]
    ctx.session.add_all(segments)
    ctx.session.flush()
    ids = [s.id for s in segments]
    drafts = chunk_blocks(blocks, ctx.cfg.chunking, _short_title(src), count)
    _add_chunks(ctx.session, src, drafts, FTS_CONFIG.get(lang, "simple"), lambda d: ids)
    return len(drafts)


def speaker_names(session: Session, src: Source) -> dict[str, str]:
    """diarization label → person name, for labels mapped with `tkb speakers label`."""
    rows = session.execute(
        select(Speaker.diarization_label, Person.name)
        .join(Person, Person.id == Speaker.person_id)
        .where(Speaker.source_id == src.id)
    ).all()
    return dict(rows)


def _chunk_av(ctx: ParseContext, src: Source, f: DetectedFile, count) -> None:
    session, cfg = ctx.session, ctx.cfg
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
    ctx = ParseContext(session=session, cfg=cfg, gateway=gateway, metadata_lookup=False)
    run_stage(
        session,
        src,
        "chunk",
        partial(parser_for(src.modality).chunk, ctx, src, f, get_token_counter()),
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
        vectors = gateway.embed(
            [embedding_text(c) for c in batch], role=gateway.role_for("embedding", src.local_only)
        )
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


# --- built-in parsers ------------------------------------------------------------------------


class DocumentParser(Parser):
    name = "documents"
    modalities = frozenset(DOCUMENT_MODALITIES)
    output = "docling.json"

    def parse(self, ctx, items):
        _parse_documents(ctx, items)

    def chunk(self, ctx, src, f, count):
        _chunk_document(ctx, src, f, count)


class ImageParser(Parser):
    name = "images"
    modalities = frozenset({"image"})
    output = "image.json"

    def parse(self, ctx, items):
        _parse_images(ctx, items)

    def chunk(self, ctx, src, f, count):
        _chunk_image(ctx, src, f, count)


class AudioVideoParser(Parser):
    """One parser for both, so Whisper and pyannote each load once for all recordings."""

    name = "audio_video"
    modalities = frozenset({"audio", "video"})

    def output_for(self, src):
        # parse = model outputs only; turns are built from them in the (cheap) chunk stage.
        # video: frames.json is written last, after diarization.json
        return "frames.json" if src.modality == "video" else "diarization.json"

    def parse(self, ctx, items):
        _parse_av(ctx, items)

    def chunk(self, ctx, src, f, count):
        _chunk_av(ctx, src, f, count)


for _parser in (DocumentParser(), ImageParser(), AudioVideoParser()):
    register_parser(_parser)
