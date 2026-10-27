"""Ingest queue (BUILD_GUIDE_v2 Milestone 13): requests from the web UI/API, one at a time.

`enqueue()` adds a row to `ingest_queue`; a worker (`tkb worker`, or the thread `tkb serve`
starts) claims the oldest queued row with `FOR UPDATE SKIP LOCKED` and runs the normal
pipeline on it, writing progress to the row so the UI can show it. Stages stay idempotent, so
an item interrupted by a crash is simply queued again when a worker starts.

GPU rule (AGENTS.md: never two large models at once): every ingest — worker or `tkb ingest` —
holds the Postgres advisory lock GPU_LOCK while it runs, so two never overlap.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select, text, update
from sqlalchemy.orm import Session

from thesiskb.core.models import QueueItem, Source

log = logging.getLogger(__name__)

GPU_LOCK = 7_231_001  # pg advisory lock key shared by every ingest
POLL_SECONDS = 2.0
PROGRESS_EVERY = 1.0  # seconds between progress writes


def _now() -> datetime:
    return datetime.now(UTC)


@contextmanager
def gpu_lock(engine, wait: bool = True) -> Iterator[bool]:
    """Hold the ingest lock on its own connection; yields False if busy and wait=False."""
    with engine.connect() as conn:
        if wait:  # blocks until free; returns void (no usable value)
            conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": GPU_LOCK})
            got = True
        else:
            got = bool(
                conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": GPU_LOCK}).scalar()
            )
        try:
            yield got
        finally:
            if got:
                conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": GPU_LOCK})


def enqueue(
    session: Session,
    kind: str,
    paths: list[str] | None = None,
    source_id=None,
    **options: Any,
) -> QueueItem:
    item = QueueItem(kind=kind, paths=paths or [], source_id=source_id, options=options)
    session.add(item)
    session.commit()
    return item


def claim(session: Session) -> QueueItem | None:
    """The oldest queued item, marked running; None if the queue is empty."""
    item = session.scalar(
        select(QueueItem)
        .where(QueueItem.status == "queued")
        .order_by(QueueItem.id)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    if item is None:
        session.rollback()
        return None
    item.status, item.started_at, item.error = "running", _now(), None
    session.commit()
    return item


def requeue_interrupted(session: Session) -> int:
    """Items left 'running' by a crashed worker go back to the queue (stages are idempotent)."""
    n = session.execute(
        update(QueueItem).where(QueueItem.status == "running").values(status="queued")
    ).rowcount
    session.commit()
    return n


def files_for(session: Session, item: QueueItem, cfg):
    """DetectedFiles for an item. Re-ingest: the source's file (or CSV row), forced."""
    from thesiskb.ingest.detect import detect
    from thesiskb.ingest.pipeline import supported_modalities

    modalities = supported_modalities(cfg)
    if item.kind == "reingest":
        src = session.get(Source, item.source_id)
        if src is None:
            raise ValueError("source no longer exists")
        file_path = Path(src.path.split("#", 1)[0])
        if not file_path.exists():
            raise FileNotFoundError(f"file moved or deleted: {file_path}")
        files = [f for f in detect([file_path]) if f.source_path == src.path]
        return files, files and files[0].sha256 == src.sha256  # unchanged content: force
    missing = [p for p in item.paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"not found: {', '.join(missing)}")
    return [f for f in detect([Path(p) for p in item.paths]) if f.modality in modalities], False


def run_item(engine, item_id: int, gateway=None) -> None:
    """Run one claimed item to the end, writing progress and the report to its row."""
    from thesiskb.core.config import get_pipeline_config
    from thesiskb.ingest.pipeline import STAGES, ingest

    if gateway is None:
        from thesiskb.llm import get_gateway

        gateway = get_gateway()
    cfg = get_pipeline_config()
    with Session(engine) as session, Session(engine) as status:
        item = session.get(QueueItem, item_id)
        opts = item.options or {}
        state = {"steps": 0, "last": 0.0}
        try:
            files, force = files_for(session, item, cfg)
            total = len(files) * (len(STAGES) - (0 if opts.get("extract", True) else 1))

            def progress(stage, name, done=None, of=None):
                if done is None:
                    state["steps"] += 1
                if time.monotonic() - state["last"] < PROGRESS_EVERY and done is not None:
                    return
                state["last"] = time.monotonic()
                p = {"stage": stage, "file": name, "steps": state["steps"], "total": total}
                if done is not None:
                    p |= {"done": done, "of": of}
                status.execute(update(QueueItem).where(QueueItem.id == item_id).values(progress=p))
                status.commit()

            report = ingest(
                session,
                files,
                cfg,
                gateway,
                source_type=opts.get("source_type"),
                force=force,
                extract=opts.get("extract", True),
                local_only=opts.get("local_only", False),
                progress=progress,
            )
            failed = report.pop("failed")
            item = session.get(QueueItem, item_id)
            item.report = dict(report) | {"files": len(files), "failed": failed}
            item.status = "failed" if failed else "done"
            item.error = "; ".join(f"{n}: {e}" for n, e in failed)[:2000] or None
        except Exception as e:  # noqa: BLE001 - recorded on the row and in the log
            log.exception("ingest queue item %s failed", item_id)
            session.rollback()
            item = session.get(QueueItem, item_id)
            item.status, item.error = "failed", f"{type(e).__name__}: {e}"
        item.finished_at = _now()
        session.commit()


def work(
    engine,
    stop: threading.Event | None = None,
    hold: Callable[[], Any] | None = None,
    once: bool = False,
) -> None:
    """Worker loop. `hold()` returns a context manager held while an item runs (the server
    passes its answer lock, so questions wait instead of loading a second model)."""
    stop = stop or threading.Event()
    with Session(engine) as session:
        if n := requeue_interrupted(session):
            log.info("re-queued %d interrupted ingest item(s)", n)
    while not stop.is_set():
        # lock first, then claim: an item never shows "running" while `tkb ingest` holds the GPU
        with gpu_lock(engine, wait=False) as free:
            item_id = None
            if free:
                with Session(engine) as session:
                    item = claim(session)
                    item_id = item.id if item else None
            if item_id is not None:
                log.info("ingest queue: running item %s", item_id)
                with hold() if hold else _null():
                    run_item(engine, item_id)
                continue
        if once:
            return
        stop.wait(POLL_SECONDS)


@contextmanager
def _null():
    yield


def start_worker_thread(engine, hold=None) -> threading.Event:
    stop = threading.Event()
    threading.Thread(
        target=work, args=(engine, stop, hold), name="ingest-worker", daemon=True
    ).start()
    return stop
