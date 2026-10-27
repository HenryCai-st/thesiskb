"""Ingest queue and changed-file replacement on a real Postgres (pipeline faked)."""

import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session

from tests.conftest import requires_db
from thesiskb.core.config import PipelineConfig
from thesiskb.core.models import Person, QueueItem, Source, Speaker, Task
from thesiskb.ingest import pipeline, queue

pytestmark = [pytest.mark.db, requires_db]


@pytest.fixture
def engine(db_url):
    eng = create_engine(db_url)
    with Session(eng) as s:
        s.execute(delete(QueueItem))
        s.execute(delete(Source).where(Source.path.like("%queue-test%")))
        s.commit()
    yield eng
    eng.dispose()


def test_claim_is_fifo_and_skips_locked_rows(engine):
    with Session(engine) as s:
        a = queue.enqueue(s, "ingest", ["a"]).id
        b = queue.enqueue(s, "ingest", ["b"]).id
    with Session(engine) as s1, Session(engine) as s2:
        # s1 holds a row lock on "a" (not committed yet): s2 must get "b" instead of waiting
        locked = s1.scalar(
            select(QueueItem).where(QueueItem.id == a).with_for_update(skip_locked=True)
        )
        assert locked.id == a
        assert queue.claim(s2).id == b
        s1.rollback()
    with Session(engine) as s:
        assert queue.claim(s).id == a
        assert queue.claim(s) is None


def test_interrupted_items_are_requeued(engine):
    with Session(engine) as s:
        item = queue.enqueue(s, "ingest", ["x"])
        queue.claim(s)
        assert queue.requeue_interrupted(s) == 1
        s.refresh(item)
        assert item.status == "queued"


def test_run_item_records_progress_report_and_errors(engine, monkeypatch, tmp_path):
    calls = []

    def fake_ingest(session, files, cfg, gateway, progress=None, **kw):
        calls.append(kw)
        progress("parse", "a.pdf")
        return {"parse": 1, "chunk": 1, "summarize": 0, "embed": 1, "extract": 0, "failed": []}

    monkeypatch.setattr(pipeline, "ingest", fake_ingest)
    monkeypatch.setattr(queue, "files_for", lambda s, item, cfg: ([object()], False))
    with Session(engine) as s:
        ok = queue.enqueue(s, "ingest", [str(tmp_path)], extract=False).id
        queue.claim(s)
    queue.run_item(engine, ok, gateway=object())
    with Session(engine) as s:
        item = s.get(QueueItem, ok)
        assert item.status == "done" and item.report["parse"] == 1 and item.report["files"] == 1
        assert item.progress["stage"] == "parse" and calls[0]["extract"] is False

    def missing(s, item, cfg):
        raise FileNotFoundError("not found: /nope")

    monkeypatch.setattr(queue, "files_for", missing)
    with Session(engine) as s:
        bad = queue.enqueue(s, "ingest", ["/nope"]).id
        queue.claim(s)
    queue.run_item(engine, bad, gateway=object())
    with Session(engine) as s:
        item = s.get(QueueItem, bad)
        assert item.status == "failed" and "not found" in item.error and item.finished_at


def test_gpu_lock_is_exclusive(engine):
    with queue.gpu_lock(engine) as first:
        assert first
        with queue.gpu_lock(engine, wait=False) as second:
            assert second is False
    with queue.gpu_lock(engine, wait=False) as again:
        assert again


def test_changed_file_replaces_old_version_and_keeps_confirmed_work(engine, tmp_path):
    path = (tmp_path / "queue-test.pdf").as_posix()
    with Session(engine) as s:
        person = Person(name=f"Queue Tester {uuid.uuid4().hex[:6]}")
        old = Source(sha256=os.urandom(32).hex(), path=path, modality="pdf", source_type="note")
        s.add_all([person, old])
        s.flush()
        s.add_all(
            [
                Task(description="Send the draft.", source_id=old.id, confirmed=True),
                Task(description="Book a room", source_id=old.id, status="done"),
                Task(description="Unreviewed idea", source_id=old.id),
                Speaker(source_id=old.id, diarization_label="SPEAKER_00", person_id=person.id),
            ]
        )
        new = Source(sha256=os.urandom(32).hex(), path=path, modality="pdf", replaces=old.id)
        s.add(new)
        s.flush()
        s.add_all(
            [
                Task(description="send the draft", source_id=new.id),
                Speaker(source_id=new.id, diarization_label="SPEAKER_00"),
            ]
        )
        s.commit()
        assert pipeline.retire_replaced(s, new, PipelineConfig(processed_dir=tmp_path)) == 1
        assert s.get(Source, old.id) is None
        tasks = {t.description: t for t in s.scalars(select(Task).where(Task.source_id == new.id))}
        assert tasks["send the draft"].confirmed  # matched: confirmation carried over
        assert tasks["Book a room"].status == "done"  # not in the new version: moved, kept
        assert "Unreviewed idea" not in tasks  # unreviewed extraction is redone instead
        sp = s.scalar(select(Speaker).where(Speaker.source_id == new.id))
        assert sp.person_id == person.id
        s.delete(person)
        s.delete(new)
        s.commit()


def test_register_links_a_changed_file_to_its_old_version(engine, tmp_path):
    from thesiskb.ingest.detect import DetectedFile, register

    f = tmp_path / "queue-test.pdf"
    with Session(engine) as s:
        v1 = register(s, DetectedFile(path=f, sha256=os.urandom(32).hex(), modality="pdf"))
        s.commit()
        v2 = register(s, DetectedFile(path=f, sha256=os.urandom(32).hex(), modality="pdf"))
        s.commit()
        assert v2.replaces == v1.id and v2.path == v1.path


def test_api_queues_ingest_and_reingest(engine, tmp_path):
    from thesiskb.api import app as app_module

    def session_override():
        with Session(engine) as s:
            yield s
            s.commit()

    app_module.app.dependency_overrides[app_module.get_session] = session_override
    try:
        client = TestClient(app_module.app)
        missing = client.post("/api/ingest", json={"paths": ["/definitely/missing"]})
        assert missing.status_code == 400
        r = client.post("/api/ingest", json={"paths": [str(tmp_path)]}).json()
        jobs = client.get("/api/jobs").json()
        assert jobs[0]["id"] == r["queued"] and jobs[0]["status"] == "queued"
        with Session(engine) as s:
            src = Source(
                sha256=os.urandom(32).hex(), path=f"{tmp_path}/queue-test.pdf", modality="pdf"
            )
            s.add(src)
            s.commit()
            sid = src.id
        client.post(f"/api/sources/{sid}/reingest")
        assert client.get("/api/jobs").json()[0]["kind"] == "reingest"
        assert client.post(f"/api/sources/{uuid.uuid4()}/reingest").status_code == 404
    finally:
        app_module.app.dependency_overrides.clear()
