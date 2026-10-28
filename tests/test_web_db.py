"""Web sources on the test database: provenance, routing, queue, forget-person (no network)."""

import os
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session

from tests.conftest import requires_db
from tests.test_web import PAGE, fetcher
from thesiskb.core.config import PipelineConfig
from thesiskb.core.models import Person, QueueItem, Source
from thesiskb.ingest import web

pytestmark = [pytest.mark.db, requires_db]


@pytest.fixture
def engine(db_url):
    eng = create_engine(db_url)

    def clean():
        with Session(eng) as s:
            s.execute(delete(Source).where(Source.path.like("%m17test%")))
            s.execute(delete(Person).where(Person.name == "Ada M17test"))
            s.execute(delete(QueueItem))
            s.commit()

    clean()
    yield eng
    clean()
    eng.dispose()


def test_ingest_web_records_url_time_person_and_skips_summaries(engine, tmp_path, monkeypatch):
    from thesiskb.ingest import pipeline
    from thesiskb.ingest.detect import register

    calls = {}

    def fake_ingest(session, files, cfg, gateway, **kw):  # the real pipeline needs Docling/GPU
        calls.update(kw)
        for f in files:
            register(session, f, kw.get("source_type"), False)
        session.commit()
        return {"parse": len(files), "failed": []}

    monkeypatch.setattr(pipeline, "ingest", fake_ingest)
    monkeypatch.setattr(web, "render", lambda url: pytest.fail("not needed"))
    big = b"\x89PNG" + b"0" * 20_000
    f = fetcher({
        "https://m17test.example/projects/sideby": (PAGE, "text/html"),
        "https://m17test.example/images/sideby/cover.png": (big, "image/png"),
    })  # fmt: skip
    cfg = PipelineConfig(raw_dir=tmp_path / "raw")
    with Session(engine) as s:
        ada = Person(name="Ada M17test")
        s.add(ada)
        s.commit()
        report, sources, failed = web.ingest_web(
            s, ["https://m17test.example/projects/sideby", "https://m17test.example/missing"],
            cfg, None, fetcher=f, summarize=False, media=False, person_id=ada.id,
        )  # fmt: skip
        assert calls["summarize"] is False and calls["source_type"] == "web"
        assert [u for u, _ in failed] == ["https://m17test.example/missing"]  # HTTP 404
        by_name = {src.path.rsplit("/", 1)[-1]: src for src in sources}
        page, image = by_name["main.html"], by_name["img-01.png"]
        assert page.origin == "https://m17test.example/projects/sideby"
        assert image.origin == "https://m17test.example/images/sideby/cover.png"
        assert isinstance(page.retrieved_at, datetime)
        assert {src.person_id for src in sources} == {ada.id}
        assert page.verification_status == "unverified" and page.source_type == "web"
        assert image.title.endswith("img-01.png")


def test_explicit_web_question_routes_to_web_only(engine):
    from thesiskb.retrieval.router import route_question

    with Session(engine) as s:
        s.add(Person(name="Ada M17test"))
        s.commit()
        r = route_question(s, None, "Search the web for Ada M17test's publications")
        assert r.paths == ["web"] and r.person_id  # results will be linked to Ada
        r = route_question(s, None, "What did Ada M17test say about the web?", use_llm=False)
        assert "web" not in r.paths


def test_api_queues_urls_as_web_items(engine):
    from thesiskb.api import app as app_module

    def session_override():
        with Session(engine) as s:
            yield s
            s.commit()

    app_module.app.dependency_overrides[app_module.get_session] = session_override
    try:
        c = TestClient(app_module.app)
        r = c.post("/api/ingest", json={"paths": ["https://m17test.example/"], "site": True})
        job = c.get("/api/jobs").json()[0]
        assert job["id"] == r.json()["queued"] and job["kind"] == "url"
        assert job["options"]["site"] is True and job["options"]["max_pages"] == 50
        mixed = c.post("/api/ingest", json={"paths": ["https://m17test.example/", "data/raw"]})
        assert mixed.status_code == 400
    finally:
        app_module.app.dependency_overrides.clear()


def test_forget_person_deletes_web_lookups_about_them(engine, tmp_path, monkeypatch):
    from contextlib import contextmanager

    from sqlalchemy.orm import sessionmaker
    from typer.testing import CliRunner

    from thesiskb import cli

    maker = sessionmaker(bind=engine)

    @contextmanager
    def scope():
        s = maker()
        try:
            yield s
            s.commit()
        finally:
            s.close()

    monkeypatch.setattr("thesiskb.core.db.session_scope", scope)
    monkeypatch.setattr("thesiskb.llm.get_gateway", lambda: None)
    snap = tmp_path / "main.html"
    snap.write_text("<p>Ada M17test works at …</p>", encoding="utf-8")
    with Session(engine) as s:
        ada = Person(name="Ada M17test")
        s.add(ada)
        s.flush()
        s.add(Source(sha256=os.urandom(32).hex(), path=f"{snap.as_posix()}#m17test",
                     modality="document", source_type="web", person_id=ada.id))  # fmt: skip
        other = Source(sha256=os.urandom(32).hex(), path="/x/m17test-other.html",
                       modality="document", source_type="web")  # fmt: skip
        s.add(other)
        s.commit()
        other_id = other.id
    snap_path = f"{snap.as_posix()}#m17test"
    out = CliRunner().invoke(cli.app, ["forget-person", "Ada M17test", "--yes"])
    assert out.exit_code == 0, out.output
    assert "1 web page(s)" in out.output
    with Session(engine) as s:
        assert s.scalar(select(Source).where(Source.path == snap_path)) is None
        assert s.get(Source, other_id) is not None  # not about Ada: kept
