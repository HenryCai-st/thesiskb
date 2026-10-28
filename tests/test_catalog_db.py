"""Models tab "Use" through the queue on the test database (downloads faked)."""

import os
import shutil

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session

from tests.conftest import requires_db
from thesiskb.core.config import REPO_ROOT, get_models_config
from thesiskb.core.models import Chunk, Entity, QueueItem, Source
from thesiskb.ingest import queue
from thesiskb.llm import catalog as cat

pytestmark = [pytest.mark.db, requires_db]


@pytest.fixture
def env(db_url, tmp_path, monkeypatch):
    d = tmp_path / "config"
    shutil.copytree(REPO_ROOT / "config", d)
    pipe = d / "pipeline.yaml"  # no merging (it would snapshot into the real data/ folder)
    pipe.write_text(
        pipe.read_text("utf-8").replace("auto_merge: true", "auto_merge: false"), "utf-8"
    )
    monkeypatch.setenv("TKB_CONFIG_DIR", str(d))
    monkeypatch.setenv("TKB_MODELS_FILE", "models.yaml")
    monkeypatch.setenv("TKB_DATABASE_URL", db_url)
    cat.reload_configs()
    downloads = []
    monkeypatch.setattr(
        cat, "download", lambda o, progress: (downloads.append(o.id), progress(5, 10))
    )
    eng = create_engine(db_url)
    with Session(eng) as s:
        s.execute(delete(QueueItem))
        s.commit()
    yield eng, d, downloads
    with Session(eng) as s:
        s.execute(delete(Source).where(Source.path.like("%m16test%")))
        s.commit()
    eng.dispose()
    monkeypatch.undo()
    cat.reload_configs()


def _client(engine):
    from thesiskb.api import app as app_module

    def session_override():
        with Session(engine) as s:
            yield s
            s.commit()

    app_module.app.dependency_overrides[app_module.get_session] = session_override
    return app_module, TestClient(app_module.app)


def test_use_speech_model_queues_downloads_and_switches(env):
    engine, cfg_dir, downloads = env
    app_module, c = _client(engine)
    try:
        r = c.post("/api/models/use", json={"component": "asr", "option": "whisper-medium"})
        item_id = r.json()["queued"]
        assert get_models_config().role("asr").model == "large-v3-turbo"  # not before it runs
        with Session(engine) as s:
            assert queue.claim(s).id == item_id
        queue.run_item(engine, item_id)
        job = next(j for j in c.get("/api/jobs").json() if j["id"] == item_id)
        assert job["status"] == "done" and job["options"]["option"] == "whisper-medium"
        assert downloads == ["whisper-medium"]
        # what the next audio ingest reads (pipeline: get_models_config().role("asr"))
        assert get_models_config().role("asr").model == "medium"
        assert "model: medium" in (cfg_dir / "models.yaml").read_text("utf-8")

        assert (
            c.post("/api/models/use", json={"component": "asr", "option": "nope"}).status_code
            == 404
        )
        online = c.post("/api/models/use", json={"component": "answer", "option": "claude-haiku"})
        if not os.environ.get("ANTHROPIC_API_KEY"):
            assert online.status_code == 400 and "ANTHROPIC_API_KEY" in online.text
    finally:
        app_module.app.dependency_overrides.clear()


def test_embedding_change_needs_confirmation_and_reembeds(env, monkeypatch):
    engine, _, _ = env
    with Session(engine) as s:
        src = Source(sha256=os.urandom(32).hex(), path="/x/m16test.pdf", modality="pdf")
        s.add(src)
        s.flush()
        s.add(Chunk(source_id=src.id, text="hello", locator={"page": 1}, embedding=[0.0] * 1024))
        s.commit()
        sid = src.id

    class Gw:
        def role_for(self, role, local_only):
            return role

        def embed(self, texts, role="embedding"):
            assert get_models_config().role(role).model == "ollama/snowflake-arctic-embed2"
            return [[1.0] + [0.0] * 1023 for _ in texts]

    monkeypatch.setattr("thesiskb.llm.get_gateway", lambda: Gw())
    app_module, c = _client(engine)
    try:
        body = {"component": "embedding", "option": "arctic-embed2"}
        refused = c.post("/api/models/use", json=body)
        assert refused.status_code == 409 and "re-embeds all" in refused.json()["detail"]
        item_id = c.post("/api/models/use", json=body | {"confirm_reembed": True}).json()["queued"]
        with Session(engine) as s:
            queue.claim(s)
        queue.run_item(engine, item_id, gateway=Gw())
        with Session(engine) as s:
            item = s.get(QueueItem, item_id)
            assert item.status == "done", item.error
            assert item.report["re_embedded"] >= 1
            v = s.scalar(select(Chunk.embedding).where(Chunk.source_id == sid))
            assert float(v[0]) == 1.0  # new vectors
            # entity names get vectors from the new model at the next ingest
            assert s.scalar(select(Entity.id).where(Entity.embedding.is_not(None))) is None
    finally:
        app_module.app.dependency_overrides.clear()
