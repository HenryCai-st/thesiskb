"""Migration test against a real Postgres+pgvector. Skipped unless the DB is reachable."""

import os
import uuid

import pytest
from sqlalchemy import create_engine, inspect, text

URL = os.environ.get("TKB_DATABASE_URL")

pytestmark = pytest.mark.db


def _reachable(url: str | None) -> bool:
    if not url:
        return False
    try:
        with create_engine(url, connect_args={"connect_timeout": 2}).connect():
            return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _reachable(URL), reason="TKB_DATABASE_URL not set or unreachable")
def test_migrations_create_schema_and_roundtrip():
    from alembic import command
    from alembic.config import Config

    from thesiskb.core.config import REPO_ROOT

    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "src/thesiskb/migrations"))
    cfg.set_main_option("sqlalchemy.url", URL)
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")

    engine = create_engine(URL)
    tables = set(inspect(engine).get_table_names())
    expected = {
        "sources",
        "segments",
        "chunks",
        "people",
        "speakers",
        "entities",
        "relations",
        "tasks",
        "jobs",
        "eval_runs",
    }
    assert expected <= tables

    sid = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO sources (id, sha256, path, modality) VALUES (:id, :h, 'x', 'pdf')"),
            {"id": sid, "h": uuid.uuid4().hex * 2},
        )
        conn.execute(
            text(
                "INSERT INTO chunks (id, source_id, text, embedding, tsv, locator) "
                "VALUES (:id, :sid, 'hello world', :emb, to_tsvector('simple', 'hello world'), "
                "'{\"page\": 1}')"
            ),
            {"id": uuid.uuid4(), "sid": sid, "emb": "[" + ",".join(["0.1"] * 1024) + "]"},
        )
        hit = conn.execute(
            text("SELECT count(*) FROM chunks WHERE tsv @@ plainto_tsquery('simple', 'hello')")
        ).scalar()
        assert hit == 1
        conn.execute(text("DELETE FROM sources WHERE id = :id"), {"id": sid})
