"""Migration test against a real Postgres+pgvector (the *_test database only)."""

import uuid

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from tests.conftest import TEST_DB_URL, requires_db

pytestmark = [pytest.mark.db, requires_db]


def test_migrations_create_schema_and_roundtrip(db_url):
    from alembic import command
    from alembic.config import Config

    from thesiskb.core.config import REPO_ROOT

    assert make_url(TEST_DB_URL).database.endswith("_test")  # never wipe a real KB
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "src/thesiskb/migrations"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")

    engine = create_engine(db_url)
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
    engine.dispose()
