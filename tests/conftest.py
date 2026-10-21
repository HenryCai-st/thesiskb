import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from thesiskb.core.config import REPO_ROOT, get_models_config

# DB tests run against a separate database whose name must end in "_test": the migration test
# drops every table. Default: <your TKB_DATABASE_URL database>_test on the same server.
_base = os.environ.get("TKB_TEST_DATABASE_URL") or os.environ.get("TKB_DATABASE_URL")
TEST_DB_URL: str | None = None
if _base:
    url = make_url(_base)
    if not (url.database or "").endswith("_test"):
        url = url.set(database=f"{url.database}_test")
    TEST_DB_URL = url.render_as_string(hide_password=False)


def _ensure_test_db() -> bool:
    if not TEST_DB_URL:
        return False
    url = make_url(TEST_DB_URL)
    try:
        admin = create_engine(
            url.set(database="postgres"),
            isolation_level="AUTOCOMMIT",
            connect_args={"connect_timeout": 2},
        )
        with admin.connect() as conn:
            exists = conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": url.database}
            )
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{url.database}"'))
        admin.dispose()
        return True
    except Exception:  # noqa: BLE001 — no server: DB tests are skipped
        return False


DB_AVAILABLE = _ensure_test_db()
requires_db = pytest.mark.skipif(not DB_AVAILABLE, reason="no Postgres reachable for tests")


@pytest.fixture(scope="session")
def db_url() -> str:
    """Test database URL with the schema at head."""
    assert TEST_DB_URL and make_url(TEST_DB_URL).database.endswith("_test")
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "src/thesiskb/migrations"))
    cfg.set_main_option("sqlalchemy.url", TEST_DB_URL)
    command.upgrade(cfg, "head")
    return TEST_DB_URL


@pytest.fixture
def config_dir() -> Path:
    return REPO_ROOT / "config"


@pytest.fixture
def models(config_dir):
    get_models_config.cache_clear()
    return get_models_config(config_dir)
