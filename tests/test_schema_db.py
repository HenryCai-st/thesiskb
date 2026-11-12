"""An outdated schema (git pull without `tkb db upgrade`) is reported, not a crash later."""

import pytest
from fastapi.testclient import TestClient

from tests.conftest import requires_db
from thesiskb import doctor
from thesiskb.doctor import Status

pytestmark = [pytest.mark.db, requires_db]


def test_schema_at_head_is_ok(db_url):
    assert doctor.schema_behind(db_url) is None
    assert doctor.check_db(db_url).status == Status.OK


def test_schema_behind_the_code_names_the_fix(db_url, monkeypatch):
    monkeypatch.setattr(doctor, "head_revision", lambda: "9999")  # code newer than the database
    problem = doctor.schema_behind(db_url)
    assert "needs 9999" in problem and "tkb db upgrade" in problem
    r = doctor.check_db(db_url)
    assert r.status == Status.FAIL and "tkb db upgrade" in r.detail


def test_server_refuses_to_start_on_an_old_schema(db_url, monkeypatch):
    from thesiskb.api import app as app_module

    monkeypatch.setattr(doctor, "head_revision", lambda: "9999")
    monkeypatch.setenv("TKB_NO_WORKER", "1")
    with pytest.raises(RuntimeError, match="tkb db upgrade"), TestClient(app_module.app):
        pass


def test_unreachable_database_is_left_to_other_checks():
    assert doctor.schema_behind("postgresql+psycopg://u:p@127.0.0.1:1/none") is None
