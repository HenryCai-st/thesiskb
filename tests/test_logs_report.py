"""Log file (JSON lines with source/stage context) and the secret-free bug report zip."""

import json
import logging
import zipfile

import pytest

from thesiskb.core import logs
from thesiskb.core.config import get_settings


@pytest.fixture
def clean_root():
    """Start without handlers (other tests may have set up logging) and restore afterwards."""
    root = logging.getLogger()
    before, level = list(root.handlers), root.level
    root.handlers = []
    yield
    for h in root.handlers:
        h.close()
    root.handlers, root.level = before, level


def test_log_file_is_json_with_context(tmp_path, clean_root):
    path = logs.setup_logging(tmp_path)
    assert logs.setup_logging(tmp_path) == path  # idempotent: one file handler
    try:
        raise ValueError("bad epub")
    except ValueError:
        logging.getLogger("thesiskb.ingest").exception(
            "parse failed for x.epub", extra={"sha256": "ab12", "stage": "parse"}
        )
    for h in logging.getLogger().handlers:
        h.flush()
    rec = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["level"] == "ERROR" and rec["sha256"] == "ab12" and rec["stage"] == "parse"
    assert "ValueError: bad epub" in rec["traceback"]


def test_report_has_no_secrets(tmp_path, monkeypatch, clean_root):
    from thesiskb import report

    monkeypatch.setenv("TKB_HF_TOKEN", "hf_SuperSecretToken123")
    monkeypatch.setenv("TKB_API_TOKEN", "api-token-456")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-789")
    monkeypatch.setenv("TKB_DATABASE_URL", "postgresql+psycopg://u:dbPassw0rd@localhost:1/x")
    get_settings.cache_clear()
    monkeypatch.setattr(logs, "log_dir", lambda: tmp_path)
    monkeypatch.setattr(report, "_environment", lambda: {"db": get_settings().database_url})
    (tmp_path / "thesiskb.log").write_text(
        "connect postgresql://u:dbPassw0rd@host failed\nkey=sk-ant-secret-789\n", encoding="utf-8"
    )
    try:
        out = report.build_report(tmp_path / "r.zip", trace={"q": "token hf_SuperSecretToken123"})
    finally:
        get_settings.cache_clear()
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        blob = "".join(z.read(n).decode("utf-8") for n in names)
    assert {"environment.json", "thesiskb.log", "trace.json"} <= set(names)
    assert not any("people" in n or n.endswith(".env") for n in names)
    for secret in ("hf_SuperSecretToken123", "api-token-456", "sk-ant-secret-789", "dbPassw0rd"):
        assert secret not in blob
    assert "***" in blob


def test_eval_summary_counts_errors_separately():
    from thesiskb.evaluation import QuestionResult, summarize

    results = [
        QuestionResult(id="a", type="general", status="ran", retrieval_hit=True, citation_hit=True),
        QuestionResult(id="b", type="general", status="skipped", reason="not ingested"),
        QuestionResult(id="c", type="general", status="error", reason="error: InternalServerError"),
    ]
    s = summarize(results)
    assert (s["ran"], s["skipped"], s["errors"]) == (1, 1, 1)
    assert s["retrieval_hit_rate"] == 1.0
