"""Doctor checks with subprocess/HTTP mocked."""

import httpx
import pytest
from typer.testing import CliRunner

from thesiskb import doctor
from thesiskb.doctor import CheckResult, Status


def test_gpu_parses_nvidia_smi(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(doctor, "_run", lambda *a, **k: "RTX Test, 16376, 1024\n")
    r = doctor.check_gpu()
    assert r.status == Status.OK
    assert "RTX Test" in r.detail and "16.0 GB" in r.detail


def test_gpu_missing(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _: None)
    assert doctor.check_gpu().status == Status.FAIL


def test_ollama_reports_missing_models(monkeypatch, models):
    class Resp:
        def json(self):
            return {"models": [{"name": "bge-m3:latest"}]}

    monkeypatch.setattr(doctor.httpx, "get", lambda *a, **k: Resp())
    r = doctor.check_ollama("http://x", models)
    assert r.status == Status.WARN
    assert "qwen3:14b" in r.detail and "bge-m3" not in r.detail


def test_ollama_unreachable(monkeypatch, models):
    def boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(doctor.httpx, "get", boom)
    assert doctor.check_ollama("http://x", models).status == Status.FAIL


def test_db_unreachable():
    r = doctor.check_db("postgresql+psycopg://u:p@127.0.0.1:1/none")
    assert r.status == Status.FAIL


def test_tesseract_missing_language(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _: "tesseract")
    monkeypatch.setattr(
        doctor, "_run", lambda *a, **k: 'List of available languages in "x" (2):\neng\nosd\n'
    )
    r = doctor.check_tesseract(["deu", "eng"])
    assert r.status == Status.WARN and "deu" in r.detail


@pytest.mark.parametrize("fail,code", [(False, 0), (True, 1)])
def test_cli_doctor_exit_code(monkeypatch, fail, code):
    from thesiskb import cli

    results = [CheckResult(name="X", status=Status.FAIL if fail else Status.OK, detail="d")]
    monkeypatch.setattr(doctor, "run_checks", lambda *a, **k: results)
    monkeypatch.setattr(doctor, "query_gpus", lambda: [])
    out = CliRunner().invoke(cli.app, ["doctor"])
    assert out.exit_code == code, out.output
    assert "answer" in out.output
