"""Models tab: catalogue, writing the choice, API keys (Milestone 16). No models downloaded."""

import json
import logging
import shutil
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from thesiskb.core.config import REPO_ROOT, get_models_config, get_pipeline_config
from thesiskb.llm import catalog as cat

SECRET = "sk-ant-test-0123456789-do-not-leak"


@pytest.fixture
def cfg_dir(tmp_path, monkeypatch):
    """A copy of config/ in tmp, used as TKB_CONFIG_DIR; .env in a tmp working directory."""
    d = tmp_path / "config"
    shutil.copytree(REPO_ROOT / "config", d)
    monkeypatch.setenv("TKB_CONFIG_DIR", str(d))
    monkeypatch.setenv("TKB_MODELS_FILE", "models.yaml")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cat.reload_configs()
    yield d
    monkeypatch.undo()
    cat.reload_configs()


def test_catalogue_is_valid_and_online_only_where_a_local_fallback_exists(cfg_dir):
    c = cat.load_catalog()
    assert {"answer", "vision", "embedding", "reranker", "ocr", "asr", "diarization"} <= set(
        c.components
    )
    for key, comp in c.components.items():
        assert len({o.id for o in comp.options}) == len(comp.options)
        for o in comp.options:
            assert o.local or (key == "answer" and o.env_key), (key, o.id)
            assert o.local or o.licence
    # the current config is recognised
    assert cat.current_option(c.components["answer"]) == "qwen3-14b"
    assert cat.current_option(c.components["asr"]) == "whisper-large-v3-turbo"
    assert cat.current_option(c.components["ocr"]) == "deu-eng"
    assert "ANTHROPIC_API_KEY" in c.env_keys() and "TKB_HF_TOKEN" in c.env_keys()


def test_switching_speech_model_writes_models_yaml_and_keeps_comments(cfg_dir):
    c = cat.load_catalog()
    cat.apply_choice(c, "asr", "whisper-small")
    text = (cfg_dir / "models.yaml").read_text("utf-8")
    assert "# Not called through LiteLLM; used by ingest/audio.py." in text  # comments kept
    role = get_models_config().role("asr")  # caches were cleared: the new value is read
    assert (role.model, role.backend, role.vram_gb) == ("small", "faster_whisper", 1.5)
    assert role.model_extra["compute_type"] == "float16"  # other keys untouched
    assert cat.current_option(c.components["asr"]) == "whisper-small"


def test_local_answer_model_switches_all_llm_roles_online_only_answer(cfg_dir):
    c = cat.load_catalog()
    cat.apply_choice(c, "answer", "qwen3-8b")
    m = get_models_config()
    assert {m.role(r).model for r in ("answer", "judge", "router", "extraction")} == {
        "ollama_chat/qwen3:8b"
    }
    assert m.role("judge").params["reasoning_effort"] == "none"  # each role keeps its params

    cat.apply_choice(c, "answer", "claude-haiku")
    m = get_models_config()
    assert m.role("answer").model == "anthropic/claude-haiku-5-5"
    assert "num_ctx" not in m.role("answer").params  # online params replace Ollama's
    assert m.role("answer_local").model == "ollama_chat/qwen3:8b"  # local-only sources
    assert m.role("judge").model == "ollama_chat/qwen3:8b"
    from thesiskb.llm.gateway import LLMGateway

    gw = LLMGateway(m, "http://localhost:11434")
    assert gw.role_for("answer", local_only=True) == "answer_local"

    cat.apply_choice(c, "answer", "qwen3-14b")  # back: num_ctx again
    assert get_models_config().role("answer").params["num_ctx"] == 12288


def test_ocr_writes_pipeline_yaml(cfg_dir):
    c = cat.load_catalog()
    cat.apply_choice(c, "ocr", "eng")
    assert get_pipeline_config().model_extra["ocr"]["languages"] == ["eng"]
    assert "# Born-digital PDFs don't need OCR" in (cfg_dir / "pipeline.yaml").read_text("utf-8")


def test_set_key_writes_env_and_replaces_the_old_line(cfg_dir, tmp_path):
    (tmp_path / ".env").write_text(
        "TKB_OLLAMA_BASE_URL=http://localhost:11434\n# ANTHROPIC_API_KEY=old\n", encoding="utf-8"
    )
    c = cat.load_catalog()
    cat.set_key(c, "ANTHROPIC_API_KEY", SECRET)
    cat.set_key(c, "ANTHROPIC_API_KEY", SECRET + "2")
    lines = (tmp_path / ".env").read_text("utf-8").splitlines()
    assert lines == ["TKB_OLLAMA_BASE_URL=http://localhost:11434", f"ANTHROPIC_API_KEY={SECRET}2"]
    assert cat.key_status(c)["ANTHROPIC_API_KEY"] is True
    with pytest.raises(ValueError):
        cat.set_key(c, "PATH", "x")  # only keys the catalogue names
    with pytest.raises(ValueError):
        cat.set_key(c, "ANTHROPIC_API_KEY", "a\nTKB_DATABASE_URL=evil")


def test_keys_api_never_returns_or_logs_the_key(cfg_dir, tmp_path, monkeypatch, caplog):
    from thesiskb.api import app as app_module

    monkeypatch.setattr(cat, "ollama_models", lambda: set())
    monkeypatch.setattr(cat, "tesseract_langs", lambda: {"deu", "eng"})
    caplog.set_level(logging.DEBUG)
    remote = TestClient(app_module.app)  # client host "testclient": not this machine
    r = remote.post("/api/keys", json={"name": "ANTHROPIC_API_KEY", "value": SECRET})
    assert r.status_code == 403 and SECRET not in r.text
    assert remote.get("/api/models").json()["keys_editable"] is False

    local = TestClient(app_module.app, client=("127.0.0.1", 50000))
    bad = local.post("/api/keys", json={"value": SECRET})  # no name: no echo of the body
    assert bad.status_code == 400 and SECRET not in bad.text
    ok = local.post("/api/keys", json={"name": "ANTHROPIC_API_KEY", "value": SECRET})
    assert ok.json() == {"name": "ANTHROPIC_API_KEY", "set": True}
    m = local.get("/api/models")
    assert m.json()["keys"]["ANTHROPIC_API_KEY"] is True and SECRET not in m.text
    claude = next(
        o for c in m.json()["components"] if c["key"] == "answer" for o in c["options"]
        if o["id"] == "claude-haiku"
    )  # fmt: skip
    assert claude["installed"] is True  # online model usable once its key is set
    assert SECRET not in caplog.text

    monkeypatch.setenv("TKB_BIND_HOST", "0.0.0.0")  # server reachable from the network
    assert (
        local.post("/api/keys", json={"name": "ANTHROPIC_API_KEY", "value": "x"}).status_code == 403
    )


def test_ollama_pull_progress_sums_layers(monkeypatch):
    lines = [
        {"status": "pulling manifest"},
        {"digest": "a", "total": 100, "completed": 50},
        {"digest": "b", "total": 300, "completed": 0},
        {"digest": "a", "total": 100, "completed": 100},
        {"digest": "b", "total": 300, "completed": 300},
        {"status": "success"},
    ]

    class Resp:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            return (json.dumps(x) for x in lines)

    @contextmanager
    def fake_stream(method, url, json=None, timeout=None):
        assert url.endswith("/api/pull") and json == {"model": "qwen3:8b", "stream": True}
        yield Resp()

    monkeypatch.setattr(cat.httpx, "stream", fake_stream)
    seen = []
    cat.pull_ollama("qwen3:8b", lambda d, t: seen.append((d, t)))
    assert seen == [(50, 100), (50, 400), (100, 400), (400, 400)]
