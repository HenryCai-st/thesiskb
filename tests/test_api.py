"""REST/OpenAI/MCP surfaces with the answer pipeline faked (no models, no GPU)."""

import asyncio
import json
import uuid

import pytest
from fastapi.testclient import TestClient

from thesiskb.answer.generate import Answer, Citation, Pinpoint, Sentence
from thesiskb.api import app as app_module
from thesiskb.core.config import get_settings


def _fake_answer(*args, **kwargs) -> Answer:
    cite = Citation(
        n=1,
        chunk_id=uuid.uuid4(),
        filename="talk.mp3",
        title="Talk",
        context_header="Meeting: Talk",
        locator={"t_start": 20, "t_end": 30},
        verification_status="unverified",
        text="[0:22] Will.i.am: I got a Spark.",
    )
    return Answer(
        question=args[2] if len(args) > 2 else "q",
        text="He got a Spark [S1]. He invented time travel [S1].",
        sentences=[
            Sentence(
                text="He got a Spark [S1].",
                citations=[1],
                status="supported",
                pinpoints=[Pinpoint(n=1, t=22, speaker="Will.i.am", line="[0:22] ...")],
            ),
            Sentence(text="He invented time travel [S1].", citations=[1], status="unsupported"),
        ],
        context=[cite],
        route={"paths": ["person"], "person": "Will.i.am", "by": "rules"},
        trace_id="t-123",
    )


@pytest.fixture
def client(monkeypatch):
    import thesiskb.answer.paths as paths

    monkeypatch.setattr(paths, "answer", _fake_answer)
    monkeypatch.setattr(
        app_module,
        "citation_links",
        lambda session, cid: {"source_id": "src-1", "open_url": "/files/src-1#t=20"},
    )
    return TestClient(app_module.app)


def test_health(client):
    assert client.get("/api/health").json()["ok"] is True


def test_ask_adds_links_and_pinpoint_urls(client):
    d = client.post("/api/ask", json={"question": "What did Will say?"}).json()
    assert d["context"][0]["links"]["open_url"] == "/files/src-1#t=20"
    assert d["sentences"][0]["pinpoints"][0]["open_url"] == "/files/src-1#t=22"


def test_openai_chat_completion_markdown(client):
    body = {
        "model": "thesiskb",
        "messages": [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "What did Will say?"},
        ],
    }
    d = client.post("/v1/chat/completions", json=body).json()
    content = d["choices"][0]["message"]["content"]
    assert d["object"] == "chat.completion" and d["model"] == "thesiskb"
    # absolute links, so chat clients on another host (Open WebUI) can open them
    assert "**Sources**" in content and "(http://testserver/files/src-1#t=20)" in content
    assert "⚠ Not supported" in content and "time travel" in content
    assert "(http://testserver/?trace=t-123)" in content
    assert d["thesiskb"]["trace_id"] == "t-123"


def test_public_url_overrides_link_base(client, monkeypatch):
    monkeypatch.setenv("TKB_PUBLIC_URL", "http://kb.lan:8000/")
    get_settings.cache_clear()
    try:
        body = {"messages": [{"role": "user", "content": "q?"}]}
        content = client.post("/v1/chat/completions", json=body).json()["choices"][0]["message"]
        assert "(http://kb.lan:8000/files/src-1#t=20)" in content["content"]
    finally:
        monkeypatch.delenv("TKB_PUBLIC_URL")
        get_settings.cache_clear()


def test_openai_streaming(client):
    body = {
        "model": "thesiskb",
        "stream": True,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "q?"}]}],
    }
    with client.stream("POST", "/v1/chat/completions", json=body) as r:
        lines = [ln for ln in r.iter_lines() if ln]
    assert lines[-1] == "data: [DONE]"
    first = json.loads(lines[0].removeprefix("data: "))
    assert first["choices"][0]["delta"]["role"] == "assistant"


def test_models_and_missing_question(client):
    assert client.get("/v1/models").json()["data"][0]["id"] == "thesiskb"
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400


def test_token_required_when_configured(client, monkeypatch):
    monkeypatch.setenv("TKB_API_TOKEN", "s3cret")
    get_settings.cache_clear()
    try:
        assert client.get("/v1/models").status_code == 401
        ok = client.get("/v1/models", headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200
        assert client.get(f"/files/{uuid.uuid4()}").status_code == 401  # media need ?token=
    finally:
        monkeypatch.delenv("TKB_API_TOKEN")
        get_settings.cache_clear()


def test_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "<title>ThesisKB</title>" in r.text


def test_mcp_tools_listed():
    from thesiskb.mcp.server import server

    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"search", "get_chunk", "get_source", "list_tasks", "person_profile", "ask"} <= names
