"""Gateway tests with LiteLLM mocked — no network, no GPU."""

from types import SimpleNamespace

import pytest

from thesiskb.llm import gateway as gw


@pytest.fixture
def gateway(models):
    return gw.LLMGateway(models, "http://ollama:11434")


def test_chat_uses_role_model_and_ollama_base(gateway, monkeypatch):
    seen = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        msg = SimpleNamespace(content="hello")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    monkeypatch.setattr(gw.litellm, "completion", fake_completion)
    out = gateway.chat([{"role": "user", "content": "hi"}], temperature=0.7)

    assert out == "hello"
    assert seen["model"] == "ollama_chat/qwen3:14b"
    assert seen["api_base"] == "http://ollama:11434"
    assert seen["temperature"] == 0.7  # call param overrides models.yaml


def test_embed_sorts_by_index_and_checks_dim(gateway, monkeypatch):
    def fake_embedding(**kwargs):
        assert kwargs["model"] == "ollama/bge-m3"
        return SimpleNamespace(
            data=[
                {"index": 1, "embedding": [1.0] * 1024},
                {"index": 0, "embedding": [0.0] * 1024},
            ]
        )

    monkeypatch.setattr(gw.litellm, "embedding", fake_embedding)
    vecs = gateway.embed(["a", "b"])
    assert vecs[0][0] == 0.0 and vecs[1][0] == 1.0


def test_embed_wrong_dim_raises(gateway, monkeypatch):
    monkeypatch.setattr(
        gw.litellm,
        "embedding",
        lambda **_: SimpleNamespace(data=[{"index": 0, "embedding": [0.0] * 3}]),
    )
    with pytest.raises(ValueError, match="dim"):
        gateway.embed(["a"])


def test_embed_empty(gateway):
    assert gateway.embed([]) == []


def test_rerank_cross_encoder_sorted(gateway):
    class FakeCE:
        def predict(self, pairs, batch_size=32):
            return [0.1, 0.9, 0.5]

    gateway._cross_encoders["BAAI/bge-reranker-v2-m3"] = FakeCE()
    out = gateway.rerank("q", ["a", "b", "c"], top_n=2)
    assert [r.index for r in out] == [1, 2]


def test_rerank_litellm_backend(gateway, monkeypatch):
    gateway.models.roles["reranker"].backend = "litellm"
    monkeypatch.setattr(
        gw.litellm,
        "rerank",
        lambda **_: SimpleNamespace(
            results=[{"index": 0, "relevance_score": 0.2}, {"index": 1, "relevance_score": 0.8}]
        ),
    )
    assert [r.index for r in gateway.rerank("q", ["a", "b"])] == [1, 0]
