"""LiteLLM gateway: chat(), embed(), rerank() with the model chosen per role from models.yaml."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import litellm
from pydantic import BaseModel

from thesiskb.core.config import ModelsConfig, RoleConfig, get_models_config, get_settings

Message = dict[str, Any]

litellm.suppress_debug_info = True  # it prints a "Provider List" banner to stdout otherwise


class RerankResult(BaseModel):
    index: int
    score: float


class LocalOnlyError(RuntimeError):
    """Content of a local_only source would have gone to a non-local model."""


class LLMGateway:
    def __init__(self, models: ModelsConfig, ollama_base_url: str) -> None:
        self.models = models
        self.ollama_base_url = ollama_base_url
        self._cross_encoders: dict[str, Any] = {}

    def role_for(self, role: str, local_only: bool) -> str:
        """The role to use for content that may include local_only sources (BUILD_GUIDE §10).

        In API mode `answer` may be a cloud model; content from a local_only source must then
        go to `<role>_local` (e.g. `answer_local` on Ollama). Without one, refuse rather than
        send it."""
        if not local_only or self.models.role(role).is_local:
            return role
        alt = f"{role}_local"
        if alt in self.models.roles and self.models.role(alt).is_local:
            return alt
        raise LocalOnlyError(
            f"a local_only source needs a local model for role '{role}': add a local "
            f"'{alt}' role to models.yaml (see config/examples/models.api.yaml)"
        )

    def _call_kwargs(self, role: RoleConfig) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": role.model}
        api_base = role.api_base or (
            self.ollama_base_url if role.provider.startswith("ollama") else None
        )
        if api_base:
            kwargs["api_base"] = api_base.replace("{ollama_base_url}", self.ollama_base_url)
        if role.api_key:
            kwargs["api_key"] = role.api_key
        return kwargs

    def chat(
        self,
        messages: list[Message],
        role: str = "answer",
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        **params: Any,
    ) -> str:
        """Return the assistant text for `messages`. Per-call params override models.yaml."""
        cfg = self.models.role(role)
        kwargs = self._call_kwargs(cfg) | cfg.params | params
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = litellm.completion(messages=messages, **kwargs)
        return resp.choices[0].message.content or ""

    def chat_with_logprobs(
        self, messages: list[Message], role: str = "vision", **params: Any
    ) -> tuple[str, list[float]]:
        """Like chat(), plus per-token logprobs (empty if the provider returns none)."""
        cfg = self.models.role(role)
        kwargs = self._call_kwargs(cfg) | cfg.params | params | {"logprobs": True}
        resp = litellm.completion(messages=messages, **kwargs)
        choice = resp.choices[0]
        lp = getattr(choice, "logprobs", None)
        content = getattr(lp, "content", None) if lp is not None else None
        if content is None and isinstance(lp, dict):
            content = lp.get("content")
        logprobs = [
            float(t["logprob"] if isinstance(t, dict) else t.logprob) for t in content or []
        ]
        return choice.message.content or "", logprobs

    def embed(self, texts: list[str], role: str = "embedding") -> list[list[float]]:
        if not texts:
            return []
        cfg = self.models.role(role)
        resp = litellm.embedding(input=texts, **self._call_kwargs(cfg))
        data = sorted(resp.data, key=lambda d: d["index"])
        vectors = [list(d["embedding"]) for d in data]
        dim = getattr(cfg, "dim", None)
        if dim and vectors and len(vectors[0]) != dim:
            raise ValueError(
                f"Embedding dim {len(vectors[0])} != configured dim {dim} for role '{role}'"
            )
        return vectors

    def rerank(
        self, query: str, documents: list[str], top_n: int | None = None, role: str = "reranker"
    ) -> list[RerankResult]:
        """Return documents' indices sorted by relevance (best first)."""
        if not documents:
            return []
        cfg = self.models.role(role)
        if cfg.backend == "litellm":  # Cohere, Jina, ... in API mode
            resp = litellm.rerank(query=query, documents=documents, **self._call_kwargs(cfg))
            results = [
                RerankResult(index=r["index"], score=float(r["relevance_score"]))
                for r in resp.results
            ]
        else:  # a registered backend (llm/backends.py), e.g. the local cross_encoder
            scores = self._reranker(cfg).predict([(query, d) for d in documents], batch_size=16)
            results = [RerankResult(index=i, score=float(s)) for i, s in enumerate(scores)]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_n] if top_n else results

    def _reranker(self, cfg: RoleConfig) -> Any:
        key = f"{cfg.backend}:{cfg.model}"
        if key not in self._cross_encoders:  # loaded once per process
            from thesiskb.llm.backends import create_backend

            self._cross_encoders[key] = create_backend("reranker", cfg, get_settings())
        return self._cross_encoders[key]


def image_message(prompt: str, jpeg: bytes) -> list[Message]:
    """One user message with text + an inline JPEG, in OpenAI content-part format."""
    import base64

    url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ]


@lru_cache
def get_gateway() -> LLMGateway:
    return LLMGateway(get_models_config(), get_settings().ollama_base_url)
