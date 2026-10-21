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


class LLMGateway:
    def __init__(self, models: ModelsConfig, ollama_base_url: str) -> None:
        self.models = models
        self.ollama_base_url = ollama_base_url
        self._cross_encoders: dict[str, Any] = {}

    def _call_kwargs(self, role: RoleConfig) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": role.model}
        api_base = role.api_base or (
            self.ollama_base_url if role.provider.startswith("ollama") else None
        )
        if api_base:
            kwargs["api_base"] = api_base
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
        if cfg.backend == "cross_encoder":
            scores = self._cross_encoder(cfg).predict(
                [(query, d) for d in documents], batch_size=16
            )
            results = [RerankResult(index=i, score=float(s)) for i, s in enumerate(scores)]
        elif cfg.backend == "litellm":
            resp = litellm.rerank(query=query, documents=documents, **self._call_kwargs(cfg))
            results = [
                RerankResult(index=r["index"], score=float(r["relevance_score"]))
                for r in resp.results
            ]
        else:
            raise ValueError(f"Unknown reranker backend '{cfg.backend}'")
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_n] if top_n else results

    def _cross_encoder(self, cfg: RoleConfig) -> Any:
        if cfg.model not in self._cross_encoders:
            try:
                import torch
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise RuntimeError(
                    "Reranker backend 'cross_encoder' needs: uv sync --extra local"
                ) from e
            extra = cfg.model_extra or {}
            kwargs: dict[str, Any] = {"max_length": extra.get("max_length", 1024)}
            if extra.get("fp16", True) and torch.cuda.is_available():
                # fp16 halves VRAM (~1.1 GB) so it fits next to the answer LLM without
                # spilling into shared system memory, which is ~20x slower on Windows
                kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
            self._cross_encoders[cfg.model] = CrossEncoder(cfg.model, **kwargs)
        return self._cross_encoders[cfg.model]


@lru_cache
def get_gateway() -> LLMGateway:
    return LLMGateway(get_models_config(), get_settings().ollama_base_url)
