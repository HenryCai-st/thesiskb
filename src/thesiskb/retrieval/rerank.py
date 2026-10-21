"""Cross-encoder reranking of hybrid hits (model from the `reranker` role)."""

from __future__ import annotations

from thesiskb.llm.gateway import LLMGateway
from thesiskb.retrieval.hybrid import Hit


def rerank(gateway: LLMGateway, query: str, hits: list[Hit], top_k: int = 8) -> list[Hit]:
    if not hits:
        return []
    docs = [f"{h.context_header}\n{h.text}" if h.context_header else h.text for h in hits]
    ranked = gateway.rerank(query, docs, top_n=top_k)
    return [hits[r.index].model_copy(update={"score": r.score}) for r in ranked]
