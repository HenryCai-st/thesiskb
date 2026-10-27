"""Hybrid retrieval: pgvector cosine search + Postgres full-text search, fused with RRF."""

from __future__ import annotations

import re
import uuid
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from thesiskb.llm.gateway import LLMGateway


class SearchFilters(BaseModel):
    source_types: list[str] | None = None  # e.g. ["paper", "book"]
    verified_only: bool = False
    source_ids: list[uuid.UUID] | None = None
    chunk_ids: list[uuid.UUID] | None = None  # e.g. chunks where a given person speaks
    summaries: bool = False  # True: search only the summary chunks (summary path)


class Hit(BaseModel):
    chunk_id: uuid.UUID
    source_id: uuid.UUID
    text: str
    context_header: str | None
    locator: dict[str, Any]
    source_path: str
    source_title: str | None
    source_type: str | None
    verification_status: str
    local_only: bool = False
    score: float = 0.0  # RRF score, then reranker score
    ranks: dict[str, int] = Field(default_factory=dict)  # {"vector": 3, "fts": 1}

    @property
    def filename(self) -> str:
        return self.source_path.rsplit("/", 1)[-1]


_COLUMNS = """
    c.id AS chunk_id, c.source_id, c.text, c.context_header, c.locator,
    s.path AS source_path, s.title AS source_title, s.source_type,
    s.verification_status, s.local_only
"""


def _where(filters: SearchFilters) -> tuple[str, dict[str, Any]]:
    clauses, params = [], {}
    # summary chunks paraphrase whole sections; only the summary path searches them
    clauses.append("c.locator ? 'summary'" if filters.summaries else "NOT (c.locator ? 'summary')")
    if filters.source_types:
        clauses.append("s.source_type IN :source_types")
        params["source_types"] = filters.source_types
    if filters.verified_only:
        clauses.append("s.verification_status = 'verified'")
    if filters.source_ids:
        clauses.append("c.source_id IN :source_ids")
        params["source_ids"] = filters.source_ids
    if filters.chunk_ids:
        clauses.append("c.id IN :chunk_ids")
        params["chunk_ids"] = filters.chunk_ids
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


def _expanding(sql: str, params: dict[str, Any]):
    stmt = text(sql)
    for key in ("source_types", "source_ids", "chunk_ids"):
        if key in params:
            stmt = stmt.bindparams(bindparam(key, expanding=True))
    return stmt


def fts_query_terms(query: str) -> str:
    """OR-query of the question's words for to_tsquery (stopwords are dropped per language)."""
    words = re.findall(r"[\wäöüß]{2,}", query.lower())
    return " | ".join(dict.fromkeys(words))


def vector_search(session: Session, qvec: list[float], k: int, filters: SearchFilters) -> list[Hit]:
    where, params = _where(filters)
    sql = f"""
        SELECT {_COLUMNS}
        FROM chunks c JOIN sources s ON s.id = c.source_id
        WHERE c.embedding IS NOT NULL {where}
        ORDER BY c.embedding <=> CAST(:qvec AS vector)
        LIMIT :k
    """
    params |= {"qvec": "[" + ",".join(f"{x:.6f}" for x in qvec) + "]", "k": k}
    rows = session.execute(_expanding(sql, params), params).mappings().all()
    return [Hit(**r) for r in rows]


def fts_search(session: Session, query: str, k: int, filters: SearchFilters) -> list[Hit]:
    terms = fts_query_terms(query)
    if not terms:
        return []
    where, params = _where(filters)
    sql = f"""
        WITH q AS (
            SELECT to_tsquery('english', :terms) || to_tsquery('german', :terms) AS tsq
        )
        SELECT {_COLUMNS}
        FROM chunks c JOIN sources s ON s.id = c.source_id, q
        WHERE c.tsv @@ q.tsq {where}
        ORDER BY ts_rank_cd(c.tsv, q.tsq) DESC
        LIMIT :k
    """
    params |= {"terms": terms, "k": k}
    rows = session.execute(_expanding(sql, params), params).mappings().all()
    return [Hit(**r) for r in rows]


def rrf_fuse(result_lists: dict[str, list[Hit]], k: int = 60, limit: int = 30) -> list[Hit]:
    """Reciprocal rank fusion: score = Σ 1 / (k + rank)."""
    fused: dict[uuid.UUID, Hit] = {}
    for name, hits in result_lists.items():
        for rank, hit in enumerate(hits, start=1):
            h = fused.setdefault(hit.chunk_id, hit.model_copy(update={"score": 0.0, "ranks": {}}))
            h.score += 1.0 / (k + rank)
            h.ranks[name] = rank
    return sorted(fused.values(), key=lambda h: h.score, reverse=True)[:limit]


def hybrid_search(
    session: Session,
    gateway: LLMGateway,
    query: str,
    filters: SearchFilters | None = None,
    top_k: int = 30,
    rrf_k: int = 60,
) -> list[Hit]:
    filters = filters or SearchFilters()
    qvec = gateway.embed([query])[0]
    return rrf_fuse(
        {
            "vector": vector_search(session, qvec, top_k, filters),
            "fts": fts_search(session, query, top_k, filters),
        },
        k=rrf_k,
        limit=top_k,
    )
