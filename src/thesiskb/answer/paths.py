"""Answer paths as context providers + one orchestrator (`answer`).

Each path returns citable passages; the orchestrator routes the question, collects passages
from every chosen path, numbers them [S1].. and runs one grounded generation plus the usual
citation/support checks. Multi-path questions therefore need no special handling.

  general     hybrid search + rerank over all sources
  literature  papers/books: LLM sub-queries → hybrid search each → rerank on the question →
              per-chunk evidence summary with a relevance score (PaperQA2-style) → keep the best
  person      statements extracted for the person + chunks where they speak or that they
              authored, searched with the question
  task        open tasks from SQL, sorted by due date
  summary     whole-document questions: the source's summary plus every part's summary, in order
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from thesiskb.answer.generate import REFUSAL, Answer, Citation, generate_from, hit_citations
from thesiskb.answer.tasks import TASK_RULES, task_citations
from thesiskb.core.config import RetrievalConfig, get_pipeline_config
from thesiskb.core.models import Chunk, Person, Source, Speaker, Statement
from thesiskb.ingest.extract import quote_in
from thesiskb.llm.gateway import LLMGateway
from thesiskb.retrieval.hybrid import Hit, SearchFilters, hybrid_search, rrf_fuse
from thesiskb.retrieval.rerank import rerank
from thesiskb.retrieval.router import Route, route_question

log = logging.getLogger(__name__)

MAX_PASSAGES = 12


class Ctx(BaseModel):
    """What a provider hands to the orchestrator."""

    cites: list[Citation] = Field(default_factory=list)
    rules: str = ""
    timings: dict[str, float] = Field(default_factory=dict)
    retrieval: list[dict[str, Any]] = Field(default_factory=list)  # for the answer trace
    keep_all: bool = False  # never truncated by merge() (summary: every part must be shown)


def hit_rows(hits: list[Hit]) -> list[dict[str, Any]]:
    """Compact hit list for the answer trace."""
    return [
        {
            "chunk_id": str(h.chunk_id),
            "file": h.filename,
            "score": round(h.score, 4),
            "ranks": h.ranks,
        }
        for h in hits
    ]


def _top_hits(
    session, gateway, query, filters, cfg: RetrievalConfig, k: int | None = None, trace=None
):
    hits = hybrid_search(session, gateway, query, filters, cfg.hybrid_top_k, cfg.rrf_k)
    top = rerank(gateway, query, hits, k or cfg.rerank_top_k)
    kept = [h for h in top if h.score >= cfg.min_rerank_score]
    if trace is not None:
        trace.append(
            {
                "query": query,
                "candidates": hit_rows(hits),
                "reranked": hit_rows(top),
                "kept": len(kept),
            }
        )
    return kept


# --- general -------------------------------------------------------------------------------


def general_ctx(session, gateway, question, cfg, filters, route) -> Ctx:
    t0 = time.perf_counter()
    trace: list[dict[str, Any]] = []
    top = _top_hits(session, gateway, question, filters, cfg, trace=trace)
    return Ctx(
        cites=hit_citations(top),
        timings={"search+rerank": time.perf_counter() - t0},
        retrieval=trace,
    )


# --- literature ----------------------------------------------------------------------------

SUBQUERY_PROMPT = """Rewrite the research question into {n} short, different search queries for \
a library of papers (vary the wording and cover the sub-aspects). Return JSON: \
{{"queries": ["...", "..."]}}

Question: {question}"""

SUMMARY_PROMPT = """Question: {question}

Passage ({header}):
{text}

Summarise only what this passage says that helps answer the question, in 1-3 sentences, keeping \
method names, numbers and claims exactly as written. Then rate its relevance 0-10 (0 = nothing \
relevant). Return JSON: {{"summary": "...", "relevance": 0-10}}"""

LITERATURE_RULES = """- Passages marked "Relevant evidence" are question-focused summaries of a \
paper chunk; cite them like any passage. Synthesise across papers: group by approach or theme, \
note agreements and disagreements, and name methods as the papers do."""


class _Queries(BaseModel):
    queries: list[str]


class _Summary(BaseModel):
    summary: str
    relevance: float = Field(ge=0, le=10)


def sub_queries(gateway: LLMGateway, question: str, n: int = 3) -> list[str]:
    try:
        raw = gateway.chat(
            [{"role": "user", "content": SUBQUERY_PROMPT.format(n=n, question=question)}],
            role="router",
            response_format=_Queries,
        )
        qs = [q.strip() for q in _Queries.model_validate(json.loads(raw)).queries if q.strip()]
    except (ValidationError, ValueError):
        qs = []
    return list(dict.fromkeys([question, *qs[:n]]))


def evidence_summary(gateway: LLMGateway, question: str, c: Citation) -> _Summary | None:
    prompt = SUMMARY_PROMPT.format(
        question=question, header=c.context_header or c.filename, text=c.text
    )
    try:
        raw = gateway.chat(
            [{"role": "user", "content": prompt}],
            role=gateway.role_for("router", c.local_only),
            response_format=_Summary,
        )
        return _Summary.model_validate(json.loads(raw))
    except (ValidationError, ValueError):
        return None


def literature_ctx(session, gateway, question, cfg, filters, route) -> Ctx:
    lit = (get_pipeline_config().model_extra or {}).get("literature", {})
    t0 = time.perf_counter()
    f = (filters or SearchFilters()).model_copy(update={"source_types": ["paper", "book"]})
    queries = sub_queries(gateway, question, lit.get("sub_queries", 3))
    t1 = time.perf_counter()
    lists = {
        f"q{i}": hybrid_search(session, gateway, q, f, cfg.hybrid_top_k, cfg.rrf_k)
        for i, q in enumerate(queries)
    }
    hits: list[Hit] = rrf_fuse(lists, k=cfg.rrf_k, limit=cfg.hybrid_top_k)
    top = [
        h
        for h in rerank(gateway, question, hits, lit.get("candidates", 10))
        if h.score >= cfg.min_rerank_score
    ]
    t2 = time.perf_counter()
    cites = hit_citations(top)
    if lit.get("evidence_summaries", True):
        scored = []
        for c in cites:
            s = evidence_summary(gateway, question, c)
            if s is None:
                scored.append((5.0, c))  # keep, unsummarised
            elif s.relevance >= lit.get("min_relevance", 3):
                c.summary = s.summary
                scored.append((s.relevance, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        cites = [c for _, c in scored[: cfg.rerank_top_k]]
    t3 = time.perf_counter()
    trace = [{"query": q, "candidates": hit_rows(lists[f"q{i}"])} for i, q in enumerate(queries)]
    trace.append(
        {
            "query": f"(fused, reranked on the question; {len(hits)} candidates)",
            "reranked": hit_rows(top),
            "kept": len(cites),
        }
    )
    return Ctx(
        cites=cites,
        rules=LITERATURE_RULES,
        timings={"sub-queries": t1 - t0, "search+rerank": t2 - t1, "summaries": t3 - t2},
        retrieval=trace,
    )


# --- person --------------------------------------------------------------------------------

PERSON_RULES = """- The question is about {name}. Attribute to {name} only what the passages \
show {name} said, wrote or prefers ("Statement by" passages, lines starting "[m:ss] {name}:", or \
papers {name} authored); other speakers' lines are context, never {name}'s view. Group the \
answer by topic and give each claim's timestamp or page."""


def person_chunk_ids(session: Session, person: Person) -> list[uuid.UUID]:
    """Chunks where the person speaks (via speaker labels) or that come from papers they wrote."""
    labels = session.execute(
        select(Speaker.source_id, Speaker.diarization_label).where(Speaker.person_id == person.id)
    ).all()
    ids: list[uuid.UUID] = []
    for source_id, label in labels:
        for c in session.scalars(
            select(Chunk).where(Chunk.source_id == source_id, ~Chunk.locator.has_key("summary"))
        ):
            if label in (c.locator.get("speakers") or []):
                ids.append(c.id)
    names = [person.name, *(person.aliases or [])]
    authors = func.array_to_string(Source.authors, "|")
    authored = session.scalars(
        select(Source.id).where(or_(*[authors.ilike(f"%{n}%") for n in names]))
    ).all()
    if authored:
        ids += session.scalars(select(Chunk.id).where(Chunk.source_id.in_(authored))).all()
    return list(dict.fromkeys(ids))


def statement_citations(session: Session, person: Person, start: int = 1) -> list[Citation]:
    cites = []
    rows = session.scalars(select(Statement).where(Statement.person_id == person.id)).all()
    for st in rows:
        chunk = session.get(Chunk, st.source_chunk_id) if st.source_chunk_id else None
        src = session.get(Source, st.source_id)
        if chunk is None or src is None:
            continue
        # the transcript line holding the quote (keeps its "[m:ss] Name:" for pinpointing)
        line = next(
            (ln for ln in chunk.text.splitlines() if quote_in(st.evidence_quote, ln)),
            f'"{st.evidence_quote}"',
        )
        cites.append(
            Citation(
                n=start + len(cites),
                chunk_id=chunk.id,
                filename=src.path.rsplit("/", 1)[-1],
                title=src.title,
                context_header=chunk.context_header,
                locator=chunk.locator,
                verification_status=src.verification_status,
                text=f"Statement by {person.name} on {st.topic}: {st.stance}\n{line}",
                kind="statement",
                local_only=src.local_only,
            )
        )
    return cites


def person_ctx(session, gateway, question, cfg, filters, route: Route) -> Ctx:
    t0 = time.perf_counter()
    person = session.get(Person, uuid.UUID(route.person_id)) if route.person_id else None
    if person is None:
        return Ctx()
    ids = person_chunk_ids(session, person)
    hits = []
    trace: list[dict[str, Any]] = []
    if ids:
        f = (filters or SearchFilters()).model_copy(update={"chunk_ids": ids})
        hits = _top_hits(session, gateway, question, f, cfg, k=6, trace=trace)
    # statements first (they carry the attribution), then the person's chunks for context
    cites = statement_citations(session, person) + hit_citations(hits)
    return Ctx(
        cites=cites,
        rules=PERSON_RULES.format(name=person.name),
        timings={"person": time.perf_counter() - t0},
        retrieval=trace,
    )


# --- entity (graph-lite) -------------------------------------------------------------------

ENTITY_RULES = """- "Entities:" passages list things the knowledge base extracted from that \
passage, with relation facts. Group the answer by entity, merge name variants of the same thing \
(e.g. "Jetson Nano Super" and "Jetson Orin Nano"), and cite the passage for each claim."""


def entity_ctx(session, gateway, question, cfg, filters, route) -> Ctx:
    from thesiskb.retrieval.graph import entity_evidence

    t0 = time.perf_counter()
    cites = []
    for ev in entity_evidence(session, question):
        chunk = session.get(Chunk, ev.chunk_id)
        src = session.get(Source, chunk.source_id) if chunk else None
        if chunk is None or src is None:
            continue
        listing = "\n".join(
            f"- {e.name} ({e.type})" + (": " + "; ".join(e.facts) if e.facts else "")
            for e in ev.entities
        )
        cites.append(
            Citation(
                n=len(cites) + 1,
                chunk_id=chunk.id,
                filename=src.path.rsplit("/", 1)[-1],
                title=src.title,
                context_header=chunk.context_header,
                locator=chunk.locator,
                verification_status=src.verification_status,
                text=f"Entities:\n{listing}\n\n{chunk.text}",
                kind="entity",
                local_only=src.local_only,
            )
        )
    return Ctx(
        cites=cites,
        rules=ENTITY_RULES,
        timings={"graph": time.perf_counter() - t0},
        retrieval=[
            {"query": f"entity graph: {question}", "chunks": [str(c.chunk_id) for c in cites]}
        ],
    )


# --- summary (whole documents) ------------------------------------------------------------

SUMMARY_RULES = """- The first passage summarises the whole source; the others ("Summary of <part>") \
summarise its parts, in order. Answer with an overview that covers EVERY part in order: one \
short paragraph or bullet per part that cites THAT part's own passage, not the whole-source \
summary (cite the whole-source summary only in the opening sentence). Say how the parts connect. \
Do not skip a part and add nothing the passages do not say."""


def summary_ctx(session, gateway, question, cfg, filters, route) -> Ctx:
    """Pick the source the question is about (best-matching summary), then show all its parts."""
    t0 = time.perf_counter()
    f = (filters or SearchFilters()).model_copy(update={"summaries": True})
    hits = hybrid_search(session, gateway, question, f, cfg.hybrid_top_k, cfg.rrf_k)
    top = rerank(gateway, question, hits, 5)
    if not top:
        return Ctx(retrieval=[{"query": question, "candidates": hit_rows(hits), "kept": 0}])
    source_id = top[0].source_id
    rows = session.scalars(
        select(Chunk).where(Chunk.source_id == source_id, Chunk.locator.has_key("summary"))
    ).all()
    rows = sorted(
        rows, key=lambda c: (c.locator["summary"] != "document", c.locator.get("index", 0))
    )
    src = session.get(Source, source_id)
    cites = [
        Citation(
            n=i,
            chunk_id=c.id,
            filename=src.path.rsplit("/", 1)[-1],
            title=src.title,
            context_header=c.context_header,
            locator=c.locator,
            verification_status=src.verification_status,
            text=c.text,
            kind="summary",
            local_only=src.local_only,
        )
        for i, c in enumerate(rows, start=1)
    ]
    trace = {
        "query": question,
        "candidates": hit_rows(hits),
        "reranked": hit_rows(top),
        "source": src.path.rsplit("/", 1)[-1],
        "parts": [c.locator.get("part") for c in rows if c.locator["summary"] == "section"],
        "kept": len(cites),
    }
    return Ctx(
        cites=cites,
        rules=SUMMARY_RULES,
        timings={"summaries": time.perf_counter() - t0},
        retrieval=[trace],
        keep_all=True,
    )


def coverage(ans: Answer) -> dict[str, Any] | None:
    """Which part summaries the answer cited; None if no part summaries were shown."""
    parts = [
        c for c in ans.context if c.kind == "summary" and c.locator.get("summary") == "section"
    ]
    if not parts:
        return None
    cited = {n for s in ans.sentences for n in s.citations}
    missing = [c.locator.get("part") for c in parts if c.n not in cited]
    return {"parts": len(parts), "cited": len(parts) - len(missing), "missing": missing}


# --- task ----------------------------------------------------------------------------------


def task_ctx(session, gateway, question, cfg, filters, route) -> Ctx:
    t0 = time.perf_counter()
    return Ctx(
        cites=task_citations(session), rules=TASK_RULES, timings={"sql": time.perf_counter() - t0}
    )


PROVIDERS: dict[str, Callable[..., Ctx]] = {
    "general": general_ctx,
    "literature": literature_ctx,
    "person": person_ctx,
    "task": task_ctx,
    "entity": entity_ctx,
    "summary": summary_ctx,
}


def merge(ctxs: list[Ctx], limit: int = MAX_PASSAGES) -> list[Citation]:
    """Renumber passages from all paths [S1].., dropping exact duplicates, path order kept."""
    seen: set[tuple] = set()
    out: list[Citation] = []
    whole = sum(len(c.cites) for c in ctxs if c.keep_all)
    rest = sum(not c.keep_all for c in ctxs)
    limit = max(limit, whole + 3 * rest)
    per_path = max(3, (limit - whole) // max(1, rest))
    for ctx in sorted(ctxs, key=lambda c: not c.keep_all):  # summaries first
        for c in ctx.cites if ctx.keep_all else ctx.cites[:per_path]:
            key = (c.chunk_id, c.kind, c.text[:200])
            if key in seen:
                continue
            seen.add(key)
            out.append(c.model_copy(update={"n": len(out) + 1}))
    return out[:limit]


def answer(
    session: Session,
    gateway: LLMGateway,
    question: str,
    cfg: RetrievalConfig,
    filters: SearchFilters | None = None,
    path: str = "auto",
    verify_support: bool = True,
    channel: str = "cli",
) -> Answer:
    """Answer a question and store its trace (answer_traces), also when answering fails."""
    from thesiskb.answer.trace import save_trace

    state: dict[str, Any] = {}
    try:
        ans = _answer(session, gateway, question, cfg, filters, path, verify_support, state)
    except Exception as e:
        save_trace(session, gateway, Answer(question=question, text="", **state), channel, error=e)
        raise
    ans.trace_id = save_trace(session, gateway, ans, channel)
    return ans


def _answer(session, gateway, question, cfg, filters, path, verify_support, state) -> Answer:
    from thesiskb.answer.verify import verify

    t0 = time.perf_counter()
    if path == "auto":
        route = route_question(session, gateway, question)
    else:
        route = Route(paths=[path], by="forced")
        if path == "person":
            forced = route_question(session, None, question, use_llm=False)
            route.person, route.person_id = forced.person, forced.person_id
    timings = {"route": time.perf_counter() - t0}
    retrieval: list[dict[str, Any]] = []
    state |= {"route": route.model_dump(), "timings": timings, "retrieval": retrieval}

    ctxs = []
    for p in route.paths:
        ctx = PROVIDERS[p](session, gateway, question, cfg, filters, route)
        timings |= {f"{p}:{k}": v for k, v in ctx.timings.items()}
        retrieval += [{"path": p} | r for r in ctx.retrieval]
        ctxs.append(ctx)
    cites = merge(ctxs)
    state["context"] = cites
    if not cites:
        return Answer(
            question=question,
            text=REFUSAL,
            text_with_ids=REFUSAL,
            refused=True,
            timings=timings,
            route=route.model_dump(),
            retrieval=retrieval,
        )
    t1 = time.perf_counter()
    rules = "\n".join(c.rules for c in ctxs if c.rules)
    ans = generate_from(gateway, question, cites, extra_rules=rules)
    t2 = time.perf_counter()
    verify(gateway, ans, check_support=verify_support and not ans.refused)
    ans.timings = timings | {"generate": t2 - t1, "verify": time.perf_counter() - t2}
    ans.route = route.model_dump()
    cov = coverage(ans)
    for r in retrieval:
        if r.get("path") == "summary" and cov is not None:
            r["coverage"] = cov  # parts the answer left out are visible in the trace
    ans.retrieval = retrieval
    return ans
