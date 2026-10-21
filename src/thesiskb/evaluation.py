"""`tkb eval`: run eval/questions.yaml, score retrieval and citations, store in eval_runs."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from thesiskb.answer.generate import answer_question
from thesiskb.core.config import REPO_ROOT, RetrievalConfig
from thesiskb.core.models import Chunk, EvalRun, Source
from thesiskb.llm.gateway import LLMGateway
from thesiskb.retrieval.hybrid import SearchFilters

PATH_FILTERS = {"literature": ["paper", "book"]}


class GoldQuestion(BaseModel):
    model_config = {"extra": "allow"}

    id: str
    question: str
    type: str
    path: str | list[str] = "general"
    expected_sources: list[str] = Field(default_factory=list)


class QuestionResult(BaseModel):
    id: str
    type: str
    status: str  # ran | skipped
    reason: str | None = None
    retrieval_hit: bool | None = None
    citation_hit: bool | None = None
    refused: bool | None = None
    invalid_citations: int = 0
    uncited_sentences: int = 0
    supported: int = 0
    checked: int = 0
    latency_s: float | None = None
    answer: str | None = None


def load_gold(path: Path) -> tuple[dict[str, str], list[GoldQuestion]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("fixtures", {}), [GoldQuestion(**q) for q in data["questions"]]


def _source_by_filename(session: Session) -> dict[str, Source]:
    return {s.path.rsplit("/", 1)[-1]: s for s in session.scalars(select(Source))}


def run_eval(
    session: Session,
    gateway: LLMGateway,
    cfg: RetrievalConfig,
    gold_path: Path,
    types: set[str] | None = None,
) -> tuple[list[QuestionResult], dict[str, Any]]:
    fixtures, questions = load_gold(gold_path)
    by_name = _source_by_filename(session)
    results: list[QuestionResult] = []

    for q in questions:
        if types and q.type not in types:
            continue
        names = [fixtures.get(s, s) for s in q.expected_sources]
        missing = [n for n in names if n not in by_name]
        if missing:
            results.append(
                QuestionResult(id=q.id, type=q.type, status="skipped", reason="not ingested")
            )
            continue
        expected = {by_name[n].id for n in names}
        paths = [q.path] if isinstance(q.path, str) else q.path
        source_types = PATH_FILTERS.get(paths[0]) if len(paths) == 1 else None
        ans = answer_question(
            session, gateway, q.question, cfg, SearchFilters(source_types=source_types)
        )
        chunk_source = {c.chunk_id: _chunk_source(session, c.chunk_id) for c in ans.context}
        cited_sources = {chunk_source[c.chunk_id] for c in ans.cited}
        results.append(
            QuestionResult(
                id=q.id,
                type=q.type,
                status="ran",
                retrieval_hit=bool(expected & set(chunk_source.values())),
                citation_hit=bool(expected & cited_sources),
                refused=ans.refused,
                invalid_citations=sum(s.status == "invalid_citation" for s in ans.sentences),
                uncited_sentences=sum(s.status == "uncited" for s in ans.sentences),
                supported=sum(s.status == "supported" for s in ans.sentences),
                checked=sum(s.status in ("supported", "unsupported") for s in ans.sentences),
                latency_s=round(sum(ans.timings.values()), 1),
                answer=ans.text,
            )
        )

    summary = summarize(results)
    session.add(
        EvalRun(
            git_sha=_git_sha(),
            config_hash=config_hash(),
            results={"summary": summary, "questions": [r.model_dump() for r in results]},
        )
    )
    session.commit()
    return results, summary


def _chunk_source(session: Session, chunk_id):
    return session.scalar(select(Chunk.source_id).where(Chunk.id == chunk_id))


def summarize(results: list[QuestionResult]) -> dict[str, Any]:
    ran = [r for r in results if r.status == "ran"]
    n = len(ran) or 1
    checked = sum(r.checked for r in ran)
    return {
        "ran": len(ran),
        "skipped": len(results) - len(ran),
        "retrieval_hit_rate": round(sum(bool(r.retrieval_hit) for r in ran) / n, 3),
        "citation_hit_rate": round(sum(bool(r.citation_hit) for r in ran) / n, 3),
        "invalid_citations": sum(r.invalid_citations for r in ran),
        "uncited_sentences": sum(r.uncited_sentences for r in ran),
        "support_rate": round(sum(r.supported for r in ran) / checked, 3) if checked else None,
        "mean_latency_s": round(sum(r.latency_s or 0 for r in ran) / n, 1),
    }


def previous_summary(session: Session) -> dict[str, Any] | None:
    runs = session.scalars(select(EvalRun).order_by(EvalRun.id.desc()).limit(2)).all()
    return runs[1].results.get("summary") if len(runs) > 1 else None


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def config_hash() -> str:
    h = hashlib.sha256()
    for name in ("models.yaml", "pipeline.yaml"):
        p = REPO_ROOT / "config" / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]
