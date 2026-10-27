"""Answer traces: store and print what happened for one question (`tkb trace`)."""

from __future__ import annotations

import logging
import traceback
import uuid
from typing import Any

from sqlalchemy import String, cast, select
from sqlalchemy.orm import Session

from thesiskb.answer.generate import Answer, locator_label
from thesiskb.core.models import AnswerTrace

log = logging.getLogger(__name__)

TRACE_ROLES = ("router", "answer", "judge", "embedding", "reranker")


def _models(gateway: Any) -> dict[str, str]:
    out = {}
    for r in TRACE_ROLES:
        try:
            out[r] = gateway.models.role(r).model
        except Exception:  # noqa: BLE001 - fakes and missing roles
            continue
    return out


def save_trace(
    session: Session, gateway: Any, ans: Answer, channel: str, error: BaseException | None = None
) -> str | None:
    """Write one trace row; never lets a tracing problem break answering."""
    d = ans.model_dump(mode="json")
    row = AnswerTrace(
        id=uuid.uuid4(),
        channel=channel,
        question=ans.question,
        answer=ans.text or None,
        refused=ans.refused,
        route=ans.route,
        retrieval=d["retrieval"],
        context=d["context"],
        sentences=d["sentences"],
        models=_models(gateway),
        timings=ans.timings,
        error="".join(traceback.format_exception(error)) if error else None,
    )
    try:
        if error is not None:
            session.rollback()  # the failed query may have aborted the transaction
        session.add(row)
        session.commit()
    except Exception:  # noqa: BLE001
        session.rollback()
        log.warning("could not store the answer trace", exc_info=True)
        return None
    return str(row.id)


def get_trace(session: Session, ref: str = "last") -> AnswerTrace | None:
    """'last', a full id, or an unambiguous id prefix (as printed by `tkb ask`)."""
    if ref == "last":
        return session.scalar(select(AnswerTrace).order_by(AnswerTrace.created_at.desc()).limit(1))
    try:
        return session.get(AnswerTrace, uuid.UUID(ref))
    except ValueError:
        pass
    if not ref or not all(ch in "0123456789abcdef-" for ch in ref.lower()):
        return None
    q = select(AnswerTrace).where(cast(AnswerTrace.id, String).like(f"{ref.lower()}%")).limit(2)
    found = session.scalars(q).all()
    return found[0] if len(found) == 1 else None


def trace_out(t: AnswerTrace) -> dict[str, Any]:
    return {
        "id": str(t.id),
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "channel": t.channel,
        "question": t.question,
        "answer": t.answer,
        "refused": t.refused,
        "route": t.route,
        "retrieval": t.retrieval or [],
        "context": t.context or [],
        "sentences": t.sentences or [],
        "models": t.models or {},
        "timings": t.timings or {},
        "error": t.error,
    }


def format_trace(t: dict[str, Any], hits: int = 5) -> str:
    """Plain-text trace for the terminal."""
    route = t.get("route") or {}
    lines = [
        f"Trace {t['id']}  ({t['channel']}, {t['created_at']})",
        f"Question: {t['question']}",
        f"Route: {', '.join(route.get('paths', []))} (by {route.get('by', '?')})"
        + (f" — {route['reason']}" if route.get("reason") else ""),
        "Models: " + ", ".join(f"{k}={v}" for k, v in t["models"].items()),
        "",
        "Queries sent to the database:",
    ]
    for r in t["retrieval"]:
        lines.append(f"  [{r.get('path')}] {r.get('query')}")
        if r.get("candidates") is not None:
            lines.append(f"      {len(r['candidates'])} candidates (vector + full-text, fused)")
        for h in (r.get("reranked") or [])[:hits]:
            lines.append(f"      {h['score']:.3f}  {h['file']}  {h['chunk_id'][:8]}")
        if r.get("kept") is not None:
            lines.append(f"      kept {r['kept']} for the answer")
    lines += ["", "Passages shown to the model:"]
    for c in t["context"]:
        lines.append(f"  [S{c['n']}] {c['filename']}, {locator_label(c['locator'])} ({c['kind']})")
    lines += ["", "Sentences:"]
    for s in t["sentences"]:
        cites = ",".join(f"S{n}" for n in s["citations"]) or "-"
        note = f" — {s['note']}" if s.get("note") else ""
        lines.append(f"  {s['status']:<16} [{cites}] {s['text'][:100]}{note}")
    lines += ["", "Timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in t["timings"].items())]
    if t.get("error"):
        lines += ["", "Error:", t["error"]]
    return "\n".join(lines)
