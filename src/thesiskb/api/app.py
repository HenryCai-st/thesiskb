"""FastAPI app: REST API, OpenAI-compatible chat endpoint, file serving and the web UI.

Run with `tkb serve` (binds 127.0.0.1 by default). If TKB_API_TOKEN is set, every /api and /v1
request needs `Authorization: Bearer <token>`; the UI asks for it once and keeps it in the
browser. Questions are answered one at a time (one GPU): a lock serialises /api/ask and
/v1/chat/completions.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from thesiskb import __version__
from thesiskb.core.config import get_pipeline_config, get_settings
from thesiskb.core.db import get_engine
from thesiskb.core.models import Chunk, Job, Person, Source, Speaker, Statement, Task

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
_answer_lock = threading.Lock()


def get_session() -> Iterator[Session]:
    session = sessionmaker(bind=get_engine())()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def require_token(authorization: str | None = Header(default=None)) -> None:
    token = get_settings().api_token
    if token and authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="missing or wrong bearer token")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """The ingest queue worker runs inside the server; questions wait while it holds the GPU."""
    import os

    stop = None
    if os.environ.get("TKB_NO_WORKER") != "1":
        from thesiskb.ingest.queue import start_worker_thread

        stop = start_worker_thread(get_engine(), hold=lambda: _answer_lock)
    yield
    if stop is not None:
        stop.set()


app = FastAPI(title="ThesisKB", version=__version__, lifespan=lifespan)


@app.exception_handler(Exception)
async def log_unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Unexpected errors go to the log file with the request, and back as readable text."""
    log.error("%s %s failed", request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


api = Depends(require_token)


# --- models --------------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str
    path: str = "auto"
    verified_only: bool = False
    source_types: list[str] | None = None
    verify_support: bool = True


class IngestRequest(BaseModel):
    paths: list[str] = Field(default_factory=lambda: ["data/raw"])
    source_type: str | None = None
    extract: bool = True


def source_out(s: Source) -> dict[str, Any]:
    return {
        "id": str(s.id),
        "sha256": s.sha256,
        "filename": s.path.rsplit("/", 1)[-1],
        "path": s.path,
        "modality": s.modality,
        "source_type": s.source_type,
        "title": s.title,
        "authors": s.authors,
        "published": s.published.isoformat() if s.published else None,
        "origin": s.origin,
        "language": s.language,
        "verification_status": s.verification_status,
        "file_url": f"/files/{s.id}",
    }


def citation_links(session: Session, chunk_id: uuid.UUID) -> dict[str, Any]:
    """URLs a client needs to show a citation: the file, and where in it."""
    chunk = session.get(Chunk, chunk_id)
    if chunk is None:
        return {}
    src = session.get(Source, chunk.source_id)
    loc = chunk.locator or {}
    out: dict[str, Any] = {
        "source_id": str(src.id),
        "modality": src.modality,
        "file_url": f"/files/{src.id}",
    }
    if loc.get("page") is not None:
        out["open_url"] = f"/files/{src.id}#page={loc['page']}"
    elif loc.get("part") is not None:  # EPUB/HTML: rendered view, scrolled to the passage
        out["open_url"] = f"/view/{src.id}{text_fragment(chunk.text)}"
    elif loc.get("t_start") is not None:
        out["open_url"] = f"/files/{src.id}#t={int(loc['t_start'])}"
    else:
        out["open_url"] = (
            f"/files/{src.id}?upright=1" if src.modality == "image" else out["file_url"]
        )
    return out


def text_fragment(text: str, words: int = 16) -> str:
    """URL text fragment (#:~:text=) for the first words of a passage; browsers scroll to it."""
    from urllib.parse import quote

    first = text.strip().split("\n", 1)[0].removeprefix("Image: ")
    snippet = " ".join(first.split()[:words])
    return "#:~:text=" + quote(snippet, safe="").replace("-", "%2D") if snippet else ""


# --- API -----------------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "version": __version__}


@app.post("/api/ask", dependencies=[api])
def ask(req: AskRequest, session: Session = Depends(get_session)) -> dict[str, Any]:
    return _ask(req, session, "api")


def _ask(req: AskRequest, session: Session, channel: str) -> dict[str, Any]:
    from thesiskb.answer.paths import answer
    from thesiskb.llm import get_gateway
    from thesiskb.retrieval.hybrid import SearchFilters

    filters = SearchFilters(source_types=req.source_types, verified_only=req.verified_only)
    with _answer_lock:
        ans = answer(
            session,
            get_gateway(),
            req.question,
            get_pipeline_config().retrieval,
            filters,
            path=req.path,
            verify_support=req.verify_support,
            channel=channel,
        )
    out = ans.model_dump(mode="json")
    for c in out["context"]:
        c["links"] = citation_links(session, uuid.UUID(c["chunk_id"]))
    by_n = {c["n"]: c for c in out["context"]}
    for s in out["sentences"]:
        for p in s["pinpoints"]:
            src_id = by_n.get(p["n"], {}).get("links", {}).get("source_id")
            p["open_url"] = f"/files/{src_id}#t={int(p['t'])}" if src_id else None
            if p.get("frame") and src_id:
                p["frame_url"] = f"/frames/{src_id}/{Path(p['frame']).name}"
    return out


@app.get("/api/traces/{ref}", dependencies=[api])
def answer_trace(ref: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    """Queries, hits, passages and judge verdicts of one answer ('last' = most recent)."""
    from thesiskb.answer.trace import get_trace, trace_out

    t = get_trace(session, ref)
    if t is None:
        raise HTTPException(status_code=404, detail="trace not found")
    out = trace_out(t)
    for c in out["context"]:
        c["links"] = citation_links(session, uuid.UUID(c["chunk_id"]))
    return out


@app.post("/api/ingest", dependencies=[api])
def ingest(req: IngestRequest, session: Session = Depends(get_session)) -> dict[str, Any]:
    """Queue an ingest of files/folders on the server; the worker runs one item at a time."""
    from thesiskb.ingest.queue import enqueue

    missing = [p for p in req.paths if not Path(p).exists()]
    if missing:
        raise HTTPException(status_code=400, detail=f"not found on the server: {missing}")
    item = enqueue(
        session, "ingest", [str(Path(p)) for p in req.paths],
        source_type=req.source_type, extract=req.extract,
    )  # fmt: skip
    return {"queued": item.id, "status_url": "/api/jobs"}


@app.post("/api/sources/{source_id}/reingest", dependencies=[api])
def reingest(source_id: uuid.UUID, session: Session = Depends(get_session)) -> dict[str, Any]:
    """Queue a re-run of every stage for one source (a changed file becomes its new version)."""
    from thesiskb.ingest.queue import enqueue

    src = session.get(Source, source_id)
    if src is None:
        raise HTTPException(status_code=404)
    item = enqueue(session, "reingest", [src.path], source_id=src.id)
    return {"queued": item.id, "status_url": "/api/jobs"}


@app.get("/api/jobs", dependencies=[api])
def jobs(limit: int = 20, session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    """Recent queue items, newest first, with progress for the running one."""
    from thesiskb.core.models import QueueItem

    rows = session.scalars(select(QueueItem).order_by(QueueItem.id.desc()).limit(limit))
    return [
        {
            "id": q.id,
            "kind": q.kind,
            "paths": q.paths,
            "source_id": str(q.source_id) if q.source_id else None,
            "status": q.status,
            "progress": q.progress,
            "report": q.report,
            "error": q.error,
            "created_at": q.created_at.isoformat() if q.created_at else None,
            "started_at": q.started_at.isoformat() if q.started_at else None,
            "finished_at": q.finished_at.isoformat() if q.finished_at else None,
        }
        for q in rows
    ]


@app.get("/api/sources", dependencies=[api])
def sources(session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    n_chunks = dict(
        session.execute(select(Chunk.source_id, func.count()).group_by(Chunk.source_id)).all()
    )
    out = []
    for s in session.scalars(select(Source).order_by(Source.path)):
        jobs = session.scalars(select(Job).where(Job.source_id == s.id)).all()
        out.append(
            source_out(s)
            | {
                "chunks": n_chunks.get(s.id, 0),
                "stages": {j.stage: j.status for j in jobs},
            }
        )
    return out


@app.get("/api/sources/{source_id}", dependencies=[api])
def source(source_id: uuid.UUID, session: Session = Depends(get_session)) -> dict[str, Any]:
    s = session.get(Source, source_id)
    if s is None:
        raise HTTPException(status_code=404)
    speakers = session.execute(
        select(Speaker.diarization_label, Person.name)
        .outerjoin(Person, Person.id == Speaker.person_id)
        .where(Speaker.source_id == s.id)
    ).all()
    return source_out(s) | {"speakers": {label: name for label, name in speakers}}


@app.get("/api/chunks/{chunk_id}", dependencies=[api])
def chunk(chunk_id: uuid.UUID, session: Session = Depends(get_session)) -> dict[str, Any]:
    c = session.get(Chunk, chunk_id)
    if c is None:
        raise HTTPException(status_code=404)
    src = session.get(Source, c.source_id)
    pages = {b["page"] for b in (c.locator or {}).get("boxes", [])}
    return {
        "id": str(c.id),
        "text": c.text,
        "context_header": c.context_header,
        "locator": c.locator,
        "source": source_out(src),
        "links": citation_links(session, c.id),
        "page_sizes": page_sizes(session, src.id, pages) if pages else {},
    }


@app.get("/api/tasks", dependencies=[api])
def tasks(include_done: bool = False, session: Session = Depends(get_session)):
    from thesiskb.answer.tasks import open_tasks

    out = []
    for t, owner in open_tasks(session, include_done=include_done):
        out.append(
            {
                "id": str(t.id),
                "description": t.description,
                "owner": owner or t.owner_raw,
                "due_date": t.due_date.isoformat() if t.due_date else None,
                "due_raw": t.due_raw,
                "status": t.status,
                "confirmed": t.confirmed,
                "evidence_quote": t.evidence_quote,
                "chunk_id": str(t.source_chunk_id) if t.source_chunk_id else None,
                "links": citation_links(session, t.source_chunk_id) if t.source_chunk_id else {},
            }
        )
    return out


@app.post("/api/tasks/{task_id}/{action}", dependencies=[api])
def task_action(task_id: uuid.UUID, action: str, session: Session = Depends(get_session)):
    t = session.get(Task, task_id)
    if t is None:
        raise HTTPException(status_code=404)
    if action == "confirm":
        t.confirmed = True
    elif action == "done":
        t.status, t.confirmed = "done", True
    elif action == "reopen":
        t.status = "open"
    else:
        raise HTTPException(status_code=400, detail="action: confirm | done | reopen")
    return {"id": str(t.id), "status": t.status, "confirmed": t.confirmed}


@app.get("/api/people/{name}", dependencies=[api])
def person(name: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    from thesiskb.ingest.extract import resolve_person

    p = resolve_person(session, name)
    if p is None:
        raise HTTPException(status_code=404, detail=f"no person '{name}'")
    authors = func.array_to_string(Source.authors, "|")
    names = [p.name, *(p.aliases or [])]
    papers = session.scalars(
        select(Source).where(or_(*[authors.ilike(f"%{n}%") for n in names]))
    ).all()
    recordings = session.execute(
        select(Source, Speaker.diarization_label)
        .join(Speaker, Speaker.source_id == Source.id)
        .where(Speaker.person_id == p.id)
    ).all()
    statements = session.scalars(select(Statement).where(Statement.person_id == p.id)).all()
    return {
        "name": p.name,
        "aliases": p.aliases,
        "role": p.role,
        "papers": [source_out(s) for s in papers],
        "recordings": [source_out(s) | {"label": label} for s, label in recordings],
        "statements": [
            {
                "topic": st.topic,
                "stance": st.stance,
                "quote": st.evidence_quote,
                "links": citation_links(session, st.source_chunk_id) if st.source_chunk_id else {},
            }
            for st in statements
        ],
    }


# --- files ---------------------------------------------------------------------------------


@app.get("/files/{source_id}")
def file(
    source_id: uuid.UUID,
    upright: bool = False,
    session: Session = Depends(get_session),
    token: str | None = Query(default=None),
):
    """The original file (Range requests supported, so players can seek). Media elements can't
    send headers, so the token may also come as ?token=."""
    expected = get_settings().api_token
    if expected and token != expected:
        raise HTTPException(status_code=401)
    from thesiskb.ingest.pipeline import processed_dir

    s = session.get(Source, source_id)
    if s is None:
        raise HTTPException(status_code=404)
    path = Path(s.path.split("#", 1)[0])  # "<file>.csv#<row>": one row of a meeting CSV
    if upright and s.modality == "image":
        candidate = processed_dir(get_pipeline_config(), s.sha256) / "upright.jpg"
        path = candidate if candidate.exists() else path
    if not path.exists():
        raise HTTPException(status_code=410, detail="file moved; re-run tkb ingest on it")
    # inline, so PDFs open in the browser viewer at #page=N
    return FileResponse(path, content_disposition_type="inline", filename=path.name)


@app.get("/view/{source_id}", response_class=HTMLResponse)
def view(
    source_id: uuid.UUID,
    session: Session = Depends(get_session),
    token: str | None = Query(default=None),
) -> str:
    """EPUB/HTML sources rendered from their parsed text (one readable page, no scripts)."""
    expected = get_settings().api_token
    if expected and token != expected:
        raise HTTPException(status_code=401)
    from thesiskb.ingest.pipeline import processed_dir

    s = session.get(Source, source_id)
    if s is None:
        raise HTTPException(status_code=404)
    out = processed_dir(get_pipeline_config(), s.sha256)
    cached = out / "view.html"
    if not cached.exists():
        if not (out / "docling.json").exists():
            raise HTTPException(status_code=404, detail="not parsed yet; run tkb ingest")
        from docling_core.types.doc import DoclingDocument

        doc = DoclingDocument.load_from_json(out / "docling.json")
        cached.write_text(doc.export_to_html(), encoding="utf-8")
    return cached.read_text(encoding="utf-8")


@app.get("/pages/{source_id}/{page}.png")
def page_image(
    source_id: uuid.UUID,
    page: int,
    session: Session = Depends(get_session),
    token: str | None = Query(default=None),
):
    """A PDF page rendered as PNG (cached), so citations can show the page with the cited
    regions highlighted in any browser — embedded PDF viewers are not always available."""
    expected = get_settings().api_token
    if expected and token != expected:
        raise HTTPException(status_code=401)
    from thesiskb.ingest.pipeline import processed_dir

    s = session.get(Source, source_id)
    if s is None or s.modality != "pdf":
        raise HTTPException(status_code=404)
    out = processed_dir(get_pipeline_config(), s.sha256) / "pages" / f"{page}.png"
    if not out.exists():
        try:
            import pypdfium2 as pdfium
        except ImportError as e:
            raise HTTPException(
                status_code=501, detail="needs pypdfium2 (uv sync --extra local)"
            ) from e
        pdf = pdfium.PdfDocument(s.path)
        if not 1 <= page <= len(pdf):
            raise HTTPException(status_code=404)
        out.parent.mkdir(parents=True, exist_ok=True)
        pdf[page - 1].render(scale=2).to_pil().save(out)
        pdf.close()
    return FileResponse(out)


def page_sizes(session: Session, source_id: uuid.UUID, pages: set[int]) -> dict[int, list[float]]:
    """Page sizes in PDF points (from the page segments), to place bounding boxes."""
    from thesiskb.core.models import Segment

    rows = session.scalars(
        select(Segment).where(Segment.source_id == source_id, Segment.kind == "page")
    )
    return {
        r.locator["page"]: r.locator.get("size")
        for r in rows
        if r.locator.get("page") in pages and r.locator.get("size")
    }


@app.get("/frames/{source_id}/{name}")
def frame(
    source_id: uuid.UUID,
    name: str,
    session: Session = Depends(get_session),
    token: str | None = Query(default=None),
):
    expected = get_settings().api_token
    if expected and token != expected:
        raise HTTPException(status_code=401)
    from thesiskb.ingest.pipeline import processed_dir

    s = session.get(Source, source_id)
    if s is None or not name.endswith(".jpg") or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=404)
    path = processed_dir(get_pipeline_config(), s.sha256) / "frames" / name
    if not path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(path)


# --- OpenAI-compatible endpoint (Open WebUI and other clients) ------------------------------

MODEL_ID = "thesiskb"


@app.get("/v1/models", dependencies=[api])
def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "thesiskb"}],
    }


def render_markdown(ans: dict[str, Any], base_url: str = "") -> str:
    """Answer text + a source list with links, for chat clients that only show text."""
    lines = [ans["text"]]
    flagged = [s for s in ans["sentences"] if s["status"] in ("unsupported", "invalid_citation")]
    if flagged:
        lines.append(
            "\n⚠ Not supported by the cited sources: "
            + " / ".join(f"“{s['text'][:80]}”" for s in flagged)
        )
    used = {n for s in ans["sentences"] for n in s["citations"]}
    cited = [c for c in ans["context"] if c["n"] in used]
    if cited:
        lines.append("\n**Sources**")
        from thesiskb.answer.generate import locator_label

        for c in cited:
            where = locator_label(c["locator"])
            url = base_url + c.get("links", {}).get("open_url", "")
            lines.append(f"- [S{c['n']}] [{c['filename']}, {where}]({url})")
    if ans.get("trace_id"):
        lines.append(f"\n[How this answer was found]({base_url}/?trace={ans['trace_id']})")
    return "\n".join(lines)


@app.post("/v1/chat/completions", dependencies=[api])
def chat_completions(
    body: dict[str, Any], request: Request, session: Session = Depends(get_session)
):
    messages = body.get("messages") or []
    question = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), None)
    if isinstance(question, list):  # content parts
        question = " ".join(p.get("text", "") for p in question if p.get("type") == "text")
    if not question:
        raise HTTPException(status_code=400, detail="no user message")
    ans = _ask(AskRequest(question=question), session, "chat")
    # absolute links: chat clients (Open WebUI) resolve relative ones against their own host
    base = get_settings().public_url or body.get("base_url") or str(request.base_url)
    content = render_markdown(ans, base.rstrip("/"))
    created = int(time.time())
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    if body.get("stream"):

        def events():
            chunk = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": content},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            end = chunk | {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(end)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "thesiskb": {
            "route": ans.get("route"),
            "trace_id": ans.get("trace_id"),
            "date": date.today().isoformat(),
        },
    }


# --- UI ------------------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")
