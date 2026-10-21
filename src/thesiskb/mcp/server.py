"""MCP server exposing the knowledge base to agents and LLM clients (stdio: `tkb mcp`).

Tools: search, get_chunk, get_source, list_tasks, person_profile, ask. Everything returned is
plain JSON with chunk IDs and locators, so a client can cite exactly like `tkb ask` does.
The tools read the local database; `search`/`ask` also use the local models (one at a time).
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

from mcp.server.mcpserver import MCPServer

from thesiskb.core.config import get_pipeline_config
from thesiskb.core.db import session_scope

server = MCPServer(
    name="thesiskb",
    instructions=(
        "Private thesis knowledge base (papers, photos of notes, meeting recordings, videos). "
        "Use `search` to find passages and cite their chunk_id; `ask` returns a finished, "
        "cited answer. `list_tasks` gives the user's deadlines from SQL; `person_profile` "
        "what a person said. Never present KB content as your own knowledge."
    ),
)
_lock = threading.Lock()


def _passage(session, chunk_id: uuid.UUID, score: float | None = None) -> dict[str, Any]:
    from thesiskb.answer.generate import locator_label
    from thesiskb.core.models import Chunk, Source

    c = session.get(Chunk, chunk_id)
    s = session.get(Source, c.source_id)
    return {
        "chunk_id": str(c.id),
        "source_id": str(s.id),
        "filename": s.path.rsplit("/", 1)[-1],
        "title": s.title,
        "where": locator_label(c.locator),
        "locator": c.locator,
        "header": c.context_header,
        "verification_status": s.verification_status,
        "text": c.text,
        **({"score": round(score, 4)} if score is not None else {}),
    }


@server.tool(description="Hybrid search (vector + keyword, reranked) over all sources.")
def search(query: str, k: int = 8, source_types: list[str] | None = None) -> list[dict]:
    """Return the k best passages for `query`; optionally restrict to source types
    (paper, book, note, meeting, interview, demo)."""
    from thesiskb.answer.paths import _top_hits
    from thesiskb.llm import get_gateway
    from thesiskb.retrieval.hybrid import SearchFilters

    with _lock, session_scope() as session:
        hits = _top_hits(
            session,
            get_gateway(),
            query,
            SearchFilters(source_types=source_types),
            get_pipeline_config().retrieval,
            k=k,
        )
        return [_passage(session, h.chunk_id, h.score) for h in hits]


@server.tool(description="One passage (chunk) by its ID, with its source and exact location.")
def get_chunk(chunk_id: str) -> dict:
    with session_scope() as session:
        return _passage(session, uuid.UUID(chunk_id))


@server.tool(description="Metadata of one source file: title, authors, type, verification.")
def get_source(source_id: str) -> dict:
    from thesiskb.api.app import source_out
    from thesiskb.core.models import Source

    with session_scope() as session:
        s = session.get(Source, uuid.UUID(source_id))
        if s is None:
            raise ValueError(f"no source {source_id}")
        return source_out(s)


@server.tool(description="The user's tasks/deadlines from the tasks table, sorted by due date.")
def list_tasks(include_done: bool = False) -> list[dict]:
    from thesiskb.answer.tasks import open_tasks

    with session_scope() as session:
        return [
            {
                "id": str(t.id),
                "description": t.description,
                "owner": owner or t.owner_raw,
                "due_date": t.due_date.isoformat() if t.due_date else None,
                "due_as_written": t.due_raw,
                "status": t.status,
                "confirmed": t.confirmed,
                "evidence_quote": t.evidence_quote,
                "chunk_id": str(t.source_chunk_id) if t.source_chunk_id else None,
            }
            for t, owner in open_tasks(session, include_done=include_done)
        ]


@server.tool(description="What a person said/prefers (with quotes), recordings, papers.")
def person_profile(name: str) -> dict:
    from thesiskb.api.app import person

    with session_scope() as session:
        try:
            return person(name, session)
        except Exception as e:  # HTTPException 404 → readable error for the client
            raise ValueError(getattr(e, "detail", str(e))) from e


@server.tool(description="A finished answer with [S#] citations, routed like `tkb ask`.")
def ask(question: str) -> dict:
    from thesiskb.answer.paths import answer
    from thesiskb.llm import get_gateway

    with _lock, session_scope() as session:
        ans = answer(session, get_gateway(), question, get_pipeline_config().retrieval)
        return {
            "answer": ans.text,
            "route": ans.route,
            "sentences": [
                {"text": s.text, "citations": s.citations, "status": s.status}
                for s in ans.sentences
            ],
            "sources": [{"label": f"S{c.n}", **_passage(session, c.chunk_id)} for c in ans.cited],
        }


def main() -> None:
    import logging
    import sys

    # stdout is the MCP channel: send all logging to stderr
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    import litellm

    litellm.suppress_debug_info = True
    server.run("stdio")


if __name__ == "__main__":
    main()
