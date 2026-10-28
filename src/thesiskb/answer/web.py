"""Web path (Milestone 17 Part B, ADR 0018): search the web only when the question asks for it.

search   the self-hosted SearXNG (docker compose --profile web), JSON API
fetch    the top results through the polite fetcher (robots.txt, delay, back-off) and
         ingest them as sources: source_type "web", unverified, without extraction or
         summaries (answers come back in about a minute, not ten)
answer   hybrid search + rerank restricted to those pages; cited like any other source
person   if the question names a known person, the pages are linked to them, so
         `tkb forget-person` deletes them
"""

from __future__ import annotations

import logging
import re
import time
import uuid

import httpx

from thesiskb.core.config import get_pipeline_config, get_settings
from thesiskb.retrieval.router import WEB_WORDS

log = logging.getLogger(__name__)

MAX_RESULTS = 4  # pages fetched and ingested per question

WEB_RULES = """- Passages from web pages were fetched just now for this question; they are \
unverified. Say that the information comes from the web pages cited."""


class WebSearchError(RuntimeError):
    pass


def asks_for_web(question: str) -> bool:
    return bool(WEB_WORDS.search(question))


_TRIGGERS = re.compile(
    r"\b(please|bitte|search(\s+for)?|look\s*up|check|find|web\s*search|such\w*(\s+nach)?|"
    r"recherchier\w*|schau\w*(\s+nach)?|(on\s+|in\s+|im\s+)?(the\s+)?(web|internet|online)|"
    r"for|nach|about|über)\b",
    re.I,
)


def search_query(question: str) -> str:
    """The question without the request to search ("Look up X online" → "X")."""
    q = re.sub(r"\s+", " ", _TRIGGERS.sub(" ", question)).strip(" ,.:;!?")
    return q or question


def web_search(query: str, n: int = 10) -> list[dict]:
    url = get_settings().searxng_url.rstrip("/") + "/search"
    try:
        r = httpx.get(url, params={"q": query, "format": "json", "safesearch": 1}, timeout=30)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise WebSearchError(
            f"web search is not reachable at {url} ({type(e).__name__}): start it with "
            "`docker compose --profile web up -d`"
        ) from e
    results = [x for x in r.json().get("results", []) if str(x.get("url", "")).startswith("http")]
    return results[:n]


def web_sources(session, gateway, question: str, person_id: str | None = None):
    """Search, fetch and ingest; returns (source ids, trace rows)."""
    from thesiskb.core.db import get_engine
    from thesiskb.ingest.queue import gpu_lock
    from thesiskb.ingest.web import Fetcher, ingest_web

    query = search_query(question)
    results = web_search(query)
    fetcher = Fetcher()
    urls = [x["url"] for x in results if fetcher.allowed(x["url"])][:MAX_RESULTS]
    trace = [{"query": query, "results": [x["url"] for x in results], "fetched": urls}]
    if not urls:
        return [], trace
    with gpu_lock(get_engine(), wait=False) as free:
        if not free:  # the queue worker may be waiting for this question to finish: never wait
            raise WebSearchError("an ingest is running; ask again when it has finished")
        _, sources, failed = ingest_web(
            session, urls, get_pipeline_config(), gateway, fetcher=fetcher, extract=False,
            summarize=False, images=False, media=False,
            person_id=uuid.UUID(person_id) if person_id else None,
        )  # fmt: skip
    trace[0]["failed"] = [u for u, _ in failed]
    return [s.id for s in sources], trace


def web_ctx(session, gateway, question, cfg, filters, route):
    from thesiskb.answer.generate import hit_citations
    from thesiskb.answer.paths import Ctx, _top_hits
    from thesiskb.retrieval.hybrid import SearchFilters

    t0 = time.perf_counter()
    ids, trace = web_sources(session, gateway, question, route.person_id)
    t1 = time.perf_counter()
    if not ids:
        return Ctx(timings={"web": t1 - t0}, retrieval=trace)
    scoped = (filters or SearchFilters()).model_copy(update={"source_ids": ids})
    top = _top_hits(session, gateway, search_query(question), scoped, cfg, trace=trace)
    return Ctx(
        cites=hit_citations(top),
        rules=WEB_RULES,
        timings={"web search+fetch": t1 - t0, "search+rerank": time.perf_counter() - t1},
        retrieval=trace,
    )
