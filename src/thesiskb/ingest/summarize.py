"""Section and document summaries at ingest (BUILD_GUIDE_v2 Milestone 12).

Retrieval hands the answer model at most ~12 passages, so "summarize the book" saw whatever 12
chunks were most similar to the word "summarize" — chapters could go missing. This stage writes
one summary per part (top-level section of a document; ~10-minute window of a recording) and one
summary of the whole source built from the part summaries: a structure-based form of RAPTOR
(tree of summaries), using the document's own headings instead of clustering.

Summaries are stored as chunks whose locator has `"summary": "section" | "document"`, covering
the part's pages/time/blocks. Normal search leaves them out; the `summary` answer path uses them.
Cached per source in data/processed/<sha>/summaries.json, keyed by the parts' text hashes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from thesiskb.core.models import Chunk, Source
from thesiskb.ingest.chunking import approx_tokens
from thesiskb.llm.gateway import LLMGateway

log = logging.getLogger(__name__)

PART_INPUT_TOKENS = 5000  # per LLM call; longer parts are summarized in pieces, then combined
WINDOW_TOKENS = 2500  # recordings have no sections: consecutive windows of about this size
MIN_SOURCE_TOKENS = 400  # shorter sources need no summary: their chunks fit in any answer
MIN_PART_TOKENS = 300  # shorter sections are joined to the following one
SKIP_MODALITIES = {"image"}

PROMPT_VERSION = "2"  # part of the cache key: changing a prompt re-summarizes
PART_PROMPT = """Summarise this part of "{title}" ({part}) for someone who has not read it.
Use only what the text says - add no examples, features or conclusions of your own. Cover its \
main points, claims, methods, decisions and results in order; keep names, numbers and terms \
exactly as written. If the text is placeholder (e.g. "Lorem ipsum") or has no real content, \
say exactly that in one sentence. Otherwise 3-8 sentences of plain prose, no heading.

{text}"""

DOC_PROMPT = """Below are summaries of every part of "{title}", in order. Write a summary of the \
whole {kind} in 3-12 sentences that mentions every part, keeps their order, and states how the \
parts connect. Use only what the summaries say. Keep names, numbers and terms exactly as \
written. Plain prose, no heading.

{parts}"""


@dataclass
class Part:
    label: str
    chunks: list[Chunk]

    @property
    def text(self) -> str:
        return "\n\n".join(c.text for c in self.chunks)


def is_summary(chunk: Chunk) -> bool:
    return bool((chunk.locator or {}).get("summary"))


def _order_key(c: Chunk) -> tuple:
    loc = c.locator or {}
    return (
        loc.get("page") or 0,
        loc.get("order") or 0,
        loc.get("t_start") or 0,
        loc.get("line") or 0,
        (loc.get("boxes") or [{}])[0].get("bbox", [0, 0])[1],
    )


def _section_of(c: Chunk) -> str:
    """Top-level section from the context header ("Paper: X › 2 Methods › 2.1 …" → "2 Methods")."""
    parts = (c.context_header or "").split(" › ")
    return parts[1] if len(parts) > 1 else ""


def _ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def split_parts(src: Source, chunks: list[Chunk]) -> list[Part]:
    """Documents: consecutive chunks of the same top-level section. Recordings: time windows."""
    chunks = sorted(chunks, key=_order_key)
    parts: list[Part] = []
    if src.modality in ("audio", "video"):
        current: list[Chunk] = []
        for c in chunks:
            current.append(c)
            if sum(approx_tokens(x.text) for x in current) >= WINDOW_TOKENS:
                parts.append(Part(label="", chunks=current))
                current = []
        if current:
            parts.append(Part(label="", chunks=current))
        for p in parts:
            a, z = p.chunks[0].locator.get("t_start"), p.chunks[-1].locator.get("t_end")
            p.label = f"{_ts(a or 0)}–{_ts(z or a or 0)}"
        return parts
    sections: list[tuple[str, list[Chunk]]] = []
    for c in chunks:
        label = _section_of(c) or "Front matter"
        if sections and sections[-1][0] == label:
            sections[-1][1].append(c)
        else:
            sections.append((label, [c]))
    # short sections (web pages: "Tools", "Team", …) are joined to the next one: one LLM call
    # per few lines would be wasteful, and the label keeps the first and last heading
    first = ""
    for label, members in sections:
        if parts and sum(approx_tokens(c.text) for c in parts[-1].chunks) < MIN_PART_TOKENS:
            parts[-1].chunks += members
            parts[-1].label = f"{first} – {label}"
        else:
            first = label
            parts.append(Part(label=label, chunks=list(members)))
    return parts


def _covering_locator(chunks: list[Chunk], kind: str, label: str) -> dict[str, Any]:
    """Locator spanning the chunks: first/last page, time range or block range."""
    locs = [c.locator or {} for c in chunks]
    out: dict[str, Any] = {"summary": kind, "part": label}
    pages = [p for loc in locs for p in (loc.get("page"), loc.get("page_end")) if p is not None]
    if pages:
        out["page"], out["page_end"] = min(pages), max(pages)
    starts = [loc["t_start"] for loc in locs if loc.get("t_start") is not None]
    if starts:
        out["t_start"] = min(starts)
        out["t_end"] = max(loc.get("t_end") or loc["t_start"] for loc in locs if "t_start" in loc)
    orders = [o for loc in locs for o in (loc.get("order"), loc.get("order_end")) if o is not None]
    if orders:
        out["order"], out["order_end"] = min(orders), max(orders)
    return out


def _chat(gateway: LLMGateway, src: Source, prompt: str) -> str:
    return gateway.chat(
        [{"role": "user", "content": prompt}],
        role=gateway.role_for("extraction", src.local_only),
        max_tokens=1200,
    ).strip()


def summarize_part(gateway: LLMGateway, src: Source, title: str, part: Part) -> str:
    """One call per piece of ≤ PART_INPUT_TOKENS; pieces are then combined."""
    pieces: list[list[Chunk]] = [[]]
    for c in part.chunks:
        if (
            pieces[-1]
            and sum(approx_tokens(x.text) for x in pieces[-1]) + approx_tokens(c.text)
            > PART_INPUT_TOKENS
        ):
            pieces.append([])
        pieces[-1].append(c)
    outs = [
        _chat(
            gateway,
            src,
            PART_PROMPT.format(
                title=title, part=part.label, text="\n\n".join(c.text for c in piece)
            ),
        )
        for piece in pieces
    ]
    if len(outs) == 1:
        return outs[0]
    joined = "\n\n".join(outs)
    return _chat(gateway, src, PART_PROMPT.format(title=title, part=part.label, text=joined))


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def summarize_source(
    session: Session,
    src: Source,
    gateway: LLMGateway,
    cache_file: Path,
    on_part: Callable[[int, int], None] | None = None,
) -> int:
    """(Re)create the summary chunks of one source; returns how many were written."""
    session.execute(
        delete(Chunk).where(Chunk.source_id == src.id, Chunk.locator.has_key("summary"))
    )
    if src.modality in SKIP_MODALITIES:
        return 0
    chunks = [c for c in session.scalars(select(Chunk).where(Chunk.source_id == src.id))]
    if sum(approx_tokens(c.text) for c in chunks) < MIN_SOURCE_TOKENS:
        return 0
    title = src.title or src.path.rsplit("/", 1)[-1]
    kind = {"book": "book", "paper": "paper", "meeting": "meeting", "web": "web page"}.get(
        src.source_type or "", "document"
    )
    cache: dict[str, str] = {}
    if cache_file.exists():
        cache = json.loads(cache_file.read_text(encoding="utf-8"))

    parts = split_parts(src, chunks)
    summaries: list[str] = []
    for i, part in enumerate(parts):
        key = _hash(PROMPT_VERSION + part.label + "\n" + part.text)
        if key not in cache:
            cache[key] = summarize_part(gateway, src, title, part)
            cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=1), "utf-8")
        summaries.append(cache[key])
        if on_part:
            on_part(i + 1, len(parts) + 1)

    if len(parts) == 1:
        doc_summary = summaries[0]
    else:
        listing = "\n\n".join(f"[{p.label}]\n{s}" for p, s in zip(parts, summaries, strict=True))
        key = _hash(PROMPT_VERSION + "document\n" + listing)
        if key not in cache:
            cache[key] = _chat(
                gateway, src, DOC_PROMPT.format(title=title, kind=kind, parts=listing)
            )
            cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=1), "utf-8")
        doc_summary = cache[key]
    if on_part:
        on_part(len(parts) + 1, len(parts) + 1)

    header = (chunks[0].context_header or title).split(" › ")[0]
    fts = chunks[0].fts_config
    rows = [
        Chunk(
            source_id=src.id,
            text=f"Summary of the whole {kind} ({len(parts)} parts): {doc_summary}",
            context_header=f"{header} › Summary",
            fts_config=fts,
            locator=_covering_locator(chunks, "document", "whole " + kind) | {"parts": len(parts)},
        )
    ]
    if len(parts) > 1:
        rows += [
            Chunk(
                source_id=src.id,
                text=f"Summary of {p.label}: {s}",
                context_header=f"{header} › {p.label} › Summary",
                fts_config=fts,
                locator=_covering_locator(p.chunks, "section", p.label) | {"index": i},
            )
            for i, (p, s) in enumerate(zip(parts, summaries, strict=True))
        ]
    session.add_all(rows)
    session.flush()
    return len(rows)
