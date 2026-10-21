"""Grounded answer generation with per-sentence citations.

The LLM sees the reranked chunks as passages [S1]..[Sn] and must cite them; labels are mapped
back to chunk IDs afterwards (small models copy short labels far more reliably than UUIDs, and the
"S" keeps them apart from a paper's own bibliography numbers like [377]). `text_with_ids` carries
the canonical `[chunk:<uuid>]` form.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from thesiskb.core.config import RetrievalConfig
from thesiskb.llm.gateway import LLMGateway
from thesiskb.retrieval.hybrid import Hit, SearchFilters, hybrid_search
from thesiskb.retrieval.rerank import rerank

REFUSAL = "The knowledge base does not contain enough evidence to answer this."

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered context passages from the \
user's private knowledge base.

Rules:
- End every sentence and every bullet point with the passage label(s) it comes from, like [S1] or \
[S2][S5] — including introductory and summary sentences. Only cite labels that appear in the context. \
Numbers like [377] inside the passages are the source's own references; do not copy them.
- Never use outside knowledge. If the passages do not answer the question, reply exactly: "{REFUSAL}"
- If the passages answer only part of the question, answer that part and say what is missing.
- Be concise: a short paragraph or a few bullet points. List dated items in chronological order.
- Answer in the language of the question.
- Your answer covers only the documents in this knowledge base; when asked about the "state of \
the art" or trends, say it reflects these documents, not the whole field."""


class Citation(BaseModel):
    n: int
    chunk_id: uuid.UUID
    filename: str
    title: str | None
    context_header: str | None
    locator: dict[str, Any]
    verification_status: str
    text: str


SentenceStatus = Literal[
    "supported", "unsupported", "uncited", "invalid_citation", "structural", "unchecked"
]


class Sentence(BaseModel):
    text: str
    citations: list[int] = Field(default_factory=list)
    status: SentenceStatus = "unchecked"
    note: str | None = None


class Answer(BaseModel):
    question: str
    text: str
    text_with_ids: str = ""
    refused: bool = False
    sentences: list[Sentence] = Field(default_factory=list)
    context: list[Citation] = Field(default_factory=list)  # everything shown to the LLM
    timings: dict[str, float] = Field(default_factory=dict)

    @property
    def cited(self) -> list[Citation]:
        used = {n for s in self.sentences for n in s.citations}
        return [c for c in self.context if c.n in used]


def locator_label(loc: dict[str, Any]) -> str:
    if loc.get("page") is not None:
        p, e = loc["page"], loc.get("page_end")
        return f"p. {p}" if not e or e == p else f"pp. {p}–{e}"
    if loc.get("t_start") is not None:
        return f"{_ts(loc['t_start'])}–{_ts(loc.get('t_end') or loc['t_start'])}"
    if loc.get("image"):
        parts = ["image"]
        if loc.get("regions"):
            parts.append(f"{len(loc['regions'])} region(s)")
        if loc.get("rotation"):
            parts.append(f"shown rotated {loc['rotation']}°")
        return ", ".join(parts)
    return ""


def _ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m // 60}:{m % 60:02d}:{s:02d}" if m >= 60 else f"{m}:{s:02d}"


def format_context(cites: list[Citation]) -> str:
    parts = []
    for c in cites:
        # display hints (e.g. image rotation) are for the user, not facts for the LLM
        where = "image" if c.locator.get("image") else locator_label(c.locator)
        head = f"[S{c.n}] {c.filename}{', ' + where if where else ''}"
        if c.context_header:
            head += f" | {c.context_header}"
        parts.append(f"{head}\n{c.text}")
    return "\n\n".join(parts)


# [S1], [S1, S3], [S1; 3]
_CITE = re.compile(r"\[(S\d+(?:\s*[,;]\s*S?\d+)*)\]")
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_ABBREV = re.compile(
    r"\b(et al|e\.g|i\.e|cf|vs|etc|Fig|Figs|Eq|Eqs|Sec|Tab|No|Dr|Prof|approx)\.", re.I
)
_PROTECT = "․"  # one-dot leader: stands in for "." inside abbreviations while splitting


def split_answer(text: str) -> list[Sentence]:
    """Split into sentences/bullets, keeping citation markers with their sentence."""
    protected = _ABBREV.sub(lambda m: m.group(0)[:-1] + _PROTECT, text)
    pieces = [p.replace(_PROTECT, ".").strip() for p in _SPLIT.split(protected)]
    sentences: list[Sentence] = []
    for piece in pieces:
        if not piece:
            continue
        # a citation group that got split off ("... steps. [S1]") belongs to the previous one
        if sentences and not re.search(r"\w", _CITE.sub("", piece)):
            sentences[-1] = _sentence(f"{sentences[-1].text} {piece}")
            continue
        sentences.append(_sentence(piece))
    _inherit_list_citations(sentences)
    return sentences


_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def _inherit_list_citations(sentences: list[Sentence]) -> None:
    """Uncited bullets under a cited lead-in ending in ':' inherit its citations.

    "Here is the schedule [S1]:" followed by "- Nov: start thesis" is a normal way to write a
    list; each bullet is still support-checked against the inherited passages.
    """
    lead: Sentence | None = None
    for s in sentences:
        if _BULLET.match(s.text):
            if not s.citations and lead is not None:
                s.citations = list(lead.citations)
                s.note = "citation from list lead-in"
        else:
            ends_colon = _CITE.sub("", s.text).rstrip().endswith(":")
            lead = s if ends_colon and s.citations else None


def _labels(group: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", group)]


def _sentence(piece: str) -> Sentence:
    nums = [n for group in _CITE.findall(piece) for n in _labels(group)]
    s = Sentence(text=piece, citations=list(dict.fromkeys(nums)))
    if not nums and re.sub(r"[*_#`\s-]", "", piece).endswith(":"):
        s.status = "structural"  # list heading like "- **Learning-free sampling** includes:"
    return s


def to_chunk_ids(text: str, cites: list[Citation]) -> str:
    by_n = {c.n: c.chunk_id for c in cites}

    def repl(m: re.Match) -> str:
        return "".join(
            f"[chunk:{by_n[n]}]" if n in by_n else f"[invalid:S{n}]" for n in _labels(m.group(1))
        )

    return _CITE.sub(repl, text)


def _citations(hits: list[Hit]) -> list[Citation]:
    return [
        Citation(
            n=i,
            chunk_id=h.chunk_id,
            filename=h.filename,
            title=h.source_title,
            context_header=h.context_header,
            locator=h.locator,
            verification_status=h.verification_status,
            text=h.text,
        )
        for i, h in enumerate(hits, start=1)
    ]


def generate(gateway: LLMGateway, question: str, hits: list[Hit]) -> Answer:
    cites = _citations(hits)
    user = f"Context:\n\n{format_context(cites)}\n\nQuestion: {question}"
    raw = gateway.chat(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        role="answer",
    ).strip()
    refused = REFUSAL.rstrip(".").lower() in raw.lower()
    return Answer(
        question=question,
        text=raw,
        text_with_ids=to_chunk_ids(raw, cites),
        refused=refused,
        sentences=split_answer(raw),
        context=cites,
    )


def answer_question(
    session: Session,
    gateway: LLMGateway,
    question: str,
    cfg: RetrievalConfig,
    filters: SearchFilters | None = None,
    verify_support: bool = True,
) -> Answer:
    from thesiskb.answer.verify import verify

    t0 = time.perf_counter()
    hits = hybrid_search(session, gateway, question, filters, cfg.hybrid_top_k, cfg.rrf_k)
    t1 = time.perf_counter()
    top = rerank(gateway, question, hits, cfg.rerank_top_k)
    top = [h for h in top if h.score >= cfg.min_rerank_score]
    t2 = time.perf_counter()
    if not top:
        return Answer(
            question=question,
            text=REFUSAL,
            text_with_ids=REFUSAL,
            refused=True,
            timings={"search": t1 - t0, "rerank": t2 - t1},
        )
    ans = generate(gateway, question, top)
    t3 = time.perf_counter()
    verify(gateway, ans, check_support=verify_support and not ans.refused)
    t4 = time.perf_counter()
    ans.timings = {"search": t1 - t0, "rerank": t2 - t1, "generate": t3 - t2, "verify": t4 - t3}
    return ans
