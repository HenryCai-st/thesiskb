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
- Passages from recordings have lines like "[0:22] Name: text". When you use one, say who said \
it and when, e.g. "Will.i.am says (0:22) ...". "(on screen)" lines describe what the video shows.
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
    text: str  # the evidence the support judge checks against
    kind: str = "chunk"  # chunk | task | statement
    summary: str | None = None  # literature path: question-focused evidence summary
    score: float | None = None
    local_only: bool = False  # from a source that must never reach a cloud model


SentenceStatus = Literal[
    "supported", "unsupported", "uncited", "invalid_citation", "structural", "unchecked"
]


class Pinpoint(BaseModel):
    """The line inside a cited recording chunk that a sentence most likely refers to."""

    n: int  # citation label S<n>
    t: float  # seconds
    speaker: str | None = None
    line: str
    frame: str | None = None  # nearest keyframe image for video
    frame_t: float | None = None


class Sentence(BaseModel):
    text: str
    citations: list[int] = Field(default_factory=list)
    status: SentenceStatus = "unchecked"
    note: str | None = None
    pinpoints: list[Pinpoint] = Field(default_factory=list)


class Answer(BaseModel):
    question: str
    text: str
    text_with_ids: str = ""
    refused: bool = False
    sentences: list[Sentence] = Field(default_factory=list)
    context: list[Citation] = Field(default_factory=list)  # everything shown to the LLM
    timings: dict[str, float] = Field(default_factory=dict)
    route: dict[str, Any] | None = None  # paths chosen by the router and why
    retrieval: list[dict[str, Any]] = Field(default_factory=list)  # queries sent + hits, per path
    trace_id: str | None = None  # answer_traces row (tkb trace <id>)

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
    if loc.get("line") is not None:
        a, z = loc["line"], loc.get("line_end") or loc["line"]
        return f"line {a}" if a == z else f"lines {a}–{z}"
    if loc.get("image"):
        parts = ["image"]
        if loc.get("regions"):
            parts.append(f"{len(loc['regions'])} region(s)")
        if loc.get("rotation"):
            parts.append(f"shown rotated {loc['rotation']}°")
        return ", ".join(parts)
    if loc.get("part") is not None:
        return loc["part"] or "section"
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
        body = f"Relevant evidence: {c.summary}" if c.summary else c.text
        parts.append(f"{head}\n{body}")
    return "\n\n".join(parts)


# [S1], [S1, S3], [S1; 3]
_CITE = re.compile(r"\[(S\d+(?:\s*[,;]\s*S?\d+)*)\]")
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_ABBREV = re.compile(
    r"\b(et al|e\.g|i\.e|cf|vs|etc|Fig|Figs|Eq|Eqs|Sec|Tab|No|Dr|Prof|approx)\.", re.I
)
_PROTECT = "․"  # one-dot leader: stands in for "." inside abbreviations while splitting


_QUOTED = re.compile(r"\"[^\"\n]+\"|“[^”\n]+”|„[^“\n]+“")
_LIST_NUMBER = re.compile(r"(?m)^(\s*\d{1,3})\.(?=\s)")
# private-use stand-ins for sentence punctuation inside quotes, restored after splitting
_HIDE = {".": "", "!": "", "?": ""}


def _hide_punct(m: re.Match) -> str:
    return "".join(_HIDE.get(ch, ch) for ch in m.group(0))


def split_answer(text: str) -> list[Sentence]:
    """Split into sentences/bullets, keeping citation markers with their sentence.
    Abbreviations ("et al.") and quoted speech ('"Now I got it. I can take it..."') stay whole."""
    protected = _ABBREV.sub(lambda m: m.group(0)[:-1] + _PROTECT, text)
    protected = _LIST_NUMBER.sub(lambda m: m.group(1) + _PROTECT, protected)  # "1. Proposal"
    protected = _QUOTED.sub(_hide_punct, protected)
    restore = {_PROTECT: ".", **{v: k for k, v in _HIDE.items()}}
    pieces = ["".join(restore.get(ch, ch) for ch in p).strip() for p in _SPLIT.split(protected)]
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


_LINE = re.compile(r"^\[(\d+(?::\d{2}){1,2})\] (?:(\(on screen\)) )?(?:([^:\n]{1,60}): )?(.*)$")
_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "is",
    "are",
    "was",
    "it",
    "that",
    "this",
    "with",
    "for",
    "as",
    "at",
    "he",
    "she",
    "they",
    "says",
    "said",
    "s1",
    "s2",
    "s3",
}


def _seconds(stamp: str) -> float:
    parts = [int(p) for p in stamp.split(":")]
    return float(sum(p * 60**i for i, p in enumerate(reversed(parts))))


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[\wäöüß']+", text.lower()) if w not in _STOP and len(w) > 2}


def pinpoint(sentence: Sentence, cites: list[Citation], min_overlap: float = 0.25) -> None:
    """For citations of recordings, find the transcript line the sentence draws on.

    Lines look like "[0:22] Will.i.am: ..." (turns) or "[0:35] (on screen) ..." (keyframes).
    A timestamp written in the sentence ("(0:22)") wins; otherwise word overlap decides.
    """
    by_n = {c.n: c for c in cites}
    words = _content_words(_CITE.sub("", sentence.text))
    stamps = {_seconds(s) for s in re.findall(r"\b(\d{1,2}:\d{2})\b", sentence.text)}
    for n in sentence.citations:
        c = by_n.get(n)
        if c is None or c.locator.get("t_start") is None:
            continue
        best, best_score = None, 0.0
        for raw in c.text.splitlines():
            m = _LINE.match(raw.strip())
            if not m:
                continue
            t = _seconds(m.group(1))
            line_words = _content_words(m.group(4))
            score = len(words & line_words) / max(1, len(words))
            if any(abs(t - s) <= 2 for s in stamps):
                score += 1.0
            if score > best_score:
                best_score = score
                best = Pinpoint(
                    n=n, t=t, speaker=None if m.group(2) else m.group(3), line=raw.strip()
                )
        if best is None or best_score < min_overlap:
            continue
        if any(p.t == best.t and p.speaker == best.speaker for p in sentence.pinpoints):
            continue  # two cited passages pointing at the same transcript line
        frames = c.locator.get("frames") or []
        if frames:
            near = min(frames, key=lambda f: abs(f["t"] - best.t))
            best.frame, best.frame_t = near["frame"], near["t"]
        sentence.pinpoints.append(best)


def to_chunk_ids(text: str, cites: list[Citation]) -> str:
    by_n = {c.n: c.chunk_id for c in cites}

    def repl(m: re.Match) -> str:
        return "".join(
            f"[chunk:{by_n[n]}]" if n in by_n else f"[invalid:S{n}]" for n in _labels(m.group(1))
        )

    return _CITE.sub(repl, text)


def hit_citations(hits: list[Hit], start: int = 1) -> list[Citation]:
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
            score=h.score,
            local_only=h.local_only,
        )
        for i, h in enumerate(hits, start=start)
    ]


def generate(gateway: LLMGateway, question: str, hits: list[Hit]) -> Answer:
    return generate_from(gateway, question, hit_citations(hits))


def generate_from(
    gateway: LLMGateway,
    question: str,
    cites: list[Citation],
    extra_rules: str = "",
    system_prompt: str = SYSTEM_PROMPT,
) -> Answer:
    """One grounded answer over prepared passages (chunks, task rows, statements...)."""
    system = system_prompt + (f"\n{extra_rules}" if extra_rules else "")
    user = f"Context:\n\n{format_context(cites)}\n\nQuestion: {question}"
    raw = gateway.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        role=gateway.role_for("answer", any(c.local_only for c in cites)),
    ).strip()
    refused = REFUSAL.rstrip(".").lower() in raw.lower()
    sentences = split_answer(raw)
    for s in sentences:
        pinpoint(s, cites)
    return Answer(
        question=question,
        text=raw,
        text_with_ids=to_chunk_ids(raw, cites),
        refused=refused,
        sentences=sentences,
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
