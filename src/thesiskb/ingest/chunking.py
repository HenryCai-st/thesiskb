"""Modality-agnostic, structure-aware chunker.

Parsers turn a source into a list of `Block`s (paragraph, table, caption, transcript turn, ...)
that each carry a section path and a locator. The chunker packs consecutive blocks of the same
section into chunks of ~target tokens, never crossing a section boundary, with a small overlap
of trailing blocks/sentences from the previous chunk of the same section.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

from thesiskb.core.config import ChunkingConfig

TokenCounter = Callable[[str], int]


class Block(BaseModel):
    text: str
    kind: str = "text"  # text|table|caption|list|turn|frame|...
    section: list[str] = Field(default_factory=list)
    locator: dict[str, Any] = Field(default_factory=dict)  # e.g. {"page": 3, "bbox": [l,t,r,b]}


class ChunkDraft(BaseModel):
    text: str
    section: list[str]
    context_header: str
    locator: dict[str, Any]
    n_tokens: int


def approx_tokens(text: str) -> int:
    """Cheap fallback (~1.3 tokens per word) when no tokenizer is installed."""
    return max(1, round(len(text.split()) * 1.3))


@lru_cache
def get_token_counter(tokenizer_name: str | None = "BAAI/bge-m3") -> TokenCounter:
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(tokenizer_name)
            return lambda text: len(tok.encode(text, add_special_tokens=False))
        except Exception:  # noqa: BLE001 — transformers missing or offline
            pass
    return approx_tokens


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_RE.split(text.strip()) if s]


def merge_locators(locators: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine block locators into one chunk locator.

    Page-based: {"page": first, "page_end": last, "boxes": [{"page", "bbox"}, ...]}
    Time-based: {"t_start": min, "t_end": max}
    Image-based: {"image": path, "rotation": deg, "regions": [[x0, y0, x1, y1], ...]}
    """
    out: dict[str, Any] = {}
    images = [loc for loc in locators if loc.get("image")]
    if images:
        out["image"] = images[0]["image"]
        out["rotation"] = images[0].get("rotation", 0)
        regions = [loc["region"] for loc in images if loc.get("region")]
        if regions:
            out["regions"] = regions
    pages = [loc["page"] for loc in locators if loc.get("page") is not None]
    if pages:
        out["page"] = min(pages)
        out["page_end"] = max(pages)
        boxes = [
            {"page": loc["page"], "bbox": loc["bbox"]}
            for loc in locators
            if loc.get("bbox") is not None
        ]
        if boxes:
            out["boxes"] = boxes
    starts = [loc["t_start"] for loc in locators if loc.get("t_start") is not None]
    ends = [loc["t_end"] for loc in locators if loc.get("t_end") is not None]
    if starts:
        out["t_start"] = min(starts)
        out["t_end"] = max(ends) if ends else None
    lines = [loc["line"] for loc in locators if loc.get("line") is not None]
    if lines:  # plain-text sources (plug-in parsers)
        out["line"] = min(lines)
        out["line_end"] = max(
            locator.get("line_end") or locator["line"]
            for locator in locators
            if locator.get("line") is not None
        )
    speakers = sorted({loc["speaker"] for loc in locators if loc.get("speaker")})
    if speakers:
        out["speakers"] = speakers
    frames = [{"t": loc["t_start"], "frame": loc["frame"]} for loc in locators if loc.get("frame")]
    if frames:
        out["frames"] = frames
    return out


def _split_oversized(block: Block, max_tokens: int, count: TokenCounter) -> list[Block]:
    """Split a block longer than max_tokens at sentence boundaries (word split as last resort)."""
    pieces: list[str] = []
    current: list[str] = []
    for sentence in split_sentences(block.text) or [block.text]:
        if count(sentence) > max_tokens:
            words = sentence.split()
            step = max(1, int(max_tokens / 1.3))
            pieces.extend(" ".join(words[i : i + step]) for i in range(0, len(words), step))
            continue
        if current and count(" ".join([*current, sentence])) > max_tokens:
            pieces.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return [block.model_copy(update={"text": p}) for p in pieces]


def _overlap_tail(blocks: list[Block], budget: int, count: TokenCounter) -> list[Block]:
    """Trailing content of the previous chunk (whole sentences) that fits in `budget` tokens."""
    if budget <= 0 or not blocks:
        return []
    last = blocks[-1]
    tail: list[str] = []
    for sentence in reversed(split_sentences(last.text)):
        if count(" ".join([sentence, *tail])) > budget:
            break
        tail.insert(0, sentence)
    return [last.model_copy(update={"text": " ".join(tail)})] if tail else []


def make_header(title: str | None, section: list[str]) -> str:
    return " › ".join([p for p in [title, *section] if p])


def chunk_blocks(
    blocks: list[Block],
    cfg: ChunkingConfig,
    title: str | None = None,
    count: TokenCounter = approx_tokens,
) -> list[ChunkDraft]:
    chunks: list[ChunkDraft] = []
    overlap_budget = int(cfg.target_tokens * cfg.overlap_ratio)

    def emit(group: list[Block], section: list[str]) -> None:
        text = "\n\n".join(b.text for b in group if b.text.strip())
        if not text.strip():
            return
        chunks.append(
            ChunkDraft(
                text=text,
                section=section,
                context_header=make_header(title, section),
                locator=merge_locators([b.locator for b in group]),
                n_tokens=count(text),
            )
        )

    group: list[Block] = []
    group_tokens = 0
    section: list[str] | None = None
    has_new_content = False

    for raw in blocks:
        if not raw.text.strip():
            continue
        for block in (
            _split_oversized(raw, cfg.max_tokens, count)
            if count(raw.text) > cfg.max_tokens
            else [raw]
        ):
            n = count(block.text)
            if block.section != section:
                if has_new_content:
                    emit(group, section or [])
                group, group_tokens, section, has_new_content = [], 0, block.section, False
            elif group_tokens + n > cfg.target_tokens and has_new_content:
                emit(group, section or [])
                group = _overlap_tail(group, min(overlap_budget, cfg.max_tokens - n), count)
                group_tokens = sum(count(b.text) for b in group)
                has_new_content = False
            group.append(block)
            group_tokens += n
            has_new_content = True

    if has_new_content:
        emit(group, section or [])
    return chunks
