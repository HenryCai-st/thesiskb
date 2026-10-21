"""Docling wrapper: PDF/Office → page-located blocks with section paths.

Two steps so the GPU part runs once:
  convert()        Docling (GPU layout/table models) → data/processed/<sha>/docling.json
  extract_blocks() docling.json → blocks with {page, bbox} locators (CPU, cheap to re-run)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from thesiskb.ingest.chunking import Block

SKIP_LABELS = {"page_header", "page_footer", "document_index"}
SKIP_SECTIONS = re.compile(r"^(references|bibliography|literatur(verzeichnis)?)$", re.I)
_NUMBERED = re.compile(r"^(?:[A-Z]|\d+)(?:\.\d+)*\.?\s+\S")
_RUNNING_AUTHOR = re.compile(r"\bet al\.?$")  # "Yang et al." running page header
_TOP_LEVEL_WORDS = re.compile(
    r"^(abstract|introduction|contents|references|bibliography|acknowledg(e)?ments?|appendix.*)$",
    re.I,
)


class ParsedDocument(BaseModel):
    title: str | None = None
    blocks: list[Block] = Field(default_factory=list)
    page_texts: dict[int, str] = Field(default_factory=dict)
    page_sizes: dict[int, tuple[float, float]] = Field(default_factory=dict)


class DoclingConverter:
    """Holds the Docling models; create once per batch, then `close()` to free VRAM."""

    def __init__(self, do_ocr: bool = False) -> None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions(do_ocr=do_ocr, do_table_structure=True)
        self._conv = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )

    def convert(self, src: Path, out_json: Path) -> None:
        doc = self._conv.convert(src).document
        out_json.parent.mkdir(parents=True, exist_ok=True)
        doc.save_as_json(out_json)

    def close(self) -> None:
        del self._conv
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _heading_depth(text: str) -> int | None:
    """Depth from numbering ("2.1 Foo" → 2); None if unnumbered."""
    m = re.match(r"^((?:[A-Z]|\d+)(?:\.\d+)*)\.?\s", text)
    if not m or not _NUMBERED.match(text):
        return None
    return m.group(1).count(".") + 1


def _furniture(items: list[tuple[str, int]], min_pages: int = 3) -> set[str]:
    """Short strings that repeat on many pages: running headers/footers."""
    pages_by_text: dict[str, set[int]] = {}
    for text, page in items:
        t = _norm(text)
        if 0 < len(t) <= 80:
            pages_by_text.setdefault(t, set()).add(page)
    return {t for t, pages in pages_by_text.items() if len(pages) >= min_pages}


def _strip_furniture(text: str, furniture: set[str]) -> str:
    t = _norm(text)
    for f in sorted(furniture, key=len, reverse=True):
        if t.startswith(f + " "):
            t = t[len(f) + 1 :]
        if t.endswith(" " + f):
            t = t[: -len(f) - 1]
    return t


def extract_blocks(docling_json: Path) -> ParsedDocument:
    from docling_core.types.doc import DoclingDocument, SectionHeaderItem, TableItem, TextItem

    doc = DoclingDocument.load_from_json(docling_json)
    page_sizes = {n: (p.size.width, p.size.height) for n, p in doc.pages.items()}

    items: list[tuple[Any, int]] = [(item, level) for item, level in doc.iterate_items()]
    furniture = _furniture(
        [(it.text, it.prov[0].page_no) for it, _ in items if isinstance(it, TextItem) and it.prov]
    )

    title: str | None = None
    stack: list[tuple[str, bool]] = []  # (heading text, is_numbered)
    skipping = False
    blocks: list[Block] = []
    page_texts: dict[int, list[str]] = {}

    for item, _level in items:
        label = str(getattr(item, "label", ""))
        if label in SKIP_LABELS or not getattr(item, "prov", None):
            continue
        prov = item.prov[0]
        page = prov.page_no

        if isinstance(item, SectionHeaderItem) or label == "title":
            text = _norm(item.text)
            if not text or text in furniture or _RUNNING_AUTHOR.search(text):
                continue
            if title is None and page == 1:
                title = text
                continue
            depth = _heading_depth(text)
            if depth is not None:
                stack = [s for s in stack[: depth - 1] if s[1]]
                stack.append((text, True))
            elif not stack or _TOP_LEVEL_WORDS.match(text.rstrip(":")):
                stack = [(text, False)]
            else:
                # unnumbered sub-heading: child of the deepest numbered heading
                stack = [s for s in stack if s[1]] + [(text, False)]
            skipping = bool(SKIP_SECTIONS.match(stack[0][0].rstrip(":")))
            continue

        if skipping:
            continue

        if isinstance(item, TableItem):
            text, kind = item.export_to_markdown(doc=doc), "table"
            caption = item.caption_text(doc)
            if caption:
                text = f"{caption}\n\n{text}"
        elif isinstance(item, TextItem):
            text = _strip_furniture(item.text, furniture)
            if not text or text in furniture:
                continue
            kind = {"caption": "caption", "list_item": "list", "footnote": "footnote"}.get(
                label, "text"
            )
        else:
            continue  # pictures (captions come as separate items), formulas without text

        height = page_sizes.get(page, (0, 0))[1]
        bb = prov.bbox.to_top_left_origin(page_height=height) if height else prov.bbox
        locator = {"page": page, "bbox": [round(v, 1) for v in (bb.l, bb.t, bb.r, bb.b)]}
        blocks.append(Block(text=text, kind=kind, section=[s[0] for s in stack], locator=locator))
        page_texts.setdefault(page, []).append(text)

    return ParsedDocument(
        title=title,
        blocks=blocks,
        page_texts={p: "\n\n".join(ts) for p, ts in page_texts.items()},
        page_sizes=page_sizes,
    )


_GERMAN = {"der", "die", "und", "das", "nicht", "ist", "mit", "für", "auf", "sich", "eine", "wir"}
_ENGLISH = {"the", "and", "of", "to", "is", "in", "that", "for", "with", "we", "are", "this"}


def detect_language(text: str) -> str:
    """'de' or 'en' by stopword counts; enough to pick an FTS configuration."""
    words = re.findall(r"[a-zäöüß]+", text.lower()[:20000])
    de = sum(w in _GERMAN for w in words)
    en = sum(w in _ENGLISH for w in words)
    return "de" if de > en else "en"


FTS_CONFIG = {"de": "german", "en": "english"}
