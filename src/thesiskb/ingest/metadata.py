"""Bibliographic metadata lookup for papers (optional, network).

Only the paper identifier (arXiv id) is sent to the public arXiv API — never file content.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date

import httpx
from pydantic import BaseModel, Field

ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class PaperMetadata(BaseModel):
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    published: date | None = None
    origin: str | None = None  # "arXiv:2209.00796v15" or "doi:..."


def arxiv_id_from_name(name: str) -> str | None:
    m = ARXIV_ID.search(name)
    return m.group(0) if m else None


def parse_arxiv_atom(xml: str, arxiv_id: str) -> PaperMetadata | None:
    entry = ET.fromstring(xml).find("a:entry", _ATOM)
    if entry is None or entry.find("a:title", _ATOM) is None:
        return None
    title = re.sub(r"\s+", " ", entry.findtext("a:title", "", _ATOM)).strip()
    authors = [a.findtext("a:name", "", _ATOM).strip() for a in entry.findall("a:author", _ATOM)]
    published = entry.findtext("a:published", "", _ATOM)[:10]
    doi = entry.findtext("arxiv:doi", None, _ATOM)
    return PaperMetadata(
        title=title or None,
        authors=[a for a in authors if a],
        published=date.fromisoformat(published) if published else None,
        origin=f"doi:{doi}" if doi else f"arXiv:{arxiv_id}",
    )


def lookup_arxiv(arxiv_id: str, timeout: float = 10) -> PaperMetadata | None:
    try:
        resp = httpx.get(
            "https://export.arxiv.org/api/query",
            params={"id_list": arxiv_id.split("v")[0]},
            timeout=timeout,
        )
        resp.raise_for_status()
        meta = parse_arxiv_atom(resp.text, arxiv_id)
    except (httpx.HTTPError, ET.ParseError, ValueError):
        return None
    if meta and meta.origin and meta.origin.startswith("arXiv:"):
        meta.origin = f"arXiv:{arxiv_id}"
    return meta
