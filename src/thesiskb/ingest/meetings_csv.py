"""Meeting exports in CSV (Granola): one source per row, so a growing export only adds rows.

Granola's CSV export has no header row. Columns as exported (2026-10):
  0 note id (UUID) · 1 owner e-mail · 2 date (dd.mm.yyyy) · 3 owner name · 4 created (ISO 8601)
  5 AI summary (Markdown) · 6 notes · 7 transcript ("Speaker A: …", one turn per line)
The e-mail and owner name are never indexed. CSVs in another layout yield no units (skipped).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import sys
from datetime import date, datetime
from pathlib import Path

from thesiskb.ingest.chunking import Block

log = logging.getLogger(__name__)

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T")
_TURN = re.compile(r"^(Speaker [A-Z0-9]+|[^:]{1,40}):\s+(.*)$")
_HEADING = re.compile(r"^#{1,6}\s+(.*)$")
ID, DATE, CREATED, SUMMARY, NOTES, TRANSCRIPT = 0, 2, 4, 5, 6, 7


def _rows(path: Path) -> list[list[str]]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))  # transcripts are long single fields
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return [r for r in csv.reader(fh) if r]


def _is_granola(row: list[str]) -> bool:
    return len(row) >= 8 and bool(_UUID.match(row[ID])) and bool(_ISO.match(row[CREATED]))


def csv_units(path: Path) -> list[tuple[str, str]]:
    """(row id, sha256 of the row's indexed content) per meeting row; [] for other CSVs."""
    try:
        rows = _rows(path)
    except (OSError, csv.Error, UnicodeDecodeError) as e:
        log.warning("cannot read %s: %s", path.name, e)
        return []
    units = [r for r in rows if _is_granola(r)]
    if rows and not units:
        log.info("skip %s: not a known meeting export (Granola) layout", path.name)
    out = []
    for r in units:
        content = "\x1f".join([r[ID], r[DATE], r[SUMMARY], r[NOTES], r[TRANSCRIPT]])
        out.append((r[ID], hashlib.sha256(content.encode("utf-8")).hexdigest()))
    return out


def write_meeting(path: Path, unit: str, out: Path) -> None:
    """Parse stage: the row's fields we index → meeting.json (no e-mail, no owner name)."""
    row = next((r for r in _rows(path) if _is_granola(r) and r[ID] == unit), None)
    if row is None:
        raise ValueError(f"row {unit} not found in {path.name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "id": row[ID],
        "date": row[DATE],
        "created": row[CREATED],
        "summary": row[SUMMARY],
        "notes": row[NOTES],
        "transcript": row[TRANSCRIPT],
    }
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def meeting_date(data: dict) -> date | None:
    for fmt, value in (("%d.%m.%Y", data.get("date")), ("%Y-%m-%d", data.get("created", "")[:10])):
        try:
            return datetime.strptime(value or "", fmt).date()
        except ValueError:
            continue
    return None


def meeting_title(data: dict) -> str:
    first = next(
        (m.group(1).strip() for ln in data["summary"].splitlines() if (m := _HEADING.match(ln))),
        "Meeting",
    )
    return f"{first} ({data['date']})" if data.get("date") else first


def meeting_blocks(data: dict) -> list[Block]:
    """AI summary (by heading), notes, then transcript turns; locator line = position in that
    order, speaker labels as in the transcript ("Speaker A")."""
    blocks: list[Block] = []
    line = 0
    heading = ""
    for raw in data["summary"].splitlines():
        line += 1
        text = raw.strip()
        if not text:
            continue
        if m := _HEADING.match(text):
            heading = m.group(1).strip()
            continue
        section = ["AI summary", heading] if heading else ["AI summary"]
        blocks.append(Block(text=text, section=section, locator={"line": line}))
    for raw in data.get("notes", "").splitlines():
        line += 1
        if raw.strip():
            blocks.append(Block(text=raw.strip(), section=["Notes"], locator={"line": line}))
    for raw in data["transcript"].splitlines():
        line += 1
        text = raw.strip()
        if not text:
            continue
        loc: dict = {"line": line}
        if m := _TURN.match(text):
            loc["speaker"] = m.group(1)
        blocks.append(Block(text=text, section=["Transcript"], locator=loc))
    return blocks
