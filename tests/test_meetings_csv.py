"""Meeting CSV (Granola export): one unit per row; a grown export only adds units."""

import csv
import json

from thesiskb.ingest.detect import detect
from thesiskb.ingest.meetings_csv import (
    csv_units,
    meeting_blocks,
    meeting_date,
    meeting_title,
    write_meeting,
)

SUMMARY = (
    "### Project Scope\n\n- Start with a concrete use case\n- Abstract later\n\n"
    "### Topic\n\n- AI at work"
)
ROW1 = [
    "58c5760f-155e-4361-b9e8-0d3fe277bbba", "someone@example.com", "15.10.2026", "Ann",
    "2026-10-15T11:30:04.008Z", SUMMARY, "",
    "Speaker A: Shall we start?\nSpeaker B: Yes, with one use case.",
]  # fmt: skip
ROW2 = [
    "7a0e1f2b-0000-4000-8000-000000000002", "someone@example.com", "22.10.2026", "Ann",
    "2026-10-22T09:00:00.000Z", "### Follow-up\n\n- Interviews planned", "",
    "Speaker A: Next week interviews.",
]  # fmt: skip


def _write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)


def test_units_per_row_and_growth_adds_only_new_rows(tmp_path):
    f = tmp_path / "granola.csv"
    _write(f, [ROW1])
    first = csv_units(f)
    _write(f, [ROW1, ROW2])  # the export grew by one meeting
    second = csv_units(f)
    assert [u for u, _ in first] == [ROW1[0]]
    assert second[0] == first[0]  # unchanged row keeps its sha → ingest skips it
    assert len(second) == 2


def test_other_csvs_are_skipped(tmp_path):
    f = tmp_path / "table.csv"
    _write(f, [["name", "value"], ["a", "1"]])
    assert csv_units(f) == []
    assert detect([f]) == []


def test_detect_gives_row_sources(tmp_path):
    f = tmp_path / "granola.csv"
    _write(f, [ROW1, ROW2])
    found = detect([tmp_path])
    assert {d.modality for d in found} == {"meeting_csv"}
    assert sorted(d.source_path.rsplit("#", 1)[1] for d in found) == sorted([ROW1[0], ROW2[0]])
    assert found[0].name.startswith("granola.csv#")


def test_meeting_json_has_no_email_or_owner(tmp_path):
    f = tmp_path / "granola.csv"
    _write(f, [ROW1])
    out = tmp_path / "meeting.json"
    write_meeting(f, ROW1[0], out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "example.com" not in out.read_text(encoding="utf-8")
    assert "Ann" not in data.values()
    assert meeting_title(data) == "Project Scope (15.10.2026)"
    assert str(meeting_date(data)) == "2026-10-15"


def test_blocks_keep_summary_sections_and_speakers():
    data = {"date": "15.10.2026", "summary": SUMMARY, "notes": "", "transcript": ROW1[7]}
    blocks = meeting_blocks(data)
    assert blocks[0].section == ["AI summary", "Project Scope"]
    assert blocks[2].section == ["AI summary", "Topic"] and blocks[2].text == "- AI at work"
    turns = [b for b in blocks if b.section == ["Transcript"]]
    assert [b.locator["speaker"] for b in turns] == ["Speaker A", "Speaker B"]
    lines = [b.locator["line"] for b in blocks]
    assert lines == sorted(lines) and len(set(lines)) == len(lines)
