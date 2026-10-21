"""Date resolution and extraction checks (LLM faked)."""

import json
from datetime import date

import pytest

from thesiskb.core.models import Chunk, Source
from thesiskb.ingest.dates import anchor_for, parse_explicit, resolve_due
from thesiskb.ingest.extract import Extraction, extract_chunk, fields_for, quote_in, verified

ANCHOR = date(2026, 10, 10)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("10.10.2026", date(2026, 10, 10)),
        ("2027-01-15", date(2027, 1, 15)),
        ("05.2027", date(2027, 5, 31)),
        ("Jan 2027", date(2027, 1, 31)),
        ("3 March 2027", date(2027, 3, 3)),
        ("March 3, 2027", date(2027, 3, 3)),
        ("Nov", date(2026, 11, 30)),  # same year: after the anchor month
        ("March", date(2027, 3, 31)),  # next year: before the anchor month
        ("Oktober", date(2026, 10, 31)),
        ("März", date(2027, 3, 31)),
        ("15 Nov", date(2026, 11, 15)),
        ("tomorrow", date(2026, 10, 11)),
        ("in 2 weeks", date(2026, 10, 24)),
        ("next Friday", date(2026, 10, 16)),  # anchor is a Saturday
        ("bis Freitag", date(2026, 10, 16)),
        ("sometime", None),
        (None, None),
    ],
)
def test_resolve_due(raw, expected):
    assert resolve_due(raw, ANCHOR) == expected


def test_anchor_is_earliest_full_date():
    assert anchor_for(["Nov", "10.10.2026", "05.2027", "Jan 2027"], date(2020, 1, 1)) == ANCHOR
    assert anchor_for(["Nov", None], date(2026, 9, 1)) == date(2026, 9, 1)
    assert parse_explicit("05.2027") == (date(2027, 5, 31), True)


def test_quote_in_normalises_whitespace_and_bullets():
    text = "10.10.2026 - Proposal\nNov - Start thesis\n- weekly meeting\n- Supervisor"
    assert quote_in("Nov – Start thesis", text)
    assert quote_in("- weekly meeting\n- Supervisor", text)
    assert quote_in("NOV - START THESIS.", text)
    assert not quote_in("Dec - Start thesis", text)
    assert not quote_in("", text)


def test_verified_drops_items_without_evidence():
    x = Extraction.model_validate(
        {
            "tasks": [
                {
                    "description": "Proposal",
                    "due": "10.10.2026",
                    "evidence_quote": "10.10.2026 - Proposal",
                },
                {"description": "Invented", "due": "Dec", "evidence_quote": "Dec - defend thesis"},
            ],
            "relations": [
                {"subject": "A", "predicate": "uses", "object": "B", "evidence_quote": "not there"}
            ],
            "entities": [{"name": "Thesis", "type": "concept"}],
        }
    )
    kept, dropped = verified(x, "10.10.2026 - Proposal\nNov - Start thesis")
    assert [t.description for t in kept.tasks] == ["Proposal"]
    assert kept.relations == [] and len(kept.entities) == 1
    assert dropped == 2


def _src(**kw):
    return Source(sha256="x" * 64, path="p", **kw)


def test_fields_depend_on_source():
    assert fields_for(_src(modality="image", source_type="note")) == [
        "tasks",
        "entities",
        "relations",
    ]
    assert fields_for(_src(modality="pdf", source_type="paper")) == ["entities", "relations"]
    assert fields_for(_src(modality="audio", source_type="meeting")) == [
        "tasks",
        "statements",
        "entities",
        "relations",
    ]
    assert fields_for(_src(modality="video", source_type="demo")) == [
        "statements",
        "entities",
        "relations",
    ]


class FakeGateway:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def chat(self, messages, role="answer", **kw):
        self.calls.append((role, messages, kw))
        return self.payload if isinstance(self.payload, str) else json.dumps(self.payload)


def test_extract_chunk_uses_schema_and_drops_unrequested_fields():
    gw = FakeGateway(
        {
            "tasks": [{"description": "x", "evidence_quote": "x"}],
            "statements": [{"person": "P", "topic": "t", "stance": "s", "evidence_quote": "x"}],
        }
    )
    out = extract_chunk(gw, Chunk(text="x", context_header="Paper: X"), ["entities", "relations"])
    assert out.tasks == [] and out.statements == []
    role, messages, kw = gw.calls[0]
    assert role == "extraction" and kw["response_format"] is Extraction
    assert '"tasks"' not in messages[0]["content"]


def test_extract_chunk_garbage_is_empty():
    assert extract_chunk(FakeGateway("no json"), Chunk(text="x"), ["tasks"]) == Extraction()


def test_month_only_detection():
    from thesiskb.ingest.dates import is_month_only

    assert all(is_month_only(r) for r in ["Nov", "Jan 2027", "05.2027", "März"])
    assert not any(is_month_only(r) for r in ["10.10.2026", "2027-01-15", "3 March 2027", None])


def test_task_row_shows_month_not_invented_day():
    from thesiskb.answer.tasks import task_row_text
    from thesiskb.core.models import Task

    t = Task(
        description="Mid-term",
        due_date=date(2027, 1, 31),
        due_raw="Jan 2027",
        status="open",
        confirmed=False,
    )
    assert "due January 2027 (written: Jan 2027)" in task_row_text(t, None)
    t2 = Task(
        description="Proposal",
        due_date=date(2026, 10, 10),
        due_raw="10.10.2026",
        status="open",
        confirmed=False,
    )
    assert "due 10 October 2026" in task_row_text(t2, None)
