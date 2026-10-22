"""extract_source on a real Postgres (test DB) with the LLM faked."""

import json
from datetime import date

import pytest
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session

from tests.conftest import requires_db
from thesiskb.core.models import Chunk, EntityMention, Person, Source, Statement, Task
from thesiskb.ingest.extract import extract_source

pytestmark = [pytest.mark.db, requires_db]

NOTE = "10.10.2026 - Proposal\nNov - Start thesis\nJan 2027 - mid term presentation"


class FakeLLM:
    def role_for(self, role, local_only=False):  # fakes are local
        return role

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def chat(self, messages, role="answer", **kw):
        self.calls += 1
        return json.dumps(self.payload)


PAYLOAD = {
    "tasks": [
        {"description": "Proposal", "due": "10.10.2026", "evidence_quote": "10.10.2026 - Proposal"},
        {"description": "Start thesis", "due": "Nov", "evidence_quote": "Nov - Start thesis"},
        {"description": "Invented", "due": "Dec", "evidence_quote": "Dec - something"},
    ],
    "statements": [],
    "entities": [{"name": "Thesis", "type": "concept"}],
    "relations": [],
}


@pytest.fixture
def session(db_url, tmp_path):
    engine = create_engine(db_url)
    with Session(engine) as s:
        src = Source(sha256="e" * 64, path="/x/note.jpg", modality="image", source_type="note")
        s.add(src)
        s.flush()
        s.add(
            Chunk(
                source_id=src.id,
                text=NOTE,
                context_header="Note: note.jpg",
                locator={"image": "/x/note.jpg"},
            )
        )
        s.commit()
        yield s, src, tmp_path / "extract.json"
        s.execute(delete(Source).where(Source.id == src.id))
        s.commit()
    engine.dispose()


def test_extract_resolves_dates_drops_unverified_and_caches(session):
    s, src, cache = session
    llm = FakeLLM(PAYLOAD)
    rep = extract_source(s, src, llm, cache, reference_date=date(2026, 1, 1))
    s.commit()
    tasks = {t.description: t for t in s.scalars(select(Task).where(Task.source_id == src.id))}
    assert set(tasks) == {"Proposal", "Start thesis"}  # "Invented" had no verbatim evidence
    assert rep.dropped_no_evidence == 1
    assert tasks["Start thesis"].due_date == date(2026, 11, 30)  # anchored on 10.10.2026
    assert all(t.source_chunk_id and t.evidence_quote for t in tasks.values())
    assert s.scalar(select(EntityMention).limit(1)) is not None

    extract_source(s, src, llm, cache)  # same chunk text: served from cache
    assert llm.calls == 1


def test_confirmed_task_survives_reindex(session):
    s, src, cache = session
    extract_source(s, src, FakeLLM(PAYLOAD), cache)
    s.commit()
    proposal = s.scalar(select(Task).where(Task.description == "Proposal"))
    proposal.confirmed = True
    proposal.description = "Proposal (edited by me)"
    s.commit()

    # re-index: chunk replaced (new id, text slightly changed), extraction runs again
    s.execute(delete(Chunk).where(Chunk.source_id == src.id))
    new_chunk = Chunk(
        source_id=src.id,
        text=NOTE + "\n05.2027 - Submission",
        context_header="Note: note.jpg",
        locator={"image": "/x/note.jpg"},
    )
    s.add(new_chunk)
    s.commit()
    extract_source(s, src, FakeLLM({**PAYLOAD, "tasks": PAYLOAD["tasks"][1:]}), cache)
    s.commit()

    kept = s.get(Task, proposal.id)
    assert kept is not None and kept.description == "Proposal (edited by me)"
    assert kept.source_chunk_id == new_chunk.id  # re-linked through its evidence quote
    names = [t.description for t in s.scalars(select(Task).where(Task.source_id == src.id))]
    assert sorted(names) == ["Proposal (edited by me)", "Start thesis"]


def test_statements_resolve_people_by_alias(session, db_url):
    s, src, cache = session
    src.modality = "audio"
    src.source_type = "meeting"
    s.add(Person(name="Will.i.am", aliases=["Will I.M"]))
    c = s.scalar(select(Chunk).where(Chunk.source_id == src.id))
    c.text = "[0:22] Will I.M: I can take this with me everywhere I go."
    s.commit()
    payload = {
        "statements": [
            {
                "person": "Will I.M",
                "topic": "portability",
                "stance": "takes it everywhere",
                "evidence_quote": "I can take this with me everywhere I go",
            }
        ]
    }
    extract_source(s, src, FakeLLM(payload), cache)
    s.commit()
    st = s.scalar(select(Statement).where(Statement.source_id == src.id))
    person = s.get(Person, st.person_id)
    assert person.name == "Will.i.am" and st.person_raw == "Will I.M"
    s.delete(person)
    s.commit()


def test_failed_chunk_is_not_cached_and_retried(session):
    s, src, cache = session

    class Flaky:
        calls = 0

        def role_for(self, role, local_only=False):
            return role

        def chat(self, messages, role="answer", **kw):
            Flaky.calls += 1
            return "{ truncated" if Flaky.calls <= 2 else json.dumps(PAYLOAD)

    rep = extract_source(s, src, Flaky(), cache)
    assert rep.failed_chunks == 1
    assert json.loads(cache.read_text(encoding="utf-8")) == {}  # nothing cached for it
    rep = extract_source(s, src, Flaky(), cache)  # next run tries again and succeeds
    assert rep.failed_chunks == 0 and rep.tasks == 2
