"""forget-person and verify against the test database (models faked)."""

import pytest
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from tests.conftest import requires_db
from thesiskb import cli
from thesiskb.core.models import Person, Source, Speaker, Statement, Task

pytestmark = [pytest.mark.db, requires_db]


@pytest.fixture
def kb(db_url, monkeypatch, tmp_path):
    from contextlib import contextmanager

    engine = create_engine(db_url)
    maker = sessionmaker(bind=engine)

    @contextmanager
    def scope():
        s = maker()
        try:
            yield s
            s.commit()
        finally:
            s.close()

    monkeypatch.setattr("thesiskb.core.db.session_scope", scope)
    rechunked = []
    monkeypatch.setattr(
        "thesiskb.ingest.pipeline.rechunk", lambda session, src, cfg, gw: rechunked.append(src.path)
    )
    monkeypatch.setattr("thesiskb.llm.get_gateway", lambda: None)

    with Session(engine) as s:
        rec = Source(
            sha256="r" * 64,
            path="/x/meeting_with_prof.mp3",
            modality="audio",
            source_type="meeting",
        )
        s.add(rec)
        s.flush()
        prof = Person(name="Prof. Test", aliases=["Test"])
        s.add(prof)
        s.flush()
        s.add_all(
            [
                Speaker(source_id=rec.id, diarization_label="SPEAKER_01", person_id=prof.id),
                Statement(
                    person_id=prof.id,
                    person_raw="Prof. Test",
                    topic="method",
                    stance="prefers small models",
                    evidence_quote="small models",
                    source_id=rec.id,
                ),
                Task(
                    description="Send draft",
                    owner_person_id=prof.id,
                    owner_raw="Prof. Test",
                    source_id=rec.id,
                    evidence_quote="send the draft",
                ),
            ]
        )
        s.commit()
        yield engine, rec.id, rechunked
        s.execute(delete(Source).where(Source.id == rec.id))
        s.execute(delete(Person).where(Person.name == "Prof. Test"))
        s.commit()
    engine.dispose()


def test_forget_person_removes_their_data_and_reindexes(kb):
    engine, rec_id, rechunked = kb
    out = CliRunner().invoke(cli.app, ["forget-person", "Test", "--yes"])
    assert out.exit_code == 0, out.output
    with Session(engine) as s:
        assert s.scalar(select(Person).where(Person.name == "Prof. Test")) is None
        assert s.scalars(select(Statement).where(Statement.source_id == rec_id)).all() == []
        spk = s.scalar(select(Speaker).where(Speaker.source_id == rec_id))
        assert spk.person_id is None and spk.diarization_label == "SPEAKER_01"
        task = s.scalar(select(Task).where(Task.source_id == rec_id))
        assert task.owner_person_id is None  # task kept, no longer linked to the person
        assert s.get(Source, rec_id) is not None  # recording kept without --delete-recordings
    assert rechunked == ["/x/meeting_with_prof.mp3"]


def test_forget_person_needs_confirmation(kb):
    engine, rec_id, _ = kb
    out = CliRunner().invoke(cli.app, ["forget-person", "Test"], input="n\n")
    assert out.exit_code == 1
    with Session(engine) as s:
        assert s.scalar(select(Person).where(Person.name == "Prof. Test")) is not None


def test_forget_person_can_drop_recordings(kb):
    engine, rec_id, rechunked = kb
    out = CliRunner().invoke(cli.app, ["forget-person", "Test", "--delete-recordings", "--yes"])
    assert out.exit_code == 0, out.output
    with Session(engine) as s:
        assert s.get(Source, rec_id) is None
    assert rechunked == []


def test_verify_sets_status(kb):
    engine, rec_id, _ = kb
    out = CliRunner().invoke(cli.app, ["verify", "meeting_with_prof", "--by", "Tester"])
    assert out.exit_code == 0, out.output
    with Session(engine) as s:
        src = s.get(Source, rec_id)
        assert (src.verification_status, src.verified_by) == ("verified", "Tester")
        assert src.verified_at is not None
    CliRunner().invoke(cli.app, ["verify", "meeting_with_prof", "--status", "unverified"])
    with Session(engine) as s:
        assert s.get(Source, rec_id).verified_by is None
