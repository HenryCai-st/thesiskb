"""Plug-in points: parsers, model backends, extractors (with the example plug-ins)."""

import json
import wave
from pathlib import Path

import pytest
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session

from tests.conftest import requires_db
from thesiskb.core.config import PipelineConfig, RoleConfig
from thesiskb.core.models import Chunk, Source
from thesiskb.ingest import extract
from thesiskb.ingest.detect import detect
from thesiskb.ingest.parsers import load_plugins, parser_for, supported_modalities
from thesiskb.llm.backends import SingleSpeaker, backends, create_backend

NOTES = """# Kick-off meeting
Attendees: me, supervisor

Decisions:
We use diffusion models for the main experiments.
Data comes from the public benchmark.

Next steps:
Draft the related-work chapter by Nov.
"""


def test_text_parser_plugin_registers_suffix_and_parses():
    load_plugins(["examples.plugins.text_notes"])
    assert "text" in supported_modalities()
    assert parser_for("text").name == "text_notes"

    from examples.plugins.text_notes import parse_text

    blocks = parse_text(NOTES)
    assert blocks[0]["section"] == ["Kick-off meeting"]
    decisions = next(b for b in blocks if b["section"][-1] == "Decisions")
    assert decisions["section"] == ["Kick-off meeting", "Decisions"]
    assert decisions["line"] == 5 and decisions["line_end"] == 6
    assert "diffusion models" in decisions["text"]


def test_detect_sees_plugin_suffix(tmp_path):
    load_plugins(["examples.plugins.text_notes"])
    (tmp_path / "minutes.txt").write_text(NOTES, encoding="utf-8")
    assert [f.modality for f in detect([tmp_path])] == ["text"]


@pytest.mark.db
@requires_db
def test_text_notes_ingested_end_to_end(tmp_path, db_url):
    from thesiskb.ingest import pipeline

    load_plugins(["examples.plugins.text_notes"])
    note = tmp_path / "raw" / "kickoff_minutes.txt"
    note.parent.mkdir()
    note.write_text(NOTES, encoding="utf-8")
    cfg = PipelineConfig(processed_dir=tmp_path / "processed")

    class FakeGateway:
        def role_for(self, role, local_only=False):  # fakes are local
            return role

        def embed(self, texts, role="embedding"):
            return [[1.0] + [0.0] * 1023 for _ in texts]

    engine = create_engine(db_url)
    with Session(engine) as s:
        try:
            files = detect([note])
            rep = pipeline.ingest(
                s, files, cfg, FakeGateway(), metadata_lookup=False, extract=False
            )
            assert (rep["parse"], rep["chunk"], rep["embed"]) == (1, 1, 1), rep
            src = s.scalar(select(Source).where(Source.sha256 == files[0].sha256))
            chunks = s.scalars(select(Chunk).where(Chunk.source_id == src.id)).all()
            assert chunks and all(c.locator.get("line") for c in chunks)
            assert {tuple(c.context_header.split(" › ")[1:]) for c in chunks} >= {
                ("Kick-off meeting", "Decisions")
            }
            again = pipeline.ingest(
                s, files, cfg, FakeGateway(), metadata_lookup=False, extract=False
            )
            assert (again["parse"], again["chunk"], again["embed"]) == (0, 0, 0)
        finally:
            s.execute(delete(Source).where(Source.path.like("%kickoff_minutes.txt")))
            s.commit()
    engine.dispose()


@pytest.fixture
def decisions_plugin():
    load_plugins(["examples.plugins.decisions"])
    from examples.plugins.decisions import DecisionsExtractor

    extract.register_extractor(DecisionsExtractor())
    yield
    extract._EXTRACTORS.pop("decisions", None)
    extract._model_for.cache_clear()


def test_extractor_plugin_joins_schema_and_evidence_check(decisions_plugin):
    meeting = Source(sha256="m" * 64, path="p", modality="audio", source_type="meeting")
    paper = Source(sha256="p" * 64, path="p", modality="pdf", source_type="paper")
    assert extract.fields_for(meeting)[-1] == "decisions"
    assert "decisions" not in extract.fields_for(paper)

    class FakeLLM:
        def chat(self, messages, role="answer", **kw):
            assert "decisions" in kw["response_format"].model_fields
            assert '"decisions"' in messages[0]["content"]
            return json.dumps(
                {
                    "decisions": [
                        {
                            "decision": "Use diffusion models",
                            "evidence_quote": "We use diffusion models",
                        },
                        {"decision": "Invented", "evidence_quote": "never said"},
                    ]
                }
            )

    chunk = Chunk(text="We use diffusion models for the main experiments.")
    x = extract.extract_chunk(FakeLLM(), chunk, extract.fields_for(meeting))
    kept, dropped = extract.verified(x, chunk.text)
    assert [d.decision for d in kept.decisions] == ["Use diffusion models"]
    assert dropped == 1


def test_builtin_field_names_cannot_be_overridden():
    class Bad(extract.Extractor):
        key = "tasks"
        item = extract.XTask

        def write(self, session, src, items):
            return 0

    with pytest.raises(ValueError, match="built-in"):
        extract.register_extractor(Bad())


def _wav(path: Path, seconds: float = 2.0) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


def test_single_speaker_diarization_backend(tmp_path):
    assert {"pyannote", "none"} <= set(backends("diarization"))
    dia = create_backend("diarization", RoleConfig(model="-", backend="none"), None)
    assert isinstance(dia, SingleSpeaker)
    turns = dia.diarize(_wav(tmp_path / "a.wav"), tmp_path / "d.json")
    assert turns == [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]


def test_unknown_backend_lists_known_ones():
    with pytest.raises(ValueError, match="known"):
        create_backend("asr", RoleConfig(model="x", backend="nope"), None)
