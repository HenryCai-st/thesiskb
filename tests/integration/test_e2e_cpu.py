"""End-to-end ingestion of a 1-page PDF and a ~10 s speech clip with the real parsers on CPU
(Docling, faster-whisper tiny), the real chunker and the test database — no Ollama, no GPU:
embeddings are faked, diarization uses the single-speaker backend.

Run with:  uv run pytest -m integration
CI: .github/workflows/ci.yml job `integration` (installs CPU torch, docling, faster-whisper,
espeak-ng). The audio part is skipped where espeak-ng is missing (e.g. Windows).
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session

from tests.conftest import requires_db
from tests.integration.fixtures import make_pdf, make_speech, tts_available
from thesiskb.core.config import ModelsConfig, PipelineConfig, RoleConfig, get_models_config
from thesiskb.core.models import Chunk, Source
from thesiskb.ingest.detect import detect
from thesiskb.retrieval.hybrid import SearchFilters, fts_search

pytestmark = [pytest.mark.integration, pytest.mark.db, requires_db]


class KeywordEmbedder:
    """Deterministic stand-in for bge-m3 (no Ollama in CI)."""

    def role_for(self, role, local_only=False):
        return role

    def embed(self, texts, role="embedding"):
        out = []
        for t in texts:
            v = [0.0] * 1024
            for i, word in enumerate(("diffusion", "supervisor", "sampling", "proposal")):
                v[i] = float(word in t.lower())
            v[1023] = 1.0
            out.append(v)
        return out


@pytest.fixture
def env(tmp_path, db_url, monkeypatch):
    import thesiskb.ingest.pipeline as pipeline
    from thesiskb.llm import ollama

    base = get_models_config()
    roles = dict(base.roles)
    roles["asr"] = RoleConfig(model="tiny", backend="faster_whisper", compute_type="int8")
    roles["diarization"] = RoleConfig(model="-", backend="none")
    monkeypatch.setattr(pipeline, "get_models_config", lambda: ModelsConfig(roles=roles))
    monkeypatch.setattr(ollama, "unload_all", lambda url: [])

    raw = tmp_path / "raw"
    raw.mkdir()
    cfg = PipelineConfig(processed_dir=tmp_path / "processed")
    engine = create_engine(db_url)
    with Session(engine) as s:
        yield s, raw, cfg, pipeline
        s.execute(delete(Source).where(Source.path.like(f"{raw.as_posix()}%")))
        s.commit()
    engine.dispose()


def test_pdf_parsed_chunked_and_searchable(env):
    pytest.importorskip("docling")
    s, raw, cfg, pipeline = env
    pdf = make_pdf(raw / "synthetic_paper.pdf")

    rep = pipeline.ingest(
        s, detect([pdf]), cfg, KeywordEmbedder(), metadata_lookup=False, extract=False
    )
    assert rep["failed"] == [], rep
    assert (rep["parse"], rep["chunk"], rep["embed"]) == (1, 1, 1)

    src = s.scalar(select(Source).where(Source.path == pdf.resolve().as_posix()))
    assert src.title.startswith("Denoising Diffusion Models")
    chunks = s.scalars(select(Chunk).where(Chunk.source_id == src.id)).all()
    assert chunks and all(c.locator.get("page") == 1 and c.locator.get("boxes") for c in chunks)
    sections = {c.context_header.split(" › ")[-1] for c in chunks}
    assert {"1 Introduction", "2 Sampling"} <= sections

    hits = fts_search(s, "DDIM sampling steps", 5, SearchFilters(source_ids=[src.id]))
    assert hits and "DDIM" in hits[0].text

    again = pipeline.ingest(
        s, detect([pdf]), cfg, KeywordEmbedder(), metadata_lookup=False, extract=False
    )
    assert (again["parse"], again["chunk"], again["embed"]) == (0, 0, 0)


@pytest.mark.skipif(not tts_available(), reason="needs espeak-ng and ffmpeg")
def test_speech_transcribed_with_timestamps(env):
    pytest.importorskip("faster_whisper")
    s, raw, cfg, pipeline = env
    wav = make_speech(raw / "synthetic_meeting.wav")

    rep = pipeline.ingest(s, detect([wav]), cfg, KeywordEmbedder(), extract=False)
    assert rep["failed"] == [], rep

    src = s.scalar(select(Source).where(Source.path == wav.resolve().as_posix()))
    chunks = s.scalars(select(Chunk).where(Chunk.source_id == src.id)).all()
    text = " ".join(c.text for c in chunks).lower()
    # whisper "tiny" on a synthetic voice mishears some words ("supervisor prefers" came back
    # as "super minor preference" on CI): require most of the key words, not all
    keywords = ("test recording", "diffusion", "models", "proposal", "october", "supervisor")
    assert sum(k in text for k in keywords) >= 4, text
    assert "unknown speaker" not in text  # every word belongs to the single speaker
    loc = chunks[0].locator
    assert loc["t_start"] < 2 and 5 < loc["t_end"] < 20
    assert loc["speakers"] == ["SPEAKER_00"]
    assert "[0:0" in chunks[0].text  # lines carry their own timestamps
