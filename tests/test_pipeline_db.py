"""End-to-end ingest + hybrid search on a real Postgres, with Docling and models faked."""

import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session

from tests.conftest import requires_db
from thesiskb.core.config import PipelineConfig
from thesiskb.core.models import Chunk, Source
from thesiskb.ingest import pipeline
from thesiskb.ingest.chunking import Block
from thesiskb.ingest.detect import detect
from thesiskb.ingest.documents import ParsedDocument
from thesiskb.retrieval.hybrid import SearchFilters, hybrid_search

pytestmark = [pytest.mark.db, requires_db]


class FakeConverter:
    calls = 0

    def __init__(self, do_ocr=False):
        pass

    def convert(self, src: Path, out: Path) -> None:
        FakeConverter.calls += 1
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}", encoding="utf-8")

    def close(self):
        pass


PARSED = ParsedDocument(
    title="Fake Paper",
    blocks=[
        Block(
            text="Denoising diffusion probabilistic models add noise.",
            section=["2 DDPM"],
            locator={"page": 2, "bbox": [0, 0, 1, 1]},
        ),
        Block(
            text="Score-based generative models estimate the score.",
            section=["3 SGM"],
            locator={"page": 3, "bbox": [0, 0, 1, 1]},
        ),
    ],
    page_texts={2: "DDPM page", 3: "SGM page"},
    page_sizes={2: (612, 792), 3: (612, 792)},
)


class FakeGateway:
    """Texts mentioning 'score' point along axis 0, everything else along axis 1."""

    def embed(self, texts, role="embedding"):
        out = []
        for t in texts:
            v = [0.0] * 1024
            v[0 if "score" in t.lower() else 1] = 1.0
            out.append(v)
        return out


@pytest.fixture
def setup(tmp_path, monkeypatch, db_url):
    import thesiskb.ingest.documents as documents

    monkeypatch.setattr(documents, "DoclingConverter", FakeConverter)
    monkeypatch.setattr(pipeline, "extract_blocks", lambda p: PARSED)
    monkeypatch.setattr("thesiskb.llm.ollama.unload_all", lambda url: [])
    pdf = tmp_path / "raw" / "fake paper.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(os.urandom(64))
    cfg = PipelineConfig(processed_dir=tmp_path / "processed")
    engine = create_engine(db_url)
    with Session(engine) as session:
        yield session, cfg, pdf
        session.execute(delete(Source).where(Source.path.like("%fake paper.pdf")))
        session.commit()
    engine.dispose()


def test_ingest_is_idempotent_and_searchable(setup):
    session, cfg, pdf = setup
    FakeConverter.calls = 0
    files = detect([pdf])

    first = pipeline.ingest(
        session, files, cfg, FakeGateway(), metadata_lookup=False, extract=False
    )
    assert (first["parse"], first["chunk"], first["embed"]) == (1, 1, 1)
    assert first["failed"] == []

    second = pipeline.ingest(
        session, files, cfg, FakeGateway(), metadata_lookup=False, extract=False
    )
    assert (second["parse"], second["chunk"], second["embed"]) == (0, 0, 0)
    assert FakeConverter.calls == 1

    # DB lost (e.g. reset) but stage output on disk: parse is not redone
    session.execute(delete(Source).where(Source.sha256 == files[0].sha256))
    session.commit()
    third = pipeline.ingest(
        session, files, cfg, FakeGateway(), metadata_lookup=False, extract=False
    )
    assert FakeConverter.calls == 1
    assert (third["chunk"], third["embed"]) == (1, 1)

    src = session.scalar(select(Source).where(Source.sha256 == files[0].sha256))
    chunks = session.scalars(select(Chunk).where(Chunk.source_id == src.id)).all()
    assert len(chunks) == 2
    assert all(c.locator.get("page") and c.embedding is not None for c in chunks)
    assert all(c.context_header.startswith("Paper: Fake Paper") for c in chunks)

    hits = hybrid_search(
        session, FakeGateway(), "score models?", SearchFilters(source_ids=[src.id]), top_k=5
    )
    assert hits and "fts" in hits[0].ranks
    assert "Score-based" in hits[0].text


def test_moved_file_updates_path_without_rework(setup, tmp_path):
    session, cfg, pdf = setup
    pipeline.ingest(
        session, detect([pdf]), cfg, FakeGateway(), metadata_lookup=False, extract=False
    )
    moved = pdf.with_name("fake paper.pdf").parent / "sub"
    moved.mkdir()
    new_path = moved / "fake paper.pdf"
    pdf.rename(new_path)
    report = pipeline.ingest(
        session, detect([new_path]), cfg, FakeGateway(), metadata_lookup=False, extract=False
    )
    assert (report["parse"], report["chunk"], report["embed"]) == (0, 0, 0)
    src = session.scalar(select(Source).where(Source.path.like("%sub/fake paper.pdf")))
    assert src is not None
