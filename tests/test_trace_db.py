"""Answer traces: every answer (and every failed one) leaves a row that `tkb trace` can show."""

import uuid

import pytest
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session

from tests.conftest import requires_db
from thesiskb.answer import paths
from thesiskb.answer.generate import Answer, Citation, Sentence
from thesiskb.answer.trace import format_trace, get_trace, trace_out
from thesiskb.core.config import RetrievalConfig
from thesiskb.core.models import AnswerTrace

pytestmark = [pytest.mark.db, requires_db]

CITE = Citation(
    n=1,
    chunk_id=uuid.uuid4(),
    filename="book.epub",
    title="Book",
    context_header="Book: Chapter 2",
    locator={"page": 4},
    verification_status="unverified",
    text="Sampling can be accelerated with DDIM.",
)
RETRIEVAL = [
    {
        "query": "how to sample faster?",
        "candidates": [
            {
                "chunk_id": str(CITE.chunk_id),
                "file": "book.epub",
                "score": 0.03,
                "ranks": {"fts": 1},
            }
        ],
        "reranked": [
            {"chunk_id": str(CITE.chunk_id), "file": "book.epub", "score": 0.91, "ranks": {}}
        ],
        "kept": 1,
    }
]


class FakeGateway:
    class models:  # noqa: N801 - mimics gateway.models.role(name).model
        @staticmethod
        def role(name):
            return type("R", (), {"model": f"fake/{name}"})


@pytest.fixture
def session(db_url):
    engine = create_engine(db_url)
    with Session(engine) as s:
        s.execute(delete(AnswerTrace))
        s.commit()
        yield s
    engine.dispose()


@pytest.fixture
def fake_pipeline(monkeypatch):
    monkeypatch.setitem(
        paths.PROVIDERS,
        "general",
        lambda *a: paths.Ctx(cites=[CITE], retrieval=RETRIEVAL, timings={"search": 0.1}),
    )
    monkeypatch.setattr(
        paths,
        "generate_from",
        lambda gw, q, cites, extra_rules="": Answer(
            question=q,
            text="DDIM speeds up sampling [S1].",
            sentences=[Sentence(text="DDIM speeds up sampling [S1].", citations=[1])],
            context=cites,
        ),
    )


def test_answer_stores_queries_hits_passages_and_verdicts(session, fake_pipeline):
    ans = paths.answer(
        session, FakeGateway(), "how to sample faster?", RetrievalConfig(), path="general",
        verify_support=False, channel="api",
    )  # fmt: skip
    assert ans.trace_id and ans.retrieval[0]["path"] == "general"
    t = trace_out(get_trace(session, ans.trace_id[:8]))  # prefix works, like `tkb trace`
    assert t["channel"] == "api" and t["route"]["paths"] == ["general"]
    assert t["retrieval"][0]["reranked"][0]["score"] == 0.91
    assert t["context"][0]["filename"] == "book.epub"
    assert t["sentences"][0]["citations"] == [1]
    assert t["models"]["answer"] == "fake/answer"
    text = format_trace(t)
    assert "how to sample faster?" in text and "0.910  book.epub" in text
    assert "kept 1 for the answer" in text
    assert get_trace(session, "last").id == uuid.UUID(ans.trace_id)


def test_failed_answer_still_leaves_a_trace(session, monkeypatch):
    def boom(*a):
        raise RuntimeError("Ollama not reachable")

    monkeypatch.setitem(paths.PROVIDERS, "general", boom)
    with pytest.raises(RuntimeError):
        paths.answer(session, FakeGateway(), "q?", RetrievalConfig(), path="general")
    t = trace_out(get_trace(session, "last"))
    assert "Ollama not reachable" in t["error"] and t["route"]["paths"] == ["general"]


def test_unknown_or_ambiguous_trace_ref(session):
    assert get_trace(session, "zzz") is None
    assert get_trace(session, "last") is None
