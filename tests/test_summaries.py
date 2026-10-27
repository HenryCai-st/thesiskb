"""Whole-document questions: part splitting, summary chunks, the summary path's merge/coverage."""

import uuid

import pytest

from thesiskb.answer.generate import Answer, Citation, Sentence, locator_label
from thesiskb.answer.paths import Ctx, coverage, merge
from thesiskb.core.models import Chunk, Source
from thesiskb.ingest import summarize as summ
from thesiskb.retrieval import router


def _chunk(header, words=300, **loc):
    return Chunk(id=uuid.uuid4(), text="word " * words, context_header=header, locator=loc)


def test_document_parts_follow_top_level_sections_in_reading_order():
    src = Source(modality="epub", path="b.epub")
    chunks = [
        _chunk("Book: B › Chapter 2", part="Chapter 2", order=5),
        _chunk("Book: B › Chapter 1", part="Chapter 1", order=0),
        _chunk("Book: B › Chapter 1 › 1.1", part="Chapter 1 › 1.1", order=2),
        _chunk("Book: B › Copyright", words=20, part="Copyright", order=9),
    ]
    parts = summ.split_parts(src, chunks)
    assert [p.label for p in parts] == ["Chapter 1", "Chapter 2", "Copyright"]
    assert [len(p.chunks) for p in parts] == [2, 1, 1]


def test_short_sections_absorb_the_next_one():
    src = Source(modality="document", path="p.html")
    chunks = [
        _chunk("Web: P › Tools", words=30, part="Tools", order=0),
        _chunk("Web: P › Team", words=30, part="Team", order=1),
        _chunk("Web: P › Problem", part="Problem", order=2),
        _chunk("Web: P › Design", part="Design", order=3),
    ]
    parts = summ.split_parts(src, chunks)
    assert [p.label for p in parts] == ["Tools – Problem", "Design"]


def test_recordings_are_split_into_time_windows(monkeypatch):
    monkeypatch.setattr(summ, "WINDOW_TOKENS", 500)
    src = Source(modality="audio", path="m.mp3")
    chunks = [_chunk("Meeting: M", t_start=60 * i, t_end=60 * i + 59) for i in range(4)]
    parts = summ.split_parts(src, chunks)
    assert [p.label for p in parts] == ["0:00–1:59", "2:00–3:59"]


class FakeGateway:
    def __init__(self):
        self.prompts = []

    def role_for(self, role, local_only=False):
        return role

    def chat(self, messages, role="answer", **kw):
        self.prompts.append(messages[0]["content"])
        return f"summary #{len(self.prompts)}"


class FakeSession:
    def __init__(self, chunks):
        self.chunks, self.added = chunks, []

    def execute(self, stmt):
        pass

    def scalars(self, stmt):
        return iter(self.chunks)

    def add_all(self, rows):
        self.added += rows

    def flush(self):
        pass


def test_summarize_source_writes_part_and_document_summaries_and_caches(tmp_path):
    src = Source(id=uuid.uuid4(), modality="epub", path="b.epub", title="B", source_type="book")
    chunks = [
        _chunk("Book: B › Chapter 1", part="Chapter 1", order=0),
        _chunk("Book: B › Chapter 2", part="Chapter 2", order=1),
        _chunk("Book: B › Chapter 3", part="Chapter 3", order=2),
    ]
    for c in chunks:
        c.fts_config = "english"
    gw, session = FakeGateway(), FakeSession(chunks)
    n = summ.summarize_source(session, src, gw, tmp_path / "s.json")
    assert n == 4 and len(gw.prompts) == 4  # 3 parts + the whole book
    doc, *parts = session.added
    assert doc.locator["summary"] == "document" and doc.locator["parts"] == 3
    assert "[Chapter 2]" in gw.prompts[-1]  # the book summary sees every part, in order
    assert [p.locator["part"] for p in parts] == ["Chapter 1", "Chapter 2", "Chapter 3"]
    assert [p.locator["index"] for p in parts] == [0, 1, 2]
    assert (
        parts[1].locator["order"] == 1
        and parts[1].context_header == "Book: B › Chapter 2 › Summary"
    )

    gw2, session2 = FakeGateway(), FakeSession(chunks)
    summ.summarize_source(session2, src, gw2, tmp_path / "s.json")
    assert gw2.prompts == []  # unchanged text: everything from the cache


def test_short_sources_and_images_get_no_summary(tmp_path):
    gw = FakeGateway()
    img = Source(id=uuid.uuid4(), modality="image", path="p.jpg")
    assert summ.summarize_source(FakeSession([_chunk("x")]), img, gw, tmp_path / "a.json") == 0
    note = Source(id=uuid.uuid4(), modality="document", path="n.md")
    short = [_chunk("Document: N", words=50, order=0)]
    assert summ.summarize_source(FakeSession(short), note, gw, tmp_path / "b.json") == 0
    assert gw.prompts == []


@pytest.mark.parametrize(
    "question,is_summary",
    [
        ("Summarize me the book", True),
        ("Give me an overview of the survey paper", True),
        ("What is the meeting about?", True),
        ("Fasse das Buch zusammen", True),
        ("What methods does the survey list for faster sampling?", False),
        ("Summarize what Will.i.am says about portability", False),
    ],
)
def test_summary_rule(question, is_summary):
    assert ("summary" in router.rule_route(question, []).paths) is is_summary


def _cite(n, kind="chunk", **loc):
    return Citation(
        n=n, chunk_id=uuid.UUID(int=n), filename="f", title=None, context_header=None,
        locator=loc, verification_status="unverified", text=f"t{n}", kind=kind,
    )  # fmt: skip


def test_merge_keeps_every_part_summary_and_puts_them_first():
    parts = Ctx(
        cites=[_cite(100 + i, "summary", summary="section") for i in range(15)], keep_all=True
    )
    search = Ctx(cites=[_cite(i) for i in range(1, 11)])
    out = merge([search, parts])
    assert sum(c.kind == "summary" for c in out) == 15  # more than MAX_PASSAGES, none dropped
    assert out[0].kind == "summary" and sum(c.kind == "chunk" for c in out) >= 3


def test_coverage_lists_parts_the_answer_did_not_cite():
    ctx = [
        _cite(1, "summary", summary="document", part="whole book"),
        _cite(2, "summary", summary="section", part="Chapter 1"),
        _cite(3, "summary", summary="section", part="Chapter 2"),
        _cite(4, "summary", summary="section", part="Chapter 3"),
    ]
    ans = Answer(
        question="q",
        text="",
        context=ctx,
        sentences=[Sentence(text="a", citations=[1, 2]), Sentence(text="b", citations=[4])],
    )
    assert coverage(ans) == {"parts": 3, "cited": 2, "missing": ["Chapter 2"]}
    assert coverage(Answer(question="q", text="", context=[_cite(9)])) is None


def test_summary_locator_label():
    loc = {"summary": "section", "part": "2 Methods", "page": 3, "page_end": 9}
    assert locator_label(loc) == "summary of 2 Methods, pp. 3–9"
    assert locator_label({"summary": "document", "part": "whole book", "order": 0}) == (
        "summary of whole book"
    )
