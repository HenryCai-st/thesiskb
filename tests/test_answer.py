import json
import uuid

from thesiskb.answer.generate import (
    Answer,
    Citation,
    format_context,
    locator_label,
    split_answer,
    to_chunk_ids,
)
from thesiskb.answer.verify import check_citations, judge_support
from thesiskb.retrieval.hybrid import Hit, fts_query_terms, rrf_fuse


def _cite(n, text="passage text"):
    return Citation(
        n=n,
        chunk_id=uuid.uuid4(),
        filename="paper.pdf",
        title="T",
        context_header="Paper: T › 1 Intro",
        locator={"page": n},
        verification_status="unverified",
        text=text,
    )


def test_split_keeps_citations_and_abbreviations():
    s = split_answer(
        "Diffusion is popular [S1]. Watson et al. [317] optimise steps [S2][S3].\n"
        "- **Learning-free** includes:\n- DDIM [S1, S4]."
    )
    assert [x.citations for x in s] == [[1], [2, 3], [], [1, 4]]
    assert "et al. [317]" in s[1].text  # not split at "al.", paper ref not a citation
    assert s[2].status == "structural"


def test_bullets_inherit_citation_from_cited_lead_in():
    s = split_answer("Here is the schedule [S1]:\n- Nov: start\n- Jan: mid-term [S2]\nDone.")
    assert [x.citations for x in s] == [[1], [1], [2], []]
    assert s[1].note == "citation from list lead-in"
    # an uncited lead-in passes nothing on
    s = split_answer("Schedule:\n- Nov: start")
    assert s[1].citations == []


def test_dangling_citation_joins_previous_sentence():
    s = split_answer("Claim one.\n[S2]")
    assert len(s) == 1 and s[0].citations == [2]


def test_to_chunk_ids_and_invalid():
    cites = [_cite(1)]
    out = to_chunk_ids("A [S1]. B [S9].", cites)
    assert f"[chunk:{cites[0].chunk_id}]" in out and "[invalid:S9]" in out


def test_check_citations_flags():
    ans = Answer(question="q", text="", context=[_cite(1), _cite(2)])
    ans.sentences = split_answer("Good [S1]. Bad [S7]. Missing citation.")
    check_citations(ans)
    assert [s.status for s in ans.sentences] == ["unchecked", "invalid_citation", "uncited"]


class FakeGateway:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, role="answer", **kw):
        self.calls.append((role, messages))
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def test_judge_marks_unsupported_one_call_per_claim():
    ans = Answer(question="q", text="", context=[_cite(1, "alpha"), _cite(2, "beta")])
    ans.sentences = split_answer("True claim [S1]. Made-up claim [S2].")
    gw = FakeGateway(
        json.dumps({"supported": True}), json.dumps({"supported": False, "reason": "x"})
    )
    judge_support(gw, ans)
    assert [s.status for s in ans.sentences] == ["supported", "unsupported"]
    assert [role for role, _ in gw.calls] == ["judge", "judge"]
    # each call only sees the passages its claim cites
    assert "alpha" in gw.calls[0][1][0]["content"] and "beta" not in gw.calls[0][1][0]["content"]


def test_judge_garbage_leaves_unchecked():
    ans = Answer(question="q", text="", context=[_cite(1)])
    ans.sentences = split_answer("Claim [S1].")
    judge_support(FakeGateway("not json"), ans)
    assert ans.sentences[0].status == "unchecked"


def test_format_context_and_locators():
    ctx = format_context([_cite(1, "hello")])
    assert ctx.startswith("[S1] paper.pdf, p. 1 | Paper: T › 1 Intro\nhello")
    assert locator_label({"page": 3, "page_end": 5}) == "pp. 3–5"
    assert locator_label({"t_start": 65, "t_end": 130}) == "1:05–2:10"


def _hit(i):
    return Hit(
        chunk_id=uuid.UUID(int=i),
        source_id=uuid.UUID(int=0),
        text=f"t{i}",
        context_header=None,
        locator={"page": 1},
        source_path="/x/a.pdf",
        source_title=None,
        source_type="paper",
        verification_status="unverified",
    )


def test_rrf_prefers_items_in_both_lists():
    fused = rrf_fuse({"vector": [_hit(1), _hit(2)], "fts": [_hit(3), _hit(2)]}, k=60)
    assert fused[0].chunk_id == uuid.UUID(int=2)
    assert fused[0].ranks == {"vector": 2, "fts": 2}


def test_fts_query_terms():
    assert fts_query_terms("How is diffusion model developing?") == (
        "how | is | diffusion | model | developing"
    )
    assert fts_query_terms("?!") == ""


def test_quotes_are_not_split():
    s = split_answer('He says, "Now I got the Spark. I can take it! Right?" [S1]. Next [S2].')
    assert len(s) == 2
    assert s[0].citations == [1] and s[0].text.startswith('He says, "Now I got the Spark. I can')


def test_pinpoint_finds_line_and_frame():
    c = Citation(
        n=1,
        chunk_id=uuid.uuid4(),
        filename="v.mp4",
        title=None,
        context_header=None,
        locator={
            "t_start": 0,
            "t_end": 120,
            "frames": [{"t": 60, "frame": "f60.jpg"}, {"t": 90, "frame": "f90.jpg"}],
        },
        verification_status="unverified",
        text="[0:00] SPEAKER_00: Hi, welcome to my kitchen.\n\n"
        "[1:04] SPEAKER_00: Almost 70 trillion operations per second, 25 watts and $249.\n\n"
        "[1:30] (on screen) A Jetson box. Text: Orin Nano",
    )
    s = split_answer("It costs $249 at 25 watts [S1].")[0]
    from thesiskb.answer.generate import pinpoint

    pinpoint(s, [c])
    assert len(s.pinpoints) == 1
    p = s.pinpoints[0]
    assert (p.t, p.speaker, p.frame) == (64.0, "SPEAKER_00", "f60.jpg")


def test_numbered_list_items_stay_whole():
    s = split_answer(
        "Schedule [S1]:\n1. **Proposal** - due 10.10.2026 [S1]\n2. **Final** - March [S2]"
    )
    assert [x.text for x in s[1:]] == [
        "1. **Proposal** - due 10.10.2026 [S1]",
        "2. **Final** - March [S2]",
    ]
    assert [x.citations for x in s[1:]] == [[1], [2]]
