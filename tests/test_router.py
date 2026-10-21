"""Router rules, LLM merge and passage merging (LLM faked, no DB)."""

import json
import uuid

import pytest

from thesiskb.answer.generate import Citation
from thesiskb.answer.paths import Ctx, merge
from thesiskb.core.models import Person
from thesiskb.retrieval import router
from thesiskb.retrieval.graph import asked_type

WILL = Person(id=uuid.uuid4(), name="Will.i.am", aliases=["Will I.M"])
PROF = Person(id=uuid.uuid4(), name="Prof. Example", aliases=["Example"])


@pytest.mark.parametrize(
    "question,paths,person",
    [
        ("Show me the schedule of my thesis", ["task"], None),
        ("Wann ist meine Frist für die Abgabe?", ["task"], None),
        ("What does Will I.M say about portability?", ["person"], "Will I.M"),
        ("What did Prof. Example say about my deadline?", ["person", "task"], "Prof. Example"),
        ("Which NVIDIA products appear in my sources?", ["entity"], None),
        ("How is diffusion model developing?", ["general"], None),
        ("An example of a diffusion model?", ["person"], "Example"),  # alias is a common word
    ],
)
def test_rule_route(question, paths, person):
    # last case: a person alias that is a common word ("Example") matches anywhere — a known
    # limitation of name rules (ADR 0007); keep such aliases out of people.yaml
    r = router.rule_route(question, [WILL, PROF])
    assert r.paths == paths
    assert r.person == person


def test_find_person_prefers_longest_match():
    p, name = router.find_person("what does prof. example think", [WILL, PROF])
    assert p is PROF and name == "Prof. Example"


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def chat(self, messages, role="answer", **kw):
        assert role == "router"
        return self.payload if isinstance(self.payload, str) else json.dumps(self.payload)


def test_llm_route_filters_unknown_paths():
    out = router.llm_route(FakeLLM({"paths": ["literature", "magic", "literature"]}), "q", [])
    assert out.paths == ["literature"]
    assert router.llm_route(FakeLLM("not json"), "q", []) is None
    assert router.llm_route(FakeLLM({"paths": ["magic"]}), "q", []) is None


@pytest.mark.parametrize(
    "q,etype",
    [
        ("Which NVIDIA products appear?", "product"),
        ("List all datasets mentioned", "dataset"),
        ("Which methods speed up sampling?", "method"),
        ("Who spoke?", None),
    ],
)
def test_asked_type(q, etype):
    assert asked_type(q) == etype


def _cite(i, kind="chunk", text="t"):
    return Citation(
        n=i,
        chunk_id=uuid.UUID(int=i),
        filename="f",
        title=None,
        context_header=None,
        locator={},
        verification_status="unverified",
        text=text,
        kind=kind,
    )


def test_merge_renumbers_dedupes_and_balances_paths():
    a = Ctx(cites=[_cite(i) for i in range(1, 11)])
    b = Ctx(cites=[_cite(1), _cite(50, kind="task", text="row")])  # first is a duplicate of a's
    out = merge([a, b], limit=8)
    assert [c.n for c in out] == list(range(1, len(out) + 1))
    assert sum(c.kind == "task" for c in out) == 1  # second path not crowded out
    assert len({(c.chunk_id, c.kind) for c in out}) == len(out)
