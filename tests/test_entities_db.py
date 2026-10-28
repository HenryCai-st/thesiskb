"""Entity merging, near-duplicate resolution and the graph API on the test database."""

import math
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session

from tests.conftest import requires_db
from thesiskb.core.models import Chunk, Entity, EntityMention, EntityMerge, Relation, Source
from thesiskb.evaluation import entity_metrics
from thesiskb.ingest.entities import EntityIndex, resolve_entities

pytestmark = [pytest.mark.db, requires_db]

TAG = "m14test"  # every entity here starts with it, so the fixture can clean up


def _vec(angle: float) -> list[float]:
    """Unit vector in the first two dims: cosine similarity of two names = cos(angle diff)."""
    return [math.cos(angle), math.sin(angle)] + [0.0] * 1022


class FakeGateway:
    def __init__(self, angles: dict[str, float]):
        self.angles = angles
        self.calls: list[list[str]] = []
        self.other: dict[str, int] = {}  # names from other tests: one orthogonal axis each

    def role_for(self, role, local_only):
        return role

    def embed(self, texts, role="embedding"):
        self.calls.append(list(texts))
        return [self._one(t) for t in texts]

    def _one(self, name: str) -> list[float]:
        if name in self.angles:
            return _vec(self.angles[name])
        v = [0.0] * 1024
        v[2 + self.other.setdefault(name, len(self.other)) % 1022] = 1.0
        return v


@pytest.fixture
def engine(db_url):
    eng = create_engine(db_url)

    def clean():
        with Session(eng) as s:
            s.execute(delete(Entity).where(Entity.key.like(f"{TAG}%")))
            s.execute(delete(Source).where(Source.path.like(f"%{TAG}%")))
            s.commit()

    clean()
    yield eng
    clean()
    eng.dispose()


def _kb(s: Session) -> tuple[Source, list[Chunk]]:
    src = Source(sha256=uuid.uuid4().hex * 2, path=f"/x/{TAG}.pdf", modality="pdf", title="T")
    s.add(src)
    s.flush()
    chunks = [Chunk(source_id=src.id, text=f"passage {i}", locator={"page": i}) for i in (1, 2)]
    s.add_all(chunks)
    s.flush()
    return src, chunks


def test_index_finds_key_and_alias_and_creates_new(engine):
    with Session(engine) as s:
        idx = EntityIndex(s)
        a = idx.get(f"{TAG} Lake-Surtan", "place")
        assert idx.get(f"{TAG} lake surtan", "other") is a  # same key
        a.aliases = [f"{TAG} Surtan Lake"]
        s.flush()
        assert EntityIndex(s).get(f"{TAG} surtan lake", "other").id == a.id  # alias, new index
        s.commit()
        assert s.scalar(select(Entity).where(Entity.key == f"{TAG} lake surtan")).type == "place"


def test_resolve_merges_variants_suggests_lookalikes_and_skips_numbers(engine):
    names = {
        f"{TAG} GAN": 0.0,
        f"{TAG} GANs": 0.40,  # cos 0.92: spelling variant → merged
        f"{TAG} possible futures": 1.0,
        f"{TAG} probable futures": 1.38,  # cos 0.93, no shared neighbour → suggestion
        f"{TAG} Song [287]": 2.0,
        f"{TAG} Song [282]": 2.05,  # cos 0.999 but different numbers → left alone
    }
    with Session(engine) as s:
        _, (c1, c2) = _kb(s)
        idx = EntityIndex(s)
        ents = {n: idx.get(n, "concept") for n in names}
        gan, gans = ents[f"{TAG} GAN"], ents[f"{TAG} GANs"]
        # GAN: two mentions (kept); GANs: one mention, shared with GAN, plus a relation
        s.add_all(
            [
                EntityMention(entity_id=gan.id, chunk_id=c1.id),
                EntityMention(entity_id=gan.id, chunk_id=c2.id),
                EntityMention(entity_id=gans.id, chunk_id=c1.id),
                Relation(subj=gans.id, pred="improves", obj=ents[f"{TAG} Song [287]"].id),
                Relation(subj=gans.id, pred="is", obj=gan.id),  # becomes a self-loop: dropped
            ]
        )
        s.flush()
        rep = resolve_entities(s, FakeGateway(names))
        s.commit()
        assert rep.embedded >= 6 and (rep.merged, rep.suggested) == (1, 1)

        assert s.get(Entity, gans.id) is None
        s.refresh(gan)
        assert gan.aliases == [f"{TAG} GANs"]
        rels = s.scalars(select(Relation).where(Relation.subj == gan.id)).all()
        assert [(r.pred, r.obj == ents[f"{TAG} Song [287]"].id) for r in rels] == [
            ("improves", True)
        ]
        mentions = s.scalars(
            select(EntityMention.chunk_id).where(EntityMention.entity_id == gan.id)
        )
        assert sorted(map(str, mentions)) == sorted([str(c1.id), str(c2.id)])

        log = s.scalars(
            select(EntityMerge).where(EntityMerge.kept_id.in_([e.id for e in ents.values()]))
        ).all()
        merged = [(m.other_name, m.reason) for m in log if m.status == "merged"]
        suggested = [m.other_name for m in log if m.status == "suggested"]
        assert merged == [(f"{TAG} GANs", "variant")]
        assert len(suggested) == 1 and "futures" in suggested[0]
        # a re-ingest later finds the merged name through the alias
        assert EntityIndex(s).get(f"{TAG} gans", "other").id == gan.id

        gw = FakeGateway(names)
        assert resolve_entities(s, gw).embedded == 0 and gw.calls == []  # nothing new: no call


def test_graph_api_neighbourhood_sources_and_evidence(engine):
    from thesiskb.api import app as app_module

    with Session(engine) as s:
        src, (c1, c2) = _kb(s)
        idx = EntityIndex(s)
        a, b, c = (idx.get(f"{TAG} {n}", "method") for n in ("DDIM", "sampling speed", "DDPM"))
        s.add_all(
            [
                EntityMention(entity_id=a.id, chunk_id=c1.id),
                EntityMention(entity_id=a.id, chunk_id=c2.id),
                Relation(subj=a.id, pred="improves", obj=b.id, chunk_id=c2.id),
                Relation(subj=b.id, pred="limits", obj=c.id, chunk_id=c1.id),
            ]
        )
        s.commit()
        ids = {"a": str(a.id), "b": str(b.id), "c": str(c.id)}
        src_id = str(src.id)
        c1_id, c2_id = str(c1.id), str(c2.id)

    def session_override():
        with Session(engine) as s:
            yield s

    app_module.app.dependency_overrides[app_module.get_session] = session_override
    try:
        client = TestClient(app_module.app)
        hits = client.get("/api/entities", params={"q": f"{TAG} dd"}).json()
        assert [h["name"] for h in hits] == [f"{TAG} DDIM", f"{TAG} DDPM"]  # most mentions first

        one = client.get(f"/api/entities/{ids['a']}/graph", params={"hops": 1}).json()
        assert {n["id"] for n in one["nodes"]} == {ids["a"], ids["b"]}
        assert [n["seed"] for n in one["nodes"] if n["id"] == ids["a"]] == [True]
        (edge,) = one["edges"]
        assert (edge["source"], edge["target"], edge["label"]) == (ids["a"], ids["b"], "improves")
        evidence = client.get(f"/api/chunks/{edge['chunk_id']}").json()  # an edge's evidence
        assert evidence["text"] == "passage 2"

        two = client.get(f"/api/entities/{ids['a']}/graph", params={"hops": 2}).json()
        assert {n["id"] for n in two["nodes"]} == set(ids.values())
        assert all(e["chunk_id"] for e in two["edges"])

        detail = client.get(f"/api/entities/{ids['a']}").json()
        (srcs,) = detail["sources"]
        assert srcs["id"] == src_id and len(srcs["chunks"]) == 2
        # "sampling speed" is only named in a relation: its source is the relation's chunk
        (rel_src,) = client.get(f"/api/entities/{ids['b']}").json()["sources"]
        assert [c["id"] for c in rel_src["chunks"]] == sorted([c1_id, c2_id])
        assert client.get("/static/vendor/cytoscape-3.34.2.min.js").status_code == 200
    finally:
        app_module.app.dependency_overrides.clear()


def test_entity_metrics_counts_duplicates_and_wrong_merges(engine, tmp_path):
    gold = tmp_path / "q.yaml"
    gold.write_text(
        "questions: []\nentity_variants:\n"
        f"  same: [['{TAG} A', '{TAG} a-'], ['{TAG} B', '{TAG} Bs'], ['{TAG} gone', '{TAG} x']]\n"
        f"  different: [['{TAG} B', '{TAG} C'], ['{TAG} A', '{TAG} B']]\n",
        encoding="utf-8",
    )
    with Session(engine) as s:
        idx = EntityIndex(s)
        idx.get(f"{TAG} A", "other")
        idx.get(f"{TAG} B", "other")
        idx.get(f"{TAG} Bs", "other")  # B / Bs not merged → a duplicate
        c = idx.get(f"{TAG} C", "other")
        c.aliases = [f"{TAG} B"]  # an alias never hides another entity's own name
        s.flush()
        m = entity_metrics(s, gold)
        s.rollback()
    assert m["entity_variant_groups"] == 2  # A / a- (one key) and B / Bs; "gone" not in the KB
    assert m["entity_duplicates"] == [f"{TAG} B / {TAG} Bs"]
    assert m["entity_wrong_merges"] == []
