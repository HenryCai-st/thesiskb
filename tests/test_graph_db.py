"""Graph-lite lookups and person chunk selection on the test database."""

import pytest
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session

from tests.conftest import requires_db
from thesiskb.answer.paths import person_chunk_ids
from thesiskb.core.models import (
    Chunk,
    Entity,
    EntityMention,
    Person,
    Relation,
    Source,
    Speaker,
)
from thesiskb.retrieval.graph import entity_evidence, neighbours

pytestmark = [pytest.mark.db, requires_db]


@pytest.fixture
def kb(db_url):
    engine = create_engine(db_url)
    with Session(engine) as s:
        video = Source(
            sha256="v" * 64,
            path="/x/Introducing-NVIDIA-Jetson.mp4",
            modality="video",
            title="Introducing NVIDIA Jetson",
        )
        audio = Source(
            sha256="a" * 64,
            path="/x/DGX Spark.mp3",
            modality="audio",
            title="NVIDIA DGX Spark Has Arrived",
        )
        paper = Source(
            sha256="p" * 64,
            path="/x/p.pdf",
            modality="pdf",
            source_type="paper",
            authors=["Ada Lovelace"],
        )
        s.add_all([video, audio, paper])
        s.flush()
        cv = Chunk(
            source_id=video.id,
            text="[1:20] SPEAKER_00: $249",
            locator={"t_start": 80, "speakers": ["SPEAKER_00"]},
        )
        ca = Chunk(
            source_id=audio.id,
            text="[0:22] Will: I got a Spark",
            locator={"t_start": 22, "speakers": ["SPEAKER_03"]},
        )
        ca2 = Chunk(
            source_id=audio.id,
            text="[1:00] Other: hi",
            locator={"t_start": 60, "speakers": ["SPEAKER_04"]},
        )
        cp = Chunk(source_id=paper.id, text="A paper", locator={"page": 1})
        s.add_all([cv, ca, ca2, cp])
        jetson = Entity(
            name="Jetson Orin Nano Super", type="product", aliases=[], key="jetson orin nano super"
        )
        spark = Entity(name="DGX Spark", type="product", aliases=[], key="dgx spark")
        llms = Entity(name="LLMs", type="concept", aliases=[], key="llms")
        s.add_all([jetson, spark, llms])
        s.flush()
        s.add_all(
            [
                EntityMention(entity_id=jetson.id, chunk_id=cv.id),
                EntityMention(entity_id=spark.id, chunk_id=ca.id),
                Relation(subj=spark.id, pred="enables running", obj=llms.id, chunk_id=ca.id),
            ]
        )
        will = Person(name="Will", aliases=["Will I.M"])
        ada = Person(name="Ada Lovelace", aliases=[])
        s.add_all([will, ada])
        s.flush()
        s.add(Speaker(source_id=audio.id, diarization_label="SPEAKER_03", person_id=will.id))
        s.commit()
        yield (
            s,
            dict(jetson=jetson, spark=spark, llms=llms, will=will, ada=ada, ca=ca, cv=cv, cp=cp),
        )
        s.execute(delete(Source).where(Source.id.in_([video.id, audio.id, paper.id])))
        s.execute(delete(Entity).where(Entity.id.in_([jetson.id, spark.id, llms.id])))
        s.execute(delete(Person).where(Person.id.in_([will.id, ada.id])))
        s.commit()
    engine.dispose()


def test_products_found_across_sources_via_title_seed(kb):
    s, k = kb
    # "NVIDIA" is not an entity in this fixture, so there is no seed: every entity of the asked
    # type ("products") is a candidate, from both recordings
    ev = entity_evidence(s, "Which NVIDIA products appear in my sources?")
    names = {e.name for x in ev for e in x.entities}
    assert {"Jetson Orin Nano Super", "DGX Spark"} <= names
    spark_ev = next(x for x in ev if x.chunk_id == k["ca"].id)
    assert "DGX Spark enables running LLMs" in spark_ev.entities[0].facts


def test_neighbours_recursive(kb):
    s, k = kb
    assert k["llms"].id in neighbours(s, [k["spark"].id], hops=1)
    assert neighbours(s, [], hops=2) == set()


def test_person_chunks_from_speaker_labels_and_authorship(kb):
    s, k = kb
    assert person_chunk_ids(s, k["will"]) == [k["ca"].id]  # not the other speaker's chunk
    assert person_chunk_ids(s, k["ada"]) == [k["cp"].id]  # authored paper
