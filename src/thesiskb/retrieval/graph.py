"""Entity lookups over the `entities` / `relations` / `entity_mentions` tables (graph-lite, SQL).

For aggregation questions ("Which NVIDIA products appear in my sources?") similarity search has
nothing to match: no single passage is "a list of NVIDIA products". The extracted tables are.

  seeds       entities whose name occurs in the question ("NVIDIA")
  type        entity type the question asks for ("products" → product)
  candidates  entities of that type co-occurring with a seed (same source, or the seed named in
              the source title), plus seeds' neighbours up to `hops` relations away
              (recursive SQL)
  evidence    the chunks that mention the candidates, one passage per chunk listing the
              entities and their relation facts; the chunk text is what the judge checks

Decision (ADR 0007): this is enough at thesis scale; no LightRAG/Neo4j.
"""

from __future__ import annotations

import re
import uuid
from collections import defaultdict

from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select, text
from sqlalchemy.orm import Session

from thesiskb.core.models import Chunk, Entity, EntityMention, Relation, Source

TYPE_WORDS = {
    "product": r"products?|devices?|computers?|hardware|tools?|produkte?|geräte?",
    "method": r"methods?|techniques?|approach(es)?|algorithms?|methoden?|verfahren",
    "model": r"models?|architectures?|modelle?",
    "dataset": r"datasets?|benchmarks?|datensätze?",
    "metric": r"metrics?|measures?|scores?|metriken?",
    "organization": r"companies|company|organi[sz]ations?|universit(y|ies)|labs?|firmen?",
    "person": r"people|persons?|researchers?|authors?|speakers?|personen|leute",
    "concept": r"concepts?|ideas?|topics?|themen?|konzepte?",
}
_STOP = {
    "which",
    "what",
    "who",
    "appear",
    "my",
    "sources",
    "the",
    "and",
    "each",
    "one",
    "for",
    "is",
    "are",
    "in",
    "of",
    "all",
    "mentioned",
    "list",
    "a",
    "an",
    "it",
    "do",
    "does",
}


class EntityFact(BaseModel):
    entity_id: uuid.UUID
    name: str
    type: str
    facts: list[str] = Field(default_factory=list)  # "DGX Spark enables running LLMs"


class EntityEvidence(BaseModel):
    chunk_id: uuid.UUID
    entities: list[EntityFact]


def asked_type(question: str) -> str | None:
    for etype, pattern in TYPE_WORDS.items():
        if re.search(rf"\b({pattern})\b", question, re.I):
            return etype
    return None


def seed_entities(session: Session, question: str, limit: int = 5) -> list[Entity]:
    """Entities whose name (≥ 3 chars) appears in the question as a whole word."""
    words = {w.lower() for w in re.findall(r"[\w.+-]{3,}", question)} - _STOP
    if not words:
        return []
    rows = session.scalars(
        select(Entity)
        .where(func.lower(Entity.name).in_(words))
        .order_by(func.length(Entity.name).desc())
        .limit(limit)
    ).all()
    # multi-word names ("DGX Spark") as substrings of the question
    q = question.lower()
    multi = session.scalars(
        select(Entity)
        .where(Entity.name.contains(" "), func.strpos(q, func.lower(Entity.name)) > 0)
        .limit(limit)
    ).all()
    return list({e.id: e for e in [*multi, *rows]}.values())


def neighbours(session: Session, seed_ids: list[uuid.UUID], hops: int = 2) -> set[uuid.UUID]:
    """Entities within `hops` relations of the seeds (either direction), recursive SQL."""
    if not seed_ids:
        return set()
    rows = session.execute(
        text("""
            WITH RECURSIVE walk(id, depth) AS (
                SELECT unnest(CAST(:seeds AS uuid[])), 0
                UNION
                SELECT CASE WHEN r.subj = w.id THEN r.obj ELSE r.subj END, w.depth + 1
                FROM relations r JOIN walk w ON r.subj = w.id OR r.obj = w.id
                WHERE w.depth < :hops
            )
            SELECT DISTINCT id FROM walk
        """),
        {"seeds": [str(s) for s in seed_ids], "hops": hops},
    ).scalars()
    return set(rows)


def co_occurring(session: Session, seeds: list[Entity], etype: str) -> set[uuid.UUID]:
    """Entities of `etype` mentioned in a source that mentions a seed or has it in its title."""
    if not seeds:
        return set()
    seed_ids = [s.id for s in seeds]
    src_with_seed = (
        select(Chunk.source_id)
        .join(EntityMention, EntityMention.chunk_id == Chunk.id)
        .where(EntityMention.entity_id.in_(seed_ids))
    )
    titled = select(Source.id).where(
        or_(
            *[Source.title.ilike(f"%{s.name}%") for s in seeds],
            *[Source.path.ilike(f"%{s.name}%") for s in seeds],
        )
    )
    rows = session.scalars(
        select(Entity.id)
        .join(EntityMention, EntityMention.entity_id == Entity.id)
        .join(Chunk, Chunk.id == EntityMention.chunk_id)
        .where(
            Entity.type == etype,
            or_(Chunk.source_id.in_(src_with_seed), Chunk.source_id.in_(titled)),
        )
    )
    return set(rows)


def entity_evidence(
    session: Session, question: str, max_entities: int = 12, max_chunks: int = 8
) -> list[EntityEvidence]:
    seeds = seed_entities(session, question)
    etype = asked_type(question)
    ids: set[uuid.UUID] = set()
    if etype:
        ids |= co_occurring(session, seeds, etype)
        if not seeds:  # "Which datasets appear in my sources?" — all of that type
            ids |= set(
                session.scalars(select(Entity.id).where(Entity.type == etype).limit(max_entities))
            )
    hop_ids = neighbours(session, [s.id for s in seeds])
    if etype:  # keep only neighbours of the asked type (seeds' relations still feed facts)
        hop_ids = set(
            session.scalars(select(Entity.id).where(Entity.id.in_(hop_ids), Entity.type == etype))
        )
    ids |= hop_ids
    if not ids:
        ids = {s.id for s in seeds}
    if not ids:
        return []

    ents = {e.id: e for e in session.scalars(select(Entity).where(Entity.id.in_(ids)))}
    ranked = sorted(  # most-mentioned first
        ents,
        key=lambda i: (
            -(
                session.scalar(
                    select(func.count())
                    .select_from(EntityMention)
                    .where(EntityMention.entity_id == i)
                )
                or 0
            )
        ),
    )[:max_entities]

    by_chunk: dict[uuid.UUID, list[EntityFact]] = defaultdict(list)
    for eid in ranked:
        e = ents[eid]
        facts = [
            f"{e.name} {pred} {obj}"
            for pred, obj in session.execute(
                select(Relation.pred, Entity.name)
                .join(Entity, Entity.id == Relation.obj)
                .where(Relation.subj == eid)
                .limit(6)
            )
        ]
        chunk_ids = session.scalars(
            select(EntityMention.chunk_id).where(EntityMention.entity_id == eid).limit(2)
        ).all()
        for cid in chunk_ids:
            by_chunk[cid].append(EntityFact(entity_id=eid, name=e.name, type=e.type, facts=facts))
    evidence = [EntityEvidence(chunk_id=c, entities=es) for c, es in by_chunk.items()]
    evidence.sort(key=lambda ev: -len(ev.entities))
    return evidence[:max_chunks]
