"""One entity per real-world thing: normalized keys, near-duplicate merges, merge log (ADR 0014).

key        lowercase (casefold), Unicode NFKC, hyphens/underscores → spaces, collapsed
           whitespace: "lake surtan" and "Lake-Surtan" are one entity, found by key at ingest
resolve    after an ingest, new entity names are embedded (bge-m3) and compared with their
           nearest names. A pair is merged automatically when
             - the names differ only in plural endings or spaces ("GAN" / "GANs",
               "Auto-Encoder" / "Autoencoder") and similarity ≥ 0.80
             - similarity ≥ 0.95
             - similarity ≥ 0.93 and the two share a relation neighbour
           and listed as a suggestion when similarity ≥ 0.90. Names with different numbers
           ("Song et al. [287]" / "[282]", "Figure 2" / "Figure 4") are never paired.
           Thresholds come from the pairs in the thesis KB (ADR 0014): above 0.95 they were
           all spelling, plural or translation variants; "possible" / "probable futures" (0.93,
           no shared neighbour) are different terms.
merge      mentions and relations move to the kept entity (the more-mentioned one), the
           other name becomes an alias, and the merge is logged in `entity_merges`
"""

from __future__ import annotations

import logging
import re
import unicodedata
import uuid
from collections.abc import Callable

from pydantic import BaseModel
from sqlalchemy import delete, func, or_, select, text, update
from sqlalchemy.orm import Session

from thesiskb.core.models import Entity, EntityMention, EntityMerge, Relation
from thesiskb.llm.gateway import LLMGateway

log = logging.getLogger(__name__)

AUTO = 0.95
AUTO_SHARED = 0.93  # with at least one shared relation neighbour
VARIANT = 0.80  # same words up to plural endings and spaces
SUGGEST = 0.90
NEAREST = 8
EMBED_BATCH = 64


def entity_key(name: str) -> str:
    k = unicodedata.normalize("NFKC", name).casefold()
    k = re.sub(r"[-_‐-―]+", " ", k)
    return re.sub(r"\s+", " ", k).strip()


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("sses", "xes", "ches", "shes")):
        return word[:-2]
    if len(word) > 2 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _variant_key(name: str) -> str:
    """Singular words, no spaces: "Variational Auto-Encoders" ~ "variational autoencoder"."""
    return "".join(_singular(w) for w in re.findall(r"\w+", entity_key(name)))


def decide(a: str, b: str, similarity: float, shared: int) -> tuple[str, str] | None:
    """(status, reason) for a candidate pair, or None to leave it alone."""
    if re.findall(r"\d+", a) != re.findall(r"\d+", b):
        return None
    if similarity >= VARIANT and _variant_key(a) == _variant_key(b):
        return "merged", "variant"
    if similarity >= AUTO:
        return "merged", "similar"
    if similarity >= AUTO_SHARED and shared > 0:
        return "merged", "shared"
    if similarity >= SUGGEST:
        return "suggested", "similar"
    return None


class EntityIndex:
    """Entities by key and alias key, for get-or-create during extraction."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.by_key: dict[str, Entity] = {}
        entities = session.scalars(select(Entity)).all()
        for e in entities:
            for name in e.aliases or []:
                self.by_key.setdefault(entity_key(name), e)
        for e in entities:
            self.by_key[e.key] = e  # an entity's own key wins over another's alias

    def get(self, name: str, etype: str) -> Entity:
        key = entity_key(name)
        ent = self.by_key.get(key)
        if ent is None:
            ent = Entity(name=name.strip(), type=etype, aliases=[], key=key)
            self.session.add(ent)
            self.session.flush()
            self.by_key[key] = ent
        return ent


def mention_count(session: Session, entity_id: uuid.UUID) -> int:
    return (
        session.scalar(
            select(func.count())
            .select_from(EntityMention)
            .where(EntityMention.entity_id == entity_id)
        )
        or 0
    )


def shared_neighbours(session: Session, a: uuid.UUID, b: uuid.UUID) -> int:
    return (
        session.scalar(
            text("""
            SELECT count(*) FROM (
                SELECT CASE WHEN subj = :a THEN obj ELSE subj END FROM relations
                WHERE (subj = :a OR obj = :a) AND subj <> obj
                INTERSECT
                SELECT CASE WHEN subj = :b THEN obj ELSE subj END FROM relations
                WHERE (subj = :b OR obj = :b) AND subj <> obj
            ) n
        """),
            {"a": a, "b": b},
        )
        or 0
    )


def merge_entities(
    session: Session,
    keep: Entity,
    drop: Entity,
    reason: str,
    similarity: float | None = None,
    shared: int = 0,
) -> None:
    session.execute(
        text(
            "INSERT INTO entity_mentions (entity_id, chunk_id) SELECT :keep, chunk_id "
            "FROM entity_mentions WHERE entity_id = :drop ON CONFLICT DO NOTHING"
        ),
        {"keep": keep.id, "drop": drop.id},
    )
    session.execute(update(Relation).where(Relation.subj == drop.id).values(subj=keep.id))
    session.execute(update(Relation).where(Relation.obj == drop.id).values(obj=keep.id))
    session.execute(delete(Relation).where(Relation.subj == keep.id, Relation.obj == keep.id))
    # earlier merges into `drop` now point at `keep` (the FK would delete them with `drop`)
    session.execute(
        update(EntityMerge)
        .where(EntityMerge.kept_id == drop.id, EntityMerge.status == "merged")
        .values(kept_id=keep.id)
    )
    keep.aliases = sorted({*(keep.aliases or []), drop.name, *(drop.aliases or [])} - {keep.name})
    session.add(
        EntityMerge(
            kept_id=keep.id,
            other_name=drop.name,
            status="merged",
            reason=reason,
            similarity=similarity,
            shared_neighbours=shared,
        )
    )
    session.delete(drop)
    session.flush()


class ResolveReport(BaseModel):
    embedded: int = 0
    merged: int = 0
    suggested: int = 0


def resolve_entities(
    session: Session, gateway: LLMGateway, before_merge: Callable[[], object] | None = None
) -> ResolveReport:
    """Embed names that have no vector yet and merge or suggest their near-duplicates.
    `before_merge` runs once, before the first merge (the pipeline saves a graph snapshot)."""
    rep = ResolveReport()
    new = session.scalars(select(Entity).where(Entity.embedding.is_(None))).all()
    role = gateway.role_for("embedding", True)  # entity names may come from local-only sources
    for i in range(0, len(new), EMBED_BATCH):
        batch = new[i : i + EMBED_BATCH]
        for e, v in zip(batch, gateway.embed([e.name for e in batch], role=role), strict=True):
            e.embedding = v
    session.flush()
    rep.embedded = len(new)

    pairs: dict[frozenset, tuple[float, uuid.UUID, uuid.UUID]] = {}
    for e in new:
        dist = Entity.embedding.cosine_distance(e.embedding)
        for other, d in session.execute(
            select(Entity.id, dist)
            .where(Entity.id != e.id, Entity.embedding.is_not(None))
            .order_by(dist)
            .limit(NEAREST)
        ):
            if 1 - d >= VARIANT:
                pairs[frozenset((e.id, other))] = (1 - d, e.id, other)

    gone: set[uuid.UUID] = set()
    for sim, a_id, b_id in sorted(pairs.values(), key=lambda p: -p[0]):
        if a_id in gone or b_id in gone:
            continue
        a, b = session.get(Entity, a_id), session.get(Entity, b_id)
        shared = shared_neighbours(session, a.id, b.id)
        verdict = decide(a.name, b.name, sim, shared)
        if verdict is None:
            continue
        status, reason = verdict
        keep, drop = sorted(
            (a, b), key=lambda x: (-mention_count(session, x.id), len(x.name), x.name)
        )
        if status == "merged":
            if before_merge and not rep.merged:
                before_merge()
            merge_entities(session, keep, drop, reason, sim, shared)
            gone.add(drop.id)
            rep.merged += 1
        elif not session.scalar(
            select(EntityMerge.id).where(
                EntityMerge.status == "suggested",
                or_(
                    (EntityMerge.kept_id == a.id) & (EntityMerge.other_id == b.id),
                    (EntityMerge.kept_id == b.id) & (EntityMerge.other_id == a.id),
                ),
            )
        ):
            session.add(
                EntityMerge(
                    kept_id=keep.id,
                    other_id=drop.id,
                    other_name=drop.name,
                    status="suggested",
                    reason=reason,
                    similarity=sim,
                    shared_neighbours=shared,
                )
            )
            rep.suggested += 1
    session.flush()
    if rep.merged or rep.suggested:
        log.info("entities: %d merged, %d suggested", rep.merged, rep.suggested)
    return rep
