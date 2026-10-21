"""LLM extraction over chunks: tasks, person statements, entities and relations.

One structured-output call per chunk (role `extraction`), with the fields that make sense for the
source type: tasks only from notes/meetings/interviews, statements only from recordings with
speakers, entities and relations from everything. Every extracted row must carry an
evidence_quote that appears verbatim (modulo whitespace/case) in its chunk, or it is dropped.

Results are cached per chunk text in data/processed/<sha>/extract.json, so re-indexing a source
only re-runs the LLM for chunks whose text changed. Writing to the DB replaces the source's
unconfirmed tasks and its statements; confirmed tasks are kept and re-linked to the chunk that
contains their evidence quote.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, create_model
from sqlalchemy import any_, delete, func, select
from sqlalchemy.orm import Session

from thesiskb.core.models import (
    Chunk,
    Entity,
    EntityMention,
    Person,
    Relation,
    Source,
    Statement,
    Task,
)
from thesiskb.ingest.dates import anchor_for, resolve_due
from thesiskb.llm.gateway import LLMGateway

log = logging.getLogger(__name__)

TASK_SOURCES = {"note", "meeting", "interview"}
STATEMENT_MODALITIES = {"audio", "video"}
EntityType = Literal[
    "method",
    "model",
    "dataset",
    "metric",
    "concept",
    "product",
    "organization",
    "person",
    "place",
    "other",
]


class XTask(BaseModel):
    description: str
    owner: str | None = None
    due: str | None = None
    evidence_quote: str
    confidence: float = Field(0.8, ge=0, le=1)


class XStatement(BaseModel):
    person: str
    topic: str
    stance: str
    evidence_quote: str
    confidence: float = Field(0.8, ge=0, le=1)


class XEntity(BaseModel):
    name: str
    type: EntityType = "other"


class XRelation(BaseModel):
    subject: str
    predicate: str
    object: str
    evidence_quote: str
    confidence: float = Field(0.8, ge=0, le=1)


class Extraction(BaseModel):
    tasks: list[XTask] = Field(default_factory=list)
    statements: list[XStatement] = Field(default_factory=list)
    entities: list[XEntity] = Field(default_factory=list)
    relations: list[XRelation] = Field(default_factory=list)


PROMPT = """Extract structured facts from one passage of the user's private knowledge base.
Source: {header}

Return JSON with exactly these keys:
{fields}

Rules:
- evidence_quote: copy a short span from the passage EXACTLY (same words, same order), at most \
one or two sentences. Do not paraphrase it.
- Only extract what the passage states. If there is nothing for a key, return an empty list.
- Keep dates exactly as written ("Nov", "Jan 2027", "05.2027", "next Friday").
- confidence: 0-1, how sure you are the item is stated in the passage.

Passage:
{text}"""

FIELD_DOCS = {
    "tasks": '"tasks": every to-do, deadline, milestone, appointment or meeting — each dated line '
    "is its own item: {description, owner (person responsible as written, or null), due (date as "
    "written, or null), evidence_quote, confidence}",
    "statements": '"statements": opinions, preferences, recommendations or claims a named '
    "speaker makes: {person (speaker name as in the passage), topic (a few words), stance (what "
    "they say/prefer, one sentence), evidence_quote, confidence}",
    "entities": '"entities": important named things — methods, models, datasets, metrics, '
    "concepts, products, organizations, people: {name, type: one of method|model|dataset|metric|"
    "concept|product|organization|person|place|other}. At most 12.",
    "relations": '"relations": explicit relations between those entities, e.g. "DDIM - improves '
    '- sampling speed": {subject, predicate (2-4 words), object, evidence_quote, confidence}. '
    "At most 8.",
}


# --- plug-in extractors ---------------------------------------------------------------------


class Extractor(ABC):
    """An extra extraction schema (BUILD_GUIDE §8 "Extractor per schema").

    Its field is added to the same per-chunk LLM call as the built-ins (no extra calls), items
    with an `evidence_quote` not found verbatim in the chunk are dropped, and `write` stores the
    rest — it is called once per source with every chunk's items and should replace what it
    stored for that source before. Example: a "claims" extractor writing to its own table, or
    to a JSON file under data/processed/<sha>/.

        class Claims(Extractor):
            key = "claims"
            doc = '"claims": factual claims with numbers: {claim, value, evidence_quote}'
            item = Claim  # pydantic model with an evidence_quote field

            def applies(self, src): return src.source_type == "paper"
            def write(self, session, src, items): ...

        register_extractor(Claims())
    """

    key: str = ""
    doc: str = ""
    item: type[BaseModel]
    needs_evidence: bool = True

    def applies(self, src: Source) -> bool:
        return True

    @abstractmethod
    def write(self, session: Session, src: Source, items: list[tuple[Chunk, list]]) -> int:
        """Store the verified items; return how many were stored."""


_EXTRACTORS: dict[str, Extractor] = {}


def register_extractor(ex: Extractor) -> Extractor:
    if ex.key in Extraction.model_fields:
        raise ValueError(f"'{ex.key}' is a built-in extraction field")
    _EXTRACTORS[ex.key] = ex
    _model_for.cache_clear()
    return ex


@lru_cache
def _model_for(plugin_keys: tuple[str, ...]) -> type[Extraction]:
    if not plugin_keys:
        return Extraction
    extra = {k: (list[_EXTRACTORS[k].item], Field(default_factory=list)) for k in plugin_keys}
    return create_model("ExtractionWithPlugins", __base__=Extraction, **extra)


def extraction_model(fields: list[str]) -> type[Extraction]:
    """The response schema for these fields: built-ins + any plug-in extractors among them."""
    return _model_for(tuple(f for f in fields if f in _EXTRACTORS))


def fields_for(src: Source) -> list[str]:
    fields = []
    if (src.source_type or "") in TASK_SOURCES:
        fields.append("tasks")
    if src.modality in STATEMENT_MODALITIES:
        fields.append("statements")
    plugins = [k for k, ex in _EXTRACTORS.items() if ex.applies(src)]
    return [*fields, "entities", "relations", *plugins]


def _norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"[\s\-–—•*]+", " ", s).strip(" .\"'")


def quote_in(quote: str, text: str) -> bool:
    """Evidence must appear verbatim (whitespace, case, bullets and dashes normalised)."""
    q = _norm(quote)
    return bool(q) and q in _norm(text)


def _cache_key(chunk: Chunk, fields: list[str]) -> str:
    return hashlib.sha1(("|".join(fields) + "\n" + chunk.text).encode()).hexdigest()


def extract_chunk(
    gateway: LLMGateway, chunk: Chunk, fields: list[str], role: str = "extraction"
) -> Extraction:
    prompt = PROMPT.format(
        header=chunk.context_header or "",
        fields="\n".join(FIELD_DOCS.get(f) or _EXTRACTORS[f].doc for f in fields),
        text=chunk.text,
    )
    model = extraction_model(fields)
    raw = gateway.chat([{"role": "user", "content": prompt}], role=role, response_format=model)
    try:
        out = model.model_validate(json.loads(raw))
    except (ValidationError, ValueError) as e:
        log.warning("unparsable extraction for chunk %s: %s", chunk.id, e)
        return model()
    for f in set(model.model_fields) - set(fields):
        setattr(out, f, [])
    return out


def verified(x: Extraction, text: str) -> tuple[Extraction, int]:
    """Drop items whose evidence quote is not in the chunk. Returns (kept, n_dropped)."""
    checked = ["tasks", "statements", "relations"] + [
        k for k, ex in _EXTRACTORS.items() if ex.needs_evidence and k in type(x).model_fields
    ]
    update = {
        k: [i for i in getattr(x, k) if quote_in(getattr(i, "evidence_quote", "") or "", text)]
        for k in checked
    }
    kept = x.model_copy(update=update)
    dropped = sum(len(getattr(x, k)) - len(update[k]) for k in checked)
    return kept, dropped


def resolve_person(session: Session, name: str | None) -> Person | None:
    if not name:
        return None
    n = name.strip().lower()
    for p in session.scalars(select(Person)):
        if n == p.name.lower() or n in {a.lower() for a in p.aliases or []}:
            return p
    return None


def _entity(session: Session, cache: dict[str, Entity], name: str, etype: str) -> Entity:
    key = name.strip().lower()
    if key in cache:
        return cache[key]
    ent = session.scalar(
        select(Entity).where(
            (func.lower(Entity.name) == key) | (name.strip() == any_(Entity.aliases))
        )
    )
    if ent is None:
        ent = Entity(name=name.strip(), type=etype, aliases=[])
        session.add(ent)
        session.flush()
    cache[key] = ent
    return ent


class ExtractReport(BaseModel):
    chunks: int = 0
    llm_calls: int = 0
    tasks: int = 0
    statements: int = 0
    entities: int = 0
    relations: int = 0
    dropped_no_evidence: int = 0
    plugins: dict[str, int] = Field(default_factory=dict)  # plug-in key → rows stored


def extract_source(
    session: Session,
    src: Source,
    gateway: LLMGateway,
    cache_path: Path,
    reference_date: date | None = None,
) -> ExtractReport:
    fields = fields_for(src)
    chunks = session.scalars(select(Chunk).where(Chunk.source_id == src.id)).all()
    cache: dict[str, dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    rep = ExtractReport(chunks=len(chunks))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[tuple[Chunk, Extraction]] = []
    for i, c in enumerate(chunks, 1):
        key = _cache_key(c, fields)
        if key in cache:
            x = extraction_model(fields).model_validate(cache[key])
        else:
            x = extract_chunk(gateway, c, fields, gateway.role_for("extraction", src.local_only))
            cache[key] = x.model_dump()
            rep.llm_calls += 1
            # save after every call: an interrupted run on a long paper resumes where it stopped
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            if rep.llm_calls % 10 == 0:
                log.info("extract: %d/%d chunks", i, len(chunks))
        x, dropped = verified(x, c.text)
        rep.dropped_no_evidence += dropped
        results.append((c, x))

    _write(session, src, results, reference_date or _source_date(src), rep)
    for k in fields:
        if k in _EXTRACTORS:
            stored = _EXTRACTORS[k].write(session, src, [(c, getattr(x, k)) for c, x in results])
            rep.plugins[k] = stored
    return rep


def _source_date(src: Source) -> date:
    if src.published:
        return src.published
    if src.ingested_at:
        return src.ingested_at.date()
    return date.today()


def _write(session, src: Source, results, ref: date, rep: ExtractReport) -> None:
    # tasks: replace unconfirmed ones, keep confirmed ones (re-linked below)
    session.execute(delete(Task).where(Task.source_id == src.id, Task.confirmed.is_(False)))
    session.execute(delete(Statement).where(Statement.source_id == src.id))
    chunk_ids = [c.id for c, _ in results]
    if chunk_ids:
        session.execute(delete(Relation).where(Relation.chunk_id.in_(chunk_ids)))
        session.execute(delete(EntityMention).where(EntityMention.chunk_id.in_(chunk_ids)))

    confirmed = session.scalars(
        select(Task).where(Task.source_id == src.id, Task.confirmed.is_(True))
    ).all()
    for t in confirmed:  # re-link to the chunk that now holds the evidence
        t.source_chunk_id = next(
            (c.id for c, _ in results if t.evidence_quote and quote_in(t.evidence_quote, c.text)),
            t.source_chunk_id,
        )
    have = {_norm(t.description) for t in confirmed}

    anchor = anchor_for([t.due for _, x in results for t in x.tasks], ref)
    ent_cache: dict[str, Entity] = {}
    for c, x in results:
        for t in x.tasks:
            if _norm(t.description) in have:
                continue
            have.add(_norm(t.description))
            owner = resolve_person(session, t.owner)
            session.add(
                Task(
                    description=t.description.strip(),
                    owner_person_id=owner.id if owner else None,
                    owner_raw=t.owner,
                    due_date=resolve_due(t.due, anchor),
                    due_raw=t.due,
                    source_chunk_id=c.id,
                    source_id=src.id,
                    evidence_quote=t.evidence_quote,
                    confidence=t.confidence,
                )
            )
            rep.tasks += 1
        for s in x.statements:
            person = resolve_person(session, s.person)
            session.add(
                Statement(
                    person_id=person.id if person else None,
                    person_raw=s.person,
                    topic=s.topic,
                    stance=s.stance,
                    evidence_quote=s.evidence_quote,
                    source_id=src.id,
                    source_chunk_id=c.id,
                    confidence=s.confidence,
                )
            )
            rep.statements += 1
        mentioned: set = set()
        for e in x.entities:
            if not e.name.strip():
                continue
            ent = _entity(session, ent_cache, e.name, e.type)
            if ent.id not in mentioned:
                session.add(EntityMention(entity_id=ent.id, chunk_id=c.id))
                mentioned.add(ent.id)
                rep.entities += 1
        for r in x.relations:
            subj = _entity(session, ent_cache, r.subject, "other")
            obj = _entity(session, ent_cache, r.object, "other")
            session.add(
                Relation(
                    subj=subj.id,
                    pred=r.predicate.strip(),
                    obj=obj.id,
                    chunk_id=c.id,
                    confidence=r.confidence,
                )
            )
            rep.relations += 1
    session.flush()
