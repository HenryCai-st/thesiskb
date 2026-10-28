"""SQLAlchemy ORM models for BUILD_GUIDE §6. Schema changes go through Alembic only."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIM = 1024


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class Base(DeclarativeBase):
    pass


class Source(Base):
    """One row per input file."""

    __tablename__ = "sources"

    id: Mapped[uuid.UUID] = _uuid_pk()
    sha256: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    modality: Mapped[str] = mapped_column(String(16), nullable=False)  # pdf|epub|image|audio|video
    source_type: Mapped[str | None] = mapped_column(String(32))  # paper|book|note|meeting|...
    title: Mapped[str | None] = mapped_column(Text)
    authors: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    published: Mapped[date | None] = mapped_column(Date)
    origin: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(16))
    verification_status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="unverified"
    )
    verified_by: Mapped[str | None] = mapped_column(Text)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    consent_note: Mapped[str | None] = mapped_column(Text)
    local_only: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    ingested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    pipeline_version: Mapped[str | None] = mapped_column(String(32))
    # a changed file's new version points at the old one until that is retired after ingest
    replaces: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("sources.id", ondelete="SET NULL")
    )
    # web pages: `origin` is the URL, this is when it was fetched; `person_id` = the person a
    # web lookup was about (removed by forget-person)
    retrieved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    person_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("people.id", ondelete="SET NULL")
    )

    __table_args__ = (Index("ix_sources_source_type", "source_type"),)


class Person(Base):
    __tablename__ = "people"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(Text, nullable=False)
    aliases: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    role: Mapped[str | None] = mapped_column(String(32))


class Speaker(Base):
    """Diarization label within one source, optionally mapped to a person."""

    __tablename__ = "speakers"

    id: Mapped[uuid.UUID] = _uuid_pk()
    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"), nullable=False
    )
    diarization_label: Mapped[str] = mapped_column(String(32), nullable=False)
    person_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("people.id", ondelete="SET NULL")
    )


class Segment(Base):
    """Normalized unit before chunking: page, transcript turn, image region, keyframe."""

    __tablename__ = "segments"

    id: Mapped[uuid.UUID] = _uuid_pk()
    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # page|turn|image|frame
    text: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    speaker_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("speakers.id", ondelete="SET NULL")
    )
    locator: Mapped[dict] = mapped_column(JSONB, nullable=False)


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = _uuid_pk()
    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"), nullable=False
    )
    segment_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, server_default="{}"
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    context_header: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    tsv: Mapped[str | None] = mapped_column(TSVECTOR)
    fts_config: Mapped[str] = mapped_column(String(16), nullable=False, server_default="simple")
    speaker_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("speakers.id", ondelete="SET NULL")
    )
    locator: Mapped[dict] = mapped_column(JSONB, nullable=False)

    __table_args__ = (
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        Index("ix_chunks_speaker_id", "speaker_id"),
        Index("ix_chunks_source_id", "source_id"),
    )


class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    aliases: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    key: Mapped[str] = mapped_column(Text, nullable=False)  # entity_key(name); unique
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))  # of the name

    __table_args__ = (
        Index("ix_entities_lower_name", func.lower(name)),
        Index("ux_entities_key", "key", unique=True),
    )


class EntityMerge(Base):
    """Log of automatic entity merges and of near-duplicates left as suggestions."""

    __tablename__ = "entity_merges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kept_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    other_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE")
    )
    other_name: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)  # merged | suggested
    reason: Mapped[str] = mapped_column(String(16), nullable=False)  # key|variant|similar|shared
    similarity: Mapped[float | None] = mapped_column(Float)
    shared_neighbours: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Relation(Base):
    __tablename__ = "relations"

    id: Mapped[uuid.UUID] = _uuid_pk()
    subj: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    pred: Mapped[str] = mapped_column(Text, nullable=False)
    obj: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"))
    confidence: Mapped[float | None] = mapped_column(Float)


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = _uuid_pk()
    description: Mapped[str] = mapped_column(Text, nullable=False)
    owner_person_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("people.id", ondelete="SET NULL")
    )
    owner_raw: Mapped[str | None] = mapped_column(Text)
    due_date: Mapped[date | None] = mapped_column(Date)
    due_raw: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="open")
    # chunk link is SET NULL on re-index and restored from evidence_quote; source_id stays
    source_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("chunks.id", ondelete="SET NULL")
    )
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE")
    )
    evidence_quote: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    confirmed: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    notes: Mapped[str | None] = mapped_column(Text)
    start_date: Mapped[date | None] = mapped_column(Date)
    estimate_hours: Mapped[float | None] = mapped_column(Float)
    # a plan's parts point at the goal they lead to; deleting the goal deletes them
    parent_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    origin: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="extracted"
    )  # extracted | manual | plan (a plan item is hidden until confirmed = accepted)

    __table_args__ = (
        Index("ix_tasks_due_date", "due_date"),
        Index("ix_tasks_source_id", "source_id"),
        Index("ix_tasks_parent_id", "parent_id"),
    )


class TaskProposal(Base):
    """A change to a task that extraction found in a newer source; applied only on accept."""

    __tablename__ = "task_proposals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    field: Mapped[str] = mapped_column(String(16), nullable=False)  # due_date | status
    value: Mapped[str] = mapped_column(Text, nullable=False)  # ISO date or "done"
    value_raw: Mapped[str | None] = mapped_column(Text)
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE")
    )
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("chunks.id", ondelete="SET NULL"))
    evidence_quote: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Statement(Base):
    """What a person says or prefers about a topic, with a verbatim evidence quote."""

    __tablename__ = "statements"

    id: Mapped[uuid.UUID] = _uuid_pk()
    person_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("people.id", ondelete="SET NULL")
    )
    person_raw: Mapped[str | None] = mapped_column(Text)
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    stance: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_quote: Mapped[str] = mapped_column(Text, nullable=False)
    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"), nullable=False
    )
    source_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("chunks.id", ondelete="SET NULL")
    )
    confidence: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        Index("ix_statements_person_id", "person_id"),
        Index("ix_statements_source_id", "source_id"),
    )


class EntityMention(Base):
    __tablename__ = "entity_mentions"

    entity_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), primary_key=True
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )

    __table_args__ = (Index("ix_entity_mentions_chunk_id", "chunk_id"),)


class Job(Base):
    """Pipeline job table: status per source per stage (no broker in v1)."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"), nullable=False
    )
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="pending")
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("ix_jobs_source_stage", "source_id", "stage", unique=True),)


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    git_sha: Mapped[str | None] = mapped_column(String(40))
    config_hash: Mapped[str | None] = mapped_column(String(64))
    results: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class AnswerTrace(Base):
    """One row per answered question: what was searched, what the model saw, how it was judged."""

    __tablename__ = "answer_traces"

    id: Mapped[uuid.UUID] = _uuid_pk()
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    channel: Mapped[str] = mapped_column(String(16), nullable=False)  # cli|api|chat|mcp|eval
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text)
    refused: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    route: Mapped[dict | None] = mapped_column(JSONB)
    retrieval: Mapped[list | None] = mapped_column(JSONB)  # per path: query, candidates, reranked
    context: Mapped[list | None] = mapped_column(JSONB)  # passages [S#] shown to the model
    sentences: Mapped[list | None] = mapped_column(JSONB)  # status + judge reason per claim
    models: Mapped[dict | None] = mapped_column(JSONB)
    timings: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (Index("ix_answer_traces_created_at", "created_at"),)


class QueueItem(Base):
    """One ingest request from the web UI/API, processed one at a time by the worker."""

    __tablename__ = "ingest_queue"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # ingest | reingest
    paths: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("sources.id", ondelete="SET NULL")
    )
    options: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="queued")
    progress: Mapped[dict | None] = mapped_column(JSONB)  # stage, file, steps done/total
    report: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("ix_ingest_queue_status", "status", "id"),)
