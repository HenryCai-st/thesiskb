"""Initial schema (BUILD_GUIDE §6).

Revision ID: 0001
Revises:
Create Date: 2026-10-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql as pg

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID = pg.UUID(as_uuid=True)


def _id() -> sa.Column:
    return sa.Column("id", UUID, primary_key=True)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "sources",
        _id(),
        sa.Column("sha256", sa.String(64), nullable=False, unique=True),
        sa.Column("path", sa.Text, nullable=False),
        sa.Column("modality", sa.String(16), nullable=False),
        sa.Column("source_type", sa.String(32)),
        sa.Column("title", sa.Text),
        sa.Column("authors", pg.ARRAY(sa.Text)),
        sa.Column("published", sa.Date),
        sa.Column("origin", sa.Text),
        sa.Column("language", sa.String(16)),
        sa.Column(
            "verification_status", sa.String(16), nullable=False, server_default="unverified"
        ),
        sa.Column("verified_by", sa.Text),
        sa.Column("verified_at", sa.DateTime(timezone=True)),
        sa.Column("consent_note", sa.Text),
        sa.Column("local_only", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("ingested_at", sa.DateTime(timezone=True)),
        sa.Column("pipeline_version", sa.String(32)),
    )
    op.create_index("ix_sources_source_type", "sources", ["source_type"])

    op.create_table(
        "people",
        _id(),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("aliases", pg.ARRAY(sa.Text), nullable=False, server_default="{}"),
        sa.Column("role", sa.String(32)),
    )

    op.create_table(
        "speakers",
        _id(),
        sa.Column(
            "source_id", UUID, sa.ForeignKey("sources.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("diarization_label", sa.String(32), nullable=False),
        sa.Column("person_id", UUID, sa.ForeignKey("people.id", ondelete="SET NULL")),
    )

    op.create_table(
        "segments",
        _id(),
        sa.Column(
            "source_id", UUID, sa.ForeignKey("sources.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("text", sa.Text, nullable=False, server_default=""),
        sa.Column("speaker_id", UUID, sa.ForeignKey("speakers.id", ondelete="SET NULL")),
        sa.Column("locator", pg.JSONB, nullable=False),
    )

    op.create_table(
        "chunks",
        _id(),
        sa.Column(
            "source_id", UUID, sa.ForeignKey("sources.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("segment_ids", pg.ARRAY(UUID), nullable=False, server_default="{}"),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("context_header", sa.Text),
        sa.Column("embedding", Vector(1024)),
        sa.Column("tsv", pg.TSVECTOR),
        sa.Column("fts_config", sa.String(16), nullable=False, server_default="simple"),
        sa.Column("speaker_id", UUID, sa.ForeignKey("speakers.id", ondelete="SET NULL")),
        sa.Column("locator", pg.JSONB, nullable=False),
    )
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")
    op.create_index("ix_chunks_speaker_id", "chunks", ["speaker_id"])
    op.create_index("ix_chunks_source_id", "chunks", ["source_id"])

    op.create_table(
        "entities",
        _id(),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("aliases", pg.ARRAY(sa.Text), nullable=False, server_default="{}"),
    )

    op.create_table(
        "relations",
        _id(),
        sa.Column("subj", UUID, sa.ForeignKey("entities.id", ondelete="CASCADE"), nullable=False),
        sa.Column("pred", sa.Text, nullable=False),
        sa.Column("obj", UUID, sa.ForeignKey("entities.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_id", UUID, sa.ForeignKey("chunks.id", ondelete="CASCADE")),
        sa.Column("confidence", sa.Float),
    )

    op.create_table(
        "tasks",
        _id(),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("owner_person_id", UUID, sa.ForeignKey("people.id", ondelete="SET NULL")),
        sa.Column("owner_raw", sa.Text),
        sa.Column("due_date", sa.Date),
        sa.Column("due_raw", sa.Text),
        sa.Column("status", sa.String(16), nullable=False, server_default="open"),
        sa.Column(
            "source_chunk_id",
            UUID,
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("confidence", sa.Float),
        sa.Column("confirmed", sa.Boolean, nullable=False, server_default="false"),
    )
    op.create_index("ix_tasks_due_date", "tasks", ["due_date"])

    op.create_table(
        "jobs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "source_id", UUID, sa.ForeignKey("sources.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("stage", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("error", sa.Text),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_jobs_source_stage", "jobs", ["source_id", "stage"], unique=True)

    op.create_table(
        "eval_runs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("git_sha", sa.String(40)),
        sa.Column("config_hash", sa.String(64)),
        sa.Column("results", pg.JSONB, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )


def downgrade() -> None:
    for table in (
        "eval_runs",
        "jobs",
        "tasks",
        "relations",
        "entities",
        "chunks",
        "segments",
        "speakers",
        "people",
        "sources",
    ):
        op.drop_table(table)
