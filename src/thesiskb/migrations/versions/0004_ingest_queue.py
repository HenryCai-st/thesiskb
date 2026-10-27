"""Ingest queue (Milestone 13) and the link from a changed file's new source to its old one.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ingest_queue",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("kind", sa.String(16), nullable=False),  # ingest | reingest
        sa.Column("paths", pg.ARRAY(sa.Text), nullable=False, server_default="{}"),
        sa.Column(
            "source_id", pg.UUID(as_uuid=True), sa.ForeignKey("sources.id", ondelete="SET NULL")
        ),
        sa.Column("options", pg.JSONB, nullable=False, server_default="{}"),
        sa.Column("status", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("progress", pg.JSONB),
        sa.Column("report", pg.JSONB),
        sa.Column("error", sa.Text),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_ingest_queue_status", "ingest_queue", ["status", "id"])
    op.add_column(
        "sources",
        sa.Column(
            "replaces", pg.UUID(as_uuid=True), sa.ForeignKey("sources.id", ondelete="SET NULL")
        ),
    )


def downgrade() -> None:
    op.drop_column("sources", "replaces")
    op.drop_index("ix_ingest_queue_status", "ingest_queue")
    op.drop_table("ingest_queue")
