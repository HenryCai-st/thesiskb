"""Answer traces (Milestone 10): queries, hits, passages and judge verdicts per question.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "answer_traces",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("channel", sa.String(16), nullable=False),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("answer", sa.Text),
        sa.Column("refused", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("route", pg.JSONB),
        sa.Column("retrieval", pg.JSONB),
        sa.Column("context", pg.JSONB),
        sa.Column("sentences", pg.JSONB),
        sa.Column("models", pg.JSONB),
        sa.Column("timings", pg.JSONB),
        sa.Column("error", sa.Text),
    )
    op.create_index("ix_answer_traces_created_at", "answer_traces", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_answer_traces_created_at", "answer_traces")
    op.drop_table("answer_traces")
