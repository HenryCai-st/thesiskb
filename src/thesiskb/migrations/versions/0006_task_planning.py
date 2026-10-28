"""Task editing and planning (Milestone 15): new task fields and proposed task updates.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("tasks", sa.Column("notes", sa.Text))
    op.add_column("tasks", sa.Column("start_date", sa.Date))
    op.add_column("tasks", sa.Column("estimate_hours", sa.Float))
    op.add_column(
        "tasks",
        sa.Column(
            "parent_id", pg.UUID(as_uuid=True), sa.ForeignKey("tasks.id", ondelete="CASCADE")
        ),
    )
    # extracted (from a source) | manual (typed by the user) | plan (proposed by "plan")
    op.add_column(
        "tasks", sa.Column("origin", sa.String(16), nullable=False, server_default="extracted")
    )
    op.create_index("ix_tasks_parent_id", "tasks", ["parent_id"])
    op.create_table(
        "task_proposals",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "task_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("tasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("field", sa.String(16), nullable=False),  # due_date | status
        sa.Column("value", sa.Text, nullable=False),
        sa.Column("value_raw", sa.Text),  # the date as written in the source
        sa.Column(
            "source_id", pg.UUID(as_uuid=True), sa.ForeignKey("sources.id", ondelete="CASCADE")
        ),
        sa.Column(
            "chunk_id", pg.UUID(as_uuid=True), sa.ForeignKey("chunks.id", ondelete="SET NULL")
        ),
        sa.Column("evidence_quote", sa.Text),
        sa.Column("status", sa.String(16), nullable=False, server_default="open"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("task_proposals")
    op.drop_index("ix_tasks_parent_id", table_name="tasks")
    for col in ("origin", "parent_id", "estimate_hours", "start_date", "notes"):
        op.drop_column("tasks", col)
