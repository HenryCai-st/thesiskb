"""Extraction tables (Milestone 5): statements, entity mentions; tasks survive re-indexing.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID = pg.UUID(as_uuid=True)


def upgrade() -> None:
    # tasks: keep the source and the evidence quote so a re-index (which replaces chunks) can
    # re-link a task instead of deleting it — confirmed tasks must never be lost
    op.add_column("tasks", sa.Column("source_id", UUID))
    op.add_column("tasks", sa.Column("evidence_quote", sa.Text))
    op.add_column(
        "tasks", sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now())
    )
    op.execute(
        "UPDATE tasks t SET source_id = c.source_id FROM chunks c WHERE c.id = t.source_chunk_id"
    )
    op.alter_column("tasks", "source_chunk_id", nullable=True)
    op.drop_constraint("tasks_source_chunk_id_fkey", "tasks", type_="foreignkey")
    op.create_foreign_key(None, "tasks", "chunks", ["source_chunk_id"], ["id"], ondelete="SET NULL")
    op.create_foreign_key(None, "tasks", "sources", ["source_id"], ["id"], ondelete="CASCADE")
    op.create_index("ix_tasks_source_id", "tasks", ["source_id"])

    op.create_table(
        "statements",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("person_id", UUID, sa.ForeignKey("people.id", ondelete="SET NULL")),
        sa.Column("person_raw", sa.Text),
        sa.Column("topic", sa.Text, nullable=False),
        sa.Column("stance", sa.Text, nullable=False),
        sa.Column("evidence_quote", sa.Text, nullable=False),
        sa.Column(
            "source_id", UUID, sa.ForeignKey("sources.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("source_chunk_id", UUID, sa.ForeignKey("chunks.id", ondelete="SET NULL")),
        sa.Column("confidence", sa.Float),
    )
    op.create_index("ix_statements_person_id", "statements", ["person_id"])
    op.create_index("ix_statements_source_id", "statements", ["source_id"])

    op.create_table(
        "entity_mentions",
        sa.Column(
            "entity_id", UUID, sa.ForeignKey("entities.id", ondelete="CASCADE"), primary_key=True
        ),
        sa.Column(
            "chunk_id", UUID, sa.ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
        ),
    )
    op.create_index("ix_entity_mentions_chunk_id", "entity_mentions", ["chunk_id"])
    op.create_index("ix_entities_lower_name", "entities", [sa.text("lower(name)")])


def downgrade() -> None:
    op.drop_index("ix_entities_lower_name", "entities")
    op.drop_table("entity_mentions")
    op.drop_table("statements")
    op.drop_index("ix_tasks_source_id", "tasks")
    op.drop_constraint("tasks_source_id_fkey", "tasks", type_="foreignkey")
    op.drop_constraint("tasks_source_chunk_id_fkey", "tasks", type_="foreignkey")
    op.execute("DELETE FROM tasks WHERE source_chunk_id IS NULL")
    op.create_foreign_key(
        "tasks_source_chunk_id_fkey",
        "tasks",
        "chunks",
        ["source_chunk_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.alter_column("tasks", "source_chunk_id", nullable=False)
    for col in ("created_at", "evidence_quote", "source_id"):
        op.drop_column("tasks", col)
