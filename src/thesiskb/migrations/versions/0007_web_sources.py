"""Web sources (Milestone 17): retrieval time, and the person a web lookup was about.

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # web pages keep their URL in `origin` (like "arXiv:…" for papers); this is when it was fetched
    op.add_column("sources", sa.Column("retrieved_at", sa.DateTime(timezone=True)))
    # web search results about a person: `tkb forget-person` deletes them
    op.add_column(
        "sources",
        sa.Column(
            "person_id", pg.UUID(as_uuid=True), sa.ForeignKey("people.id", ondelete="SET NULL")
        ),
    )


def downgrade() -> None:
    op.drop_column("sources", "person_id")
    op.drop_column("sources", "retrieved_at")
