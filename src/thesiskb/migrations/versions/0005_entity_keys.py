"""Entity keys, name embeddings and the merge log (Milestone 14); merges existing duplicates.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-28
"""

import re
import unicodedata
from collections import defaultdict
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql as pg

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _key(name: str) -> str:  # frozen copy of thesiskb.ingest.entities.entity_key
    k = unicodedata.normalize("NFKC", name).casefold()
    k = re.sub(r"[-_‐-―]+", " ", k)
    return re.sub(r"\s+", " ", k).strip()


def upgrade() -> None:
    op.add_column("entities", sa.Column("key", sa.Text))
    op.add_column("entities", sa.Column("embedding", Vector(1024)))
    op.create_table(
        "entity_merges",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "kept_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("entities.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # suggestions point at the other entity; for merges it is gone, only its name is kept
        sa.Column(
            "other_id", pg.UUID(as_uuid=True), sa.ForeignKey("entities.id", ondelete="CASCADE")
        ),
        sa.Column("other_name", sa.Text, nullable=False),
        sa.Column("status", sa.String(16), nullable=False),  # merged | suggested
        sa.Column("reason", sa.String(16), nullable=False),  # key | variant | similar | shared
        sa.Column("similarity", sa.Float),
        sa.Column("shared_neighbours", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    conn = op.get_bind()
    rows = conn.execute(
        sa.text(
            "SELECT e.id, e.name, e.aliases, count(m.chunk_id) AS n FROM entities e "
            "LEFT JOIN entity_mentions m ON m.entity_id = e.id GROUP BY e.id"
        )
    ).all()
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[_key(r.name)].append(r)
    for key, group in groups.items():
        group.sort(key=lambda r: (-r.n, len(r.name), r.name))
        keep, rest = group[0], group[1:]
        conn.execute(
            sa.text("UPDATE entities SET key = :k WHERE id = :i"), {"k": key, "i": keep.id}
        )
        for d in rest:  # same steps as thesiskb.ingest.entities.merge_entities
            p = {"keep": keep.id, "drop": d.id}
            conn.execute(
                sa.text(
                    "INSERT INTO entity_mentions (entity_id, chunk_id) SELECT :keep, chunk_id "
                    "FROM entity_mentions WHERE entity_id = :drop ON CONFLICT DO NOTHING"
                ),
                p,
            )
            conn.execute(sa.text("UPDATE relations SET subj = :keep WHERE subj = :drop"), p)
            conn.execute(sa.text("UPDATE relations SET obj = :keep WHERE obj = :drop"), p)
            conn.execute(
                sa.text(
                    "UPDATE entities SET aliases = (SELECT array_agg(DISTINCT a) FROM "
                    "unnest(aliases || CAST(:names AS text[])) a) WHERE id = :keep"
                ),
                {"keep": keep.id, "names": [d.name, *(d.aliases or [])]},
            )
            conn.execute(
                sa.text(
                    "INSERT INTO entity_merges (kept_id, other_name, status, reason) "
                    "VALUES (:keep, :name, 'merged', 'key')"
                ),
                {"keep": keep.id, "name": d.name},
            )
            conn.execute(sa.text("DELETE FROM entities WHERE id = :drop"), p)
    conn.execute(sa.text("DELETE FROM relations WHERE subj = obj"))
    op.alter_column("entities", "key", nullable=False)
    op.create_index("ux_entities_key", "entities", ["key"], unique=True)


def downgrade() -> None:
    # merged entities are not split again; their names stay in `aliases`
    op.drop_index("ux_entities_key", table_name="entities")
    op.drop_table("entity_merges")
    op.drop_column("entities", "embedding")
    op.drop_column("entities", "key")
