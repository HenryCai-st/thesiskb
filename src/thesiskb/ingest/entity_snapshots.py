"""Entity graph snapshots: save before merging, restore to switch back (ADR 0015).

A snapshot is one gzipped JSON file in data/backups/entities/ with the rows of `entities`,
`entity_mentions`, `relations` and `entity_merges` (not the name embeddings: they are
recomputed). The pipeline saves one before the first automatic merge of an ingest
("before-merge"); restoring saves the current graph first ("before-restore"), so a restore
can itself be undone. Only these two automatic kinds are pruned (`entities.keep_snapshots`).

Restoring replaces the whole entity graph. Mentions and relations whose chunk no longer
exists (a source re-ingested or forgotten since) are skipped and counted. With
`entities.auto_merge: true` the next ingest merges near-duplicates again; set it to false to
keep a restored graph as it is.
"""

from __future__ import annotations

import gzip
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Table, delete, insert, select
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Session

from thesiskb.core.config import PipelineConfig
from thesiskb.core.models import Chunk, Entity, EntityMention, EntityMerge, Relation

_AUTOMATIC_NAME = re.compile(r"\d{8}-\d{6}-(before-merge|before-restore)(-\d+)?\.json\.gz")
# table → columns saved (entity_merges.id is a sequence: new ids on restore, order by time)
TABLES: dict[str, tuple[Table, list[str]]] = {
    "entities": (Entity.__table__, ["id", "name", "type", "aliases", "key"]),
    "entity_mentions": (EntityMention.__table__, ["entity_id", "chunk_id"]),
    "relations": (Relation.__table__, ["id", "subj", "pred", "obj", "chunk_id", "confidence"]),
    "entity_merges": (
        EntityMerge.__table__,
        [
            "kept_id",
            "other_id",
            "other_name",
            "status",
            "reason",
            "similarity",
            "shared_neighbours",
            "created_at",
        ],
    ),
}


class SnapshotInfo(BaseModel):
    path: Path
    label: str
    created: str
    counts: dict[str, int]


class RestoreReport(BaseModel):
    restored: dict[str, int] = Field(default_factory=dict)
    skipped_missing_chunk: int = 0
    saved_current: Path | None = None


def snapshot_dir(cfg: PipelineConfig) -> Path:
    return cfg.data_dir / "backups" / "entities"


def _dump(value):
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _load(table: Table, row: dict) -> dict:
    out = {}
    for name, value in row.items():
        col = table.c[name]
        if value is not None and isinstance(col.type, UUID):
            value = uuid.UUID(value)
        elif value is not None and isinstance(col.type, DateTime):
            value = datetime.fromisoformat(value)
        out[name] = value
    return out


def save_snapshot(session: Session, cfg: PipelineConfig, label: str = "manual") -> Path:
    """Write the current entity graph (as this session sees it) to a new snapshot file."""
    now = datetime.now(UTC)
    data: dict = {"version": 1, "label": label, "created": now.isoformat(), "tables": {}}
    for name, (table, cols) in TABLES.items():
        rows = session.execute(select(*[table.c[c] for c in cols])).mappings().all()
        data["tables"][name] = [{k: _dump(v) for k, v in r.items()} for r in rows]
    folder = snapshot_dir(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{now:%Y%m%d-%H%M%S}-{label}"
    path = folder / f"{stem}.json.gz"
    n = 2
    while path.exists():
        path = folder / f"{stem}-{n}.json.gz"
        n += 1
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    _prune(cfg)
    return path


def read_snapshot(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def list_snapshots(cfg: PipelineConfig) -> list[SnapshotInfo]:
    out = []
    for p in sorted(snapshot_dir(cfg).glob("*.json.gz"), reverse=True):
        data = read_snapshot(p)
        out.append(
            SnapshotInfo(
                path=p,
                label=data["label"],
                created=data["created"],
                counts={k: len(v) for k, v in data["tables"].items()},
            )
        )
    return out


def _prune(cfg: PipelineConfig) -> None:
    keep = cfg.entities.keep_snapshots
    automatic = [
        p
        for p in sorted(snapshot_dir(cfg).glob("*.json.gz"), reverse=True)
        if _AUTOMATIC_NAME.fullmatch(p.name)
    ]
    for p in automatic[keep:]:
        p.unlink()


def restore_snapshot(session: Session, cfg: PipelineConfig, path: Path) -> RestoreReport:
    """Replace the entity graph with the snapshot's (after saving the current one)."""
    data = read_snapshot(path)
    rep = RestoreReport(saved_current=save_snapshot(session, cfg, "before-restore"))
    session.execute(delete(Entity))  # cascades to mentions, relations and the merge log
    chunks = set(session.scalars(select(Chunk.id)))
    tables = data["tables"]
    entities = {r["id"] for r in tables["entities"]}

    def ok(table: str, row: dict) -> bool:
        if table == "entity_merges":
            return row["kept_id"] in entities and row["other_id"] in (None, *entities)
        if table == "relations" and not (row["subj"] in entities and row["obj"] in entities):
            return False
        chunk = row.get("chunk_id")
        if chunk is not None and uuid.UUID(chunk) not in chunks:
            rep.skipped_missing_chunk += 1
            return False
        return True

    for name, (table, _cols) in TABLES.items():
        rows = [_load(table, r) for r in tables.get(name, []) if ok(name, r)]
        if rows:
            session.execute(insert(table), rows)
        rep.restored[name] = len(rows)
    session.flush()
    return rep
