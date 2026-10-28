# ADR 0015 — Entity graph snapshots before merging

Date: 2026-10-28 · Status: accepted

Context: automatic merging (ADR 0014) changes the graph in place and has no undo. The user
asked for a backup of the graph before merging, kept inside the project, to switch back if
merging does not help.

## Decisions
- **What**: the rows of `entities` (without name embeddings, which are recomputed),
  `entity_mentions`, `relations` and `entity_merges`, as one gzipped JSON file in
  `data/backups/entities/<time>-<label>.json.gz`. Not a `pg_dump`: chunks, sources and tasks
  are not touched by merging, and a table-level restore would roll them back too.
- **Where**: inside the project folder but under `data/`, which is git-ignored. Snapshots hold
  entity names from private sources, so they are never committed (AGENTS.md).
- **When**: the pipeline saves a "before-merge" snapshot right before the first automatic
  merge of an ingest (no file when nothing is merged). `tkb entities snapshot` saves one by
  hand. Only automatic snapshots are pruned (newest `entities.keep_snapshots`, default 10).
- **Restore** (`tkb entities restore <#|file>`): saves the current graph ("before-restore"),
  then replaces the entity tables with the snapshot's rows. Rows pointing at chunks that no
  longer exist are skipped and counted. Refused while an ingest holds the GPU lock.
- **Switch off**: `entities.auto_merge: false` skips embedding and merging after ingest, so a
  restored graph stays as it is. Name keys (ADR 0014) still apply.
- The thesis KB's graph from before M14 was rebuilt from the pre-migration database backup
  (migration 0005's key merges only) and saved as `pre-m14-unmerged`.
