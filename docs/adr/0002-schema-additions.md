# ADR 0002 — Small additions to the §6 schema

Date: 2026-10-21 · Status: accepted

- `sources.local_only boolean` — needed by §10 (router must stay local when any retrieved chunk is
  local-only). Added now to avoid an early migration.
- `chunks.fts_config text default 'simple'` — §6 asks for a per-language FTS configuration; storing
  it per chunk lets the query side build `to_tsquery(<config>, ...)` consistently.
- `jobs (source_id, stage)` is unique — one status row per file per stage keeps stages idempotent.
- HNSW index uses `vector_cosine_ops` (bge-m3 embeddings are compared by cosine).
