# ADR 0013 — Ingest queue, re-ingest from the web UI, changed files, meeting CSVs

Date: 2026-10-27 · Status: accepted

Context: users want to start and watch ingest from the web UI, re-run one source, and keep a
growing Granola export current (feedback point 6, BUILD_GUIDE_v2 Milestone 13).

## Decisions
- **Queue table `ingest_queue`** (migration 0004), not the `jobs` table the guide suggested:
  `jobs` is one row per source *per stage* with a required source id, while a request
  ("ingest this folder") exists before any source does. Jobs keep recording stage status.
- **Worker**: claims the oldest queued row with `SELECT … FOR UPDATE SKIP LOCKED`, runs the
  normal pipeline, writes progress (stage, file, steps done/total) to the row about once a
  second, and the report or error at the end. `tkb serve` runs it as a thread; `tkb worker`
  runs it without the web server. Items left "running" by a crash are re-queued at start
  (stages are idempotent). No library (Procrastinate etc.): a 120-line loop is enough for one
  user and one GPU.
- **One ingest at a time on the GPU**: every ingest (worker or `tkb ingest`) holds a Postgres
  advisory lock; the worker takes it before claiming, so an item never shows "running" while
  it waits. Inside `tkb serve`, the worker also holds the answer lock: questions wait while an
  item runs (as before with the old background ingest), so two large models never share VRAM.
- **API**: `POST /api/ingest` (queues; 400 if a path is missing on the server),
  `POST /api/sources/{id}/reingest`, `GET /api/jobs`. The web UI's Sources tab has an ingest
  field, a queue panel (polls every 2 s while something runs) and a re-ingest button per source.
- **Changed files**: a new sha256 at a known path registers a new source with
  `replaces = <old id>` (inheriting type, local-only and verification status). After the new
  version is fully ingested, older sources at that path are retired: confirmed or finished
  tasks are matched to the new tasks by normalized description (confirmation/status/owner
  copied) or moved to the new source if no longer present; speaker names are copied to the
  same diarization labels; then the old source and its processed folder are deleted.
  Re-ingest of an unchanged file forces every stage.
- **Meeting CSVs (Granola)**: one source per row, path `<file>.csv#<note id>`, sha256 of the
  row's indexed fields, so a grown export ingests only new rows and an edited row replaces its
  old version. Columns follow Granola's header-less export (id, e-mail, date, owner, created,
  AI summary, notes, transcript); other CSV layouts are skipped. The e-mail and owner name are
  never stored. Blocks: AI summary by heading, notes, transcript turns with their speaker
  label; line locators. No Granola API connector (free plan; the CSV has transcripts).
