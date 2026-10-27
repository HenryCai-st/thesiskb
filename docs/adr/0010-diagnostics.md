# ADR 0010 — Diagnostics: answer traces, absolute chat links, log file, bug report

Date: 2026-10-27 · Status: accepted

Context: users saw correct answers with broken source links and "not supported" flags, and
asked to see what the model asks the database (feedback points 1 and 9, BUILD_GUIDE_v2 M10).

## Decisions
- **Traces in Postgres** (`answer_traces`, migration 0003), one row per `answer()` call, also
  when answering fails (the error and traceback are stored, then re-raised). Each answer path
  reports its own queries and hits (`Ctx.retrieval`): candidates after fusion, reranked hits
  with scores, and how many were kept. Plus route, passages shown, sentences with the judge's
  reason, model names and timings. `channel` tells cli/api/chat/mcp/eval apart. No new service:
  Arize Phoenix / Langfuse were considered and left for later; one table is enough to answer
  "why this answer".
- Traces store the passage text the model saw (it is already in the same database). There is
  no automatic pruning yet.
- **Absolute links in `/v1` chat replies**: `TKB_PUBLIC_URL`, else the request's base URL.
  Relative links were resolved by Open WebUI against its own host. The REST API and web UI keep
  relative links (same origin).
- **Log file**: standard `logging` with a rotating JSON-lines handler in `data/logs/`, set up
  once per process by `tkb` (all commands incl. `serve`) and the MCP server. Console behaviour
  is unchanged (warnings only; ingest's rich handler at INFO). Context fields via `extra=`
  (`sha256`, `stage`). Unhandled API errors are logged with the request path.
- **`tkb report`**: zip with log tail, doctor results, package versions, the active models
  file and pipeline.yaml. Every known secret (settings, `.env`, `*KEY/TOKEN/SECRET/PASSWORD`
  environment variables, models.yaml api keys, URL passwords) is replaced by `***`.
  `people.yaml` and `.env` are never added; the last trace only with `--trace`.
