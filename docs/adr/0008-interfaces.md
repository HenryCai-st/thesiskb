# ADR 0008 — Interfaces: FastAPI + own small UI, MCP over stdio

Date: 2026-10-22 · Status: accepted (answers BUILD_GUIDE §11.5 "Open WebUI vs custom")

## One FastAPI app (`tkb serve`)
- REST: `/api/ask`, `/api/ingest` (background), `/api/tasks` (+ confirm/done/reopen),
  `/api/sources`, `/api/sources/{id}`, `/api/chunks/{id}`, `/api/people/{name}`.
- Files: `/files/{source_id}` serves originals with HTTP Range (players can seek), images
  upright with `?upright=1`; `/pages/{id}/{n}.png` renders a PDF page (pypdfium2, cached);
  `/frames/{id}/{name}` serves keyframes.
- OpenAI-compatible `/v1/models` and `/v1/chat/completions` (incl. `stream: true`) so Open WebUI
  or any OpenAI client can use the KB as a "model". The reply is markdown: the answer, a
  warning listing unsupported sentences, and a source list with links.
- Questions run one at a time (lock); ingest holds the same lock (GPU).
- Binds 127.0.0.1. `TKB_API_TOKEN` enables bearer auth; `tkb serve --host 0.0.0.0` refuses to
  start without it. Media elements can't send headers, so file URLs accept `?token=`.

## Own UI instead of Open WebUI as the main front end
Milestone 7's acceptance — click a citation, see the exact page or jump to the timestamp —
needs a viewer next to the answer: Open WebUI shows our links but cannot render a page with the
cited boxes or seek a player. The UI is one static HTML file (no build step): answer with [S#]
chips and ⚠ flags; viewer shows PDF pages as images with the chunk's bounding boxes drawn on
top, photos upright with post-it regions, audio/video players started at the pinpointed second,
keyframe thumbnails. Open WebUI still works through `/v1` for chat-style use.

PDF pages are rendered server-side because embedded PDF viewers are not available everywhere
(the IDE's preview pane showed a blank frame); this also makes the bounding boxes visible.

## MCP server (`tkb mcp`, stdio)
Tools `search`, `get_chunk`, `get_source`, `list_tasks`, `person_profile`, `ask`, using the MCP
Python SDK 2.x (`MCPServer`, formerly FastMCP). Logging goes to stderr; stdout is the protocol.

## Verified (2026-10-22, local GPU)
- Person question in the UI → "▶ 0:24 Will.i.am" opens the audio player at 0:24.
- Task question → photo shown upright; answer says "January 2027", not the internal sort date.
- PDF chunk → page 13 rendered with both cited paragraphs boxed.
- `/v1/chat/completions` (normal and streaming); video answer pinpoint 1:20 (VTT 1:21.8),
  keyframe and video (206 Range) served.
