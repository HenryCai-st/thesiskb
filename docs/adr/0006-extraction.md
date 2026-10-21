# ADR 0006 — Extraction: one structured call per chunk, verbatim evidence, dates in code

Date: 2026-10-21 · Status: accepted

## One call per chunk, fields by source type
`ingest/extract.py` sends each chunk once to the `extraction` role (qwen3:14b, thinking off,
JSON-schema structured output via LiteLLM → Ollama `format`) asking only for what fits the source:

| source | tasks | statements | entities / relations |
|---|---|---|---|
| note, meeting, interview | ✓ | | ✓ |
| audio / video | (if meeting/interview) | ✓ | ✓ |
| paper, book, demo | | (demo video: ✓) | ✓ |

The prompt must say that *every dated line* is an item: asked for "tasks", the model left out
"Proposal" and "Final" as "milestones".

## Verbatim evidence or nothing
Every task, statement and relation carries an `evidence_quote` that must occur in the chunk
(case, whitespace, bullets and dashes normalised), otherwise it is dropped and counted. This is
what makes extracted rows citable: each row links to its chunk and the quote shows why.

## Dates resolved in code (`ingest/dates.py`)
The LLM copies due dates as written; code resolves them. Month-only dates take the first such
month on/after an anchor — the earliest explicit date among the source's tasks, else the source
date — and fall on the month's last day. The schedule note gives Nov → 2026-11-30 and
March → 2027-03-31 because the note starts at 10.10.2026. German month and weekday names work.

## Tasks survive re-indexing
Re-indexing replaces chunks. Tasks now store `source_id` and `evidence_quote`, and their chunk
link is `ON DELETE SET NULL` (migration 0002). Re-extraction replaces unconfirmed tasks; tasks the
user confirmed or edited (`tkb tasks review`) are kept and re-linked to the chunk holding their
evidence quote.

## Q3 answers come from SQL
`answer/tasks.py`: open tasks sorted by due date become the passages; the LLM only phrases them,
with the usual citation and support checks. This also fixes the schedule order that the photo
alone could not give (March before January on the hand-drawn loop).

## Cost
Results are cached per chunk text (`extract.json`, written after every call). Papers are the
slow part: entities + relations for 81 survey chunks took 40 min on the 4070 Ti SUPER; use
`tkb ingest --no-extract` to skip it.

## Known gaps (for Milestone 6)
- Entity duplicates ("DDPM" / "DDPMs"), running-footer noise ("ACM Computing Surveys"), and
  entities seen only in relations typed "other" (356 of 1036 on the survey). Merge/normalise
  before deciding on a graph layer.
- Speaker labels (SPEAKER_02) are per recording; statements by unlabelled speakers are only
  attributable within their source.
