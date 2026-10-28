# Using ThesisKB day to day

This is the practical guide: what to do with a new paper, a photo of the whiteboard, or a
meeting recording, and how to get answers you can trust. Setup is in the [README](../README.md).

## The loop

```
put files in data/raw/  →  tkb ingest data/raw  →  label speakers  →  review tasks
                                       ↓
                 ask (web UI / tkb ask / Claude via MCP)  →  check the cited source
```

Re-running `tkb ingest data/raw` is always safe: finished work is skipped, so just run it after
adding files. Each file is identified by its content (SHA-256): renaming or moving a file does
not re-process it, and exact duplicates are ingested once.

## 1. Adding material

| You have | Put it in `data/raw/` as | What happens |
|---|---|---|
| Paper / book chapter | `.pdf` (`.docx`, `.pptx`, `.md` also work) | Pages, sections, tables, figure captions; title/authors/date looked up for arXiv files |
| E-book | `.epub` | Chapters and sections; citations name the chapter (no pages); title/author from the book |
| Saved web page | `.html` (browser "Save page as", the `_files` folder next to it is ignored) | Text by section, images by their alt text; videos hosted on the site are not included |
| Photo of whiteboard, post-its, handwritten notes | `.jpg` / `.png` / `.heic` | Turned upright, read by the vision model (handwriting) and OCR; post-its become one item each |
| Meeting / interview recording | `.mp3`, `.wav`, `.m4a` | Transcribed with timestamps, split by speaker |
| Video (interview, product demo) | `.mp4`, `.mkv`, `.mov` | As audio, plus keyframes described by the vision model |
| Granola meeting export | `.csv` as exported (AI summary + transcript per meeting) | One source per meeting row; re-exporting a longer CSV only adds the new meetings. E-mail and owner name are not stored |

```bash
uv run tkb ingest data/raw                     # everything new
uv run tkb ingest data/raw/meetings --type meeting
uv run tkb ingest data/raw/private --local-only   # never sent to a cloud model (API mode)
uv run tkb sources                             # what is in the KB, and each file's stages
```

**From the web UI** (Sources tab): type a folder or file path on the server (default
`data/raw`) and press **Ingest**, or press **re-ingest** next to one source. Requests wait in a
queue and run one at a time; the panel shows the current stage and file. Questions wait while
an ingest runs (both need the GPU). Without the web server, `uv run tkb worker` processes the
queue. `tkb ingest` in a terminal waits if the queue is busy, and vice versa.

**Changed files**: edit a file in place (same path) and ingest again — the new version replaces
the old one once it is fully processed. Tasks you confirmed or marked done are kept, and
speaker names carry over.

Source types steer answering: `paper`/`book` feed the literature path, `meeting`/`interview`/`note`
are scanned for tasks. Defaults: e-books → book, web pages → web, other documents → paper, photos → note, audio and meeting CSVs → meeting, video → demo.

Every source longer than a page also gets a summary per part and one for the whole source
(the `summarize` stage, ~10–20 s per part with the local model; cached, so re-runs are free).

**How long it takes** (RTX 4070 Ti SUPER): a 60-page paper ~1.5 min to index plus ~40 min of
extraction (entities/relations — skip with `--no-extract` and run later); a photo ~15 s; 3 min of
audio ~20 s; a 2-min video ~1.5 min. Long jobs resume where they stopped if interrupted.

**Tip — recordings:** say names out loud at the start ("I'm Anna, this is Prof. Example").
`tkb speakers list` turns "my name is …" into a suggestion.

## 2. Telling the KB who is speaking

Speaker separation gives labels (`SPEAKER_00`, `SPEAKER_01`, ...) per recording. Map the ones that
matter to people once:

```bash
uv run tkb speakers list meeting_2026-11-04       # talk time, first words, name suggestions
uv run tkb speakers label meeting_2026-11-04 SPEAKER_01 "Prof. Example" --alias "Example" --role supervisor
```

The recording is re-indexed with the name, so answers say "Prof. Example (12:40): ..." and
`tkb person "Prof. Example"` shows everything they said across recordings. Aliases also catch
how Whisper spells a name ("Will I.M." for Will.i.am). Avoid aliases that are common words.

Files are matched by any part of the filename or a sha256 prefix in every command that takes
`<file>`.

## 3. Tasks and deadlines

Tasks are extracted from notes, photos and meetings, each with a verbatim quote as evidence.
Dates are resolved in code: "Nov" in a note that starts "10.10.2026" becomes November 2026.

```bash
uv run tkb tasks                 # open tasks by due date, with where each came from
uv run tkb tasks review          # confirm / edit / delete, one by one
uv run tkb tasks done 93a4       # id prefix from `tkb tasks`
```

Confirm the ones that are right: confirmed and edited tasks are kept even when a source is
re-processed. The web UI's Tasks tab does the same with buttons.

## 4. Asking

Web UI (recommended): `uv run tkb serve`, open http://127.0.0.1:8000.
Terminal: `uv run tkb ask "..."`.

The router picks the path; you can force one with `--path`:

| Ask like | Path | Answer comes from |
|---|---|---|
| "What is the state of the art for designing diffusion models?" | literature | papers only: several searches, evidence summaries, synthesis |
| "What does Prof. Example prefer for the evaluation?" | person | only what that person said or wrote |
| "Show me my thesis schedule", "When is my mid-term?" | task | the tasks table (SQL), sorted by date |
| "Which datasets appear in my sources?" | entity | the extracted entities/relations tables |
| "Summarize the book", "What is the meeting about?" | summary | summaries of every part (chapter, section, 10-minute window) written at ingest; the answer says which parts it covered |
| anything else | general | search over everything |

Questions in German work too ("Wann ist meine Zwischenpräsentation?").

### Reading an answer

- Every sentence ends with `[S1]`-style citations. In the UI, click one: PDFs open at the page
  with the passage boxed, photos upright, recordings start playing at the quoted second.
- `↳ S2 at 0:24 Prof. Example` under a sentence is the exact moment it refers to.
- ⚠ marks a sentence the second check could not match to its cited passage, or one without
  a citation. The line under it ("why: …") is the checker's reason. Treat it as unverified —
  open the source.
- **Details** (under the answer) shows how it was found: the route, every query sent to the
  database (literature questions are split into several), the top hits with scores, the
  passages the model saw and the models used. In the terminal: `uv run tkb trace last` (or the
  id printed after `tkb ask`). Chat clients get a "How this answer was found" link to the same
  view.
- "The knowledge base does not contain enough evidence to answer this." is a feature: nothing
  relevant was found, so nothing was made up.
- Answers only cover what you ingested. "State of the art" means *in your library*.

### Trusting sources

```bash
uv run tkb verify 2209.00796 --by "Your Name"        # you checked this source
uv run tkb verify sketchy_blog --status disputed
uv run tkb ask "..." --verified-only                 # answer only from verified sources
```

The status is shown next to every citation. For the thesis itself, cite the original papers —
use ThesisKB answers as a reading guide, never as a source (BUILD_GUIDE §8).

### Looking at the graph

The web UI's **Graph** tab shows the entities extracted from your sources (methods, models,
concepts, people, …) and the relations between them. Search a name, pick it, and choose 1 or 2
hops. Click a node to list the sources and passages that mention it; click an edge ("DDIM —
improves — sampling speed") to open the passage the relation was read from.

Spelling variants are merged into one entity automatically: "Monte-Carlo sampling" and "Monte
Carlo sampling", "GAN" and "GANs", "Bachelor Project" and "Bachelor Projekt". Names with
different numbers ("Figure 2" / "Figure 4") are never merged; look-alikes that may be different
things ("possible" / "probable futures") are only listed as suggestions. To see what was merged:

```bash
uv run tkb entities merges      # automatic merges (newest first), then suggestions
```

A wrong merge cannot be undone from the UI; report it with the two names (`tkb report`). The
variants checked by `tkb eval` are in `eval/questions.yaml` under `entity_variants`; add yours.

## 5. Using it from Claude (or another agent)

```bash
claude mcp add thesiskb -- uv --directory /path/to/thesiskb run tkb mcp
```

Claude can then `search` your KB, read chunks, list your tasks and person profiles, or `ask` for
a cited answer — e.g. "Using thesiskb, draft my related-work outline and cite chunk IDs."
Open WebUI or any OpenAI client: base URL `http://localhost:8000/v1`, model `thesiskb`.

## 6. Privacy

- Everything runs locally by default; nothing leaves the machine (except the arXiv id of
  arXiv-named PDFs for metadata — turn off with `metadata_lookup: false` in pipeline.yaml).
- **API mode** (`TKB_MODELS_FILE=examples/models.api.yaml`): retrieved passages go to the cloud
  provider with each question. Mark private sources `tkb local-only <file>`; their content
  then only goes to the local fallback models, or the request is refused.
- Recordings of people: get consent and note how (`tkb consent <file> "verbal, all 3, 2026-11-04"`);
  if someone asks,
  `tkb forget-person "Name"` removes their statements, speaker labels and profile and re-indexes
  their recordings without the name. `--delete-recordings` removes those recordings from the
  index; delete the original files yourself.
- Never commit `data/`, `.env` or recordings; the repo's `.gitignore` already excludes them.

## 7. Keeping it healthy

```bash
uv run tkb doctor        # DB, GPU, Ollama models, cloud keys, ffmpeg, tesseract
uv run tkb eval          # gold questions (eval/questions.yaml) — run after changing models
uv run tkb reindex <file>   # re-chunk + re-embed one file from its cached outputs
uv run tkb trace last    # how the last answer was found (queries, hits, judge reasons)
uv run tkb report        # zip for a bug report: logs, doctor, versions, config (no secrets)
docker exec thesiskb-postgres pg_dump -U thesiskb thesiskb > backup.sql   # back up the DB
```

Every command and the server write a log to `data/logs/thesiskb.log` (one JSON object per line,
rotated at 5 MB, 5 files kept). Ingest failures include the file's sha256, the stage and the
traceback. To report a problem, run `uv run tkb report` (add `--trace` to include the last
answer) and send the zip from `data/logs/`. Secrets are replaced by `***`; the log does contain
file names, so look through it first. `.env`, `people.yaml`, your files and chunk text are
never included.

Add your own gold questions to `eval/questions.yaml` as you go (a question, the file that should
answer it, a few facts the answer must contain). `tkb eval` then tells you whether a new model or
setting made things better or worse.

## Troubleshooting

| Symptom | Fix |
|---|---|
| After a reboot: "Can't reach the database" or `APIConnectionError` | start **Docker Desktop** and the **Ollama** app (both need to run), then re-run the command — finished stages are skipped |
| `tkb doctor`: Database FAIL | `docker compose up -d` (Docker Desktop must be running) |
| Ollama: missing models | `ollama pull <name>` for each listed |
| Diarization fails: 401/403 | `TKB_HF_TOKEN` in `.env`, and accept the model terms on Hugging Face |
| No Hugging Face token / CPU only | set `diarization: {backend: none}` in models.yaml (one speaker per file) |
| Photos read sideways | the vision role must return logprobs (keep it on Ollama `/v1`) |
| First answer slow (~60 s) | models loading; `tkb serve` keeps them loaded between questions |
| A sentence flagged ⚠ that looks right | read its "why:" line and **Details**; the checker is strict about names/numbers; open the source |
| Source links in Open WebUI don't open | links point at the address the request came to; if Open WebUI reaches ThesisKB under another name, set `TKB_PUBLIC_URL=http://<host>:8000` in `.env`. With `TKB_API_TOKEN` set, files also need `?token=` |
| Something failed and you need help | `uv run tkb report`, then send the zip from `data/logs/` |
| Changed models.yaml embedding model | `tkb ingest data/raw --force` (all vectors must be re-made) |
