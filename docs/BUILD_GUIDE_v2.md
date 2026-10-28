# ThesisKB — Build Guide v2 (user feedback round 1)

This guide covers the nine feedback points from the first users of v0.1.0. It continues
[BUILD_GUIDE.md](../BUILD_GUIDE.md): same format (**Goal → Tasks → Acceptance**), and
milestone numbers continue after Milestone 9 (Linux server, still on hold).

Before writing code, read [coding-guide.md](coding-guide.md). It defines what is in scope
and when a milestone is done. Anything not listed in a milestone below is out of scope.

## 1. The feedback: what users want, and what we found

| # | Feedback | What the user actually wants | Diagnosis | Milestone |
|---|---|---|---|---|
| 1 | Chat gives a correct answer, but the source link fails or is marked "not supported by the cited source". Show the exact query sent to the database. | Clickable sources that open, and a way to see *why* an answer was flagged. | **Confirmed.** Two separate problems. (a) Links in `/v1` chat replies are relative (`/files/…`); Open WebUI resolves them against its own host, so they break. (b) The ⚠ note comes from the per-claim judge, but no reason is shown and nothing about the question (route, sub-queries, retrieved chunks, scores, judge verdicts) is stored. | M10 |
| 2 | "Summarize the book" covers chapters 1 and 3 but skips 2; asking about chapter 2 works. Is the model too small? | Whole-document questions (summaries, overviews) that cover the whole document. | **Confirmed, not mainly model size.** Retrieval sends at most 8–12 passages (~600 tokens each) chosen by similarity to the question. Nothing in the pipeline ever sees the whole book. A bigger model would get the same 12 passages. Duplicate entities (point 4) are not the cause. | M12 |
| 3 | A "Models" tab: options per component (OCR, speech, …) with sizes; choosing one downloads it; online models take an API key. | Change models without editing YAML, and know the size/VRAM cost first. | Models are set per role in `config/models.yaml` and per backend in `pipeline.yaml`. Ollama can list, pull (with byte progress) and delete models over HTTP. | M16 |
| 4 | Visualize the knowledge base. "lake surtan" vs "lake-surtan" are duplicates. Do entities need managing? | See what is in the KB; no duplicate entities. | Entities are matched by exact name, so spelling variants become separate nodes. String matching alone is known to miss duplicates ([DEG-RAG](https://arxiv.org/pdf/2510.14271)). Whether users need full manual editing is **open**; start with automatic merging plus merge/rename. | M14 |
| 5 | EPUB fails to ingest. Also HTML/URLs with images, layout, video and audio. | Ingest books in EPUB and web pages. | **EPUB cause found.** Docling converts EPUB fine, but EPUB/HTML have no pages, so every item lacks page provenance and `extract_blocks()` drops all of them → 0 blocks → 0 chunks. Fix: section/anchor locators for page-less documents. URL ingest is new work. | M11 (files), M17 (URLs) |
| 6 | Granola CSV grows over time. Re-run ingest for a source from the web UI. Make ingest a queue started from the web UI. | Keep growing sources up to date without the terminal. | Ingest runs only from the CLI. A changed file gets a new sha256 and is ingested as a *new* source, so the old version stays. Granola's built-in CSV export has no transcripts; its [API](https://pipeline.zoominfo.com/sales/granola-api) (Business plan) or MCP server does. | M13 |
| 7 | Person background check: query Google/Bing/other sources, summarize into one graph. What does web crawling need? | Look up public information about a person and add it to the KB. | New capability with **legal risk** (GDPR profiling of people from scraped data). Needs a search API, polite fetching, storage of fetched pages as sources, and a legal check first. | M17 (gated) |
| 8 | Tasks: manual edits, LLM updates, planning backwards from a deadline into weekly steps, week view with arrows or a calendar with hover. | Plan thesis work week by week, not only list extracted tasks. | Tasks are read-only in the UI except confirm/done. No start date, estimate or parent task. | M15 |
| 9 | Monitoring and bug-fixing for other users: logs when something fails. | When something goes wrong on a user's PC, get enough information to fix it. | Errors print to the console only; no log file, no stored answer traces, no way to bundle a bug report. | M10 |

## 2. Order of work

1. **M10 Diagnostics** first: every later bug report depends on it, and it fixes point 1.
2. **M11 EPUB/HTML files**: a bug fix; users are blocked today.
3. **M12 Whole-document answers**.
4. **M13 Ingest queue and re-ingest**.
5. **M14 Entities and graph view**.
6. **M15 Task editing and planning**.
7. **M16 Models tab**.
8. **M17 Web sources** (URL ingest, then web search). Starts only after the legal questions in section 4 are answered.

Commit and tag at the end of each milestone (`m10`, `m11`, …). Run `tkb eval` after any
change to retrieval, prompts or models, and report the diff.

## 3. Milestones

### Milestone 10 — Diagnostics: answer traces, working links, logs (points 1, 9)
Goal: every answer can be explained afterwards, and every failure leaves a readable record.

- **Absolute source links.** Add `TKB_PUBLIC_URL` (default: the request's base URL). `/v1/chat/completions` renders absolute links so they open from Open WebUI. Open WebUI has no documented structured-sources format for this case; community pipes append markdown links, which is what we already do ([pipe docs](https://docs.openwebui.com/features/extensibility/plugin/functions/pipe)).
- **Answer trace.** One row per question in a new `answer_traces` table (Alembic migration): question, route and router reason, sub-queries, retrieved chunk ids with fusion/rerank scores, passage ids sent to the model, model role names, judge verdict *and reason* per claim, timings, error if any. Written by `answer()`; nothing else changes in answering.
- **Show the trace.** `tkb trace <id|last>` prints it. In the web UI, each answer gets a "Details" toggle showing the queries, the retrieved passages and why a sentence was flagged. The chat reply includes the trace id.
- **Logs.** Standard `logging` with a rotating file handler writing to `data/logs/thesiskb.log` (JSON lines: time, level, module, source sha, stage, message, traceback). CLI, server and ingest all use it. Console output stays as it is.
- **Bug report bundle.** `tkb report` writes a zip with: the last N log lines, `tkb doctor` output, versions, config files with secrets removed, and optionally the last answer trace. It never includes `.env`, raw files, chunk text or person names unless the user passes `--include-content`.

Not in scope: external tracing services. [Arize Phoenix](https://langfuse.com/compare/arize-phoenix) (one container, OpenTelemetry) is the option to revisit if the built-in trace view is not enough; Langfuse needs ClickHouse, Redis and S3 when self-hosted.

Acceptance:
- A source link in Open WebUI opens the right page/timestamp.
- For any answer, `tkb trace last` shows the sub-queries, retrieved chunks with scores, and the judge's reason for every flagged sentence.
- An ingest failure leaves a log entry with file name, sha256, stage and traceback.
- `tkb report` output contains no secrets (test checks for the DB password, HF token and API keys).

### Milestone 11 — EPUB and HTML files (point 5, files only)
Goal: EPUB and saved HTML pages ingest with working citations.

- `DoclingConverter`: enable `InputFormat.EPUB` and `InputFormat.HTML` alongside PDF ([supported formats](https://docling-project.github.io/docling/usage/supported_formats/)).
- `extract_blocks()`: when items have no page provenance, keep them and give them a **non-page locator**: `{"section": [...], "anchor": "<file-in-epub>#<id>", "order": n}`. Never drop text silently; log a warning if a document yields 0 blocks.
- Viewer: open page-less sources as rendered HTML scrolled to the anchor (EPUB chapters are XHTML inside the zip). Citations show the section path instead of a page number.
- Images inside EPUB/HTML go through the existing image path (caption/OCR) only if Docling exposes them as picture items; no new image pipeline.
- Fallback for EPUBs that Docling rejects (invalid XHTML is common in real files): read the spine with `ebooklib` + BeautifulSoup ([example](https://bitsgalore.org/2023/03/09/extracting-text-from-epub-files-in-python)). Add it only if one of the test EPUBs needs it.
- Test sources: the user will provide real EPUB and HTML files. Tests use small synthetic files; real files are kept out of git.

Acceptance: the user's failing EPUB ingests; a question about chapter 2 cites chapter 2 and the link opens that chapter; `tkb eval` shows no regression.

### Milestone 12 — Whole-document answers (point 2)
Goal: "summarize the book / paper / meeting" covers all parts of the document.

- **Summaries at ingest** (new stage `summarize`, idempotent, cached in `data/processed/<sha>/summaries.json`): one summary per section (chapter), then one per document built from the section summaries. This is the simple, structure-based form of [RAPTOR](https://huggingface.co/papers/2401.18059) (tree of summaries); we use the document's own headings instead of clustering. Long sections are summarized in parts first.
- Store summaries as chunks with `kind="summary"` and a locator covering the section (first and last page or anchor), so they are searchable and citable.
- **Summary route.** Router recognises overview questions ("summarize", "overview", "what is X about", "main points"). The `summary` path fetches the document summary plus all section summaries of the target source, in document order, instead of top-k similarity.
- **Coverage check.** The answer for a summary question lists which sections it used; the trace records any section left out.
- Gold questions: add at least two whole-document questions (one book with ≥ 3 chapters, one long meeting) with the expected sections.

Not in scope: changing chunk size, top-k or the answer model for this problem.

Acceptance: "Summarize <book>" mentions every chapter of the test book; per-chapter questions keep their current scores in `tkb eval`.

### Milestone 13 — Ingest queue, re-ingest, growing sources (point 6)
Goal: start and watch ingest from the web UI; keep changing files current.

- **Queue.** Use the existing `jobs` table as a queue: a worker process (`tkb worker`, also started by `tkb serve`) claims jobs with `SELECT … FOR UPDATE SKIP LOCKED` and runs them one at a time (GPU rule: one large model at a time). A library such as [Procrastinate](https://procrastinate.readthedocs.io/en/stable/discussions.html) is the alternative if the simple loop proves unreliable; decide in an ADR.
- **Web UI.** Sources tab: "Ingest folder" (path on the server), "Re-ingest" per source, and a queue panel with status, current stage and progress. API: `POST /ingest`, `POST /sources/{id}/reingest`, `GET /jobs`.
- **Changed files.** A file at a known path with a new sha256 replaces the old version: the new source records `replaces=<old id>`, and the old source's chunks are removed after the new ingest succeeds. Confirmed tasks and speaker labels from the old version are kept when they match.
- **Append-only CSV** (Granola exports and similar): split into one unit per row (meeting) and hash each row, so re-ingesting a grown CSV only processes new rows.
- **Granola.** Its built-in CSV has titles and summaries, no transcripts. Getting transcripts via the Granola API (Business/Enterprise) or MCP server is a separate decision; ask the users which plan they have before building a connector.

Acceptance: from the web UI, add a folder, watch it progress, and re-ingest one source; re-ingesting a CSV with 2 new rows processes only those rows; a changed PDF leaves exactly one current version.

### Milestone 14 — Entities: merge duplicates, graph view (point 4)
Goal: one node per real-world thing, and a way to look at the graph.

- **Normalized names.** Entity key = lowercase, Unicode-normalized, hyphens/underscores → spaces, collapsed whitespace. "lake surtan" and "lake-surtan" become one entity. Migration merges existing duplicates.
- **Candidate merges.** For remaining near-duplicates, compare name embeddings plus shared neighbours ([DEG-RAG](https://arxiv.org/pdf/2510.14271) reports large graph reductions this way). Above a high threshold merge automatically; between thresholds list them as suggestions.
- ~~Manage (merge/rename/alias UI)~~ — dropped: users do not need to edit entities (section 4). `tkb entities merges` lists automatic merges so a wrong one can be reported.
- **Graph view.** New "Graph" tab with [Cytoscape.js](https://github.com/cytoscape/js-graph-lib-comparison/blob/master/README.md) (works for hundreds to a few thousand nodes). Search an entity, show its neighbourhood (1–2 hops), click a node to see its sources, click an edge to see the evidence chunk.

Acceptance: no duplicate pairs among the test set's known variants; the graph tab shows an entity's neighbourhood and every edge opens its evidence; `tkb eval` rel-01/rel-02 do not regress.

Added after M14 (user request): **graph snapshots.** Save the entity graph to `data/backups/entities/` before automatic merging, `tkb entities snapshots` / `restore` / `snapshot` to switch back and forth, and `entities.auto_merge` to turn merging off (ADR 0015).

### Milestone 15 — Task editing, planning, week and calendar views (point 8)
Goal: plan thesis work by week, with the user in control.

- **Edit.** Add, edit and delete tasks in the UI and CLI: title, owner, due date, status, notes. New fields (migration): `start_date`, `estimate_hours`, `parent_id`, `origin` (`extracted` / `manual` / `plan`). Manual tasks need no evidence link; extracted tasks keep theirs.
- **Plan.** "Plan towards a deadline": the user picks a goal task with a due date and the tasks/sources it depends on. The LLM proposes parts (e.g. "Task A part 1"); **dates are computed in code**, working backwards from the deadline using estimates and available weeks. Proposed items are saved with `origin=plan`, unconfirmed, and appear only after the user accepts them.
- **LLM updates.** The model may *propose* changes to tasks (new due date found in a newer source, task done per a meeting). Proposals go to the review queue; the model never edits tasks directly.
- **Views.** Week view with ◀ ▶ buttons (default). Month calendar where hovering or clicking a week shows its tasks. Build both as a small grid in the existing static UI; use a library only if the grid becomes hard to maintain.

Acceptance: create a December deadline with two dependent tasks, get a weekly plan, accept it, and see it in both views; rejected plan items leave no trace in the task list; existing task eval rows do not regress.

### Milestone 16 — Models tab (point 3)
Goal: choose and download models per component from the UI.

- **Catalogue** `config/model_catalog.yaml`: per component (answer LLM, vision, embeddings, reranker, OCR, speech recognition, diarization): candidate models with download size, approximate VRAM, licence, local/online, and the role or backend key it sets. Curated by us; no automatic discovery.
- **Tab.** Shows current choice, installed or not, size, VRAM. "Use" downloads if needed (Ollama [`/api/pull`](https://mintlify.com/ollama/ollama/api/introduction) streams bytes completed/total; Hugging Face models via `huggingface_hub` with progress) and then writes the choice to `config/models.yaml` / `pipeline.yaml`.
- **Online models.** Paste an API key into a password field; the server writes it to `.env` and never returns it (UI shows only "set" / "not set"). Only allowed when the server listens on localhost. `local_only` sources keep using local models regardless.
- Changing the embedding model requires re-embedding everything; the UI says so and queues the job (M13) only after confirmation.
- Downloads run as jobs in the M13 queue, one at a time.

Acceptance: switch the speech model from the UI and the next audio ingest uses it; an API key set in the UI works and is never visible in any API response or log; `tkb eval` after switching the answer model is reported.

### Milestone 17 — Web sources (points 5 and 7) — gated by section 4
Goal: add web pages to the KB, and search the web on request, within the law.

Part A — URL ingest (do first):
- `tkb ingest <url>` and an "Add URL" field. Fetch once, store a snapshot (HTML + main text) under `data/raw/web/`, sha256 of the snapshot, locator = URL + anchor, retrieval date recorded.
- Main text via trafilatura or Docling HTML; [Crawl4AI](https://www.zenrows.com/blog/crawl4ai) only for pages that need JavaScript. Embedded images go through the image path; embedded audio/video only when it is a direct file link (no platform downloaders).
- Polite fetching: check `robots.txt` with [`urllib.robotparser`](https://docs.python.org/3/library/urllib.robotparser.html), descriptive User-Agent, one request at a time per host with a delay, back off on 429/503.
- Page clutter: navigation, footers and "more projects" cards must not become chunks (found in M11: the SideBy page's project carousel was retrieved for a SideBy question). Use main-content extraction (trafilatura) for web pages; check with gold question web-01.
- Videos embedded in a page (`<video src>` on the same site) are downloaded and go through the video path; the M11 saved copy has none of them.
- Whole-site ingest (`tkb ingest <url> --site`): same host only, `sitemap.xml` first, else follow links; `--max-pages` (default 50). Each page is its own source. No crawling of other hosts.

Part B — web search (only after the legal check):
- A `web` route used only when the user asks for it explicitly. Search via self-hosted [SearXNG](https://geekflare.com/guides/brave-search-api-alternatives/) (private) or an API such as Brave or Tavily ([comparison](https://blog.serverworks.co.jp/tavily-vs-brave-search-api-comparison)) — one backend, chosen in an ADR.
- Results become sources (Part A) marked `verification_status=unverified`; answers cite them like any other source.
- **Person lookups**: limited to what the legal check allows (e.g. public professional information: affiliation, publications, talks). Results stay linked to the person so `tkb forget-person` removes them.

Acceptance (A): a saved URL ingests and its citation opens the snapshot at the right section. Acceptance (B): defined after the legal check.

## 4. Open questions (answer before or during the milestone; record as ADRs)

### Answers (user, 2026-10-27)

1. **Legal:** private, non-commercial use outside the EU; GDPR compliance is not required.
   M17 Part B is no longer gated. Keep the technical safeguards anyway (robots.txt, rate
   limits, `forget-person`, `unverified` status).
2. **Granola:** free plan. The CSV holds the AI summary and the transcript of each meeting,
   ordered by time. M13 ingests the CSV (one unit per meeting row); no API connector.
3. **Entities:** users do not need to edit entities. M14 keeps automatic merging and the
   graph view; manual merge/rename/alias is dropped.
4. **Planning:** plans count holidays as normal days by default; the user can exclude
   single days or whole weeks.
5. **Test data:** an EPUB (gold question: "What is the difference between chapter 1 and
   chapter 2?") and the page <https://www.viviencai.com/projects/sideby>, also saved under
   `data/raw/` as "SideBy _ Vivien Cai Portfolio" (the saved copy has broken layout, so the
   live URL is the reference).
6. **Whole-site ingest:** wanted ("grab all pages of https://www.viviencai.com"). Added to
   M17 Part A as a bounded same-site crawl: read `sitemap.xml` first, else follow links on
   the same host only, obey robots.txt, with a page limit (default 50) and delay. Each page
   becomes its own source.

### Original questions

1. **Legal (M17 B).** The EDPB's draft [Guidelines 03/2026 on web scraping](https://www.edpb.europa.eu/system/files/2026-07/edpb_guidelines_2020603_webscraping_v1_en_0.pdf) say GDPR applies to scraped personal data, public availability alone is not a legal basis, and untargeted scraping is hard to justify ([summary](https://www.skwschwarz.de/en/news/ki-flash-edpb-guidelines-web-scraping-ai)). Background checks on people are profiling. Who is looked up, for what purpose, and on what legal basis? Needs a qualified answer, not ours.
2. **Granola (M13).** Which Granola plan do users have? Is the CSV (no transcripts) enough?
3. **Entities (M14).** Do users want to edit entities beyond merge/rename/alias?
4. **Planning (M15).** Should plans respect user availability (holidays, hours per week)? Default: no.
5. **Test data (M11, M17).** The user provides real EPUB and HTML/URL samples when M11 starts.

### Ideas found during milestones (not scheduled)

- **M14:** a trailing acronym keeps variants apart ("variational autoencoders (VAEs)" vs
  "Variational Autoencoders", "Variational lower bound (VLB)" vs "variational lower bound").
  Dropping a parenthesised acronym in the variant check would merge them; not in M14's task list.
- **M15:** tasks you type or plan have no evidence passage, so the task *answer* path (which
  cites a chunk for every row) does not use them yet; they show in the Tasks tab and CLI only.
  Citing "your task list" as a source would need a new citation kind.

## 5. Research notes

- Open WebUI pipes and reserved arguments: <https://docs.openwebui.com/features/extensibility/plugin/development/reserved-args>
- RAPTOR (ICLR 2024): <https://proceedings.iclr.cc/paper_files/paper/2024/hash/8a2acd174940dbca361a6398a4f9df91-Abstract-Conference.html>
- Langfuse vs Phoenix: <https://www.zenml.io/blog/langfuse-vs-phoenix>
- R2R graph deduplication: <https://r2r-docs.sciphi.ai/cookbooks/advanced-graphrag>
- Graph visualisation libraries: <https://memgraph.com/blog/you-want-a-fast-easy-to-use-and-popular-graph-visualization-tool>
- Robust EPUB parsing: <https://dev.to/jacob_gong/how-we-built-a-robust-epub-parsing-and-rebuilding-pipeline-in-python-29f9>
- Docling vs Crawl4AI: <https://toolhalla.ai/compare/docling-vs-crawl4ai>
- Polite crawling: <https://scrappey.com/qa/crawling/what-is-polite-crawling>
- Granola export tools: <https://github.com/moona3k/granola-export>
- Ollama API: <https://lumadock.com/tutorials/ollama-api-guide>
