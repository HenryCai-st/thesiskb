# ThesisKB — Build Guide

A private, multimodal knowledge base with verified, cited retrieval. Built first as a
Windows prototype, published on GitHub as a reusable framework, then deployed to a Linux server.

This document is written for both humans and coding agents. Each milestone has a goal,
concrete tasks and acceptance criteria. Work through milestones in order; do not start
a milestone until the previous one passes its acceptance criteria.

---

## 0. Context and goals

### First use case: a bachelor thesis

Inputs:
- Papers and books (PDF, EPUB)
- Photos of whiteboards, post-its and handwritten drafts
- Audio recordings of meetings
- Videos of interviews and product demonstrations

Starting volume: 100–1,000 pages of documents, ~10 hours of audio/video. Expect this to change.

Target questions (each needs a different mechanism — see section 3):

| # | Example question | Type | Mechanism |
|---|---|---|---|
| Q1 | "What is state of the art for designing good diffusion models?" | Literature synthesis | Retrieval + rerank + evidence summaries + citations |
| Q2 | "What does Professor X prefer?" | Person-centric aggregation | Filter by person/speaker + entity graph + quotes with timestamps |
| Q3 | "Who does what until when?" | Structured facts | Extraction into a `tasks` table at ingestion time, then SQL. **Not RAG.** |

### Non-goals for v1
- No fine-tuning of models.
- No multi-user permissions (single user / small team, behind VPN).
- No automatic web crawling. External papers are added by the user (optional arXiv/Semantic Scholar import comes later).

### Design principles
1. **Every answer is traceable.** Every chunk points to a source file and an exact location (page + bbox, or timestamp range, or image region). Answers must cite chunk IDs; uncited claims are flagged.
2. **"Verified" is metadata, not model magic.** Sources carry a `verification_status`; retrieval can be restricted to verified sources.
3. **Stages are idempotent and cached.** Every file is keyed by its SHA-256. Each pipeline stage writes its output to disk; re-running skips completed work.
4. **One heavy GPU model at a time.** 16 GB VRAM cannot hold the LLM, Whisper and a vision model together. Ingestion runs in stage batches.
5. **The LLM is swappable.** All model calls go through one OpenAI-compatible interface (local Ollama or a cloud API via LiteLLM).
6. **Reuse before building.** Wrap proven libraries; only write glue, data model and evaluation.

---

## 1. Prior art — check before building

People have built most of the pieces. Spend Milestone 0 trying these, then decide what to reuse.

| Project | What it is | How to use it here |
|---|---|---|
| **RAGFlow** | Full self-hosted RAG platform: deep document parsing, hybrid retrieval, reranking, citations, optional GraphRAG, local models via Ollama | Baseline. If it answers Q1 well out of the box, borrow its ideas or even use it as the doc layer. Heavy Docker stack. |
| **kotaemon** | Clean, customizable RAG UI + framework, local or API models, optional LightRAG/GraphRAG | Lighter baseline; good reference for UI and citation display. |
| **LightRAG** (HKUDS) | Graph + vector retrieval, server with API | Candidate for the knowledge-graph layer (Milestone 6). |
| **RAG-Anything** (HKUDS) | Multimodal RAG on top of LightRAG: images, tables, equations in documents | Reference for parsing figures/tables/equations in papers. |
| **VideoRAG** (HKUDS) | RAG over long videos | Reference for the video pipeline. |
| **PaperQA2** (FutureHouse) | Agentic RAG specialized for scientific papers: metadata-aware embeddings, LLM reranking, contextual summarization, cited answers, contradiction detection; any LiteLLM model | Best reference for Q1. Use its approach (or call it directly) for the literature path, and as a quality benchmark. |
| **Docling** / **MinerU** / **Marker** | PDF/Office → structured Markdown/JSON with layout, tables, formulas, page numbers | Document parsing. |
| **faster-whisper**, **WhisperX**, **pyannote** | Transcription, word-level timestamps, speaker diarization | Audio/video. |
| **Open WebUI** | Chat UI for Ollama / OpenAI-compatible endpoints | Optional front end. |

**Decision rule after Milestone 0:**
- Build our own thin framework (this guide) because none of the above covers Q3 (task extraction) and Q2 (person-centric, cross-modal) well, and a thesis-oriented, documented, hackable repo is itself the deliverable.
- Reuse Docling, faster-whisper/WhisperX, embeddings/rerankers, pgvector, LiteLLM, and optionally LightRAG and PaperQA2 as libraries.

---

## 2. Hardware budget (prototype PC)

Spec: 16 GB VRAM, 32 GB RAM, ~50 GB disk (expandable).

### VRAM plan (load one at a time)

| Model role | Suggested class (check current releases) | Approx. VRAM |
|---|---|---|
| Answer LLM | ~14B dense at Q4 (e.g. Qwen3-14B), or ~30B MoE with ~3B active at Q4 (e.g. Qwen3-30B-A3B) | 9–16 GB (+ context) |
| Extraction LLM (entities, tasks) | Same model, or a 7–9B model for speed | 5–9 GB |
| Vision LLM (captions, handwriting) | 7–12B vision-language model (e.g. Qwen-VL or Gemma multimodal class) | 6–12 GB |
| ASR | Whisper large-v3-turbo via faster-whisper (int8/fp16) | 3–6 GB |
| Diarization | pyannote | ~1–2 GB |
| Embeddings | bge-m3 (multilingual, good for German + English) | ~2 GB |
| Reranker | bge-reranker-v2-m3 | ~1–2 GB |

Embeddings + reranker can stay loaded alongside the answer LLM if the LLM is ≤ ~12 GB.
Model names change fast — pin whatever is best at your VRAM size when you start, and record it in `config/models.yaml`.

### RAM
- WSL2 by default takes up to half of host RAM. Set it explicitly (see 4.1): 20–24 GB for WSL, the rest for Windows.
- Postgres + pgvector for this data size needs < 2 GB.

### Disk — 50 GB is too tight. Plan for ≥ 200 GB.

| Item | Size |
|---|---|
| Docker images (Postgres, optional Neo4j/Open WebUI) | 3–8 GB |
| Python env + CUDA wheels (torch etc.) | 8–12 GB |
| LLMs (one answer model + one vision model) | 15–30 GB |
| Whisper, embeddings, reranker, Docling models | 4–6 GB |
| Raw data: 1,000 pages PDFs | 0.5–2 GB |
| Raw data: 10 h video at 1080p | 10–30 GB |
| Processed data (audio WAV, keyframes, JSON) | 3–10 GB |

If disk can't grow yet: keep raw video on an external drive; the pipeline only needs the extracted audio (16 kHz mono, ~1 GB / 10 h) and keyframes.

### Expected processing times (rough, single 16 GB GPU)
- 1,000 PDF pages through Docling: tens of minutes.
- 10 h audio through faster-whisper turbo: roughly 15–40 min; diarization adds similar.
- Captioning a few hundred images/keyframes: tens of minutes.
- Indexing: minutes.

---

## 3. Architecture

```
            ┌─────────── inputs (data/raw/) ───────────┐
            │  pdf/epub   images   audio   video        │
            └──────────────────┬───────────────────────┘
                               ▼
        Ingestion pipeline (stage-based, idempotent, GPU-aware)
   detect → parse/transcribe/caption → normalize → chunk → embed
                          │                    │
                          ▼                    ▼
                 extract (LLM):         index (pgvector + FTS)
                 entities, people,
                 tasks, claims
                          │
                          ▼
   ┌────────────── PostgreSQL (single DB) ─────────────────┐
   │ sources │ segments │ chunks+embeddings │ people │       │
   │ entities/relations │ tasks │ jobs │ eval runs           │
   └───────────────────────────────────────────────────────┘
                          ▲
       Query layer (FastAPI): router → {literature RAG | person RAG | task SQL}
       → hybrid search → rerank → answer with citations → citation check
                          ▲
        Interfaces: CLI, REST API, MCP server, optional Open WebUI
                          ▲
        LLM gateway (LiteLLM): Ollama locally / cloud API
```

### Why Postgres for everything (v1)
- pgvector for embeddings, built-in full-text search for BM25-style keyword search → hybrid search in one DB.
- Relational tables for tasks, people, sources and verification status (Q3 is plain SQL).
- One container, trivial to back up and to move to Linux.
- At 20k–100k chunks, pgvector is more than fast enough. Revisit Qdrant only past millions of chunks.
- Knowledge graph: start with `entities` / `relations` tables in Postgres. Add LightRAG or Neo4j in Milestone 6 only if relation questions fail.

### Communication between components
- **Prototype:** no message broker. A `jobs` table in Postgres (status per file per stage) plus a single worker process that picks the next job for the currently loaded model. This is enough for one machine.
- **Server:** same code; optionally swap the job table for Redis + RQ/Celery if you want workers on several machines. Keep the `JobQueue` interface abstract so this is a config change.
- **Model calls:** HTTP to Ollama (or vLLM on the server) through LiteLLM. Never import model-specific SDKs in pipeline code.
- **Tools for agents/LLM clients:** an MCP server exposing `search`, `get_chunk`, `get_source`, `list_tasks`, `person_profile`.

### Query routing (one router, three paths)
1. **Literature path (Q1):** query rewriting into sub-queries → hybrid search restricted to `source_type in (paper, book)` → rerank → per-chunk evidence summary relevant to the question (PaperQA2-style) → answer with citations → citation check. Prompt must state the answer only covers documents in the KB; "SOTA" is only as current as what was ingested.
2. **Person path (Q2):** resolve person via `people` + aliases → retrieve chunks where `speaker = person` or `author = person` or entity link to person → group by source and theme → answer with quotes (page or timestamp) for each claim.
3. **Task path (Q3):** SQL over `tasks` (owner, due date, status) → LLM only formats the result; each row links to the transcript segment it came from.

---

## 4. Environment setup

### 4.1 Windows prototype — develop inside WSL2
Do all development inside WSL2 (Ubuntu). This makes the later move to Linux nearly free, and most of the ML tooling (pyannote, Docling GPU, vLLM) is far less painful on Linux.

1. Install/update NVIDIA Windows driver (it provides CUDA inside WSL2; do **not** install a Linux GPU driver inside WSL).
2. `wsl --install -d Ubuntu-24.04`
3. Create `%UserProfile%\.wslconfig`:
   ```ini
   [wsl2]
   memory=22GB
   swap=8GB
   ```
   then `wsl --shutdown`.
4. Docker Desktop with the WSL2 backend (or Docker Engine inside WSL). Verify `docker run --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`.
5. Keep the repo and data **inside the WSL filesystem** (`~/thesiskb`), not on `/mnt/c` — file I/O across the boundary is very slow.
6. If the Windows disk is small, move the WSL distro to another drive (`wsl --export` / `--import`) or point `data/` at a mounted larger drive.
7. Ollama: install inside WSL (Linux install script) so the setup matches the server. Set `OLLAMA_MAX_LOADED_MODELS=1` and `OLLAMA_KEEP_ALIVE=5m`.
8. System packages: `ffmpeg`, `tesseract-ocr` (+ `deu`, `eng` language packs), `python3.11+`, `uv` for Python env management.
9. Hugging Face token (for pyannote gated models): accept the model terms on Hugging Face first.

### 4.2 Linux server (later)
Same `docker-compose.yml` with a `gpu` profile. Swap Ollama for vLLM if you need concurrency. Put everything behind Tailscale/WireGuard; expose only the UI/API with auth. Nightly `pg_dump` + rsync of `data/raw`.

---

## 5. Repository layout

```
thesiskb/
├── README.md                  # what it is, quickstart, screenshots
├── AGENTS.md                  # rules for coding agents (see section 9)
├── LICENSE                    # e.g. Apache-2.0 or MIT
├── docker-compose.yml         # postgres(+pgvector), optional: open-webui, neo4j
├── .env.example               # DB URL, HF token, API keys (never commit .env)
├── config/
│   ├── models.yaml            # model per role, quantization, endpoint
│   ├── pipeline.yaml          # chunk sizes, OCR langs, keyframe interval, etc.
│   └── people.yaml            # known people + aliases (seed for person resolution)
├── data/                      # git-ignored
│   ├── raw/{docs,images,audio,video}/
│   └── processed/<sha256>/    # per-file stage outputs (json, wav, frames)
├── src/thesiskb/
│   ├── core/                  # config, logging, db session, models (pydantic + SQLAlchemy)
│   ├── llm/                   # LiteLLM gateway, embeddings, reranker, prompt templates
│   ├── ingest/
│   │   ├── detect.py          # modality detection, hashing, dedupe
│   │   ├── documents.py       # Docling wrapper
│   │   ├── images.py          # OCR + VLM caption
│   │   ├── audio.py           # faster-whisper/WhisperX + diarization
│   │   ├── video.py           # ffmpeg audio + scene keyframes → audio.py + images.py
│   │   ├── chunking.py        # unified chunker with locators
│   │   ├── extract.py         # LLM extraction: people, entities, tasks, claims
│   │   └── pipeline.py        # stage runner, GPU-aware scheduling, job table
│   ├── retrieval/
│   │   ├── hybrid.py          # pgvector + FTS, reciprocal rank fusion
│   │   ├── rerank.py
│   │   └── router.py          # literature / person / task paths
│   ├── answer/
│   │   ├── generate.py        # grounded answer with chunk-ID citations
│   │   └── verify.py          # citation existence + support check
│   ├── api/                   # FastAPI app (REST + OpenAI-compatible /v1/chat)
│   ├── mcp/                   # MCP server exposing KB tools
│   └── cli.py                 # `tkb ingest`, `tkb ask`, `tkb tasks`, `tkb eval`, `tkb verify`
├── eval/
│   ├── questions.yaml         # gold questions with expected sources
│   └── run_eval.py
├── tests/                     # unit tests per module, small fixture files
└── docs/                      # architecture, decisions (ADR), hardware notes
```

---

## 6. Data model (PostgreSQL)

```sql
-- One row per input file
sources(
  id uuid pk, sha256 text unique, path text, modality text,         -- pdf|epub|image|audio|video
  source_type text,                                                -- paper|book|note|meeting|interview|demo
  title text, authors text[], published date, origin text,          -- DOI, URL, "recorded 2026-10-14"
  language text,
  verification_status text default 'unverified',                   -- unverified|verified|disputed
  verified_by text, verified_at timestamptz,
  consent_note text,                                               -- for recordings of people
  ingested_at timestamptz, pipeline_version text
)

-- Normalized units before chunking (a page, a transcript turn, an image region, a keyframe)
segments(
  id uuid pk, source_id fk, kind text,                             -- page|turn|image|frame
  text text, speaker_id fk null,
  locator jsonb                                                    -- {page, bbox} | {t_start, t_end} | {frame_ts, region}
)

chunks(
  id uuid pk, source_id fk, segment_ids uuid[],
  text text, context_header text,                                  -- e.g. "Paper: X (2024) › Section 3.2"
  embedding vector(1024), tsv tsvector,
  speaker_id fk null, locator jsonb
)

people(id uuid pk, name text, aliases text[], role text)           -- e.g. supervisor, student, interviewee
speakers(id uuid pk, source_id fk, diarization_label text, person_id fk null)  -- SPEAKER_00 → Prof. X (manual or confirmed)

entities(id uuid pk, name text, type text, aliases text[])
relations(id uuid pk, subj fk, pred text, obj fk, chunk_id fk, confidence real)

tasks(
  id uuid pk, description text, owner_person_id fk null, owner_raw text,
  due_date date null, due_raw text, status text default 'open',
  source_chunk_id fk, confidence real,
  confirmed boolean default false                                  -- human review
)

jobs(id, source_id, stage, status, error, started_at, finished_at)
eval_runs(id, git_sha, config_hash, results jsonb, created_at)
```

Indexes: HNSW on `chunks.embedding`, GIN on `chunks.tsv`, btree on `tasks.due_date`, `chunks.speaker_id`, `sources.source_type`.
Use a `simple` or language-specific FTS configuration per chunk language (German and English differ).

---

## 7. Milestones

Each milestone: **Goal → Tasks → Acceptance criteria.** Commit and tag at the end of each.

### Milestone 0 — Baseline and gold questions (1–2 days)
Goal: know what "good" looks like before writing code.
- Collect a **small fixture set**: 5 papers, 1 book chapter, 10 photos (incl. 3 post-its, 2 handwritten), 1 meeting recording (~20 min), 1 interview video (~15 min).
- Write `eval/questions.yaml`: ~25 questions covering Q1/Q2/Q3, each with expected source(s) and, for Q3, the expected task rows.
- Try RAGFlow or kotaemon (and PaperQA2 on the papers) with a local model. Note what works and what fails.

Acceptance: gold set exists; a short `docs/baseline.md` records baseline results.

### Milestone 1 — Skeleton and infrastructure
- Repo layout, `uv` project, `ruff` + `pytest`, pre-commit.
- `docker-compose.yml` with Postgres + pgvector; Alembic migrations for section 6.
- `config/*.yaml` loading via pydantic-settings.
- LLM gateway: LiteLLM wrapper with `chat()`, `embed()`, `rerank()`; model per role from config.
- Ollama running in WSL; pull answer model + embeddings.

Acceptance: `tkb doctor` checks DB, GPU (`nvidia-smi`), Ollama, ffmpeg, tesseract and prints model/VRAM summary. Tests pass in CI (CPU-only, mocked models).

### Milestone 2 — Documents end-to-end (text RAG MVP)
- `detect.py`: hash, dedupe, modality detection, register `sources`.
- `documents.py`: Docling → per-page segments with bbox; keep tables as Markdown; capture figure captions; read DOI/title/authors (fallback: first-page heuristics, optional Crossref lookup).
- `chunking.py`: structure-aware chunks (~400–800 tokens, overlap ~10–15%), never cross section boundaries; add `context_header`.
- Embeddings with bge-m3; FTS column.
- `hybrid.py`: vector + FTS, reciprocal rank fusion; `rerank.py` with cross-encoder; top-k 30 → rerank → 8.
- `generate.py`: prompt requires `[chunk:<id>]` citations per sentence; refuse when evidence missing.
- `verify.py`: every cited ID exists and was in context; second LLM pass (or NLI) checks each sentence is supported; unsupported sentences are marked.
- CLI: `tkb ingest data/raw/docs`, `tkb ask "..."` prints answer + sources with page numbers.

Acceptance: on Q1 gold questions, ≥ 80% of answers cite at least one expected source; zero citations to non-existent chunks; re-running ingest on the same files does no work.

### Milestone 3 — Images, photos, post-its
- `images.py`: OCR (Tesseract or PaddleOCR) + VLM description. For handwriting, VLM transcription is usually better than OCR — store both and keep the higher-confidence one.
- Post-its: ask the VLM for a structured list of notes (`[{text, color, position}]`) so each note becomes its own segment.
- Store image path + region in `locator`; UI/CLI shows the image next to the citation.
- GPU scheduling: run all images in one batch after unloading the answer LLM.

Acceptance: questions about content on photos are answerable with a citation pointing to the right image.

### Milestone 4 — Audio and video
- `audio.py`: faster-whisper (large-v3-turbo) with word timestamps; diarization (pyannote, or WhisperX which combines both); language auto-detect (German/English).
- Segments = speaker turns with `{t_start, t_end}`; chunk by consecutive turns (~60–120 s windows).
- `speakers` table: unknown `SPEAKER_00` labels; `tkb speakers label <source>` CLI to map labels to `people` (manual is fine for v1; voice embeddings for auto-matching later).
- `video.py`: ffmpeg → 16 kHz mono WAV → `audio.py`; scene-change keyframes (`ffmpeg -vf "select='gt(scene,0.3)'"`, plus one frame every N seconds as fallback) → `images.py`; align frame captions to the transcript timeline.
- Product demo videos: keyframe captions describe screens/UI; tie them to what was said at that timestamp.

Acceptance: "What did Prof. X say about Y?" returns quotes with correct speaker and timestamps (±5 s); video answers cite timestamp and keyframe.

### Milestone 5 — Extraction: people, tasks, claims
- `extract.py` runs the LLM over meeting/interview chunks with a strict JSON schema (use Ollama structured outputs / JSON schema mode):
  - **Tasks:** `{description, owner, due, evidence_quote}`. Resolve `owner` against `people` aliases; resolve relative dates ("next Friday") against the recording date in `sources`.
  - **Person statements/preferences:** `{person, topic, stance, evidence_quote}`.
  - **Entities/relations** (methods, datasets, concepts, people) for papers and transcripts.
- Post-its and handwritten drafts can also produce tasks.
- Every extracted row stores `source_chunk_id` and `confidence`; the evidence quote must appear verbatim in the chunk, otherwise discard.
- `tkb tasks` lists open tasks by due date; `tkb tasks review` lets the user confirm/edit (sets `confirmed=true`).

Acceptance: on gold Q3 rows, recall ≥ 80%, no task without a valid evidence link; Q3 answers come from SQL, not free generation.

### Milestone 6 — Router, person profiles, optional graph
- `router.py`: classify the question (LLM or rules) into literature / person / task / general; allow multi-path.
- Person path: profile = statements + authored papers + speaker quotes, grouped by topic; answer cites each claim.
- Literature path: adopt PaperQA2-style steps (sub-queries, per-chunk relevance summaries, then synthesis). Optionally call PaperQA2 directly on the paper subset and compare.
- Graph: only if gold questions about relations fail. Options: query `relations` table with recursive SQL; or integrate LightRAG with a Postgres backend; or Neo4j.

Acceptance: all three gold question types meet their targets; `tkb eval` shows no regression vs Milestone 5.

### Milestone 7 — Interfaces
- FastAPI: `/ingest`, `/ask`, `/tasks`, `/sources/{id}`, `/chunks/{id}` + an OpenAI-compatible `/v1/chat/completions` so Open WebUI (or any client) can use the KB as a "model".
- MCP server with tools `search`, `get_chunk`, `get_source`, `list_tasks`, `person_profile`.
- Simple UI (Open WebUI, or a small Streamlit/Gradio app) that renders citations as links to page/timestamp/image.

Acceptance: ask from the UI, click a citation, see the exact page or jump to the timestamp.

### Milestone 8 — Framework polish and GitHub release
- README with quickstart (WSL2 and Linux), hardware table, screenshots.
- `docs/architecture.md`, ADRs for key choices (pgvector over Qdrant, no broker in v1, etc.).
- Plugin points: `Parser` interface per modality, `ModelProvider` per role, `Extractor` per schema — so others can add a modality or swap a model without touching the core.
- CI: lint, unit tests with mocked models, a tiny integration test with a 1-page PDF and 10 s audio clip on CPU.
- Example config for "API mode" (cloud LLM, local KB) and "fully local".
- No real data, recordings or personal names in the repo; fixtures are synthetic or openly licensed.

### Milestone 9 — Linux server deployment
- `docker compose --profile gpu up` on the server; Ollama or vLLM container.
- Tailscale/WireGuard; API auth (token); TLS if exposed.
- Backups: nightly `pg_dump`, `data/raw` sync, restore test.
- Move data: `pg_dump`/restore + copy `data/`; `sha256` keys make re-ingest unnecessary.

---

## 8. Verification and evaluation

- **Source verification:** `tkb verify <source> --status verified --by <name>`. Retrieval flag `--verified-only`. Show status next to every citation.
- **Answer verification:** (1) citation IDs exist and were in context; (2) each sentence is supported by its cited chunk (LLM judge or NLI model); (3) unsupported sentences shown as such, or the answer is regenerated once.
- **Evaluation:** `tkb eval` runs `eval/questions.yaml`, records retrieval hit rate (expected source in top-k), citation precision, answer support rate, task recall, latency. Store in `eval_runs` with git SHA and config hash. Run after every milestone and every model change.
- **Thesis integrity:** cite the original papers in the thesis, never the system's output. Treat Q1 answers as a reading guide.

---

## 9. AGENTS.md (copy into the repo root for coding agents)

```markdown
# Rules for coding agents
- Read BUILD_GUIDE.md first. Work on one milestone at a time; check its acceptance criteria before moving on.
- Python 3.11+, uv, ruff, pytest, type hints, pydantic models at module boundaries.
- All model calls go through src/thesiskb/llm/ (LiteLLM). Never import a vendor SDK elsewhere.
- Every pipeline stage must be idempotent, keyed by source sha256, and write outputs to data/processed/<sha256>/.
- Every chunk must have a locator. Never create chunks without source_id and locator.
- Never load two large GPU models at once; use the pipeline scheduler.
- Tests must run on CPU with mocked models. Put heavy/GPU tests behind a `gpu` pytest marker.
- Never commit data/, .env, model weights, recordings or personal information.
- Schema changes only via Alembic migrations.
- When a design decision is made, add a short ADR in docs/adr/.
- Run `tkb eval` after changes to retrieval, prompts or models and report the diff.
```

---

## 10. Privacy, consent and legal (Germany / EU)

- Meeting and interview recordings contain personal data, and voice-based diarization processes biometric-like data. Get consent from participants, note it in `sources.consent_note`, and allow deleting a person's data (`tkb forget-person`).
- In API mode, retrieved chunks are sent to the API provider with each question. Add a `local_only` flag on sources; the router must use the local model whenever any retrieved chunk is `local_only`.
- Respect licenses of papers/books: keep them private; do not publish the index or chunks.
- Check your university's rules on using AI tools for thesis work.

---

## 11. Open decisions (record answers as ADRs)
1. Answer model: 14B dense vs ~30B MoE — decide from `tkb eval` results on your hardware.
2. Handwriting: OCR vs VLM — decide from Milestone 3 fixtures.
3. Graph layer needed at all? — decide after Milestone 6 gold results.
4. Keep own literature path or delegate to PaperQA2?
5. UI: Open WebUI vs custom.
