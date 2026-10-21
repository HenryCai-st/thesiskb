# Architecture

ThesisKB is one Python package (`src/thesiskb`), one PostgreSQL database (with pgvector), a folder
of stage outputs per input file, and local models behind one gateway. Decisions are recorded in
[docs/adr](adr/).

```
 data/raw/  (your files, never modified)
     │  tkb ingest
     ▼
 detect ── sha256, modality, dedupe ──► sources
     │
     ▼  parse: one GPU batch per parser (ingest/parsers.py registry)
 documents  Docling ─────────────► processed/<sha>/docling.json
 images     rotation by VLM logprobs, VLM read + Tesseract ─► image.json, upright.jpg
 audio      ffmpeg → faster-whisper → pyannote ─► audio.wav, asr.json, diarization.json
 video      audio path + scene keyframes → VLM captions ─► frames/, frames.json
     │
     ▼  chunk (CPU; `tkb reindex` re-runs it)
 Blocks with locators {page,bbox} | {t_start,t_end,speaker} | {image,region} | {line}
   → section-aware chunks (~600 tokens, 12% overlap; 60–120 s windows for recordings)
   → segments + chunks rows
     │
     ▼  embed: bge-m3 vectors + per-language tsvector
     ▼  extract: one structured LLM call per chunk → tasks, statements, entities/relations
        (+ plug-in extractors); verbatim evidence quote required; dates resolved in code
     ▼
 ┌──────────────────────────── PostgreSQL ─────────────────────────────┐
 │ sources  segments  chunks(+vector, tsv)  people  speakers  tasks    │
 │ statements  entities  entity_mentions  relations  jobs  eval_runs   │
 └─────────────────────────────────────────────────────────────────────┘
     ▲
 question ─► router (rules + LLM) ─► paths (context providers, answer/paths.py)
     general     hybrid search: pgvector ∪ FTS, RRF, cross-encoder rerank
     literature  sub-queries → search papers → rerank → per-chunk evidence summaries
     person      their statements + chunks where they speak / papers they wrote
     task        open tasks from SQL, by due date
     entity      seeds + type + co-occurrence + 2-hop relations (recursive SQL)
     ▼
 merge + renumber passages [S1..Sn] → one grounded generation (role: answer)
     ▼
 verify: citations exist and were shown; every sentence cited; per-claim support judge
 pinpoint: sentence → transcript line (timestamp, speaker, nearest keyframe)
     ▼
 tkb ask · web UI + REST + /v1 OpenAI-compatible (tkb serve) · MCP tools (tkb mcp)
```

## Modules

| Package | Responsibility |
|---|---|
| `core/` | settings (`.env`, `TKB_*`), YAML config, DB engine, ORM models |
| `llm/` | the only place that calls models: LiteLLM gateway (`chat`, `chat_with_logprobs`, `embed`, `rerank`, `role_for` for local-only content), backends registry for reranker / ASR / diarization, Ollama VRAM housekeeping |
| `ingest/` | detect, parser registry + built-in parsers (documents, images, audio, video), chunking, metadata (arXiv), dates, extraction (+ extractor registry), pipeline stages and job table |
| `retrieval/` | hybrid search, rerank, router, graph-lite entity lookups |
| `answer/` | answer paths, grounded generation, verification, task path |
| `api/` | FastAPI app (REST, OpenAI-compatible `/v1`, file/page/frame serving) and the static web UI |
| `mcp/` | MCP server on stdio |
| `evaluation.py` | `tkb eval`: gold questions, metrics, `eval_runs` |
| `cli.py` | the `tkb` command |

## Invariants (see also AGENTS.md)

- Every chunk has a `source_id` and a locator; every extracted task/statement/relation has a
  verbatim evidence quote from its chunk.
- Stages are idempotent and keyed by sha256; stage outputs on disk count as done, so a lost DB is
  rebuilt without GPU work.
- One heavy GPU model family at a time: parse runs per parser batch and unloads Ollama models
  before Docling/Whisper/pyannote; questions and ingestion share one lock in the API.
- Schema changes only via Alembic (`src/thesiskb/migrations`).
- Content of a `local_only` source only reaches models for which `RoleConfig.is_local` holds.

## Extending

| To add | Implement | Register |
|---|---|---|
| a file type | `ingest.parsers.Parser` (`parse` batch → outputs, `chunk` → `pipeline.write_blocks`) | `register_parser(...)` |
| a model for a chat/vision/embedding role | nothing — change `model:` in models.yaml (any LiteLLM provider) | — |
| a reranker / ASR / diarization backend | factory returning an object with the documented methods | `@register_backend(kind, name)`, then `backend: name` in models.yaml |
| an extraction schema | `ingest.extract.Extractor` (`key`, `doc`, `item`, `write`) | `register_extractor(...)` |
| an answer path | a provider `(session, gateway, question, cfg, filters, route) -> Ctx` | `answer.paths.PROVIDERS[name]` + router rule |

Plug-in modules are imported from `plugins:` in `config/pipeline.yaml` or the
`thesiskb.plugins` entry-point group. Working examples: [examples/plugins](../examples/plugins).
