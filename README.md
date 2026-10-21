# ThesisKB

A private, multimodal knowledge base with verified, cited retrieval — papers, photos of
whiteboards, meeting recordings and interview videos, all searchable with answers that cite the
exact page, image or timestamp.

> Status: **Milestone 5 (extraction)**. PDFs, photos, audio and video are indexed and answerable
> with page, image or timestamp + speaker citations. Tasks, people's statements and entities are
> extracted into tables; task questions are answered from SQL.
> See [BUILD_GUIDE.md](BUILD_GUIDE.md) for the full plan and [AGENTS.md](AGENTS.md) for rules
> coding agents follow.

## Quickstart (Windows prototype)

Prerequisites: Docker Desktop, [Ollama](https://ollama.com), ffmpeg,
[Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) with `deu` + `eng` language packs,
[uv](https://docs.astral.sh/uv/), an NVIDIA GPU (16 GB VRAM recommended). On Windows Tesseract is
not added to PATH: set `TKB_TESSERACT_CMD` (and `TKB_TESSDATA_DIR` if the language packs are in
a user folder) in `.env`.

```bash
uv sync --extra local        # create .venv incl. Docling, CUDA torch, reranker (~6 GB)
cp .env.example .env         # adjust if needed
docker compose up -d         # Postgres + pgvector
uv run tkb db upgrade        # create the schema
ollama pull qwen3:14b        # answer/extraction model
ollama pull bge-m3           # embeddings
ollama pull qwen2.5vl:7b     # vision (photos, handwriting)
uv run tkb doctor            # check everything
```

## Usage

```bash
uv run tkb ingest data/raw               # idempotent: re-running skips finished stages
uv run tkb sources                       # index: original filename ↔ sha256, title, stages
uv run tkb ask "How is diffusion model developing?" --type paper
uv run tkb eval                          # gold questions in eval/questions.yaml
uv run tkb open <chunk-id>               # open a cited file (photos open upright)
uv run tkb speakers list <file>          # who speaks when, with name suggestions
uv run tkb speakers label <file> SPEAKER_03 "Will.i.am" --alias "Will I.M"
uv run tkb reindex <file>                # redo chunk + embed from cached outputs (no GPU)
uv run tkb tasks                         # open tasks by due date, each linked to its evidence
uv run tkb tasks review                  # confirm / edit / delete extracted tasks
uv run tkb ask "Show me my thesis schedule" --path task   # answered from the tasks table
uv run tkb statements --person "Will.i.am"
```

Recordings are transcribed with faster-whisper and split into speakers with pyannote; answers
show each quote's speaker and timestamp (`↳ S1 at 0:22 Will.i.am`) and, for video, the nearest
keyframe. Speaker separation needs a Hugging Face read token (`TKB_HF_TOKEN` in `.env`) and
accepted terms for `pyannote/speaker-diarization-community-1`. See
[ADR 0005](docs/adr/0005-audio-video.md).

Photos are turned upright automatically (the rotation the vision model reads most
confidently), transcribed by the vision model, and also OCR'd with Tesseract; both readings are
kept in `data/processed/<sha256>/image.json`. See [ADR 0004](docs/adr/0004-image-reading.md).

Original files are never renamed or copied. Each file is identified by the SHA-256 of its
content; stage outputs live in `data/processed/<sha256>/`, and moving or renaming a file only
updates its path in the `sources` table. Answers cite passages as `[S1]`…; the source list shows
the original filename, page range and chunk ID. Sentences without a citation, or that the
support check finds unsupported, are flagged with ⚠.

Set `OLLAMA_MAX_LOADED_MODELS=1` and `OLLAMA_KEEP_ALIVE=5m` for Ollama so only one large
model sits in VRAM at a time.

## Development

```bash
uv run pytest                # CPU-only, models mocked; DB tests use a separate *_test database
uv run ruff check . && uv run ruff format .
uv run pre-commit install    # lint on every commit
```

Models per role live in [config/models.yaml](config/models.yaml); swap a local model for a cloud
one by changing its LiteLLM name (e.g. `openai/gpt-...`).

## Layout

```
config/            models.yaml, pipeline.yaml, people.yaml
src/thesiskb/
  core/            settings, YAML config, DB engine, ORM models
  llm/             LiteLLM gateway: chat(), embed(), rerank(); Ollama VRAM housekeeping
  ingest/          detect (hash/dedupe), documents (Docling), images (VLM + OCR),
                   audio (Whisper + pyannote), video (keyframes), chunking, metadata,
                   extract (tasks/statements/entities), dates, pipeline
  retrieval/       hybrid search (pgvector + FTS, RRF), cross-encoder rerank
  answer/          cited answer generation, task path (SQL), citation + support checks
  evaluation.py    `tkb eval`
  migrations/      Alembic migrations
  doctor.py        environment checks
  cli.py           `tkb` command
docs/adr/          architecture decision records
data/              raw inputs and processed outputs (git-ignored)
```

## License

MIT. Never commit private data, recordings or personal names — `data/` and `.env` are git-ignored.
