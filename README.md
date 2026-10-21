# ThesisKB

A private, multimodal knowledge base with verified, cited retrieval — papers, photos of
whiteboards, meeting recordings and interview videos, all searchable with answers that cite the
exact page, image or timestamp.

> Status: **Milestone 1 (skeleton & infrastructure)**. See [BUILD_GUIDE.md](BUILD_GUIDE.md) for
> the full plan and [AGENTS.md](AGENTS.md) for rules coding agents follow.

## Quickstart (Windows prototype)

Prerequisites: Docker Desktop, [Ollama](https://ollama.com), ffmpeg,
[uv](https://docs.astral.sh/uv/), an NVIDIA GPU (16 GB VRAM recommended).

```bash
uv sync                      # create .venv and install dependencies
cp .env.example .env         # adjust if needed
docker compose up -d         # Postgres + pgvector
uv run tkb db upgrade        # create the schema
ollama pull qwen3:14b        # answer/extraction model
ollama pull bge-m3           # embeddings
uv run tkb doctor            # check everything
```

Set `OLLAMA_MAX_LOADED_MODELS=1` and `OLLAMA_KEEP_ALIVE=5m` for Ollama so only one large
model sits in VRAM at a time.

## Development

```bash
uv run pytest                # CPU-only, models mocked; DB test runs if Postgres is up
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
  llm/             LiteLLM gateway: chat(), embed(), rerank()
  migrations/      Alembic migrations
  doctor.py        environment checks
  cli.py           `tkb` command
docs/adr/          architecture decision records
data/              raw inputs and processed outputs (git-ignored)
```

## License

MIT. Never commit private data, recordings or personal names — `data/` and `.env` are git-ignored.
