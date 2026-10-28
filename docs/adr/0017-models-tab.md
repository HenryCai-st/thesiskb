# ADR 0017 — Models tab: curated catalogue, downloads in the queue, API keys in .env

Date: 2026-10-28 · Status: accepted

Context: changing a model meant editing YAML without knowing its size or VRAM cost first
(feedback point 3, BUILD_GUIDE_v2 Milestone 16).

## Decisions
- **Catalogue** `config/model_catalog.yaml`, curated by hand: per component the options the
  code supports, with download size, approximate VRAM, licence, local/online and what it
  sets (roles in the active models file, or `ocr.languages` in pipeline.yaml). No discovery.
- **Answer LLM sets four roles** (answer, judge, router, extraction): they share weights, and
  a second local LLM would not fit next to the first in 16 GB. Each role keeps its own params;
  an option's `params` replace only `answer`'s (Ollama's `num_ctx` must not go to a cloud API).
- **Online only for the answer LLM**: it is the only role with a local fallback for
  local-only sources (`answer_local`, created from the current local model on the first
  switch to an online one). Reranking, embeddings and vision have no such fallback in the
  code (the reranker sends passages without checking `local_only`), so they stay local.
  Embedding options must have 1024 dimensions (database column).
- **"Use" is a queue item** (`kind = model`, M13): one at a time, holding the GPU lock.
  Ollama models via `/api/pull` (bytes completed/total summed over layers); Hugging Face
  models via `snapshot_download` into the normal HF cache (where faster-whisper, pyannote and
  sentence-transformers look), progress = bytes in the repo's cache folder vs the catalogue
  size. Tesseract language packs are only checked, not downloaded. Then the choice is written
  with ruamel.yaml (new dependency: keeps the files' comments) and cached configs and the
  gateway are dropped, so the next question or ingest uses it. Transcripts record the
  speech model (`asr.json` → `model`).
- **Embedding change** returns 409 until the user confirms; the item then re-embeds every
  chunk and clears entity-name vectors (they are recomputed, with the usual snapshot before
  any merge).
- **API keys**: `POST /api/keys` writes `NAME=value` to `.env` and the process environment,
  only for names the catalogue uses (plus `TKB_HF_TOKEN`), one line, and only when the
  request comes from this machine and `tkb serve` listens on localhost. The body is parsed
  by hand (a FastAPI validation error would echo it); responses and logs carry only the
  key's name and "set" / "not set".
- The choice is written to the committed `config/models.yaml`; it shows as a local change
  in git, which is intended (the file is the user's setup).
