# ADR 0009 — Framework: plug-in registries, three model setups, local-only enforcement

Date: 2026-10-22 · Status: accepted

## Plug-in points (BUILD_GUIDE §7 Milestone 8)
- `ingest/parsers.py` — `Parser` per modality with its own file suffixes; the pipeline groups
  pending files by parser and calls `parse(ctx, batch)` once per parser (one GPU model family
  per batch), then `chunk(ctx, src, f, count)` per file. `pipeline.write_blocks` gives plug-ins
  the standard chunk stage.
- `llm/backends.py` — backends for roles LiteLLM doesn't cover (reranker, asr, diarization),
  selected by `backend:` in models.yaml. Added `diarization: none` (single speaker) for CPU
  setups and machines without the gated pyannote model.
- `ingest/extract.py` — `Extractor` per extra schema. Its field joins the same per-chunk LLM
  call (no extra calls), the response schema is built with `pydantic.create_model`, and items
  without a verbatim evidence quote are dropped like the built-ins. Built-in fields can't be
  overridden.
- Plug-ins are imported from `plugins:` in pipeline.yaml or the `thesiskb.plugins` entry-point
  group. `examples/plugins/` has a `.txt` notes parser and a "decisions" extractor, both tested.

The built-in extractors were not rewritten into the registry: they are tuned and tested, and
the registry only has to make *adding* schemas possible.

## Model setups
`TKB_MODELS_FILE` selects `models.yaml` (fully local), `examples/models.api.yaml` (cloud LLM for
answer/judge/router/extraction, everything else local) or `examples/models.cpu.yaml`. In API
mode vision stays local (rotation detection needs token logprobs) and embeddings stay local
(stored vectors; cloud embeddings would ship every chunk). `.env` is exported to the process
environment so LiteLLM finds provider keys; `tkb doctor` lists cloud roles and missing keys.

## local_only enforcement (BUILD_GUIDE §10)
`LLMGateway.role_for(role, local_only)` returns the role itself if it is local
(`RoleConfig.is_local`: Ollama, localhost endpoint or in-process backend), else `<role>_local`,
else raises `LocalOnlyError`. Every call that sees source content goes through it: answer,
support judge, literature summaries, extraction, vision (photos, keyframes) and embeddings.
Questions alone (router, query embedding) are not source content.

## CI integration test
`tests/integration` generates a 1-page PDF by hand (no PDF library) and ~10 s of speech
(espeak-ng; the Windows voice locally) and runs the real Docling and faster-whisper (tiny, CPU)
through the pipeline into the test DB, with embeddings faked and diarization `none`. The CI job
installs CPU-only PyTorch instead of the CUDA wheels in the lock file.
