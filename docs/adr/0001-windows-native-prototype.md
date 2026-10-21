# ADR 0001 — Start Milestone 1 on native Windows, move heavy ML to WSL2 later

Date: 2026-10-21 · Status: accepted

## Context
BUILD_GUIDE §4.1 recommends developing inside WSL2 (Ubuntu). The prototype PC already has
Docker Desktop, Ollama (Windows build), ffmpeg and Python on Windows, but no Ubuntu WSL distro.

## Decision
Milestone 1 (skeleton, DB, config, LLM gateway, `tkb doctor`) runs natively on Windows:
Postgres in Docker Desktop, Ollama via its Windows service on `localhost:11434`. All code is
cross-platform (no Windows-only paths or APIs), and every endpoint comes from `TKB_*` env vars.

Before Milestone 4 (pyannote/WhisperX, which are painful on Windows), clone the repo into an
Ubuntu WSL2 distro and run the pipeline there; Ollama and Postgres stay reachable on localhost.

## Consequences
- Fast start with what is installed; no WSL migration needed for Milestones 1–3.
- Ollama's `OLLAMA_MAX_LOADED_MODELS=1` must be set as a Windows user env var (not in WSL).
- CI runs on Linux, which keeps us honest about portability.
