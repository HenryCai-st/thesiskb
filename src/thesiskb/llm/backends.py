"""Model-backend plug-in point for roles that do not go through LiteLLM (BUILD_GUIDE §8
"ModelProvider per role"): reranker, asr (speech-to-text), diarization.

Chat, vision, embedding and LiteLLM rerank roles are swapped by changing `model:` in
config/models.yaml (any LiteLLM provider). Everything else is selected by the role's `backend:`
and created through this registry:

    from thesiskb.llm.backends import register_backend

    @register_backend("asr", "my_asr")
    def make(role_cfg, settings):
        return MyTranscriber(role_cfg.model)   # needs .transcribe(wav, out_json) and .close()

Interfaces (duck-typed):
  reranker     .predict(list[(query, doc)], batch_size) -> list[float]
  asr          .transcribe(wav: Path, out_json: Path) -> dict; .close()
  diarization  .diarize(wav: Path, out_json: Path) -> list[{start, end, speaker}]; .close()
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

Factory = Callable[[Any, Any], Any]  # (RoleConfig, Settings) -> backend instance
_BACKENDS: dict[tuple[str, str], Factory] = {}


def register_backend(kind: str, name: str) -> Callable[[Factory], Factory]:
    def deco(factory: Factory) -> Factory:
        _BACKENDS[(kind, name)] = factory
        return factory

    return deco


def create_backend(kind: str, role_cfg, settings) -> Any:
    try:
        factory = _BACKENDS[(kind, role_cfg.backend)]
    except KeyError as e:
        known = sorted(n for k, n in _BACKENDS if k == kind)
        raise ValueError(f"unknown {kind} backend '{role_cfg.backend}' (known: {known})") from e
    return factory(role_cfg, settings)


def backends(kind: str) -> list[str]:
    return sorted(n for k, n in _BACKENDS if k == kind)


# --- built-ins -----------------------------------------------------------------------------


@register_backend("asr", "faster_whisper")
def _faster_whisper(role_cfg, settings):
    from thesiskb.ingest.audio import Transcriber

    return Transcriber(role_cfg.model, (role_cfg.model_extra or {}).get("compute_type", "float16"))


@register_backend("diarization", "pyannote")
def _pyannote(role_cfg, settings):
    from thesiskb.ingest.audio import Diarizer

    return Diarizer(role_cfg.model, settings.hf_token)


class SingleSpeaker:
    """Diarization stand-in: one speaker for the whole recording. For solo recordings, CPU-only
    machines, or when the gated pyannote model is not available (no Hugging Face token)."""

    def diarize(self, wav: Path, out_json: Path) -> list[dict[str, Any]]:
        import wave

        with wave.open(str(wav)) as w:  # header only: no numpy needed on CPU-only installs
            duration = w.getnframes() / w.getframerate()
        turns = [{"start": 0.0, "end": round(duration, 2), "speaker": "SPEAKER_00"}]
        out_json.write_text(json.dumps(turns), encoding="utf-8")
        return turns

    def close(self) -> None:
        pass


@register_backend("diarization", "none")
def _single_speaker(role_cfg, settings):
    return SingleSpeaker()


@register_backend("reranker", "cross_encoder")
def _cross_encoder(role_cfg, settings):
    try:
        import torch
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise RuntimeError("Reranker backend 'cross_encoder' needs: uv sync --extra local") from e
    extra = role_cfg.model_extra or {}
    kwargs: dict[str, Any] = {"max_length": extra.get("max_length", 1024)}
    if extra.get("fp16", True) and torch.cuda.is_available():
        # fp16 halves VRAM (~1.1 GB) so it fits next to the answer LLM without spilling into
        # shared system memory, which is ~20x slower on Windows
        kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    return CrossEncoder(role_cfg.model, **kwargs)
