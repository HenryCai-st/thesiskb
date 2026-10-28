"""Model catalogue for the Models tab: what is offered, installed, chosen; download and switch.

config/model_catalog.yaml lists the options per component (answer LLM, vision, embeddings,
reranker, OCR, speech, diarization). "Use" runs as an item in the ingest queue (one at a
time, holding the GPU lock): download (Ollama /api/pull with bytes completed/total, or
huggingface_hub into the normal HF cache), then write the choice into the active models
file (or pipeline.yaml for OCR) with comments kept, then clear the cached configs so the
next ingest or question uses it. Embedding changes also re-embed every chunk.

API keys for online models go to .env and the process environment; they are never returned,
only "set" / "not set" (ADR 0017).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from thesiskb.core.config import (
    get_models_config,
    get_pipeline_config,
    get_settings,
)

log = logging.getLogger(__name__)

Progress = Callable[[int, int], None]  # bytes done, bytes total (0 = unknown)
KEY_NAME = re.compile(r"[A-Z][A-Z0-9_]{2,63}")
EXTRA_KEYS = ("TKB_HF_TOKEN",)  # gated Hugging Face models (pyannote)


class Option(BaseModel):
    model_config = {"extra": "forbid"}

    id: str
    label: str
    model: str | None = None
    value: Any = None  # for pipeline components (OCR languages)
    local: bool = True
    env_key: str | None = None
    download: dict[str, Any] = Field(default_factory=dict)
    size_gb: float | None = None
    vram_gb: float | None = None
    licence: str = ""
    note: str = ""
    set: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] | None = None


class Component(BaseModel):
    label: str
    help: str = ""
    roles: list[str] = Field(default_factory=list)
    pipeline: str | None = None  # dotted pipeline.yaml key, instead of roles
    reembed: bool = False
    options: list[Option]

    def option(self, option_id: str) -> Option:
        for o in self.options:
            if o.id == option_id:
                return o
        raise KeyError(option_id)


class Catalog(BaseModel):
    components: dict[str, Component]

    def env_keys(self) -> list[str]:
        keys = {o.env_key for c in self.components.values() for o in c.options if o.env_key}
        return sorted(keys) + list(EXTRA_KEYS)


def catalog_path() -> Path:
    return get_settings().config_dir / "model_catalog.yaml"


def load_catalog(path: Path | None = None) -> Catalog:
    import yaml

    return Catalog.model_validate(yaml.safe_load((path or catalog_path()).read_text("utf-8")))


def models_path() -> Path:
    s = get_settings()
    p = Path(s.models_file)
    return p if p.is_absolute() else s.config_dir / p


def reload_configs() -> None:
    """Forget cached configs and the gateway: the next call reads the files again."""
    from thesiskb.llm.gateway import get_gateway

    get_settings.cache_clear()
    get_models_config.cache_clear()
    get_pipeline_config.cache_clear()
    get_gateway.cache_clear()


# --- what is chosen / installed ------------------------------------------------------------


def current_option(comp: Component) -> str | None:
    if comp.pipeline:
        value = get_pipeline_config().model_dump()
        for part in comp.pipeline.split("."):
            value = (value or {}).get(part) if isinstance(value, dict) else None
        return next((o.id for o in comp.options if o.value == value), None)
    roles = get_models_config().roles
    role = roles.get(comp.roles[0])
    if role is None:
        return None
    for o in comp.options:
        if o.model == role.model and o.set.get("backend", role.backend) == role.backend:
            return o.id
    return None


def ollama_models() -> set[str] | None:
    """Names Ollama has, with and without ':latest'; None if Ollama is not reachable."""
    try:
        r = httpx.get(f"{get_settings().ollama_base_url}/api/tags", timeout=3)
        r.raise_for_status()
    except httpx.HTTPError:
        return None
    names = set()
    for m in r.json().get("models", []):
        names.add(m["name"])
        names.add(m["name"].removesuffix(":latest"))
    return names


def tesseract_langs() -> set[str] | None:
    s = get_settings()
    cmd = shutil.which(s.tesseract_cmd) or (
        s.tesseract_cmd if Path(s.tesseract_cmd).exists() else None
    )
    if not cmd:
        return None
    args = [cmd, "--list-langs"] + (["--tessdata-dir", s.tessdata_dir] if s.tessdata_dir else [])
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return {line.strip() for line in out.splitlines()[1:] if line.strip()}


def hf_cached(repo: str, check: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(repo, check), str)


def is_installed(o: Option, ollama: set[str] | None, langs: set[str] | None) -> bool | None:
    """True / False, or None when it cannot be checked (Ollama or Tesseract not found)."""
    if not o.local:
        return bool(o.env_key and os.environ.get(o.env_key))
    d = o.download
    if "ollama" in d:
        return None if ollama is None else d["ollama"] in ollama
    if "hf" in d:
        return hf_cached(d["hf"], d.get("check", "config.json"))
    if "tesseract" in d:
        return None if langs is None else set(d["tesseract"]) <= langs
    return True  # nothing to download


def status(catalog: Catalog) -> list[dict[str, Any]]:
    ollama, langs = ollama_models(), tesseract_langs()
    out = []
    for key, comp in catalog.components.items():
        out.append(
            {
                "key": key,
                "label": comp.label,
                "help": comp.help,
                "reembed": comp.reembed,
                "current": current_option(comp),
                "options": [
                    o.model_dump(include={"id", "label", "model", "local", "env_key", "size_gb",
                                          "vram_gb", "licence", "note"})
                    | {"installed": is_installed(o, ollama, langs),
                       "gated": bool(o.download.get("gated"))}
                    for o in comp.options
                ],
            }
        )  # fmt: skip
    return out


# --- download --------------------------------------------------------------------------------


def pull_ollama(name: str, progress: Progress) -> None:
    """Ollama streams one JSON line per update; sum bytes over the layers (digests)."""
    layers: dict[str, tuple[int, int]] = {}
    url = f"{get_settings().ollama_base_url}/api/pull"
    with httpx.stream("POST", url, json={"model": name, "stream": True}, timeout=None) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.strip():
                continue
            msg = json.loads(line)
            if msg.get("error"):
                raise RuntimeError(f"ollama pull {name}: {msg['error']}")
            if msg.get("digest") and msg.get("total"):
                layers[msg["digest"]] = (msg.get("completed", 0), msg["total"])
                progress(sum(c for c, _ in layers.values()), sum(t for _, t in layers.values()))


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def download_hf(repo: str, expected_gb: float | None, progress: Progress) -> None:
    """snapshot_download into the HF cache; progress = bytes in the repo's cache folder."""
    from huggingface_hub import constants, snapshot_download

    folder = Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}"
    total = int((expected_gb or 0) * 1e9)
    done = threading.Event()

    def watch() -> None:
        while not done.wait(1.0):
            progress(min(_dir_size(folder), total) if total else _dir_size(folder), total)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    try:
        snapshot_download(repo, token=get_settings().hf_token or None)
    finally:
        done.set()
        t.join()
    size = _dir_size(folder)
    progress(size, size)


def download(o: Option, progress: Progress) -> None:
    d = o.download
    if "ollama" in d:
        pull_ollama(d["ollama"], progress)
    elif "hf" in d:
        if d.get("gated") and not get_settings().hf_token:
            raise RuntimeError(
                f"{o.label} is gated on Hugging Face: set TKB_HF_TOKEN (Models tab, keys) and "
                f"accept the terms on huggingface.co/{d['hf']}"
            )
        download_hf(d["hf"], o.size_gb, progress)
    elif "tesseract" in d:
        langs = tesseract_langs()
        missing = sorted(set(d["tesseract"]) - (langs or set()))
        if missing:
            raise RuntimeError(
                f"Tesseract language pack(s) missing: {', '.join(missing)}. Install them "
                "(see docs/usage.md, Troubleshooting); they are not downloaded here."
            )


# --- write the choice ------------------------------------------------------------------------


def _yaml():
    from ruamel.yaml import YAML

    y = YAML()
    y.preserve_quotes = True
    y.width = 4096
    return y


def apply_choice(catalog: Catalog, component: str, option_id: str) -> None:
    """Write an option into models.yaml (roles) or pipeline.yaml (OCR), keeping comments."""
    comp = catalog.components[component]
    o = comp.option(option_id)
    y = _yaml()
    if comp.pipeline:
        path = get_settings().config_dir / "pipeline.yaml"
        data = y.load(path.read_text("utf-8"))
        node = data
        *parents, last = comp.pipeline.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[last] = list(o.value) if isinstance(o.value, list) else o.value
    else:
        path = models_path()
        data = y.load(path.read_text("utf-8"))
        roles = data["roles"]
        targets = comp.roles if o.local else comp.roles[:1]  # online: only `answer`
        first, local_alt = comp.roles[0], f"{comp.roles[0]}_local"
        if (
            not o.local
            and first in roles
            and local_alt not in roles
            and get_models_config().role(first).is_local
        ):  # local-only sources keep using the local model
            from copy import deepcopy

            roles[local_alt] = deepcopy(roles[first])
        for name in targets:
            role = roles.setdefault(name, {})
            role["model"] = o.model
            for k, v in o.set.items():
                role[k] = v
            if o.vram_gb is not None and o.local:
                role["vram_gb"] = int(o.vram_gb) if float(o.vram_gb).is_integer() else o.vram_gb
            elif not o.local:
                role.pop("vram_gb", None)
            if o.params is not None and name == comp.roles[0]:
                params = role.get("params")
                if params is None:
                    role["params"] = dict(o.params)
                else:  # in place: comments on the kept keys stay
                    for k in [k for k in params if k not in o.params]:
                        del params[k]
                    for k, v in o.params.items():
                        params[k] = v
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        y.dump(data, fh)
    reload_configs()
    log.info("models: %s set to %s", component, o.label)


# --- API keys ------------------------------------------------------------------------------


def env_path() -> Path:
    return Path(".env")  # the file Settings and load_dotenv read (current directory)


def key_status(catalog: Catalog) -> dict[str, bool]:
    return {k: bool(os.environ.get(k)) for k in catalog.env_keys()}


def set_key(catalog: Catalog, name: str, value: str) -> None:
    """Write NAME=value to .env (replacing an earlier line) and to this process's env."""
    if name not in catalog.env_keys() or not KEY_NAME.fullmatch(name):
        raise ValueError(f"unknown key name {name!r}")
    value = value.strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError("the key must be one non-empty line")
    path = env_path()
    lines = path.read_text("utf-8").splitlines() if path.exists() else []
    pattern = re.compile(rf"^\s*#?\s*{re.escape(name)}\s*=")
    lines = [ln for ln in lines if not pattern.match(ln)] + [f"{name}={value}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ[name] = value
    reload_configs()
    log.info("API key %s updated in .env", name)  # the name only, never the value
