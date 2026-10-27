"""`tkb report`: one zip with what is needed to debug a problem on another PC.

Contains: recent log lines, `tkb doctor` results, versions, and the model/pipeline config.
Secrets (DB password, tokens, API keys) are replaced by *** everywhere. Never included:
.env, people.yaml, raw files, chunk text. The last answer trace (question, answer and
passages) is added only with `--trace`.
"""

from __future__ import annotations

import json
import os
import platform
import re
import sys
import zipfile
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from thesiskb import __version__
from thesiskb.core.config import (
    REPO_ROOT,
    get_models_config,
    get_pipeline_config,
    get_settings,
)

PACKAGES = (
    "litellm", "docling", "faster-whisper", "pyannote.audio", "torch", "sentence-transformers",
    "sqlalchemy", "psycopg", "pgvector", "fastapi", "mcp",
)  # fmt: skip
SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)", re.I)
URL_PASSWORD = re.compile(r"(://[^:/@\s]+:)[^@\s]+@")


def secrets() -> list[str]:
    """Every secret value we know of: settings, .env, environment, models.yaml api keys."""
    s = get_settings()
    found = [s.hf_token, s.api_token]
    m = URL_PASSWORD.search(s.database_url)
    if m:
        found.append(s.database_url[m.end(1) : m.end() - 1])
    env_file = REPO_ROOT / ".env"
    pairs = list(os.environ.items())
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            k, sep, v = line.partition("=")
            if sep and not k.strip().startswith("#"):
                pairs.append((k.strip(), v.strip().strip("\"'")))
    found += [v for k, v in pairs if SECRET_ENV.search(k)]
    found += [r.api_key for r in get_models_config().roles.values()]
    return sorted({v for v in found if v and len(v) >= 4}, key=len, reverse=True)


def redact(text: str, known: list[str]) -> str:
    for v in known:
        text = text.replace(v, "***")
    return URL_PASSWORD.sub(r"\1***@", text)


def _log_tail(log_dir: Path, lines: int) -> str:
    files = sorted(log_dir.glob("thesiskb.log*"), key=lambda p: p.stat().st_mtime)
    out: list[str] = []
    for f in reversed(files):  # newest first, until we have enough lines
        out = f.read_text(encoding="utf-8", errors="replace").splitlines() + out
        if len(out) >= lines:
            break
    return "\n".join(out[-lines:])


def _environment() -> dict[str, Any]:
    from thesiskb.doctor import run_checks

    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    cfg = get_pipeline_config()
    ocr = (cfg.model_extra or {}).get("ocr", {}).get("languages", [])
    try:
        checks = [
            r.model_dump(mode="json") for r in run_checks(get_settings(), get_models_config(), ocr)
        ]
    except Exception as e:  # noqa: BLE001 - the report must still be written
        checks = [{"name": "doctor", "status": "error", "detail": f"{type(e).__name__}: {e}"}]
    return {
        "thesiskb": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "models_file": get_settings().models_file,
        "doctor": checks,
    }


def build_report(out: Path, log_lines: int = 1000, trace: dict[str, Any] | None = None) -> Path:
    from thesiskb.core.logs import log_dir

    known = secrets()
    settings = get_settings()
    configs = [settings.config_dir / settings.models_file, settings.config_dir / "pipeline.yaml"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        env = json.dumps(_environment(), indent=2, ensure_ascii=False, default=str)
        z.writestr("environment.json", redact(env, known))
        z.writestr("thesiskb.log", redact(_log_tail(log_dir(), log_lines), known))
        for c in configs:
            if c.exists():
                z.writestr(f"config/{c.name}", redact(c.read_text(encoding="utf-8"), known))
        if trace is not None:
            z.writestr("trace.json", redact(json.dumps(trace, indent=2, ensure_ascii=False), known))
    return out


def default_report_path() -> Path:
    from thesiskb.core.logs import log_dir

    return log_dir() / f"report-{datetime.now():%Y%m%d-%H%M%S}.zip"
