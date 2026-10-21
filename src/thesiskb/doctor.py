"""Environment checks for `tkb doctor`: DB, GPU, Ollama, ffmpeg, tesseract, model/VRAM summary."""

from __future__ import annotations

import shutil
import subprocess
from enum import StrEnum

import httpx
from pydantic import BaseModel

from thesiskb.core.config import ModelsConfig, Settings


class Status(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


class CheckResult(BaseModel):
    name: str
    status: Status
    detail: str


class GpuInfo(BaseModel):
    name: str
    total_mb: int
    used_mb: int


def _run(cmd: list[str], timeout: float = 10) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True).stdout


def query_gpus() -> list[GpuInfo]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    for line in out.strip().splitlines():
        name, total, used = (p.strip() for p in line.split(","))
        gpus.append(GpuInfo(name=name, total_mb=int(total), used_mb=int(used)))
    return gpus


def check_gpu() -> CheckResult:
    if not shutil.which("nvidia-smi"):
        return CheckResult(name="GPU", status=Status.FAIL, detail="nvidia-smi not found")
    try:
        gpus = query_gpus()
    except (subprocess.SubprocessError, ValueError) as e:
        return CheckResult(name="GPU", status=Status.FAIL, detail=f"nvidia-smi failed: {e}")
    if not gpus:
        return CheckResult(name="GPU", status=Status.FAIL, detail="no GPU reported")
    detail = "; ".join(
        f"{g.name}: {g.used_mb / 1024:.1f}/{g.total_mb / 1024:.1f} GB used" for g in gpus
    )
    return CheckResult(name="GPU", status=Status.OK, detail=detail)


def check_db(database_url: str) -> CheckResult:
    from sqlalchemy import create_engine, text

    try:
        engine = create_engine(database_url, connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            version = conn.execute(text("SHOW server_version")).scalar()
            vector = conn.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ).scalar()
            has_alembic = conn.execute(
                text("SELECT to_regclass('public.alembic_version') IS NOT NULL")
            ).scalar()
            revision = (
                conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
                if has_alembic
                else None
            )
        engine.dispose()
    except Exception as e:  # noqa: BLE001 — any connection error is a failed check
        msg = str(e).splitlines()[0]
        return CheckResult(
            name="Database",
            status=Status.FAIL,
            detail=f"cannot connect ({msg}). Try: docker compose up -d",
        )
    if not vector or not revision:
        return CheckResult(
            name="Database",
            status=Status.WARN,
            detail=f"Postgres {version} reachable but schema missing. Run: tkb db upgrade",
        )
    return CheckResult(
        name="Database",
        status=Status.OK,
        detail=f"Postgres {version}, pgvector {vector}, schema rev {revision}",
    )


def check_ollama(base_url: str, models: ModelsConfig) -> CheckResult:
    try:
        tags = httpx.get(f"{base_url}/api/tags", timeout=3).json()
    except (httpx.HTTPError, ValueError) as e:
        return CheckResult(
            name="Ollama", status=Status.FAIL, detail=f"not reachable at {base_url} ({e})"
        )
    pulled = {m["name"] for m in tags.get("models", [])}
    pulled |= {n.removesuffix(":latest") for n in pulled}
    wanted = {
        r.model.split("/", 1)[1] for r in models.roles.values() if r.provider.startswith("ollama")
    }
    missing = sorted(w for w in wanted if w not in pulled)
    if missing:
        return CheckResult(
            name="Ollama",
            status=Status.WARN,
            detail=f"reachable; missing models: {', '.join(missing)} (ollama pull <name>)",
        )
    return CheckResult(
        name="Ollama", status=Status.OK, detail=f"reachable; {len(wanted)} configured models pulled"
    )


def check_binary(name: str, version_args: list[str], required: bool = True) -> CheckResult:
    path = shutil.which(name)
    if not path:
        return CheckResult(
            name=name, status=Status.FAIL if required else Status.WARN, detail="not found on PATH"
        )
    try:
        first = _run([path, *version_args]).strip().splitlines()[0]
    except (subprocess.SubprocessError, IndexError) as e:
        return CheckResult(name=name, status=Status.WARN, detail=f"found but failed: {e}")
    return CheckResult(name=name, status=Status.OK, detail=first)


def check_tesseract(languages: list[str]) -> CheckResult:
    """Tesseract is first needed in Milestone 3, so missing is a warning, not a failure."""
    path = shutil.which("tesseract")
    if not path:
        return CheckResult(
            name="tesseract", status=Status.WARN, detail="not found (needed from Milestone 3)"
        )
    try:
        out = _run([path, "--list-langs"])
    except subprocess.SubprocessError as e:
        return CheckResult(name="tesseract", status=Status.WARN, detail=f"failed: {e}")
    installed = {line.strip() for line in out.splitlines()[1:]}  # line 0 is a header
    missing = [lang for lang in languages if lang not in installed]
    if missing:
        return CheckResult(
            name="tesseract",
            status=Status.WARN,
            detail=f"installed; missing language packs: {', '.join(missing)}",
        )
    return CheckResult(
        name="tesseract", status=Status.OK, detail=f"languages: {', '.join(languages)}"
    )


def run_checks(
    settings: Settings, models: ModelsConfig, ocr_languages: list[str]
) -> list[CheckResult]:
    return [
        check_db(settings.database_url),
        check_gpu(),
        check_ollama(settings.ollama_base_url, models),
        check_binary("ffmpeg", ["-version"]),
        check_tesseract(ocr_languages),
    ]
