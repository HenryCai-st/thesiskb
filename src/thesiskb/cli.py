"""`tkb` command line."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from thesiskb.core.config import (
    REPO_ROOT,
    get_models_config,
    get_pipeline_config,
    get_settings,
)

app = typer.Typer(help="ThesisKB — private, multimodal knowledge base.", no_args_is_help=True)
db_app = typer.Typer(help="Database schema management.", no_args_is_help=True)
app.add_typer(db_app, name="db")
console = Console()

_STATUS_STYLE = {"ok": "[green]OK[/]", "warn": "[yellow]WARN[/]", "fail": "[red]FAIL[/]"}


@app.command()
def doctor() -> None:
    """Check DB, GPU, Ollama, ffmpeg, tesseract and print the model/VRAM plan."""
    from thesiskb.doctor import Status, query_gpus, run_checks

    settings = get_settings()
    models = get_models_config()
    ocr_langs = (get_pipeline_config().model_extra or {}).get("ocr", {}).get("languages", [])

    checks = Table(title="Environment")
    checks.add_column("Check")
    checks.add_column("Status")
    checks.add_column("Detail", overflow="fold")
    results = run_checks(settings, models, ocr_langs)
    for r in results:
        checks.add_row(r.name, _STATUS_STYLE[r.status], r.detail)
    console.print(checks)

    try:
        total_gb = sum(g.total_mb for g in query_gpus()) / 1024
    except Exception:  # noqa: BLE001 — no GPU just means no budget column
        total_gb = None

    plan = Table(title="Models per role (loaded one heavy model at a time)")
    plan.add_column("Role")
    plan.add_column("Model")
    plan.add_column("Backend")
    plan.add_column("≈VRAM", justify="right")
    plan.add_column("Fits", justify="center")
    for name, role in models.roles.items():
        fits = "-" if total_gb is None else ("yes" if role.vram_gb <= total_gb else "[red]no[/]")
        plan.add_row(name, role.model, role.backend, f"{role.vram_gb:g} GB", fits)
    console.print(plan)

    if total_gb is not None:
        answer = models.roles.get("answer")
        resident = sum(
            models.roles[r].vram_gb for r in ("embedding", "reranker") if r in models.roles
        )
        if answer:
            combo = answer.vram_gb + resident
            ok = combo <= total_gb
            console.print(
                f"Answer LLM + embeddings + reranker ≈ {combo:g} GB of {total_gb:.1f} GB "
                f"→ {'can stay loaded together' if ok else '[yellow]must swap[/]'}"
            )

    if any(r.status == Status.FAIL for r in results):
        raise typer.Exit(code=1)


def _alembic_config():
    from alembic.config import Config

    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    cfg.set_main_option("sqlalchemy.url", get_settings().database_url)
    return cfg


@db_app.command("upgrade")
def db_upgrade(revision: str = "head") -> None:
    """Apply Alembic migrations."""
    from alembic import command

    command.upgrade(_alembic_config(), revision)
    console.print(f"[green]Database upgraded to {revision}[/]")


@db_app.command("current")
def db_current() -> None:
    """Show the current schema revision."""
    from alembic import command

    command.current(_alembic_config(), verbose=False)


if __name__ == "__main__":
    app()
