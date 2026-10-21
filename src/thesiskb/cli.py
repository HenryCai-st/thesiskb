"""`tkb` command line."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
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


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(..., help="Files or folders, e.g. data/raw"),
    source_type: str | None = typer.Option(
        None, "--type", help="paper|book|note|meeting|interview|demo (default: by file type)"
    ),
    force: bool = typer.Option(False, help="Redo all stages even if already done"),
) -> None:
    """Ingest documents (Milestone 2: PDF/Office). Other modalities are registered only."""
    import logging

    from thesiskb.core.db import session_scope
    from thesiskb.ingest.detect import DOCUMENT_MODALITIES, detect
    from thesiskb.ingest.pipeline import ingest_documents
    from thesiskb.llm import get_gateway

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("LiteLLM", "httpx", "docling", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = get_pipeline_config()
    files = detect(paths)
    docs = [f for f in files if f.modality in DOCUMENT_MODALITIES]
    others = [f for f in files if f.modality not in DOCUMENT_MODALITIES]
    console.print(f"Found {len(files)} files: {len(docs)} documents, {len(others)} other")
    for f in others:
        console.print(f"  [dim]skip (needs a later milestone): {f.path.name} ({f.modality})[/]")

    with session_scope() as session:
        report = ingest_documents(
            session,
            docs,
            cfg,
            get_gateway(),
            source_type=source_type,
            force=force,
            metadata_lookup=bool((cfg.model_extra or {}).get("metadata_lookup", True)),
        )
    failed = report.pop("failed")
    work = ", ".join(f"{stage}: {n}" for stage, n in report.items())
    console.print(f"Work done — {work}" + ("" if any(report.values()) else " (nothing to do)"))
    for name, err in failed:
        console.print(f"[red]FAILED[/] {name}: {err}")
    if failed:
        raise typer.Exit(code=1)


@app.command()
def ask(
    question: str,
    source_type: list[str] | None = typer.Option(
        None, "--type", help="Restrict to source types (repeatable), e.g. --type paper"
    ),
    verified_only: bool = typer.Option(False, "--verified-only"),
    no_verify: bool = typer.Option(False, "--no-verify", help="Skip the LLM support check"),
    show_context: bool = typer.Option(False, help="Also print every retrieved passage"),
    as_json: bool = typer.Option(False, "--json", help="Print the full Answer as JSON"),
) -> None:
    """Answer a question from the knowledge base, with citations."""
    from thesiskb.answer.generate import answer_question, locator_label
    from thesiskb.core.db import session_scope
    from thesiskb.llm import get_gateway
    from thesiskb.retrieval.hybrid import SearchFilters

    filters = SearchFilters(source_types=source_type or None, verified_only=verified_only)
    with session_scope() as session:
        ans = answer_question(
            session,
            get_gateway(),
            question,
            get_pipeline_config().retrieval,
            filters,
            verify_support=not no_verify,
        )
    if as_json:
        import sys

        sys.stdout.write(ans.model_dump_json(indent=2) + "\n")
        return

    marks = {
        "unsupported": "[red]⚠ unsupported[/]",
        "uncited": "[yellow]⚠ uncited[/]",
        "invalid_citation": "[red]⚠ invalid citation[/]",
    }
    console.print()
    for s in ans.sentences:
        flag = marks.get(s.status)
        note = f" [dim]({escape(s.note)})[/]" if flag and s.note else ""
        console.print(escape(s.text) + (f"  {flag}{note}" if flag else ""))
    if not ans.sentences:
        console.print(escape(ans.text))

    shown = ans.context if show_context else ans.cited
    if shown:
        console.print("\n[bold]Sources[/]")
        for c in shown:
            status = "" if c.verification_status == "unverified" else f" ({c.verification_status})"
            console.print(
                escape(f"  [S{c.n}] {c.filename}, {locator_label(c.locator)}{status}")
                + f"\n      [dim]{escape(c.context_header or '')}  chunk:{c.chunk_id}[/]"
            )
    t = ans.timings
    console.print(f"\n[dim]{' · '.join(f'{k} {v:.1f}s' for k, v in t.items())}[/]")


@app.command()
def sources() -> None:
    """List ingested sources: original filename ↔ sha256, title and status."""
    from sqlalchemy import func, select

    from thesiskb.core.db import session_scope
    from thesiskb.core.models import Chunk, Job, Source

    table = Table(title="Sources")
    for col in ("sha256", "File", "Type", "Title", "Chunks", "Stages", "Status"):
        table.add_column(col, overflow="fold")
    with session_scope() as session:
        n_chunks = dict(
            session.execute(select(Chunk.source_id, func.count()).group_by(Chunk.source_id)).all()
        )
        for s in session.scalars(select(Source).order_by(Source.path)):
            jobs = session.scalars(select(Job).where(Job.source_id == s.id)).all()
            stages = " ".join(f"{j.stage}:{j.status}" for j in sorted(jobs, key=lambda j: j.id))
            table.add_row(
                s.sha256[:12],
                s.path.rsplit("/", 1)[-1],
                s.source_type or "",
                s.title or "",
                str(n_chunks.get(s.id, 0)),
                stages,
                s.verification_status,
            )
    console.print(table)


@app.command("eval")
def eval_cmd(
    gold: Path = typer.Option(REPO_ROOT / "eval" / "questions.yaml", help="Gold questions file"),
    question_type: list[str] | None = typer.Option(None, "--type", help="Only these types"),
) -> None:
    """Run the gold questions and report retrieval/citation metrics (stored in eval_runs)."""
    from thesiskb.core.db import session_scope
    from thesiskb.evaluation import previous_summary, run_eval
    from thesiskb.llm import get_gateway

    with session_scope() as session:
        results, summary = run_eval(
            session,
            get_gateway(),
            get_pipeline_config().retrieval,
            gold,
            set(question_type) if question_type else None,
        )
        prev = previous_summary(session)

    table = Table(title="Eval")
    for col in ("id", "type", "status", "retrieved", "cited", "refused", "supported", "flags", "s"):
        table.add_column(col)
    yn = {True: "[green]yes[/]", False: "[red]no[/]", None: "-"}
    for r in results:
        flags = []
        if r.invalid_citations:
            flags.append(f"{r.invalid_citations} invalid")
        if r.uncited_sentences:
            flags.append(f"{r.uncited_sentences} uncited")
        table.add_row(
            r.id,
            r.type,
            r.status if r.status == "ran" else f"[dim]{r.reason}[/]",
            yn[r.retrieval_hit],
            yn[r.citation_hit],
            yn[r.refused],
            f"{r.supported}/{r.checked}" if r.status == "ran" else "-",
            ", ".join(flags),
            f"{r.latency_s}" if r.latency_s is not None else "-",
        )
    console.print(table)
    for key, value in summary.items():
        before = f"  [dim](prev {prev[key]})[/]" if prev and key in prev else ""
        console.print(f"  {key}: {value}{before}")


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
