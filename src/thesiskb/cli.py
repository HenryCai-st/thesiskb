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
    """Ingest documents (PDF/Office), images, audio and video."""
    import logging
    from collections import Counter

    from thesiskb.core.db import session_scope
    from thesiskb.ingest.detect import detect
    from thesiskb.ingest.pipeline import SUPPORTED_MODALITIES
    from thesiskb.ingest.pipeline import ingest as run_ingest
    from thesiskb.llm import get_gateway

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("LiteLLM", "httpx", "docling", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = get_pipeline_config()
    files = detect(paths)
    supported = [f for f in files if f.modality in SUPPORTED_MODALITIES]
    by_kind = Counter(f.modality for f in supported)
    console.print(
        f"Found {len(files)} files: " + ", ".join(f"{n} {m}" for m, n in sorted(by_kind.items()))
    )
    for f in files:
        if f.modality not in SUPPORTED_MODALITIES:
            console.print(f"  [dim]skip (needs a later milestone): {f.path.name} ({f.modality})[/]")

    with session_scope() as session:
        report = run_ingest(
            session,
            supported,
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
    from thesiskb.ingest.audio import ts

    for s in ans.sentences:
        flag = marks.get(s.status)
        note = f" [dim]({escape(s.note)})[/]" if flag and s.note else ""
        console.print(escape(s.text) + (f"  {flag}{note}" if flag else ""))
        for p in s.pinpoints:
            who = f" {p.speaker}" if p.speaker else " (on screen)"
            frame = f" · keyframe {ts(p.frame_t)} {p.frame}" if p.frame else ""
            console.print(f"    [dim]↳ S{p.n} at {ts(p.t)}{escape(who)}{escape(frame)}[/]")
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
        console.print("[dim]Open a source: tkb open <chunk-id>[/]")
    t = ans.timings
    console.print(f"\n[dim]{' · '.join(f'{k} {v:.1f}s' for k, v in t.items())}[/]")


speakers_app = typer.Typer(
    help="Map diarization labels (SPEAKER_00) to people.", no_args_is_help=True
)
app.add_typer(speakers_app, name="speakers")


def _resolve_source(session, ref: str):
    """A source by sha256 prefix or (part of) its filename."""
    from sqlalchemy import select

    from thesiskb.core.models import Source

    rows = session.scalars(select(Source)).all()
    hits = [s for s in rows if s.sha256.startswith(ref.lower())] or [
        s for s in rows if ref.lower() in s.path.rsplit("/", 1)[-1].lower()
    ]
    if len(hits) != 1:
        names = ", ".join(s.path.rsplit("/", 1)[-1] for s in hits[:5]) or "none"
        console.print(f"[red]'{escape(ref)}' matches {len(hits)} sources ({escape(names)})[/]")
        raise typer.Exit(code=1)
    return hits[0]


@speakers_app.command("list")
def speakers_list(source: str = typer.Argument(..., help="sha256 prefix or part of filename")):
    """Show speakers of a recording with talking time, a sample quote and name suggestions."""
    from thesiskb.core.db import session_scope
    from thesiskb.ingest.audio import Transcript, suggest_names, ts
    from thesiskb.ingest.pipeline import processed_dir, speaker_names

    with session_scope() as session:
        src = _resolve_source(session, source)
        path = processed_dir(get_pipeline_config(), src.sha256) / "transcript.json"
        if not path.exists():
            console.print("[red]No transcript yet — run tkb ingest first[/]")
            raise typer.Exit(code=1)
        transcript = Transcript.model_validate_json(path.read_text("utf-8"))
        names = speaker_names(session, src)
        filename = src.path.rsplit("/", 1)[-1]
    suggestions = suggest_names(transcript)

    table = Table(title=filename)
    for col in ("Label", "Person", "Suggested", "Talk", "Turns", "First words"):
        table.add_column(col, overflow="fold")
    for label in transcript.speakers:
        turns = [t for t in transcript.turns if t.speaker == label]
        talk = sum(t.end - t.start for t in turns)
        first = turns[0]
        table.add_row(
            label,
            names.get(label, "[dim]—[/]"),
            suggestions.get(label, ""),
            f"{talk:.0f}s",
            str(len(turns)),
            escape(f"[{ts(first.start)}] {first.text[:90]}"),
        )
    console.print(table)
    console.print('[dim]Map one: tkb speakers label <source> <LABEL> "<name>" [--alias ...][/]')


@speakers_app.command("label")
def speakers_label(
    source: str = typer.Argument(..., help="sha256 prefix or part of filename"),
    label: str = typer.Argument(..., help="e.g. SPEAKER_03"),
    name: str = typer.Argument(..., help='Person\'s name, e.g. "Will.i.am"'),
    alias: list[str] | None = typer.Option(None, "--alias", help="Other spellings (repeatable)"),
    role: str | None = typer.Option(None, help="e.g. supervisor, interviewee"),
) -> None:
    """Map a speaker label to a person, then re-chunk and re-embed the recording."""
    from sqlalchemy import func, select

    from thesiskb.core.db import session_scope
    from thesiskb.core.models import Person, Speaker
    from thesiskb.ingest.pipeline import rechunk
    from thesiskb.llm import get_gateway

    with session_scope() as session:
        src = _resolve_source(session, source)
        spk = session.scalar(
            select(Speaker).where(Speaker.source_id == src.id, Speaker.diarization_label == label)
        )
        if spk is None:
            console.print(f"[red]{escape(label)} not found; see tkb speakers list {source}[/]")
            raise typer.Exit(code=1)
        wanted = name.lower()
        person = session.scalar(
            select(Person).where((func.lower(Person.name) == wanted) | Person.aliases.any(name))
        )
        if person is None:
            person = Person(name=name, aliases=[], role=role)
            session.add(person)
        person.aliases = sorted(set(person.aliases or []) | set(alias or []))
        if role:
            person.role = role
        session.flush()
        spk.person_id = person.id
        session.commit()
        console.print(f"{label} → {escape(person.name)}; re-indexing {escape(src.path)} …")
        rechunk(session, src, get_pipeline_config(), get_gateway())
    console.print("[green]done[/]")


@app.command()
def reindex(source: str = typer.Argument(..., help="sha256 prefix or part of filename")) -> None:
    """Redo chunk + embed for one source from its cached stage outputs (no GPU models)."""
    from thesiskb.core.db import session_scope
    from thesiskb.ingest.pipeline import rechunk
    from thesiskb.llm import get_gateway

    with session_scope() as session:
        src = _resolve_source(session, source)
        name = src.path.rsplit("/", 1)[-1]
        rechunk(session, src, get_pipeline_config(), get_gateway())
    console.print(f"[green]re-indexed[/] {escape(name)}")


@app.command("open")
def open_chunk(chunk_id: str) -> None:
    """Open the file a citation points to (images open upright)."""
    import os
    import uuid

    from thesiskb.answer.generate import locator_label
    from thesiskb.core.db import session_scope
    from thesiskb.core.models import Chunk, Source
    from thesiskb.ingest.pipeline import processed_dir

    with session_scope() as session:
        chunk = session.get(Chunk, uuid.UUID(chunk_id))
        if chunk is None:
            console.print(f"[red]No chunk {chunk_id}[/]")
            raise typer.Exit(code=1)
        src = session.get(Source, chunk.source_id)
        target = Path(src.path)
        upright = processed_dir(get_pipeline_config(), src.sha256) / "upright.jpg"
        if src.modality == "image" and upright.exists():
            target = upright
        console.print(f"{escape(src.path)} — {locator_label(chunk.locator)}")
        for fr in chunk.locator.get("frames", []):
            console.print(f"  keyframe {fr['t']:.1f}s: {escape(fr['frame'])}")
        if chunk.locator.get("t_start") is not None:
            console.print(
                f"[dim]jump to {locator_label(chunk.locator).split('–')[0]} in your player[/]"
            )
        console.print(f"[dim]opening {escape(str(target))}[/]")
    if hasattr(os, "startfile"):
        os.startfile(target)  # Windows
    else:
        import subprocess

        subprocess.run(["xdg-open", str(target)], check=False)


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
    cols = (
        "id",
        "type",
        "status",
        "retrieved",
        "cited",
        "refused",
        "supported",
        "quote",
        "flags",
        "s",
    )
    for col in cols:
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
            yn[r.quote_hit] + ("" if r.keyframe_hit is None else f" kf:{yn[r.keyframe_hit]}"),
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
