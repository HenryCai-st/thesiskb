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
    extract: bool = typer.Option(
        True, help="Run LLM extraction (tasks, statements, entities); slow on long papers"
    ),
    local_only: bool = typer.Option(
        False, "--local-only", help="Never send these files' content to a cloud model"
    ),
) -> None:
    """Ingest documents (PDF/Office), images, audio and video."""
    from collections import Counter

    from rich.logging import RichHandler
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    from thesiskb.core.db import get_engine, session_scope

    # logs go through the same console as the progress bars, so they print above them
    from thesiskb.core.logs import set_console_handler, setup_logging
    from thesiskb.ingest.detect import detect
    from thesiskb.ingest.pipeline import ingest as run_ingest
    from thesiskb.ingest.pipeline import supported_modalities
    from thesiskb.ingest.queue import gpu_lock
    from thesiskb.llm import get_gateway

    setup_logging()
    set_console_handler(RichHandler(console=console, show_path=False, show_time=False))
    cfg = get_pipeline_config()
    modalities = supported_modalities(cfg)  # loads plug-ins (new file types) before detect
    files = detect(paths)
    supported = [f for f in files if f.modality in modalities]
    by_kind = Counter(f.modality for f in supported)
    console.print(
        f"Found {len(files)} files: " + ", ".join(f"{n} {m}" for m, n in sorted(by_kind.items()))
    )
    for f in files:
        if f.modality not in modalities:
            console.print(f"  [dim]skip (no parser for {f.modality}): {f.path.name}[/]")

    # one step = one file through one stage; already-done steps complete instantly
    # bar and counts first: on a narrow terminal only the file name at the end gets cut
    columns = (
        BarColumn(bar_width=24),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        TextColumn("{task.description}"),
    )
    with session_scope() as session, Progress(*columns, console=console) as bar:
        total = bar.add_task("steps (file x stage)", total=len(supported) * (5 if extract else 4))
        chunks = bar.add_task("", total=1, visible=False)

        def fit(text: str) -> str:  # the bar and counts take ~60 columns; cut only the label
            room = max(12, console.width - 62)
            return text if len(text) <= room else text[: room - 1] + "…"

        def progress(stage, name, done=None, of=None):
            if done is None:  # a file finished a stage
                bar.update(total, advance=1, description=fit(f"{stage} ✓ {name}"))
                bar.update(chunks, visible=False)
            else:  # summarize part by part, extract chunk by chunk
                label = fit(f"{stage} {name}")
                bar.update(chunks, completed=done, total=of, visible=True, description=label)

        with gpu_lock(get_engine(), wait=False) as free:
            if not free:  # the web UI's queue (or another tkb ingest) is using the GPU
                console.print("[yellow]Another ingest is running; waiting for it to finish…[/]")
        with gpu_lock(get_engine()):
            report = run_ingest(
                session,
                supported,
                cfg,
                get_gateway(),
                source_type=source_type,
                force=force,
                metadata_lookup=bool((cfg.model_extra or {}).get("metadata_lookup", True)),
                extract=extract,
                local_only=local_only,
                progress=progress,
            )
    console.print()  # end the progress-bar line
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
    path: str = typer.Option(
        "auto", help="auto (router) | general | literature | person | task | entity | summary"
    ),
) -> None:
    """Answer a question from the knowledge base, with citations."""
    from thesiskb.answer.generate import locator_label
    from thesiskb.answer.paths import answer
    from thesiskb.core.db import session_scope
    from thesiskb.llm import get_gateway
    from thesiskb.retrieval.hybrid import SearchFilters

    filters = SearchFilters(source_types=source_type or None, verified_only=verified_only)
    with session_scope() as session:
        ans = answer(
            session,
            get_gateway(),
            question,
            get_pipeline_config().retrieval,
            filters,
            path=path,
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
    if ans.route:
        who = f" ({ans.route['person']})" if ans.route.get("person") else ""
        console.print(
            f"\n[dim]route: {' + '.join(ans.route['paths'])}{escape(who)} "
            f"[{ans.route['by']}: {escape(ans.route.get('reason') or '')}][/]"
        )
    console.print(f"[dim]{' · '.join(f'{k} {v:.1f}s' for k, v in t.items())}[/]")
    for r in ans.retrieval:
        if cov := r.get("coverage"):
            missing = f"; not covered: {', '.join(cov['missing'])}" if cov["missing"] else ""
            console.print(f"[dim]parts covered: {cov['cited']}/{cov['parts']}{escape(missing)}[/]")
    if ans.trace_id:
        console.print(f"[dim]trace: tkb trace {ans.trace_id[:8]}[/]")


@app.command()
def trace(
    ref: str = typer.Argument("last", help="'last' or a trace id (prefix)"),
    as_json: bool = typer.Option(False, "--json"),
    hits: int = typer.Option(5, help="Reranked hits to show per query"),
) -> None:
    """Show how a question was answered: queries, hits, passages, judge verdicts."""
    from thesiskb.answer.trace import format_trace, get_trace, trace_out
    from thesiskb.core.db import session_scope

    with session_scope() as session:
        t = get_trace(session, ref)
        if t is None:
            console.print(f"[red]No trace found for {escape(ref)}[/]")
            raise typer.Exit(code=1)
        out = trace_out(t)
    if as_json:
        import json
        import sys

        sys.stdout.write(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        return
    console.print(escape(format_trace(out, hits)), highlight=False)


@app.command()
def report(
    out: Path | None = typer.Option(
        None, help="Zip file to write (default: data/logs/report-*.zip)"
    ),
    lines: int = typer.Option(1000, help="Log lines to include"),
    with_trace: bool = typer.Option(
        False, "--trace", help="Also include the last answer trace (question, answer, passages)"
    ),
) -> None:
    """Bundle logs, doctor results, versions and config (secrets removed) for a bug report."""
    from thesiskb.report import build_report, default_report_path

    trace_data = None
    if with_trace:
        from thesiskb.answer.trace import get_trace, trace_out
        from thesiskb.core.db import session_scope

        with session_scope() as session:
            t = get_trace(session, "last")
            trace_data = trace_out(t) if t else None
    path = build_report(out or default_report_path(), lines, trace_data)
    console.print(f"[green]Report written:[/] {escape(str(path))}")
    console.print(
        "[dim]Contains log lines (with file names), doctor results, versions and config; "
        "secrets are replaced by ***. Look through it before sharing.[/]"
    )


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
    from sqlalchemy import any_, func, select

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
            select(Person).where(
                (func.lower(Person.name) == wanted) | (name == any_(Person.aliases))
            )
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


tasks_app = typer.Typer(
    help="Tasks extracted from notes and meetings.", invoke_without_command=True
)
app.add_typer(tasks_app, name="tasks")


def _task_by_prefix(session, prefix: str):
    from sqlalchemy import String, cast, select

    from thesiskb.core.models import Task

    hits = session.scalars(select(Task).where(cast(Task.id, String).like(f"{prefix}%"))).all()
    if len(hits) != 1:
        console.print(f"[red]'{escape(prefix)}' matches {len(hits)} tasks[/]")
        raise typer.Exit(code=1)
    return hits[0]


@tasks_app.callback()
def tasks_list(
    ctx: typer.Context,
    all_: bool = typer.Option(False, "--all", help="Include done tasks"),
) -> None:
    """List open tasks by due date (no subcommand), or review/confirm/done them."""
    if ctx.invoked_subcommand:
        return
    from thesiskb.answer.generate import locator_label
    from thesiskb.answer.tasks import open_tasks
    from thesiskb.core.db import session_scope
    from thesiskb.core.models import Chunk, Source

    table = Table(title="Tasks")
    for col in ("id", "Due", "Written", "Task", "Owner", "Status", "✓", "Source"):
        table.add_column(col, overflow="fold")
    with session_scope() as session:
        for t, owner in open_tasks(session, include_done=all_):
            src = session.get(Source, t.source_id) if t.source_id else None
            chunk = session.get(Chunk, t.source_chunk_id) if t.source_chunk_id else None
            where = src.path.rsplit("/", 1)[-1] if src else "?"
            if chunk:
                where += f", {locator_label(chunk.locator)}"
            elif src:
                where += " [red](evidence link lost)[/]"
            table.add_row(
                str(t.id)[:8],
                t.due_date.isoformat() if t.due_date else "—",
                escape(t.due_raw or ""),
                escape(t.description),
                escape(owner or t.owner_raw or ""),
                t.status,
                "✓" if t.confirmed else "",
                where,
            )
    console.print(table)
    console.print("[dim]tkb tasks review · tkb tasks done <id> · tkb tasks confirm <id>[/]")


@tasks_app.command("review")
def tasks_review() -> None:
    """Go through unconfirmed tasks: confirm, edit, delete or skip each one."""
    from datetime import date

    from sqlalchemy import select

    from thesiskb.core.db import session_scope
    from thesiskb.core.models import Task
    from thesiskb.ingest.dates import resolve_due

    with session_scope() as session:
        todo = session.scalars(
            select(Task).where(Task.confirmed.is_(False)).order_by(Task.due_date)
        ).all()
        if not todo:
            console.print("Nothing to review.")
            return
        for t in todo:
            console.print(
                f"\n[bold]{escape(t.description)}[/]  due {t.due_date or '—'} "
                f"(written: {escape(t.due_raw or '—')}), owner: {escape(t.owner_raw or '—')}\n"
                f'  [dim]evidence: "{escape(t.evidence_quote or "")}"[/]'
            )
            choice = typer.prompt("[c]onfirm / [e]dit / [d]elete / [s]kip / [q]uit", default="c")
            if choice.startswith("q"):
                break
            if choice.startswith("d"):
                session.delete(t)
            elif choice.startswith("e"):
                t.description = typer.prompt("description", default=t.description)
                t.owner_raw = typer.prompt("owner", default=t.owner_raw or "") or None
                due = typer.prompt(
                    "due (as written, e.g. 2027-01-31 or Jan 2027)", default=t.due_raw or ""
                )
                if due != (t.due_raw or ""):
                    t.due_raw = due or None
                    t.due_date = resolve_due(due, t.due_date or date.today())
                t.confirmed = True
            elif choice.startswith("c"):
                t.confirmed = True
            session.commit()
    console.print("[green]saved[/]")


@tasks_app.command("confirm")
def tasks_confirm(task_id: str) -> None:
    """Mark a task as checked by you (kept even when its source is re-extracted)."""
    from thesiskb.core.db import session_scope

    with session_scope() as session:
        t = _task_by_prefix(session, task_id)
        t.confirmed = True
        console.print(f"confirmed: {escape(t.description)}")


@tasks_app.command("done")
def tasks_done(task_id: str) -> None:
    """Mark a task as done (also confirms it)."""
    from thesiskb.core.db import session_scope

    with session_scope() as session:
        t = _task_by_prefix(session, task_id)
        t.status, t.confirmed = "done", True
        console.print(f"done: {escape(t.description)}")


@app.command()
def person(name: str = typer.Argument(..., help='Name or alias, e.g. "Will.i.am"')) -> None:
    """Person profile: statements grouped by topic, recordings they speak in, papers they wrote."""
    from collections import defaultdict

    from sqlalchemy import func, or_, select

    from thesiskb.core.db import session_scope
    from thesiskb.core.models import Source, Speaker, Statement
    from thesiskb.ingest.extract import resolve_person

    with session_scope() as session:
        p = resolve_person(session, name)
        if p is None:
            console.print(
                f"[red]No person '{escape(name)}'. Label a speaker first: "
                'tkb speakers label <file> <LABEL> "<name>"[/]'
            )
            raise typer.Exit(code=1)
        console.print(
            f"[bold]{escape(p.name)}[/]"
            + (f" ({escape(p.role)})" if p.role else "")
            + (f"  aliases: {escape(', '.join(p.aliases))}" if p.aliases else "")
        )

        recordings = session.execute(
            select(Source.path, Speaker.diarization_label)
            .join(Speaker, Speaker.source_id == Source.id)
            .where(Speaker.person_id == p.id)
        ).all()
        if recordings:
            console.print("\n[bold]Speaks in[/]")
            for path, label in recordings:
                console.print(f"  {escape(path.rsplit('/', 1)[-1])} ({label})")

        authors = func.array_to_string(Source.authors, "|")
        names = [p.name, *(p.aliases or [])]
        papers = session.scalars(
            select(Source).where(or_(*[authors.ilike(f"%{n}%") for n in names]))
        ).all()
        if papers:
            console.print("\n[bold]Authored[/]")
            for s in papers:
                year = f" ({s.published.year})" if s.published else ""
                console.print(f"  {escape(s.title or s.path)}{year}")

        by_topic = defaultdict(list)
        for st in session.scalars(select(Statement).where(Statement.person_id == p.id)):
            by_topic[st.topic].append(st)
        if by_topic:
            console.print("\n[bold]Statements by topic[/]")
            for topic, sts in sorted(by_topic.items()):
                console.print(f"  [bold]{escape(topic)}[/]")
                for st in sts:
                    console.print(f'    {escape(st.stance)}  [dim]"{escape(st.evidence_quote)}"[/]')
        console.print(f'\n[dim]Ask: tkb ask "What does {escape(p.name)} think about …?"[/]')


@app.command()
def statements(
    person: str | None = typer.Option(None, help="Name, alias or speaker label (e.g. SPEAKER_02)"),
) -> None:
    """What people say or prefer (extracted from recordings), with evidence."""
    from sqlalchemy import select

    from thesiskb.answer.generate import locator_label
    from thesiskb.core.db import session_scope
    from thesiskb.core.models import Chunk, Person, Source, Statement
    from thesiskb.ingest.extract import quote_in, resolve_person

    table = Table(title="Statements")
    for col in ("Person", "Topic", "Stance", "Source"):
        table.add_column(col, overflow="fold")
    with session_scope() as session:
        stmt = select(Statement, Person.name).outerjoin(Person, Person.id == Statement.person_id)
        if person:
            p = resolve_person(session, person)
            stmt = stmt.where(
                (Statement.person_id == p.id) if p else (Statement.person_raw == person)
            )
        for st, name in session.execute(stmt.order_by(Person.name, Statement.person_raw)):
            src = session.get(Source, st.source_id)
            chunk = session.get(Chunk, st.source_chunk_id) if st.source_chunk_id else None
            where = src.path.rsplit("/", 1)[-1][:40]
            if chunk:
                # the transcript line holding the quote gives the exact time
                line = next(
                    (ln for ln in chunk.text.splitlines() if quote_in(st.evidence_quote, ln)), ""
                )
                stamp = line[1 : line.index("]")] if line.startswith("[") and "]" in line else ""
                where += f", {stamp or locator_label(chunk.locator)}"
            table.add_row(
                escape(name or st.person_raw or "?"),
                escape(st.topic),
                escape(st.stance) + f'\n[dim]"{escape(st.evidence_quote)}"[/]',
                escape(where),
            )
    console.print(table)


@app.command()
def verify(
    source: str = typer.Argument(..., help="sha256 prefix or part of filename"),
    status: str = typer.Option("verified", help="verified | disputed | unverified"),
    by: str | None = typer.Option(None, "--by", help="Who checked it (shown with citations)"),
) -> None:
    """Record that you checked a source. `tkb ask --verified-only` then uses only these."""
    from datetime import UTC, datetime

    from thesiskb.core.db import session_scope

    if status not in ("verified", "disputed", "unverified"):
        console.print("[red]--status must be verified, disputed or unverified[/]")
        raise typer.Exit(code=1)
    with session_scope() as session:
        src = _resolve_source(session, source)
        src.verification_status = status
        src.verified_by = by if status != "unverified" else None
        src.verified_at = datetime.now(UTC) if status != "unverified" else None
        name = src.path.rsplit("/", 1)[-1]
    console.print(f"{escape(name)}: {status}" + (f" by {escape(by)}" if by else ""))


@app.command()
def consent(
    source: str = typer.Argument(..., help="sha256 prefix or part of filename"),
    note: str = typer.Argument(..., help='e.g. "verbal consent of all 3 participants, 2026-11-04"'),
) -> None:
    """Record how consent was given for a recording of people (BUILD_GUIDE §10)."""
    from thesiskb.core.db import session_scope

    with session_scope() as session:
        src = _resolve_source(session, source)
        src.consent_note = note
        name = src.path.rsplit("/", 1)[-1]
    console.print(f"{escape(name)}: consent noted")


@app.command("local-only")
def local_only_cmd(
    source: str = typer.Argument(..., help="sha256 prefix or part of filename"),
    off: bool = typer.Option(False, "--off", help="Allow cloud models again"),
) -> None:
    """Keep a source's content away from cloud models (API mode); local fallbacks are used."""
    from thesiskb.core.db import session_scope

    with session_scope() as session:
        src = _resolve_source(session, source)
        src.local_only = not off
        name = src.path.rsplit("/", 1)[-1]
    state = "may be sent to cloud models" if off else "local models only"
    console.print(f"{escape(name)}: {state}")


@app.command("forget-person")
def forget_person(
    name: str = typer.Argument(..., help="Name or alias"),
    delete_recordings: bool = typer.Option(
        False, help="Also remove every recording they speak in from the index"
    ),
    yes: bool = typer.Option(False, "--yes", help="Don't ask for confirmation"),
) -> None:
    """Delete a person's data (BUILD_GUIDE §10): statements, speaker labels, profile; their
    name is removed from transcripts by re-indexing. Original files are never deleted."""
    import shutil

    from sqlalchemy import delete, select, update

    from thesiskb.core.db import session_scope
    from thesiskb.core.models import Source, Speaker, Statement, Task
    from thesiskb.ingest.extract import resolve_person
    from thesiskb.ingest.pipeline import processed_dir, rechunk
    from thesiskb.llm import get_gateway

    cfg = get_pipeline_config()
    with session_scope() as session:
        p = resolve_person(session, name)
        if p is None:
            console.print(f"[red]No person '{escape(name)}'[/]")
            raise typer.Exit(code=1)
        source_ids = list(
            session.scalars(select(Speaker.source_id).where(Speaker.person_id == p.id).distinct())
        )
        sources = [session.get(Source, i) for i in source_ids]
        n_st = len(session.scalars(select(Statement.id).where(Statement.person_id == p.id)).all())
        console.print(
            f"[bold]{escape(p.name)}[/]: {n_st} statements, speaker in {len(sources)} "
            f"recording(s): " + ", ".join(escape(s.path.rsplit("/", 1)[-1]) for s in sources)
        )
        if delete_recordings:
            console.print(
                "[yellow]--delete-recordings: those recordings leave the index "
                "(their processed files are deleted; your originals are kept)[/]"
            )
        if not yes and not typer.confirm("Delete this person's data?"):
            raise typer.Exit(code=1)

        session.execute(delete(Statement).where(Statement.person_id == p.id))
        session.execute(
            update(Task).where(Task.owner_person_id == p.id).values(owner_person_id=None)
        )
        session.execute(update(Speaker).where(Speaker.person_id == p.id).values(person_id=None))
        session.delete(p)
        session.commit()

        gateway = get_gateway()
        for src in sources:
            label = src.path.rsplit("/", 1)[-1]
            if delete_recordings:
                shutil.rmtree(processed_dir(cfg, src.sha256), ignore_errors=True)
                session.delete(src)
                session.commit()
                console.print(
                    f"  removed from index: {escape(label)} (original: {escape(src.path)})"
                )
            else:
                # chunks/extractions are rebuilt with the anonymous label (SPEAKER_03)
                (processed_dir(cfg, src.sha256) / "extract.json").unlink(missing_ok=True)
                rechunk(session, src, cfg, gateway)
                console.print(f"  re-indexed without the name: {escape(label)}")
    console.print(
        "[green]done[/] — the voice itself is still in those recordings; use "
        "--delete-recordings (and delete the originals) to remove it entirely."
    )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Use 0.0.0.0 only behind a VPN, with a token"),
    port: int = typer.Option(8000),
) -> None:
    """Start the REST API + web UI (and the OpenAI-compatible /v1 endpoint)."""
    import uvicorn

    settings = get_settings()
    if host not in ("127.0.0.1", "localhost") and not settings.api_token:
        console.print("[red]Refusing to listen on a network interface without TKB_API_TOKEN[/]")
        raise typer.Exit(code=1)
    console.print(f"ThesisKB on http://{host}:{port}  (API docs: /docs)")
    uvicorn.run("thesiskb.api.app:app", host=host, port=port, log_level="info")


@app.command()
def worker(
    once: bool = typer.Option(False, "--once", help="Process the queue, then exit"),
) -> None:
    """Process the ingest queue (items added from the web UI/API) without the web server.
    `tkb serve` already runs a worker; use this on a machine that only ingests."""
    from rich.logging import RichHandler

    from thesiskb.core.db import get_engine
    from thesiskb.core.logs import set_console_handler
    from thesiskb.ingest.queue import work

    set_console_handler(RichHandler(console=console, show_path=False, show_time=True))
    console.print("Waiting for ingest requests… (Ctrl+C to stop)" if not once else "")
    try:
        work(get_engine(), once=once)
    except KeyboardInterrupt:
        console.print("stopped")


@app.command("mcp")
def mcp_cmd() -> None:
    """Run the MCP server on stdio (for Claude Desktop/Code and other MCP clients)."""
    from thesiskb.mcp.server import main

    main()


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
        target = Path(src.path.split("#", 1)[0])  # CSV row → the CSV file
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
    oracle: bool = typer.Option(False, help="Use each question's gold path instead of the router"),
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
            oracle=oracle,
        )
        prev = previous_summary(session)

    table = Table(title="Eval")
    cols = (
        "id",
        "type",
        "route",
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
            ("+".join(r.routed) + ("" if r.route_ok else " [red]✗[/]"))
            if r.status == "ran"
            else f"[dim]{r.reason}[/]",
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


entities_app = typer.Typer(help="Entities in the knowledge graph.", no_args_is_help=True)
app.add_typer(entities_app, name="entities")


@entities_app.command("merges")
def entities_merges(
    limit: int = typer.Option(100, help="Rows per table"),
):
    """Automatic entity merges (newest first), then near-duplicates left as suggestions."""
    from sqlalchemy import select

    from thesiskb.core.db import session_scope
    from thesiskb.core.models import Entity, EntityMerge

    with session_scope() as session:
        for status, title in (("merged", "Automatic merges"), ("suggested", "Suggestions")):
            rows = session.execute(
                select(EntityMerge, Entity.name)
                .join(Entity, Entity.id == EntityMerge.kept_id)
                .where(EntityMerge.status == status)
                .order_by(EntityMerge.created_at.desc(), EntityMerge.id.desc())
                .limit(limit)
            ).all()
            table = Table(title=f"{title} ({len(rows)})")
            for col in ("Kept", "Merged" if status == "merged" else "Similar to", "Why", "Sim."):
                table.add_column(col, overflow="fold")
            for m, kept in rows:
                why = {
                    "key": "same name",
                    "variant": "spelling/plural",
                    "shared": "similar + neighbour",
                }
                table.add_row(
                    escape(kept),
                    escape(m.other_name),
                    why.get(m.reason, m.reason),
                    f"{m.similarity:.3f}" if m.similarity is not None else "",
                )
            console.print(table)
    console.print("[dim]A wrong merge? Report it with the two names (docs/usage.md).[/]")


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


def main() -> None:
    """Entry point of `tkb`: the CLI, with a readable message when the database is down."""
    from sqlalchemy.exc import OperationalError

    from thesiskb.core.logs import setup_logging

    setup_logging()
    try:
        app()
    except OperationalError as e:
        if "connect" not in str(e).lower() and "timeout" not in str(e).lower():
            raise
        url = get_settings().database_url.rsplit("@", 1)[-1]  # host:port/db, no password
        console.print(
            f"[red]Can't reach the database ({escape(url)}).[/]\n"
            "Is Docker Desktop running? Start it, then run:  docker compose up -d\n"
            "[dim]Check everything with: uv run tkb doctor[/]"
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
