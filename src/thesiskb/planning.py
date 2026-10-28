"""Task editing, plans towards a deadline and proposed task updates (Milestone 15, ADR 0016).

edit       add / edit / delete tasks from the UI and CLI. Typed tasks have origin "manual",
           are confirmed and need no evidence link.
plan       the user picks a goal task with a due date, the tasks (and sources) it depends
           on, hours per week and days/weeks to leave out. The LLM only proposes the parts
           ("Literature review part 1: …") with hour estimates; `schedule` computes the dates
           in code, backwards from the deadline. Parts are saved with origin "plan",
           unconfirmed, parent = the goal, and stay hidden until the plan is accepted;
           rejecting deletes them.
proposals  extraction may find a newer source with another due date for a known task, or a
           task reported as done. It never edits the task: it adds a `task_proposals` row
           that the user accepts or rejects.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import date, timedelta

from pydantic import BaseModel, Field
from sqlalchemy import and_, delete, not_, select
from sqlalchemy.orm import Session

from thesiskb.core.models import Chunk, Source, Task, TaskProposal
from thesiskb.llm.gateway import LLMGateway

log = logging.getLogger(__name__)

HOURS_PER_WEEK = 10.0
MAX_PARTS = 20


def hidden_plan_items():
    """SQL condition for plan parts not accepted yet: never listed, shown or answered from."""
    return and_(Task.origin == "plan", Task.confirmed.is_(False))


def visible_tasks():
    return not_(hidden_plan_items())


# --- edit ------------------------------------------------------------------------------------


class TaskEdit(BaseModel):
    """Fields a user can set; None leaves a field unchanged on update."""

    description: str | None = None
    owner: str | None = None
    due_date: date | None = None
    start_date: date | None = None
    estimate_hours: float | None = Field(None, ge=0)
    status: str | None = Field(None, pattern="^(open|done)$")
    notes: str | None = None
    clear: list[str] = Field(default_factory=list)  # fields to set to empty


def apply_edit(session: Session, t: Task, e: TaskEdit) -> Task:
    from thesiskb.ingest.extract import resolve_person

    for f in ("description", "due_date", "start_date", "estimate_hours", "status", "notes"):
        v = getattr(e, f)
        if v is not None:
            setattr(t, f, v.strip() if isinstance(v, str) else v)
    if e.due_date is not None:
        t.due_raw = None  # the date is the user's now, not as written in a source
    if e.owner is not None:
        person = resolve_person(session, e.owner)
        t.owner_raw = e.owner.strip() or None
        t.owner_person_id = person.id if person else None
    for f in e.clear:
        if f == "owner":
            t.owner_raw = t.owner_person_id = None
        elif f in ("due_date", "start_date", "estimate_hours", "notes"):
            setattr(t, f, None)
            if f == "due_date":
                t.due_raw = None
    if not (t.description or "").strip():
        raise ValueError("a task needs a description")
    t.confirmed = True  # edited by the user = checked by the user
    session.flush()
    return t


def add_task(session: Session, e: TaskEdit) -> Task:
    t = Task(description=e.description or "", origin="manual", status="open")
    session.add(t)
    return apply_edit(session, t, e)


# --- plan ------------------------------------------------------------------------------------


def parse_excluded(items: list[str]) -> set[date]:
    """'2026-12-24', '2026-12-24..2027-01-02' (inclusive) or ISO weeks like '2026-W52'."""
    out: set[date] = set()
    for raw in items:
        s = raw.strip()
        if not s:
            continue
        if m := re.fullmatch(r"(\d{4})-W(\d{1,2})", s, re.I):
            monday = date.fromisocalendar(int(m[1]), int(m[2]), 1)
            out |= {monday + timedelta(days=i) for i in range(7)}
        elif ".." in s:
            a, b = (date.fromisoformat(x.strip()) for x in s.split("..", 1))
            out |= {a + timedelta(days=i) for i in range((b - a).days + 1)}
        else:
            out.add(date.fromisoformat(s))
    return out


class Slot(BaseModel):
    start: date
    end: date


def schedule(
    hours: list[float],
    deadline: date,
    hours_per_week: float = HOURS_PER_WEEK,
    excluded: set[date] | None = None,
) -> list[Slot]:
    """Dates for parts done in this order, packed backwards so the last one ends the day before
    the deadline. Every day (holidays too) gives hours_per_week / 7, except excluded days."""
    if hours_per_week <= 0:
        raise ValueError("hours per week must be positive")
    excluded = excluded or set()
    per_day = hours_per_week / 7
    day, left_today = deadline - timedelta(days=1), per_day
    slots: list[Slot] = []
    for h in reversed(hours):
        need = max(h, 0.0)
        while day in excluded:
            day, left_today = day - timedelta(days=1), per_day
        end = day
        while True:
            while day in excluded:
                day, left_today = day - timedelta(days=1), per_day
            take = min(need, left_today)
            need -= take
            left_today -= take
            if need <= 1e-9:
                break
            day, left_today = day - timedelta(days=1), per_day
        slots.append(Slot(start=day, end=end))
        if left_today <= 1e-9:  # this day is full: the previous part ends the day before
            day, left_today = day - timedelta(days=1), per_day
    return list(reversed(slots))


class XPart(BaseModel):
    title: str
    part_of: str  # title of the task it belongs to
    estimate_hours: float = Field(ge=0.5, le=200)
    notes: str = ""


class XPlan(BaseModel):
    parts: list[XPart] = Field(default_factory=list, max_length=MAX_PARTS)


PLAN_PROMPT = """You help a student plan work towards a deadline.
Goal: {goal} (due {due})
It depends on these tasks, in this order:
{deps}
{sources}
Split each task above into 1-4 concrete parts, in the order they must be done. Name every part
"<task title> part <n>: <what to do>". Give each part an estimate in hours. Use only the tasks
and material above; do not add new topics. Do not give dates.

Return JSON: {{"parts": [{{"title", "part_of" (the task title exactly as given),
"estimate_hours", "notes" (one short sentence or empty)}}]}}"""


class PlanItem(BaseModel):
    id: uuid.UUID
    title: str
    part_of: str | None
    estimate_hours: float | None
    start_date: date | None
    due_date: date | None
    notes: str | None


class PlanProposal(BaseModel):
    goal_id: uuid.UUID
    goal: str
    deadline: date
    items: list[PlanItem]
    warnings: list[str] = Field(default_factory=list)


def _source_context(session: Session, source_ids: list[uuid.UUID]) -> str:
    lines = []
    for sid in source_ids:
        src = session.get(Source, sid)
        if src is None:
            continue
        summary = session.scalar(
            select(Chunk.text).where(
                Chunk.source_id == sid, Chunk.locator["summary"].astext == "document"
            )
        )
        title = src.title or src.path.rsplit("/", 1)[-1]
        lines.append(f"- {title}: {(summary or '')[:1200]}".rstrip(": "))
    return "Material to work through:\n" + "\n".join(lines) + "\n" if lines else ""


def propose_plan(
    session: Session,
    gateway: LLMGateway,
    goal_id: uuid.UUID,
    depends_on: list[uuid.UUID],
    source_ids: list[uuid.UUID] | None = None,
    hours_per_week: float = HOURS_PER_WEEK,
    excluded: list[str] | None = None,
    today: date | None = None,
) -> PlanProposal:
    """Ask the LLM for parts, schedule them, save them as hidden plan items (replacing an
    earlier proposal for this goal that was not accepted)."""
    goal = session.get(Task, goal_id)
    if goal is None:
        raise ValueError("no such goal task")
    if goal.due_date is None:
        raise ValueError("the goal task needs a due date")
    deps = [session.get(Task, d) for d in depends_on]
    deps = [d for d in deps if d is not None and d.id != goal.id]
    if not deps:
        raise ValueError("pick at least one task the goal depends on")
    skip = parse_excluded(excluded or [])
    today = today or date.today()

    dep_lines = "\n".join(
        f"- {d.description}"
        + (f" (about {d.estimate_hours:g} h)" if d.estimate_hours else "")
        + (f" — notes: {d.notes}" if d.notes else "")
        for d in deps
    )
    prompt = PLAN_PROMPT.format(
        goal=goal.description,
        due=goal.due_date.isoformat(),
        deps=dep_lines,
        sources=_source_context(session, source_ids or []),
    )
    # tasks are personal: the plan is made by the local model even if a cloud role exists
    raw = gateway.chat(
        [{"role": "user", "content": prompt}],
        role=gateway.role_for("extraction", True),
        response_format=XPlan,
    )
    parts = XPlan.model_validate(json.loads(raw)).parts

    # keep the user's order of tasks; a part naming an unknown task joins the nearest by title
    by_title = {d.description.strip().lower(): d for d in deps}
    ordered: list[tuple[XPart, Task]] = []
    for d in deps:
        mine = [p for p in parts if _owner_task(p, by_title, deps) is d]
        if not mine:  # the model skipped it: schedule the task itself as one part
            mine = [
                XPart(
                    title=d.description, part_of=d.description, estimate_hours=d.estimate_hours or 4
                )
            ]
        if d.estimate_hours:  # the user's estimate wins: scale the parts to it
            total = sum(p.estimate_hours for p in mine)
            mine = [
                p.model_copy(update={"estimate_hours": p.estimate_hours * d.estimate_hours / total})
                for p in mine
            ]
        ordered += [(p, d) for p in mine]

    slots = schedule([p.estimate_hours for p, _ in ordered], goal.due_date, hours_per_week, skip)
    reject_plan(session, goal.id)
    items = []
    for (p, d), slot in zip(ordered, slots, strict=True):
        t = Task(
            description=p.title.strip(),
            notes=p.notes.strip() or None,
            estimate_hours=round(p.estimate_hours, 1),
            start_date=slot.start,
            due_date=slot.end,
            owner_raw=d.owner_raw,
            owner_person_id=d.owner_person_id,
            parent_id=goal.id,
            origin="plan",
            confirmed=False,
        )
        session.add(t)
        session.flush()
        items.append(_item(t, d.description))

    warnings = []
    if slots and slots[0].start < today:
        warnings.append(
            f"The plan would have to start on {slots[0].start.isoformat()}, before today: "
            "allow more hours per week or move the deadline."
        )
    return PlanProposal(
        goal_id=goal.id, goal=goal.description, deadline=goal.due_date, items=items,
        warnings=warnings,
    )  # fmt: skip


def _owner_task(p: XPart, by_title: dict[str, Task], deps: list[Task]) -> Task:
    key = p.part_of.strip().lower()
    if key in by_title:
        return by_title[key]
    words = set(re.findall(r"\w{3,}", f"{p.part_of} {p.title}".lower()))
    return max(deps, key=lambda d: len(words & set(re.findall(r"\w{3,}", d.description.lower()))))


def _item(t: Task, part_of: str | None) -> PlanItem:
    return PlanItem(
        id=t.id,
        title=t.description,
        part_of=part_of,
        estimate_hours=t.estimate_hours,
        start_date=t.start_date,
        due_date=t.due_date,
        notes=t.notes,
    )


def pending_plan(session: Session, goal_id: uuid.UUID) -> list[Task]:
    return list(
        session.scalars(
            select(Task)
            .where(Task.parent_id == goal_id, hidden_plan_items())
            .order_by(Task.start_date, Task.due_date)
        )
    )


def accept_plan(session: Session, goal_id: uuid.UUID) -> int:
    items = pending_plan(session, goal_id)
    for t in items:
        t.confirmed = True
    session.flush()
    return len(items)


def reject_plan(session: Session, goal_id: uuid.UUID) -> int:
    n = session.execute(delete(Task).where(Task.parent_id == goal_id, hidden_plan_items()))
    session.flush()
    return n.rowcount or 0


# --- proposals from extraction ---------------------------------------------------------------


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w{3,}", text.lower())) - {"the", "and", "for", "mit", "und", "der"}


def same_task(a: str, b: str) -> bool:
    """Two descriptions of one task: the same numbers ("chapter 2" is not "chapter 3"), and most
    words of the shorter one appear in the longer one."""
    if re.findall(r"\d+", a) != re.findall(r"\d+", b):
        return False
    wa, wb = _words(a), _words(b)
    small = min(len(wa), len(wb))
    return small >= 2 and len(wa & wb) >= 0.7 * small


def propose_updates(session: Session, src: Source, src_date: date) -> int:
    """Compare this source's freshly extracted tasks with known tasks from older sources (and
    the user's own): a different due date or a "done" becomes an open proposal."""
    new = session.scalars(select(Task).where(Task.source_id == src.id)).all()
    if not new:
        return 0
    known = session.execute(
        select(Task, Source)
        .outerjoin(Source, Source.id == Task.source_id)
        .where(
            Task.source_id.is_distinct_from(src.id),
            Task.source_id.is_distinct_from(src.replaces),  # its own older version
            Task.status != "done",
            visible_tasks(),
        )
    ).all()
    made = 0
    for t in new:
        for k, ksrc in known:
            if not same_task(t.description, k.description):
                continue
            kdate = (
                (ksrc.published or (ksrc.ingested_at and ksrc.ingested_at.date())) if ksrc else None
            )
            if kdate and kdate >= src_date:
                continue  # only a newer source may update a task
            changes = []
            if t.due_date and t.due_date != k.due_date:
                changes.append(("due_date", t.due_date.isoformat(), t.due_raw))
            if t.status == "done":
                changes.append(("status", "done", None))
            for field, value, value_raw in changes:
                exists = session.scalar(
                    select(TaskProposal.id).where(
                        TaskProposal.task_id == k.id,
                        TaskProposal.field == field,
                        TaskProposal.value == value,
                    )
                )
                if exists:
                    continue  # also when it was rejected: don't ask again
                session.add(
                    TaskProposal(
                        task_id=k.id,
                        field=field,
                        value=value,
                        value_raw=value_raw,
                        source_id=src.id,
                        chunk_id=t.source_chunk_id,
                        evidence_quote=t.evidence_quote,
                    )
                )
                made += 1
    session.flush()
    return made


def decide_proposal(session: Session, proposal_id: int, accept: bool) -> TaskProposal:
    p = session.get(TaskProposal, proposal_id)
    if p is None or p.status != "open":
        raise ValueError("no open proposal with that id")
    if accept:
        t = session.get(Task, p.task_id)
        if p.field == "due_date":
            t.due_date, t.due_raw = date.fromisoformat(p.value), p.value_raw
        elif p.field == "status":
            t.status = p.value
        t.confirmed = True
    p.status = "accepted" if accept else "rejected"
    session.flush()
    return p
