"""Task path (Q3: "who does what until when?"): rows come from SQL, the LLM only phrases them.

Each open task becomes a passage [S<n>] — its due date (resolved), description, owner and the
verbatim evidence quote — cited back to the chunk it was extracted from, so the usual citation
checks and the support judge apply unchanged.
"""

from __future__ import annotations

import time

from sqlalchemy import select
from sqlalchemy.orm import Session

from thesiskb.answer.generate import REFUSAL, Answer, Citation, generate_from
from thesiskb.core.models import Chunk, Person, Source, Task
from thesiskb.llm.gateway import LLMGateway
from thesiskb.planning import visible_tasks

TASK_RULES = """- "Task:" passages are rows of the user's task table, already sorted by due date. \
Keep that (chronological) order when listing several. State each date as written in the source \
("written: Jan 2027" → "January 2027"); the resolved "due" date is only for ordering and is \
often just the end of a month — never present it as an exact day the source did not give."""


def open_tasks(session: Session, include_done: bool = False) -> list[tuple[Task, str | None]]:
    stmt = (
        select(Task, Person.name)
        .outerjoin(Person, Person.id == Task.owner_person_id)
        .order_by(Task.due_date.is_(None), Task.due_date, Task.description)
    )
    if not include_done:
        stmt = stmt.where(Task.status != "done")
    return list(session.execute(stmt.where(visible_tasks())).all())  # no unaccepted plan items


def task_row_text(t: Task, owner: str | None) -> str:
    from thesiskb.ingest.dates import is_month_only

    if t.due_date is None:
        due = "no date"
    elif is_month_only(t.due_raw):
        # the day is only a sort key (end of month): never show it as if the source gave it
        due = t.due_date.strftime("%B %Y")
    else:
        due = t.due_date.strftime("%d %B %Y").lstrip("0")
    written = f" (written: {t.due_raw})" if t.due_raw else ""
    who = owner or t.owner_raw or "unassigned"
    flags = f" | status: {t.status}" + (" | confirmed" if t.confirmed else "")
    evidence = f'\nevidence: "{t.evidence_quote}"' if t.evidence_quote else ""
    return f"Task: {t.description} | due {due}{written} | owner: {who}{flags}{evidence}"


def task_citations(
    session: Session, rows: list[tuple[Task, str | None]] | None = None, start: int = 1
) -> list[Citation]:
    rows = open_tasks(session) if rows is None else rows
    cites = []
    for t, owner in rows:
        chunk = session.get(Chunk, t.source_chunk_id) if t.source_chunk_id else None
        src = session.get(Source, t.source_id) if t.source_id else None
        if chunk is None or src is None:
            continue  # no evidence link: never shown as a fact
        cites.append(
            Citation(
                n=start + len(cites),
                chunk_id=chunk.id,
                filename=src.path.rsplit("/", 1)[-1],
                title=src.title,
                context_header=chunk.context_header,
                locator=chunk.locator,
                verification_status=src.verification_status,
                text=task_row_text(t, owner),
                kind="task",
                local_only=src.local_only,
            )
        )
    return cites


def answer_tasks(
    session: Session, gateway: LLMGateway, question: str, verify_support: bool = True
) -> Answer:
    from thesiskb.answer.verify import verify

    t0 = time.perf_counter()
    cites = task_citations(session)
    t1 = time.perf_counter()
    if not cites:
        return Answer(
            question=question,
            text=REFUSAL,
            text_with_ids=REFUSAL,
            refused=True,
            timings={"sql": t1 - t0},
        )
    ans = generate_from(gateway, question, cites, extra_rules=TASK_RULES)
    t2 = time.perf_counter()
    verify(gateway, ans, check_support=verify_support and not ans.refused)
    ans.timings = {"sql": t1 - t0, "generate": t2 - t1, "verify": time.perf_counter() - t2}
    return ans
