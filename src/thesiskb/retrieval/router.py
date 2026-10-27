"""Route a question to one or more answer paths (BUILD_GUIDE §3, Milestone 6).

  literature  papers/books: sub-queries, evidence summaries, synthesis   (Q1)
  person      what a known person says/prefers/wrote                      (Q2)
  task        deadlines, schedule, who does what until when — SQL         (Q3)
  general     hybrid search over everything (recordings, notes, ...)
  summary     overview of one whole source (book, paper, meeting), from ingest-time summaries

High-precision rules run first: a known person's name or alias in the question adds `person`,
schedule/deadline words add `task`. An LLM call (role `router`, JSON) classifies the rest and may
add paths; if it fails, rules decide and `general` is the fallback. Paths whose sources do not
exist in the KB (e.g. literature without papers) fall back to `general`.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from thesiskb.core.models import Chunk, Entity, Person, Source, Task
from thesiskb.llm.gateway import LLMGateway

log = logging.getLogger(__name__)

Path = Literal["literature", "person", "task", "entity", "general", "summary"]
PATHS: tuple[str, ...] = ("literature", "person", "task", "entity", "general", "summary")

# "Summarize the book", "Give me an overview of the paper", "What is the meeting about?"
_DOC = (
    r"(book|paper|article|document|chapter|report|meeting|recording|talk|video|page|survey|"
    r"buch|artikel|dokument|besprechung|vortrag)"
)
_SUMMARY_WORDS = re.compile(
    rf"\b(summari[sz]e|summary|overview|recap|main points|key points|gist|zusammenfass\w*|"
    rf"überblick)\b.{{0,60}}\b{_DOC}\b|\bwhat (is|was) (the|this|that) {_DOC} about\b|"
    rf"\bfasse\b.{{0,60}}\bzusammen\b",
    re.I,
)

# "Which NVIDIA products appear in my sources?", "List all datasets mentioned"
_ENTITY_WORDS = re.compile(
    r"\b(which|what|list|all|welche)\b.{0,60}\b(appear|appears|mentioned|occur|in my "
    r"(sources|notes|files|recordings|papers)|across|kommen vor|erwähnt)\b",
    re.I,
)

_TASK_WORDS = re.compile(
    r"\b(schedule|deadlines?|due|tasks?|to-?dos?|milestones?|timeline|until when|by when|"
    r"who does what|when is my|when are my|what do i have to|zeitplan|frist(en)?|termin(e)?|"
    r"aufgaben?|bis wann|wer macht was)\b",
    re.I,
)

_LITERATURE_WORDS = re.compile(
    r"\b(papers?|survey|literature|research|study|studies|publication|state of the art|sota|"
    r"method|approach|forschung|literatur|veröffentlichung)\b",
    re.I,
)


def _is_author(session: Session, person_id: str) -> bool:
    import uuid

    p = session.get(Person, uuid.UUID(person_id))
    if p is None:
        return False
    authors = func.array_to_string(Source.authors, "|")
    names = [p.name, *(p.aliases or [])]
    return bool(
        session.scalar(select(func.count()).where(or_(*[authors.ilike(f"%{n}%") for n in names])))
    )


ROUTER_PROMPT = """Classify a question for a personal thesis knowledge base. Paths:
- literature: what research papers/books say — methods, state of the art, findings, comparisons \
of techniques in the literature.
- person: what a specific named person says, thinks, prefers, recommends or wrote.
- task: the user's own deadlines, schedule, milestones, to-dos, who does what until when.
- entity: lists or aggregations across sources — which products / methods / datasets / people \
appear, everything connected to X.
- summary: a summary or overview of one whole source — "summarize the book", "what is the \
meeting about", "main points of the paper".
- general: anything else — products, recordings, meetings, notes, demos, facts in any source.
A question can need more than one path (e.g. "What did Prof. X say about my deadline?" = \
person + task).

Known people: {people}

Return JSON: {{"paths": ["..."], "person": "<name as in the question, or null>", \
"reason": "<few words>"}}

Question: {question}"""


class Route(BaseModel):
    paths: list[str] = Field(default_factory=lambda: ["general"])
    person: str | None = None  # name as written in the question
    person_id: str | None = None  # resolved Person.id
    reason: str = ""
    by: str = "rules"  # rules | llm | rules+llm | forced


class _LLMRoute(BaseModel):
    paths: list[str]
    person: str | None = None
    reason: str = ""


def known_people(session: Session) -> list[Person]:
    return list(session.scalars(select(Person)))


def find_person(question: str, people: list[Person]) -> tuple[Person, str] | None:
    """A known person whose name or alias appears in the question (longest match wins)."""
    q = question.lower()
    best: tuple[Person, str] | None = None
    for p in people:
        for name in [p.name, *(p.aliases or [])]:
            pattern = r"(?<![\w.])" + re.escape(name.lower()) + r"(?![\w])"
            if name and re.search(pattern, q) and (best is None or len(name) > len(best[1])):
                best = (p, name)
    return best


def rule_route(question: str, people: list[Person]) -> Route:
    paths: list[str] = []
    person = find_person(question, people)
    if person:
        paths.append("person")
    if _TASK_WORDS.search(question):
        paths.append("task")
    if _ENTITY_WORDS.search(question):
        paths.append("entity")
    if _SUMMARY_WORDS.search(question):
        paths.append("summary")
    route = Route(paths=paths or ["general"], reason="rules")
    if person:
        route.person, route.person_id = person[1], str(person[0].id)
    return route


def llm_route(gateway: LLMGateway, question: str, people: list[Person]) -> _LLMRoute | None:
    names = ", ".join(p.name for p in people) or "none"
    try:
        raw = gateway.chat(
            [{"role": "user", "content": ROUTER_PROMPT.format(people=names, question=question)}],
            role="router",
            response_format=_LLMRoute,
        )
        out = _LLMRoute.model_validate(json.loads(raw))
    except (ValidationError, ValueError) as e:
        log.warning("router LLM output unusable: %s", e)
        return None
    out.paths = [p for p in dict.fromkeys(out.paths) if p in PATHS]
    return out if out.paths else None


def available_paths(session: Session) -> set[str]:
    have = {"general"}
    if session.scalar(select(func.count()).where(Source.source_type.in_(("paper", "book")))):
        have.add("literature")
    if session.scalar(select(func.count()).select_from(Task)):
        have.add("task")
    if session.scalar(select(func.count()).select_from(Person)):
        have.add("person")
    if session.scalar(select(func.count()).select_from(Entity)):
        have.add("entity")
    if session.scalar(select(func.count()).where(Chunk.locator.has_key("summary"))):
        have.add("summary")
    return have


def route_question(
    session: Session, gateway: LLMGateway | None, question: str, use_llm: bool = True
) -> Route:
    people = known_people(session)
    route = rule_route(question, people)
    llm = llm_route(gateway, question, people) if (use_llm and gateway) else None
    if llm:
        paths = list(dict.fromkeys([*[p for p in route.paths if p != "general"], *llm.paths]))
        route.reason = llm.reason or route.reason
        route.by = "rules+llm" if route.paths != ["general"] else "llm"
        if "person" in paths and not route.person_id and llm.person:
            match = find_person(llm.person, people)
            route.person = llm.person
            if match:
                route.person_id = str(match[0].id)
        route.paths = paths
    have = available_paths(session)
    if "person" in route.paths and not route.person_id:
        have.discard("person")  # an unknown person can't be filtered on: search everything
    if (
        "person" in route.paths
        and "literature" in route.paths
        and route.person_id
        and not _LITERATURE_WORDS.search(question)
        and not _is_author(session, route.person_id)
    ):
        route.paths.remove("literature")  # costly path that can't hold this person's views
    kept = [p for p in route.paths if p in have]
    if len(kept) < len(route.paths) and "general" not in kept:
        kept.append("general")
    # entity lookups only see passages linked to extracted entities; never let them replace
    # search (comparisons routed to entity alone lost unlinked evidence)
    if "entity" in kept and not {"general", "literature"} & set(kept):
        kept.append("general")
    # a whole-document question: the part summaries cover it; top-k search would only add a
    # few similar chunks and crowd the context
    if "summary" in kept and _SUMMARY_WORDS.search(question):
        kept = [p for p in kept if p not in ("general", "literature", "entity")]
    route.paths = kept or ["general"]
    return route
