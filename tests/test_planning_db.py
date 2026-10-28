"""Task editing, plans and proposed updates through the API, on the test database."""

import json
import os
from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.orm import Session

from tests.conftest import requires_db
from thesiskb.core.models import Chunk, Source, Task, TaskProposal
from thesiskb.planning import propose_updates

pytestmark = [pytest.mark.db, requires_db]

TAG = "m15test"


class FakeGateway:
    """Returns the plan the 'model' proposes; records the prompt."""

    def __init__(self, parts):
        self.parts, self.prompts = parts, []

    def role_for(self, role, local_only):
        return role

    def chat(self, messages, role="answer", response_format=None, **kw):
        self.prompts.append(messages[0]["content"])
        return json.dumps({"parts": self.parts})


@pytest.fixture
def engine(db_url):
    eng = create_engine(db_url)

    def clean():
        with Session(eng) as s:
            s.execute(delete(Task).where(Task.description.like(f"%{TAG}%")))
            s.execute(delete(Source).where(Source.path.like(f"%{TAG}%")))
            s.commit()

    clean()
    yield eng
    clean()
    eng.dispose()


@pytest.fixture
def client(engine, monkeypatch):
    from thesiskb.api import app as app_module

    def session_override():
        with Session(engine) as s:
            yield s
            s.commit()

    app_module.app.dependency_overrides[app_module.get_session] = session_override
    yield app_module, TestClient(app_module.app)
    app_module.app.dependency_overrides.clear()


def _mine(c) -> list[dict]:
    return [t for t in c.get("/api/tasks?include_done=true").json() if TAG in t["description"]]


def test_add_edit_delete_manual_tasks(client):
    _, c = client
    t = c.post("/api/tasks", json={"description": f"{TAG} Thesis draft", "due_date": "2026-12-15"})
    assert t.status_code == 200
    t = t.json()
    assert (t["origin"], t["confirmed"], t["chunk_id"]) == ("manual", True, None)
    assert c.post("/api/tasks", json={"description": " "}).status_code == 400

    e = c.patch(
        f"/api/tasks/{t['id']}",
        json={"owner": "Henry", "estimate_hours": 6, "notes": "two chapters", "status": "done"},
    ).json()
    assert (e["owner"], e["estimate_hours"], e["notes"], e["status"]) == (
        "Henry", 6, "two chapters", "done"
    )  # fmt: skip
    e = c.patch(f"/api/tasks/{t['id']}", json={"clear": ["owner", "due_date"]}).json()
    assert e["owner"] is None and e["due_date"] is None and e["notes"] == "two chapters"

    assert c.delete(f"/api/tasks/{t['id']}").status_code == 200
    assert _mine(c) == []


def test_december_deadline_plan_accept_and_reject(client, engine, monkeypatch):
    """Acceptance: a December deadline with two dependent tasks gets a weekly plan; accepted
    items show up (with dates for the week/month views); a rejected plan leaves no trace."""
    app_module, c = client
    goal = c.post(
        "/api/tasks", json={"description": f"{TAG} Hand in thesis", "due_date": "2026-12-18"}
    ).json()
    lit = c.post(
        "/api/tasks", json={"description": f"{TAG} Literature review", "estimate_hours": 12}
    ).json()
    study = c.post("/api/tasks", json={"description": f"{TAG} User study"}).json()
    parts = [
        {"title": f"{TAG} User study part 1: recruit", "part_of": f"{TAG} User study",
         "estimate_hours": 4},
        {"title": f"{TAG} Literature review part 1: search", "part_of": f"{TAG} Literature review",
         "estimate_hours": 3},
        {"title": f"{TAG} Literature review part 2: write", "part_of": f"{TAG} Literature review",
         "estimate_hours": 3},
        {"title": f"{TAG} User study part 2: run", "part_of": "user study", "estimate_hours": 6},
    ]  # fmt: skip
    gw = FakeGateway(parts)
    monkeypatch.setattr("thesiskb.llm.get_gateway", lambda: gw)
    body = {
        "goal_id": goal["id"],
        "depends_on": [lit["id"], study["id"]],
        "hours_per_week": 14,
        "exclude": ["2026-W50"],
    }
    before = len(_mine(c))

    # reject: nothing stays behind
    p = c.post("/api/plans", json=body).json()
    assert len(p["items"]) == 4 and len(_mine(c)) == before  # proposals are hidden
    assert c.post(f"/api/plans/{goal['id']}/reject").json() == {"rejected": 4}
    with Session(engine) as s:
        assert s.scalar(select(func.count()).where(Task.parent_id == goal["id"])) == 0
    assert len(_mine(c)) == before

    # propose again and accept
    p = c.post("/api/plans", json=body).json()
    titles = [i["title"] for i in p["items"]]
    assert titles[:2] == [
        f"{TAG} Literature review part 1: search",
        f"{TAG} Literature review part 2: write",
    ]
    # the user's 12 h for the literature review wins over the model's 3 + 3
    assert [i["estimate_hours"] for i in p["items"]] == [6.0, 6.0, 4.0, 6.0]
    ends = [date.fromisoformat(i["due_date"]) for i in p["items"]]
    starts = [date.fromisoformat(i["start_date"]) for i in p["items"]]
    assert ends == sorted(ends) and max(ends) < date(2026, 12, 18)
    assert all(not (date(2026, 12, 7) <= d <= date(2026, 12, 13)) for d in ends + starts)  # W50
    assert "Literature review" in gw.prompts[0] and "2026-12-18" in gw.prompts[0]
    assert len(c.get(f"/api/plans/{goal['id']}").json()) == 4  # kept, still pending
    assert c.post(f"/api/plans/{goal['id']}/accept").json() == {"accepted": 4}
    shown = {t["description"]: t for t in _mine(c)}
    assert all(shown[t]["origin"] == "plan" and shown[t]["start_date"] for t in titles)

    # a plan for a goal without a due date is refused with a readable message
    r = c.post("/api/plans", json=body | {"goal_id": lit["id"], "depends_on": [study["id"]]})
    assert r.status_code == 400 and "due date" in r.json()["detail"]


def test_newer_source_proposes_due_date_and_done_never_edits(client, engine):
    _, c = client
    with Session(engine) as s:
        old = Source(sha256=os.urandom(32).hex(), path=f"/x/{TAG}-old.md", modality="document",
                     source_type="meeting", published=date(2026, 10, 1))  # fmt: skip
        new = Source(sha256=os.urandom(32).hex(), path=f"/x/{TAG}-new.md", modality="document",
                     source_type="meeting", published=date(2026, 10, 20))  # fmt: skip
        s.add_all([old, new])
        s.flush()
        ch = Chunk(source_id=new.id, text="Expose now due 30.11.", locator={"line": 1})
        s.add(ch)
        s.flush()
        known = Task(description=f"Submit expose to Weber {TAG}", due_date=date(2026, 11, 15),
                     source_id=old.id, confirmed=True)  # fmt: skip
        done_known = Task(description=f"Book the lab room {TAG}", source_id=old.id)
        s.add_all([known, done_known])
        s.flush()
        s.add_all(
            [
                Task(description=f"submit the expose to Weber {TAG}", due_date=date(2026, 11, 30),
                     due_raw="30.11.", source_id=new.id, source_chunk_id=ch.id,
                     evidence_quote="Expose now due 30.11."),
                Task(description=f"Lab room booked {TAG}", source_id=new.id, status="done"),
                Task(description=f"book the lab room {TAG}", source_id=new.id, status="done"),
            ]
        )  # fmt: skip
        s.flush()
        assert propose_updates(s, new, date(2026, 10, 20)) == 2
        assert propose_updates(s, new, date(2026, 10, 20)) == 0  # not twice
        # an older source never proposes
        assert propose_updates(s, old, date(2026, 10, 1)) == 0
        s.commit()
        known_id, done_id = known.id, done_known.id

    props = [p for p in c.get("/api/task-proposals").json() if TAG in p["task"]["description"]]
    by_field = {p["field"]: p for p in props}
    assert by_field["due_date"]["value"] == "2026-11-30"
    assert by_field["due_date"]["evidence_quote"] == "Expose now due 30.11."
    with Session(engine) as s:  # nothing changed yet
        assert s.get(Task, known_id).due_date == date(2026, 11, 15)
        assert s.get(Task, done_id).status == "open"

    c.post(f"/api/task-proposals/{by_field['due_date']['id']}/accept")
    c.post(f"/api/task-proposals/{by_field['status']['id']}/reject")
    with Session(engine) as s:
        t = s.get(Task, known_id)
        assert (t.due_date, t.due_raw) == (date(2026, 11, 30), "30.11.")
        assert s.get(Task, done_id).status == "open"
        assert {
            p.status
            for p in s.scalars(
                select(TaskProposal).where(TaskProposal.task_id.in_([known_id, done_id]))
            )
        } == {"accepted", "rejected"}
    assert c.post(f"/api/task-proposals/{by_field['status']['id']}/accept").status_code == 404
