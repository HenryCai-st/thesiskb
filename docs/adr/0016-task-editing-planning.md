# ADR 0016 — Task editing, plans towards a deadline, proposed updates, week/month views

Date: 2026-10-28 · Status: accepted

Context: tasks were only extracted; users want to add and change them, plan thesis work by
week towards a deadline, and stay in control of what the model changes (feedback point 8,
BUILD_GUIDE_v2 Milestone 15).

## Decisions
- **Fields** (migration 0006): `notes`, `start_date`, `estimate_hours`, `parent_id` (→ the
  goal a plan part leads to; deleting the goal deletes its parts), `origin` = extracted |
  manual | plan. Typed tasks are `manual`, confirmed, and have no evidence link; the
  "tasks without evidence" eval counter only looks at extracted tasks. The task *answer*
  path still cites only tasks with an evidence chunk (unchanged).
- **Plan**: the user picks a goal with a due date, ticks the tasks it depends on in the order
  they must be done, optionally sources (their M12 document summary is given to the model),
  hours per week (default 10) and days to leave out (`2026-12-24`, ranges `a..b`, ISO weeks
  `2026-W52`). The local extraction model (always local: tasks are personal) only returns
  parts with hour estimates; a user estimate on a task wins (parts are scaled to it).
  `schedule()` computes dates in code: every day — holidays included, as decided — gives
  hours/7, excluded days give nothing, parts are packed backwards so the last ends the day
  before the deadline. A plan that would start before today gets a warning, not a refusal.
- **Hidden until accepted**: parts are saved as `origin=plan`, unconfirmed, and filtered out
  of every task list, view and answer (`visible_tasks()`); accept confirms them, reject (or
  a new proposal for the same goal) deletes them, so a rejected plan leaves no row.
- **Proposed updates** (`task_proposals`): after extraction, each new task is compared with
  known tasks from *older* sources (or typed ones): same numbers and ≥ 70 % of the shorter
  description's words. A different due date, or `status: "done"` (new extraction field),
  becomes an open proposal with the evidence quote. The model never edits a task; a
  rejected proposal is not asked again. Cached extractions have no status field and stay
  "open" (no re-extraction forced).
- **Views**: a small grid in the existing static UI, no calendar library: week view (default,
  ◀ ▶, tasks on their due day, multi-day plan parts on every day they span) and month view
  (hover or click a week to list its tasks). Edit, plan and proposal forms open in the side
  panel. CLI: `tkb tasks add | edit | delete | plan`, proposals in `tkb tasks review`.
