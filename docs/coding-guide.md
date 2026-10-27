# Coding guide — scope and definition of done

Rules for anyone (human or agent) implementing a milestone from
[BUILD_GUIDE.md](../BUILD_GUIDE.md) or [BUILD_GUIDE_v2.md](BUILD_GUIDE_v2.md).
[AGENTS.md](../AGENTS.md) still applies; this guide adds the scope rules.

## 1. Scope: build only what the milestone lists

- Work on **one milestone at a time**. Start by copying its task list and acceptance
  criteria into the PR/commit description; that list is the whole job.
- **No extra features.** If it is not in the milestone's tasks, do not build it — not as a
  "small extra", not behind a flag, not as an unused option. Write it down instead (section 4).
- **No speculative generality.** No plugin points, settings, abstract base classes or
  parameters for cases nobody asked for. Add a setting only when the milestone names it or
  two real users need different values.
- **No drive-by refactors.** Change existing code only where the task needs it. A refactor
  that is needed gets its own commit with no behaviour change.
- **Prefer the smallest change** that meets the acceptance criteria: reuse existing modules,
  tables and UI patterns before adding new ones. A new dependency needs a reason in the
  commit message; a new service (container, daemon) needs an ADR.
- **"Not in scope" lines in the guide are binding.** If one turns out to block acceptance,
  stop and ask; do not work around it.

## 2. Before you write code

1. Read the milestone and the code it touches.
2. Reproduce the problem (for bug-driven milestones) with a test or a script, and note the
   cause in the commit or ADR.
3. If a task is ambiguous, pick the narrowest reading and say so in the commit message, or
   ask the user. Do not resolve ambiguity by building both options.

## 3. Definition of done (every milestone)

- [ ] Every task in the milestone is done, and nothing else.
- [ ] Each acceptance criterion is checked, and how is written in the commit/PR
      (test name, command output or screenshot).
- [ ] Tests: new behaviour has CPU-only tests with mocked models; GPU tests behind the
      `gpu` marker. `uv run pytest` and `uv run ruff check` pass; CI is green.
- [ ] Retrieval, prompt or model changes: `tkb eval` run before and after, diff reported.
- [ ] Schema changes only through an Alembic migration, with a downgrade.
- [ ] Each design decision has a short ADR in `docs/adr/`.
- [ ] User-facing changes are reflected in `docs/usage.md` (and `README.md` if it is a
      headline feature). No other doc rewrites.
- [ ] No `data/`, `.env`, model weights, recordings or personal information committed.
- [ ] Tag the milestone (`mNN`).

## 4. Ideas that are out of scope

Good ideas found during a milestone go into the "Open questions" section of the build guide
(or a GitHub issue) with one line on why. They are built only after they are added to a
milestone.

## 5. Code style reminders

- Match the surrounding code: naming, comment density, module layout.
- Pydantic models at module boundaries; type hints everywhere.
- All model calls through `src/thesiskb/llm/`; no vendor SDK imports elsewhere.
- Pipeline stages idempotent, keyed by sha256, outputs in `data/processed/<sha256>/`.
- Every chunk has `source_id` and a locator.
- Errors the user can act on get a readable message (what failed, what to do), and the
  full traceback goes to the log.
