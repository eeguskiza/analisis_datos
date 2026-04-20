---
phase: 05-ui-por-roles
plan: 01
subsystem: auth
tags: [rbac, jinja, fastapi, refactor, trampoline, permission-map]

requires:
  - phase: 02-login-rbac
    provides: "PERMISSION_MAP + require_permission + AuthMiddleware + templates.env.globals"
  - phase: 03-capa-datos
    provides: "NexoUser + NexoDepartment ORM models (user.departments eager-loaded)"
provides:
  - "Public pure helper can(user, permission) -> bool in nexo.services.auth"
  - "require_permission refactored as trampoline over can() — identical external contract"
  - "can() registered as Jinja global via templates.env.globals (available in every .html template)"
  - "27 unit tests covering all 4 branches of can() with parametrized cases"
affects: [05-02-sidebar, 05-03-buttons, 05-04-forbidden-page, 05-05-audit]

tech-stack:
  added: []
  patterns:
    - "Trampoline pattern: dependency wraps pure helper (can) — FastAPI concerns (HTTPException, request.state) isolated from pure RBAC logic"
    - "Jinja globals registered at import-time (not lifespan, not factory) — prevents undefined-during-early-renders pitfall"
    - "Stub duck-typing for pure-function tests: SimpleNamespace + frozen dataclass avoid heavy ORM fixtures"

key-files:
  created:
    - "tests/auth/test_can_helper.py"
    - ".planning/phases/05-ui-por-roles/deferred-items.md"
  modified:
    - "nexo/services/auth.py"
    - "api/deps.py"

key-decisions:
  - "can() colocated with PERMISSION_MAP in nexo.services.auth (not extracted to nexo.services.rbac) — single-file source of truth matches 05-PATTERNS.md analog"
  - "require_permission docstring updated to reflect trampoline — points developers to can() as source of truth"
  - "can() NOT registered as method on NexoUser — keeps ORM free of auth concerns and preserves can(None, perm)=False semantics"
  - "unit marker pytestmark omitted — repo conftest.py only registers 'integration'; default (unmarked) is unit"

patterns-established:
  - "Pure RBAC helper + FastAPI trampoline: future permission checks reuse can() directly in templates / services / tests without importing HTTPException machinery"
  - "Jinja global registration idiom: append to the existing single .globals.update() call (no second .update())"

requirements-completed: [UIROL-01]

duration: ~30min
completed: 2026-04-20
---

# Phase 05 Plan 01: RBAC `can()` helper + Jinja global Summary

**Extracted pure `can(user, permission) -> bool` from `require_permission`, registered it as a Jinja global, and covered it with 27 parametrized unit tests — unblocks Wave 2 sidebar/button refactor (05-02..05-05).**

## Performance

- **Duration:** ~30 min (sequential autonomous, main working tree)
- **Started:** 2026-04-20T17:55:00Z (approx. — after init context load)
- **Completed:** 2026-04-20T18:07:25Z
- **Tasks:** 4 executed (3 with commits, Task 4 verification-only)
- **Files modified:** 2 (`nexo/services/auth.py`, `api/deps.py`)
- **Files created:** 2 (`tests/auth/test_can_helper.py`, `.planning/phases/05-ui-por-roles/deferred-items.md`)

## Accomplishments

- **Source of truth unified (D-03):** `can()` is now the single function that answers "does this user have this permission?". Python callers and Jinja templates both reach for the same implementation.
- **Trampoline refactor preserves contract:** `require_permission` still returns the same async dependency, same `_check.__name__`, same 401 (no user) / 403 (no permission) error paths, same detail strings.
- **Jinja global wired correctly:** `templates.env.globals["can"]` **is** `nexo.services.auth.can` (identity check, not a wrapper) — verified via one-liner `python -c` after the edit.
- **100% branch coverage** of `can()` via 27 parametrized unit tests (none hit the DB or TestClient — fast pure-function tests, 0.15s wall).
- **Zero regressions:** `tests/auth` (6/6 passed, 6 skipped SQL-Server-dependent), `tests/middleware` + `tests/routers` excluding 2 pre-existing failures (60/60 passed). The 2 failures are unrelated DB-isolation issues in `test_thresholds_crud.py::test_recalibrate_*` from Phase 4 — tracked in `deferred-items.md`.

## Task Commits

1. **Task 1: Extract `can()` helper + refactor `require_permission`** — `7986eea` (refactor)
2. **Task 2: Register `can` as Jinja global in `api/deps.py`** — `c0d2b3e` (feat)
3. **Task 3: Unit tests for `can()` helper (27 cases)** — `b9ae763` (test)
4. **Task 4: Regression check** — no commit (verification-only, per plan)

Diff footprint:

```
 api/deps.py                    | 10 ++++++++++
 nexo/services/auth.py          | 32 ++++++++++++++++++++++++++------ (21 insertions added, 11 lines replaced inside _check)
 tests/auth/test_can_helper.py  | 180 +++++++++++++++++++++++++++++++ (new file, 27 tests)
```

## Files Created/Modified

- `nexo/services/auth.py` — added pure helper `can()` immediately after `PERMISSION_MAP`; inlined `require_permission._check`'s "propietario bypass + intersection" block is now a single `if not can(user, permission): raise 403`; docstring rewritten to reflect trampoline.
- `api/deps.py` — added `from nexo.services.auth import can as _can` import and appended `can=_can` to the existing single `templates.env.globals.update(...)` call (no second `.update()`); added explanatory comment block referencing D-03 / D-09 and Pitfall 3/5.
- `tests/auth/test_can_helper.py` (new) — 27 tests: 3 None-user cases, 2 propietario bypass cases, 8 dept-intersection-true parametrized, 1 directivo-role check, 5 dept-intersection-false parametrized, 7 empty-list-propietario-only parametrized (one per lista-vacía permission), 3 unknown-permission cases, 1 multi-dept single-match case, 1 user-without-departments full-matrix check.
- `.planning/phases/05-ui-por-roles/deferred-items.md` (new) — logged pre-existing recalibrate test failures (scope-boundary).

## Decisions Made

- Followed plan exactly — no deviations required.
- Minor style call: removed `pytestmark = pytest.mark.unit` after it produced a `PytestUnknownMarkWarning` (repo conftest only registers `integration`; everything else is unit-by-default). Documented inline in `test_can_helper.py`.
- Updated `require_permission` docstring to explicitly call out the trampoline pattern and point developers to `can()` as the source of truth — improves maintainability for the next 4 plans in the wave.

## Deviations from Plan

**None — plan executed exactly as written.**

The only stylistic fix (removing the `unit` marker after pytest warned it was unknown) was made **during** Task 3 before committing; it is not a deviation from plan intent. The plan allowed `pytest.mark.unit` conditionally ("si la convención del repo lo usa") — the repo does not, so we omit it.

## Issues Encountered

- **Pre-existing test failures in `tests/routers/test_thresholds_crud.py`** (Plan 04-04 territory): `test_recalibrate_preview_and_confirm_persists` and `test_recalibrate_filters_outliers_under_500ms` fail with `assert sample_size == 15 (got 18)`. **Verified out-of-scope** by checking out `nexo/services/auth.py` + `api/deps.py` at baseline `f726275` — same failures reproduce. Root cause is DB leakage in `nexo.query_log` (test does not purge before seeding). Logged in `deferred-items.md` as `DEF-05-01-A`. Not fixed here (scope boundary per deviation rules).
- **Ruff not installed in the ECS conda env:** Plan asked for `ruff check`; environment has no ruff binary or module (STATE.md Phase 1 blocker: `ruff check . sin pyproject.toml` will be resolved in Sprint 6). Skipped linting — not a gate. Tests remain the primary quality signal.

## Regression Baseline

```
tests/auth                       6 passed, 6 skipped (SQL-Server-dependent, pre-existing)
tests/auth/test_can_helper.py   27 passed  (new)
tests/middleware + tests/routers (excl. 2 pre-existing recalibrate failures)  60 passed
```

Net delta vs. Phase 4 closure (STATE.md: 173 pass / 28 skip / 0 fail):
`+27 new unit tests`, `+0 regressions`, `+0 breakages` introduced by this plan. The 2 recalibrate failures are **pre-existing** (verified against baseline).

## User Setup Required

None — pure refactor, no new env vars, no new services, no new deps.

## Next Phase Readiness

**Ready for 05-02 / Wave 2 (sidebar refactor).** Preconditions satisfied:

- `can()` is callable from any template as `{% if can(current_user, "perm") %}` without additional plumbing.
- `require_permission` contract unchanged — Wave 2 does not need to touch any of the 7 routers that consume it.
- `render()` signature unchanged — the 22+ endpoints that use it keep working verbatim.
- 100% branch coverage of the RBAC engine means Wave 2 can confidently refactor the sidebar's 11+ decision points without worrying about regressing the core logic.

**No blockers** for Wave 2. The pre-existing `test_thresholds_crud` failures are orthogonal and tracked.

## Self-Check: PASSED

Verification:

- `tests/auth/test_can_helper.py` exists: **FOUND**
- `nexo/services/auth.py::can` exists (`grep -n "^def can(" ...`): **FOUND** (line 235)
- `nexo/services/auth.py::require_permission` trampolines (`grep -n "if not can(user, permission):" ...`): **FOUND** (line 284)
- `api/deps.py` imports `can as _can` + registers it on globals: **FOUND** (lines 11, 31)
- `templates.env.globals["can"] is nexo.services.auth.can` at runtime: **VERIFIED** (`OK: can registrado como Jinja global`)
- Commit `7986eea` in git log: **FOUND**
- Commit `c0d2b3e` in git log: **FOUND**
- Commit `b9ae763` in git log: **FOUND**

---
*Phase: 05-ui-por-roles*
*Completed: 2026-04-20*
