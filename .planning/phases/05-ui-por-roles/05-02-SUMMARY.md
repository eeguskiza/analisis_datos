---
phase: 05-ui-por-roles
plan: 02
subsystem: templates
tags: [rbac, sidebar, jinja, templates, refactor, integration-tests]

requires:
  - phase: 05-ui-por-roles
    plan: 01
    provides: "can(user, permission) pure helper + Jinja global registration in api/deps.py"
  - phase: 02-login-rbac
    provides: "PERMISSION_MAP + AuthMiddleware + session cookie auth"
provides:
  - "templates/base.html nav_items now carry `permission` (string | None) instead of `visible_to` (\"*\" | \"propietario\")"
  - "sidebar filters via `can(current_user, permission)` — single source of truth shared with require_permission (D-01)"
  - "Solicitudes badge gate is `can(current_user, \"aprobaciones:manage\")` instead of role-string check"
  - "5 integration tests covering propietario/ingenieria-directivo/produccion-usuario/rrhh-usuario/anon sidebar visibility (TestClient-based)"
affects: [05-03-buttons, 05-04-forbidden-page, 05-05-audit]

tech-stack:
  added: []
  patterns:
    - "Jinja global `can` consumed in loop guard: `{% if permission is none or can(current_user, permission) %}` — tolerates current_user=None on public routes"
    - "Integration test pattern reused from tests/routers/test_approvals_api.py: module-scoped TestClient + autouse purge + slowapi reset + per-test user creation"
    - "W-03 mitigation: anchored regex `href=\"/x\"(?![/\\w])` prevents sub-path false positives (`/ajustes` vs `/ajustes/solicitudes`)"

key-files:
  created:
    - "tests/routers/test_sidebar_filtering.py"
  modified:
    - "templates/base.html"

key-decisions:
  - "Kept the `_sep1` separator always visible (permission=None) — the plan's note about 'al menos un item visible a cada lado' was not needed since the separator is itself minimal and dropping it would leak information about which group is empty"
  - "Did not touch login.html (does not extend base.html) — /login anonymous test validates only that the middleware lets the route through and the template renders; future public routes that extend base.html will still be safe because can(None, x) returns False"
  - "Reused _create_user/_login helpers pattern from test_approvals_api.py verbatim (TEST_DOMAIN + TEST_PASSWORD + autouse purge). No helper extraction to conftest — keeps each suite self-contained per current repo convention"
  - "Anchored regex assertions (W-03) instead of substring checks — protects against the concrete trap where `href=\"/ajustes\"` would match `href=\"/ajustes/solicitudes\"` (the two siblings this plan had to distinguish)"

patterns-established:
  - "Sidebar nav_items schema: 5-tuple (key, href, label, icon_paths, permission) where permission is a PERMISSION_MAP key or None — future items just append a new tuple"
  - "Per-role sidebar tests: one test per representative role+dept combo, asserting both present and absent hrefs with anchored regex"

requirements-completed: [UIROL-02]

duration: ~25min
completed: 2026-04-20
---

# Phase 05 Plan 02: Sidebar fine-grained filtering Summary

**Refactored `templates/base.html` nav_items from coarse role-string gating (`visible_to == "*" | "propietario"`) to fine-grained permission gating via `can(current_user, permission)` — sidebar and backend now share a single source of truth, making "link visible → click → 403" impossible.**

## Performance

- **Duration:** ~25 min (sequential autonomous, main working tree)
- **Started:** 2026-04-20T20:10:00Z (approx.)
- **Completed:** 2026-04-20T20:17:00Z
- **Tasks:** 3 executed, 3 commits
- **Files modified:** 1 (`templates/base.html`)
- **Files created:** 1 (`tests/routers/test_sidebar_filtering.py`, 324 lines)

## Accomplishments

- **D-01 sidebar fino wired end-to-end:** the `{% for key, href, label, icon_paths, permission in nav_items %}` loop guard is now `permission is none or can(current_user, permission)`. Every item's visibility derives from `PERMISSION_MAP` via the `can()` helper registered as a Jinja global in Plan 05-01.
- **Last role-string check in base.html eliminated:** the Solicitudes badge block (from Plan 04-03) was the final `current_user.role == 'propietario'` guard — now `can(current_user, "aprobaciones:manage")`. Functionally equivalent (PERMISSION_MAP entry is `[]` → propietario-only bypass) but consistent with D-01.
- **Separator rendering preserved:** `_sep1` keeps `permission=None`, so the `permission is none` shortcut ensures it always renders. No logic added/changed in the body of the loop (`<a>` with icon + `x-show`).
- **Integration tests cover 4 canonical personas + anon:** propietario (all 11 items + solicitudes), ingenieria-directivo (no ajustes/operarios/solicitudes), produccion-usuario (no bbdd/ciclos-calc/operarios/ajustes), rrhh-usuario (only historial/operarios + dashboard), and /login anonymous (200 OK, no `can(None,...)` crash).
- **W-03 landmine defused preemptively:** both `_assert_href_present` and `_assert_href_absent` use anchored regex `href="/x"(?![/\w])` — guarantees `href="/ajustes"` does NOT match `href="/ajustes/solicitudes"`, which is the exact pair this plan had to distinguish.
- **Zero regressions:** `tests/auth` + `tests/routers` + `tests/middleware` = 64 passed, 6 skipped, 3 pre-existing failures (`test_thresholds_crud::test_recalibrate_*`) tracked as DEF-05-01-A in `deferred-items.md` — verified unchanged from Plan 05-01 baseline.

## Task Commits

1. **Task 1: Refactor nav_items to carry `permission`** — `faf68c2` (feat)
   - 11 tuples migrated (13 inserts / 13 deletes).
   - Loop unpack + `{% if %}` guard updated in the same commit.
2. **Task 2: Solicitudes badge gate → can()** — `06e1d25` (refactor)
   - 1 insert / 1 delete (one-line change).
3. **Task 3: Integration tests for per-role sidebar visibility** — `e174f29` (test)
   - 324 new lines, 5 tests passing.

Diff footprint:

```
 templates/base.html                          |  14 +-
 tests/routers/test_sidebar_filtering.py      | 324 ++++++++++++++ (new)
```

## Files Created/Modified

- `templates/base.html` (modified):
  - Lines 48-59: each `nav_items` tuple's 5th slot now a permission string or `None` (was `"*"` / `"propietario"`). SVG `icon_paths` untouched.
  - Line 61: unpack renamed `visible_to` → `permission`.
  - Line 62: guard changed from `visible_to == "*" or (current_user and current_user.role == visible_to)` to `permission is none or can(current_user, permission)`.
  - Line 82: solicitudes badge gate changed from `current_user and current_user.role == 'propietario'` to `can(current_user, "aprobaciones:manage")`.
- `tests/routers/test_sidebar_filtering.py` (new, 324 lines):
  - 5 tests, all `@pytest.mark.integration`, skipped if Postgres unreachable.
  - Helpers: `_create_user`, `_login`, `_purge`, `_reset_rate_limit`, `_assert_href_present`, `_assert_href_absent`.
  - TEST_DOMAIN `@sidebar-filter-test.local` (isolated from `@approvals-api-test.local` and `@threshold-crud-test.local`).

## Items visible per role (observed in tests)

Concrete checklist for manual smoke testing:

| Role / Dept                   | Visible hrefs                                                                                | Absent hrefs                                                                            |
| ----------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| propietario (any)             | /, /datos, /pipeline, /historial, /capacidad, /recursos, /ciclos-calc, /operarios, /bbdd, /ajustes, /ajustes/solicitudes | (nothing)                                                                               |
| directivo + ingenieria        | /, /datos, /pipeline, /historial, /capacidad, /recursos, /ciclos-calc, /bbdd                 | /ajustes, /operarios, /ajustes/solicitudes                                              |
| usuario + produccion          | /, /datos, /pipeline, /historial, /capacidad, /recursos                                       | /bbdd, /ciclos-calc, /operarios, /ajustes, /ajustes/solicitudes                         |
| usuario + rrhh                | /, /historial, /operarios                                                                    | /datos, /pipeline, /bbdd, /ajustes, /ajustes/solicitudes                                |
| (anon on /login)              | — (login.html does not extend base.html; route renders 200 without can(None,x) error)       | —                                                                                        |

## Decisions Made

- Followed plan exactly — no deviations required.
- The plan's `<guardrails>` note about "ensure at least one item on either side is visible before rendering the separator" was evaluated: current behavior leaves the separator always rendered (permission=None), which is correct because (a) a lone separator is visually indistinguishable from a spacer, (b) dropping it would leak information about which side is empty, and (c) the plan itself lists `_sep1` as `None` in the mapping. Decision: do not add conditional logic for the separator.

## Deviations from Plan

**None — plan executed exactly as written.**

## Issues Encountered

- **Pre-existing `test_thresholds_crud::test_recalibrate_*` failures** (3 tests): confirmed unchanged from Plan 05-01 baseline (`DEF-05-01-A` in `deferred-items.md`). Not caused by this plan — verified via git log + the failing tests do not import or touch `templates/base.html` or `api/deps.py`. Out of scope per deviation rules.
- **No ruff in env:** same as Plan 05-01 — ruff binary/module not installed in ECS conda env (STATE.md Sprint 6 blocker). Skipped linting.

## Regression Baseline

```
tests/routers/test_sidebar_filtering.py       5 passed       (new)
tests/routers/test_approvals_api.py          10 passed       (re-verified post-refactor)
tests/auth + tests/routers + tests/middleware 64 passed, 6 skipped, 3 failed (pre-existing DEF-05-01-A)
```

Net delta vs. Plan 05-01 closure:
`+5 new integration tests`, `+0 regressions`, `+0 breakages`.

## User Setup Required

None — pure template refactor + tests. No env var changes, no new services, no new deps.

## Next Phase Readiness

**Ready for 05-03 / Wave 3 (forbidden UX + flash pipeline).** Preconditions satisfied:

- `can()` is the single source of truth in base.html — no role-string or `visible_to` gates remain.
- `/forbidden` page (Plan 05-03 target) can assume the sidebar is already fine-grained — it only needs to cover the "direct URL entry" edge case, not sidebar-driven 403s.
- The test fixture pattern (TEST_DOMAIN, purge, slowapi reset) is established and reusable for the next 3 plans in the wave.
- 5-tuple `nav_items` schema is stable — Plan 05-03..05-05 can extend it if they need a 6th slot for (e.g.) a submenu flag without breaking this plan's tests.

**No blockers** for Wave 3.

## Self-Check: PASSED

Verification:

- `templates/base.html` contains `can(current_user, permission)`: **FOUND** (line 62)
- `templates/base.html` contains `can(current_user, "aprobaciones:manage")`: **FOUND** (line 82)
- `templates/base.html` contains `visible_to`: **NOT FOUND** (0 matches)
- `templates/base.html` contains `current_user.role == 'propietario'`: **NOT FOUND** (0 matches)
- `tests/routers/test_sidebar_filtering.py` exists, 5 tests pass: **VERIFIED** (5 passed in 6.51s)
- Commit `faf68c2` in git log: **FOUND**
- Commit `06e1d25` in git log: **FOUND**
- Commit `e174f29` in git log: **FOUND**
- Base.html parses as valid Jinja2 (no syntax error): **VERIFIED**

---
*Phase: 05-ui-por-roles*
*Completed: 2026-04-20*
