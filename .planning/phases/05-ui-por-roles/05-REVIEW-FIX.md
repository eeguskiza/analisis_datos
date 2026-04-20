---
phase: 05-ui-por-roles
type: review-fix
fixed_at: 2026-04-20
review_path: .planning/phases/05-ui-por-roles/05-REVIEW.md
iteration: 1
findings_in_scope: 1
fixed: 1
deferred: 7
skipped: 0
status: complete
---

# Phase 5 (ui-por-roles) — Code Review Fix Report

**Fixed at:** 2026-04-20
**Source review:** `.planning/phases/05-ui-por-roles/05-REVIEW.md`
**Iteration:** 1

## Summary

- **HIGH findings in scope:** 1 (HI-01).
- **Fixed:** 1 — HI-01 (flash cookie race on chained-403).
- **Deferred to Phase 5.1 polish candidates:** 7 (MD-01..MD-04, LO-01..LO-03).
- **Skipped:** 0.
- **Status:** `complete` — sole HIGH finding resolved; MEDIUMs/LOWs tracked in
  `deferred-items.md` per objective brief.

## Findings table

| Finding | Severity | Status | Commit / Reason |
|---------|----------|--------|-----------------|
| HI-01 — `FlashMiddleware.delete_cookie` races with handler-set flash | HIGH | fixed | `f821e46` (fix + regression test) |
| MD-01 — `recursos.html` drawer edits not gated for read-only | MEDIUM | deferred | `deferred-items.md` §DEF-05-REVIEW-MD-01 |
| MD-02 — `base.html` separator relies on implicit invariant | MEDIUM | deferred | `deferred-items.md` §DEF-05-REVIEW-MD-02 |
| MD-03 — 403 handler parses `exc.detail` as string prefix | MEDIUM | deferred | `deferred-items.md` §DEF-05-REVIEW-MD-03 |
| MD-04 — `ajustes_conexion.html` stale line references | MEDIUM | deferred | `deferred-items.md` §DEF-05-REVIEW-MD-04 |
| LO-01 — `ajustes_conexion.html` `showToast` wrong arity | LOW | deferred | `deferred-items.md` §DEF-05-REVIEW-LO-01 |
| LO-02 — `PERMISSION_MAP` docstring stale plan IDs | LOW | deferred | `deferred-items.md` §DEF-05-REVIEW-LO-02 |
| LO-03 — `_PERMISSION_LABELS` unaccented convention undocumented | LOW | deferred | `deferred-items.md` §DEF-05-REVIEW-LO-03 |

## Fixed Issues

### HI-01 — Flash cookie race on chained-403

**Files modified:**

- `nexo/middleware/flash.py` — added guard before `delete_cookie`.
- `tests/middleware/test_flash.py` — added `_reemit` endpoint dummy +
  `test_flash_does_not_clobber_newly_set_cookie_chained_403` regression test.

**Commit:** `f821e46 — fix(05-review): HI-01 flash cookie race — skip delete
when response re-emits flash`

**Applied fix (Option 2 from review — inspect response headers):** before
calling `response.delete_cookie(_FLASH_COOKIE, path="/")`, iterate
`response.headers.getlist("set-cookie")` and skip the delete if any header
already starts with `nexo_flash=`. Robust against any downstream handler that
re-emits the cookie — not coupled to `http_exception_handler_403` specifically.
Option 1 (request.state sentinel) rejected because it would couple middleware
to one specific handler; Option 2 lifts the check to the response-cookie
layer where the actual collision happens.

**Verification:**

1. Syntax check (`python -c "import ast; ast.parse(...)"`) — OK for both files.
2. Pre-fix sanity: temporarily reverted `flash.py` and ran the new test —
   FAILED with "Esperaba 1 Set-Cookie para nexo_flash, obtuve 2" (two
   Set-Cookie headers: `nexo_flash=mensaje_nuevo; Max-Age=60` followed by
   `nexo_flash=""; Max-Age=0`). Confirms the regression test catches the
   exact HI-01 symptom.
3. Post-fix: all 5 flash tests pass (including the new regression).
4. Full suite (`tests/middleware` + `tests/routers`): **72 passed, 3 failed**.
   The 3 failures are pre-existing DEF-05-01-A (`test_recalibrate_*` —
   Phase 4 test-fixture leakage in `nexo.query_log`, already tracked).
   No new regressions introduced by HI-01 fix.

## Deferred Issues

All 7 MEDIUM/LOW findings were appended to
`.planning/phases/05-ui-por-roles/deferred-items.md` under a new section
"Phase 5.1 polish candidates" with:

- Severity (MEDIUM / LOW).
- File path + line references.
- Original issue description.
- Suggested fix sketch (verbatim or condensed from review).

See `deferred-items.md` §DEF-05-REVIEW-MD-01 through §DEF-05-REVIEW-LO-03.

## Final test counts (post-fix)

Command:

```bash
NEXO_PG_HOST=localhost NEXO_PG_PORT=5433 NEXO_PG_USER=oee \
  NEXO_PG_PASSWORD=oee NEXO_PG_DB=oee_planta \
  NEXO_PG_APP_USER= NEXO_PG_APP_PASSWORD= \
  NEXO_SECRET_KEY=testsecretkeytestsecretkeytestsecretkey \
  pytest tests/middleware tests/routers -q
```

Result: **72 passed, 3 failed** (3 pre-existing, tracked as DEF-05-01-A).

Flash-specific: **5 passed, 0 failed** (4 baseline + 1 new HI-01 regression).

---

_Fixed: 2026-04-20_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
