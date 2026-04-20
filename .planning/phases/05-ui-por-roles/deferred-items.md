# Phase 05 — Deferred Items

Issues found during plan execution that are out-of-scope for the current
plan but worth tracking.

---

## Plan 05-01 — discovered during regression sweep (Task 4)

### DEF-05-01-A: `tests/routers/test_thresholds_crud.py` recalibrate tests leak DB rows

**Status:** Pre-existing from Phase 4 closure (baseline `f726275`).

**Tests failing:**

- `tests/routers/test_thresholds_crud.py::test_recalibrate_preview_and_confirm_persists`
  — asserts `preview["sample_size"] == 15`, gets `18`.
- `tests/routers/test_thresholds_crud.py::test_recalibrate_filters_outliers_under_500ms`
  — same class of failure.
- `tests/routers/test_thresholds_crud.py::test_recalibrate_insufficient_data_returns_400`
  — observed during Plan 05-04 regression sweep (2026-04-20). Sample-size
  contamination likely causes recalibrate to find ≥10 samples where the
  test expected <10 (so it returns 200 instead of 400). Same root-cause
  class as the other two; reverting to `f0d984c~3` reproduces.

**Root cause (suspected):** `nexo.query_log` rows from prior test runs or
from co-running tests in the same suite are not purged before the
recalibrate preview runs, so `compute_factor` sees more samples than the
test seeded. The test fixture likely needs to truncate / delete query_log
rows for the specific `user_id + endpoint` tuple before the test body
runs.

**Reproduction:**

```bash
git checkout f726275 -- nexo/services/auth.py api/deps.py  # baseline
NEXO_PG_HOST=localhost NEXO_PG_PORT=5433 NEXO_PG_USER=oee \
NEXO_PG_PASSWORD=oee NEXO_PG_DB=oee_planta \
NEXO_SECRET_KEY=testsecretkeytestsecretkeytestsecretkey \
pytest tests/routers/test_thresholds_crud.py::test_recalibrate_preview_and_confirm_persists \
       tests/routers/test_thresholds_crud.py::test_recalibrate_filters_outliers_under_500ms
# → 2 failed (same error)
```

**Not in scope for Plan 05-01:** the 05-01 refactor (extract `can()`,
Jinja global) does not touch `nexo/query_log`, `compute_factor`, or the
`/api/thresholds/*/recalibrate` endpoint. Reverting auth.py/deps.py to
the baseline reproduces the failure.

**Recommended resolution plan:** open a small fix plan (Phase 4
follow-up or Phase 6 test-hardening) to make the recalibrate test
fixture purge `nexo.query_log` rows for the synthetic user **before**
seeding, rather than relying on teardown-only.

**Re-observed during Plan 05-05 (2026-04-20):** Regression sweep after
button gating + HTML GET hardening re-reproduced the exact same failure
on `test_recalibrate_insufficient_data_returns_400` (200 instead of
400). Scope-boundary: Plan 05-05 touches only `templates/*.html`,
`api/routers/pages.py` (route deps), and new test files — no overlap
with `nexo/query_log` or `/api/thresholds/*/recalibrate`. Confirmed
pre-existing; tracked here, not fixed by Plan 05-05.
