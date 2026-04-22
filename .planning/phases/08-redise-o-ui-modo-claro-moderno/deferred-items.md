# Phase 8 — Deferred Items

Out-of-scope discoveries logged during plan execution. Do NOT fix inside Phase 8
plans that don't own the file; carry forward to the appropriate plan/phase.

## 2026-04-22 — Plan 08-01 execution

- **tests/routers/test_thresholds_crud.py::test_recalibrate_insufficient_data_returns_400**
  — FAILS on pre-plan HEAD (`3a9d836`) and on post-plan HEAD. Verified isolated from
  Plan 08-01 scope (no CSS/Tailwind/BRANDING.md touch). Endpoint
  `/api/thresholds/pipeline%2Frun/recalibrate?confirm=false` returns 200 where the
  test asserts 400 (insufficient data). Pre-existing bug in the thresholds CRUD
  path or test-data setup. Owner: whichever Phase 8 plan (or a Phase 4 backport
  fix) next touches thresholds / recalibrate; or ignore and open a separate
  bugfix commit.
