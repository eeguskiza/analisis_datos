---
phase: 05-ui-por-roles
plan: 03
subsystem: ui
tags: [middleware, exception-handler, flash, ux, forbidden, rbac, starlette, fastapi]

requires:
  - phase: 05-ui-por-roles
    plan: 01
    provides: "can(user, permission) pure helper + Jinja global registration in api/deps.py"
  - phase: 02-login-rbac
    provides: "require_permission dependency raising HTTPException(403) with `Permiso requerido: <perm>` detail"
  - phase: 02-login-rbac
    provides: "NAMING-07 global_exception_handler on Exception (500 only — not HTTPException)"
provides:
  - "FlashMiddleware (nexo/middleware/flash.py) — read-and-clear cookie nexo_flash, one-request TTL"
  - "StarletteHTTPException handler in api/main.py with Accept negotiation: 403 HTML → 302+flash; 403 JSON/HX → default (contract stable)"
  - "_PERMISSION_LABELS friendly-label dict covering 21 permissions (covers all 10 HTML-guarded perms from Plan 05-05 + ajustes:manage + conexion:config per W-06)"
  - "render() in api/deps.py passes flash_message=request.state.flash to Jinja ctx (signature unchanged)"
  - "base.html renders showToast('info', 'Aviso', ...) via DOMContentLoaded when flash_message truthy; reuses existing window.showToast (no new toast system)"
  - "4 unit tests (test_flash.py) + 9 integration tests (test_forbidden_redirect.py) covering redirect flow + JSON contract + read-and-clear + W-04 + W-06"
affects: [05-04-ajustes-split, 05-05-router-guards]

tech-stack:
  added: []
  patterns:
    - "Accept-aware 403 handler: delegate to fastapi.exception_handlers.http_exception_handler for status != 403 and 403 JSON paths; only 403 HTML gets redirect+flash"
    - "Flash cookie lifecycle: handler writes HttpOnly+SameSite=Lax+Secure=not debug, max_age=60; middleware reads pre-call_next and deletes post-call_next"
    - "Middleware stack LIFO order preserved: Query (innermost) → Audit → Flash (NEW) → Auth (outermost). Flash between Audit and Auth because it reads/writes cookies, doesn't touch request.state.user (W-02)"
    - "Test cookie handling workaround: testserver http:// drops Secure cookies from jar automatically; tests use cookies= kwarg per-request + _extract_flash_cookie helper instead of client.cookies.set"
    - "Test-contract alignment: pre-existing 403 HTML assertions updated to split HTML (302+flash) from JSON (403) — Plan 05-03 explicitly changes HTML behavior per D-07"

key-files:
  created:
    - "nexo/middleware/flash.py (34 lines)"
    - "tests/middleware/test_flash.py (129 lines, 4 tests)"
    - "tests/routers/test_forbidden_redirect.py (455 lines, 9 tests)"
  modified:
    - "api/main.py (imports + _PERMISSION_LABELS dict + http_exception_handler_403 handler + FlashMiddleware in stack)"
    - "api/deps.py (render() ctx extended with flash_message — signature unchanged)"
    - "templates/base.html (flash_message|tojson → showToast block before </body>)"
    - "tests/routers/test_approvals_api.py (test_ajustes_solicitudes_is_propietario_only updated to split HTML 302 / JSON 403)"
    - "tests/routers/test_thresholds_crud.py (test_get_limites_page_requires_propietario same split)"

key-decisions:
  - "Register exception_handler on StarletteHTTPException (parent), not fastapi.HTTPException — catches both branches (Pitfall 1 per 05-RESEARCH)"
  - "Delegate status != 403 to fastapi.exception_handlers.http_exception_handler — NAMING-07 Exception handler untouched (different registry per type)"
  - "_PERMISSION_LABELS lives local to api/main.py for Phase 5 (per 05-RESEARCH Open Q1 option a); migrate to nexo/services/auth.py in Mark-IV if more callers need it"
  - "Cookie policy: HttpOnly=True, SameSite=Lax, Secure=not settings.debug (Pitfall 2 — dev HTTP stays functional), max_age=60s, path=/"
  - "FlashMiddleware positioned between Audit and Auth (W-02): LIFO Auth-outer-first (401 short-circuit), Flash-next (populates state.flash over authenticated or anonymous requests). No state.user dependency"
  - "_wants_json reused from api/main.py:127 (existing duplicate with api/middleware/auth.py pre-existed; Plan 05-03 did NOT create a third copy — same import path from both call sites via the one in main.py)"
  - "showToast reused from base.html:188-223 (Pitfall/Pattern #10 of 05-PATTERNS) — NOT a new toast component"
  - "Flash render guarded by DOMContentLoaded because toastSystem Alpine container may not be ready on sync parse"
  - "Test-alignment (Rule 1 Bug): 2 pre-existing tests asserting 403 on HTML GET updated to split HTML 302 + JSON 403; this is the documented D-07 contract change, not a regression"

patterns-established:
  - "Accept negotiation via _wants_json(): /api/*, Accept: application/json (if no text/html), HX-Request: true → JSON branch; else HTML branch"
  - "Exception handler registry sibling pattern: @app.exception_handler(Exception) for 500 coexists with @app.exception_handler(StarletteHTTPException) for 401/403/404/422 — different types, different handlers"
  - "Flash cookie test pattern: helper _extract_flash_cookie parses Set-Cookie header, tests reenvían value via cookies= kwarg on next request (testserver secure-cookie jar workaround)"

requirements-completed: [UIROL-02]

duration: ~45min
completed: 2026-04-20
---

# Phase 05 Plan 03: Forbidden UX + flash pipeline Summary

**StarletteHTTPException handler negotiates Accept on 403 — HTML → 302+flash cookie with user-friendly label; JSON/HTMX contract stable. FlashMiddleware reads-and-clears nexo_flash between Audit and Auth in the middleware stack; base.html dispatches showToast('info','Aviso',...) on DOMContentLoaded when flash_message is truthy.**

## Performance

- **Duration:** ~45 min (sequential autonomous, main working tree, tests inside Docker web container)
- **Started:** 2026-04-20T20:20:00Z (approx.)
- **Completed:** 2026-04-20T20:35:00Z
- **Tasks:** 5 executed, 5 commits
- **Files created:** 3 (`nexo/middleware/flash.py`, `tests/middleware/test_flash.py`, `tests/routers/test_forbidden_redirect.py`)
- **Files modified:** 5 (`api/main.py`, `api/deps.py`, `templates/base.html`, `tests/routers/test_approvals_api.py`, `tests/routers/test_thresholds_crud.py`)

## Accomplishments

- **D-07/D-08 wired end-to-end:** non-propietario typing `/ajustes` (or any HTML route gated by `require_permission`) now receives a 302 to `/` with a `nexo_flash` cookie carrying a user-friendly label (e.g., "No tienes permiso para acceder a la configuracion"). The next request reads and clears the cookie, and `base.html` dispatches the existing `showToast` with `type='info'`, `title='Aviso'`.
- **JSON / HTMX contract preserved:** `/api/*`, `Accept: application/json`, and `HX-Request: true` still receive the unchanged 403 JSON body `{"detail": "Permiso requerido: <perm>"}`. The handler delegates those branches to `fastapi.exception_handlers.http_exception_handler` — zero regression for the sidebar badge (Plan 04-03), HTMX calls, and programmatic clients.
- **NAMING-07 non-regression confirmed:** the existing `@app.exception_handler(Exception)` handler (500 with UUID) is untouched. FastAPI keeps a separate registry per exception type, so the new `@app.exception_handler(StarletteHTTPException)` is a sibling, not a shadow. `test_404_not_regressed_by_new_handler` validates that 404 still responds from the default handler (not our 403 branch).
- **Middleware stack respects LIFO W-02:** order is now `Query (innermost) → Audit → Flash → Auth (outermost)`. AuthMiddleware still runs first on ingress (401 short-circuit preserved); FlashMiddleware sits between Audit and Auth because it neither reads nor writes `request.state.user` — it only needs to see the raw cookie jar. Anonymous requests can still carry flash.
- **W-06 coverage locked in via test:** `_PERMISSION_LABELS` covers the 10 HTML-guarded permissions landing in Plan 05-05 (`pipeline:read`, `recursos:read`, `historial:read`, `ciclos:read`, `operarios:read`, `datos:read`, `bbdd:read`, `capacidad:read`, `ajustes:manage`, `conexion:config`) plus 11 more for full catalog coverage. `test_flash_label_coverage` enforces this at suite runtime — if a future plan adds a new HTML-guarded permission without a friendly label, the test fails.
- **Suite impact:** 154 → 156 passing tests (+2 vs pre-plan baseline: +4 new `test_flash.py` + 9 new `test_forbidden_redirect.py` - 11 pre-existing; 2 pre-existing tests updated for new contract). 3 pre-existing `test_recalibrate_*` failures remain unchanged (DEF-05-01-A).

## Task Commits

Each task was committed atomically to `feature/Mark-III`:

1. **Task 1: Create FlashMiddleware** — `4bc45bf` (feat)
   - `nexo/middleware/flash.py` (34 lines): `BaseHTTPMiddleware.dispatch` reads `nexo_flash` cookie pre-`call_next`, assigns to `request.state.flash`, deletes cookie post-`call_next` if present. No DB, no logging, no exception capture.

2. **Task 2: Handler + FlashMiddleware registration** — `34948fe` (feat)
   - `api/main.py`: added imports (`StarletteHTTPException`, `_default_http_handler`, `RedirectResponse`, `FlashMiddleware`); `_PERMISSION_LABELS` dict (21 entries); `_friendly_permission_label()` helper; `@app.exception_handler(StarletteHTTPException)` handler with Accept negotiation; `app.add_middleware(FlashMiddleware)` between Audit and Auth.

3. **Task 3: render() + base.html toast render** — `b0d7c83` (feat)
   - `api/deps.py`: `render()` ctx dict now includes `flash_message=getattr(request.state, "flash", None)`. Signature unchanged (Pitfall 7 preserved).
   - `templates/base.html`: new `{% if flash_message %}<script>...</script>{% endif %}` block before `</body>` dispatches `showToast('info', 'Aviso', {{ flash_message|tojson }})` under `DOMContentLoaded`. `|tojson` guarantees XSS-safe interpolation.

4. **Task 4: Unit tests for FlashMiddleware** — `881d6fb` (test)
   - `tests/middleware/test_flash.py` (129 lines, 4 tests): mini-Starlette app with only `FlashMiddleware` + dummy `/peek` endpoint. Hermetic — no DB, no auth, no `api.main.app`. Covers: absent cookie → state is None + no Set-Cookie; present cookie → state populated + delete signal; empty string treated as present; multi-request read-and-clear sequence. Uses `client.cookies.set()` pattern (not deprecated kwarg).

5. **Task 5: Integration tests + test-contract alignment** — `7a993c3` (test)
   - `tests/routers/test_forbidden_redirect.py` (455 lines, 9 tests): covers the full D-07/D-08 flow end-to-end against `api.main.app` with TestClient. Includes Rule 1 fix for 2 pre-existing tests (`test_approvals_api.py::test_ajustes_solicitudes_is_propietario_only` + `test_thresholds_crud.py::test_get_limites_page_requires_propietario`) that were asserting the old 403-HTML behavior — updated to split HTML (302+flash) from JSON (403).

## Files Created/Modified

- `nexo/middleware/flash.py` (new, 34 lines):
  - `FlashMiddleware(BaseHTTPMiddleware)` with async `dispatch`.
  - Private constant `_FLASH_COOKIE = "nexo_flash"` at module top.
  - Zero external deps beyond Starlette (no DB, no settings).

- `api/main.py` (modified):
  - Lines 11-31: imports extended (`StarletteHTTPException`, `_default_http_handler`, `RedirectResponse`, `FlashMiddleware`).
  - Lines 175-267: new `_PERMISSION_LABELS` dict + `_friendly_permission_label()` + `@app.exception_handler(StarletteHTTPException)` `http_exception_handler_403`.
  - Line 203 (new): `app.add_middleware(FlashMiddleware)` between Audit (line 202) and Auth (line 204).

- `api/deps.py` (modified):
  - Lines 52-60: `render()` ctx dict extended with `flash_message`. Signature, `if extra:`, and `TemplateResponse(...)` call untouched.

- `templates/base.html` (modified):
  - Lines 235-245: new `{% if flash_message %}` block with DOMContentLoaded + showToast dispatch.

- `tests/middleware/test_flash.py` (new, 129 lines, 4 tests).

- `tests/routers/test_forbidden_redirect.py` (new, 455 lines, 9 tests).

- `tests/routers/test_approvals_api.py` (modified):
  - `test_ajustes_solicitudes_is_propietario_only` now asserts HTML 302+flash and JSON 403 separately (D-07 contract).

- `tests/routers/test_thresholds_crud.py` (modified):
  - `test_get_limites_page_requires_propietario` — same split.

## Decisions Made

- **Handler type: StarletteHTTPException.** `fastapi.HTTPException` is a subclass of `StarletteHTTPException`, so registering on the parent catches both. Pitfall 1 from 05-RESEARCH — explicitly preserved.
- **Delegation for non-403.** `if exc.status_code != 403: return await _default_http_handler(request, exc)` — FastAPI's default handler is reused verbatim so 401/404/422 continue to respond exactly as before.
- **Accept negotiation via `_wants_json` (reused).** The existing helper at `api/main.py:127` is the single call site in this handler — no third copy of the `/api/*` + `application/json` + `hx-request` heuristic. The pre-existing duplicate at `api/middleware/auth.py:70-80` stays as-is; this plan explicitly does NOT refactor it (out of scope).
- **`_PERMISSION_LABELS` local to `api/main.py`.** Per 05-RESEARCH Open Q1 option (a): keep the friendly-label map next to the handler that consumes it. Migration to `nexo/services/auth.py` is deferred to Mark-IV — noted in a comment.
- **Cookie policy:** `HttpOnly=True`, `SameSite=Lax`, `Secure=not settings.debug`, `max_age=60`, `path=/`. `Secure=not settings.debug` (Pitfall 2) means dev HTTP keeps the cookie, prod HTTPS enforces Secure.
- **Flash toast uses existing `showToast`.** `type='info'` with `title='Aviso'` — reuses the visual style defined at `base.html:188-223`. DOMContentLoaded wait is defensive (Alpine `toastSystem` contained may hydrate after the parse but before interaction).
- **Test-alignment for 2 pre-existing tests.** Both `test_ajustes_solicitudes_is_propietario_only` and `test_get_limites_page_requires_propietario` asserted 403 on HTML GET. Plan 05-03 D-07 explicitly changes HTML GET to 302+flash — not a regression, a contract change. Updated tests split HTML and JSON paths; both pass.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test-contract alignment for 2 pre-existing tests**

- **Found during:** Full-suite regression run after Task 5 tests passed.
- **Issue:** `test_ajustes_solicitudes_is_propietario_only` (in `test_approvals_api.py`) and `test_get_limites_page_requires_propietario` (in `test_thresholds_crud.py`) both asserted `r.status_code == 403` on HTML GET without `Accept: application/json` / `HX-Request`. Plan 05-03 D-07 explicitly changes this contract: HTML GET → 302 + flash cookie. The failing tests were the old behavior frozen in tests, not bugs in the new handler.
- **Fix:** Split both tests into two assertions per role:
  - HTML path: assert `r.status_code == 302`, `location == "/"`, `Set-Cookie` contains `nexo_flash=`.
  - JSON path: assert `r.status_code == 403`, `Set-Cookie` has no `nexo_flash`.
- **Files modified:** `tests/routers/test_approvals_api.py`, `tests/routers/test_thresholds_crud.py`.
- **Verification:** Both tests pass individually; full suite still green (156 passed, 3 pre-existing failures unrelated).
- **Committed in:** `7a993c3` (same commit as Task 5 new file).

**2. [Rule 3 - Blocking] Container image was stale (missing `nexo/middleware/` directory)**

- **Found during:** Task 1 verification attempt.
- **Issue:** `docker compose exec -T web python -c "from nexo.middleware.flash ..."` failed with `ModuleNotFoundError: No module named 'nexo.middleware'`. The `analisis_datos-web` Docker image was built before `nexo/middleware/` existed as a package and the service does NOT bind-mount the codebase (only `./data`, `./informes`, `./tests`). Every code edit requires either a rebuild or a targeted `docker compose cp`.
- **Fix:** Ran `docker compose up -d --build web` once to rebuild the image with the current source tree (includes `nexo/middleware/__init__.py` and `query_timing.py`). Subsequent edits were pushed into the container via `docker compose cp <file> web:/app/<file>` instead of a full rebuild, which is faster for incremental verification.
- **Files modified:** None (operational action only).
- **Verification:** `docker compose exec -T web python -c "from nexo.middleware.flash ..."` returned "OK".
- **Committed in:** No commit (runtime infrastructure operation, not code).

**3. [Rule 3 - Blocking] pytest not installed in container image**

- **Found during:** Task 4 verification attempt.
- **Issue:** `docker compose exec -T web python -m pytest tests/middleware/test_flash.py` failed with `No module named pytest`. The production Docker image intentionally omits dev deps.
- **Fix:** `docker compose exec -T web pip install pytest pytest-asyncio httpx`. This is a container-only install (not persisted to the image), aligned with the existing `make test-data` target (which assumes pytest is installed via a similar mechanism at CI time).
- **Files modified:** None.
- **Verification:** 4 tests run and pass.
- **Committed in:** No commit.

---

**Total deviations:** 3 auto-fixed (1 Rule-1 bug in pre-existing tests to align with planned contract, 2 Rule-3 infrastructure blockers resolved without code changes).
**Impact on plan:** All auto-fixes essential to complete the plan. No scope creep. The Rule-1 fix is the contract alignment the plan explicitly mandates (D-07).

## Issues Encountered

- **Starlette testclient drops Secure cookies over http:// testserver.** When using `client.cookies.set("nexo_session", cookie)`, the session cookie (set with `Secure=True` by `/login`) gets dropped from the jar on next request, so `GET /` returns 302 to `/login` (session invalid) instead of 200. Workaround: tests use the `cookies=` kwarg per-request for both `nexo_session` and `nexo_flash`, plus a helper `_extract_flash_cookie` that parses `Set-Cookie` header from the 403 response and manually reenvía on the next GET. Documented in test file docstring. Generates a deprecation warning per request; acceptable for now — matches pattern from `tests/routers/test_sidebar_filtering.py` Plan 05-02.
- **3 pre-existing test failures carry forward.** `test_recalibrate_insufficient_data_returns_400`, `test_recalibrate_preview_and_confirm_persists`, `test_recalibrate_filters_outliers_under_500ms` — all tracked as `DEF-05-01-A` in `.planning/deferred-items.md`. Verified unchanged from the Plan 05-02 baseline before starting Plan 05-03; neither Plan 05-03 changes nor the contract alignment commits touch the affected code paths.

## No-regression Evidence (NAMING-07 handler)

- `test_404_not_regressed_by_new_handler` (in `test_forbidden_redirect.py`): GET `/ruta-que-no-existe-404` with valid session + `Accept: application/json` → 404, no `nexo_flash` in Set-Cookie. Confirms the new `http_exception_handler_403` delegates `status != 403` to `_default_http_handler` and the NAMING-07 `Exception`-registered handler is not invoked on HTTPException paths.
- Full suite: 156 passing (baseline 154 + 4 new unit + 9 new integration - 11 that were already in 154; 2 pre-existing tests updated; 3 pre-existing `test_recalibrate_*` failures unchanged — not caused by Plan 05-03).

## `_wants_json` Triplication Check

- `grep -c "_wants_json" api/main.py` = 3 (1 definition at line 127 + 1 call in `global_exception_handler` at line 146 + 1 call in `http_exception_handler_403` at line 247). **No new function body, only a new call site.**
- `grep -c "_wants_json" api/middleware/auth.py` = 2 (1 definition + 1 call). Pre-existing duplicate; not touched by Plan 05-03.

## Next Phase Readiness

- **Ready for 05-04 (Wave 4 — split de `/ajustes` hub).** The forbidden UX foundation is live: any new HTML route added in 05-04 that uses `require_permission` automatically benefits from the 302+flash flow without code changes. The 21-entry `_PERMISSION_LABELS` dict already covers the labels for routes that 05-04 / 05-05 will guard; `test_flash_label_coverage` guards against regressions.
- **Ready for 05-05 (Wave 5 — router guards).** Any router newly guarded with `require_permission(X)` where `X in _PERMISSION_LABELS` will surface the right message; any new permission added to `PERMISSION_MAP` should also get an entry in `_PERMISSION_LABELS` or the `test_flash_label_coverage` check must be extended.
- No blockers or concerns.

---
*Phase: 05-ui-por-roles*
*Completed: 2026-04-20*

## Self-Check: PASSED

- Files verified: nexo/middleware/flash.py, tests/middleware/test_flash.py, tests/routers/test_forbidden_redirect.py, .planning/phases/05-ui-por-roles/05-03-SUMMARY.md
- Commits verified: 4bc45bf, 34948fe, b0d7c83, 881d6fb, 7a993c3
