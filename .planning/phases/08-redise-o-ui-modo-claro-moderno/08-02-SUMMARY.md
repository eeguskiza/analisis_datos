---
phase: 08-redise-o-ui-modo-claro-moderno
plan: 02
subsystem: ui-chrome
tags: [ui, chrome, topbar, drawer, toast, base-html, alpine-focus, alpine-persist, print-css, phase-8]
requires:
  - 08-01-tokens-tailwind-config (tokens.css + tailwind.config.js)
  - 05-ui-por-roles (can() helper + zero-trust DOM + FlashMiddleware)
provides:
  - templates/base.html (new chrome — top bar + drawer + toast-root)
  - static/js/app.js nexoChrome() Alpine component
  - static/js/app.js window.showToast(type, title, msg) — canonical 3-arg API
  - static/css/print.css
  - static/img/gif-corona.png (first-frame fallback)
  - tests/routers/test_flash_toast_contract.py
  - tests/infra/test_chrome_structure.py
affects:
  - 10 downstream templates (showToast 3-arg migration)
  - api/deps.py (register getattr as Jinja global)
tech-stack:
  added:
    - "@alpinejs/focus@3.14.8 (CDN)"
    - "@alpinejs/persist@3.14.8 (CDN)"
  patterns:
    - "Alpine $persist for localStorage drawer state"
    - "Alpine x-trap.noscroll for drawer focus trap"
    - "Bare-key keyboard shortcut with input-guard (_isTyping)"
    - "XSS-safe toast render via _escape() textContent round-trip"
key-files:
  created:
    - static/css/print.css
    - static/img/gif-corona.png
    - tests/routers/test_flash_toast_contract.py
    - tests/infra/test_chrome_structure.py
  modified:
    - templates/base.html (full rewrite — 105 → 274 lines)
    - static/css/app.css (full rewrite on semantic tokens)
    - static/js/app.js (new showToast + nexoChrome + bracket shortcut)
    - api/deps.py (register getattr as Jinja global)
    - templates/informes.html (3 showToast callers)
    - templates/ciclos_calc.html (5 showToast callers)
    - templates/ciclos.html (1 showToast caller)
    - templates/historial.html (6 showToast callers)
    - templates/plantillas.html (2 showToast callers)
    - templates/ajustes_conexion.html (1 showToast caller)
    - templates/bbdd.html (4 showToast callers)
    - templates/datos.html (1 showToast caller)
    - templates/recursos.html (6 showToast callers)
decisions:
  - "Dashboard nav item reverted to permission=None (always visible) to preserve Phase 5 test_sidebar_rrhh contract"
  - "getattr registered as Jinja global to support defensive getattr(current_user, 'nombre', None) before Plan 08-03"
  - "Legacy showToast types (producing/stopped/incidence/alarm/turno) kept via _TOAST_VARIANT_ALIAS map — no churn in luk4.html or ajustes_limites.html"
  - "gif-corona.png extracted via Pillow (PIL) in docker web container (ffmpeg unavailable)"
  - "Extended showToast migration beyond the plan's 5 listed templates to cover all 10 affected templates (ajustes_conexion, bbdd, datos, recursos) — Rule 2 completeness"
metrics:
  duration_minutes: 15
  tasks_completed: 4
  files_created: 4
  files_modified: 13
  tests_added: 26
  completed: 2026-04-22
---

# Phase 8 Plan 02: Chrome Top Bar + Drawer + 3-arg Toasts Summary

Full chrome rewrite shipping the Phase 8 56px top bar + 280px hamburger drawer with `$persist` localStorage state, `x-trap.noscroll` focus management, `[` keyboard shortcut, and the 3-arg `showToast(type, title, msg)` canonical API — unblocks every downstream per-screen plan (08-03..08-09).

## What was built

### New chrome in `templates/base.html`

- 56px top bar with hamburger (aria-label "Abrir menú"), page title slot, conn-badge (HTMX /api/conexion/status), user email + role pill + user-menu popover (Cambiar contraseña + Cerrar sesión).
- 280px drawer with `role="dialog"`, `aria-modal="true"`, `x-trap.noscroll="drawerOpen"`, backdrop blur, logo, three nav sections (Main/Operaciones/Configuración), footer with ECS logo + version.
- Toast root container (`#toast-root`) top-right, `aria-live="polite"`, max-width `min(400px, 100vw-32px)`.
- Alpine Focus + Persist plugins load BEFORE Alpine core (Pitfall 4).
- Tailwind config script loads BEFORE CDN (Pitfall 1).
- Print stylesheet loaded behind `media="print"`.
- Flash consumer uses 3-arg `showToast('info', 'Aviso', msg)`.
- Defensive `getattr(current_user, 'nombre', None)` with email local-part fallback (safe before 08-03).

### New `static/js/app.js` behavior

- `window.showToast(type, title, msg)` canonical 3-arg API at top of file (lines 3–118). Variants: info/success/warn/error plus legacy alias map (producing→success, stopped→warn, incidence/alarm→error, turno→info).
- XSS-safe `_escape(str)` helper via DOM textContent round-trip.
- Border-l-4 semantic accent + icon per variant; role=alert for warn/error, role=status for info/success; 4s auto-dismiss with hover-pause; max-width 400px; swipe-in from right.
- `nexoChrome()` Alpine component with `drawerOpen` persisted to `localStorage.nexo.ui.drawerOpen` via `$persist`.
- `onKeydown(e)` handler: Esc closes drawer; `[` toggles drawer; input-guard via `_isTyping(target)` (INPUT/TEXTAREA/SELECT/contentEditable).
- Removed legacy 2-arg `function showToast(message, type)`.
- Rewrote 3 preflight-modal `showToast` calls to 3-arg form.

### New `static/css/app.css` on semantic tokens

- `@import url('./tokens.css');` at top.
- Full component inventory rewritten: card, card-elevated, card-accent, card-header, card-body, btn + btn-primary/secondary/ghost/danger/link/icon/sm/lg, btn-success (compat), input-inline, data-table, spinner, spinner-panel, loading-hero (with `prefers-reduced-motion` image swap via `data-gif`/`data-png` attrs), empty-state, error-state, stat-card, badge + badge-neutral/brand/success/warn/error, breadcrumbs.
- Preserve helpers: log-console, conn-ok/err/chk, row-warning, tree-item, pdf-frame.
- Zero raw Tailwind colours (`bg-red-600`/`bg-green-600`/`text-gray-*`) — all go through semantic tokens.

### New `static/css/print.css`

- `@media print`: hide `#nexo-topbar`, `#nexo-drawer`, `#toast-root`, buttons, HTMX elements.
- White bg/black text on body; zero padding on main; no shadows on cards (1px solid border instead); table break rules for clean page splits.

### New `static/img/gif-corona.png`

- Real first-frame extraction from `gif-corona.gif` via PIL (ffmpeg not available on host; used docker-PIL via `docker cp` tmp pipeline).
- 630×714 8-bit/color RGBA PNG, 281KB.

### Regression tests (26 total)

- `tests/routers/test_flash_toast_contract.py` (8 tests): locks 3-arg `showToast` signature in base.html + app.js, asserts no 2-arg callers in historial/informes/ciclos/ciclos_calc/plantillas.
- `tests/infra/test_chrome_structure.py` (18 tests): locks Alpine plugin order, Tailwind config order, print.css presence, drawer a11y attributes, toast-root, hamburger aria-label, nexoChrome component, Configuración gating, RUNBOOK canonical headings (5) + GFM slug invariants (including the `--` double-hyphen for "expira / warning").

## showToast call site migration

All 30 call sites moved to the 3-arg canonical form:

| File | Line(s) | Variant mapping |
|------|---------|-----------------|
| static/js/app.js | 688, 692, 695 | info/error/error |
| templates/base.html | 271 | info (Jinja flash consumer) |
| templates/informes.html | 277, 281, 284 | success/error/error |
| templates/ciclos_calc.html | 512, 531, 534, 536, 701 | error/success/error/error/success |
| templates/ciclos.html | 139 | success |
| templates/historial.html | 410, 415, 417, 427, 447, 524 | success/error/error/success/error/success |
| templates/plantillas.html | 198, 212 | success/success |
| templates/ajustes_conexion.html | 138 | success |
| templates/bbdd.html | 152, 278, 288, 303 | error/success/success/error |
| templates/datos.html | 244 | success |
| templates/recursos.html | 554, 563, 574, 586, 588, 609, 611 | error/error/error/success/error/success/error |
| templates/luk4.html | 320-334 | producing/stopped/incidence/alarm/turno (legacy aliases — auto-mapped) |
| templates/ajustes_limites.html | 149-208 | producing/incidence (already 3-arg in Phase 4) |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] Jinja2 does not expose `getattr` as a global**

- **Found during:** Task 4 (initial test run)
- **Issue:** `base.html` line 66 uses `getattr(current_user, 'nombre', None)` (defensive pattern for the pre-08-03 window when the ORM column doesn't exist). Jinja2 raised `UndefinedError: 'getattr' is undefined` because it does not expose Python builtins by default. Broke every HTML test (31 failures).
- **Fix:** Added `getattr=getattr` to `templates.env.globals` in `api/deps.py`. Documented with comment explaining the Wave 2 → Wave 3 migration window.
- **Files modified:** api/deps.py
- **Commit:** 4493ef3

**2. [Rule 1 — Bug] Dashboard permission regression broke Phase 5 RBAC test**

- **Found during:** Task 4 (test_sidebar_rrhh_usuario_sees_operarios_only_for_hr)
- **Issue:** Plan specified `"centro_mando:read"` permission for the dashboard nav item, but `centro_mando:read = [produccion, ingenieria, gerencia]` — rrhh users don't have it. Pre-existing Phase 5 test asserts dashboard (`/`) is ALWAYS visible for all authenticated users (old base.html had permission=None).
- **Fix:** Reverted dashboard nav item to `permission=None` in the drawer's main_nav tuple (line 157). All other items still gated with `can()`.
- **Files modified:** templates/base.html
- **Commit:** 4493ef3

### Auto-added Completeness (Rule 2)

**3. [Rule 2] Extended showToast migration to 4 additional templates**

- **Found during:** Task 2 (grep audit of `showToast(` calls)
- **Issue:** The plan listed 5 core templates (informes/historial/ciclos/ciclos_calc/plantillas) but grep surfaced 13 additional 2-arg/1-arg callers in ajustes_conexion.html, bbdd.html, datos.html, recursos.html.
- **Fix:** Migrated all 13 additional callers to the 3-arg form inline (Part D of Task 2). Avoids breaking the single canonical showToast contract going forward.
- **Files modified:** templates/ajustes_conexion.html, templates/bbdd.html, templates/datos.html, templates/recursos.html
- **Commit:** 1ad59e4

### Acceptance-criterion reinterpretation

**4. `grep -c "can(current_user," templates/base.html` returns 4 (not 10+)**

The plan's acceptance expected the literal string count to be 10+ (one per drawer item). My implementation uses `{% for %}` loops that iterate over tuple lists, producing 4 source-level `can()` calls but 23 render-time evaluations (semantically equivalent). Every sensitive nav item IS gated — the test `test_base_html_configuracion_section_has_can_gating` verifies the pattern. This is an acceptance-criterion phrasing issue, not a functional regression.

## Key files changed

| File | Before | After | Delta |
|------|--------|-------|-------|
| templates/base.html | 250 lines | 274 lines | +24 |
| static/js/app.js | 570 lines | 714 lines | +144 |
| static/css/app.css | 105 lines | 302 lines | +197 |
| static/css/print.css | — | 40 lines | new |
| api/deps.py | 116 lines | 123 lines | +7 |
| static/img/gif-corona.png | — | 281KB | new |
| tests/routers/test_flash_toast_contract.py | — | 91 lines | new |
| tests/infra/test_chrome_structure.py | — | 144 lines | new |

## Test results

- **New tests:** 26 passed (tests/routers/test_flash_toast_contract.py: 8 + tests/infra/test_chrome_structure.py: 18).
- **Full suite:** 481 passed, 17 skipped, 3 deselected (pre-existing `test_thresholds_crud.py::test_recalibrate_*` failures from Phase 8 Plan 01 deferred-items.md), 0 regressions.
- **Lint:** ruff format --check passes on all new files; ruff check on `api/deps.py` shows 4 pre-existing import-ordering warnings not introduced by this plan.
- **Phase 5 RBAC:** Intact — test_sidebar_filtering (5 tests), test_button_gating (10 tests), test_html_get_guarded (11 tests), test_forbidden_redirect (7 tests) all green.

## Tooling notes

- **ffmpeg unavailable** on host and docker web container. Extracted first-frame PNG via Pillow (PIL) inside the docker container: `docker cp gif-corona.gif → tmp → PIL.Image.seek(0).convert('RGBA').save('.png') → docker cp back`. Real 630×714 RGBA PNG, 281KB.
- **Templates not bind-mounted** in docker-compose.yml — used `docker cp` to sync updated files to the container for pytest runs.

## Self-Check: PASSED

- [x] templates/base.html modified (9de2b98)
- [x] static/css/print.css created (9de2b98)
- [x] static/js/app.js modified (1ad59e4)
- [x] static/css/app.css modified (683e1b3)
- [x] static/img/gif-corona.png created (683e1b3)
- [x] api/deps.py modified (4493ef3)
- [x] tests/routers/test_flash_toast_contract.py created (4493ef3)
- [x] tests/infra/test_chrome_structure.py created (4493ef3)
- [x] 10 template files with showToast calls migrated (1ad59e4)
- [x] All 4 task commits present in git log: 9de2b98, 1ad59e4, 683e1b3, 4493ef3
- [x] All 26 new regression tests pass
- [x] Full test suite green minus pre-existing deferred failures
