---
phase: 08-redise-o-ui-modo-claro-moderno
plan: 04
subsystem: ui
tags: [landing, bienvenida, saludo, reloj, phase-8, ui, redirect, jinja-filter]
dependency_graph:
  requires:
    - plan: 08-02
      provides: "base.html chrome (topbar + drawer + toast) + Alpine core + nexoChrome()"
    - plan: 08-03
      provides: "nexo.users.nombre column + ORM wiring (happy path for display_name)"
  provides:
    - "GET /bienvenida landing route (authenticated, no RBAC gate)"
    - "Jinja filter hora_saludo(datetime) with Europe/Madrid tz + 3-band greeting"
    - "Alpine component bienvenidaPage() with setInterval tick + destroy cleanup"
    - "Post-login redirect target switched from / to /bienvenida"
  affects:
    - "Any test that asserted login redirects to /  — now must assert /bienvenida (none found in tree)"
    - "UX flow: users now land on a transitional screen before Centro de Mando"
tech_stack:
  added:
    - "zoneinfo.ZoneInfo (stdlib, Python 3.9+) — Europe/Madrid timezone resolution"
  patterns:
    - "Server-rendered time-banded greeting via Jinja filter (deterministic, no client clock trust)"
    - "Client-side reloj via Alpine x-data + setInterval + destroy() cleanup (Pitfall 6)"
    - "Defensive getattr fallback for display_name (guards against NULL nombre rows)"
    - "Mobile role swap via hidden/md:hidden twin headings (Display 32 → Heading 20)"
key_files:
  created:
    - templates/bienvenida.html
    - tests/routers/test_bienvenida.py
  modified:
    - api/deps.py
    - api/routers/auth.py
    - api/routers/pages.py
    - static/js/app.js
    - .planning/phases/08-redise-o-ui-modo-claro-moderno/deferred-items.md
decisions:
  - "No require_permission on /bienvenida — AuthMiddleware global is sufficient (any authenticated user lands here)"
  - "hora_saludo registered as Jinja filter (not global) so templates use pipe syntax {{ None|hora_saludo }}"
  - "Europe/Madrid is server-authoritative for the greeting band — client clock not consulted for saludo"
  - "Mobile role swap implemented with twin <h2> elements + hidden/md:hidden rather than @media CSS override (matches existing Tailwind patterns in base.html)"
  - "must_change_password branch in auth.py preserved — /cambiar-password continues to take priority over /bienvenida"
metrics:
  duration_minutes: 10
  tasks_completed: 3
  files_touched: 6
  tests_added: 17
  completed_at: "2026-04-22T18:56:00Z"
---

# Phase 8 Plan 08-04: Bienvenida Landing Summary

**One-liner:** New `/bienvenida` landing screen post-login with server-rendered
saludo por franja (Europe/Madrid), client-side reloj via Alpine tick, and CTA
to Centro de Mando — replaces `/` as the default post-login redirect target.

## What Shipped

### New route + template

- **GET `/bienvenida`** — registered in `api/routers/pages.py` without
  `require_permission(...)` (AuthMiddleware global is the single gate). Injects
  `now_madrid: datetime` so the Jinja template can render day + date in
  Spanish without relying on container locale settings.

- **`templates/bienvenida.html`** — extends `base.html` (Plan 08-02 chrome).
  Centered column (`max-w-[640px]`) with 4 stacked elements:
  1. Saludo — `{{ None|hora_saludo }}, {{ display_name }}` (Display 32 desktop
     / Heading 20 mobile via twin `<h2>` + `hidden`/`md:hidden`).
  2. Día + fecha — `Es {dia}, {n} de {mes} de {YYYY}` from 7-day + 12-month
     literal arrays indexed by `now_madrid.weekday()` / `now_madrid.month`.
  3. Reloj — `<div x-text="clock">` bound to Alpine `bienvenidaPage()`;
     `role="timer"` + `font-variant-numeric: tabular-nums` (no jitter).
  4. CTA — `<a href="/" class="btn btn-primary btn-lg" autofocus>` with arrow
     svg (enter-to-continue after login).

- **Display name fallback** — `getattr(current_user, 'nombre', None) or
  current_user.email.split('@')[0]|capitalize`. Plan 08-03 already shipped the
  `nombre` column but pre-existing rows may have `nombre IS NULL`; the fallback
  to the email local-part guarantees the landing always renders something
  human-readable.

### Post-login redirect

- `api/routers/auth.py:login_post` — target changed from `/` to `/bienvenida`.
  The `must_change_password` branch (→ `/cambiar-password`) is preserved
  unchanged; forced-change continues to take priority.

### Jinja filter

- `hora_saludo(now=None)` in `api/deps.py`:
  - 06:00..11:59 → `"Buenos días"`
  - 12:00..20:59 → `"Buenas tardes"`
  - 21:00..05:59 → `"Buenas noches"`
  - Accepts `None` (→ current Madrid time), naive `datetime` (assumed Madrid),
    or aware `datetime` (converted to Madrid via `astimezone`).

### Alpine component

- `bienvenidaPage()` in `static/js/app.js`:
  - `init()` — tick once, then `setInterval(tick, 1000)`.
  - `destroy()` — `clearInterval(_timer)` (Pitfall 6: avoids timer leak on
    HTMX swap or Alpine teardown).
  - `tick()` — pads H/M/S to 2 digits and sets `clock = 'HH:MM:SS'` (24h).
  - No CSS transition on the clock update — `prefers-reduced-motion` honored
    naturally since the only change is `textContent`.

## Test coverage

`tests/routers/test_bienvenida.py` — 17 tests, all passing:

**Unit (filter bands, no DB):**
- 8 parametrized cases on hour boundary (00, 05, 06, 11, 12, 20, 21, 23).
- Naive datetime assumed Madrid.
- `None` → current time fallback (returns one of the 3 bands).
- UTC 10:00 → Madrid 12:00 (tardes, DST-aware).

**Integration (require Postgres):**
- Post-login propietario → 303 → `/bienvenida`.
- Post-login `must_change_password=True` → 303 → `/cambiar-password` (branch
  preservation).
- GET `/bienvenida` as propietario → 200 with CTA + `x-data=bienvenidaPage()`
  + greeting band + `x-text="clock"`.
- GET `/bienvenida` as `usuario/rrhh` → 200 (no permission wall).
- GET `/bienvenida` sin cookie → 302 → `/login` (AuthMiddleware global).
- GET `/bienvenida` render Spanish day-name + month-name + literal "Es … de …".

**Phase 5 RBAC regression preserved:** all 41 tests from
`test_rbac_smoke.py` + `test_html_get_guarded.py` + `test_sidebar_filtering.py`
+ `test_button_gating.py` + `test_forbidden_redirect.py` still pass.

## Commits

| Task | Hash    | Message                                                                              |
| ---- | ------- | ------------------------------------------------------------------------------------ |
| 1    | 103c650 | feat(08-04): add hora_saludo filter + bienvenidaPage Alpine + login redirect to /bienvenida |
| 2    | cf58648 | feat(08-04): add templates/bienvenida.html + GET /bienvenida route                   |
| 3    | baeb3fb | test(08-04): regression for /bienvenida landing + hora_saludo filter                 |

## Deviations from Plan

None — plan executed as written with minor cosmetic adjustments:

- **Minor (style)** — Mobile role swap implemented with twin `<h2>` elements
  + `hidden`/`md:hidden` pair (desktop + mobile) rather than a single element
  with a `@media (max-width: 640px)` CSS override. Matches the existing
  Tailwind utility-first pattern in `base.html` (e.g. `hidden sm:inline` on
  the topbar email). Not a deviation from the UI-SPEC contract; same visual
  outcome.

- **Minor (test structure)** — Plan specified 5 tests; shipped 17 (all
  passing) to cover boundary conditions more thoroughly:
  * 8 hora_saludo band boundaries (vs. 3 suggested).
  * Additional tz-conversion test (UTC → Madrid DST-aware).
  * Additional test for Spanish day/month rendering in the template body.
  * `usuario` role test complementing the `propietario` test.

## Handoff

- **Any test asserting login redirects to `/`** → must now assert
  `/bienvenida`. Grep confirmed no existing test held this assertion; the 7
  matches for `location == "/"` in `tests/routers/` are all 403-redirect
  targets (Plan 05-03 flow, unrelated to post-login).
- **Drawer "Centro Mando" active state** — when user is on `/bienvenida`,
  `page == "bienvenida"` → no drawer item is marked active. Intentional
  (transitional state, do NOT add a drawer entry for Bienvenida).
- **Timezone** — the server IS authoritative for the greeting band. If the
  server runs outside Europe/Madrid, zoneinfo handles the conversion; the
  filter always resolves the Madrid hour.
- **Accessibility** — the reloj has `role="timer"` + `aria-label="Hora actual"`.
  Screen readers will periodically announce clock updates; if that proves too
  chatty in practice (Mark-IV feedback), consider `aria-live="off"` + an
  opt-in toggle.

## Deferred Issues

- **3 pre-existing failures in `tests/routers/test_thresholds_crud.py`** —
  `test_recalibrate_insufficient_data_returns_400`,
  `test_recalibrate_preview_and_confirm_persists`,
  `test_recalibrate_filters_outliers_under_500ms`. Confirmed pre-existing via
  checkout of the pre-Task1 commit. Logged in
  `.planning/phases/08-redise-o-ui-modo-claro-moderno/deferred-items.md`.
  Owner: next Phase 8 plan (or a Phase 4 backport) that touches thresholds
  recalibrate semantics.

## Self-Check: PASSED

- `templates/bienvenida.html` — exists (verified `test -f`).
- `tests/routers/test_bienvenida.py` — exists.
- `api/deps.py` — contains `def hora_saludo` + `templates.env.filters["hora_saludo"]`.
- `api/routers/auth.py` — contains `target = "/cambiar-password" if user.must_change_password else "/bienvenida"`.
- `api/routers/pages.py` — contains `@router.get("/bienvenida")` + `now_madrid`.
- `static/js/app.js` — contains `function bienvenidaPage()` + `clearInterval(this._timer)`.
- Commits verified in `git log`:
  - `103c650` — Task 1.
  - `cf58648` — Task 2.
  - `baeb3fb` — Task 3.
- Test suite: 17 bienvenida tests pass + 41 Phase 5 RBAC tests pass + 15
  other Phase 8 tests pass (73 relevant tests green).
