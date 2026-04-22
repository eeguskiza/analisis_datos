---
phase: 08-redise-o-ui-modo-claro-moderno
plan: 04
type: execute
wave: 4
depends_on: [08-03]
files_modified:
  - templates/bienvenida.html
  - api/routers/auth.py
  - api/routers/pages.py
  - api/deps.py
  - static/js/app.js
  - tests/routers/test_bienvenida.py
autonomous: true
gap_closure: false
requirements: [UIREDO-02, UIREDO-05, UIREDO-06]
tags: [landing, bienvenida, saludo, reloj, phase-8, ui]
user_setup: []

must_haves:
  truths:
    - "A new GET `/bienvenida` route renders `templates/bienvenida.html` guarded by authentication (no specific permission — every authenticated user sees it)."
    - "Post-login redirect in `api/routers/auth.py` changes from `/` to `/bienvenida` (EXCEPT the must_change_password redirect stays `/cambiar-password`)."
    - "The greeting band is computed server-side via a Jinja filter `hora_saludo(now)` (06–11→Buenos días, 12–20→Buenas tardes, 21–05→Buenas noches) using Europe/Madrid timezone."
    - "The greeting uses `user.nombre` (Plan 08-03 column) with fallback to email local-part capitalized."
    - "The reloj ticks every second via Alpine `setInterval` inside a `bienvenidaPage()` component; the interval is cleared on Alpine `destroy()` to avoid leaks (Pitfall 6)."
    - "A primary `btn-lg` CTA `Ir a Centro de Mando` navigates to `/`."
    - "The page honours `prefers-reduced-motion` — no gratuitous animation on the reloj (tick is a text update, not an animated transition)."
    - "Mobile (`max-width: 640px`) reduces the greeting role from Display 32/600 to Heading 20/600 per UI-SPEC."
    - "The top bar + drawer (Plan 08-02) are available via the shared `base.html`; drawer accessible with `[` + hamburger as on other screens."
  artifacts:
    - path: "templates/bienvenida.html"
      provides: "New landing template — greeting, day-of-week, date, reloj, primary CTA"
      contains: "{% extends \"base.html\" %}\n{% block page_title %}Bienvenida{% endblock %}"
    - path: "api/routers/pages.py"
      provides: "GET /bienvenida route"
      contains: "@router.get(\"/bienvenida\")"
    - path: "api/routers/auth.py"
      provides: "Post-login redirect target swapped from / to /bienvenida"
      contains: "target = \"/cambiar-password\" if user.must_change_password else \"/bienvenida\""
    - path: "api/deps.py"
      provides: "`hora_saludo` Jinja filter registered on templates.env"
      contains: "templates.env.filters[\"hora_saludo\"] = hora_saludo"
    - path: "static/js/app.js"
      provides: "bienvenidaPage() Alpine component with setInterval + destroy cleanup"
      contains: "function bienvenidaPage()"
    - path: "tests/routers/test_bienvenida.py"
      provides: "Tests for route, redirect, greeting filter, template rendering, reloj markup presence"
      contains: "def test_post_login_redirects_to_bienvenida + def test_hora_saludo_filter_bands"
  key_links:
    - from: "api/routers/auth.py:login_post"
      to: "/bienvenida"
      via: "RedirectResponse target"
      pattern: "/bienvenida"
    - from: "templates/bienvenida.html"
      to: "static/js/app.js bienvenidaPage()"
      via: "Alpine x-data + setInterval tick"
      pattern: "x-data=\"bienvenidaPage\\(\\)\""
---

<objective>
Ship the `/bienvenida` landing screen that replaces `/` as the
post-login target. It renders a time-banded greeting, a live reloj, the
current date in Spanish, and a primary CTA into Centro de Mando. All
chrome (top bar + drawer + toast) comes from `base.html` (Plan 08-02),
so this plan only adds the landing-specific content block and the
`/bienvenida` route + redirect + Jinja filter.

Purpose: UIREDO-02 extension per CONTEXT D-23 requires a landing
screen; UI-SPEC §Landing Screen is the source of truth. The landing
keeps the drawer accessible so users can navigate directly without
bouncing through Centro de Mando.

Output: 1 new Jinja template, 1 new route, 1 route rewrite (login
redirect), 1 Jinja filter, 1 Alpine component, 1 test file.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/08-redise-o-ui-modo-claro-moderno/08-CONTEXT.md
@.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md
@.planning/phases/08-redise-o-ui-modo-claro-moderno/08-02-SUMMARY.md
@.planning/phases/08-redise-o-ui-modo-claro-moderno/08-03-SUMMARY.md
@api/routers/auth.py
@api/routers/pages.py
@api/deps.py
@templates/base.html
@static/js/app.js
@CLAUDE.md

<interfaces>
<!-- Current login redirect in api/routers/auth.py:141 -->

```python
target = "/cambiar-password" if user.must_change_password else "/"
response: Response = RedirectResponse(target, status_code=303)
```

Plan 08-04 changes the `else` branch to `/bienvenida`. The
`must_change_password` branch stays as-is.

<!-- pages.py routes pattern (from existing /datos, /historial) -->

Existing routes use:
```python
@router.get("/datos", dependencies=[Depends(require_permission("datos:read"))])
def datos_page(request: Request, ...):
    return render("datos.html", request, _common_extra("datos"))
```

For `/bienvenida`, there is NO specific permission — every authenticated
user lands here. Therefore NO `require_permission(...)` dependency; the
`AuthMiddleware` already protects authenticated routes globally.

<!-- Jinja filter registration in api/deps.py -->

```python
templates.env.globals.update(app_name=..., can=_can, ...)
# Add:
templates.env.filters["hora_saludo"] = hora_saludo
```

<!-- Landing layout from UI-SPEC §Landing Screen -->

```
┌──────────────── Top bar (shared) ──────────────────┐
│ [≡]  Bienvenida   conn-badge · user menu         │
├──────────────────────────────────────────────────┤
│  [Saludo, {nombre}]     ← Display 32/600         │
│  [Es {lunes}, {22 de abril de 2026}] ← Body muted│
│  [HH:MM:SS]             ← Display 32/600 mono    │
│  [ Ir a Centro de Mando → ]  ← btn-primary btn-lg │
└──────────────────────────────────────────────────┘
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add hora_saludo filter in api/deps.py + tick helper in static/js/app.js + post-login redirect change</name>
  <read_first>
    - `api/deps.py` — see where templates.env is configured.
    - `api/routers/auth.py:141-153` — the current `/` target and redirect.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Saludo rules (hour bands, D-23)".
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-RESEARCH.md` §Pitfall 6 (setInterval leak).
    - Python stdlib: `zoneinfo.ZoneInfo` (Python 3.9+ standard).
  </read_first>
  <action>
### Part A — `api/deps.py` — register the `hora_saludo` filter

Append (AFTER the existing `templates.env.globals.update(...)` block):

```python
# ── Plan 08-04: hora_saludo filter (D-23 — saludo por franja) ───────────────
from datetime import datetime
from zoneinfo import ZoneInfo

_MADRID = ZoneInfo("Europe/Madrid")


def hora_saludo(now: datetime | None = None) -> str:
    """Return the time-banded Spanish greeting for the given instant.

    Bands (D-23):
      06:00..11:59 → 'Buenos días'
      12:00..20:59 → 'Buenas tardes'
      21:00..05:59 → 'Buenas noches'

    Uses Europe/Madrid (server tz is authoritative — UI-SPEC §Landing).
    """
    if now is None:
        now = datetime.now(_MADRID)
    else:
        if now.tzinfo is None:
            now = now.replace(tzinfo=_MADRID)
        else:
            now = now.astimezone(_MADRID)
    h = now.hour
    if 6 <= h < 12:
        return "Buenos días"
    if 12 <= h < 21:
        return "Buenas tardes"
    return "Buenas noches"


templates.env.filters["hora_saludo"] = hora_saludo
```

The filter accepts an optional `datetime`; if called with no arg
(`{{ None|hora_saludo }}` style) it falls back to "now".

### Part B — `api/routers/auth.py` — change post-login redirect

Locate the line (current file at line 141):

```python
target = "/cambiar-password" if user.must_change_password else "/"
```

Replace with:

```python
target = "/cambiar-password" if user.must_change_password else "/bienvenida"
```

Do not change anything else in the login flow.

### Part C — `static/js/app.js` — `bienvenidaPage()` Alpine component

Append near the end of the file (after the `nexoChrome()` function from
Plan 08-02):

```js
// ── Landing: reloj (Plan 08-04 / UI-SPEC §Landing) ─────────────────────────
// Ticks every second. Interval cleared on destroy() (Pitfall 6).

function bienvenidaPage() {
  return {
    clock: '',
    _timer: null,
    init() {
      this.tick();
      this._timer = setInterval(() => this.tick(), 1000);
    },
    destroy() {
      if (this._timer) { clearInterval(this._timer); this._timer = null; }
    },
    tick() {
      const d = new Date();
      const hh = String(d.getHours()).padStart(2, '0');
      const mm = String(d.getMinutes()).padStart(2, '0');
      const ss = String(d.getSeconds()).padStart(2, '0');
      this.clock = `${hh}:${mm}:${ss}`;
    },
  };
}

window.bienvenidaPage = bienvenidaPage;
```

The tick is pure JS; no animation, no CSS transition — respects
`prefers-reduced-motion` naturally.
  </action>
  <acceptance_criteria>
    - `grep -c "def hora_saludo" api/deps.py` returns 1.
    - `grep -c "templates.env.filters\[\"hora_saludo\"\]" api/deps.py` returns 1.
    - `grep -c "/bienvenida" api/routers/auth.py` returns 1 or more.
    - `grep -c "function bienvenidaPage()" static/js/app.js` returns 1.
    - `grep -c "clearInterval(this._timer)" static/js/app.js` returns 1 or more.
    - `ruff check api/ nexo/` exit 0, `mypy api/ nexo/` exit 0.
  </acceptance_criteria>
  <verify>
    <automated>grep -q "def hora_saludo" api/deps.py &amp;&amp; grep -q "/bienvenida" api/routers/auth.py &amp;&amp; grep -q "function bienvenidaPage()" static/js/app.js &amp;&amp; ruff check api/ nexo/ &amp;&amp; mypy api/ nexo/</automated>
  </verify>
  <done>Filter registered. Redirect changed. Alpine component exists with proper interval cleanup.</done>
</task>

<task type="auto">
  <name>Task 2: Create templates/bienvenida.html + GET /bienvenida route</name>
  <read_first>
    - `templates/base.html` (from 08-02) — confirm `{% block page_title %}` + `{% block content %}` slots.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Landing Screen" in full.
    - `api/routers/pages.py` — existing route pattern (uses `render()` helper).
  </read_first>
  <action>
### Part A — Create `templates/bienvenida.html`

```jinja
{% extends "base.html" %}
{% block title %}Bienvenida · {{ app_name }}{% endblock %}
{% block page_title %}Bienvenida{% endblock %}

{% block content %}
{# Display name = nombre if populated (Plan 08-03), else email local-part capitalized. #}
{% set display_name = current_user.nombre if current_user and current_user.nombre
   else (current_user.email.split('@')[0]|capitalize if current_user else '') %}

<section x-data="bienvenidaPage()"
         class="max-w-[640px] mx-auto py-16 flex flex-col items-center text-center gap-6">

  {# Saludo (Display on desktop, Heading on mobile per UI-SPEC) #}
  <h2 class="text-display text-heading md:block hidden">
    {{ None|hora_saludo }}, {{ display_name }}
  </h2>
  <h2 class="text-heading text-heading md:hidden block">
    {{ None|hora_saludo }}, {{ display_name }}
  </h2>

  {# Día y fecha. Server-rendered en castellano; el reloj es client-side. #}
  {% set dia_semana = [
      "Lunes", "Martes", "Miércoles", "Jueves",
      "Viernes", "Sábado", "Domingo"
  ][(now_madrid.weekday())] %}
  {% set meses = [
      "enero", "febrero", "marzo", "abril", "mayo", "junio",
      "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
  ] %}
  <p class="text-body text-muted">
    Es {{ dia_semana }}, {{ now_madrid.day }} de {{ meses[now_madrid.month - 1] }} de {{ now_madrid.year }}
  </p>

  {# Reloj (client-side tick, Alpine). #}
  <div class="text-display text-heading font-mono tabular-nums" x-text="clock"
       aria-label="Hora actual" role="timer"></div>

  {# Primary CTA #}
  <a href="/" class="btn btn-primary btn-lg" autofocus>
    Ir a Centro de Mando
    <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.75" viewBox="0 0 24 24" aria-hidden="true">
      <path stroke-linecap="round" stroke-linejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6"/>
    </svg>
  </a>

  {# TODO Mark-IV: widgets configurables per-user (D-23 deferred). #}
</section>
{% endblock %}
```

### Part B — `api/routers/pages.py` — add the route

Append to the existing router block:

```python
from datetime import datetime
from zoneinfo import ZoneInfo

_MADRID_TZ = ZoneInfo("Europe/Madrid")


@router.get("/bienvenida")
def bienvenida_page(request: Request):
    extra = _common_extra("bienvenida")
    extra["now_madrid"] = datetime.now(_MADRID_TZ)
    return render("bienvenida.html", request, extra)
```

Do NOT add a permission dependency — landing is accessible to any
authenticated user. The global `AuthMiddleware` already redirects
unauthenticated HTML to `/login`.

### Part C — Verify the drawer "Centro Mando" active state stays correct

In `templates/base.html` (Plan 08-02), the main nav entry for `"/"` has
key `"dashboard"`. On the `/bienvenida` route, `page = "bienvenida"` — so
no drawer item is marked active. That's fine — the user is in a
transitional state. Do NOT add a "Bienvenida" nav item to the drawer.
  </action>
  <acceptance_criteria>
    - `test -f templates/bienvenida.html` returns 0.
    - `grep -c "x-data=\"bienvenidaPage()\"" templates/bienvenida.html` returns 1.
    - `grep -c "|hora_saludo" templates/bienvenida.html` returns 2 (one per breakpoint heading).
    - `grep -c "Ir a Centro de Mando" templates/bienvenida.html` returns 1.
    - `grep -c "btn btn-primary btn-lg" templates/bienvenida.html` returns 1.
    - `grep -c "@router.get(\"/bienvenida\")" api/routers/pages.py` returns 1.
    - `grep -c "now_madrid" api/routers/pages.py` returns 1 or more.
    - `grep -c "now_madrid" templates/bienvenida.html` returns 2 or more.
    - App starts locally without error.
  </acceptance_criteria>
  <verify>
    <automated>test -f templates/bienvenida.html &amp;&amp; grep -q "x-data=\"bienvenidaPage()\"" templates/bienvenida.html &amp;&amp; grep -q "@router.get(\"/bienvenida\")" api/routers/pages.py &amp;&amp; ruff check api/ nexo/</automated>
  </verify>
  <done>Template ships. Route exists. No permission guard (intentional). Drawer still works from landing.</done>
</task>

<task type="auto">
  <name>Task 3: Test suite — redirect + filter + route + template render + reloj presence</name>
  <read_first>
    - `tests/routers/` existing patterns for HTML-route tests + login helper fixtures.
    - `tests/conftest.py` for `propietario_client` or equivalent.
  </read_first>
  <action>
Create `tests/routers/test_bienvenida.py`:

```python
"""Regression for Phase 8 / Plan 08-04: /bienvenida landing.

Locks:
1. Post-login redirect targets /bienvenida (not /).
2. GET /bienvenida renders for any authenticated user.
3. hora_saludo filter returns the correct Spanish band per hour.
4. The template renders the reloj + greeting + primary CTA.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest


_MADRID = ZoneInfo("Europe/Madrid")


# ── Filter bands ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "hour, expected",
    [
        (0, "Buenas noches"),
        (5, "Buenas noches"),
        (6, "Buenos días"),
        (11, "Buenos días"),
        (12, "Buenas tardes"),
        (20, "Buenas tardes"),
        (21, "Buenas noches"),
        (23, "Buenas noches"),
    ],
)
def test_hora_saludo_filter_bands(hour: int, expected: str):
    from api.deps import hora_saludo

    dt = datetime(2026, 4, 22, hour, 0, 0, tzinfo=_MADRID)
    assert hora_saludo(dt) == expected


def test_hora_saludo_naive_datetime_assumed_madrid():
    from api.deps import hora_saludo

    naive = datetime(2026, 4, 22, 10, 0, 0)
    assert hora_saludo(naive) == "Buenos días"


def test_hora_saludo_none_uses_now():
    from api.deps import hora_saludo

    r = hora_saludo(None)
    assert r in {"Buenos días", "Buenas tardes", "Buenas noches"}


# ── Post-login redirect ─────────────────────────────────────────────────────

def test_post_login_redirects_to_bienvenida(test_client, seed_propietario):
    resp = test_client.post(
        "/login",
        data={"email": seed_propietario.email, "password": "PropietarioPass123"},
        follow_redirects=False,
    )
    assert resp.status_code in (302, 303)
    assert resp.headers["location"] == "/bienvenida"


def test_post_login_must_change_password_still_redirects_to_change(
    test_client, seed_user_must_change
):
    resp = test_client.post(
        "/login",
        data={"email": seed_user_must_change.email, "password": "InitialPass123"},
        follow_redirects=False,
    )
    assert resp.status_code in (302, 303)
    assert resp.headers["location"] == "/cambiar-password"


# ── Route render ────────────────────────────────────────────────────────────

def test_bienvenida_route_renders_for_propietario(propietario_client):
    resp = propietario_client.get("/bienvenida")
    assert resp.status_code == 200
    body = resp.text
    assert "Ir a Centro de Mando" in body
    assert 'x-data="bienvenidaPage()"' in body
    # Greeting band must be present (any of the 3)
    assert any(g in body for g in ("Buenos días", "Buenas tardes", "Buenas noches"))


def test_bienvenida_route_renders_for_usuario(usuario_client):
    resp = usuario_client.get("/bienvenida")
    assert resp.status_code == 200
    body = resp.text
    assert "Ir a Centro de Mando" in body


def test_bienvenida_route_requires_auth(test_client):
    resp = test_client.get("/bienvenida", follow_redirects=False)
    # Unauthenticated → AuthMiddleware redirects HTML to /login
    assert resp.status_code in (302, 303)
    assert resp.headers["location"].startswith("/login")
```

Reuse existing `test_client`, `propietario_client`, `usuario_client`,
`seed_propietario`, `seed_user_must_change` fixtures. If their exact
names differ, port to the nearest equivalent and log in the SUMMARY.
  </action>
  <acceptance_criteria>
    - `test -f tests/routers/test_bienvenida.py` returns 0.
    - `pytest tests/routers/test_bienvenida.py -x -q` exits 0.
    - `pytest tests/ -x -q` (full suite) exits 0.
    - `ruff check tests/` exit 0.
  </acceptance_criteria>
  <verify>
    <automated>ruff check tests/routers/test_bienvenida.py &amp;&amp; ruff format --check tests/routers/test_bienvenida.py &amp;&amp; pytest tests/routers/test_bienvenida.py -x -q &amp;&amp; pytest tests/ -x -q</automated>
  </verify>
  <done>Landing tested end-to-end. Full suite green.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Browser → setInterval | Client-only; no data sent. |
| Server → greeting | `display_name` is HTML-escaped via Jinja `{{ }}`; same for `dia_semana`/`meses` which are static literals. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-08-04-01 | XSS | `display_name` rendered in `<h2>` | mitigate | Jinja auto-escape on `{{ ... }}`. No `|safe` applied. Plan 08-03 cap 120 chars. |
| T-08-04-02 | Tampering | Client clock drifts → wrong greeting in reloj | accept | Reloj is UX, not authoritative. Next navigation re-renders with server time. |
| T-08-04-03 | DoS | setInterval leak on HTMX partial swap | mitigate | Alpine `destroy()` clears `_timer` (Pitfall 6). |
| T-08-04-04 | Authorisation bypass | `/bienvenida` missing auth check | mitigate | Global AuthMiddleware guards all non-public routes; confirmed by `test_bienvenida_route_requires_auth`. |
</threat_model>

<verification>
1. `pytest tests/ -x -q` green.
2. Manual:
   - Log out, log in as propietario. Redirect lands on `/bienvenida`.
   - Top bar shows avatar + role pill. Hamburger + `[` open drawer.
   - Saludo matches current time (Buenos días / tardes / noches).
   - Day-of-week + date in Spanish (Martes, 22 de abril de 2026).
   - Reloj ticks once per second.
   - Click "Ir a Centro de Mando" → navigate to `/`.
   - Reload `/bienvenida` repeatedly; browser memory stable (no
     interval accumulation).
   - Go to `/bienvenida` as a `usuario` of `rrhh` (no pipeline perm) —
     greeting + reloj + CTA still render; drawer only shows the items
     that user has permission for.
</verification>

<success_criteria>
- New `/bienvenida` route rendering greeting + reloj + CTA.
- Post-login redirect switched to `/bienvenida` (except must_change_password).
- `hora_saludo` filter correct on all 3 bands.
- Reloj ticks with proper destroy cleanup.
- Pre-existing tests green.
</success_criteria>

<output>
After completion, create `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-04-SUMMARY.md` with:

- Added: `templates/bienvenida.html`, `/bienvenida` route.
- Edited: `api/routers/auth.py` (redirect), `api/deps.py` (filter), `static/js/app.js` (bienvenidaPage).
- Test coverage: bands + redirects + render + auth required.
- Handoff: any test that counted on `/` being the login target needs to
  look at `/bienvenida` now — list any that were updated.
</output>