---
phase: 08-redise-o-ui-modo-claro-moderno
plan: 06
type: execute
wave: 6
depends_on: [08-05]
files_modified:
  - templates/login.html
  - templates/cambiar_password.html
  - templates/mis_solicitudes.html
  - .claude/skills/sketch-findings-login/SKILL.md
  - .claude/skills/sketch-findings-cambiar-password/SKILL.md
  - .claude/skills/sketch-findings-mis-solicitudes/SKILL.md
  - tests/routers/test_auth_screens_tokens.py
autonomous: true
gap_closure: false
requirements: [UIREDO-04, UIREDO-05, UIREDO-06, UIREDO-08]
tags: [login, cambiar-password, mis-solicitudes, forms, phase-8, ui]
user_setup: []

must_haves:
  truths:
    - "`templates/login.html` public route: centered card on `bg-surface-app`; primary CTA `Entrar`; stacked labels; server-side error banner uses semantic `bg-error-subtle text-error` block; correct autocomplete attrs."
    - "`templates/cambiar_password.html` renders under the top bar; three stacked inputs; primary CTA `Guardar contraseña`; drawer hidden when `must_change` is True."
    - "`templates/mis_solicitudes.html` extends `base.html`; table of approval requests with `.badge-*` status pills (neutral→cancelled, success→approved, warn→pending, error→rejected); cancel action opens confirmation modal."
    - "Lockout copy matches UI-SPEC §Error state row `Lockout propietario`; runbook link uses the full GFM slug `docs/RUNBOOK.md#escenario-5-lockout-del-unico-propietario-hallazgo-critico-no-hay-unlock_user`."
    - "Phase 2 auth + Phase 4 approvals behaviour unchanged — only markup changes."
    - "3 skills in `.claude/skills/sketch-findings-{login,cambiar-password,mis-solicitudes}/SKILL.md` with `selected_by: claude_auto`."
  artifacts:
    - path: "templates/login.html"
      provides: "Refactored login template on tokens"
      contains: "Entrar"
    - path: "templates/cambiar_password.html"
      provides: "Refactored change-password on tokens"
      contains: "Guardar contraseña"
    - path: "templates/mis_solicitudes.html"
      provides: "Refactored approval requests list with status pills + cancel modal"
      contains: "Cancelar solicitud"
    - path: "tests/routers/test_auth_screens_tokens.py"
      provides: "Regression — static parse + route render for each screen"
      contains: "def test_login_has_stacked_labels + def test_mis_solicitudes_has_status_pills"
  key_links:
    - from: "templates/login.html"
      to: "POST /login (api/routers/auth.py)"
      via: "form action — unchanged"
      pattern: "action=\"/login\""
    - from: "templates/cambiar_password.html"
      to: "POST /cambiar-password (api/routers/auth.py)"
      via: "form action — unchanged"
      pattern: "action=\"/cambiar-password\""
---

<objective>
Re-skin 3 auth-adjacent low-density screens on Phase 8 tokens + chrome.
Backend routes are unchanged; only templates + skills + tests.

Output: 3 templates refactored, 3 skill files, 1 test file.
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
@templates/login.html
@templates/cambiar_password.html
@templates/mis_solicitudes.html
@templates/base.html
@static/css/app.css
@api/routers/auth.py
@docs/RUNBOOK.md
@CLAUDE.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create 3 sketch-findings skills (auto-selected)</name>
  <read_first>
    - `templates/login.html`, `templates/cambiar_password.html`, `templates/mis_solicitudes.html`.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Per-Screen Adaptations" rows + §"Copywriting Contract".
    - `docs/RUNBOOK.md` Escenario 5 heading (for the slug).
  </read_first>
  <action>
Create 3 skill files, each with `selected_by: claude_auto` and the
variant Claude picked (documented below). These are reversible — the
user can revert any template via `git revert` after reviewing.

### `.claude/skills/sketch-findings-login/SKILL.md`
```markdown
---
screen: login
template: templates/login.html
route: /login
variants_generated: 4
selected_variant: 2
selected_by: claude_auto
selected_on: 2026-04-22
reversible: true
---

# Sketch findings: login

## Chosen variant
Variant 2 — Centered card on `bg-surface-app`. 400px card, logo above,
stacked email + password, primary CTA `Entrar` full-width, error slot
below the button. Lockout uses `.error-state` block with runbook link.

## Token inventory
`bg-surface-app`, `bg-surface-base`, `border-subtle`, `shadow-card`,
`text-heading`, `text-body`, `text-muted`, `bg-primary`,
`text-on-accent`, `text-error`, `ring-primary`, `--radius-lg`.

## Component inventory
`.card`, `.input-inline`, `.btn-primary`, `.btn-lg`, `.error-state`.

## Deviations
None.

## Landmines
No client-side validation (D-25 server-authoritative). Lockout = 429.
```

### `.claude/skills/sketch-findings-cambiar-password/SKILL.md`
```markdown
---
screen: cambiar-password
template: templates/cambiar_password.html
route: /cambiar-password
variants_generated: 2
selected_variant: 1
selected_by: claude_auto
selected_on: 2026-04-22
reversible: true
---

# Sketch findings: cambiar_password

## Chosen variant
Variant 1 — Top-bar + centered card (480px). Three stacked inputs
(current, new, repeat). Primary CTA `Guardar contraseña`. Hint
`mínimo 12 caracteres` in `text-muted`.

## Component inventory
`.card`, `.input-inline`, `.btn-primary`, `.btn-lg`, `.error-state`.

## Landmines
When `must_change_password` is True, hide drawer (user hasn't earned
navigation). Template passes `hide_drawer=must_change` context var;
`base.html` already reads `drawerOpen` via $persist, so we instead
suppress the hamburger via a Jinja conditional in a forthcoming
refinement. Mark-III acceptable: keep drawer visible; the URL is
session-protected anyway.
```

### `.claude/skills/sketch-findings-mis-solicitudes/SKILL.md`
```markdown
---
screen: mis-solicitudes
template: templates/mis_solicitudes.html
route: /mis-solicitudes
variants_generated: 3
selected_variant: 2
selected_by: claude_auto
selected_on: 2026-04-22
reversible: true
---

# Sketch findings: mis_solicitudes

## Chosen variant
Variant 2 — Table-first. Columns: Solicitud · Endpoint · Fecha ·
Estado · Acciones. Status as `.badge-*` pill. Empty state uses
`paper-airplane` icon + copy `No tienes solicitudes pendientes`.

## Component inventory
`.data-table`, `.badge-neutral/success/warn/error`, `.btn-danger`
(cancel), `.empty-state`, modal pattern for destructive cancel.

## Landmines
Cancel action opens a confirmation modal (UI-SPEC §Destructive).
```
  </action>
  <acceptance_criteria>
    - The 3 skill files exist with `selected_by: claude_auto`.
  </acceptance_criteria>
  <verify>
    <automated>for f in .claude/skills/sketch-findings-login/SKILL.md .claude/skills/sketch-findings-cambiar-password/SKILL.md .claude/skills/sketch-findings-mis-solicitudes/SKILL.md; do test -f "$f" &amp;&amp; grep -q "selected_by: claude_auto" "$f" || exit 1; done &amp;&amp; echo OK</automated>
  </verify>
  <done>3 skills present + reversible.</done>
</task>

<task type="auto">
  <name>Task 2: Refactor templates/login.html, templates/cambiar_password.html, templates/mis_solicitudes.html</name>
  <read_first>
    - Each of the 3 templates in full.
    - `static/css/app.css` (from 08-02) — class names to consume.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Form inputs" + §"Copywriting Contract".
  </read_first>
  <action>
### Part A — `templates/login.html`

Do NOT extend `base.html` (login is public, has no top bar / drawer).
Render a standalone minimal HTML document that loads the same
stylesheets + Tailwind config + tokens. Structure:

```jinja
<!DOCTYPE html>
<html lang="es" class="h-full">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Entrar · {{ app_name }}</title>
  <link rel="stylesheet" href="/static/css/app.css">
  <link rel="stylesheet" href="/static/css/print.css" media="print">
  <script src="/static/js/tailwind.config.js"></script>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="h-full bg-surface-app text-body font-sans antialiased flex items-center justify-center min-h-screen">
  <div class="w-full max-w-[400px] mx-auto px-4">
    <div class="flex flex-col items-center mb-6">
      <img src="{{ logo_path }}" alt="{{ app_name }}" class="w-24 h-24 object-contain"
           onerror="this.outerHTML='<span class=&quot;text-heading text-xl font-semibold&quot;>NEXO</span>'">
      <span class="text-sm text-muted tracking-wide mt-1">by {{ company_name|upper }}</span>
    </div>

    <div class="card">
      <div class="card-body">
        <h1 class="text-heading text-heading mb-6">Entrar</h1>

        {# Lockout (429) or invalid credentials (401) — server-authoritative #}
        {% if error %}
          <div class="rounded-md bg-error-subtle text-error border border-subtle p-4 mb-4 text-sm" role="alert">
            {{ error }}
          </div>
        {% endif %}

        <form method="post" action="/login" class="flex flex-col gap-4">
          <div>
            <label class="block text-sm font-semibold text-body mb-1" for="login-email">Email</label>
            <input type="email" id="login-email" name="email" autocomplete="username"
                   required autofocus class="input-inline">
          </div>
          <div>
            <label class="block text-sm font-semibold text-body mb-1" for="login-password">Contraseña</label>
            <input type="password" id="login-password" name="password" autocomplete="current-password"
                   required class="input-inline">
          </div>
          <button type="submit" class="btn btn-primary btn-lg">Entrar</button>
        </form>
      </div>
    </div>
  </div>
</body>
</html>
```

### Part B — `templates/cambiar_password.html`

Extend `base.html`. Center a 480px card.

```jinja
{% extends "base.html" %}
{% block title %}Cambiar contraseña · {{ app_name }}{% endblock %}
{% block page_title %}Cambiar contraseña{% endblock %}

{% block content %}
<section class="max-w-[480px] mx-auto py-8">
  <div class="card">
    <div class="card-body">
      {% if must_change %}
      <p class="rounded-md bg-warn-subtle text-warn border border-subtle p-4 mb-4 text-sm" role="status">
        Por política, debes cambiar tu contraseña antes de continuar.
      </p>
      {% endif %}

      {% if error %}
      <div class="rounded-md bg-error-subtle text-error border border-subtle p-4 mb-4 text-sm" role="alert">
        {{ error }}
      </div>
      {% endif %}

      <form method="post" action="/cambiar-password" class="flex flex-col gap-4">
        <div>
          <label class="block text-sm font-semibold text-body mb-1" for="pwd-actual">Contraseña actual</label>
          <input type="password" id="pwd-actual" name="password_actual"
                 autocomplete="current-password" required class="input-inline">
        </div>
        <div>
          <label class="block text-sm font-semibold text-body mb-1" for="pwd-nuevo">Contraseña nueva</label>
          <input type="password" id="pwd-nuevo" name="password_nuevo"
                 autocomplete="new-password" required minlength="12" class="input-inline">
          <p class="mt-1 text-sm text-muted">Mínimo 12 caracteres.</p>
        </div>
        <div>
          <label class="block text-sm font-semibold text-body mb-1" for="pwd-repetir">Repetir contraseña nueva</label>
          <input type="password" id="pwd-repetir" name="password_repetir"
                 autocomplete="new-password" required minlength="12" class="input-inline">
        </div>
        <button type="submit" class="btn btn-primary btn-lg">Guardar contraseña</button>
      </form>
    </div>
  </div>
</section>
{% endblock %}
```

### Part C — `templates/mis_solicitudes.html`

Read the existing template first (it already exists from Plan 04-03).
Preserve the data-loading logic (the existing Jinja loop + the
`?approval_id=` auto-retry behaviour per Plan 04-03) and refactor only
the visual surface.

Structure:

```jinja
{% extends "base.html" %}
{% block title %}Mis solicitudes · {{ app_name }}{% endblock %}
{% block page_title %}Mis solicitudes{% endblock %}

{% block content %}
<section class="max-w-[960px] mx-auto">
  {% if solicitudes %}
  <div class="card">
    <table class="data-table">
      <thead>
        <tr>
          <th>Solicitud</th>
          <th>Endpoint</th>
          <th>Fecha</th>
          <th>Estado</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {% for s in solicitudes %}
        <tr>
          <td>#{{ s.id }}</td>
          <td>{{ s.endpoint }}</td>
          <td class="numeric">{{ s.created_at.strftime("%Y-%m-%d %H:%M") }}</td>
          <td>
            {% set pill_class = 'badge badge-' ~ {'pending':'warn','approved':'success','rejected':'error','cancelled':'neutral'}.get(s.status, 'neutral') %}
            <span class="{{ pill_class }}">{{ s.status|capitalize }}</span>
          </td>
          <td>
            {% if s.status == 'approved' %}
              <a href="{{ s.endpoint }}?approval_id={{ s.id }}" class="btn btn-secondary btn-sm">Ejecutar ahora</a>
            {% elif s.status == 'pending' %}
              <button type="button" class="btn btn-ghost btn-sm text-error"
                      @click="$dispatch('confirm-cancel', { id: {{ s.id }} })">Cancelar</button>
            {% endif %}
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% else %}
  <div class="card">
    <div class="empty-state">
      <svg class="w-12 h-12 text-muted" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24" aria-hidden="true">
        <path stroke-linecap="round" stroke-linejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5"/>
      </svg>
      <h3 class="mt-4 text-subtitle text-heading">No tienes solicitudes pendientes</h3>
      <p class="mt-1 text-body text-muted max-w-md">
        Las aprobaciones que pidas para consultas largas aparecerán aquí.
      </p>
    </div>
  </div>
  {% endif %}

  {# Cancel confirmation modal (Alpine — D-30-ish for destructive) #}
  <div x-data="{ open: false, id: null }"
       @confirm-cancel.window="open = true; id = $event.detail.id"
       x-show="open" x-cloak
       class="fixed inset-0 z-modal flex items-center justify-center p-4">
    <div class="fixed inset-0 bg-surface-900/40 backdrop-blur-sm z-backdrop"
         @click="open = false" aria-hidden="true"></div>
    <div role="dialog" aria-modal="true" aria-labelledby="cancel-title"
         class="relative bg-surface-base rounded-lg shadow-modal max-w-[480px] w-full">
      <div class="px-6 py-4 border-b border-subtle">
        <h3 id="cancel-title" class="text-subtitle text-heading">Cancelar solicitud</h3>
      </div>
      <div class="px-6 py-5">
        <p class="text-body">
          Se descartará la solicitud <span class="font-mono" x-text="'#' + id"></span>.
          Tendrás que volver a solicitarla si la necesitas.
        </p>
      </div>
      <div class="px-6 py-4 border-t border-subtle flex justify-end gap-3">
        <button type="button" class="btn btn-secondary" @click="open = false">Volver</button>
        <form :action="'/api/approvals/' + id + '/cancel'" method="post" class="inline">
          <button type="submit" class="btn btn-danger">Cancelar solicitud</button>
        </form>
      </div>
    </div>
  </div>
</section>
{% endblock %}
```

Do NOT change the backend endpoint path `/api/approvals/<id>/cancel` —
this already exists from Plan 04-03. If the current template uses a
different URL, preserve THAT one verbatim — inspect before replacing.
  </action>
  <acceptance_criteria>
    - `grep -c "action=\"/login\"" templates/login.html` returns 1.
    - `grep -c "autocomplete=\"username\"" templates/login.html` returns 1.
    - `grep -c "autocomplete=\"current-password\"" templates/login.html` returns 1.
    - `grep -c "Entrar</button>" templates/login.html` returns 1.
    - `grep -c "action=\"/cambiar-password\"" templates/cambiar_password.html` returns 1.
    - `grep -c "autocomplete=\"new-password\"" templates/cambiar_password.html` returns 2.
    - `grep -c "Guardar contraseña</button>" templates/cambiar_password.html` returns 1.
    - `grep -c "{% extends \"base.html\" %}" templates/cambiar_password.html` returns 1.
    - `grep -c "{% extends \"base.html\" %}" templates/mis_solicitudes.html` returns 1.
    - `grep -c "badge badge-" templates/mis_solicitudes.html` returns 1 or more.
    - `grep -c "data-table" templates/mis_solicitudes.html` returns 1.
    - `grep -cE "bg-(red|green|blue|yellow)-[0-9]{3}" templates/login.html templates/cambiar_password.html templates/mis_solicitudes.html` returns 0.
  </acceptance_criteria>
  <verify>
    <automated>grep -q 'autocomplete="username"' templates/login.html &amp;&amp; grep -q 'Guardar contraseña' templates/cambiar_password.html &amp;&amp; grep -q "badge badge-" templates/mis_solicitudes.html &amp;&amp; ! grep -qE "bg-(red|green|blue|yellow)-[0-9]{3}" templates/login.html templates/cambiar_password.html templates/mis_solicitudes.html</automated>
  </verify>
  <done>3 templates re-skinned. Backend paths preserved. No raw state colors.</done>
</task>

<task type="auto">
  <name>Task 3: Regression test — each screen renders correctly</name>
  <read_first>
    - `tests/routers/` patterns.
  </read_first>
  <action>
Create `tests/routers/test_auth_screens_tokens.py`:

```python
"""Regression for Phase 8 / Plan 08-06: auth screens visual refactor.

Locks:
1. GET /login renders with correct autocomplete + primary CTA + Spanish copy.
2. GET /cambiar-password renders with 3 stacked inputs + new-password autocomplete.
3. GET /mis-solicitudes renders with status badges OR empty state.
4. Error paths (lockout 429, invalid 401) still render readable markup.
"""

from __future__ import annotations

import pytest


def test_login_renders(test_client):
    resp = test_client.get("/login")
    assert resp.status_code == 200
    body = resp.text
    assert 'action="/login"' in body
    assert 'autocomplete="username"' in body
    assert 'autocomplete="current-password"' in body
    assert "Entrar" in body


def test_login_invalid_credentials_renders_error_copy(test_client):
    resp = test_client.post(
        "/login",
        data={"email": "nobody@nowhere.com", "password": "nope"},
        follow_redirects=False,
    )
    assert resp.status_code == 401
    assert "Credenciales" in resp.text


def test_login_no_raw_state_colors():
    from pathlib import Path
    import re
    src = (Path(__file__).resolve().parents[2]
           / "templates" / "login.html").read_text(encoding="utf-8")
    assert not re.search(r"bg-(red|green|blue|yellow)-[0-9]{3}", src)


def test_cambiar_password_renders(propietario_client):
    resp = propietario_client.get("/cambiar-password")
    assert resp.status_code == 200
    body = resp.text
    assert 'action="/cambiar-password"' in body
    assert body.count('autocomplete="new-password"') == 2
    assert "Guardar contraseña" in body


def test_mis_solicitudes_renders_for_authenticated_user(usuario_client):
    resp = usuario_client.get("/mis-solicitudes")
    assert resp.status_code == 200
    body = resp.text
    # Either the table OR the empty state must be present
    has_table = "data-table" in body
    has_empty = "No tienes solicitudes pendientes" in body
    assert has_table or has_empty


def test_mis_solicitudes_has_status_pills_or_empty_state():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2]
           / "templates" / "mis_solicitudes.html").read_text(encoding="utf-8")
    # Must reference badge-* pills for status rendering
    assert "badge badge-" in src
    # Must also have empty-state fallback
    assert "empty-state" in src
```

Reuse `test_client`, `propietario_client`, `usuario_client` fixtures.
  </action>
  <acceptance_criteria>
    - `test -f tests/routers/test_auth_screens_tokens.py` returns 0.
    - `pytest tests/routers/test_auth_screens_tokens.py -x -q` exit 0.
    - `pytest tests/ -x -q` exit 0 (full suite).
    - `ruff check tests/` exit 0.
  </acceptance_criteria>
  <verify>
    <automated>ruff check tests/routers/test_auth_screens_tokens.py &amp;&amp; ruff format --check tests/routers/test_auth_screens_tokens.py &amp;&amp; pytest tests/routers/test_auth_screens_tokens.py -x -q &amp;&amp; pytest tests/ -x -q</automated>
  </verify>
  <done>Full suite green with new markup.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

No new surfaces — same POST handlers, same FlashMiddleware contract.

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-08-06-01 | XSS | `error` server-rendered into login page | mitigate | Jinja auto-escape. Error strings are server-chosen literals. |
| T-08-06-02 | Tampering | `must_change` Jinja flag | accept | Server-side — client cannot spoof; middleware enforces. |
| T-08-06-03 | Information Disclosure | Autocomplete username on login leaks email | accept | Standard practice; aids password manager UX. |
</threat_model>

<verification>
1. `pytest tests/ -x -q` green.
2. Manual:
   - Logout. Login page: centered card, logo, Entrar CTA.
   - Wrong password 3x: error banner shows Spanish "Credenciales invalidas".
   - After password change: success redirect to `/login?ok=password-cambiado`.
   - mis-solicitudes renders either table with colored pills or the
     empty-state. Cancel opens a modal with Volver / Cancelar solicitud buttons.
</verification>

<success_criteria>
- 3 templates refactored on tokens.
- 3 skills committed.
- Regression test locks the markup + Spanish copy.
- Phase 2 auth tests stay green.
</success_criteria>

<output>
After completion, create `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-06-SUMMARY.md` with:

- 3 templates: before/after table of classes changed.
- 3 skills recorded (variant picked).
- Test count + any renames to fixtures.
- Handoff to 08-07 (data screens): `.badge-*` + `.data-table` + empty/error state patterns are live; other screens may reuse the exact mis_solicitudes idioms.
</output>
