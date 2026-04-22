---
phase: 08-redise-o-ui-modo-claro-moderno
plan: 09
type: execute
wave: 6
depends_on: [08-05]
files_modified:
  - templates/ajustes.html
  - templates/ajustes_conexion.html
  - templates/ajustes_usuarios.html
  - templates/ajustes_auditoria.html
  - templates/ajustes_limites.html
  - templates/ajustes_rendimiento.html
  - templates/ajustes_solicitudes.html
  - .claude/skills/sketch-findings-ajustes-hub/SKILL.md
  - .claude/skills/sketch-findings-ajustes-conexion/SKILL.md
  - .claude/skills/sketch-findings-ajustes-usuarios/SKILL.md
  - .claude/skills/sketch-findings-ajustes-auditoria/SKILL.md
  - .claude/skills/sketch-findings-ajustes-limites/SKILL.md
  - .claude/skills/sketch-findings-ajustes-rendimiento/SKILL.md
  - .claude/skills/sketch-findings-ajustes-solicitudes/SKILL.md
  - tests/routers/test_ajustes_screens_tokens.py
autonomous: true
gap_closure: false
requirements: [UIREDO-04, UIREDO-05, UIREDO-06]
tags: [ajustes, hub, sub-pages, settings, phase-8, ui]
user_setup: []

must_haves:
  truths:
    - "`templates/ajustes.html` hub refactored: grid of 6 cards (Conexión, Usuarios, Auditoría, Límites, Rendimiento, Solicitudes), each card gated by `can(current_user, <perm>)` even though the hub route is propietario-only (defensive consistency per UI-SPEC)."
    - "Each card shows: Heroicon outline, title, 1-sentence description, breadcrumb-style chevron."
    - "`templates/ajustes_conexion.html` refactored: two cards — Conexión actual (read-only status) + Editar credenciales form (propietario-only `conexion:config`). Form change triggers confirmation modal."
    - "`templates/ajustes_usuarios.html` refactored: table of users + Crear usuario CTA + row actions (edit, deactivate). Edit opens dialog; deactivate opens destructive confirmation modal. This plan ALSO ships the `nombre` field from Plan 08-03 Task 2 (was already done in 08-03 but MUST be present now — verify)."
    - "`templates/ajustes_auditoria.html` refactored: filters sidebar (user, fecha, path, status) + paginated table + CSV export button. Sticky table header. Tabular-nums on timestamps."
    - "`templates/ajustes_limites.html` refactored: table of thresholds with inline edit. Recalcular factor button at top."
    - "`templates/ajustes_rendimiento.html` refactored: filters + Chart.js canvas + summary table. Chart.js series colors pulled from tokens (`--color-primary`, `--color-info`, `--color-warn`, `--color-success`)."
    - "`templates/ajustes_solicitudes.html` refactored: table of pending approvals with approve/reject actions; row expand shows request details; both actions open confirmation modals."
    - "Phase 5 RBAC + Phase 4 approval flow preserved verbatim."
    - "Breadcrumbs present on every sub-page: `Ajustes / <Sub-page>`."
    - "7 skills with `selected_by: claude_auto`."
  artifacts:
    - path: "templates/ajustes.html"
      provides: "Refactored hub with 6-card grid gated by can()"
      contains: "ajustes:manage"
    - path: "templates/ajustes_conexion.html"
      provides: "Refactored — status card + edit credentials form"
      contains: "conexion:config"
    - path: "templates/ajustes_usuarios.html"
      provides: "Refactored table + Crear usuario CTA; nombre field from 08-03 present"
      contains: "Crear usuario"
    - path: "templates/ajustes_auditoria.html"
      provides: "Refactored filters + paginated table + export"
      contains: "data-table"
    - path: "templates/ajustes_limites.html"
      provides: "Refactored thresholds table + recalcular button"
      contains: "Recalcular factor"
    - path: "templates/ajustes_rendimiento.html"
      provides: "Refactored filters + Chart.js canvas + summary"
      contains: "Chart("
    - path: "templates/ajustes_solicitudes.html"
      provides: "Refactored approvals table + approve/reject modals"
      contains: "Aprobar solicitud"
    - path: "tests/routers/test_ajustes_screens_tokens.py"
      provides: "Regression — each sub-page renders, breadcrumb present, RBAC preserved, no raw state colors"
      contains: "def test_ajustes_hub_has_six_cards"
  key_links:
    - from: "templates/ajustes.html"
      to: "6 sub-routes under /ajustes/*"
      via: "href links on cards"
      pattern: "href=\"/ajustes/"
    - from: "templates/ajustes_solicitudes.html"
      to: "POST /api/approvals/<id>/approve, POST /api/approvals/<id>/reject"
      via: "form actions — unchanged"
      pattern: "/api/approvals/"
---

<objective>
Re-skin the /ajustes hub + 6 sub-pages on Phase 8 tokens. The sub-pages
share form + table patterns; they can all be refactored together in
one plan because the Phase 5 split already isolated each into its own
template, so file ownership is clean.

Output: 7 templates refactored + 7 skills + 1 regression test file.
Phase 4 approval flow + Phase 5 RBAC preserved.
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
@templates/ajustes.html
@templates/ajustes_conexion.html
@templates/ajustes_usuarios.html
@templates/ajustes_auditoria.html
@templates/ajustes_limites.html
@templates/ajustes_rendimiento.html
@templates/ajustes_solicitudes.html
@templates/base.html
@static/css/app.css
@api/routers/pages.py
@nexo/services/auth.py
@CLAUDE.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create 7 sketch-findings skills</name>
  <read_first>
    - Each of the 7 templates in full.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Per-Screen Adaptations" rows 08-19 through 08-25 + §"Breadcrumbs (only on ajustes sub-pages)" + §"Open Questions" item 2 (sub-nav treatment).
  </read_first>
  <action>
Create 7 skill files with `selected_by: claude_auto`.

- **ajustes-hub** — 4 variants; pick v1 (6-card grid with icons + descriptions).
- **ajustes-conexion** — 3 variants; pick v2 (status card top + edit form below).
- **ajustes-usuarios** — 3 variants; pick v1 (table + Crear usuario CTA).
- **ajustes-auditoria** — 3 variants; pick v2 (filters collapsible sidebar + table right).
- **ajustes-limites** — 3 variants; pick v1 (flat table with inline edit).
- **ajustes-rendimiento** — 3 variants; pick v2 (filter bar top + chart left + summary table right).
- **ajustes-solicitudes** — 3 variants; pick v1 (table + row expand + approve/reject modals).

Each SKILL.md notes:

- "Breadcrumbs `Ajustes / <Sub-page>` preserved on every sub-page."
- "Configuración drawer section (Plan 08-02) surfaces these; cards on the hub are a second navigation axis — both must keep working (D-discretion resolution: flat items in drawer + card grid on hub)."
- "Confirmation modals for destructive actions (deactivate user, reject approval) use the UI-SPEC §Destructive actions pattern."
- "Chart.js color palette in ajustes_rendimiento pulls from tokens via JS: `getComputedStyle(document.documentElement).getPropertyValue('--color-primary').trim()` → converted to `rgb(R,G,B)` strings (the token is `R G B` space-separated; JS splits + rejoins with commas + wraps)."
  </action>
  <acceptance_criteria>
    - 7 skill files exist with `selected_by: claude_auto`.
  </acceptance_criteria>
  <verify>
    <automated>for s in ajustes-hub ajustes-conexion ajustes-usuarios ajustes-auditoria ajustes-limites ajustes-rendimiento ajustes-solicitudes; do test -f ".claude/skills/sketch-findings-${s}/SKILL.md" &amp;&amp; grep -q "selected_by: claude_auto" ".claude/skills/sketch-findings-${s}/SKILL.md" || exit 1; done &amp;&amp; echo OK</automated>
  </verify>
  <done>7 skills recorded.</done>
</task>

<task type="auto">
  <name>Task 2: Refactor 7 templates — class migration + breadcrumbs + hub 6-card grid</name>
  <read_first>
    - Each template in full.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Breadcrumbs" + §"Per-Screen Adaptations" rows.
  </read_first>
  <action>
Apply the same class migration table as Plans 08-07 / 08-08 Task 2.

### Ajustes hub (`templates/ajustes.html`)

Replace the current hub content with a 6-card grid. Each card gated by
`can(current_user, <perm>)`. Example structure for a single card:

```jinja
{% if can(current_user, 'conexion:config') %}
<a href="/ajustes/conexion" class="card hover:shadow-card transition-base ease-standard">
  <div class="card-body">
    <div class="flex items-start gap-3">
      <span class="btn-icon bg-primary-subtle text-primary shrink-0">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4"/>
        </svg>
      </span>
      <div class="min-w-0 flex-1">
        <h3 class="text-subtitle text-heading">Conexión</h3>
        <p class="text-sm text-muted mt-1">Estado y credenciales del SQL Server MES/APP.</p>
      </div>
      <svg class="w-5 h-5 text-muted" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24" aria-hidden="true">
        <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7"/>
      </svg>
    </div>
  </div>
</a>
{% endif %}
```

Repeat for the other 5 cards:

| perm | href | title | description |
|------|------|-------|-------------|
| `usuarios:manage` | `/ajustes/usuarios` | Usuarios | `Crea, edita y desactiva cuentas.` |
| `auditoria:read` | `/ajustes/auditoria` | Auditoría | `Histórico de acciones registradas por el sistema.` |
| `limites:manage` | `/ajustes/limites` | Límites | `Umbrales de preflight por endpoint.` |
| `rendimiento:read` | `/ajustes/rendimiento` | Rendimiento | `Tiempos reales vs estimados y divergencias.` |
| `aprobaciones:manage` | `/ajustes/solicitudes` | Solicitudes | `Aprueba o rechaza consultas largas pendientes.` |

Wrap the grid:

```jinja
<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
  {# cards #}
</div>
```

### Each sub-page — add breadcrumbs at the top of `{% block content %}`

Example in `ajustes_conexion.html`:

```jinja
{% block content %}
<div class="breadcrumbs">
  <a href="/ajustes">Ajustes</a>
  <span class="sep">/</span>
  <span class="current">Conexión</span>
</div>
{# rest of the content #}
{% endblock %}
```

Same breadcrumb structure in each of the 6 sub-pages with the correct
final segment label (Conexión, Usuarios, Auditoría, Límites,
Rendimiento, Solicitudes).

### Per-sub-page specific rules

- **ajustes_conexion.html** — two cards (status read-only + edit form).
  Form submit opens confirmation modal (Alpine) before POSTing the
  changes. Preserve the existing `ajustesConexionPage()` Alpine
  component (Plan 05-04 renamed it).

- **ajustes_usuarios.html** — verify the `nombre` field from Plan
  08-03 Task 2 is present. Add breadcrumb. Refactor classes. Row
  actions: edit (opens dialog modal), deactivate (destructive
  confirmation modal with copy from UI-SPEC §Destructive actions row
  `Borrar usuario`).

- **ajustes_auditoria.html** — filters collapsible sidebar on desktop,
  top collapse on mobile. Sticky table header
  (`sticky top-0 z-sticky bg-surface-subtle`). Tabular-nums on
  timestamp cells (`class="numeric"` which app.css already styles).

- **ajustes_limites.html** — table of thresholds. Inline edit via
  `<input class="input-inline">` per cell. Save per row (simple, no
  modal). Recalcular factor button in the top-right `btn btn-secondary`.

- **ajustes_rendimiento.html** — Chart.js canvas. Re-initialise the
  color palette from tokens. Add a tiny helper at the top of the page
  script:

  ```js
  function tokenRgb(name) {
    const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    // raw = "R G B"
    const [r, g, b] = raw.split(/\s+/).map(Number);
    return `rgb(${r}, ${g}, ${b})`;
  }
  const NEXO_CHART_COLORS = [
    tokenRgb('--color-primary'),
    tokenRgb('--color-info'),
    tokenRgb('--color-warn'),
    tokenRgb('--color-success'),
    tokenRgb('--color-error'),
  ];
  ```

  Then pass these to Chart.js `backgroundColor`/`borderColor` arrays.
  Do NOT change the Chart.js config API, only swap the color arrays.

- **ajustes_solicitudes.html** — table of pending approvals. Approve
  action → confirmation modal with `Aprobar solicitud` primary CTA
  (per UI-SPEC Copywriting Contract). Reject action → modal with
  `Rechazar` primary CTA. Row expand shows request details
  (JSON → formatted table or key/value list). Preserve all existing
  HTMX / Alpine / form actions.

**Search-and-replace class migration:** apply the same table as Plans
08-07/08-08 to every sub-page.

**Preserve verbatim on every sub-page:**

- Alpine + HTMX attributes.
- Form actions.
- Phase 5 `{% if can() %}` wrappers.
- `showToast` 3-arg calls.
- Chart.js canvas IDs and Alpine component names.
  </action>
  <acceptance_criteria>
    - For each of the 7 templates, `grep -cE "bg-(red|green|blue|yellow|amber)-[0-9]{3}" <template>` returns 0.
    - `grep -c "breadcrumbs" templates/ajustes_conexion.html templates/ajustes_usuarios.html templates/ajustes_auditoria.html templates/ajustes_limites.html templates/ajustes_rendimiento.html templates/ajustes_solicitudes.html` returns 6 or more.
    - `grep -c "can(current_user, 'usuarios:manage')\|can(current_user, \"usuarios:manage\")" templates/ajustes.html` returns 1 or more.
    - `grep -c "can(current_user, 'aprobaciones:manage')\|can(current_user, \"aprobaciones:manage\")" templates/ajustes.html` returns 1 or more.
    - `grep -c "tokenRgb\|NEXO_CHART_COLORS" templates/ajustes_rendimiento.html` returns 1 or more.
    - `grep -c "Aprobar solicitud" templates/ajustes_solicitudes.html` returns 1 or more.
    - `grep -c "Recalcular factor\|Recalcular" templates/ajustes_limites.html` returns 1 or more.
    - `grep -c "Crear usuario" templates/ajustes_usuarios.html` returns 1 or more.
    - `grep -c "name=\"nombre\"" templates/ajustes_usuarios.html` returns 1 or more (from 08-03 migration).
  </acceptance_criteria>
  <verify>
    <automated>! grep -qE "bg-(red|green|blue|yellow|amber)-[0-9]{3}" templates/ajustes.html templates/ajustes_conexion.html templates/ajustes_usuarios.html templates/ajustes_auditoria.html templates/ajustes_limites.html templates/ajustes_rendimiento.html templates/ajustes_solicitudes.html &amp;&amp; grep -q "breadcrumbs" templates/ajustes_conexion.html &amp;&amp; grep -q "tokenRgb\|NEXO_CHART_COLORS" templates/ajustes_rendimiento.html</automated>
  </verify>
  <done>7 templates refactored + breadcrumbs + hub card grid + Chart.js color tokenization.</done>
</task>

<task type="auto">
  <name>Task 3: Regression test</name>
  <read_first>
    - `tests/routers/` patterns.
  </read_first>
  <action>
Create `tests/routers/test_ajustes_screens_tokens.py`:

```python
"""Regression for Phase 8 / Plan 08-09: /ajustes suite visual refactor.

Locks:
- Each sub-page renders for propietario.
- No raw Tailwind state colors.
- Breadcrumbs present on every sub-page.
- RBAC wrappers preserved on hub cards + sensitive actions.
- Spanish CTAs match Copywriting Contract.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_TEMPLATES = Path(__file__).resolve().parents[2] / "templates"


AJUSTES_TEMPLATES = [
    "ajustes.html",
    "ajustes_conexion.html",
    "ajustes_usuarios.html",
    "ajustes_auditoria.html",
    "ajustes_limites.html",
    "ajustes_rendimiento.html",
    "ajustes_solicitudes.html",
]


@pytest.mark.parametrize("template", AJUSTES_TEMPLATES)
def test_no_raw_state_colors(template: str):
    src = (_TEMPLATES / template).read_text(encoding="utf-8")
    assert not re.search(r"bg-(red|green|blue|yellow|amber)-[0-9]{3}", src)


@pytest.mark.parametrize("template", [
    "ajustes_conexion.html",
    "ajustes_usuarios.html",
    "ajustes_auditoria.html",
    "ajustes_limites.html",
    "ajustes_rendimiento.html",
    "ajustes_solicitudes.html",
])
def test_sub_page_has_breadcrumbs(template: str):
    src = (_TEMPLATES / template).read_text(encoding="utf-8")
    assert "breadcrumbs" in src


def test_ajustes_hub_has_six_cards():
    src = (_TEMPLATES / "ajustes.html").read_text(encoding="utf-8")
    for perm in (
        "conexion:config", "usuarios:manage", "auditoria:read",
        "limites:manage", "rendimiento:read", "aprobaciones:manage",
    ):
        assert perm in src, f"Hub missing gated card for {perm}"


def test_ajustes_rendimiento_uses_token_colors():
    src = (_TEMPLATES / "ajustes_rendimiento.html").read_text(encoding="utf-8")
    assert "tokenRgb" in src or "NEXO_CHART_COLORS" in src


def test_ajustes_solicitudes_has_aprobar_label():
    src = (_TEMPLATES / "ajustes_solicitudes.html").read_text(encoding="utf-8")
    assert "Aprobar solicitud" in src


def test_ajustes_limites_has_recalcular():
    src = (_TEMPLATES / "ajustes_limites.html").read_text(encoding="utf-8")
    assert "Recalcular" in src


def test_ajustes_usuarios_has_crear_usuario():
    src = (_TEMPLATES / "ajustes_usuarios.html").read_text(encoding="utf-8")
    assert "Crear usuario" in src


def test_ajustes_hub_renders(propietario_client):
    resp = propietario_client.get("/ajustes")
    assert resp.status_code == 200


def test_ajustes_hub_forbidden_for_usuario(usuario_client):
    resp = usuario_client.get("/ajustes", follow_redirects=False)
    # Phase 5: HTML → 302+cookie or 403 JSON
    assert resp.status_code in (302, 303, 403)


@pytest.mark.parametrize("path", [
    "/ajustes/conexion", "/ajustes/usuarios", "/ajustes/auditoria",
    "/ajustes/limites", "/ajustes/rendimiento", "/ajustes/solicitudes",
])
def test_sub_page_renders_for_propietario(propietario_client, path: str):
    resp = propietario_client.get(path)
    assert resp.status_code == 200
```
  </action>
  <acceptance_criteria>
    - `pytest tests/routers/test_ajustes_screens_tokens.py -x -q` exit 0.
    - `pytest tests/ -x -q` exit 0.
    - `ruff check tests/routers/test_ajustes_screens_tokens.py` exit 0.
  </acceptance_criteria>
  <verify>
    <automated>ruff check tests/routers/test_ajustes_screens_tokens.py &amp;&amp; ruff format --check tests/routers/test_ajustes_screens_tokens.py &amp;&amp; pytest tests/routers/test_ajustes_screens_tokens.py -x -q &amp;&amp; pytest tests/ -x -q</automated>
  </verify>
  <done>Full suite green.</done>
</task>

</tasks>

<threat_model>
## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-08-09-01 | Elevation of Privilege | Accidental removal of `auditoria:read` / `aprobaciones:manage` / `usuarios:manage` gate | mitigate | `test_ajustes_hub_has_six_cards` + `test_ajustes_hub_forbidden_for_usuario`. |
| T-08-09-02 | Information Disclosure | Approval request detail expansion exposes server params | accept | Phase 5 already gates the route to propietario; info is by design visible. |
</threat_model>

<verification>
1. `pytest tests/` green.
2. Manual: login as propietario, visit /ajustes → 6 cards. Click each → sub-page renders with breadcrumbs. Performance chart shows data with tokenized colors. Approve a solicitud → modal confirms → action persists.
</verification>

<success_criteria>
- 7 templates refactored.
- 7 skills recorded.
- Breadcrumbs ubiquitous.
- Chart.js tokenized.
- RBAC preserved.
- Phase 4 approval flow operational.
</success_criteria>

<output>
Create `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-09-SUMMARY.md`.
</output>
