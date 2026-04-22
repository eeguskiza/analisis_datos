---
phase: 08-redise-o-ui-modo-claro-moderno
plan: 08
type: execute
wave: 6
depends_on: [08-05]
files_modified:
  - templates/capacidad.html
  - templates/operarios.html
  - templates/recursos.html
  - templates/ciclos.html
  - templates/ciclos_calc.html
  - templates/plantillas.html
  - .claude/skills/sketch-findings-capacidad/SKILL.md
  - .claude/skills/sketch-findings-operarios/SKILL.md
  - .claude/skills/sketch-findings-recursos/SKILL.md
  - .claude/skills/sketch-findings-ciclos/SKILL.md
  - .claude/skills/sketch-findings-ciclos-calc/SKILL.md
  - .claude/skills/sketch-findings-plantillas/SKILL.md
  - tests/routers/test_config_screens_tokens.py
autonomous: true
gap_closure: false
requirements: [UIREDO-04, UIREDO-05, UIREDO-06]
tags: [capacidad, operarios, recursos, ciclos, ciclos-calc, plantillas, cfg, editors, phase-8, ui]
user_setup: []

must_haves:
  truths:
    - "`templates/capacidad.html` + `templates/operarios.html` refactored: filters (date range, section), KPI cards row (capacidad only), data table, preflight amber/red for ranges > 90 days on both (preserve Alpine)."
    - "`templates/recursos.html` + `templates/ciclos.html` refactored as CRUD list screens: table with inline edit, Nuevo/Sync buttons gated by `recursos:edit`/`ciclos:edit` (Phase 5 preserved). Edit opens dialog modal."
    - "`templates/ciclos_calc.html` refactored: input panel top (card), results panel below (card). No preflight — calculation is client-side."
    - "`templates/plantillas.html` refactored as read-only list with a `bg-info-subtle` banner pointing at Mark-IV REF-04 deferral."
    - "Phase 5 RBAC wrappers preserved verbatim on every sensitive button: Nuevo recurso, Sync, Borrar, Export CSV (operarios)."
    - "`showToast` calls are 3-arg (Plan 08-02 migration)."
    - "6 sketch skills written with `selected_by: claude_auto`."
  artifacts:
    - path: "templates/capacidad.html"
      provides: "Re-skinned capacidad screen: filters + KPI cards + data table + preflight"
      contains: "badge badge-"
    - path: "templates/operarios.html"
      provides: "Re-skinned operarios screen: filters + data table + CSV export gated by operarios:export"
      contains: "operarios:export"
    - path: "templates/recursos.html"
      provides: "Re-skinned recursos CRUD: table + Nuevo + Sync gated by recursos:edit"
      contains: "recursos:edit"
    - path: "templates/ciclos.html"
      provides: "Re-skinned ciclos CRUD — same shape as recursos"
      contains: "ciclos:edit"
    - path: "templates/ciclos_calc.html"
      provides: "Re-skinned ciclos-calc: input card + results card"
      contains: "Calcular ciclos"
    - path: "templates/plantillas.html"
      provides: "Read-only list with Mark-IV banner"
      contains: "Mark-IV"
    - path: "tests/routers/test_config_screens_tokens.py"
      provides: "Regression — route renders + RBAC wrappers + Spanish CTAs + no raw state colors"
      contains: "def test_recursos_nuevo_gated_by_recursos_edit"
  key_links:
    - from: "templates/recursos.html"
      to: "POST /recursos/edit (existing endpoint)"
      via: "form action — unchanged"
      pattern: "action=\"/recursos"
    - from: "templates/operarios.html"
      to: "GET /api/operarios/export (CSV)"
      via: "Alpine-triggered fetch"
      pattern: "operarios:export"
---

<objective>
Re-skin the 6 configuration / editor screens on Phase 8 tokens.
All six share CRUD + filter + table patterns; shipping them in one
plan reduces duplication of skill + test overhead.

Output: 6 templates + 6 skills + 1 test file. Phase 5 RBAC preserved.
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
@templates/capacidad.html
@templates/operarios.html
@templates/recursos.html
@templates/ciclos.html
@templates/ciclos_calc.html
@templates/plantillas.html
@templates/base.html
@static/css/app.css
@api/routers/pages.py
@nexo/services/auth.py
@CLAUDE.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create 6 sketch-findings skills</name>
  <read_first>
    - Each of the 6 templates in full.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Per-Screen Adaptations" rows 08-08 through 08-16.
  </read_first>
  <action>
Create 6 skill files. Each uses `selected_by: claude_auto`. Summary:

- **capacidad** — 3 variants; pick v2 (filters top, KPI row, table below).
- **operarios** — 3 variants; pick v1 (filters + table + CSV export button gated).
- **recursos** — 3 variants; pick v1 (table with inline edit dialog).
- **ciclos** — 3 variants; pick v1 (same shape as recursos).
- **ciclos_calc** — 3 variants; pick v2 (input card top + results card below).
- **plantillas** — 2 variants; pick v1 (read-only list + Mark-IV banner).

For each, write `.claude/skills/sketch-findings-<screen>/SKILL.md` with:
front-matter (`screen`, `template`, `route`, `variants_generated`,
`selected_variant`, `selected_by: claude_auto`, `selected_on: 2026-04-22`,
`reversible: true`), a "Chosen variant" section summarising the pick,
a "Token inventory" listing only semantic tokens, a "Component
inventory" listing only semantic classes (`.card`, `.data-table`,
`.btn-*`, `.badge-*`, `.empty-state`, `.input-inline`), and a
"Landmines" section noting:

- `capacidad`/`operarios`: preflight amber/red for > 90 days — preserve
  Alpine modal component.
- `recursos`/`ciclos`: `can(current_user, 'recursos:edit')` / `ciclos:edit`
  wraps Nuevo + Sync + Borrar.
- `ciclos_calc`: client-side calculation — no preflight, no SSE.
- `plantillas`: read-only; add `bg-info-subtle` banner about REF-04.
  </action>
  <acceptance_criteria>
    - 6 skill files exist with `selected_by: claude_auto`.
  </acceptance_criteria>
  <verify>
    <automated>for s in capacidad operarios recursos ciclos ciclos-calc plantillas; do test -f ".claude/skills/sketch-findings-${s}/SKILL.md" &amp;&amp; grep -q "selected_by: claude_auto" ".claude/skills/sketch-findings-${s}/SKILL.md" || exit 1; done &amp;&amp; echo OK</automated>
  </verify>
  <done>6 skills committed.</done>
</task>

<task type="auto">
  <name>Task 2: Refactor 6 templates — apply the class migration table from Plan 08-07</name>
  <read_first>
    - Each template in full.
    - `templates/historial.html` / `templates/recursos.html` as the canonical table pattern reference (post-08-07).
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Per-Screen Adaptations" for specific concerns per screen.
  </read_first>
  <action>
Apply the same search-and-replace class migration table as Plan 08-07 Task 2 to all 6 templates:

| Old | New |
|-----|-----|
| `bg-brand-800` | `bg-primary-active` or `bg-surface-base` (context-dependent) |
| `bg-brand-600` | `bg-primary` |
| `text-brand-*` | `text-primary` / `text-heading` |
| `bg-surface-50/100/200` | `bg-surface-app/subtle/muted` |
| `border-surface-200` | `border-subtle` |
| `text-gray-500/600/700` | `text-muted` / `text-body` / `text-heading` |
| `bg-green-100 text-green-800` | `badge badge-success` |
| `bg-red-100 text-red-800` | `badge badge-error` |
| `bg-yellow-100 text-yellow-800` | `badge badge-warn` |
| `rounded-2xl/xl` | `rounded-lg/md` |
| `shadow-sm` | `shadow-card` |
| `font-medium` | `font-semibold` |

**Screen-specific rules:**

- **capacidad.html** — add KPI cards row using `.stat-card` class (from 08-02 CSS). Preserve any preflight amber/red modal's Alpine API (if present) and ensure the modal container uses `bg-surface-base shadow-modal rounded-lg` (post-08-02 tokens). Button `Lanzar análisis` → `Ejecutar análisis` (consistency with pipeline per Copywriting Contract — unless the existing text is different and has route-level meaning; confirm and preserve).

- **operarios.html** — CSV export button wrapped with
  `{% if can(current_user, 'operarios:export') %}` (verify this wrapper
  exists; if missing from Phase 5, add it since it's part of the
  phase-5 invariant. Check `docs/AUTH_MODEL.md` / `PERMISSION_MAP` to
  confirm `operarios:export` is a real permission — per
  `nexo/services/auth.py` it exists).

- **recursos.html** — `Nuevo recurso` button → `{% if can(current_user, 'recursos:edit') %}`. `Sincronizar` button → `{% if can(current_user, 'recursos:edit') %}`. Edit / Borrar row actions similarly. Each edit opens a dialog modal (use the `.card-elevated` + backdrop Alpine pattern from Plan 08-06 `mis_solicitudes.html` confirmation modal).

- **ciclos.html** — mirror recursos structure (per UI-SPEC "Same shape as recursos"). `{% if can(current_user, 'ciclos:edit') %}` wrappers.

- **ciclos_calc.html** — two stacked cards. Calculation is client-side. Button `Calcular ciclos` → primary CTA. Replace any remaining 1-arg `showToast('Valor no valido', 'error')` patterns with 3-arg form (08-02 should have done this — verify grep + sanity fix if any remain).

- **plantillas.html** — Prepend a banner:
  ```jinja
  <div class="rounded-md bg-info-subtle text-info border border-subtle p-4 mb-4 text-sm" role="note">
    El CRUD de plantillas se implementará en Mark-IV (ver
    <code class="font-mono">REF-04</code> en
    <code>.planning/REQUIREMENTS.md</code>). Por ahora, la lista se muestra en
    modo solo-lectura.
  </div>
  ```

**Preserve:** Alpine components, HTMX attrs, form actions, Phase 5
`can()` wrappers, showToast 3-arg calls, `.log-console` class (n/a
here), any `x-ref`/`x-id` identifiers, route paths.
  </action>
  <acceptance_criteria>
    - For each of the 6 templates, `grep -cE "bg-(red|green|blue|yellow|amber)-[0-9]{3}" <template>` returns 0.
    - `grep -c "can(current_user, 'recursos:edit')" templates/recursos.html` returns 1 or more.
    - `grep -c "can(current_user, 'ciclos:edit')" templates/ciclos.html` returns 1 or more.
    - `grep -c "can(current_user, 'operarios:export')" templates/operarios.html` returns 1 or more.
    - `grep -c "Calcular ciclos" templates/ciclos_calc.html` returns 1 or more.
    - `grep -c "REF-04" templates/plantillas.html` returns 1.
    - `grep -rn "showToast(" templates/capacidad.html templates/operarios.html templates/recursos.html templates/ciclos.html templates/ciclos_calc.html templates/plantillas.html | grep -vE "showToast\((['\"])(info|success|warn|error)\\1" | wc -l` returns 0.
  </acceptance_criteria>
  <verify>
    <automated>! grep -qE "bg-(red|green|blue|yellow|amber)-[0-9]{3}" templates/capacidad.html templates/operarios.html templates/recursos.html templates/ciclos.html templates/ciclos_calc.html templates/plantillas.html &amp;&amp; grep -q "can(current_user, 'recursos:edit')" templates/recursos.html &amp;&amp; grep -q "can(current_user, 'ciclos:edit')" templates/ciclos.html &amp;&amp; grep -q "REF-04" templates/plantillas.html</automated>
  </verify>
  <done>6 templates re-skinned. RBAC intact. Plantillas banner documents the Mark-IV deferral.</done>
</task>

<task type="auto">
  <name>Task 3: Regression test — routes, RBAC, Spanish copy</name>
  <read_first>
    - `tests/routers/` patterns + existing Phase 5 RBAC tests (many already exist for these screens).
  </read_first>
  <action>
Create `tests/routers/test_config_screens_tokens.py`:

```python
"""Regression for Phase 8 / Plan 08-08: config/editor screens refactor.

Locks:
- No raw Tailwind state colors.
- Phase 5 `can()` wrappers preserved on sensitive buttons.
- Spanish CTA labels.
- Plantillas banner flags Mark-IV deferral.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_TEMPLATES = Path(__file__).resolve().parents[2] / "templates"


@pytest.mark.parametrize("template", [
    "capacidad.html", "operarios.html", "recursos.html",
    "ciclos.html", "ciclos_calc.html", "plantillas.html",
])
def test_no_raw_state_colors(template: str):
    src = (_TEMPLATES / template).read_text(encoding="utf-8")
    assert not re.search(r"bg-(red|green|blue|yellow|amber)-[0-9]{3}", src), (
        f"{template} still has raw state colors"
    )


def test_recursos_nuevo_gated_by_recursos_edit():
    src = (_TEMPLATES / "recursos.html").read_text(encoding="utf-8")
    assert "recursos:edit" in src


def test_ciclos_gated_by_ciclos_edit():
    src = (_TEMPLATES / "ciclos.html").read_text(encoding="utf-8")
    assert "ciclos:edit" in src


def test_operarios_export_gated():
    src = (_TEMPLATES / "operarios.html").read_text(encoding="utf-8")
    assert "operarios:export" in src


def test_ciclos_calc_has_primary_cta():
    src = (_TEMPLATES / "ciclos_calc.html").read_text(encoding="utf-8")
    assert "Calcular ciclos" in src


def test_plantillas_mark_iv_banner():
    src = (_TEMPLATES / "plantillas.html").read_text(encoding="utf-8")
    assert "REF-04" in src
    assert "Mark-IV" in src


def test_capacidad_route_renders(propietario_client):
    resp = propietario_client.get("/capacidad")
    assert resp.status_code == 200


def test_operarios_route_renders(propietario_client):
    resp = propietario_client.get("/operarios")
    assert resp.status_code == 200


def test_recursos_route_renders(propietario_client):
    resp = propietario_client.get("/recursos")
    assert resp.status_code == 200


def test_ciclos_calc_route_renders(propietario_client):
    resp = propietario_client.get("/ciclos-calc")
    assert resp.status_code == 200
```
  </action>
  <acceptance_criteria>
    - `pytest tests/routers/test_config_screens_tokens.py -x -q` exit 0.
    - `pytest tests/ -x -q` exit 0.
    - `ruff check tests/routers/test_config_screens_tokens.py` exit 0.
  </acceptance_criteria>
  <verify>
    <automated>ruff check tests/routers/test_config_screens_tokens.py &amp;&amp; ruff format --check tests/routers/test_config_screens_tokens.py &amp;&amp; pytest tests/routers/test_config_screens_tokens.py -x -q &amp;&amp; pytest tests/ -x -q</automated>
  </verify>
  <done>Full suite green.</done>
</task>

</tasks>

<threat_model>
## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-08-08-01 | Elevation of Privilege | Accidental removal of `recursos:edit` gate during refactor | mitigate | Static grep test `test_recursos_nuevo_gated_by_recursos_edit`. |
| T-08-08-02 | Elevation of Privilege | Accidental removal of `operarios:export` gate | mitigate | `test_operarios_export_gated`. |
</threat_model>

<verification>
1. `pytest tests/` green.
2. Manual: visit each screen; confirm buttons hidden for non-permitted users.
</verification>

<success_criteria>
- 6 templates refactored on tokens.
- RBAC gates preserved.
- 6 skills recorded.
</success_criteria>

<output>
Create `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-08-SUMMARY.md`.
</output>
