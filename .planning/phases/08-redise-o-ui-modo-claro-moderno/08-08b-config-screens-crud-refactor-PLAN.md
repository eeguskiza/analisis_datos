---
phase: 08-redise-o-ui-modo-claro-moderno
plan: 08b
type: execute
wave: 6
depends_on: [08-05]
files_modified:
  - templates/recursos.html
  - templates/ciclos.html
  - templates/ciclos_calc.html
  - templates/plantillas.html
  - .claude/skills/sketch-findings-recursos/SKILL.md
  - .claude/skills/sketch-findings-ciclos/SKILL.md
  - .claude/skills/sketch-findings-ciclos-calc/SKILL.md
  - .claude/skills/sketch-findings-plantillas/SKILL.md
  - tests/routers/test_config_screens_crud_tokens.py
autonomous: true
gap_closure: false
requirements: [UIREDO-04, UIREDO-05, UIREDO-06]
tags: [recursos, ciclos, ciclos-calc, plantillas, cfg, crud, editors, phase-8, ui]
user_setup: []

must_haves:
  truths:
    - "`templates/recursos.html` + `templates/ciclos.html` refactored as CRUD list screens: table with inline edit, Nuevo/Sync buttons gated by `recursos:edit`/`ciclos:edit` (Phase 5 preserved). Edit opens dialog modal."
    - "`templates/ciclos_calc.html` refactored: input panel top (card), results panel below (card). No preflight — calculation is client-side."
    - "`templates/plantillas.html` refactored as read-only list with a `bg-info-subtle` banner pointing at Mark-IV REF-04 deferral."
    - "Phase 5 RBAC wrappers preserved verbatim on every sensitive button: Nuevo recurso, Sync, Borrar."
    - "`showToast` calls are 3-arg (Plan 08-02 migration invariant)."
    - "4 sketch skills written with `selected_by: claude_auto`."
    - "No files_modified overlap with 08-08a, 08-06, 08-07, 08-09a, 08-09b, 08-09c — all Wave 6 plans remain parallelizable."
  artifacts:
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
    - path: "tests/routers/test_config_screens_crud_tokens.py"
      provides: "Regression — routes render + RBAC wrappers + Spanish CTAs + no raw state colors"
      contains: "def test_recursos_nuevo_gated_by_recursos_edit"
  key_links:
    - from: "templates/recursos.html"
      to: "POST /recursos/edit (existing endpoint)"
      via: "form action — unchanged"
      pattern: "action=\"/recursos"
---

<objective>
Re-skin the 4 CRUD-shaped configuration / editor screens (recursos,
ciclos, ciclos_calc, plantillas) on Phase 8 tokens. All four share
CRUD table + form patterns.

Purpose (split from original 08-08 per checker): the original plan
touched 6 templates / 13 files. Splitting into 08-08a (filter/KPI) and
08-08b (CRUD) keeps each plan well-sized and preserves parallel Wave 6
execution.

Output: 4 templates + 4 skills + 1 test file. Phase 5 RBAC preserved.
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
  <name>Task 1: Create 4 sketch-findings skills</name>
  <read_first>
    - Each of the 4 templates in full.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Per-Screen Adaptations" rows for recursos, ciclos, ciclos_calc, plantillas.
  </read_first>
  <action>
Create 4 skill files with `selected_by: claude_auto`. Summary:

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

- `recursos`/`ciclos`: `can(current_user, 'recursos:edit')` / `ciclos:edit`
  wraps Nuevo + Sync + Borrar.
- `ciclos_calc`: client-side calculation — no preflight, no SSE.
- `plantillas`: read-only; add `bg-info-subtle` banner about REF-04.
  </action>
  <acceptance_criteria>
    - 4 skill files exist with `selected_by: claude_auto`.
  </acceptance_criteria>
  <verify>
    <automated>for s in recursos ciclos ciclos-calc plantillas; do test -f ".claude/skills/sketch-findings-${s}/SKILL.md" &amp;&amp; grep -q "selected_by: claude_auto" ".claude/skills/sketch-findings-${s}/SKILL.md" || exit 1; done &amp;&amp; echo OK</automated>
  </verify>
  <done>4 skills committed.</done>
</task>

<task type="auto">
  <name>Task 2: Refactor 4 CRUD templates — apply class migration table from Plan 08-07</name>
  <read_first>
    - Each template in full.
    - `templates/recursos.html` as the canonical CRUD table pattern (post-refactor).
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Per-Screen Adaptations" for specific concerns per screen.
  </read_first>
  <action>
Apply the same search-and-replace class migration table as Plan 08-07 Task 2 to all 4 templates:

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
`can()` wrappers, showToast 3-arg calls, any `x-ref`/`x-id` identifiers,
route paths.
  </action>
  <acceptance_criteria>
    - For each of the 4 templates, `grep -cE "bg-(red|green|blue|yellow|amber)-[0-9]{3}" <template>` returns 0.
    - `grep -c "can(current_user, 'recursos:edit')" templates/recursos.html` returns 1 or more.
    - `grep -c "can(current_user, 'ciclos:edit')" templates/ciclos.html` returns 1 or more.
    - `grep -c "Calcular ciclos" templates/ciclos_calc.html` returns 1 or more.
    - `grep -c "REF-04" templates/plantillas.html` returns 1.
    - `grep -rn "showToast(" templates/recursos.html templates/ciclos.html templates/ciclos_calc.html templates/plantillas.html | grep -vE "showToast\((['\"])(info|success|warn|error)\\1" | wc -l` returns 0.
  </acceptance_criteria>
  <verify>
    <automated>! grep -qE "bg-(red|green|blue|yellow|amber)-[0-9]{3}" templates/recursos.html templates/ciclos.html templates/ciclos_calc.html templates/plantillas.html &amp;&amp; grep -q "can(current_user, 'recursos:edit')" templates/recursos.html &amp;&amp; grep -q "can(current_user, 'ciclos:edit')" templates/ciclos.html &amp;&amp; grep -q "REF-04" templates/plantillas.html</automated>
  </verify>
  <done>4 CRUD templates re-skinned. RBAC intact. Plantillas banner documents the Mark-IV deferral.</done>
</task>

<task type="auto">
  <name>Task 3: Regression test — routes, RBAC, Spanish copy</name>
  <read_first>
    - `tests/routers/` patterns + existing Phase 5 RBAC tests (many already exist for these screens).
  </read_first>
  <action>
Create `tests/routers/test_config_screens_crud_tokens.py`:

```python
"""Regression for Phase 8 / Plan 08-08b: CRUD config/editor screens refactor.

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
    "recursos.html", "ciclos.html", "ciclos_calc.html", "plantillas.html",
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


def test_ciclos_calc_has_primary_cta():
    src = (_TEMPLATES / "ciclos_calc.html").read_text(encoding="utf-8")
    assert "Calcular ciclos" in src


def test_plantillas_mark_iv_banner():
    src = (_TEMPLATES / "plantillas.html").read_text(encoding="utf-8")
    assert "REF-04" in src
    assert "Mark-IV" in src


def test_recursos_route_renders(propietario_client):
    resp = propietario_client.get("/recursos")
    assert resp.status_code == 200


def test_ciclos_calc_route_renders(propietario_client):
    resp = propietario_client.get("/ciclos-calc")
    assert resp.status_code == 200
```
  </action>
  <acceptance_criteria>
    - `pytest tests/routers/test_config_screens_crud_tokens.py -x -q` exit 0.
    - `pytest tests/ -x -q` exit 0.
    - `ruff check tests/routers/test_config_screens_crud_tokens.py` exit 0.
  </acceptance_criteria>
  <verify>
    <automated>ruff check tests/routers/test_config_screens_crud_tokens.py &amp;&amp; ruff format --check tests/routers/test_config_screens_crud_tokens.py &amp;&amp; pytest tests/routers/test_config_screens_crud_tokens.py -x -q &amp;&amp; pytest tests/ -x -q</automated>
  </verify>
  <done>Full suite green.</done>
</task>

</tasks>

<threat_model>
## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-08-08b-01 | Elevation of Privilege | Accidental removal of `recursos:edit` gate during refactor | mitigate | Static grep test `test_recursos_nuevo_gated_by_recursos_edit`. |
| T-08-08b-02 | Elevation of Privilege | Accidental removal of `ciclos:edit` gate during refactor | mitigate | Static grep test `test_ciclos_gated_by_ciclos_edit`. |
</threat_model>

<verification>
1. `pytest tests/` green.
2. Manual: visit each screen; confirm Nuevo/Sync/Borrar buttons hidden for non-permitted users.
</verification>

<success_criteria>
- 4 templates refactored on tokens.
- RBAC gates preserved.
- 4 skills recorded.
</success_criteria>

<output>
Create `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-08b-SUMMARY.md`.
</output>
