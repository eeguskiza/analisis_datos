---
phase: 08-redise-o-ui-modo-claro-moderno
plan: 08a
type: execute
wave: 6
depends_on: [08-05]
files_modified:
  - templates/capacidad.html
  - templates/operarios.html
  - .claude/skills/sketch-findings-capacidad/SKILL.md
  - .claude/skills/sketch-findings-operarios/SKILL.md
  - tests/routers/test_config_screens_filter_kpi_tokens.py
autonomous: true
gap_closure: false
requirements: [UIREDO-04, UIREDO-05, UIREDO-06]
tags: [capacidad, operarios, cfg, filter-kpi, phase-8, ui]
user_setup: []

must_haves:
  truths:
    - "`templates/capacidad.html` refactored: filter bar (date range, section), KPI cards row using `.stat-card`, data table, preflight amber/red for ranges > 90 days preserved (Alpine component API unchanged)."
    - "`templates/operarios.html` refactored: filters + data table + CSV export button gated by `{% if can(current_user, 'operarios:export') %}` (Phase 5 preserved)."
    - "Phase 5 RBAC wrappers preserved verbatim on every sensitive button: Export CSV (operarios)."
    - "`showToast` calls are 3-arg (Plan 08-02 migration invariant)."
    - "2 sketch skills (capacidad, operarios) written with `selected_by: claude_auto`."
    - "No files_modified overlap with 08-08b, 08-06, 08-07, 08-09a, 08-09b, 08-09c — all Wave 6 plans remain parallelizable."
  artifacts:
    - path: "templates/capacidad.html"
      provides: "Re-skinned capacidad screen: filters + KPI cards + data table + preflight"
      contains: "badge badge-"
    - path: "templates/operarios.html"
      provides: "Re-skinned operarios screen: filters + data table + CSV export gated by operarios:export"
      contains: "operarios:export"
    - path: "tests/routers/test_config_screens_filter_kpi_tokens.py"
      provides: "Regression — routes render + RBAC wrappers + Spanish CTAs + no raw state colors"
      contains: "def test_operarios_export_gated"
  key_links:
    - from: "templates/operarios.html"
      to: "GET /api/operarios/export (CSV)"
      via: "Alpine-triggered fetch"
      pattern: "operarios:export"
    - from: "templates/capacidad.html"
      to: "preflight modal Alpine component"
      via: "x-data component — preserved"
      pattern: "x-data"
---

<objective>
Re-skin the 2 filter/KPI configuration screens (capacidad + operarios)
on Phase 8 tokens. Both share the filter bar + tabular data pattern.

Purpose (split from original 08-08 per checker): the original plan
touched 6 templates / 13 files which exceeded the 2-3 task / <50%
context budget. Splitting into 08-08a (filter/KPI) and 08-08b (CRUD)
keeps each plan well-sized and preserves parallel Wave 6 execution.

Output: 2 templates + 2 skills + 1 test file. Phase 5 RBAC preserved.
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
@templates/base.html
@static/css/app.css
@api/routers/pages.py
@nexo/services/auth.py
@CLAUDE.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create 2 sketch-findings skills</name>
  <read_first>
    - Each of the 2 templates in full.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Per-Screen Adaptations" rows for capacidad + operarios.
  </read_first>
  <action>
Create 2 skill files with `selected_by: claude_auto`. Summary:

- **capacidad** — 3 variants; pick v2 (filters top, KPI row, table below).
- **operarios** — 3 variants; pick v1 (filters + table + CSV export button gated).

For each, write `.claude/skills/sketch-findings-<screen>/SKILL.md` with:
front-matter (`screen`, `template`, `route`, `variants_generated`,
`selected_variant`, `selected_by: claude_auto`, `selected_on: 2026-04-22`,
`reversible: true`), a "Chosen variant" section summarising the pick,
a "Token inventory" listing only semantic tokens, a "Component
inventory" listing only semantic classes (`.card`, `.data-table`,
`.btn-*`, `.badge-*`, `.empty-state`, `.stat-card`), and a
"Landmines" section noting:

- `capacidad`/`operarios`: preflight amber/red for > 90 days — preserve
  Alpine modal component.
- `operarios`: `can(current_user, 'operarios:export')` wraps CSV export button.
  </action>
  <acceptance_criteria>
    - 2 skill files exist with `selected_by: claude_auto`.
  </acceptance_criteria>
  <verify>
    <automated>for s in capacidad operarios; do test -f ".claude/skills/sketch-findings-${s}/SKILL.md" &amp;&amp; grep -q "selected_by: claude_auto" ".claude/skills/sketch-findings-${s}/SKILL.md" || exit 1; done &amp;&amp; echo OK</automated>
  </verify>
  <done>2 skills committed.</done>
</task>

<task type="auto">
  <name>Task 2: Refactor capacidad.html + operarios.html — apply class migration table from Plan 08-07</name>
  <read_first>
    - Each template in full.
    - `templates/historial.html` as the canonical table pattern reference (post-08-07).
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Per-Screen Adaptations" for specific concerns per screen.
  </read_first>
  <action>
Apply the same search-and-replace class migration table as Plan 08-07 Task 2 to both templates:

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

**Preserve:** Alpine components, HTMX attrs, form actions, Phase 5
`can()` wrappers, showToast 3-arg calls, `.log-console` class (n/a
here), any `x-ref`/`x-id` identifiers, route paths.
  </action>
  <acceptance_criteria>
    - For each of the 2 templates, `grep -cE "bg-(red|green|blue|yellow|amber)-[0-9]{3}" <template>` returns 0.
    - `grep -c "can(current_user, 'operarios:export')" templates/operarios.html` returns 1 or more.
    - `grep -rn "showToast(" templates/capacidad.html templates/operarios.html | grep -vE "showToast\((['\"])(info|success|warn|error)\\1" | wc -l` returns 0.
  </acceptance_criteria>
  <verify>
    <automated>! grep -qE "bg-(red|green|blue|yellow|amber)-[0-9]{3}" templates/capacidad.html templates/operarios.html &amp;&amp; grep -q "can(current_user, 'operarios:export')" templates/operarios.html</automated>
  </verify>
  <done>2 templates re-skinned. RBAC intact.</done>
</task>

<task type="auto">
  <name>Task 3: Regression test — routes, RBAC, Spanish copy</name>
  <read_first>
    - `tests/routers/` patterns + existing Phase 5 RBAC tests (many already exist for these screens).
  </read_first>
  <action>
Create `tests/routers/test_config_screens_filter_kpi_tokens.py`:

```python
"""Regression for Phase 8 / Plan 08-08a: filter/KPI config screens refactor.

Locks:
- No raw Tailwind state colors.
- Phase 5 `can()` wrappers preserved on sensitive buttons.
- Routes render.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_TEMPLATES = Path(__file__).resolve().parents[2] / "templates"


@pytest.mark.parametrize("template", [
    "capacidad.html", "operarios.html",
])
def test_no_raw_state_colors(template: str):
    src = (_TEMPLATES / template).read_text(encoding="utf-8")
    assert not re.search(r"bg-(red|green|blue|yellow|amber)-[0-9]{3}", src), (
        f"{template} still has raw state colors"
    )


def test_operarios_export_gated():
    src = (_TEMPLATES / "operarios.html").read_text(encoding="utf-8")
    assert "operarios:export" in src


def test_capacidad_route_renders(propietario_client):
    resp = propietario_client.get("/capacidad")
    assert resp.status_code == 200


def test_operarios_route_renders(propietario_client):
    resp = propietario_client.get("/operarios")
    assert resp.status_code == 200
```
  </action>
  <acceptance_criteria>
    - `pytest tests/routers/test_config_screens_filter_kpi_tokens.py -x -q` exit 0.
    - `pytest tests/ -x -q` exit 0.
    - `ruff check tests/routers/test_config_screens_filter_kpi_tokens.py` exit 0.
  </acceptance_criteria>
  <verify>
    <automated>ruff check tests/routers/test_config_screens_filter_kpi_tokens.py &amp;&amp; ruff format --check tests/routers/test_config_screens_filter_kpi_tokens.py &amp;&amp; pytest tests/routers/test_config_screens_filter_kpi_tokens.py -x -q &amp;&amp; pytest tests/ -x -q</automated>
  </verify>
  <done>Full suite green.</done>
</task>

</tasks>

<threat_model>
## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-08-08a-01 | Elevation of Privilege | Accidental removal of `operarios:export` gate during refactor | mitigate | Static grep test `test_operarios_export_gated`. |
| T-08-08a-02 | Tampering | Preflight modal Alpine API drift | mitigate | Alpine attributes preserved verbatim; manual smoke on > 90 days range. |
</threat_model>

<verification>
1. `pytest tests/` green.
2. Manual: visit each screen; confirm export button hidden for non-permitted users.
</verification>

<success_criteria>
- 2 templates refactored on tokens.
- RBAC gates preserved.
- 2 skills recorded.
</success_criteria>

<output>
Create `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-08a-SUMMARY.md`.
</output>
