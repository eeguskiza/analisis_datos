---
phase: 08-redise-o-ui-modo-claro-moderno
plan: 09c
type: execute
wave: 6
depends_on: [08-05]
files_modified:
  - templates/ajustes_rendimiento.html
  - templates/ajustes_solicitudes.html
  - .claude/skills/sketch-findings-ajustes-rendimiento/SKILL.md
  - .claude/skills/sketch-findings-ajustes-solicitudes/SKILL.md
  - tests/routers/test_ajustes_rendimiento_solicitudes_tokens.py
autonomous: true
gap_closure: false
requirements: [UIREDO-04, UIREDO-05, UIREDO-06]
tags: [ajustes, rendimiento, solicitudes, approvals, chart-js, phase-8, ui]
user_setup: []

must_haves:
  truths:
    - "`templates/ajustes_rendimiento.html` refactored: filters + Chart.js canvas + summary table. Chart.js series colors pulled from tokens (`--color-primary`, `--color-info`, `--color-warn`, `--color-success`)."
    - "`templates/ajustes_solicitudes.html` refactored: table of pending approvals with approve/reject actions; row expand shows request details; both actions open confirmation modals."
    - "Phase 4 approval flow + Phase 5 RBAC preserved verbatim (`rendimiento:read` + `aprobaciones:manage`)."
    - "Breadcrumbs present on both sub-pages: `Ajustes / Rendimiento`, `Ajustes / Solicitudes`."
    - "2 skills with `selected_by: claude_auto`."
    - "No files_modified overlap with 08-09a, 08-09b, 08-08a, 08-08b, 08-06, 08-07 — all Wave 6 plans remain parallelizable."
  artifacts:
    - path: "templates/ajustes_rendimiento.html"
      provides: "Refactored filters + Chart.js canvas + summary"
      contains: "Chart("
    - path: "templates/ajustes_solicitudes.html"
      provides: "Refactored approvals table + approve/reject modals"
      contains: "Aprobar solicitud"
    - path: "tests/routers/test_ajustes_rendimiento_solicitudes_tokens.py"
      provides: "Regression — routes render, breadcrumbs, Chart.js tokenization, approval CTAs preserved, no raw state colors"
      contains: "def test_ajustes_solicitudes_has_aprobar_label"
  key_links:
    - from: "templates/ajustes_solicitudes.html"
      to: "POST /api/approvals/<id>/approve, POST /api/approvals/<id>/reject"
      via: "form actions — unchanged"
      pattern: "/api/approvals/"
---

<objective>
Re-skin /ajustes/rendimiento + /ajustes/solicitudes sub-pages on Phase
8 tokens. Includes Chart.js color tokenization for the rendimiento
chart and preserves the Phase 4 approval flow on solicitudes.

Purpose (split from original 08-09 per checker): third of three sub-
plans. Keeps each plan within the 2-3 task / <50% context budget and
preserves Wave 6 parallel execution.

Output: 2 templates refactored + 2 skills + 1 regression test file.
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
  <name>Task 1: Create 2 sketch-findings skills</name>
  <read_first>
    - Each of the 2 templates in full.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Per-Screen Adaptations" rows 08-24 (rendimiento), 08-25 (solicitudes).
  </read_first>
  <action>
Create 2 skill files with `selected_by: claude_auto`.

- **ajustes-rendimiento** — 3 variants; pick v2 (filter bar top + chart left + summary table right).
- **ajustes-solicitudes** — 3 variants; pick v1 (table + row expand + approve/reject modals).

Each SKILL.md notes:

- "Breadcrumbs `Ajustes / <Sub-page>` preserved."
- "Confirmation modals for destructive actions (reject approval) use the UI-SPEC §Destructive actions pattern."
- "Chart.js color palette in ajustes_rendimiento pulls from tokens via JS: `getComputedStyle(document.documentElement).getPropertyValue('--color-primary').trim()` → converted to `rgb(R,G,B)` strings (the token is `R G B` space-separated; JS splits + rejoins with commas + wraps)."
  </action>
  <acceptance_criteria>
    - 2 skill files exist with `selected_by: claude_auto`.
  </acceptance_criteria>
  <verify>
    <automated>for s in ajustes-rendimiento ajustes-solicitudes; do test -f ".claude/skills/sketch-findings-${s}/SKILL.md" &amp;&amp; grep -q "selected_by: claude_auto" ".claude/skills/sketch-findings-${s}/SKILL.md" || exit 1; done &amp;&amp; echo OK</automated>
  </verify>
  <done>2 skills recorded.</done>
</task>

<task type="auto">
  <name>Task 2: Refactor 2 templates — class migration + breadcrumbs + Chart.js tokenization + approval modals</name>
  <read_first>
    - Each template in full.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Breadcrumbs" + §"Per-Screen Adaptations" rows + §"Destructive actions".
  </read_first>
  <action>
Apply the same class migration table as Plans 08-07 / 08-08a Task 2.

### Breadcrumbs

At the top of each `{% block content %}`:

```jinja
<div class="breadcrumbs">
  <a href="/ajustes">Ajustes</a>
  <span class="sep">/</span>
  <span class="current">Rendimiento</span>   {# or "Solicitudes" #}
</div>
```

### Per-sub-page specific rules

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

**Preserve verbatim on both sub-pages:**

- Alpine + HTMX attributes.
- Form actions.
- Phase 5 `{% if can() %}` wrappers.
- `showToast` 3-arg calls.
- Chart.js canvas IDs and Alpine component names.
  </action>
  <acceptance_criteria>
    - For each of the 2 templates, `grep -cE "bg-(red|green|blue|yellow|amber)-[0-9]{3}" <template>` returns 0.
    - `grep -c "breadcrumbs" templates/ajustes_rendimiento.html templates/ajustes_solicitudes.html` returns 2 or more.
    - `grep -c "tokenRgb\|NEXO_CHART_COLORS" templates/ajustes_rendimiento.html` returns 1 or more.
    - `grep -c "Aprobar solicitud" templates/ajustes_solicitudes.html` returns 1 or more.
  </acceptance_criteria>
  <verify>
    <automated>! grep -qE "bg-(red|green|blue|yellow|amber)-[0-9]{3}" templates/ajustes_rendimiento.html templates/ajustes_solicitudes.html &amp;&amp; grep -q "breadcrumbs" templates/ajustes_rendimiento.html &amp;&amp; grep -q "tokenRgb\|NEXO_CHART_COLORS" templates/ajustes_rendimiento.html &amp;&amp; grep -q "Aprobar solicitud" templates/ajustes_solicitudes.html</automated>
  </verify>
  <done>2 templates refactored + breadcrumbs + Chart.js tokenization + approval modals.</done>
</task>

<task type="auto">
  <name>Task 3: Regression test</name>
  <read_first>
    - `tests/routers/` patterns.
  </read_first>
  <action>
Create `tests/routers/test_ajustes_rendimiento_solicitudes_tokens.py`:

```python
"""Regression for Phase 8 / Plan 08-09c: rendimiento + solicitudes sub-pages.

Locks:
- Each sub-page renders for propietario.
- No raw Tailwind state colors.
- Breadcrumbs present.
- Chart.js color palette tokenized.
- Approval CTAs preserved.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_TEMPLATES = Path(__file__).resolve().parents[2] / "templates"


TEMPLATES_IN_SCOPE = [
    "ajustes_rendimiento.html",
    "ajustes_solicitudes.html",
]


@pytest.mark.parametrize("template", TEMPLATES_IN_SCOPE)
def test_no_raw_state_colors(template: str):
    src = (_TEMPLATES / template).read_text(encoding="utf-8")
    assert not re.search(r"bg-(red|green|blue|yellow|amber)-[0-9]{3}", src)


@pytest.mark.parametrize("template", TEMPLATES_IN_SCOPE)
def test_sub_page_has_breadcrumbs(template: str):
    src = (_TEMPLATES / template).read_text(encoding="utf-8")
    assert "breadcrumbs" in src


def test_ajustes_rendimiento_uses_token_colors():
    src = (_TEMPLATES / "ajustes_rendimiento.html").read_text(encoding="utf-8")
    assert "tokenRgb" in src or "NEXO_CHART_COLORS" in src


def test_ajustes_solicitudes_has_aprobar_label():
    src = (_TEMPLATES / "ajustes_solicitudes.html").read_text(encoding="utf-8")
    assert "Aprobar solicitud" in src


@pytest.mark.parametrize("path", [
    "/ajustes/rendimiento", "/ajustes/solicitudes",
])
def test_sub_page_renders_for_propietario(propietario_client, path: str):
    resp = propietario_client.get(path)
    assert resp.status_code == 200
```
  </action>
  <acceptance_criteria>
    - `pytest tests/routers/test_ajustes_rendimiento_solicitudes_tokens.py -x -q` exit 0.
    - `pytest tests/ -x -q` exit 0.
    - `ruff check tests/routers/test_ajustes_rendimiento_solicitudes_tokens.py` exit 0.
  </acceptance_criteria>
  <verify>
    <automated>ruff check tests/routers/test_ajustes_rendimiento_solicitudes_tokens.py &amp;&amp; ruff format --check tests/routers/test_ajustes_rendimiento_solicitudes_tokens.py &amp;&amp; pytest tests/routers/test_ajustes_rendimiento_solicitudes_tokens.py -x -q &amp;&amp; pytest tests/ -x -q</automated>
  </verify>
  <done>Full suite green.</done>
</task>

</tasks>

<threat_model>
## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-08-09c-01 | Elevation of Privilege | Accidental removal of `aprobaciones:manage` gate on approve/reject buttons | mitigate | Route-level RBAC check preserved; buttons only shown to gated users. |
| T-08-09c-02 | Information Disclosure | Approval request detail expansion exposes server params | accept | Phase 5 already gates the route to propietario; info is by design visible. |
</threat_model>

<verification>
1. `pytest tests/` green.
2. Manual: login as propietario, visit /ajustes/rendimiento (chart shows data with tokenized colors) + /ajustes/solicitudes (approve a solicitud → modal confirms → action persists).
</verification>

<success_criteria>
- 2 templates refactored.
- 2 skills recorded.
- Chart.js tokenized.
- Phase 4 approval flow operational.
- RBAC preserved.
</success_criteria>

<output>
Create `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-09c-SUMMARY.md`.
</output>
