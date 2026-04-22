---
phase: 08-redise-o-ui-modo-claro-moderno
plan: 09b
type: execute
wave: 6
depends_on: [08-05]
files_modified:
  - templates/ajustes_auditoria.html
  - templates/ajustes_limites.html
  - .claude/skills/sketch-findings-ajustes-auditoria/SKILL.md
  - .claude/skills/sketch-findings-ajustes-limites/SKILL.md
  - tests/routers/test_ajustes_audit_limits_tokens.py
autonomous: true
gap_closure: false
requirements: [UIREDO-04, UIREDO-05, UIREDO-06]
tags: [ajustes, auditoria, limites, settings, phase-8, ui]
user_setup: []

must_haves:
  truths:
    - "`templates/ajustes_auditoria.html` refactored: filters sidebar (user, fecha, path, status) + paginated table + CSV export button. Sticky table header. Tabular-nums on timestamps."
    - "`templates/ajustes_limites.html` refactored: table of thresholds with inline edit. Recalcular factor button at top."
    - "Phase 5 RBAC preserved verbatim on `auditoria:read` + `limites:manage`."
    - "Breadcrumbs present on both sub-pages: `Ajustes / Auditoría`, `Ajustes / Límites`."
    - "2 skills with `selected_by: claude_auto`."
    - "No files_modified overlap with 08-09a, 08-09c, 08-08a, 08-08b, 08-06, 08-07 — all Wave 6 plans remain parallelizable."
  artifacts:
    - path: "templates/ajustes_auditoria.html"
      provides: "Refactored filters + paginated table + export"
      contains: "data-table"
    - path: "templates/ajustes_limites.html"
      provides: "Refactored thresholds table + recalcular button"
      contains: "Recalcular factor"
    - path: "tests/routers/test_ajustes_audit_limits_tokens.py"
      provides: "Regression — routes render, breadcrumbs, RBAC preserved, no raw state colors"
      contains: "def test_ajustes_limites_has_recalcular"
  key_links:
    - from: "templates/ajustes_auditoria.html"
      to: "GET /ajustes/auditoria/export (CSV)"
      via: "form action or Alpine fetch"
      pattern: "/ajustes/auditoria"
---

<objective>
Re-skin /ajustes/auditoria + /ajustes/limites sub-pages on Phase 8
tokens.

Purpose (split from original 08-09 per checker): splitting the large
ajustes suite refactor into three smaller plans keeps each plan within
the 2-3 task / <50% context budget and preserves Wave 6 parallel
execution.

Output: 2 templates refactored + 2 skills + 1 regression test file.
Phase 5 RBAC preserved.
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
@templates/ajustes_auditoria.html
@templates/ajustes_limites.html
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
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Per-Screen Adaptations" rows 08-22 (auditoría), 08-23 (límites).
  </read_first>
  <action>
Create 2 skill files with `selected_by: claude_auto`.

- **ajustes-auditoria** — 3 variants; pick v2 (filters collapsible sidebar + table right).
- **ajustes-limites** — 3 variants; pick v1 (flat table with inline edit).

Each SKILL.md notes:

- "Breadcrumbs `Ajustes / <Sub-page>` preserved."
- "Sticky table header on auditoría for long result sets."
- "Tabular-nums on timestamps + threshold cells."
  </action>
  <acceptance_criteria>
    - 2 skill files exist with `selected_by: claude_auto`.
  </acceptance_criteria>
  <verify>
    <automated>for s in ajustes-auditoria ajustes-limites; do test -f ".claude/skills/sketch-findings-${s}/SKILL.md" &amp;&amp; grep -q "selected_by: claude_auto" ".claude/skills/sketch-findings-${s}/SKILL.md" || exit 1; done &amp;&amp; echo OK</automated>
  </verify>
  <done>2 skills recorded.</done>
</task>

<task type="auto">
  <name>Task 2: Refactor 2 templates — class migration + breadcrumbs + sticky header + inline edits</name>
  <read_first>
    - Each template in full.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Breadcrumbs" + §"Per-Screen Adaptations" rows.
  </read_first>
  <action>
Apply the same class migration table as Plans 08-07 / 08-08a Task 2.

### Breadcrumbs

At the top of each `{% block content %}`:

```jinja
<div class="breadcrumbs">
  <a href="/ajustes">Ajustes</a>
  <span class="sep">/</span>
  <span class="current">Auditoría</span>   {# or "Límites" #}
</div>
```

### Per-sub-page specific rules

- **ajustes_auditoria.html** — filters collapsible sidebar on desktop,
  top collapse on mobile. Sticky table header
  (`sticky top-0 z-sticky bg-surface-subtle`). Tabular-nums on
  timestamp cells (`class="numeric"` which app.css already styles).

- **ajustes_limites.html** — table of thresholds. Inline edit via
  `<input class="input-inline">` per cell. Save per row (simple, no
  modal). Recalcular factor button in the top-right `btn btn-secondary`.

**Preserve verbatim:**

- Alpine + HTMX attributes.
- Form actions.
- Phase 5 `{% if can() %}` wrappers.
- `showToast` 3-arg calls.
  </action>
  <acceptance_criteria>
    - For each of the 2 templates, `grep -cE "bg-(red|green|blue|yellow|amber)-[0-9]{3}" <template>` returns 0.
    - `grep -c "breadcrumbs" templates/ajustes_auditoria.html templates/ajustes_limites.html` returns 2 or more.
    - `grep -c "Recalcular factor\|Recalcular" templates/ajustes_limites.html` returns 1 or more.
    - `grep -c "data-table" templates/ajustes_auditoria.html` returns 1 or more.
  </acceptance_criteria>
  <verify>
    <automated>! grep -qE "bg-(red|green|blue|yellow|amber)-[0-9]{3}" templates/ajustes_auditoria.html templates/ajustes_limites.html &amp;&amp; grep -q "breadcrumbs" templates/ajustes_auditoria.html &amp;&amp; grep -q "Recalcular" templates/ajustes_limites.html</automated>
  </verify>
  <done>2 templates refactored + breadcrumbs + sticky header + inline edits.</done>
</task>

<task type="auto">
  <name>Task 3: Regression test</name>
  <read_first>
    - `tests/routers/` patterns.
  </read_first>
  <action>
Create `tests/routers/test_ajustes_audit_limits_tokens.py`:

```python
"""Regression for Phase 8 / Plan 08-09b: audit + limits sub-pages refactor.

Locks:
- Each sub-page renders for propietario.
- No raw Tailwind state colors.
- Breadcrumbs present.
- Recalcular button + data-table present.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_TEMPLATES = Path(__file__).resolve().parents[2] / "templates"


TEMPLATES_IN_SCOPE = [
    "ajustes_auditoria.html",
    "ajustes_limites.html",
]


@pytest.mark.parametrize("template", TEMPLATES_IN_SCOPE)
def test_no_raw_state_colors(template: str):
    src = (_TEMPLATES / template).read_text(encoding="utf-8")
    assert not re.search(r"bg-(red|green|blue|yellow|amber)-[0-9]{3}", src)


@pytest.mark.parametrize("template", TEMPLATES_IN_SCOPE)
def test_sub_page_has_breadcrumbs(template: str):
    src = (_TEMPLATES / template).read_text(encoding="utf-8")
    assert "breadcrumbs" in src


def test_ajustes_limites_has_recalcular():
    src = (_TEMPLATES / "ajustes_limites.html").read_text(encoding="utf-8")
    assert "Recalcular" in src


def test_ajustes_auditoria_has_data_table():
    src = (_TEMPLATES / "ajustes_auditoria.html").read_text(encoding="utf-8")
    assert "data-table" in src


@pytest.mark.parametrize("path", [
    "/ajustes/auditoria", "/ajustes/limites",
])
def test_sub_page_renders_for_propietario(propietario_client, path: str):
    resp = propietario_client.get(path)
    assert resp.status_code == 200
```
  </action>
  <acceptance_criteria>
    - `pytest tests/routers/test_ajustes_audit_limits_tokens.py -x -q` exit 0.
    - `pytest tests/ -x -q` exit 0.
    - `ruff check tests/routers/test_ajustes_audit_limits_tokens.py` exit 0.
  </acceptance_criteria>
  <verify>
    <automated>ruff check tests/routers/test_ajustes_audit_limits_tokens.py &amp;&amp; ruff format --check tests/routers/test_ajustes_audit_limits_tokens.py &amp;&amp; pytest tests/routers/test_ajustes_audit_limits_tokens.py -x -q &amp;&amp; pytest tests/ -x -q</automated>
  </verify>
  <done>Full suite green.</done>
</task>

</tasks>

<threat_model>
## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-08-09b-01 | Elevation of Privilege | Accidental removal of `auditoria:read` gate | mitigate | Route-level RBAC check preserved; audit log shown only on protected route. |
| T-08-09b-02 | Information Disclosure | Audit log export leaks PII | accept | Phase 5 already gates the route to propietario; export is by design visible. |
</threat_model>

<verification>
1. `pytest tests/` green.
2. Manual: login as propietario, visit /ajustes/auditoria + /ajustes/limites. Confirm sticky header on auditoría + Recalcular button on límites.
</verification>

<success_criteria>
- 2 templates refactored.
- 2 skills recorded.
- Breadcrumbs on both sub-pages.
- RBAC preserved.
</success_criteria>

<output>
Create `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-09b-SUMMARY.md`.
</output>
