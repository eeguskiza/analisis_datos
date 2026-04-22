---
phase: 08-redise-o-ui-modo-claro-moderno
plan: 07
type: execute
wave: 6
depends_on: [08-05]
files_modified:
  - templates/pipeline.html
  - templates/historial.html
  - templates/bbdd.html
  - templates/datos.html
  - templates/informes.html
  - .claude/skills/sketch-findings-pipeline/SKILL.md
  - .claude/skills/sketch-findings-historial/SKILL.md
  - .claude/skills/sketch-findings-bbdd/SKILL.md
  - .claude/skills/sketch-findings-datos/SKILL.md
  - .claude/skills/sketch-findings-informes/SKILL.md
  - tests/routers/test_data_screens_tokens.py
autonomous: true
gap_closure: false
requirements: [UIREDO-04, UIREDO-05, UIREDO-06]
tags: [pipeline, historial, bbdd, datos, informes, phase-8, ui]
user_setup: []

must_haves:
  truths:
    - "`templates/pipeline.html` refactored: filter form uses stacked labels + `(opcional)` where relevant; Run button is `.btn btn-primary btn-lg`; log console keeps its dark `.log-console` class (preserved per UI-SPEC); preflight amber/red modals preserve their existing Alpine component names + API (only inner chrome re-styles)."
    - "`templates/historial.html` refactored: filters collapse to a top filter bar above a `.data-table`; row actions (view PDFs, delete) gated by `{% if can(current_user, 'informes:delete') %}` preserved; destructive delete opens confirmation modal; empty state uses `archive-box` icon + `Sin ejecuciones todavía` copy."
    - "`templates/bbdd.html` refactored: query form top (stacked fields), Run button triggers preflight (existing Alpine component unchanged), results table below or empty state with `magnifying-glass` icon + `Sin resultados para esta consulta`."
    - "`templates/datos.html` refactored as simplest screen: single `.card` with `Refrescar datos` CTA + last-refresh timestamp."
    - "`templates/informes.html` refactored: grid of PDF cards; empty state `document` icon + `Aún no hay informes generados`."
    - "Phase 4 preflight modal Alpine APIs (`preflightModal`, event dispatch) are preserved verbatim — only CSS/class changes."
    - "Phase 5 RBAC preserved: every `{% if can(current_user, ...) %}` in these templates unchanged."
    - "All `showToast` calls are 3-arg (Plan 08-02 already migrated — confirm test)."
    - "Sketch skills exist for each screen with `selected_by: claude_auto`."
  artifacts:
    - path: "templates/pipeline.html"
      provides: "Re-skinned pipeline screen with log console preserved and preflight modal preserved"
      contains: "btn btn-primary btn-lg"
    - path: "templates/historial.html"
      provides: "Re-skinned historial table + filter bar + delete confirmation"
      contains: "data-table"
    - path: "templates/bbdd.html"
      provides: "Re-skinned BBDD query interface with preflight modal chrome"
      contains: "Lanzar consulta"
    - path: "templates/datos.html"
      provides: "Single card refresh screen"
      contains: "Refrescar datos"
    - path: "templates/informes.html"
      provides: "PDF grid with empty state"
      contains: "informe"
    - path: "tests/routers/test_data_screens_tokens.py"
      provides: "Regression — each route renders, no raw state colors, preflight Alpine component names present, RBAC gates preserved"
      contains: "def test_pipeline_preserves_preflight_modal + def test_historial_has_delete_gating"
  key_links:
    - from: "templates/pipeline.html"
      to: "Alpine preflightModal component in static/js/app.js"
      via: "x-data=\"preflightModal()\" attribute"
      pattern: "preflightModal\\(\\)"
    - from: "templates/historial.html"
      to: "POST /historial/<id>/delete"
      via: "form action"
      pattern: "action=\"/historial/"
---

<objective>
Re-skin the 5 data-heavy screens (pipeline, historial, bbdd, datos,
informes) on Phase 8 tokens + chrome, preserving all Alpine
component APIs from Phase 4 (preflightModal) and Phase 5 RBAC
wrappers. Each screen gets an auto-selected sketch-findings skill.

Output: 5 templates refactored + 5 skills + 1 regression test file.
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
@templates/pipeline.html
@templates/historial.html
@templates/bbdd.html
@templates/datos.html
@templates/informes.html
@templates/base.html
@static/css/app.css
@static/js/app.js
@api/routers/pages.py
@CLAUDE.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create 5 sketch-findings skills (auto-selected)</name>
  <read_first>
    - Each of the 5 templates in full.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Per-Screen Adaptations" rows for these 5 + §"Copywriting Contract" §"Empty state copy" family table.
  </read_first>
  <action>
Create 5 skill files (one per screen). Each uses `selected_by: claude_auto`.

### `.claude/skills/sketch-findings-pipeline/SKILL.md`
```markdown
---
screen: pipeline
template: templates/pipeline.html
route: /pipeline
variants_generated: 4
selected_variant: 2
selected_by: claude_auto
selected_on: 2026-04-22
reversible: true
---

# Sketch findings: pipeline

## Chosen variant
Variant 2 — Two-column split on desktop: filter form + Run button
on the left (sticky), log console + PDF list + OEE dashboard on the
right. Mobile collapses to a single column. Preflight modal preserved
with re-skinned chrome.

## Token inventory
`bg-surface-app`, `bg-surface-base`, `border-subtle`, `shadow-card`,
`text-heading`, `text-body`, `text-muted`, `bg-primary`,
`text-on-accent`, `bg-warn-subtle` (amber preflight), `bg-error-subtle`
(red preflight), `--shadow-modal`.

## Component inventory
`.card`, `.input-inline`, `.btn-primary`, `.btn-lg`, `.log-console`
(preserved), `.badge-warn`, `.badge-error`, `.loading-hero` (Tier 2
during pipeline run).

## Landmines
- Do NOT touch the Alpine `preflightModal()` API.
- `showToast` calls are 3-arg post 08-02 — grep confirms.
- The `.log-console` dark background is INTENTIONAL.
```

### `.claude/skills/sketch-findings-historial/SKILL.md`
```markdown
---
screen: historial
template: templates/historial.html
route: /historial
variants_generated: 4
selected_variant: 1
selected_by: claude_auto
selected_on: 2026-04-22
reversible: true
---

# Sketch findings: historial

## Chosen variant
Variant 1 — Filter bar on top + single `.data-table` below.
Click-to-expand row reveals PDF list inline. Destructive delete in
rightmost cell opens confirmation modal.

## Component inventory
`.data-table`, `.btn-ghost`, `.btn-danger`, `.badge-*`, `.empty-state`,
`.card-elevated` (modal container).

## Landmines
`{% if can(current_user, 'informes:delete') %}` MUST wrap the delete
button — Phase 5 invariant.
```

### `.claude/skills/sketch-findings-bbdd/SKILL.md`
```markdown
---
screen: bbdd
template: templates/bbdd.html
route: /bbdd
variants_generated: 4
selected_variant: 2
selected_by: claude_auto
selected_on: 2026-04-22
reversible: true
---

# Sketch findings: bbdd

## Chosen variant
Variant 2 — Query form top (stacked fields, `(opcional)` on limit),
Run button triggers preflight, results table below, query-history card
on the right sidebar (desktop) or collapsed on mobile.

## Component inventory
`.card`, `.input-inline`, `.btn-primary`, `.btn-lg`, `.data-table`,
`.empty-state` (magnifying-glass + "Sin resultados para esta consulta"),
`.tree-item` (history sidebar), preflightModal.

## Landmines
Preflight amber/red Alpine contract identical to pipeline.
```

### `.claude/skills/sketch-findings-datos/SKILL.md`
```markdown
---
screen: datos
template: templates/datos.html
route: /datos
variants_generated: 2
selected_variant: 1
selected_by: claude_auto
selected_on: 2026-04-22
reversible: true
---

# Sketch findings: datos

## Chosen variant
Variant 1 — Single centered `.card` with "Refrescar datos" CTA
(primary) + last-refresh timestamp in `text-muted tabular-nums`.

## Component inventory
`.card`, `.btn-primary`, `.btn-lg`, `.spinner-panel` (during refresh).
```

### `.claude/skills/sketch-findings-informes/SKILL.md`
```markdown
---
screen: informes
template: templates/informes.html
route: /informes (redirects to /historial) - but the template lives on for embedded use
variants_generated: 3
selected_variant: 2
selected_by: claude_auto
selected_on: 2026-04-22
reversible: true
---

# Sketch findings: informes

## Chosen variant
Variant 2 — Grid of PDF cards (`grid-cols-1 md:grid-cols-2 lg:grid-cols-3`).
Each card: document icon, filename truncated, generated-at date, size,
download + view buttons. Empty state follows UI-SPEC family.

## Component inventory
`.card`, `.btn-secondary`, `.btn-ghost`, `.empty-state` (document icon
+ "Aún no hay informes generados" + CTA to /pipeline gated by
`pipeline:run`).

## Landmines
`/informes` redirects to `/historial` per Phase 3 wiring; the template
is still rendered from historial for embedded contexts — leave the
redirect intact.
```
  </action>
  <acceptance_criteria>
    - The 5 skill files exist and each grep matches `selected_by: claude_auto`.
  </acceptance_criteria>
  <verify>
    <automated>for s in pipeline historial bbdd datos informes; do test -f ".claude/skills/sketch-findings-${s}/SKILL.md" &amp;&amp; grep -q "selected_by: claude_auto" ".claude/skills/sketch-findings-${s}/SKILL.md" || exit 1; done &amp;&amp; echo OK</automated>
  </verify>
  <done>5 skills recorded.</done>
</task>

<task type="auto">
  <name>Task 2: Refactor 5 templates — class migration, semantic badges, RBAC + preflight preservation</name>
  <read_first>
    - Each template in full.
    - `static/css/app.css` (from 08-02) — confirm class names.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Per-Screen Adaptations" rows.
  </read_first>
  <action>
For EACH of the 5 templates, apply the following migration rules.
**Preserve verbatim:** Alpine components (`x-data`, `x-init`, `x-show`,
`x-if`, `x-bind`, `x-on`, `x-transition` — except class lists inside
transitions, which can be updated), form actions, HTMX attributes
(`hx-get`, `hx-post`, `hx-trigger`, `hx-swap`), `{% if can() %}`
wrappers, showToast calls (already 3-arg), the `.log-console` class in
pipeline.html, route paths, and any `?approval_id=` query-string logic.

**Migrate (search-and-replace class attrs):**

| Old utility | New utility |
|-------------|-------------|
| `bg-brand-800` | `bg-primary-active` (or `bg-surface-base text-heading` for header strips) |
| `bg-brand-700` | `bg-primary-hover` |
| `bg-brand-600` | `bg-primary` |
| `bg-brand-50` | `bg-primary-subtle` |
| `text-brand-800` | `text-heading` |
| `text-brand-700` | `text-primary` |
| `text-brand-600` | `text-primary` |
| `text-brand-500` | `text-primary` |
| `bg-surface-50` | `bg-surface-app` |
| `bg-surface-100` | `bg-surface-subtle` |
| `bg-surface-200` | `bg-surface-muted` |
| `border-surface-200` | `border-subtle` |
| `border-surface-300` | `border-strong` |
| `text-gray-500` | `text-muted` |
| `text-gray-600` | `text-body` |
| `text-gray-700` | `text-body` |
| `text-gray-800` | `text-heading` |
| `bg-amber-50` | `bg-warn-subtle` |
| `bg-yellow-50` | `bg-warn-subtle` |
| `bg-green-100 text-green-800` | `badge badge-success` |
| `bg-red-100 text-red-800` | `badge badge-error` |
| `bg-blue-100 text-blue-800` | `badge badge-info` |
| `bg-yellow-100 text-yellow-800` | `badge badge-warn` |
| `rounded-2xl` (cards) | `rounded-lg` |
| `rounded-xl` (buttons) | `rounded-md` |
| `shadow-sm` (cards) | `shadow-card` |
| `font-bold` (Subtitle/Heading contexts) | `font-semibold` |
| `font-medium` | `font-semibold` |
| `text-xs` (badge + table header) | `text-sm font-semibold uppercase tracking-wide` |
| `text-xs` (captions / footnotes) | `text-sm text-muted` |

**NEVER modify inside `x-transition:enter-start`, `x-transition:leave-end` classes** blindly — those control the animation; mapping `ease-out duration-400` → `transition ease-standard duration-base` is acceptable and desired (keeps the 200ms budget).

**Empty states** — for each screen with a `{% else %}` / `{% if not data %}` branch, replace ad-hoc "No hay datos" markup with the UI-SPEC empty-state pattern:

```jinja
<div class="empty-state">
  <svg class="w-12 h-12 text-muted" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24" aria-hidden="true">
    <!-- Heroicon outline for the family (see UI-SPEC table) -->
  </svg>
  <h3 class="mt-4 text-subtitle text-heading">{{ headline }}</h3>
  <p class="mt-1 text-body text-muted max-w-md">{{ body }}</p>
  {% if cta_href and can(current_user, cta_perm) %}
    <a href="{{ cta_href }}" class="btn btn-primary mt-6">{{ cta_label }}</a>
  {% endif %}
</div>
```

The exact headline + body per screen:

| Screen | Icon path (Heroicon outline) | Headline | Body | CTA |
|--------|------------------------------|----------|------|-----|
| historial (empty) | `archive-box` | `Sin ejecuciones todavía` | `Cuando lances un análisis aparecerá aquí con su informe PDF.` | `Ir a Análisis` → `/pipeline` (if `pipeline:run`) |
| bbdd (no results) | `magnifying-glass` | `Sin resultados para esta consulta` | `Ajusta los filtros de fecha o recurso y vuelve a lanzar la consulta.` | `Cambiar filtros` (scrolls to form) |
| informes (no PDFs) | `document` | `Aún no hay informes generados` | `Lanza un análisis desde la página de Análisis.` | `Ir a Análisis` → `/pipeline` (if `pipeline:run`) |

The Heroicon outline path strings are in UI-SPEC's component library references; grab them from a Heroicons mirror if convenient, otherwise use a simple stand-in SVG rectangle placeholder — the important thing is that `<svg class="w-12 h-12 text-muted">` is a valid SVG element.

**Button labels (match Copywriting Contract):**

- Pipeline run button label → `Ejecutar análisis`.
- BBDD run button label → `Lanzar consulta`.
- Datos refresh button label → `Refrescar datos`.
- Historial delete button: icon-only `.btn-icon text-muted hover:text-error` with `aria-label="Borrar ejecución"`; modal title `Borrar ejecución`; modal body `Se borrará la ejecución #{id} del {fecha} y sus PDFs asociados. Esta acción es irreversible.`; confirm label `Borrar` (red).
- Informes action buttons: `Descargar` + `Ver`.

**RBAC preservation — explicit check list per template:**

- `pipeline.html` → preserve `{% if can(current_user, 'pipeline:run') %}` wrappers around the Run button and any reschedule/retry buttons.
- `historial.html` → preserve `{% if can(current_user, 'informes:delete') %}` wrapper around delete button.
- `bbdd.html` → no per-button `can()` currently (the whole route is gated at router level); leave as-is.
- `datos.html` → route-level gated; leave as-is.
- `informes.html` → preserve any existing `{% if can() %}` wrappers.

**Do NOT strip or rename:**

- Any `id="..."` used by Alpine `x-ref` or query selectors.
- Any `data-*` attributes.
- Any HTMX `hx-*` attribute.
- The `.log-console` class name (deliberately preserved).
- Preflight modal container IDs / root elements.
  </action>
  <acceptance_criteria>
    - For each of the 5 templates, `grep -cE "bg-(red|green|blue|yellow|amber)-[0-9]{3}" <template>` returns 0.
    - `grep -cE "rounded-2xl|rounded-xl" templates/pipeline.html templates/historial.html templates/bbdd.html templates/datos.html templates/informes.html` returns 0 OR every remaining instance is in a comment / x-transition attribute (manual inspection).
    - `grep -c "preflightModal" templates/pipeline.html` returns 1 or more (Alpine API preserved).
    - `grep -c "preflightModal" templates/bbdd.html` returns 1 or more.
    - `grep -c "can(current_user, 'informes:delete')" templates/historial.html` returns 1 or more.
    - `grep -c "Ejecutar análisis" templates/pipeline.html` returns 1.
    - `grep -c "Lanzar consulta" templates/bbdd.html` returns 1.
    - `grep -c "Refrescar datos" templates/datos.html` returns 1.
    - `grep -c "data-table" templates/historial.html` returns 1 or more.
    - `grep -c "empty-state" templates/historial.html templates/bbdd.html templates/informes.html` returns 3 or more.
    - `grep -c "log-console" templates/pipeline.html` returns 1 or more (preserved).
    - `grep -rn "showToast(" templates/pipeline.html templates/historial.html templates/bbdd.html templates/datos.html templates/informes.html | grep -vE "showToast\((['\"])(info|success|warn|error)\\1" | wc -l` returns 0 (no 2-arg callers remain).
  </acceptance_criteria>
  <verify>
    <automated>! grep -qE "bg-(red|green|blue|yellow|amber)-[0-9]{3}" templates/pipeline.html templates/historial.html templates/bbdd.html templates/datos.html templates/informes.html &amp;&amp; grep -q "preflightModal" templates/pipeline.html &amp;&amp; grep -q "Ejecutar análisis" templates/pipeline.html &amp;&amp; grep -q "Lanzar consulta" templates/bbdd.html &amp;&amp; grep -q "Refrescar datos" templates/datos.html</automated>
  </verify>
  <done>5 templates re-skinned. Alpine APIs preserved. RBAC preserved. Empty states adopted.</done>
</task>

<task type="auto">
  <name>Task 3: Regression test — each route renders + preflight + RBAC intact + no raw colors</name>
  <read_first>
    - `tests/routers/` patterns.
  </read_first>
  <action>
Create `tests/routers/test_data_screens_tokens.py`:

```python
"""Regression for Phase 8 / Plan 08-07: data screens visual refactor.

Locks per screen:
- Route renders for authenticated users with the right permission.
- Preflight Alpine component preserved (pipeline, bbdd).
- RBAC wrappers preserved (historial delete).
- No raw Tailwind state colors remain in the template.
- Spanish CTA labels per Copywriting Contract.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_TEMPLATES = Path(__file__).resolve().parents[2] / "templates"


@pytest.mark.parametrize("template", [
    "pipeline.html", "historial.html", "bbdd.html", "datos.html", "informes.html",
])
def test_no_raw_state_colors(template: str):
    src = (_TEMPLATES / template).read_text(encoding="utf-8")
    assert not re.search(r"bg-(red|green|blue|yellow|amber)-[0-9]{3}", src), (
        f"{template} still has raw Tailwind state colors"
    )


def test_pipeline_preserves_preflight_modal():
    src = (_TEMPLATES / "pipeline.html").read_text(encoding="utf-8")
    assert "preflightModal" in src
    assert "Ejecutar análisis" in src


def test_bbdd_preserves_preflight_modal():
    src = (_TEMPLATES / "bbdd.html").read_text(encoding="utf-8")
    assert "preflightModal" in src
    assert "Lanzar consulta" in src


def test_historial_preserves_delete_rbac_wrapper():
    src = (_TEMPLATES / "historial.html").read_text(encoding="utf-8")
    assert "can(current_user, 'informes:delete')" in src or \
           "can(current_user, \"informes:delete\")" in src


def test_datos_has_refresh_cta():
    src = (_TEMPLATES / "datos.html").read_text(encoding="utf-8")
    assert "Refrescar datos" in src


def test_pipeline_route_renders(propietario_client):
    resp = propietario_client.get("/pipeline")
    assert resp.status_code == 200
    assert "Ejecutar análisis" in resp.text


def test_historial_route_renders(propietario_client):
    resp = propietario_client.get("/historial")
    assert resp.status_code == 200


def test_bbdd_route_renders(propietario_client):
    resp = propietario_client.get("/bbdd")
    assert resp.status_code == 200


def test_datos_route_renders(propietario_client):
    resp = propietario_client.get("/datos")
    assert resp.status_code == 200
```
  </action>
  <acceptance_criteria>
    - `pytest tests/routers/test_data_screens_tokens.py -x -q` exit 0.
    - `pytest tests/ -x -q` exit 0.
    - `ruff check tests/routers/test_data_screens_tokens.py` exit 0.
  </acceptance_criteria>
  <verify>
    <automated>ruff check tests/routers/test_data_screens_tokens.py &amp;&amp; ruff format --check tests/routers/test_data_screens_tokens.py &amp;&amp; pytest tests/routers/test_data_screens_tokens.py -x -q &amp;&amp; pytest tests/ -x -q</automated>
  </verify>
  <done>Full suite green.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

No new boundaries — visual refactor.

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-08-07-01 | Elevation of Privilege | Accidental removal of `can()` gate on delete button | mitigate | Regression test `test_historial_preserves_delete_rbac_wrapper`. |
| T-08-07-02 | Tampering | Preflight modal component rename | mitigate | Regression test `test_pipeline_preserves_preflight_modal`. |
</threat_model>

<verification>
1. `pytest tests/` green.
2. Manual walkthroughs: pipeline → preflight modal fires on large range; historial → delete modal; bbdd → query error renders empty state; datos refresh works; informes grid.
</verification>

<success_criteria>
- 5 templates refactored on tokens.
- 5 skills recorded.
- Preflight + RBAC preserved.
- Full test suite green.
</success_criteria>

<output>
Create `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-07-SUMMARY.md` summarising per-template changes.
</output>
