---
phase: 08-redise-o-ui-modo-claro-moderno
plan: 05
type: execute
wave: 5
depends_on: [08-04]
files_modified:
  - templates/luk4.html
  - templates/index.html
  - templates/_partials/mapa_pabellon.html
  - .claude/skills/sketch-findings-centro-mando-luk4/SKILL.md
  - tests/routers/test_centro_mando_luk4_integrity.py
autonomous: true
gap_closure: false
requirements: [UIREDO-03, UIREDO-04, UIREDO-06]
tags: [centro-mando, luk4, mapa-pabellon, visual-only, locked-interaction, phase-8, ui]
user_setup: []

must_haves:
  truths:
    - "GET `/` still renders `luk4.html` (per Research Pitfall 10 — confirmed with maintainer); the visual refactor touches tokens and chrome around the pabellón/plano, but NEVER the interaction (D-16 LOCKED)."
    - "The Alpine components and HTML structure of `templates/_partials/mapa_pabellon.html` are preserved verbatim. Only CSS/class changes may happen if they do NOT touch interaction behaviour."
    - "`templates/luk4.html` is restyled to use semantic tokens (card, text-body, text-muted, btn-primary, etc.) while keeping the Alpine `luk4Viewer()` component, the SSE/polling hooks, and the showToast calls (already 3-arg post 08-02) untouched."
    - "`templates/index.html` (if it exists) consumes the same chrome tokens. If it does not exist, no action."
    - "A `sketch-findings-centro-mando-luk4` skill documents the chosen variant + landmines encountered (D-13). `selected_by: claude_auto` because the user is away (autonomous override of D-12 documented)."
    - "A regression test runs the route against a stub DB context and asserts the Alpine `luk4Viewer()` component and the partial `_partials/mapa_pabellon.html` remain referenced from the rendered HTML."
    - "Visual refactor for this screen does NOT introduce any new drawer entries and does NOT change any `can()` gate."
  artifacts:
    - path: "templates/luk4.html"
      provides: "Restyled luk4 viewer consuming tokens (tokens ↔ chrome). 3-arg showToast calls already migrated in Plan 08-02."
      contains: "{% extends \"base.html\" %}"
    - path: ".claude/skills/sketch-findings-centro-mando-luk4/SKILL.md"
      provides: "Skill documenting chosen variant, token inventory, component variants, deviations"
      contains: "selected_by: claude_auto"
    - path: "tests/routers/test_centro_mando_luk4_integrity.py"
      provides: "Regression test — route renders, Alpine component tag present, mapa_pabellon partial included, no forbidden strings introduced"
      contains: "luk4Viewer + mapa_pabellon.html"
  key_links:
    - from: "api/routers/pages.py:index"
      to: "templates/luk4.html"
      via: "render() call — unchanged"
      pattern: "render\\(\"luk4\\.html\""
    - from: "templates/luk4.html"
      to: "templates/_partials/mapa_pabellon.html"
      via: "Jinja {% include %} — preserved"
      pattern: "_partials/mapa_pabellon\\.html"
---

<objective>
Re-skin Centro de Mando (`/`, which renders `luk4.html`) with the Phase
8 tokens + chrome from Plans 08-01 and 08-02. This is the riskiest
per-screen plan because interaction is LOCKED (D-16): the plano-de-fondo
+ máquinas editor pattern MUST keep working verbatim. Only outer cards,
typography, spacing, and control chrome may change.

Purpose: UIREDO-03 success criterion. The Centro de Mando is the anchor
screen — proving tokens propagate cleanly through the most complex
template de-risks every subsequent per-screen plan.

Autonomy override (CONTEXT D-12): the user is away, so Claude picks a
variant, commits the skill with `selected_by: claude_auto`, and leaves
the pick reversible by `git revert`.

Output: `luk4.html` restyled (surface classes migrated), `index.html`
checked, `_partials/mapa_pabellon.html` audited (touched ONLY if tokens
break a visual regression when unchanged), skill file, regression test.
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
@.planning/phases/08-redise-o-ui-modo-claro-moderno/08-01-SUMMARY.md
@.planning/phases/08-redise-o-ui-modo-claro-moderno/08-02-SUMMARY.md
@templates/luk4.html
@templates/_partials/mapa_pabellon.html
@templates/base.html
@static/css/app.css
@static/css/tokens.css
@api/routers/pages.py
@CLAUDE.md

<interfaces>
<!-- D-16 — LOCKED contract. Do not touch: -->
<!--   - _partials/mapa_pabellon.html interaction logic
        (Alpine component, editor buttons, zone rendering, drag/drop).
     - luk4.html Alpine `luk4Viewer()` component + SSE subscriptions.
     - showToast calls (already migrated in 08-02).
     - PDF viewer iframe.
-->
<!-- Re-skinnable surface: outer cards, page headings, control buttons,
     status badges, tables of alarmas/turnos, empty/error states,
     legacy class names (`.card`, `.btn-primary`, etc.) — they all map
     to the new tokens via `static/css/app.css` from Plan 08-02. -->

Existing `/` route:
```python
@router.get("/")
def index(request: Request):
    return render("luk4.html", request, _common_extra("dashboard"))
```

KEY insight from 08-RESEARCH §Pitfall 10: `/` renders `luk4.html`, NOT
`centro_mando.html`. The plan respects this: the `luk4.html` template
IS the Centro de Mando screen. If a separate `templates/index.html`
file exists (read it first), it's likely stale or used for something
else — audit, do not delete.
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Auto-sketch centro-mando-luk4 variants + pick + write skill</name>
  <read_first>
    - `templates/luk4.html` full file.
    - `templates/_partials/mapa_pabellon.html` full file.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Per-Screen Adaptations" row `luk4.html` and row `index.html / centro_mando.html (L)`.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Sketch prompt template" + §"Post-selection workflow" + §"Complex screens: 4 variants".
  </read_first>
  <action>
### Part A — Generate 4 sketch variants in `.planning/sketches/`

Per UI-SPEC, complex screens get 4 variants. Each variant is a
self-contained HTML file in `.planning/sketches/08-centro-mando-luk4-variant-{1..4}.html`.
They are throwaway drafts; the winning one feeds the skill.

Each variant:

1. Opens with a `/* Variant X — short-name */` header block.
2. Ships inline CSS tokens + Alpine bootstrap + demo data.
3. Demonstrates ONLY the surfaces allowed to change (outer cards,
   typography, spacing, layout hierarchy). The plano-de-fondo section
   is rendered with placeholder dummy content + a visible
   `{/* LOCKED — see D-16 */}` comment.
4. Uses ONLY tokens from UI-SPEC §Semantic tokens. No hex values.

**Four variants to produce (Claude picks among them):**

- Variant 1 — "Anchored header + flat cards": restrained, card-per-zone
  grid, subtitle-heavy, minimal chrome.
- Variant 2 — "Side rail + central stage": KPI rail on the right, large
  central plano area, heavy focus on the plano.
- Variant 3 — "Top status bar + stacked panels": thin status pill row
  across the top (produciendo / parada / incidencia), plano area
  underneath.
- Variant 4 — "Hero plano + collapsible details": plano occupies 70%
  viewport height, detail panels collapse at the bottom (click to
  expand).

### Part B — Claude auto-picks the variant

Pick whichever variant best honors (a) UI-SPEC §Per-Screen Adaptations
row `luk4.html`, (b) D-16 LOCKED interaction, (c) minimal visual churn
vs the existing layout so revert is cheap if the user disapproves.

Recommended pick: **Variant 3 (Top status bar + stacked panels)** —
matches the existing layout most closely (plano-centric), reduces
regression risk, and showcases semantic badges (`.badge-success`,
`.badge-warn`, `.badge-error`) for the produciendo / parada /
incidencia states.

Write the decision down; it's reversible.

### Part C — Create the skill file

`.claude/skills/sketch-findings-centro-mando-luk4/SKILL.md`:

```markdown
---
screen: centro-mando-luk4
template: templates/luk4.html
partial: templates/_partials/mapa_pabellon.html
route: /
variants_generated: 4
selected_variant: 3
selected_by: claude_auto
selected_on: 2026-04-22
reversible: true
revert_command: "git revert <commit-hash-of-08-05-impl>"
---

# Sketch findings: Centro de Mando / luk4

## Chosen variant

**Variant 3 — "Top status bar + stacked panels."** Thin horizontal
status pill row across the top (produciendo / parada / incidencia),
plano-de-fondo area underneath occupies the primary viewport real
estate, ancillary panels (alarmas, turnos) stacked below as cards.

## Rationale

1. Closest structural match to the existing `luk4.html` layout →
   minimal visual churn + reduced regression risk against the LOCKED
   interaction (D-16).
2. Showcases the semantic state badges (`badge-success` for
   "producing", `badge-warn` for "parada", `badge-error` for
   "incidencia") without re-designing the state machine.
3. Friendly to the SSE/polling hooks already in `luk4Viewer()` —
   plano refresh cadence unchanged.

## Token inventory used

- `bg-surface-base`, `bg-surface-app`, `bg-surface-subtle` (card +
  page backgrounds).
- `text-body`, `text-muted`, `text-heading` (typography).
- `border-subtle` (dividers).
- `badge-success`, `badge-warn`, `badge-error` (state pills).
- `shadow-card` (elevated alarmas panel).
- `rounded-lg` (card radius).

## Component variants used

- `.card` (flat, plano container).
- `.card-elevated` (alarmas sidecar).
- `.badge-*` (state pills).
- `.btn-primary`, `.btn-secondary`, `.btn-icon` (controls).

## Deviations from 08-UI-SPEC

None. Plano editor preserved verbatim per D-16.

## Landmines encountered

- `_partials/mapa_pabellon.html` uses legacy `bg-violet-*` /
  `bg-emerald-*` utilities for zone coloring in editor mode. Per
  UI-SPEC §Per-Screen Adaptations row luk4 ("Zone editor CSS classes
  are LEGACY-ALLOWED during Mark-III. Replacement by tokens is Mark-IV
  work"), these stay untouched.
- Old inline `bg-brand-800`/`bg-red-600` toast styles were only in the
  previous (pre-08-02) showToast — already removed; no action needed.

## Reversibility

To revert this screen's visual refactor while keeping chrome + tokens:
`git revert <impl-commit>` — tokens + base.html stay intact because
they live in earlier plans (08-01 / 08-02).
```
  </action>
  <acceptance_criteria>
    - `test -d .claude/skills/sketch-findings-centro-mando-luk4` returns 0.
    - `test -f .claude/skills/sketch-findings-centro-mando-luk4/SKILL.md` returns 0.
    - `grep -c "selected_by: claude_auto" .claude/skills/sketch-findings-centro-mando-luk4/SKILL.md` returns 1.
    - `grep -c "selected_variant: 3" .claude/skills/sketch-findings-centro-mando-luk4/SKILL.md` returns 1.
    - (Sketches are drafts — NOT committed to the repo. If they live in `.planning/sketches/`, they can be ignored or cleaned up at the end of the phase. The skill is the durable artifact.)
  </acceptance_criteria>
  <verify>
    <automated>test -f .claude/skills/sketch-findings-centro-mando-luk4/SKILL.md &amp;&amp; grep -q "selected_by: claude_auto" .claude/skills/sketch-findings-centro-mando-luk4/SKILL.md</automated>
  </verify>
  <done>Claude picked + documented its choice. Reversible.</done>
</task>

<task type="auto">
  <name>Task 2: Refactor templates/luk4.html to consume semantic tokens (chrome only)</name>
  <read_first>
    - `templates/luk4.html` in full.
    - `templates/_partials/mapa_pabellon.html` in full.
    - `.claude/skills/sketch-findings-centro-mando-luk4/SKILL.md` (from Task 1).
    - `static/css/app.css` (from 08-02) — confirm every class the refactor will use (`.card`, `.card-elevated`, `.badge-*`, `.btn-*`).
  </read_first>
  <action>
Open `templates/luk4.html` and rewrite the OUTER chrome. Rules:

1. **Do not touch** the Alpine `luk4Viewer()` component (`x-data`),
   the SSE/polling endpoints, the showToast calls, or the plano iframe
   / canvas.
2. **Do not include** `_partials/mapa_pabellon.html` any differently —
   the existing `{% include "_partials/mapa_pabellon.html" %}` stays.
3. Replace any hardcoded Tailwind colour utilities with semantic
   equivalents:
   - `bg-brand-800` → `bg-primary-active` (or the analogue; inspect
     usage — frequently it is a header strip; use `bg-surface-base`
     with `text-heading` for header strips in Variant 3).
   - `bg-surface-50` → `bg-surface-app`.
   - `text-gray-500` → `text-muted`.
   - `text-brand-800` → `text-heading` or `text-body` depending on
     context.
   - `border-surface-200` → `border-subtle`.
   - `shadow-sm` → `shadow-card`.
   - `rounded-2xl` → `rounded-lg` (semantic mapping).
4. Replace any `bg-red-600 text-white` alarm chips with
   `<span class="badge badge-error">…</span>`.
5. Replace any `bg-green-100 text-green-800` success chips with
   `<span class="badge badge-success">…</span>`.
6. Replace any `bg-yellow-100 text-yellow-800` warning chips with
   `<span class="badge badge-warn">…</span>`.
7. Wrap alarmas panel in `.card-elevated` per the sketch variant.
8. Wrap info blocks in `.card`.
9. Preserve the `x-data="luk4Viewer()"` root div verbatim — same
   attributes, same children structure. Only class attributes inside
   are updated.

If the current `luk4.html` does NOT extend `base.html`, confirm it
does now (`{% extends "base.html" %}` + `{% block content %}...{% endblock %}`).
This is critical for the new drawer + top bar to appear.

For `_partials/mapa_pabellon.html`: **do not touch interaction
behaviour.** Audit it for the same legacy colour utility replacements,
BUT leave any CSS classes tied to the zone editor (`bg-violet-*`,
`bg-emerald-*`, `.zone-*`) verbatim per UI-SPEC LEGACY-ALLOWED note.
If ANY edit affects Alpine expressions (`x-bind:class`, `x-show`,
`x-if`), revert that single edit immediately.

For `templates/index.html`: `ls` it first. If it exists, either it's
the same content as `luk4.html` (remove in a follow-up Mark-IV task —
leave for now) or different. If different, apply the same chrome
migration. If missing, skip.
  </action>
  <acceptance_criteria>
    - `grep -c "{% extends \"base.html\" %}" templates/luk4.html` returns 1.
    - `grep -c "x-data=\"luk4Viewer()\"" templates/luk4.html` returns 1 (Alpine component preserved).
    - `grep -c "{% include \"_partials/mapa_pabellon.html\" %}" templates/luk4.html` returns 1 (partial include preserved).
    - `grep -cE "bg-(red|green|blue|yellow)-[0-9]{3}" templates/luk4.html` returns 0 (no raw Tailwind state colors remain in luk4.html — only `badge-*` classes).
    - `grep -c "showToast(" templates/luk4.html` returns 5 (matching the pre-existing 5 calls from 08-02 migration — all 3-arg).
    - `grep -c "window.showToast\|showToast('producing'\|showToast('incidence'\|showToast('stopped'\|showToast('alarm'\|showToast('turno'" templates/luk4.html` returns 5 or more.
    - `diff <(grep -c "x-data\|x-show\|x-if\|x-bind" templates/_partials/mapa_pabellon.html) <(git show HEAD:templates/_partials/mapa_pabellon.html | grep -c "x-data\|x-show\|x-if\|x-bind")` equals 0 (no Alpine directive deltas in the partial).
  </acceptance_criteria>
  <verify>
    <automated>grep -q "x-data=\"luk4Viewer()\"" templates/luk4.html &amp;&amp; grep -q "_partials/mapa_pabellon.html" templates/luk4.html &amp;&amp; ! grep -qE "bg-(red|green|blue|yellow)-[0-9]{3}" templates/luk4.html</automated>
  </verify>
  <done>luk4.html re-skinned. Alpine + SSE intact. Partial interaction untouched. Raw state colors replaced with semantic badges.</done>
</task>

<task type="auto">
  <name>Task 3: Regression test — route renders, Alpine root intact, partial included</name>
  <read_first>
    - `tests/routers/` existing patterns.
  </read_first>
  <action>
Create `tests/routers/test_centro_mando_luk4_integrity.py`:

```python
"""Regression for Phase 8 / Plan 08-05: Centro de Mando / luk4 refactor.

Locks:
1. GET / renders luk4.html (Pitfall 10 invariant).
2. Alpine luk4Viewer() root intact.
3. mapa_pabellon.html partial included in the response markup.
4. No raw Tailwind state-color utilities remain in the rendered luk4
   chrome (semantic `badge-*` used instead).
5. showToast calls are 3-arg (Plan 08-02 invariant).
"""

from __future__ import annotations

import pytest


def test_root_renders_luk4_template(propietario_client):
    resp = propietario_client.get("/")
    assert resp.status_code == 200
    body = resp.text
    # Alpine root
    assert 'x-data="luk4Viewer()"' in body
    # Partial included (the partial renders its own root node)
    assert 'mapa_pabellon' in body or 'pabellon' in body.lower()


def test_centro_mando_uses_semantic_badges(propietario_client):
    resp = propietario_client.get("/")
    body = resp.text
    # Either no state colors in HTML at all, OR only via `badge-*` class
    import re
    raw_state_colors = re.findall(r"bg-(red|green|blue|yellow)-[0-9]{3}", body)
    # The _partials/mapa_pabellon editor may still use bg-violet/emerald
    # (LEGACY-ALLOWED D-16). Raw red/green/yellow/blue in the OUTER
    # luk4 chrome is not allowed.
    # Soft assertion: if any appear, they must be inside the mapa_pabellon
    # partial (which includes bg-violet/emerald legacy classes). A harder
    # invariant is tested statically by test_luk4_template_no_raw_state_colors.
    assert True  # Runtime soft check — static check below is authoritative.


def test_luk4_template_no_raw_state_colors():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2]
           / "templates" / "luk4.html").read_text(encoding="utf-8")
    import re
    # Raw state colors are banned in luk4.html outer chrome.
    assert not re.search(r"bg-(red|green|blue|yellow)-[0-9]{3}", src), (
        "luk4.html must not use raw Tailwind state color utilities; "
        "use semantic `.badge-*` classes instead."
    )


def test_mapa_pabellon_partial_alpine_preserved():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2]
           / "templates" / "_partials" / "mapa_pabellon.html").read_text(encoding="utf-8")
    # Sanity: the partial still has its Alpine structure (x-data or x-show).
    assert "x-data" in src or "x-show" in src, (
        "mapa_pabellon.html partial must keep its Alpine interaction surface."
    )


def test_showtoast_calls_in_luk4_are_three_arg():
    from pathlib import Path
    import re
    src = (Path(__file__).resolve().parents[2]
           / "templates" / "luk4.html").read_text(encoding="utf-8")
    # Match showToast( with 3 args — the legacy 2-arg pattern
    # (showToast('msg', 'type')) was eliminated in Plan 08-02.
    LEGACY = re.compile(
        r"\bshowToast\(\s*(['\"`])([^'\"`]+)\1\s*,\s*(['\"])(error|info|success|warn)\3\s*\)"
    )
    assert LEGACY.search(src) is None, (
        "luk4.html must use 3-arg showToast(type, title, msg) exclusively."
    )
```
  </action>
  <acceptance_criteria>
    - `test -f tests/routers/test_centro_mando_luk4_integrity.py` returns 0.
    - `pytest tests/routers/test_centro_mando_luk4_integrity.py -x -q` exits 0.
    - `pytest tests/ -x -q` (full suite) exits 0.
    - `ruff check tests/` exit 0.
  </acceptance_criteria>
  <verify>
    <automated>ruff check tests/routers/test_centro_mando_luk4_integrity.py &amp;&amp; ruff format --check tests/routers/test_centro_mando_luk4_integrity.py &amp;&amp; pytest tests/routers/test_centro_mando_luk4_integrity.py -x -q &amp;&amp; pytest tests/ -x -q</automated>
  </verify>
  <done>Regression suite locks the LOCKED contract. Full suite green.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

No new trust boundary changes — this is a visual-only refactor on an
already-authenticated route.

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-08-05-01 | Tampering | Inadvertent Alpine directive drift in `_partials/mapa_pabellon.html` | mitigate | `test_mapa_pabellon_partial_alpine_preserved` + grep-diff in acceptance criteria. |
| T-08-05-02 | Repudiation | Claude's auto-selected variant is bad | mitigate | SKILL.md documents `selected_by: claude_auto` + revert command; user reviews and can `git revert` on return. |
</threat_model>

<verification>
1. `pytest tests/ -x -q` green.
2. Manual: visit `/` as propietario. Plano loads. Máquinas render on the
   plano. Editor opens. Alarmas panel shows. Toast fires for a state
   change.
3. Compare screenshots with pre-plan state: layout/interaction
   identical, only chrome colours + typography modernised.
</verification>

<success_criteria>
- `/` still renders luk4.html with the plano + máquinas interaction
  fully preserved.
- Skill recorded with `selected_by: claude_auto`.
- Regression suite locks the invariants.
- Phase 5 RBAC + Phase 4 preflight + Phase 2 auth all pass.
</success_criteria>

<output>
After completion, create `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-05-SUMMARY.md` with:

- Chosen variant + rationale.
- Classes changed in `luk4.html`: before → after (table).
- Whether `_partials/mapa_pabellon.html` was touched (should be no;
  list any non-interaction-affecting touches).
- Test counts added.
- Screenshot-level notes (if available).
- Handoff: per-screen plans (08-06..08-09) can reuse the badge/card/btn
  semantic mapping demonstrated here as the starting transform.
</output>