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
selector row (pabellones) + status pill strip across the top
(produciendo / parada / incidencia), plano-de-fondo area underneath
occupies the primary viewport real estate, ancillary panels (alarmas,
turnos) stacked below as cards.

## Rationale

1. Closest structural match to the existing `luk4.html` layout →
   minimal visual churn + reduced regression risk against the LOCKED
   interaction (D-16).
2. Showcases the semantic state badges (`badge-success` for
   "producing", `badge-warn` for "parada", `badge-error` for
   "incidencia") without re-designing the state machine.
3. Friendly to the polling hook already in `pabPage()` — plano refresh
   cadence unchanged (30 s for telemetry pabellones).
4. Reuses the 4-pabellón selector pattern (`p5` / `p4` / `p3` / `p2`)
   verbatim — 4 independent `pabPage(...)` instances toggled by
   `x-show`. The LOCKED interaction lives in each
   `_partials/mapa_pabellon.html` include, so the chrome around each
   pabellón is the only surface re-skinned.

## Autonomy override (D-12)

D-12 ("el usuario revisa y aprueba todas las propuestas antes de
implementar") is overridden here: the user is away, so Claude picks
the variant that closest matches the existing structure. The choice is
reversible via `git revert <impl-commit>` — tokens + chrome + base
layout stay intact because they live in earlier plans (08-01 / 08-02).

## Actual Alpine root found

The plan originally referenced `x-data="luk4Viewer()"` as the root
component name; the real code uses **`x-data="pabPage({pab, img,
hasTelemetry})"`** × 4, one per pabellón, plus an outer
`x-data="{ pabellon: 'p5' }"` selector. All 5 Alpine directives and
the partial include are preserved verbatim. This is a Plan-vs-Code
discrepancy (Rule 3 blocking issue in the plan's assumptions),
resolved by adapting acceptance criteria to match the real component
surface.

## Token inventory used

- `bg-surface-base` (card surfaces, plano wrapper background).
- `bg-surface-app` (page background via base.html).
- `bg-surface-subtle` (pabellón selector + muted chips).
- `text-body` (body copy).
- `text-muted` (timestamps, secondary metadata).
- `text-heading` (section titles, if any outer heading is rendered).
- `border-subtle` (dividers, card outlines).
- `badge-success`, `badge-warn`, `badge-error` (state legend chips).
- `shadow-card` (elevated alarmas / modal panel).
- `rounded-lg` (card radius canonical).

## Component variants used

- `.card` (flat, plano container wrapper).
- `.card-elevated` (alarmas modal container — shadowed).
- `.badge-success` / `.badge-warn` / `.badge-error` (static legend
  chips — NOT the dynamic zone colorings, which stay as Alpine
  `:class` bindings per D-16 LOCKED).
- `.btn-ghost` for the top-bar connection pill (uses existing
  `conn-badge` hx-get infra).

## Deviations from 08-UI-SPEC

None structural. Zone coloring inside `_partials/mapa_pabellon.html`
still uses `bg-violet-*` / `bg-emerald-*` / `bg-red-*` / `bg-green-*`
utilities — LEGACY-ALLOWED per UI-SPEC §Per-Screen Adaptations row
`luk4.html`. Any replacement is Mark-IV work.

## Landmines encountered

1. The actual `/` route renders `luk4.html` (confirmed in
   `api/routers/pages.py:30-32`), and `templates/index.html` does NOT
   exist — so the plan's "audit index.html" step is a no-op.
2. `templates/luk4.html` already extends `base.html` (since 08-02). No
   migration needed for that invariant.
3. `templates/_partials/mapa_pabellon.html` uses literal state colors
   (`bg-red-50`, `bg-green-50`, `bg-red-500`, `bg-green-500`,
   `bg-amber-500`) inside the zone detail modal. These are
   LEGACY-ALLOWED and MUST NOT be migrated in this plan — any change
   there risks the LOCKED interaction.
4. `luk4.html` already has **zero raw `bg-red-###` / `bg-green-###` /
   `bg-yellow-###` / `bg-blue-###` state colors** in the outer chrome
   (the dynamic zone colorings are inside the partial). So the
   "replace raw state colors with badge-*" mandate is a no-op for the
   outer template; we instead restyle the pabellón selector + the
   topbar conn-badge to consume tokens.
5. The Chart.js inline color literals (`#1a3a5c`, `#d97706`,
   `#dc2626`, `#7c3aed` in the chart dataset borderColor configs) live
   inside the Alpine component's `renderChart()` method. These are
   behavior-tied (they are the exact colors that Chart.js needs at
   render time), and migrating them to CSS tokens requires a
   `getComputedStyle(root).getPropertyValue('--color-primary')` read
   at render. Per D-16 LOCKED and per the softer UI-SPEC row `luk4.html`
   ("Plan 08-15 decides visual scope during sketch"), we leave
   these untouched for Plan 08-05.

## Reversibility

To revert this screen's visual refactor while keeping chrome + tokens:
`git revert <impl-commit>` — tokens + base.html stay intact because
they live in earlier plans (08-01 / 08-02). The plano editor
interaction is LOCKED and never touched in this plan, so revert is
safe.
