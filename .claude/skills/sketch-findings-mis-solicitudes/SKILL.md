---
screen: mis-solicitudes
template: templates/mis_solicitudes.html
route: /mis-solicitudes
variants_generated: 3
selected_variant: 2
selected_by: claude_auto
selected_on: 2026-04-22
reversible: true
revert_command: "git revert <commit-hash-of-08-06-impl>"
---

# Sketch findings: mis_solicitudes

## Chosen variant

**Variant 2 — Table-first.** Extends `base.html`. One `.data-table`
inside a `.card` with columns: `#` · `Endpoint` · `Estimado (ms)` ·
`Creado` · `Estado` · `Acción`. Status rendered as `.badge-*` pill
(neutral → cancelled / expired, success → approved, warn → pending,
error → rejected, brand → consumed). Empty-state uses `.empty-state`
with paper-airplane SVG + `No tienes solicitudes pendientes` copy.
Cancel action opens an Alpine confirmation modal (Volver / Cancelar
solicitud).

## Rationale

1. Keeps the existing data model verbatim (`solicitudes` list of
   approval rows with `id`, `endpoint`, `estimated_ms`, `created_at`,
   `status`, `consumed_at`).
2. Status → pill mapping uses the semantic `.badge-*` classes (no raw
   `bg-yellow-100` / `bg-green-100` etc.). This is the FIRST screen to
   use the full `.badge-neutral/brand/success/warn/error` palette; the
   pattern is re-usable by downstream plans (08-07..08-09).
3. "Ejecutar ahora" link (for status=`approved` + `consumed_at is
   None`) preserves the Plan 04-03 flow — per-endpoint URL with
   `?approval_id=` query param, which the preflight modal's `init()`
   re-dispatches with `force=true`.
4. Cancel modal follows UI-SPEC §"Destructive actions" — confirm via
   modal (not native `confirm()`), with `Volver` as the secondary
   (safe) action and `Cancelar solicitud` as `.btn-danger`.

## Token inventory

- `.data-table` (Plan 08-02 app.css).
- `.card`, `.card-body`.
- `.badge-neutral`, `.badge-brand`, `.badge-success`, `.badge-warn`,
  `.badge-error`.
- `.empty-state`.
- `.btn-secondary` / `.btn-ghost` / `.btn-danger`.
- `z-modal` / `z-backdrop` (for the cancel modal layering).
- `bg-surface-900/40` (backdrop) — via raw Tailwind `bg-surface-900`
  which IS a semantic class (the tailwind config exposes
  `surface-900` = `var(--color-surface-900)`).

## Component inventory

- Table: `<table class="data-table">` with `.numeric` on the timestamp
  column.
- Empty state: `<div class="empty-state">` with a paper-airplane SVG.
- Modal: Alpine `x-data` + `x-show` + `x-cloak` + `role="dialog"` +
  `aria-modal="true"` + event dispatch via `$dispatch('confirm-cancel',
  {id})` from the row-level Cancel button.

## Autonomy override (D-12)

Operator AFK. Plan 08-06 explicitly authorizes Claude to pick
Variant 2. Reversible.

## Deviations

- **Per-endpoint "Ejecutar ahora" URL** — the plan's snippet shows a
  generic `{{ s.endpoint }}?approval_id=…` link, but the existing
  template uses endpoint-specific HTML paths (`/pipeline`, `/bbdd`,
  `/capacidad`, `/operarios`) because Plan 04-03 wires the preflight
  re-dispatcher per screen. We preserve the per-endpoint mapping
  verbatim — changing it would break the D-15 auto-retry contract.
- **`expired` and `consumed` badge classes** — plan only listed
  pending/approved/rejected/cancelled; we add `expired` → neutral and
  `consumed` → brand to cover every value the router can emit
  (Rule 2 completeness — the server is the source of truth for status
  strings).

## Landmines

- **Cancel endpoint is `/api/approvals/{id}/cancel`** — preserve the
  literal path. Plan 04-03 wired this; the Alpine modal builds the
  URL from the row-level `id` on submission.
- **Ownership check** — router filters `list_by_user(user.id)` so the
  template only receives the current user's rows. DO NOT add any
  user-selector UI here.
- **`?ok=cancelled`** query param — the `page_mis_solicitudes` view
  accepts it and we surface it as a success banner at the top of the
  page (preserved from the original template).
- **Approved + already-consumed (`consumed_at is not None`)** — do
  NOT render "Ejecutar ahora"; the token was burned. Render `—`
  instead.
