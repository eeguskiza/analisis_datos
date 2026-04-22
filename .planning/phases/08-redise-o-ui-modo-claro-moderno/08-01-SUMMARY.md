---
phase: 08-redise-o-ui-modo-claro-moderno
plan: 01
subsystem: ui-tokens
tags: [ui, tokens, tailwind, css-variables, branding, phase-8]
requires:
  - UIREDO-01
provides:
  - static/css/tokens.css (two-layer design tokens)
  - static/js/tailwind.config.js (Tailwind CDN config extracted from base.html)
  - docs/BRANDING.md §Tokens (Phase 8)
  - tests/infra/test_tokens_css.py (56 regression tests)
affects:
  - static/css/app.css (added @import at line 1)
tech-stack:
  added: []
  patterns:
    - Two-layer tokens (raw palette + semantic aliases)
    - CSS variable via Tailwind rgb(var(--token) / <alpha-value>) pattern
    - Light-only (no dark hooks, no prefers-color-scheme)
key-files:
  created:
    - static/css/tokens.css
    - static/js/tailwind.config.js
    - tests/infra/test_tokens_css.py
    - .planning/phases/08-redise-o-ui-modo-claro-moderno/deferred-items.md
  modified:
    - static/css/app.css
    - docs/BRANDING.md
decisions:
  - D-01 applied: two-layer tokens (raw + semantic) in one file
  - D-02 applied: Tailwind theme.extend consumes tokens via rgb(var(--x) / <alpha-value>)
  - D-03 applied: brand + surface preserved; surface 400-900 + semantic states filled
  - D-04 applied: light-only, no dark hooks
  - D-05 applied: typography, spacing aliases, radius, shadow, motion, z-index tokens
  - D-06 applied: system font stack (no CDN)
metrics:
  duration_minutes: 9
  completed_date: 2026-04-22
  tasks_completed: 3
  files_created: 4
  files_modified: 2
  tests_added: 56
---

# Phase 08 Plan 01: Tokens + Tailwind Config Summary

Ship the foundational token layer for the Phase 8 UI redesign: two-layer
`tokens.css`, Tailwind config extracted from `base.html`, BRANDING.md doc
section, and a static regression test. No templates rewritten — the app
renders identically until Plan 08-02 wires the new chrome.

## Token Inventory

### Raw palette (Layer 1)

| Group | Steps declared |
|-------|----------------|
| `brand` | 50, 100, 200, 300, 400, 500, 600, 700, 800, 900 (10 steps, kept verbatim from `base.html`) |
| `surface` | 0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900 (11 steps — `0` and `400-900` are new) |
| `success` | 50, 100, 500, 600, 700 |
| `warn` | 50, 100, 500, 600, 700 |
| `error` | 50, 100, 500, 600, 700 |
| `info` | 50, 100, 500, 600, 700 |

Format: space-separated RGB triples (e.g. `--color-brand-600: 26 58 92;`).
Zero use of `rgb()` wrapper or comma separators inside `--color-*` tokens
(verified by `test_color_tokens_do_not_wrap_rgb`).

### Semantic tokens (Layer 2) — 24 color aliases

- **Surfaces (4):** `--color-surface-base`, `--color-surface-app`,
  `--color-surface-subtle`, `--color-surface-muted`.
- **Text (5):** `--color-text-body`, `--color-text-heading`,
  `--color-text-muted`, `--color-text-disabled`, `--color-text-on-accent`.
- **Border (2):** `--color-border-subtle`, `--color-border-strong`.
- **Primary / accent (5):** `--color-primary`, `--color-primary-hover`,
  `--color-primary-active`, `--color-primary-subtle`, `--color-focus-ring`.
- **Semantic states (8):** `--color-success` + `-subtle`, `--color-warn` +
  `-subtle`, `--color-error` + `-subtle`, `--color-info` + `-subtle`.

### Typography scale

`--text-body-size` 14px / `--text-body-lh` 1.5 / weights 400 + 600
`--text-subtitle-size` 16px / lh 1.4
`--text-heading-size` 20px / lh 1.3 / ls -0.01em
`--text-display-size` 32px / lh 1.2 / ls -0.02em
`--font-sans` = system stack (D-06); `--font-mono` = `ui-monospace` stack.

### Spacing aliases

`--space-field-gap` 16px, `--space-card-padding` 24px,
`--space-card-gap` 16px, `--space-section-gap` 32px,
`--space-page-gutter-x` 24px (`-mobile` 16px),
`--space-topbar-h` 56px (`-mobile` 52px),
`--space-drawer-w` 280px, `--space-drawer-padding` 16px.

### Elevation (shadow)

`--shadow-card`, `--shadow-popover`, `--shadow-modal`, `--shadow-drawer`.

### Radius

`--radius-sm` 6px, `--radius-md` 10px, `--radius-lg` 16px,
`--radius-xl` 24px, `--radius-pill` 9999px.

### Motion

`--duration-fast` 150ms, `--duration-base` 200ms, `--duration-slow` 300ms
`--ease-standard`, `--ease-emphasized`, `--ease-accelerate`.
Reduced-motion global rule at `@media (prefers-reduced-motion: reduce)`.

### Z-index scale

`--z-base` 0, `--z-sticky` 10, `--z-topbar` 30, `--z-backdrop` 40,
`--z-drawer` 50, `--z-modal` 60, `--z-popover` 70, `--z-toast` 90,
`--z-reserved` 100 (emergency only).

## Files Created

- `static/css/tokens.css` — 184 lines, ~4.5 KB. Two-layer tokens + reduced-motion.
- `static/js/tailwind.config.js` — 128 lines. Declares `window.tailwind.config`
  consumed by Tailwind CDN; maps every semantic utility to a CSS var via
  `rgb(var(--token) / <alpha-value>)` to preserve alpha utilities like
  `bg-primary/20`. Legacy `brand.50-900` + `surface.0-900` scales preserved
  for edge cases (mapa_pabellon editor).
- `tests/infra/test_tokens_css.py` — 56 parametrized + unit regression tests.
  Ruff-format and ruff-check clean. Runs in ~0.03s.
- `.planning/phases/08-redise-o-ui-modo-claro-moderno/deferred-items.md` —
  log of one out-of-scope pre-existing failure (see Deferred Issues below).

## Files Modified

- `static/css/app.css` — prepended `@import url('./tokens.css');` as the first
  line. No other rules touched (those get rewritten in Plan 08-02).
- `docs/BRANDING.md` — appended `## Tokens (Phase 8)` section with the
  60/30/10 palette split, typography, motion, z-index, Tailwind mapping and
  WCAG 2.1 AA contrast matrix. Existing Sprint 0 content preserved verbatim.

## Test Count Added

56 tests in `tests/infra/test_tokens_css.py`:

- 24 `test_tokens_file_declares_semantic_color` parametrized over every
  required semantic color token.
- 21 `test_tokens_file_declares_layout_tokens` parametrized over radius (5),
  shadow (4), motion (6), z-index (6) tokens.
- 1 `test_color_tokens_do_not_wrap_rgb` — format regression (Pitfall 2 from
  08-RESEARCH.md).
- 1 `test_reduced_motion_block_present` — D-05 requirement.
- 1 `test_app_css_imports_tokens` — asserts `@import url('./tokens.css');` is
  line 1 of `app.css`.
- 6 `test_tailwind_config_references_tokens` parametrized — asserts
  `var(--color-…)` references present for the first 6 semantic tokens.
- 1 `test_tailwind_uses_alpha_value_pattern` — at least one
  `/ <alpha-value>` usage.
- 1 `test_branding_md_has_tokens_section` — `## Tokens (Phase 8)` +
  `60 / 30 / 10` + `` `--color-primary` `` present.

Verified with `docker compose exec -T web pytest tests/infra/test_tokens_css.py -x -q`:
all 56 pass in 0.03s.

## Deviations from Plan

None — plan executed exactly as written. The action block from Task 1 and
Task 2 was copied verbatim from the plan (every hex/RGB value preserved).

## Deferred Issues

**Out-of-scope** (logged to
`.planning/phases/08-redise-o-ui-modo-claro-moderno/deferred-items.md`):

- `tests/routers/test_thresholds_crud.py::test_recalibrate_insufficient_data_returns_400`
  fails on HEAD. Verified pre-existing against `3a9d836` (pre-plan commit)
  with the same failure mode. Completely disjoint from Plan 08-01 scope
  (touches `/api/thresholds/…/recalibrate`, no CSS/Tailwind/branding). Not
  caused by this plan; not fixed here.

## Handoff Note for Plan 08-02

The substrate for chrome redesign is live:

1. `static/css/tokens.css` is loaded on every page via `app.css @import` and
   exposes every token Plan 08-02 needs: `--space-topbar-h`, `--space-drawer-w`,
   `--z-topbar`, `--z-drawer`, `--z-backdrop`, `--shadow-drawer`,
   `--duration-base` (drawer budget), `--color-primary`, `--color-text-*`.
2. `static/js/tailwind.config.js` is a drop-in replacement for the inline
   script in `base.html:13-24`. Plan 08-02 can:
   - Remove lines 13-24 in `base.html` (the inline `<script>tailwind.config=…`).
   - Add `<script src="/static/js/tailwind.config.js"></script>` on line 13
     **before** the `cdn.tailwindcss.com` `<script>` tag (Pitfall 1 — JIT ordering).
3. Every semantic utility (`bg-surface-app`, `text-body`, `bg-primary`,
   `border-subtle`, `shadow-drawer`, `rounded-lg`, `z-topbar`,
   `transition-base`) is immediately usable. Legacy `brand.*` and `surface.50-300`
   utilities still work (compat layer) — Plan 08-02 can migrate gradually.
4. Zero template was touched — the running app looks identical. Plan 08-02 is
   where the visual change actually starts.

## Self-Check: PASSED

- [x] tokens.css exists at `/home/eeguskiza/analisis_datos/static/css/tokens.css`
- [x] tailwind.config.js exists at `/home/eeguskiza/analisis_datos/static/js/tailwind.config.js`
- [x] tests/infra/test_tokens_css.py exists and passes (56/56)
- [x] docs/BRANDING.md contains `## Tokens (Phase 8)` section
- [x] static/css/app.css line 1 is `@import url('./tokens.css');`
- [x] Commit 169b99f in git log (Task 1)
- [x] Commit 009dbcf in git log (Task 2)
- [x] Commit 8deedfa in git log (Task 3)
