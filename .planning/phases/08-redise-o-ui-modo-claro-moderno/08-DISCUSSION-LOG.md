# Phase 8: Rediseño UI (modo claro moderno) — Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `08-CONTEXT.md` — this log preserves the alternatives considered.

**Date:** 2026-04-22
**Phase:** 08-redise-o-ui-modo-claro-moderno
**Areas discussed:** Token architecture + palette, Sidebar drawer behavior, Per-screen proposal format, Execution order + rollout strategy, Form + input conventions, Loading / empty / error states, Toast/notification + print stylesheet, Landing screen (added mid-discussion)

---

## Token architecture + palette (UIREDO-01)

### Q: Token layer structure in `static/css/tokens.css`

| Option | Description | Selected |
|--------|-------------|----------|
| Two-layer (raw + semantic) | Raw scale variables + semantic aliases that templates consume | ✓ |
| Semantic-only | Only semantic tokens, no raw scale — simpler but rebrand touches every token | |
| Raw-only (Tailwind-style) | Keep current brand/surface scale exactly, just move inline to tokens.css | |

### Q: Tailwind hookup to tokens

| Option | Description | Selected |
|--------|-------------|----------|
| Tailwind v3 theme.extend reads CSS vars | `rgb(var(--color-primary) / <alpha-value>)` with space-separated RGB | ✓ |
| Arbitrary-value brackets `bg-[var(...)]` | No theme extension; inline var refs in templates | |
| Bypass Tailwind for colors | Plain CSS classes in app.css referencing tokens | |

### Q: Palette scope

| Option | Description | Selected |
|--------|-------------|----------|
| Keep current brand + fill gaps | Preserve brand.50-900 and surface.50-300, fill surface 400-900 and semantics | ✓ |
| Propose new 'modo claro moderno' palette | Audit direction, propose 2 variants, pick one | |
| Start minimal, expand per-screen | Only what Centro de Mando + sidebar need, extend organically | |

### Q: Dark mode preparation

| Option | Description | Selected |
|--------|-------------|----------|
| Strictly light-only, no hooks | No [data-theme] selectors, no prefers-color-scheme | ✓ |
| Reserve shape, no dark values | Empty dark selector block as placeholder | |
| Full dual-theme scaffold | Both palettes defined, html pinned to light | |

### Q: Tokens beyond colors (multiSelect)

| Option | Description | Selected |
|--------|-------------|----------|
| Typography tokens | Font family, size scale, line-height, weights | ✓ |
| Spacing tokens | Semantic spacing aliases on top of Tailwind scale | ✓ |
| Radius + elevation tokens | --radius-sm/md/lg + --shadow-card/popover/modal | ✓ |
| Motion tokens | --duration-fast/base/slow + --ease-standard | ✓ |

### Q: Font family

| Option | Description | Selected |
|--------|-------------|----------|
| Keep Tailwind system font stack | System UI font, no CDN, zero cost | ✓ |
| Inter via Google Fonts CDN | Cleaner modern look, external dep | |
| Inter self-hosted | Local fonts, ~200KB added to repo | |

---

## Sidebar drawer behavior (UIREDO-02)

### Q: Desktop default state

| Option | Description | Selected |
|--------|-------------|----------|
| Icon-only rail always visible + popup overlay on demand | ~56px rail, expand to overlay on click/hover | |
| Fully hidden behind hamburger | No rail, hamburger in top bar opens full drawer overlay | ✓ |
| Push layout open/closed | Drawer pushes content; content reflows on every toggle | |

### Q: Persistence between page loads

| Option | Description | Selected |
|--------|-------------|----------|
| Session-scoped via Alpine only | Each load starts closed, no storage | |
| Persisted per user via cookie | Server-side cookie survives navigation | |
| Persisted via localStorage | Browser-local, no server | ✓ |

### Q: Mobile + keyboard (multiSelect)

| Option | Description | Selected |
|--------|-------------|----------|
| Full-height drawer overlays content on mobile | Below md: 768px, slide-in + backdrop | ✓ |
| Esc closes the drawer | Standard a11y, focus returns to toggle | ✓ |
| Focus trap inside drawer when open | Tab cycles only drawer items | ✓ |
| Hamburger keyboard shortcut | Power-user quick toggle | ✓ |

### Q: Hamburger toggle + topbar location

| Option | Description | Selected |
|--------|-------------|----------|
| New minimal top bar with hamburger + user menu | Global top bar across every page | ✓ |
| Floating hamburger button only | No top bar, pinned button | |
| Page-embedded toggle, no global topbar | Per-template header, varies by page | |

---

## Per-screen proposal format (UIREDO-04)

### Q: Proposal form

| Option | Description | Selected |
|--------|-------------|----------|
| Live HTML mockups via /gsd-sketch | Throwaway HTML variants in .planning/sketches/ | ✓ |
| Side-by-side ASCII layouts | Text description + bullets, fastest, not visual | |
| External design files (Figma etc.) | Mockups outside the repo, max fidelity | |

### Q: Capturing the approved proposal

| Option | Description | Selected |
|--------|-------------|----------|
| Wrap up sketch → screen-level SKILL file | /gsd-sketch-wrap-up produces per-screen skill | ✓ |
| Inline decision in the screen's plan file | Plan cites sketch, no wrap-up artefact | |
| Commit the winning sketch into docs/ui-proposals/ | Tracked HTML reference | |

### Q: Proposal count per screen

| Option | Description | Selected |
|--------|-------------|----------|
| Exactly 2 — minimum UIREDO-04 | Two meaningfully different directions | |
| 3 by default, 2 minimum | Richer decision log | |
| Variable per screen — Claude decides | Based on screen complexity | ✓ |

### Q: Screen unit

| Option | Description | Selected |
|--------|-------------|----------|
| Each Jinja template = one screen | ~24 cycles, direct mapping | ✓ |
| Functional groups | ~7 cycles by feature domain | |
| Single sprint per screen, Claude groups when obvious | Hybrid with judgement | |

### Q: Sign-off delegation (follow-up after user added Centro de Mando interaction lock)

| Option | Description | Selected |
|--------|-------------|----------|
| You sign off only Centro de Mando; Claude picks the rest | User review on CdM; Claude picks others | |
| You sign off all chrome + CdM; Claude picks data screens | Middle ground | |
| Claude decides everything after initial token sign-off | Full delegation | |

**User's choice:** Free text — "yo quiero ver las propuestas antes de aceptarlas"
**Notes:** User wants to review ALL proposals before acceptance, across all screens. Claude generates; user picks. Centro de Mando has the additional constraint of locked plano+máquinas interaction.

---

## Execution order + rollout

### Q: First landing and order

| Option | Description | Selected |
|--------|-------------|----------|
| Chrome first, then Centro de Mando, then remaining screens | 08-01 tokens, 08-02 chrome, 08-03 CdM, 08-04..NN rest | ✓ |
| Tokens, then parallel chrome + CdM, then rest | Two simultaneous plans, faster but tighter coord | |
| Vertical slice: one screen end-to-end first, then broaden | Prove the flow on low-risk screen first | |

### Q: Branching strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Continue on feature/Mark-III directly | One branch, commits in sequence | ✓ |
| One feature/Mark-III-rediseno-ui branch | Dedicated redesign branch, merge at end | |
| One branch per screen, merge per plan | Max review granularity | |

### Q: Coexistence strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Replace immediately, one-way street | New UI replaces old per plan, no fallback | ✓ |
| Feature flag NEXO_UI_V2 with both themes | Toggle via env var, double the templates during phase | |
| Mid-phase /login UI switch toggle | Per-user flag in nexo.users | |

### Q: Regression strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Full Phase 5 suite GREEN on every screen plan | pytest tests/routers + tests/auth + tests/middleware | ✓ |
| Minimal smoke per screen, full suite at phase close | Faster per-plan, risk of compounding regressions | |
| Visual diff screenshots (Playwright) | Screenshot vs baseline, heavy CI infra | |

### Q: CI a11y tool

| Option | Description | Selected |
|--------|-------------|----------|
| pa11y-ci over compose smoke URLs | Extends Phase 7 smoke job, loops authenticated URLs | ✓ |
| axe-core via Playwright | Browser per screen, heavier | |
| Static contrast check against tokens.css only | Fastest, narrower safety net | |

### Q: Post-Phase-8 Mark-III closure

| Option | Description | Selected |
|--------|-------------|----------|
| Tag v1.0.0 + follow docs/RELEASE.md | Triggers the Phase 7 quartet docs | ✓ |
| Cut v1.0.0 now (pre-Phase-8), v1.1.0 after | Two releases: functional vs visually complete | |
| Defer release ceremony entirely | No tag, continue on Mark-III branch | |

---

## Form + input conventions

### Q: Label placement

| Option | Description | Selected |
|--------|-------------|----------|
| Top-aligned labels (stacked) | Label above input, standard, clear | ✓ |
| Floating labels inside input | Modern aesthetic, needs JS, a11y edge cases | |
| Left-aligned horizontal labels | Dense forms, old-school, breaks on mobile | |

### Q: Validation + error display

| Option | Description | Selected |
|--------|-------------|----------|
| Server-side + inline under field + flash toast | FastAPI authoritative, Phase 2/4 pattern | ✓ |
| Alpine client-side pre-check + server authoritative | Duplicated logic | |
| HTML5 native validation | Zero JS, limited styling | |

### Q: Required/optional marker

| Option | Description | Selected |
|--------|-------------|----------|
| Optional fields marked (opcional), required is default | Matches Nexo's form shape (mostly required) | ✓ |
| Required fields marked with *, optional is default | Classic, noisy | |
| No markers, rely on submit validation | Adds friction | |

---

## Loading / empty / error states

### Q: Loading pattern

| Option | Description | Selected |
|--------|-------------|----------|
| Skeleton after 300ms, content swap when ready | Shimmer matching final shape | |
| Spinner in the card/panel area | Single centered spinner | ✓ |
| Stale data + 'Actualizando...' bar | Requires cached state | |

### Q: Empty state treatment

| Option | Description | Selected |
|--------|-------------|----------|
| Icon + headline + next action button | Turn dead end into next step, RBAC via can() | ✓ |
| Text-only message centered | Minimalist, fastest | |
| Custom illustration per screen | Most polished, N illustrations to produce | |

### Q: Error state

| Option | Description | Selected |
|--------|-------------|----------|
| In-panel error block + retry + RUNBOOK link | Surfaces Phase 7 runbook at failure moment | ✓ |
| Generic error toast on panel | Doesn't surface runbook | |
| Full-page error view | Overkill for per-panel errors | |

---

## Toast / notification + print stylesheet

### Q: Flash toast placement + lifetime

| Option | Description | Selected |
|--------|-------------|----------|
| Top-right, auto-dismiss 4s, queued stack, pause on hover | Standard, Phase 5 contract preserved | ✓ |
| Bottom-center, auto-dismiss 4s | Common on mobile | |
| Inline banner above affected panel | Most precise, awkward for cross-cutting | |

### Q: Print stylesheet

| Option | Description | Selected |
|--------|-------------|----------|
| No print stylesheet, PDFs stay matplotlib path | Lowest-risk, flag as backlog if feedback demands | |
| Minimal @media print that hides chrome | ~20 lines CSS, Ctrl+P-clean | ✓ |
| Full print stylesheet per screen | Duplicates matplotlib, Mark-IV | |

---

## Landing screen (added mid-discussion — D-23)

**User addition (literal):** "quiero una landing screen siempre cuando inicies sesión con un saludo en grande al usuario y según la hora que te diga buenos días/tardes/noches y un reloj en tiempo real, y luego iremos viendo cómo customizarla según el usuario."

### Q: Landing route + post-login flow

| Option | Description | Selected |
|--------|-------------|----------|
| New route /bienvenida, login redirects there, user clicks through to Centro de Mando | Doesn't violate UIREDO-03, additive surface | ✓ |
| Landing panel stamped on top of Centro de Mando | / still routes to CdM, landing card above plano | |
| Modal overlay on first load after login | CdM renders, welcome modal with saludo + reloj | |

### Q: Landing scope for Phase 8

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 8 ships greeting + reloj + nav buttons; customization = Mark-IV | MVP landing, defer widgets | ✓ |
| Phase 8 ships everything including user customization | Widget dashboard, per-user schema | |
| Phase 8 ships static landing with no clock | Simpler, loses real-time feel | |

---

## Claude's Discretion

Areas where the user delegated to Claude (captured in CONTEXT.md §G):
- Specific iconography per empty-state / error state (Heroicons outline default)
- Z-index scale for drawer / modals / toasts (in Plan 08-01 tokens.css)
- Breakpoint list beyond `md: 768px` (Tailwind defaults expected)
- Sub-nav treatment for `/ajustes` inside the drawer
- Exact keyboard shortcut key for drawer toggle
- Saludo hour bands (06–12 / 12–21 / 21–06 initial proposal, refine in Plan 08-03)

## Deferred Ideas

- Landing customization per-user (widgets, favorites, KPIs) — Mark-IV or dedicated phase
- Dark theme — locked out of Mark-III per D-04
- Per-screen print layouts — matplotlib handles reports; Mark-IV if needed
- axe-core via Playwright — pa11y-ci adopted now; migrate later if rigor demanded
- Visual diff screenshots — functional regression only in Mark-III
