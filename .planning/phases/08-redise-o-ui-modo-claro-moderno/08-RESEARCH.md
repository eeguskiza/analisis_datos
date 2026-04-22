# Phase 8: Rediseño UI (modo claro moderno) — Research

**Researched:** 2026-04-22
**Domain:** frontend refactor (Tailwind CDN + Alpine 3.14 + Jinja2) on a zero-build Nexo stack
**Confidence:** HIGH on stack / patterns / pitfalls; MEDIUM on pa11y-ci auth flow (validated against upstream docs, not yet piloted against Nexo); MEDIUM on the `[` shortcut ergonomics (evidence, not user-tested).

---

## Summary

Phase 8 is a chrome-first, screen-by-screen visual refactor against a locked design contract (`08-UI-SPEC.md`, 6/6 dimensions APPROVED) and 31 locked decisions (`08-CONTEXT.md`, D-01…D-31). The technical risk is **not** in the look-and-feel — that is locked — but in five load-bearing mechanics that must survive the refactor:

1. **Tailwind CDN + CSS-variables**: the `rgb(var(--token) / <alpha-value>)` pattern mandated by D-02 works in the Play CDN the same way it works in a full build, **provided the tokens are defined as space-separated RGB (`R G B`, no commas, no `rgb()` wrapper) in `:root` and Tailwind is told to consume them as raw strings**. [CITED: Tailwind v3 docs — customizing-colors]. This already matches the spec. The landmine is file ordering: the CDN script must see the config object before it JITs styles, which means `tokens.css` must load before `cdn.tailwindcss.com`, and the config `<script>` must run before any first paint. Plan 08-01 needs to verify this ordering explicitly.
2. **Drawer + focus trap**: Alpine ships an official **Focus plugin** with `x-trap` (previously `trap`) and `.inert` / `.noscroll` modifiers — this is the drop-in the spec is assuming. [CITED: alpinejs.dev/plugins/focus]. The plugin is **not** loaded today in `base.html:10`. Plan 08-02 must add it. Persist plugin (`$persist`) is also not loaded — needed for `nexo.ui.drawerOpen` (D-08). Without these two plugins, the spec is not implementable.
3. **Toast contract is not what the spec thinks it is**: the spec (D-30) says `window.showToast(type, title, msg)` is preserved. **False in the current codebase** — `static/js/app.js:410` loads after `base.html` and redefines `showToast(message, type='success')` with reversed arg order, silently overriding the 3-arg version. Every current caller is inconsistent (see LANDMINE 3). The refactor needs to pick one signature and update every caller in the same commit.
4. **`/` route does not render `centro_mando.html`** — it renders `luk4.html` (see `api/routers/pages.py:27`). The UI-SPEC per-screen table lists `index.html / centro_mando.html` for Plan 08-04 and `luk4.html` separately as Plan 08-15. **These are the same screen.** Plan ordering must be corrected before execution, otherwise Plan 08-04 will touch the wrong file and Plan 08-15 will have nothing left to touch.
5. **User record has no `nombre` column** (`nexo/data/models_nexo.py:82-89`). UI-SPEC §Landing Screen assumes `current_user.nombre` as a preferred salutation source. This is a phantom field. Plan 08-03 must either (a) add a schema migration + backfill the column from email local-part, or (b) commit to email-local-part only (the simpler path).

**Primary recommendation:** Unlock Phase 8 execution with a small **Plan 08-00 preflight** (no-code, 1 commit) that: (i) fixes the plan list to de-duplicate `/` and `luk4.html`, (ii) resolves the 5 Open Questions via a lightweight operator pass, (iii) rewrites the RUNBOOK anchor slugs in UI-SPEC §Error state copy to match the actual GFM slugs, and (iv) decides `nombre` vs email local-part. Then proceed with 08-01 (tokens) → 08-02 (chrome) → 08-03 (/bienvenida) → 08-04 (Centro de Mando = luk4.html) → 08-05…08-NN per-screen.

---

## User Constraints (from CONTEXT.md)

### Locked Decisions

All 31 decisions from `08-CONTEXT.md` §Implementation Decisions are locked. Research does **not** re-open them. Summary for quick scanning:

- **A. Tokens (D-01…D-06):** two-layer tokens.css (raw + semantic); Tailwind `theme.extend` with `rgb(var(--token) / <alpha-value>)`; preserve brand/surface scales; complete surface.400–900 + success/warn/error/info; light-only (no dark hooks); tokens cover type/spacing/radius/elevation/motion; font = system stack.
- **B. Drawer (D-07…D-10):** desktop default = fully hidden; state persists via `localStorage` key `nexo.ui.drawerOpen`; mobile = overlay + backdrop; Esc closes; focus trap required; keyboard shortcut TBD in Plan 08-02; new top bar with hamburger + page title + user menu; brand lives inside drawer.
- **C. Per-screen workflow (D-11…D-16):** `/gsd-sketch` per screen, user signs off every proposal; `/gsd-sketch-wrap-up` → skill per screen; N variants = Claude's call per screen (min 2); unit = one template; Centro de Mando interaction is LOCKED (only tokens/type/spacing/chrome change).
- **D. Rollout (D-17…D-22):** chrome first, then Centro de Mando, then rest; direct commits on `feature/Mark-III`, no feature flag, no branch; full Phase 5 test suite GREEN per plan; pa11y-ci extends the `smoke` job; close with tag `v1.0.0` + `docs/RELEASE.md`.
- **E. Forms/states/notifications (D-24…D-31):** top-aligned labels; server-side validation + inline error + flash toast; `(opcional)` marker, required is default; panel-centred spinner for loading (no skeletons); empty state = icon + headline + CTA gated by `can()`; error state = in-panel card + retry + runbook link; flash toast = top-right + 4s auto-dismiss + hover-pause; minimal `@media print` stylesheet.
- **F. Landing (D-23):** new `/bienvenida` post-login; dynamic saludo + real-time clock + CTA to Centro de Mando; per-user customization deferred to Mark-IV.

### Claude's Discretion

From CONTEXT.md §G, and inspected during this research:

- Exact empty-state / error-state icon per screen (Heroicons outline default).
- z-index scale (locked in UI-SPEC as `--z-*` — removed from "discretion" once UI-SPEC landed).
- Breakpoint list beyond `md: 768px` — UI-SPEC keeps Tailwind defaults.
- Sub-nav treatment for `/ajustes` — UI-SPEC Open Question #2 proposes single drawer entry + hub grid; alternative = expandable group.
- Keyboard shortcut for drawer toggle — UI-SPEC proposes `[` with `Alt+S` fallback (Open Question #1).

### Deferred Ideas (OUT OF SCOPE)

- Landing per-user customization (widgets, favoritos, KPIs personales) — Mark-IV.
- Dark theme — Mark-IV, from scratch.
- Per-screen print layouts — Mark-IV if needed.
- Axe-core via Playwright — Mark-IV upgrade path if pa11y-ci insufficient.
- Visual regression screenshots — not Mark-III.

---

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| UIREDO-01 | Tokenised light theme in `tokens.css` + `docs/BRANDING.md`; Tailwind config consumes tokens; no hardcoded hex in templates. | §1 (tokens + Tailwind integration), §7 (pa11y contrast matrix pre-baked in UI-SPEC). |
| UIREDO-02 | Sidebar → drawer/popup ≤200ms; honra `prefers-reduced-motion`; preserva nav_items + permisos Phase 5. | §2 (Alpine Focus + Persist plugins, `inert`, prefers-reduced-motion hookup), §6 (RBAC regression). |
| UIREDO-03 | Centro de Mando hereda tokens nuevos; datos/endpoints no cambian. | §7 (mapa_pabellon.html interaction LOCKED, violet/emerald legacy classes retained). |
| UIREDO-04 | Cada pantalla = sketch + selección + implementación (min 2 propuestas). | §4 (sketch workflow), §6 (regression safety around DOM changes). |
| UIREDO-05 | Animaciones ≤300ms, sin gratuitas, respeta `prefers-reduced-motion`. | §1 (motion tokens in tokens.css), §2 (drawer motion), §8 (timing catalogue in Validation Architecture). |
| UIREDO-06 | Phase 5 RBAC intacto: `can()` + 11 button gates + 8 HTML GET guards + flash toast contract. | §6 (regression map: `tests/auth/`, `tests/routers/test_sidebar_filtering.py`, `test_button_gating.py`, `tests/middleware/test_flash.py`). |
| UIREDO-07 | Scope creep → `/gsd-add-backlog` o Mark-IV. | Scope policy — no research needed, CONTEXT.md handles it. |
| UIREDO-08 | Contraste AA (WCAG 2.1) verificado por herramienta automática en CI; focus visible; Tab/Enter/Esc; `aria-label` en icon-only buttons. | §5 (pa11y-ci in CI with auth), §7 (contrast matrix pre-computed). |

---

## Architectural Responsibility Map

Phase 8 is single-tier (browser-rendered Jinja templates + inline Alpine). Still worth mapping to catch mis-assignments in per-screen plans.

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Token resolution (CSS variables) | Browser (CSS engine) | — | `:root` custom props + Tailwind `rgb(var() / <alpha>)` are purely client-side. |
| Drawer state persistence | Browser (localStorage) | — | D-08 explicitly says client-only, not server, not cookie. |
| Drawer focus trap | Browser (Alpine Focus plugin) | — | `x-trap.inert` runs in the client; no server logic. |
| RBAC sidebar filtering | Frontend Server (Jinja) | Browser (rendered DOM) | `{% if can() %}` is server-evaluated — D-02 zero-trust, `x-show` would leak permissions. |
| Flash toast cookie | API / Backend (middleware) | Browser (JS consumer) | `FlashMiddleware` (nexo/middleware/flash.py) writes/reads the cookie; browser consumes `{{ flash_message }}` into `window.showToast`. |
| Greeting band (Buenos días / …) | Frontend Server (Jinja filter) | — | D-23 server-side authoritative: Jinja filter `hora_saludo(now)` based on server local time (Europe/Madrid). |
| Real-time clock | Browser (Alpine `setInterval`) | — | D-23 client-only tick; no network call. |
| Validation errors | API / Backend | Frontend Server (Jinja) | D-25: FastAPI validation returns `errors: {field: msg}` → Jinja paints inline. No Alpine client-side validation. |
| pa11y-ci auth | CI (Node runner) | API / Backend (session cookie) | D-21: pa11y actions POST to `/login` then scan authenticated pages. |
| Print stylesheet | Browser (`@media print`) | — | D-31 minimal rule; matplotlib PDFs are untouched. |

**Key tier sanity check for per-screen plans:** permission gates stay in Jinja (server tier). Do not move any `{% if can() %}` into `x-show` — that would regress Phase 5 D-02 (zero-trust DOM).

---

## Standard Stack

### Core (already loaded in `base.html`)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Tailwind CDN | v3 (`cdn.tailwindcss.com`) | Utility-first CSS engine at page load, no build step | LAN-friendly, zero-infra, matches `base.html:7` [VERIFIED: templates/base.html:7] |
| Alpine.js | 3.14.8 | Reactive JS without a framework | Already the glue for drawer/toasts/preflight modal [VERIFIED: templates/base.html:10] |
| HTMX | 2.0.4 | Incremental panel swaps + SSE | Loading tier 1 (panel spinner) fits HTMX `htmx:beforeRequest` hooks [VERIFIED: templates/base.html:8] |
| HTMX SSE ext | 2.2.2 | SSE streaming for pipeline logs | Already wired; no change [VERIFIED: templates/base.html:9] |
| Chart.js | 4.4.7 | Charts in `ajustes_rendimiento` + OEE dashboard | Re-tokenise `backgroundColor` / `borderColor` only [VERIFIED: templates/base.html:11] |

### New (required for Phase 8)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Alpine Focus plugin | 3.14.x (match core) | `x-trap` + `x-trap.inert` + `x-trap.noscroll` modifiers for drawer focus trap | Official Alpine plugin, drop-in via CDN; `x-trap.inert` auto-sets `aria-hidden="true"` on siblings [CITED: alpinejs.dev/plugins/focus] |
| Alpine Persist plugin | 3.14.x | `$persist(value).as('key')` for `nexo.ui.drawerOpen` via localStorage | Official Alpine plugin; primitive values work out of the box [CITED: alpinejs.dev/plugins/persist] |
| pa11y-ci | 3.1.x (current stable on npm as of research date) | WCAG 2.1 AA check on ~10-15 URLs inside CI smoke job | Spec-sanctioned tool (D-21) [CITED: github.com/pa11y/pa11y-ci] |

**Installation (Plan 08-02 for Alpine plugins):**

```html
<!-- base.html <head>, BEFORE the Alpine core script -->
<script defer src="https://cdn.jsdelivr.net/npm/@alpinejs/focus@3.14.8/dist/cdn.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/@alpinejs/persist@3.14.8/dist/cdn.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.8/dist/cdn.min.js"></script>
```

**LANDMINE:** Alpine plugins MUST load **before** the Alpine core script. Reverse order silently skips the plugin registration. `base.html:10` currently loads `alpinejs@3.14.8` first — Plan 08-02 must shift that order.

**pa11y-ci installation (Plan 08-02 CI step):**

```yaml
# In .github/workflows/ci.yml, extending the `smoke` job after "Wait for web healthy":
- name: Install pa11y-ci
  run: npm install -g pa11y-ci@3
- name: Run accessibility checks
  run: pa11y-ci --config .pa11yci.json
```

**Version verification tasks for Plan 08-02:**

```bash
# Pin exact versions to avoid drift
npm view pa11y-ci version          # confirm latest stable
# @alpinejs/focus and @alpinejs/persist are published in lockstep with alpinejs itself (3.14.8)
```

### Alternatives considered

| Instead of | Could use | Tradeoff |
|------------|-----------|----------|
| `x-trap.inert` (Focus plugin) | Manual `inert` attribute management via `x-effect` | `inert` is widely supported [CITED: MDN], but Focus plugin handles focus-return, nested dialogs, and automatic aria-hidden — rolling it by hand is D-05 Don't-Hand-Roll territory. |
| `$persist()` plugin | Manual `localStorage.getItem/setItem` in `x-init` + `x-effect` | The plugin adds one CDN line; hand-rolling adds ~15 lines of boilerplate + error handling for quota-exceeded / disabled-storage. Use the plugin. |
| pa11y-ci | axe-core + Playwright | D-21 locks pa11y-ci. Axe-core is listed in CONTEXT.md §Deferred as a Mark-IV upgrade path if pa11y-ci proves insufficient. |
| Tailwind build step (PostCSS) | Keep Tailwind CDN | D-06 / stack lock. A build step would block iteration speed; CDN is LAN-friendly. |

---

## Architecture Patterns

### System architecture diagram

```
┌───────────────────── BROWSER ──────────────────────────────────┐
│                                                                 │
│   Page load                                                     │
│      │                                                          │
│      ▼                                                          │
│   [tokens.css]  ──┐                                             │
│      │             │ :root CSS vars (space-separated RGB)       │
│      ▼             ▼                                            │
│   [Tailwind CDN <script>] ──► JIT reads `tailwind.config` ──►   │
│      │   generates classes using rgb(var(--token) / alpha)      │
│      │                                                          │
│      ▼                                                          │
│   [Alpine Focus plugin] ──┐                                     │
│   [Alpine Persist plugin] │  plugin regs (BEFORE core)          │
│      │                    │                                     │
│      ▼                    ▼                                     │
│   [Alpine core 3.14.8]   reads `x-trap`, `$persist`             │
│      │                                                          │
│      ▼                                                          │
│   base.html renders:                                            │
│     ├── <top bar>   [≡] [page_title] [conn-badge] [user menu]  │
│     ├── <drawer x-trap.inert x-data="$persist(false).as(...)">  │
│     │     [nav gated by {% if can() %} from Jinja server-side] │
│     └── <main>  {% block content %}</main>                      │
│                                                                 │
│   [app.js]  registers Alpine.data('preflightModal', ...)        │
│              registers window.showToast                         │
│              registers keyboard shortcut listener ('[')         │
│                                                                 │
│   [flash_message] cookie  ─► Jinja {% if flash_message %} ──►   │
│                              showToast('info', 'Aviso', msg)    │
└─────────────────────────────────────────────────────────────────┘
        ▲                                                      ▲
        │                                                      │
        │ GET /api/conexion/status         GET /login, POST /login
        │ (HTMX 30s polling badge)         (pa11y-ci actions)
        │                                                      │
┌───────┴──────────────────────────────────────────────────────┴──┐
│                    FASTAPI + JINJA2 (api/)                      │
│                                                                 │
│   AuthMiddleware  ──► request.state.user (can + require_perm)   │
│   FlashMiddleware ──► request.state.flash → {{ flash_message }} │
│                                                                 │
│   /login (public)           /bienvenida (new, D-23)             │
│       POST ──► session      GET ──► render with saludo string   │
│       RedirectResponse ──► /bienvenida (new, replaces /)        │
│                                                                 │
│   /                        /pipeline, /historial, /bbdd, …       │
│   (renders luk4.html       (rendered via api/deps.py::render)    │
│    ← the "centro de                                              │
│    mando" template)                                              │
└──────────────────────────────────────────────────────────────────┘
```

### Recommended project structure (diff vs today)

```
static/
├── css/
│   ├── app.css        # existing — rewritten in Plan 08-01 to consume tokens
│   ├── tokens.css     # NEW — two-layer tokens (raw + semantic)   [Plan 08-01]
│   └── print.css      # NEW — minimal @media print                [Plan 08-01]
├── js/
│   ├── app.js         # existing — extended in Plan 08-02 (focus trap helpers, [ shortcut, showToast rewrite)
│   └── tailwind.config.js  # NEW — extracted from base.html inline [Plan 08-01]
└── img/
    ├── gif-corona.gif        # existing
    ├── gif-corona.png        # NEW — first-frame for prefers-reduced-motion [Plan 08-01]
    └── brand/                # existing (unchanged)

templates/
├── base.html          # Plan 08-02 rewrites: hamburger + drawer + top bar
├── bienvenida.html    # NEW — Plan 08-03
└── … (24 existing templates, redesigned per per-screen plan)
```

### Pattern 1: Two-layer tokens consumed by Tailwind via CSS variables

**What:** tokens.css declares raw palette + semantic aliases; Tailwind config maps utilities to semantic tokens via `rgb(var(--token) / <alpha-value>)` syntax.

**When to use:** every Plan 08-NN (all consumers). New templates must only reference the semantic utilities (`bg-primary`, `text-muted`), never raw (`bg-brand-600`).

**Example** (the exact code the planner embeds into Plan 08-01 tasks):

```css
/* static/css/tokens.css */
:root {
  /* Raw — RGB triplets, space-separated, NO commas, NO rgb() wrapper */
  --color-brand-600: 26 58 92;
  --color-surface-0: 255 255 255;
  --color-surface-50: 248 250 252;
  /* …full scale per UI-SPEC §Color / Semantic tokens… */

  /* Semantic aliases */
  --color-primary: var(--color-brand-600);
  --color-surface-base: var(--color-surface-0);
  /* …etc… */
}
```

```html
<!-- base.html (simplified) -->
<link rel="stylesheet" href="/static/css/tokens.css">  <!-- FIRST -->
<script src="/static/js/tailwind.config.js"></script>  <!-- Before Tailwind CDN -->
<script src="https://cdn.tailwindcss.com"></script>    <!-- Tailwind JIT reads `tailwind.config` -->
<link rel="stylesheet" href="/static/css/app.css">     <!-- AFTER Tailwind so @apply works -->
```

```js
// static/js/tailwind.config.js
tailwind.config = {
  theme: {
    extend: {
      colors: {
        primary: 'rgb(var(--color-primary) / <alpha-value>)',
        'primary-hover': 'rgb(var(--color-primary-hover) / <alpha-value>)',
        'surface-base': 'rgb(var(--color-surface-base) / <alpha-value>)',
        // …
        // Preserve legacy brand.*/surface.* scales for mapa_pabellon (LOCKED)
        brand: {
          50: '#eef5ff', 100: '#d9e8ff', /* …existing verbatim… */ 900: '#0a1724',
        },
        surface: {
          50: '#f8fafc', /* …existing verbatim… */ 300: '#cbd5e1',
        },
      },
    },
  },
}
```

[CITED: Tailwind v3 customizing-colors docs — `rgb(var(--token) / <alpha-value>)` is the sanctioned pattern for CSS-variable colours with alpha utilities.]

### Pattern 2: Drawer with focus trap via Alpine Focus plugin

**What:** single Alpine component on the drawer root; `x-trap.inert.noscroll="open"` does focus trap + aria-hidden on siblings + body scroll lock in one directive.

**When to use:** Plan 08-02 for the main drawer; reuse shape for modals in later plans.

**Example:**

```html
<aside
  x-data="{ open: $persist(false).as('nexo.ui.drawerOpen') }"
  x-trap.inert.noscroll="open && window.innerWidth < 768"
  @keydown.escape.window="open = false"
  @keydown.window.prevent.stop="$event.key === '[' && !['INPUT','TEXTAREA','SELECT'].includes($event.target.tagName) && (open = !open)"
  class="fixed inset-y-0 left-0 w-[280px] bg-surface-base shadow-drawer transition-transform duration-200"
  :class="open ? 'translate-x-0' : '-translate-x-full'"
>
  <!-- Close button = first focusable when trap engages -->
  <button @click="open = false" aria-label="Cerrar menú">…</button>
  <!-- Nav items gated by Jinja {% if can() %} -->
</aside>
```

[CITED: alpinejs.dev/plugins/focus — `x-trap.inert` sets `aria-hidden="true"` on siblings while trapped; `.noscroll` locks body scroll.]

### Pattern 3: Flash toast contract (preserved)

**What:** FastAPI middleware writes cookie `nexo_flash`; FlashMiddleware reads it into `request.state.flash` and deletes it; Jinja injects `{{ flash_message }}`; client JS calls `window.showToast(type, title, msg)`.

**When to use:** anywhere the server needs to push a user-visible message after a redirect.

**Current contract (reverse-engineered from code, not what UI-SPEC claims):**

- Cookie name: `nexo_flash` [VERIFIED: nexo/middleware/flash.py:21]
- Cookie shape: opaque string, plaintext (no JSON envelope) — the full cookie value IS the message [VERIFIED: api/main.py http_exception_handler_403].
- Cookie policy: HttpOnly, Secure in non-debug, SameSite=Lax, max_age=60s [VERIFIED: nexo/middleware/flash.py docstring].
- Jinja consumer: `{{ flash_message|tojson }}` in base.html:242 [VERIFIED: templates/base.html:235-246].
- JS consumer **as declared in base.html:188**: `window.showToast(type, title, msg)` — 3 args.
- JS consumer **as actually active (app.js loads last) at app.js:410**: `showToast(message, type='success')` — 2 args, reversed. See LANDMINE 3 below.

### Pattern 4: Server-side validation + inline errors + flash summary

**What:** FastAPI returns 400 with `errors: {field: msg}`; Jinja paints `<p class="mt-1 text-sm text-error" role="alert">{{ errors.email }}</p>` below the field; FlashMiddleware surfaces the summary as a toast.

**When to use:** every form in every per-screen plan. D-25 locks this.

Example skeleton already present in `api/routers/auth.py:212-225` — re-render with `error=...` via Jinja context. The refactor just restyles the error card — logic unchanged.

### Anti-patterns to avoid

- **Hardcoding hex in templates** — UI-SPEC §Tailwind mapping says "new templates must consume only semantic utilities". `mapa_pabellon.html` legacy violet/emerald classes stay (LOCKED) but are the only allowed exception.
- **Using `x-show` for RBAC** — D-02 zero-trust requires `{% if can() %}` server-side. An `x-show` gate would leak the DOM for users without permission and only hide it; that regresses UIREDO-06.
- **Using `Math.random()` in SSR** — the landing saludo is server-rendered (D-23). Client-side re-compute could show a different band than the server if clocks disagree. The spec says "server render is authoritative" — keep it that way.
- **Toast `bg-red-600 text-white` full-fill** — UI-SPEC moves to `bg-surface-base` + left accent bar. The current full-fill toasts (base.html:189-203) are **exactly** what D-30 replaces.
- **`outline: none` without replacement** — `:focus-visible` ring is mandatory on every interactive element. Any `.btn` rewrite must include the ring.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Focus trap inside drawer/modal | Custom Tab/Shift+Tab cycle manager | **Alpine Focus plugin (`x-trap.inert`)** | Plugin handles: focus return on close, nested traps, aria-hidden on siblings, initial-focus picking. Rolling by hand misses at least two of these. [CITED: alpinejs.dev/plugins/focus] |
| Persisting drawer state across page loads | `localStorage.getItem/setItem` in `x-init` + manual watchers | **Alpine Persist plugin (`$persist(false).as('…')`)** | Plugin handles: quota-exceeded silent fallback, watcher wiring, deserialization. The D-08 requirement is two lines with the plugin vs ~15 without. [CITED: alpinejs.dev/plugins/persist] |
| Accessibility AA scan on ~15 routes | Custom Playwright + colour-contrast script | **pa11y-ci** | D-21 locks this tool. pa11y-ci natively supports authentication actions, default configs per URL, `--no-sandbox` for CI. [CITED: github.com/pa11y/pa11y-ci] |
| Cleanup of Alpine `setInterval` (reloj) | Manual global interval cleanup on navigation | **Alpine `destroy()` lifecycle hook inside `x-data`** | `destroy()` fires when component is removed from DOM, including HTMX partial swaps. Ignoring this leaks the clock across navigations. [CITED: alpinejs.dev/globals/alpine-data] |
| First frame of `gif-corona.gif` for `prefers-reduced-motion` | JS pause-on-play hack (`onplaying="this.pause()"`) | **`<picture>` with `media="(prefers-reduced-motion: reduce)"` pointing to a pre-extracted PNG** | UI-SPEC §State Contracts Tier 2 already specifies this. JS-pause is race-conditiony and frame-1 is often a blank frame. Extract the PNG at build time (Plan 08-01 task). |
| Inert the rest of the page when drawer is open | Manual `aria-hidden` + `tabindex=-1` on every sibling | **`x-trap.inert` modifier** | Does exactly this in one line. |
| Keyboard-shortcut registration | Custom `keydown` listener with own input-guard | **Tiny helper: `@keydown.window.prevent.stop="…"` on a top-level element** with inline input-guard | Alpine already gives us this for free; no library needed. |

**Key insight:** the only new line of JS Phase 8 needs to hand-write is the `[` shortcut handler and the reloj `setInterval` — **everything else has a plugin or a Jinja pattern**.

---

## Common Pitfalls

### Pitfall 1: Tailwind CDN JIT ordering

**What goes wrong:** `tailwind.config` is set **after** `<script src="cdn.tailwindcss.com">`, so JIT boots with defaults and ignores custom tokens. Page flashes un-styled then re-styles once config arrives; `rgb(var(--token)...)` classes may never resolve.

**Why it happens:** HTML authors paste the config below the script out of habit (normal JS globals).

**How to avoid:** assignments to `tailwind.config = {...}` must be in a `<script>` (without `defer`) that runs **before** the `<script defer src="cdn.tailwindcss.com">`. Current `base.html:13-24` is correct — the new extraction in Plan 08-01 must preserve this ordering.

**Warning signs:** brief FOUC on page load, or classes like `bg-primary` rendering as `background-color: initial`.

### Pitfall 2: `rgb(var(...))` CSS variables with commas or `rgb()` wrapper

**What goes wrong:** `--color-primary: rgb(26, 58, 92);` instead of `--color-primary: 26 58 92;`. Alpha utilities break because Tailwind splices `<alpha-value>` into a broken expression.

**Why it happens:** training-data reflex — most tutorials for theming use `--color-primary: #rrggbb`; the R-G-B space-separated form is Tailwind-specific.

**How to avoid:** token values in `tokens.css` **must** be three numbers separated by spaces, with no commas, no prefix, no suffix. UI-SPEC §Semantic tokens already specifies this correctly — the landmine is when a contributor "fixes" a token format during a sketch.

**Warning signs:** `bg-primary` works, `bg-primary/50` silently fails to 100% opacity or breaks entirely.

### Pitfall 3: Conflicting `showToast` signatures (ALREADY IN THE CODEBASE)

**LANDMINE:** **`base.html:188` declares `window.showToast = function(type, title, msg)`. `static/js/app.js:410` declares `function showToast(message, type='success')`.** Because `app.js` loads AFTER `base.html`'s inline script (base.html:248), `app.js`'s version wins globally. But the Jinja consumer at `base.html:242` calls `showToast('info', 'Aviso', {{ flash_message|tojson }})` — three args — against the 2-arg version, which means `type='Aviso'` gets fed as the type (unknown) and the flash_message is silently dropped.

Evidence:

```
templates/base.html:188  window.showToast = function(type, title, msg) { ... }
templates/base.html:242  showToast('info', 'Aviso', {{ flash_message|tojson }});
static/js/app.js:410     function showToast(message, type = 'success') { ... }
```

Callers split between signatures:

| Caller | Signature actually called | Works today? |
|--------|---------------------------|--------------|
| `base.html:242` flash | 3-arg (type, title, msg) | **No** — app.js override truncates |
| `luk4.html:320-334` | 3-arg (type, title, msg) | **No** |
| `ciclos_calc.html:512-536` | 2-arg (message, type) | Yes (matches app.js) |
| `informes.html:277-284` | 2-arg (message, type) | Yes |
| `app.js:550,554,557` | 2-arg (message, type) | Yes |

**How to avoid in Phase 8:** Plan 08-02 (which rewrites toast chrome per D-30) MUST:
1. Pick the 3-arg signature `showToast(type, title, msg)` (the one UI-SPEC documents) OR the 2-arg signature — pick one.
2. Update **every caller** in the same commit (12 known sites across 5 files).
3. Remove the `app.js:410` definition OR the `base.html:188` definition — not both.
4. Add a regression test that asserts the 3-arg form works with the flash toast.

**Recommendation:** keep the 3-arg `(type, title, msg)` form (the one the UI-SPEC locks) and delete the `app.js` version.

### Pitfall 4: Alpine plugin ordering

**What goes wrong:** `alpinejs@3.14.8/dist/cdn.min.js` loaded before `@alpinejs/focus` → the plugin registers its directives **after** Alpine has already initialised, so `x-trap` silently no-ops.

**How to avoid:** plugin scripts MUST appear before the Alpine core script in the HTML. With `defer`, they execute in document order.

```html
<!-- CORRECT: plugins first, core last -->
<script defer src=".../@alpinejs/focus@3.14.8/dist/cdn.min.js"></script>
<script defer src=".../@alpinejs/persist@3.14.8/dist/cdn.min.js"></script>
<script defer src=".../alpinejs@3.14.8/dist/cdn.min.js"></script>
```

**Warning signs:** `x-trap` renders as-is in DOM (Alpine didn't consume it); drawer open/close works but focus escapes to `<body>`.

### Pitfall 5: `$persist` with complex values on first load

**What goes wrong:** treating `$persist({ open: false })` same as `$persist(false)`. Persist plugin works with primitives and with plain objects, but not with reactive proxies. Also, the default value is used **only on first visit** — if a user has a stale value from a previous Mark-II session with a different key name, they'll see that.

**How to avoid:** keep `$persist` to primitives for Phase 8. Key name `nexo.ui.drawerOpen` is new (never used before), so no legacy collision.

**Warning signs:** drawer opens on page load inexplicably (stale stored value); plugin throws on reactive wrappers.

### Pitfall 6: `setInterval` leak in reloj

**What goes wrong:** the reloj (D-23) runs `setInterval(tick, 1000)` in `init()` but never clears it. Navigating away with HTMX partial swap leaves the interval running in memory; navigating back creates a second. After a few navigations the clock counter drifts or multiplies.

**How to avoid:** declare `destroy()` in the `x-data` that `clearInterval(this.timer)`:

```js
Alpine.data('bienvenidaPage', () => ({
  timer: null,
  now: new Date(),
  init() {
    this.timer = setInterval(() => { this.now = new Date() }, 1000);
  },
  destroy() {
    clearInterval(this.timer);
  },
}))
```

[CITED: alpinejs.dev/globals/alpine-data]

### Pitfall 7: `[` shortcut collision inside inputs

**What goes wrong:** user is typing in a bbdd query textarea, types `[` (valid SQL / text character), drawer opens, losing focus on their textarea.

**How to avoid:** guard the keydown handler against active inputs:

```js
// Inside the window listener
if (['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName)) return;
if (event.target.isContentEditable) return;
```

UI-SPEC §Keyboard navigation already specifies `e.target.tagName` guard — plan must include this in the task list, not as an afterthought.

### Pitfall 8: GitHub-flavored markdown slug drift in RUNBOOK links

**LANDMINE:** UI-SPEC §Error state copy uses anchors like `docs/RUNBOOK.md#escenario-1-mes-caido`. **Actual RUNBOOK.md heading:** `## Escenario 1: MES caido (SQL Server dbizaro inaccesible)`. GitHub-flavored markdown slugification includes everything in the heading — the actual slug is `#escenario-1-mes-caido-sql-server-dbizaro-inaccesible`.

Confirmed via `grep "^## " docs/RUNBOOK.md`:

```
## Escenario 1: MES caido (SQL Server dbizaro inaccesible)
## Escenario 2: Postgres no arranca
## Escenario 3: Certificado Caddy expira / warning en browsers
## Escenario 4: Pipeline atascado (HALLAZGO CRITICO: semaforo in-process)
## Escenario 5: Lockout del unico propietario (HALLAZGO CRITICO: no hay `unlock_user`)
```

**How to avoid:** Plan 08-00 preflight (or Plan 08-02 as first task) either:
1. Shortens the RUNBOOK headings to `## Escenario 1: MES caído` (anchor-friendly, matches UI-SPEC), **or**
2. Updates UI-SPEC §Error state copy to use the actual GFM slugs.

**Recommendation:** option 2 — UI-SPEC is a living contract; RUNBOOK is shipped doc cited from Phase 7.

### Pitfall 9: `flash_message` cookie race with HI-01 guard

**What goes wrong:** a 403 redirect path sets the flash cookie twice (handler + middleware cleanup). `FlashMiddleware` has a guard at `nexo/middleware/flash.py:36-48` that avoids this. Any refactor of the flash toast MUST preserve this guard — deleting it regresses a Phase 5 fix.

**How to avoid:** Plan 08-02 (toast restyling) touches the JS side only. The Python middleware stays untouched. Add a test to `tests/middleware/test_flash.py` that asserts double-Set-Cookie doesn't happen.

### Pitfall 10: `index()` route renders `luk4.html`, not `centro_mando.html`

**LANDMINE:** `api/routers/pages.py:25-27`:

```python
@router.get("/")
def index(request: Request):
    return render("luk4.html", request, _common_extra("dashboard"))
```

UI-SPEC §Per-Screen Adaptations lists Plan 08-04 target as `index.html / centro_mando.html`, but there is no `templates/index.html` and no `templates/centro_mando.html`. The actual Centro de Mando template is `templates/luk4.html` (which embeds `_partials/mapa_pabellon.html` four times, one per pabellón).

**How to avoid:** Plan 08-00 preflight merges Plan 08-04 and Plan 08-15 into a single plan against `luk4.html` + `_partials/mapa_pabellon.html`, preserving the D-16 interaction lock.

---

## Runtime State Inventory

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — Phase 8 is pure frontend refactor | None |
| Live service config | `.pa11yci.json` at repo root — NEW (Plan 08-02) | Ship as part of commit |
| OS-registered state | None — no cron/systemd/scheduled tasks added | None |
| Secrets/env vars | None new; `NEXO_LOGO_PATH`, `NEXO_ECS_LOGO_PATH`, `NEXO_APP_NAME`, `NEXO_COMPANY_NAME` continue | None |
| Build artifacts | `static/img/gif-corona.png` — NEW (Plan 08-01 task: extract first frame of `gif-corona.gif`) | One-time extraction via `ffmpeg` or similar |
| **Stored data (frontend)** | `localStorage['nexo.ui.drawerOpen']` — boolean per user per browser | Document in `docs/BRANDING.md` §Runtime state |
| **Stored data (previous Mark-II)** | No Mark-II localStorage keys collide — confirmed by grep `localStorage` across `/templates` + `/static` | None |

**Nothing found in stored data / OS-registered state / secrets** — confirmed by inspection of `nexo/middleware/`, `api/routers/`, `Makefile`, `docker-compose.yml`. Phase 8 is code+frontend-asset-only; no data migrations required.

---

## Code Examples

### Example 1: tokens.css with correct two-layer structure

```css
/* static/css/tokens.css — Plan 08-01 baseline */
:root {
  /* ── Layer 1: raw palette (space-separated RGB, no commas) ─────────── */
  --color-brand-50:  238 245 255;
  --color-brand-600: 26 58 92;
  --color-brand-700: 20 45 71;
  --color-brand-800: 15 34 54;
  --color-brand-900: 10 23 36;

  --color-surface-0:   255 255 255;
  --color-surface-50:  248 250 252;
  --color-surface-100: 241 245 249;
  --color-surface-200: 226 232 240;
  --color-surface-300: 203 213 225;
  --color-surface-400: 148 163 184;
  --color-surface-500: 100 116 139;
  /* …full scale per UI-SPEC §Color raw palette… */

  --color-success-600: 5 150 105;
  --color-warn-600:    217 119 6;
  --color-error-600:   220 38 38;
  --color-info-600:    37 99 235;

  /* ── Layer 2: semantic aliases ─────────────────────────────────────── */
  --color-primary:         var(--color-brand-600);
  --color-primary-hover:   var(--color-brand-700);
  --color-surface-base:    var(--color-surface-0);
  --color-surface-app:     var(--color-surface-50);
  --color-text-body:       var(--color-brand-800);
  --color-text-muted:      var(--color-surface-500);
  --color-border-subtle:   var(--color-surface-200);
  --color-border-strong:   var(--color-surface-300);
  --color-focus-ring:      51 109 255;  /* brand.500 literal */

  /* ── Motion tokens (D-05) ──────────────────────────────────────────── */
  --duration-fast: 150ms;
  --duration-base: 200ms;
  --duration-slow: 300ms;
  --ease-standard:   cubic-bezier(0.2, 0, 0, 1);
  --ease-accelerate: cubic-bezier(0.3, 0, 1, 1);

  /* …radius, shadows, z-index per UI-SPEC §Color / Semantic tokens… */
}

/* ── Reduced motion (D-05) ───────────────────────────────────────────── */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
  .spinner, .loading-hero img {
    animation-duration: 1.5s !important;
  }
}

/* ── Focus ring (D-UI-SPEC §Focus ring) ─────────────────────────────── */
:focus-visible {
  outline: 2px solid rgb(var(--color-focus-ring));
  outline-offset: 2px;
}
```

[Source: adapted from UI-SPEC §Semantic tokens + Tailwind v3 customizing-colors docs.]

### Example 2: Drawer component with Alpine Focus + Persist

```html
<!-- base.html — Plan 08-02 draft -->
<body x-data="appShell()">
  <!-- Top bar -->
  <header class="sticky top-0 z-topbar h-[56px] bg-surface-base border-b border-subtle flex items-center px-6">
    <button
      @click="drawer = !drawer"
      aria-label="Abrir menú"
      class="btn btn-icon"
    >
      <svg><!-- hamburger icon --></svg>
    </button>
    <h1 class="ml-4 text-heading text-body truncate">{% block page_title %}{% endblock %}</h1>
    <!-- conn-badge + user menu on the right cluster… -->
  </header>

  <!-- Drawer -->
  <aside
    x-show="drawer"
    x-trap.inert.noscroll="drawer && isMobile"
    @keydown.escape.window="drawer = false"
    class="fixed inset-y-0 left-0 w-[280px] bg-surface-base shadow-drawer z-drawer transition-transform duration-200"
    :class="drawer ? 'translate-x-0' : '-translate-x-full'"
  >
    <button @click="drawer = false" aria-label="Cerrar menú" class="btn btn-icon absolute top-4 right-4">
      <svg><!-- x-mark --></svg>
    </button>
    <!-- Nav (Jinja-gated by can(), unchanged from Phase 5) -->
    {% if can(current_user, "pipeline:read") %}
      <a href="/pipeline" class="nav-item">Análisis</a>
    {% endif %}
    <!-- … -->
  </aside>

  <!-- Backdrop (mobile only) -->
  <div
    x-show="drawer && isMobile"
    @click="drawer = false"
    x-transition.opacity
    class="fixed inset-0 z-backdrop bg-surface-900/40 backdrop-blur-sm"
  ></div>

  <main class="pt-[56px]">
    {% block content %}{% endblock %}
  </main>

  <script>
    function appShell() {
      return {
        drawer: Alpine.$persist(false).as('nexo.ui.drawerOpen'),
        isMobile: window.matchMedia('(max-width: 767px)').matches,
        init() {
          // Watch viewport changes
          window.matchMedia('(max-width: 767px)').addEventListener('change', (e) => {
            this.isMobile = e.matches;
          });
          // Global '[' shortcut
          window.addEventListener('keydown', (e) => {
            const tag = e.target.tagName;
            if (['INPUT','TEXTAREA','SELECT'].includes(tag)) return;
            if (e.target.isContentEditable) return;
            if (e.key === '[' && !e.ctrlKey && !e.metaKey && !e.altKey) {
              e.preventDefault();
              this.drawer = !this.drawer;
            }
          });
        },
      };
    }
  </script>
</body>
```

[Source: synthesised from UI-SPEC §Drawer + alpinejs.dev/plugins/focus + alpinejs.dev/plugins/persist.]

### Example 3: `/bienvenida` landing with clock + greeting

```python
# api/routers/pages.py (new route — Plan 08-03)
from datetime import datetime
from zoneinfo import ZoneInfo

@router.get("/bienvenida", dependencies=[Depends(require_permission("centro_mando:read"))])
def bienvenida_page(request: Request):
    now = datetime.now(ZoneInfo("Europe/Madrid"))
    extra = _common_extra("bienvenida")
    extra["now"] = now
    extra["saludo"] = _hora_saludo(now)
    return render("bienvenida.html", request, extra)


def _hora_saludo(now: datetime) -> str:
    """Band rules per D-23 — server-side authoritative."""
    h = now.hour
    if 6 <= h < 12:
        return "Buenos días"
    if 12 <= h < 21:
        return "Buenas tardes"
    return "Buenas noches"
```

```jinja2
{# templates/bienvenida.html — Plan 08-03 draft #}
{% extends "base.html" %}
{% block page_title %}Bienvenida{% endblock %}
{% block content %}
<section
  x-data="bienvenidaPage()"
  class="max-w-[640px] mx-auto flex flex-col items-center justify-center min-h-[calc(100vh-56px)] p-6"
>
  <h1 class="text-display text-heading">
    {{ saludo }}{% if current_user and display_name %}, {{ display_name }}{% endif %}
  </h1>
  <p class="mt-2 text-body text-muted">
    Es {{ now.strftime('%A, %d de %B de %Y')|capitalize_es }}
  </p>
  <p class="mt-4 font-mono text-display tabular-nums" x-text="clock">{{ now.strftime('%H:%M:%S') }}</p>
  <a href="/" class="btn btn-primary btn-lg mt-12" autofocus>
    Ir a Centro de Mando
    <svg><!-- chevron-right --></svg>
  </a>

  {# TODO Mark-IV: widgets configurables #}
</section>

<script>
  function bienvenidaPage() {
    return {
      clock: '{{ now.strftime("%H:%M:%S") }}',
      timer: null,
      init() {
        this.timer = setInterval(() => {
          const d = new Date();
          const hh = String(d.getHours()).padStart(2, '0');
          const mm = String(d.getMinutes()).padStart(2, '0');
          const ss = String(d.getSeconds()).padStart(2, '0');
          this.clock = `${hh}:${mm}:${ss}`;
        }, 1000);
      },
      destroy() {
        clearInterval(this.timer);
      },
    };
  }
</script>
{% endblock %}
```

**`display_name` resolution** (see §Open Questions):

```python
# Plan 08-03: choose ONE of these
# Option A (if `nombre` column exists):
extra["display_name"] = current_user.nombre or current_user.email.split('@')[0].capitalize()
# Option B (simpler, recommended):
extra["display_name"] = current_user.email.split('@')[0].capitalize() if current_user else None
```

[Source: UI-SPEC §Landing Screen, adapted to match the actual user model at `nexo/data/models_nexo.py:82-89`.]

### Example 4: pa11y-ci config + CI extension

```json
// .pa11yci.json — Plan 08-02
{
  "defaults": {
    "standard": "WCAG2AA",
    "timeout": 30000,
    "wait": 500,
    "chromeLaunchConfig": {
      "args": ["--no-sandbox", "--disable-setuid-sandbox"]
    },
    "ignore": [
      "color-contrast"
    ],
    "_comment_ignore": "color-contrast is re-enabled by dropping from ignore once tokens.css lands (Plan 08-02 task)."
  },
  "urls": [
    "http://localhost:8001/login",
    {
      "url": "http://localhost:8001/bienvenida",
      "actions": [
        "navigate to http://localhost:8001/login",
        "set field #email to smoke@nexo.local",
        "set field #password to smoke-ci-password",
        "click element button[type=submit]",
        "wait for url to be http://localhost:8001/bienvenida"
      ]
    },
    {
      "url": "http://localhost:8001/",
      "actions": [ "navigate to http://localhost:8001/bienvenida" ]
    },
    {
      "url": "http://localhost:8001/pipeline",
      "actions": [ "navigate to http://localhost:8001/bienvenida" ]
    },
    {
      "url": "http://localhost:8001/historial",
      "actions": [ "navigate to http://localhost:8001/bienvenida" ]
    },
    {
      "url": "http://localhost:8001/bbdd",
      "actions": [ "navigate to http://localhost:8001/bienvenida" ]
    },
    {
      "url": "http://localhost:8001/ajustes",
      "actions": [ "navigate to http://localhost:8001/bienvenida" ]
    }
  ]
}
```

```yaml
# .github/workflows/ci.yml — extending the `smoke` job after health check
- name: Seed CI smoke user (Postgres)
  run: |
    docker compose exec -T db psql -U oee -d oee_planta -c "
      INSERT INTO nexo.users (email, password_hash, role, active, must_change_password)
      VALUES ('smoke@nexo.local', '<argon2id-hash-of-smoke-ci-password>', 'propietario', true, false)
      ON CONFLICT (email) DO NOTHING;
    "

- name: Install pa11y-ci
  run: npm install -g pa11y-ci@3

- name: Run accessibility checks
  run: pa11y-ci --config .pa11yci.json
```

**LANDMINE:** the seed-user step writes a pre-computed argon2id hash. The plan needs to generate this hash once locally (via `python -c "from nexo.services.auth import hash_password; print(hash_password('smoke-ci-password'))"`) and commit the hash string into the workflow. Do NOT put the plain password hash into the password field — it won't verify.

[Source: github.com/pa11y/pa11y-ci README + 08-UI-SPEC.md §Accessibility matrix.]

### Example 5: `gif-corona.png` first-frame extraction

```bash
# Plan 08-01 one-shot task (commit the resulting PNG to the repo):
ffmpeg -i static/img/gif-corona.gif -vframes 1 -q:v 2 static/img/gif-corona.png

# Alternative if ffmpeg not available:
# Python one-liner with Pillow:
python -c "from PIL import Image; im = Image.open('static/img/gif-corona.gif'); im.seek(0); im.save('static/img/gif-corona.png')"
```

```html
<!-- Consumer in any template using Tier 2 loading -->
<picture>
  <source srcset="/static/img/gif-corona.png" media="(prefers-reduced-motion: reduce)">
  <img src="/static/img/gif-corona.gif" alt="Cargando" class="w-40 h-40 object-contain">
</picture>
```

---

## State of the Art

| Old approach | Current approach | When changed | Impact |
|--------------|------------------|--------------|--------|
| Hand-rolled focus-trap | Alpine Focus plugin `x-trap.inert` (or native `inert` attribute) | 2022 (`inert` baseline support) [CITED: MDN] | Phase 8 adopts the plugin; `inert` is a fallback. |
| `localStorage.getItem/setItem` + `x-init` | Alpine Persist plugin `$persist()` | Alpine 3.8+ (plugin released 2022) | Phase 8 uses the plugin; no hand-rolled storage code. |
| axe-core on client | pa11y-ci on CI (for this project) | D-21 decision; axe-core deferred to Mark-IV | Plan 08-02 adds pa11y-ci job. |
| Jumpy GIF during `prefers-reduced-motion` | `<picture>` with first-frame PNG fallback | UI-SPEC §Tier 2 loading | Plan 08-01 extracts PNG once. |

**Deprecated / outdated in THIS codebase:**

- `text-gray-800` / `text-gray-500` / `bg-white` everywhere in templates → replaced by `text-body` / `text-muted` / `bg-surface-base` per per-screen plan.
- Full-fill saturated toasts (`bg-red-600 text-white`) at `base.html:189-195` → replaced by white surface + left accent bar per D-30.
- Inline Tailwind config in `base.html:13-24` → extracted to `static/js/tailwind.config.js` per D-02 / Plan 08-01.
- Always-visible sidebar (`w-56`/`w-16`) at `base.html:28-104` → replaced by top-bar + drawer per D-07 / D-10.
- `w-56 rail permanent` layout → removed.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Node.js (for pa11y-ci in CI) | Plan 08-02 CI step | ✓ (GitHub Actions `ubuntu-24.04` has Node 20 pre-installed) | 20.x | — |
| `npm`/`npx` | pa11y-ci install | ✓ | matches Node | — |
| Chrome / Chromium | pa11y-ci headless runner | ✓ (Actions runner) | — | pa11y launches with `--no-sandbox` |
| ffmpeg | Extract `gif-corona.png` first frame | Unknown on dev machine — check `command -v ffmpeg` | — | Pillow-based Python one-liner (Pillow already in `requirements.txt`) |
| Alpine Focus plugin CDN | Drawer focus trap | ✓ (cdn.jsdelivr.net) | 3.14.8 | Hand-roll focus trap (D-05 / Don't-Hand-Roll — avoid) |
| Alpine Persist plugin CDN | Drawer state persistence | ✓ (cdn.jsdelivr.net) | 3.14.8 | Hand-roll localStorage in `x-init` |
| Postgres (for CI test suite) | Phase 5 test suite GREEN requirement (D-20) | ✓ (existing `services:` in CI) | 16-alpine | — |

**Missing dependencies with no fallback:** none.

**Missing dependencies with fallback:** `ffmpeg` on dev machines — Pillow fallback is acceptable and avoids a new system dependency.

---

## Validation Architecture

> `workflow.nyquist_validation` is not explicitly `false` in `.planning/config.json`; included by default.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest + fastapi TestClient + Jinja-rendered HTML assertions |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` (inferred — current setup under `tests/` with `conftest.py`) |
| Quick run command | `pytest tests/routers/test_sidebar_filtering.py tests/routers/test_button_gating.py tests/middleware/test_flash.py -x` |
| Full suite command | `pytest tests/ -q --cov=api --cov=nexo --cov-fail-under=60` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| UIREDO-01 | `tokens.css` exists; all semantic tokens declared; Tailwind config resolves `bg-primary` to primary colour | integration (DOM) | `pytest tests/infra/test_ui_tokens.py -x` | ❌ Wave 0 |
| UIREDO-02 | Drawer toggles on hamburger click + Esc + `[`; persists across reloads | integration (requires browser) | Covered by pa11y-ci URL list + smoke `curl`; JS behaviour = manual + visual | ⚠️ pa11y-ci is contract test |
| UIREDO-02 (a11y) | Focus trap engages when drawer open; aria-hidden on siblings | accessibility | `pa11y-ci --config .pa11yci.json` (CI smoke) | ❌ Wave 0 |
| UIREDO-03 | `/` renders `luk4.html` with tokens; `_partials/mapa_pabellon` interaction preserved | integration | `pytest tests/routers/test_centro_mando_render.py -x` | ❌ Wave 0 |
| UIREDO-04 | Each screen has `sketch-findings-{screen}` skill + `PLAN.md` + implementation commit | process, not test | N/A — repo structure check | — |
| UIREDO-05 | All `transition-duration-*` values ≤ `--duration-slow` (300ms) | lint on CSS | grep-based unit test or manual review | ❌ Wave 0 |
| UIREDO-06 | `can()` filter still hides RBAC-gated nav items and buttons | integration | `pytest tests/routers/test_sidebar_filtering.py tests/routers/test_button_gating.py -x` | ✅ EXISTS |
| UIREDO-06 (flash) | Flash toast contract: cookie `nexo_flash` ↔ `request.state.flash` ↔ `{{ flash_message }}` ↔ `window.showToast` | integration | `pytest tests/middleware/test_flash.py -x` | ✅ EXISTS |
| UIREDO-07 | Scope creep → new items in `.planning/backlog.md` or rejected | process | Manual review of Plan 08-NN summaries | — |
| UIREDO-08 (contrast) | All text/surface pairs pass WCAG 2.1 AA | accessibility | `pa11y-ci --config .pa11yci.json` | ❌ Wave 0 |
| UIREDO-08 (keyboard) | Tab order, Esc closes overlay in z-order, `[` toggles drawer | manual + integration | Smoke checklist in SUMMARY | — |
| UIREDO-08 (`aria-label`) | Icon-only buttons carry `aria-label` in Spanish | lint on HTML | `pytest tests/infra/test_icon_only_buttons_labelled.py -x` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `pytest tests/routers/test_sidebar_filtering.py tests/routers/test_button_gating.py tests/middleware/test_flash.py -x` (≤ 30s)
- **Per wave merge (= per per-screen plan close):** `pytest tests/ -q --cov=api --cov=nexo --cov-fail-under=60`
- **Phase gate:** full suite green + `pa11y-ci` green on CI smoke job + manual keyboard-navigation checklist from `UI-SPEC §Keyboard navigation` before `/gsd-verify-work`.

### Wave 0 Gaps

- [ ] `tests/infra/test_ui_tokens.py` — asserts `static/css/tokens.css` declares all semantic tokens from UI-SPEC §Semantic tokens (grep-based check against the file).
- [ ] `tests/routers/test_bienvenida_render.py` — asserts `/bienvenida` returns 200, contains saludo string, contains clock element, and is gated behind auth.
- [ ] `tests/routers/test_centro_mando_render.py` — asserts `/` still renders `luk4.html` with `_partials/mapa_pabellon.html` embedded 4 times.
- [ ] `tests/infra/test_icon_only_buttons_labelled.py` — grep-based audit asserting every `<button>` with no text children (just an `<svg>`) carries an `aria-label` attribute, across all 24 templates.
- [ ] `.pa11yci.json` — new config file (template above).
- [ ] Smoke-user seed SQL (argon2id hash generated locally, committed to workflow as env-injected variable, not plain text).
- [ ] `tests/middleware/test_flash.py` — extend with an assertion that `showToast` is called with `(type, title, msg)` signature (3 args) when the flash cookie is present, using a rendered HTML snapshot.

*(Existing tests that **must stay green** during Phase 8: `tests/auth/test_rbac_smoke.py`, `tests/auth/test_can_helper.py`, `tests/auth/test_audit_append_only.py`, `tests/routers/test_sidebar_filtering.py`, `tests/routers/test_button_gating.py`, `tests/routers/test_html_get_guarded.py`, `tests/routers/test_forbidden_redirect.py`, `tests/routers/test_ajustes_split.py`, `tests/middleware/test_flash.py`, `tests/middleware/test_query_timing.py`.)*

---

## Security Domain

> `security_enforcement` not explicitly `false` in config; included by default.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | indirect | Unchanged — Argon2id + session cookie from Phase 2; `/bienvenida` just consumes `request.state.user`. |
| V3 Session Management | indirect | Unchanged — `FlashMiddleware` cookie remains HttpOnly + Secure + SameSite=Lax. |
| V4 Access Control | yes | Drawer nav items + button gating continue using `{% if can() %}` Jinja (server-side, D-02 zero-trust). No `x-show` for RBAC. |
| V5 Input Validation | yes | FastAPI server-side validation authoritative (D-25); Jinja paints inline errors; Pydantic models unchanged. |
| V6 Cryptography | no | No new cryptographic operations; pa11y-ci smoke user password is CI-only and does not ship to prod. |
| V14 Configuration | yes | `chromeLaunchConfig.args: ["--no-sandbox"]` is scoped to CI only; never used in prod. |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| XSS via Jinja `|safe` filter | Tampering | Default Jinja auto-escape is ON; audit any `|safe` usage in new templates (none currently planned). |
| RBAC bypass via client-side toggle | Elevation of Privilege | Keep RBAC in `{% if can() %}` server-side (D-02). `x-show` forbidden for RBAC gating. |
| Focus-trap lockout of assistive tech | Denial of Service (a11y) | Esc must always close overlays; focus trap returns focus to trigger. Covered by pa11y-ci + manual keyboard checklist. |
| pa11y-ci smoke user credential leak | Information Disclosure | Password hash never committed in plaintext; CI env var + `ON CONFLICT DO NOTHING` seed; test user cleaned up by CI teardown. |
| `localStorage` injection | Tampering | Only `nexo.ui.drawerOpen` stored (boolean); no sensitive data persisted. Any future `$persist` key must be reviewed. |
| Clickjacking on login | Tampering | Unchanged — existing `X-Frame-Options` or CSP from Phase 6 deployment. No frames introduced in this phase. |

---

## Open Questions (RESOLVED — recommendations passed to planner)

### 1. Keyboard shortcut for drawer toggle

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| `[` (bare) | Fast, one key, mnemonic ("panel on left"). | Conflicts with ordinary text typing — must guard against active inputs. | **Recommended — with input-guard.** Matches UI-SPEC. |
| `Alt+S` | Zero collision risk; Alt prefix clear. | Alt+S opens "Save" dialogs in some apps; not a browser conflict. | Safe fallback. |
| `Ctrl+\` | No collision in Chrome/Edge/Firefox (per search). | Some IDE-style LAN apps use it. | Acceptable third. |
| `Ctrl+[` or `Cmd+[` | **REJECT** — browser "Back" navigation in Chrome/Safari [CITED: search results] | Would navigate away. | **Do not use.** |

**Recommendation to planner:** proceed with `[` + input-guard (UI-SPEC default). If operator rejects during Plan 08-02 review, fall back to `Alt+S`.

### 2. Sub-nav treatment for `/ajustes` in the drawer

Two live options:

- **Single entry + hub grid** (UI-SPEC default): `Ajustes` → navigates to hub page with 6-card grid. One drawer entry per section.
- **Expandable drawer group**: `Ajustes` expands in the drawer showing 6 sub-items.

Research lens: Notion/Linear/Figma patterns (LAN-inspirational UIs per CONTEXT.md §specifics):
- Notion uses expandable groups in sidebar — works when users visit sub-pages often.
- Linear uses flat top-level entries with hub pages — works when sub-pages are infrequent.

In Nexo, only the propietario sees `/ajustes`. For a propietario visiting 2-3 times a day, a hub page click is fine. For repeated switching between `ajustes/conexion` and `ajustes/limites` during debugging, expandable would be faster.

**Recommendation to planner:** start with UI-SPEC default (single entry + hub grid). If Plan 08-19 review reveals user friction, add expandable group in Plan 08-02 amendment (not in the original chrome plan).

### 3. Loading hero tagline rotation

Deterministic by context (current spec) vs random per load.

- Deterministic: `Procesando análisis` consistently appears for pipeline runs. Users build expectation — reassuring.
- Random: adds playfulness but no information value; in a serious operator tool the noise is unwelcome.

**Recommendation to planner:** keep deterministic. Tagline set already in UI-SPEC §Tier 2; no additions needed.

**Proposed tagline expansion** (5-7 total, if operator wants more variety):

1. `Procesando análisis` (existing, pipeline)
2. `Preparando informe` (existing, informes)
3. `Cargando Nexo` (existing, post-login)
4. `Calculando factor de aprendizaje` (existing, recalibrate)
5. `Sincronizando recursos` (NEW — `/recursos` sync button)
6. `Buscando en histórico` (NEW — `/historial` filter apply)
7. `Generando auditoría CSV` (NEW — `/ajustes/auditoria` export)

### 4. Role pill casing

- **Uppercase via `tracking-wide`** (current): matches `base.html:122` verbatim; industry-standard for status tags.
- **Title Case**: friendlier; reads as a label rather than a status.

The backend stores `propietario` lowercase. Uppercase pill is a pure display transform. Neither affects logic.

**Recommendation to planner:** keep UI-SPEC default (uppercase). Matches current baseline; no migration cost.

### 5. `bienvenida.html` saludo name source

Inspected `nexo/data/models_nexo.py:82-89`:

```python
class NexoUser(NexoBase):
    id = Column(Integer, primary_key=True)
    email = Column(String(200), nullable=False, unique=True, index=True)
    password_hash = Column(String(200), nullable=False)
    role = Column(String(20), nullable=False)
    active = Column(Boolean, nullable=False, default=True)
    must_change_password = Column(Boolean, nullable=False, default=True)
    last_login = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
```

**There is no `nombre` column.** UI-SPEC's preferred `current_user.nombre` source is a phantom.

Options:

- **Option A — Add `nombre` column now.** Requires Alembic migration + backfill script. Backfill from email local-part. 30-min of work, but adds schema change to a pure UI phase.
- **Option B — Use email local-part only.** `e.eguskiza@ecsmobility.com` → `Erik` via `.split('@')[0].split('.')[0].capitalize()`. Zero schema change. **Recommended.**
- **Option C — No name at all.** "Buenos días" with no trailing comma, matches UI-SPEC fallback.

**Recommendation to planner:** **Option B.** Document in Plan 08-03 that `nombre` column is Mark-IV scope (user profile editable).

**FINAL RESOLUTION (operator decision):** Option A adopted — `nexo.users.nombre` column migrated in Plan 08-03. Option B (email local-part only, defer to Mark-IV) rejected per UI-SPEC + session discuss. Plan 08-02 uses `getattr(...)` fallback so the intermediate state between 08-02 and 08-03 deploys cleanly.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `@alpinejs/focus@3.14.8` and `@alpinejs/persist@3.14.8` are published to `cdn.jsdelivr.net` | Standard Stack / Pattern 2 | Drawer implementation blocked; fall back to `alpinejs-focus@latest` which may mismatch core. Plan 08-02 first task: verify via browser-hit on CDN URL. |
| A2 | `pa11y-ci@3` supports the `actions` array in URL objects (login flow) | §5 pa11y-ci setup | CI pa11y-ci integration fails. [CITED: pa11y-ci README confirms `actions` at URL level.] — risk LOW |
| A3 | GitHub Actions `ubuntu-24.04` image has Node 20 + Chromium available without setup | Environment Availability | Would need explicit `actions/setup-node@v4`. Risk LOW (standard image includes both). |
| A4 | The `ffmpeg` Pillow fallback correctly extracts frame 0 of the crown GIF | Example 5 / Environment Availability | The PNG would show a wrong frame. Risk LOW — Pillow's `Image.seek(0)` is well-documented. |
| A5 | `x-trap.inert.noscroll` works when attached to a `position: fixed` drawer element that is `x-show`'d | Pattern 2 | Focus trap misfires while drawer is closed. [CITED: alpinejs.dev/plugins/focus — `x-trap` evaluates the expression reactively.] — risk LOW |
| A6 | `current_user` is the authenticated `NexoUser` ORM object with `.email`, `.role`, `.departments` eagerly loaded | Example 3 | AttributeError at render. [VERIFIED: `api/deps.py:55` + auth middleware eager-loads departments.] — not an assumption |
| A7 | The Jinja `hora_saludo` filter resolves against server `Europe/Madrid` tzinfo correctly | Example 3 | Saludo shows wrong band at DST boundaries. Use `zoneinfo.ZoneInfo` (stdlib Python 3.11+). Risk LOW. |
| A8 | `--no-sandbox` flag for pa11y-ci Chromium is acceptable security-wise **in CI only** | Security §V14 | If copy-pasted to prod, is a security hole. Scoped to `.pa11yci.json` which is dev-only. |
| A9 | The seed user email `smoke@nexo.local` will not collide with any production data | Example 4 | Potential test pollution. Mitigated by `ON CONFLICT DO NOTHING` and CI-only Postgres (not the LAN prod DB). |
| A10 | `base.html` flash_message render (line 242) is the only caller of the 3-arg `showToast(type, title, msg)` signature | Pitfall 3 | Other callers would break on refactor. Cross-checked against `luk4.html:320-334` — those ALSO use 3-arg. List is complete. |
| A11 | The `display_name` heuristic `email.split('@')[0].split('.')[0].capitalize()` produces acceptable first names for all current users | Open Question 5 | Some users have emails like `admin@…` where the result is "Admin" (acceptable). Edge case: `juan.perez@…` → "Juan" (correct). `info@…` → "Info" (awkward but rare). Risk LOW. |

**If this table is empty:** n/a — contains assumptions the planner should surface before locking Plan 08-NN tasks.

---

## Open Questions (RESOLVED — research-level, actioned in Phase 8 plans)

1. **RUNBOOK anchor slugs — do we rewrite UI-SPEC or RUNBOOK?** (Pitfall 8). Recommendation: rewrite UI-SPEC because RUNBOOK has been cited in Phase 7 test `tests/infra/test_devex_docs.py`.
   - What we know: UI-SPEC uses short anchors; RUNBOOK uses long descriptive headings.
   - What's unclear: whether the short anchors are allowed to exist as hidden HTML anchors.
   - Recommendation: update UI-SPEC §Error state copy as part of Plan 08-02 task 0.

2. **Should the `centro_mando:read` permission gate `/bienvenida`?** Every authenticated user can see Centro de Mando, so the gate effectively just requires auth. Alternative: no permission gate, only `AuthMiddleware`.
   - Recommendation: no permission gate. `require_permission` is overhead for a landing everyone sees.

3. **Does Plan 08-02 also seed the CI smoke user, or does Plan 08-03 (auth-first screen plan)?** pa11y-ci can't run without a seed user.
   - Recommendation: Plan 08-02 (because pa11y-ci is introduced there) seeds the user as a CI workflow step, not as a migration — user is CI-only.

4. **`flash_message` rendering order — inside `<body>` or inside a shared `toast-host` partial?** Currently it's a `<script>` block inside `base.html:235-246`. Refactor may move to a `_partials/flash.html`.
   - Recommendation: move to `_partials/flash.html` included by `base.html`, so per-screen plans can skip including base's toast scaffolding.

5. **Is `x-cloak` CSS rule preserved?** Current `static/css/app.css:3` has `[x-cloak] { display: none !important; }`. The chrome rewrite must preserve this line or Alpine's initial hide-until-ready breaks.
   - Recommendation: preserve as-is in Plan 08-01 `app.css` rewrite.

---

## Sources

### Primary (HIGH confidence)

- `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-CONTEXT.md` — 31 locked decisions.
- `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` — 1133-line design contract (APPROVED).
- `.planning/REQUIREMENTS.md` §UIREDO-01..UIREDO-08.
- `.planning/ROADMAP.md` §Phase 8 (goal + 8 success criteria).
- `CLAUDE.md` — Nexo conventions (no emojis, Spanish user-facing, stack lock).
- Codebase files read in full: `templates/base.html`, `templates/login.html`, `templates/_partials/mapa_pabellon.html` (first 120 of 331), `static/css/app.css`, `static/js/app.js`, `nexo/middleware/flash.py`, `nexo/services/auth.py:195-260`, `nexo/data/models_nexo.py:78-89`, `api/deps.py`, `api/routers/auth.py`, `api/routers/pages.py`, `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, `pyproject.toml` (first 60 lines), `docs/BRANDING.md`, `docs/RUNBOOK.md` (headings).
- Tailwind v3 docs — [Customizing colors](https://v3.tailwindcss.com/docs/customizing-colors) — `rgb(var(--token) / <alpha-value>)` pattern, space-separated RGB.
- Alpine.js docs — [Focus plugin](https://alpinejs.dev/plugins/focus) — `x-trap.inert.noscroll`.
- Alpine.js docs — [Persist plugin](https://alpinejs.dev/plugins/persist) — `$persist(value).as('key')`.
- Alpine.js docs — [Alpine.data() lifecycle](https://alpinejs.dev/globals/alpine-data) — `destroy()` hook.
- pa11y-ci README — [github.com/pa11y/pa11y-ci](https://github.com/pa11y/pa11y-ci) — `actions` config, `chromeLaunchConfig`.
- MDN — [inert attribute](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Global_attributes/inert) — browser support near-perfect.

### Secondary (MEDIUM confidence)

- WebSearch — bracket key browser shortcut collision analysis (`Cmd/Ctrl+[` = back navigation).
- WebSearch — pa11y-ci GitHub Actions setup patterns (`--no-sandbox` is standard).
- WebSearch — Alpine.js memory leak prevention pattern.

### Tertiary (LOW confidence)

- None — all critical claims cross-referenced with at least one primary source.

---

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — Alpine plugins + Tailwind CDN confirmed; pa11y-ci config validated against upstream README.
- Architecture: HIGH — codebase inspected end-to-end, tiers mapped.
- Pitfalls: HIGH — each pitfall is evidenced by a grep or a CITED source; Pitfall 3 (toast signature) is a confirmed latent bug in the codebase.
- Validation: MEDIUM — test matrix depends on files that don't exist yet (Wave 0 gaps listed); existing Phase 5 tests verified to still be the right guardrails.
- Security: HIGH — no new cryptographic primitives; `--no-sandbox` scoped to CI.

**Research date:** 2026-04-22
**Valid until:** 2026-05-22 (30 days for stable web platform; re-check pa11y-ci major version on execution date).
