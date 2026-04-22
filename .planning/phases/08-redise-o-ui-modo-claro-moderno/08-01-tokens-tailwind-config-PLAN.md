---
phase: 08-redise-o-ui-modo-claro-moderno
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - static/css/tokens.css
  - static/css/app.css
  - static/js/tailwind.config.js
  - docs/BRANDING.md
  - tests/infra/test_tokens_css.py
autonomous: true
gap_closure: false
requirements: [UIREDO-01]
tags: [ui, tokens, tailwind, css-variables, nexo, phase-8]
user_setup: []

must_haves:
  truths:
    - "The two-layer token file `static/css/tokens.css` exists and declares every token in UI-SPEC.md §Semantic tokens verbatim (surfaces, text, border, primary, focus-ring, state, shadow, radius, motion, z-index)."
    - "Raw values in `tokens.css` are space-separated RGB triples (no commas, no `rgb()` wrapper) so Tailwind's `rgb(var(--token) / <alpha-value>)` pattern works end-to-end."
    - "Tailwind config lives in `static/js/tailwind.config.js` (extracted from `base.html`) and maps every semantic utility listed in UI-SPEC §Tailwind mapping (`bg-surface`, `text-body`, `bg-primary`, `shadow-card`, `rounded-lg`, `z-topbar`, …) back to the CSS variables."
    - "`docs/BRANDING.md` has a new §Tokens section documenting the raw palette, semantic aliases and the 60/30/10 split exactly as UI-SPEC §Color specifies."
    - "No template is rewritten in this plan — the redesign of chrome happens in 08-02. Only tokens + config + docs + a regression test ship here."
  artifacts:
    - path: "static/css/tokens.css"
      provides: "Two-layer CSS custom properties — raw palette (brand, surface, success, warn, error, info) + semantic aliases (surface, text, border, primary, shadow, radius, motion, z-index)"
      contains: "--color-surface-base, --color-text-body, --color-primary, --shadow-card, --radius-lg, --duration-base, --z-topbar, @media (prefers-reduced-motion: reduce)"
    - path: "static/js/tailwind.config.js"
      provides: "Extracted Tailwind config consumed via `<script src=...>` in base.html; maps CSS vars to utilities via `rgb(var(--token) / <alpha-value>)`"
      contains: "window.tailwind = window.tailwind || {}; tailwind.config = {...};"
    - path: "docs/BRANDING.md"
      provides: "Canonical doc for tokens + palette + 60/30/10 split"
      contains: "## Tokens (Phase 8)"
    - path: "tests/infra/test_tokens_css.py"
      provides: "Regression test — parses tokens.css, asserts every required token exists with correct format, and asserts Tailwind config references them"
      contains: "def test_tokens_file_declares_all_semantic_colors"
  key_links:
    - from: "static/css/tokens.css"
      to: "static/js/tailwind.config.js"
      via: "CSS variable lookup at Tailwind JIT time"
      pattern: "rgb\\(var\\(--color-[a-z-]+\\) / <alpha-value>\\)"
    - from: "docs/BRANDING.md"
      to: "static/css/tokens.css"
      via: "documentation of the same tokens declared in CSS"
      pattern: "--color-(surface|text|primary|success|warn|error|info)"
---

<objective>
Ship the foundational token layer for the Phase 8 UI redesign.
Extract the Tailwind config out of `base.html` (currently inlined on lines
13-24) into a dedicated `static/js/tailwind.config.js`. Create the
two-layer token file `static/css/tokens.css` that UI-SPEC.md §Semantic
tokens mandates. Document the tokens in `docs/BRANDING.md`. Land a
static regression test so drift is caught immediately.

This plan ships **no** template rewrites. It only creates the
substrate that Plan 08-02 (chrome) and every subsequent per-screen plan
depend on.

Purpose: without tokens, no other screen can claim to honor
UIREDO-01. This plan unblocks the wave.

Output: 4 source files + 1 test + 1 doc section. Zero behaviour
change — the app renders identically until 08-02 wires the new chrome.
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
@.planning/phases/08-redise-o-ui-modo-claro-moderno/08-RESEARCH.md
@docs/BRANDING.md
@templates/base.html
@static/css/app.css
@CLAUDE.md

<interfaces>
<!-- Existing Tailwind config inline in base.html:13-24 that this plan extracts. -->

From templates/base.html (lines 13-24, to be removed by this plan after extraction):
```html
<script>
  tailwind.config = {
    theme: {
      extend: {
        colors: {
          brand:   { 50:'#eef5ff', 100:'#d9e8ff', 200:'#bcd5ff', 300:'#8ebbff', 400:'#5995ff', 500:'#336dff', 600:'#1a3a5c', 700:'#142d47', 800:'#0f2236', 900:'#0a1724' },
          surface: { 50:'#f8fafc', 100:'#f1f5f9', 200:'#e2e8f0', 300:'#cbd5e1' },
        }
      }
    }
  }
</script>
```

IMPORTANT: **This plan only MOVES the config to a new file.** Wiring
`base.html` to load the new `tailwind.config.js` file is Plan 08-02's
job — don't edit `base.html` here. The app keeps working because this
plan doesn't touch `base.html` — the inline config stays until 08-02
replaces it.
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create static/css/tokens.css + update static/css/app.css @import</name>
  <read_first>
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Semantic tokens (layer 2, tokens.css)" (lines ~218-290) and §"Color — Raw palette (layer 1)" (lines ~192-210) — the full CSS block is the source of truth.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Spacing Scale — Semantic aliases" (lines ~73-85) — spacing tokens.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Typography" — the CSS var names (`--text-body`, `--text-subtitle`, `--text-heading`, `--text-display`) that tokens.css must declare.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Reduced motion" — the `@media (prefers-reduced-motion: reduce)` rule at the bottom of tokens.css.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-RESEARCH.md` §Pitfall 1 + §Pitfall 2 — format `R G B` (spaces, no commas, no wrapper).
    - `static/css/app.css` — current file (103 lines). Tokens.css must load BEFORE app.css; app.css stays unchanged in this task other than adding `@import url('./tokens.css');` at the top.
  </read_first>
  <action>
Create `static/css/tokens.css` with the following structure. Every value is a
literal from UI-SPEC.md — do NOT paraphrase.

```css
/* ──────────────────────────────────────────────────────────────
   Nexo — Design Tokens (Phase 8, per D-01 + D-05)
   Two layers:
     Layer 1 (raw)       — palette scales used occasionally for edge cases.
     Layer 2 (semantic)  — the only layer templates should read from.
   Format: space-separated R G B for Tailwind's
           rgb(var(--color-xxx) / <alpha-value>) pattern.
   Do NOT wrap values in rgb() and do NOT use commas.
   ────────────────────────────────────────────────────────────── */

:root {
  /* ── Layer 1: Raw palette ─────────────────────────────────────── */
  /* Brand (kept from base.html:18) */
  --color-brand-50:   238 245 255;
  --color-brand-100:  217 232 255;
  --color-brand-200:  188 213 255;
  --color-brand-300:  142 187 255;
  --color-brand-400:   89 149 255;
  --color-brand-500:   51 109 255;
  --color-brand-600:   26  58  92;   /* primary accent */
  --color-brand-700:   20  45  71;   /* primary hover  */
  --color-brand-800:   15  34  54;   /* text body      */
  --color-brand-900:   10  23  36;   /* text heading   */

  /* Surface (50-300 kept; 0 + 400-900 new) */
  --color-surface-0:   255 255 255;
  --color-surface-50:  248 250 252;
  --color-surface-100: 241 245 249;
  --color-surface-200: 226 232 240;
  --color-surface-300: 203 213 225;
  --color-surface-400: 148 163 184;
  --color-surface-500: 100 116 139;
  --color-surface-600:  71  85 105;
  --color-surface-700:  51  65  85;
  --color-surface-800:  30  41  59;
  --color-surface-900:  15  23  42;

  /* Success */
  --color-success-50:  236 253 245;
  --color-success-100: 209 250 229;
  --color-success-500:  16 185 129;
  --color-success-600:   5 150 105;
  --color-success-700:   4 120  87;

  /* Warn */
  --color-warn-50:  255 251 235;
  --color-warn-100: 254 243 199;
  --color-warn-500: 245 158  11;
  --color-warn-600: 217 119   6;
  --color-warn-700: 180  83   9;

  /* Error */
  --color-error-50:  254 242 242;
  --color-error-100: 254 226 226;
  --color-error-500: 239  68  68;
  --color-error-600: 220  38  38;
  --color-error-700: 185  28  28;

  /* Info */
  --color-info-50:  239 246 255;
  --color-info-100: 219 234 254;
  --color-info-500:  59 130 246;
  --color-info-600:  37  99 235;
  --color-info-700:  29  78 216;

  /* ── Layer 2: Semantic tokens ─────────────────────────────────── */
  /* Base surfaces */
  --color-surface-base:    var(--color-surface-0);
  --color-surface-app:     var(--color-surface-50);
  --color-surface-subtle:  var(--color-surface-100);
  --color-surface-muted:   var(--color-surface-200);

  /* Text */
  --color-text-body:       var(--color-brand-800);
  --color-text-heading:    var(--color-brand-900);
  --color-text-muted:      var(--color-surface-500);
  --color-text-disabled:   var(--color-surface-400);
  --color-text-on-accent:  255 255 255;

  /* Border */
  --color-border-subtle:   var(--color-surface-200);
  --color-border-strong:   var(--color-surface-300);

  /* Accent (brand) */
  --color-primary:         var(--color-brand-600);
  --color-primary-hover:   var(--color-brand-700);
  --color-primary-active:  var(--color-brand-800);
  --color-primary-subtle:  var(--color-brand-50);
  --color-focus-ring:      var(--color-brand-500);

  /* Semantic states */
  --color-success:         var(--color-success-600);
  --color-success-subtle:  var(--color-success-50);
  --color-warn:            var(--color-warn-600);
  --color-warn-subtle:     var(--color-warn-50);
  --color-error:           var(--color-error-600);
  --color-error-subtle:    var(--color-error-50);
  --color-info:            var(--color-info-600);
  --color-info-subtle:     var(--color-info-50);

  /* Typography scale (D-05) */
  --text-body-size: 14px;
  --text-body-lh:   1.5;
  --text-body-weight-regular: 400;
  --text-body-weight-semibold: 600;
  --text-subtitle-size: 16px;
  --text-subtitle-lh:   1.4;
  --text-heading-size: 20px;
  --text-heading-lh:   1.3;
  --text-heading-ls:   -0.01em;
  --text-display-size: 32px;
  --text-display-lh:   1.2;
  --text-display-ls:   -0.02em;

  /* Font stacks (D-06) */
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               Helvetica, Arial, sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas,
               monospace;

  /* Spacing aliases (D-05) */
  --space-field-gap: 16px;
  --space-card-padding: 24px;
  --space-card-gap: 16px;
  --space-section-gap: 32px;
  --space-page-gutter-x: 24px;
  --space-page-gutter-x-mobile: 16px;
  --space-topbar-h: 56px;
  --space-topbar-h-mobile: 52px;
  --space-drawer-w: 280px;
  --space-drawer-padding: 16px;

  /* Elevation (shadow) */
  --shadow-card:     0 1px 2px 0 rgb(15 34 54 / 0.04),
                     0 1px 3px 0 rgb(15 34 54 / 0.06);
  --shadow-popover:  0 4px 6px -1px rgb(15 34 54 / 0.08),
                     0 2px 4px -2px rgb(15 34 54 / 0.06);
  --shadow-modal:    0 20px 25px -5px rgb(15 34 54 / 0.12),
                     0 8px 10px -6px rgb(15 34 54 / 0.08);
  --shadow-drawer:   4px 0 24px -4px rgb(15 34 54 / 0.12);

  /* Radius */
  --radius-sm:   6px;
  --radius-md:  10px;
  --radius-lg:  16px;
  --radius-xl:  24px;
  --radius-pill: 9999px;

  /* Motion (D-05 — budget ≤200ms drawer, ≤300ms modal) */
  --duration-fast: 150ms;
  --duration-base: 200ms;
  --duration-slow: 300ms;
  --ease-standard:    cubic-bezier(0.2, 0, 0, 1);
  --ease-emphasized:  cubic-bezier(0.3, 0, 0, 1);
  --ease-accelerate:  cubic-bezier(0.3, 0, 1, 1);

  /* Z-index scale */
  --z-base:      0;
  --z-sticky:   10;
  --z-topbar:   30;
  --z-backdrop: 40;
  --z-drawer:   50;
  --z-modal:    60;
  --z-popover:  70;
  --z-toast:    90;
  --z-reserved: 100;
}

/* Reduced motion — applies globally (D-05 + UI-SPEC §Reduced motion) */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
  /* The spinner and the loading hero GIF are exempt — they signal progress. */
  .spinner,
  .loading-hero img,
  .spinner-panel .spinner {
    animation-duration: 1.5s !important;
  }
}
```

Then update `static/css/app.css` to `@import url('./tokens.css');` as the FIRST line (above the existing `[x-cloak]` rule). Do not touch any other rule in `app.css` in this task — they get rewritten in Plan 08-02.
  </action>
  <acceptance_criteria>
    - `test -f /home/eeguskiza/analisis_datos/static/css/tokens.css` returns 0.
    - `grep -c "^  --color-surface-base:" static/css/tokens.css` returns exactly 1.
    - `grep -c "^  --z-topbar:" static/css/tokens.css` returns exactly 1.
    - `grep -c "^  --duration-base: 200ms;" static/css/tokens.css` returns exactly 1.
    - `grep -cE "rgb\\(" static/css/tokens.css` returns 0 inside `:root {}` block (the `rgb()` usages are only inside `--shadow-*` values and the reduced-motion block is outside `:root`). Sanity check: `awk '/^:root/,/^}/' static/css/tokens.css | grep -c "rgb("` returns a small positive number ONLY from the `--shadow-*` rgba-syntax tokens — NEVER from the `--color-*` tokens. Validate explicitly: `awk '/^:root/,/^}/' static/css/tokens.css | grep -E "^  --color-" | grep -c "rgb("` returns 0.
    - `head -1 static/css/app.css` is exactly `@import url('./tokens.css');`.
    - `grep -c "@media (prefers-reduced-motion: reduce)" static/css/tokens.css` returns 1 or more.
  </acceptance_criteria>
  <verify>
    <automated>test -f static/css/tokens.css &amp;&amp; head -1 static/css/app.css | grep -F "@import url('./tokens.css');" &amp;&amp; awk '/^:root/,/^}/' static/css/tokens.css | grep -E "^  --color-" | awk '{ if ($0 ~ /rgb\(/) { print "FAIL: rgb() wrapper inside --color-*: " $0; exit 1 } } END { print "OK" }'</automated>
  </verify>
  <done>tokens.css exists with all UI-SPEC §Semantic tokens, raw triple format, and the reduced-motion block. app.css imports it as first line. No template touched.</done>
</task>

<task type="auto">
  <name>Task 2: Extract Tailwind config to static/js/tailwind.config.js</name>
  <read_first>
    - `templates/base.html` lines 13-24 — current inline config (DO NOT EDIT base.html in this task — extraction only creates the new file).
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Tailwind mapping (Plan 08-01 target)" (table around line ~291).
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-RESEARCH.md` §Pitfall 1 (JIT ordering) and §Pattern 1 (two-layer tokens).
    - `static/css/tokens.css` (created in Task 1) — every CSS var name referenced here.
  </read_first>
  <action>
Create `static/js/tailwind.config.js` with the following exact content. It declares the Tailwind CDN config consumed via `tailwind.config = {...}` on the global `window.tailwind`. Every color / utility maps back to a CSS variable defined in `tokens.css`.

```js
/* ──────────────────────────────────────────────────────────────
   Nexo — Tailwind CDN config (Phase 8 / Plan 08-01, per D-02)
   Extracted from base.html. Loaded via
     <script src="/static/js/tailwind.config.js"></script>
   and MUST appear BEFORE the cdn.tailwindcss.com script tag
   (Pitfall 1 — JIT ordering). Values reference CSS vars declared
   in /static/css/tokens.css using Tailwind's
     rgb(var(--color-xxx) / <alpha-value>)
   pattern, which preserves `bg-primary/20` alpha utilities.
   ────────────────────────────────────────────────────────────── */

window.tailwind = window.tailwind || {};
window.tailwind.config = {
  theme: {
    extend: {
      colors: {
        /* Semantic surfaces (Layer 2) */
        surface: {
          DEFAULT: 'rgb(var(--color-surface-base) / <alpha-value>)',
          base:    'rgb(var(--color-surface-base) / <alpha-value>)',
          app:     'rgb(var(--color-surface-app) / <alpha-value>)',
          subtle:  'rgb(var(--color-surface-subtle) / <alpha-value>)',
          muted:   'rgb(var(--color-surface-muted) / <alpha-value>)',
          /* Raw scale preserved for legacy uses (mapa_pabellon, etc.) */
          0:   'rgb(var(--color-surface-0) / <alpha-value>)',
          50:  'rgb(var(--color-surface-50) / <alpha-value>)',
          100: 'rgb(var(--color-surface-100) / <alpha-value>)',
          200: 'rgb(var(--color-surface-200) / <alpha-value>)',
          300: 'rgb(var(--color-surface-300) / <alpha-value>)',
          400: 'rgb(var(--color-surface-400) / <alpha-value>)',
          500: 'rgb(var(--color-surface-500) / <alpha-value>)',
          600: 'rgb(var(--color-surface-600) / <alpha-value>)',
          700: 'rgb(var(--color-surface-700) / <alpha-value>)',
          800: 'rgb(var(--color-surface-800) / <alpha-value>)',
          900: 'rgb(var(--color-surface-900) / <alpha-value>)',
        },
        /* Brand scale preserved for legacy uses; new code should use `primary`. */
        brand: {
          50:  'rgb(var(--color-brand-50) / <alpha-value>)',
          100: 'rgb(var(--color-brand-100) / <alpha-value>)',
          200: 'rgb(var(--color-brand-200) / <alpha-value>)',
          300: 'rgb(var(--color-brand-300) / <alpha-value>)',
          400: 'rgb(var(--color-brand-400) / <alpha-value>)',
          500: 'rgb(var(--color-brand-500) / <alpha-value>)',
          600: 'rgb(var(--color-brand-600) / <alpha-value>)',
          700: 'rgb(var(--color-brand-700) / <alpha-value>)',
          800: 'rgb(var(--color-brand-800) / <alpha-value>)',
          900: 'rgb(var(--color-brand-900) / <alpha-value>)',
        },
        /* Primary (accent) */
        primary: {
          DEFAULT: 'rgb(var(--color-primary) / <alpha-value>)',
          hover:   'rgb(var(--color-primary-hover) / <alpha-value>)',
          active:  'rgb(var(--color-primary-active) / <alpha-value>)',
          subtle:  'rgb(var(--color-primary-subtle) / <alpha-value>)',
        },
        /* Semantic states */
        success: {
          DEFAULT: 'rgb(var(--color-success) / <alpha-value>)',
          subtle:  'rgb(var(--color-success-subtle) / <alpha-value>)',
        },
        warn: {
          DEFAULT: 'rgb(var(--color-warn) / <alpha-value>)',
          subtle:  'rgb(var(--color-warn-subtle) / <alpha-value>)',
        },
        error: {
          DEFAULT: 'rgb(var(--color-error) / <alpha-value>)',
          subtle:  'rgb(var(--color-error-subtle) / <alpha-value>)',
        },
        info: {
          DEFAULT: 'rgb(var(--color-info) / <alpha-value>)',
          subtle:  'rgb(var(--color-info-subtle) / <alpha-value>)',
        },
      },
      textColor: {
        body:      'rgb(var(--color-text-body) / <alpha-value>)',
        heading:   'rgb(var(--color-text-heading) / <alpha-value>)',
        muted:     'rgb(var(--color-text-muted) / <alpha-value>)',
        disabled:  'rgb(var(--color-text-disabled) / <alpha-value>)',
        'on-accent': 'rgb(var(--color-text-on-accent) / <alpha-value>)',
      },
      borderColor: {
        subtle: 'rgb(var(--color-border-subtle) / <alpha-value>)',
        strong: 'rgb(var(--color-border-strong) / <alpha-value>)',
      },
      ringColor: {
        primary: 'rgb(var(--color-focus-ring) / <alpha-value>)',
      },
      boxShadow: {
        card:    'var(--shadow-card)',
        popover: 'var(--shadow-popover)',
        modal:   'var(--shadow-modal)',
        drawer:  'var(--shadow-drawer)',
      },
      borderRadius: {
        sm:   'var(--radius-sm)',
        md:   'var(--radius-md)',
        lg:   'var(--radius-lg)',
        xl:   'var(--radius-xl)',
        pill: 'var(--radius-pill)',
      },
      transitionDuration: {
        fast: 'var(--duration-fast)',
        base: 'var(--duration-base)',
        slow: 'var(--duration-slow)',
      },
      transitionTimingFunction: {
        standard:    'var(--ease-standard)',
        emphasized:  'var(--ease-emphasized)',
        accelerate:  'var(--ease-accelerate)',
      },
      zIndex: {
        base:     'var(--z-base)',
        sticky:   'var(--z-sticky)',
        topbar:   'var(--z-topbar)',
        backdrop: 'var(--z-backdrop)',
        drawer:   'var(--z-drawer)',
        modal:    'var(--z-modal)',
        popover:  'var(--z-popover)',
        toast:    'var(--z-toast)',
      },
      fontFamily: {
        sans: 'var(--font-sans)',
        mono: 'var(--font-mono)',
      },
    },
  },
};
```

Do NOT edit `base.html` in this plan — Plan 08-02 does the wiring. This
file just needs to exist and be correct.
  </action>
  <acceptance_criteria>
    - `test -f /home/eeguskiza/analisis_datos/static/js/tailwind.config.js` returns 0.
    - `grep -c "window.tailwind.config = {" static/js/tailwind.config.js` returns 1.
    - `grep -c "rgb(var(--color-primary) / <alpha-value>)" static/js/tailwind.config.js` returns 1 or more (the DEFAULT primary binding).
    - `grep -c "rgb(var(--color-surface-app) / <alpha-value>)" static/js/tailwind.config.js` returns 1.
    - `grep -c "var(--shadow-card)" static/js/tailwind.config.js` returns 1.
    - `grep -c "'var(--z-topbar)'" static/js/tailwind.config.js` returns 1.
    - `node -e "const fs=require('fs'); const src=fs.readFileSync('static/js/tailwind.config.js','utf8'); eval(src); if(!window.tailwind.config.theme.extend.colors.primary.DEFAULT) process.exit(1);"` returns 0 (syntactic validity + structure).
  </acceptance_criteria>
  <verify>
    <automated>test -f static/js/tailwind.config.js &amp;&amp; node -e "global.window={}; const fs=require('fs'); const src=fs.readFileSync('static/js/tailwind.config.js','utf8'); eval(src); const c=window.tailwind.config.theme.extend; if (!c.colors.primary.DEFAULT || !c.boxShadow.card || !c.zIndex.topbar) { console.log('FAIL: missing tokens'); process.exit(1); } console.log('OK');"</automated>
  </verify>
  <done>tailwind.config.js exists, is syntactically valid JS, declares all semantic utilities mapped to CSS vars. base.html NOT touched (that's 08-02).</done>
</task>

<task type="auto">
  <name>Task 3: Document tokens in docs/BRANDING.md + regression test in tests/infra/test_tokens_css.py</name>
  <read_first>
    - `docs/BRANDING.md` — existing sections; append (do not rewrite).
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Color — 60 / 30 / 10 split" and §"Accessibility — Contrast (WCAG 2.1 AA)" table (contrast numbers).
    - `tests/infra/test_deploy_lan_doc.py` — existing pattern for doc regression tests (this is the style to emulate: parse file, assert content present).
  </read_first>
  <action>
### Part A — Append to `docs/BRANDING.md`

Append a new section at the end of `docs/BRANDING.md`. Do NOT rewrite the file;
use append-only so the existing Phase 1 branding decisions stay intact.

```markdown

## Tokens (Phase 8)

Phase 8 introduce el sistema de tokens CSS en `static/css/tokens.css`. Dos capas:

1. **Raw** — escalas (`--color-brand-*`, `--color-surface-*`, `--color-success-*`,
   etc.). Se conservan para casos residuales (editor del pabellón, mapa).
2. **Semántica** — alias consumidos por templates y `app.css` (`--color-primary`,
   `--color-surface-base`, `--color-text-body`, `--shadow-card`, `--z-topbar`).
   Los templates SOLO deben leer de esta capa.

Formato: todos los tokens de color se declaran como triples RGB separados por
espacios (`R G B`), sin comas ni wrapper `rgb()`. Tailwind los consume vía
`rgb(var(--color-xxx) / <alpha-value>)`, lo que preserva las utilidades alpha
(`bg-primary/20`).

### Paleta — 60 / 30 / 10 (UIREDO-01)

| Rol | Token | Hex | % uso |
|-----|-------|-----|-------|
| Dominante (60%) | `--color-surface-app` (`surface.50`) | `#f8fafc` | body, main content |
| Secundario (30%) | `--color-surface-base` (`surface.0`) | `#ffffff` | cards, drawer, top bar, modals, form fields |
| Acento (10%) | `--color-primary` (`brand.600`) | `#1a3a5c` | CTA primario, link, focus ring, nav activo |
| Destructivo | `--color-error` (`error.600`) | `#dc2626` | botones destructivos, estados de error |

El acento queda reservado a: (1) botón primario, (2) nav activo en drawer,
(3) focus ring, (4) links en texto, (5) selected state (checkbox/radio/tab).

### Tipografía

4 tamaños (Body 14, Subtitle 16, Heading 20, Display 32) y 2 pesos
(400 regular, 600 semibold). Fuentes system-stack (D-06) sin CDN.

### Motion

`--duration-fast: 150ms`, `--duration-base: 200ms` (drawer cap UIREDO-02),
`--duration-slow: 300ms` (modal cap UIREDO-05). Respeta
`prefers-reduced-motion` (regla global en `tokens.css`).

### Z-index

`--z-topbar: 30`, `--z-backdrop: 40`, `--z-drawer: 50`, `--z-modal: 60`,
`--z-popover: 70`, `--z-toast: 90`. Escala cerrada — no usar valores fuera.

### Tailwind mapping

Consumido vía `static/js/tailwind.config.js` (extraído de `base.html` en
Plan 08-01). Utilidades nuevas: `bg-surface`, `bg-surface-app`, `text-body`,
`text-muted`, `bg-primary`, `bg-primary-subtle`, `border-subtle`,
`shadow-card`, `z-topbar`. Utilidades legacy (`bg-brand-600`, `bg-surface-50`)
siguen disponibles para compatibilidad.

### Contraste WCAG 2.1 AA

Matriz verificada manualmente (`pa11y-ci` la validará en CI en Plan 08-10):
text-body sobre surface-app 14.2:1 (AAA), text-muted sobre surface-app 5.1:1
(AA), on-accent sobre primary 10.3:1, success sobre surface-base 4.7:1 (AA
ajustado — no aclarar), warn sobre surface-base 4.5:1 (AA ajustado — no
aclarar), error sobre surface-base 4.9:1.
```

### Part B — Create `tests/infra/test_tokens_css.py`

```python
"""Regression tests for Phase 8 tokens.css + tailwind.config.js.

Static parse — no runtime deps. Asserts:
- tokens.css declares every required semantic CSS var.
- Color tokens use space-separated RGB format (no commas, no rgb() wrapper).
- tailwind.config.js references the declared vars via the
  rgb(var(--token) / <alpha-value>) pattern.
- docs/BRANDING.md has the Phase 8 Tokens section.

Pitfall 1/2 from 08-RESEARCH.md: token drift or format regressions
break the entire Tailwind pipeline silently. Catch at CI boundary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_TOKENS_CSS = _ROOT / "static" / "css" / "tokens.css"
_TAILWIND_JS = _ROOT / "static" / "js" / "tailwind.config.js"
_BRANDING_MD = _ROOT / "docs" / "BRANDING.md"
_APP_CSS = _ROOT / "static" / "css" / "app.css"


REQUIRED_SEMANTIC_COLOR_TOKENS = [
    "--color-surface-base",
    "--color-surface-app",
    "--color-surface-subtle",
    "--color-surface-muted",
    "--color-text-body",
    "--color-text-heading",
    "--color-text-muted",
    "--color-text-disabled",
    "--color-text-on-accent",
    "--color-border-subtle",
    "--color-border-strong",
    "--color-primary",
    "--color-primary-hover",
    "--color-primary-active",
    "--color-primary-subtle",
    "--color-focus-ring",
    "--color-success",
    "--color-success-subtle",
    "--color-warn",
    "--color-warn-subtle",
    "--color-error",
    "--color-error-subtle",
    "--color-info",
    "--color-info-subtle",
]

REQUIRED_RADIUS_TOKENS = [
    "--radius-sm",
    "--radius-md",
    "--radius-lg",
    "--radius-xl",
    "--radius-pill",
]

REQUIRED_SHADOW_TOKENS = [
    "--shadow-card",
    "--shadow-popover",
    "--shadow-modal",
    "--shadow-drawer",
]

REQUIRED_MOTION_TOKENS = [
    "--duration-fast",
    "--duration-base",
    "--duration-slow",
    "--ease-standard",
    "--ease-emphasized",
    "--ease-accelerate",
]

REQUIRED_Z_TOKENS = [
    "--z-topbar",
    "--z-backdrop",
    "--z-drawer",
    "--z-modal",
    "--z-popover",
    "--z-toast",
]


@pytest.fixture(scope="module")
def tokens_css_text() -> str:
    assert _TOKENS_CSS.exists(), f"Missing {_TOKENS_CSS}"
    return _TOKENS_CSS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tailwind_js_text() -> str:
    assert _TAILWIND_JS.exists(), f"Missing {_TAILWIND_JS}"
    return _TAILWIND_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def branding_md_text() -> str:
    assert _BRANDING_MD.exists(), f"Missing {_BRANDING_MD}"
    return _BRANDING_MD.read_text(encoding="utf-8")


@pytest.mark.parametrize("token", REQUIRED_SEMANTIC_COLOR_TOKENS)
def test_tokens_file_declares_semantic_color(token: str, tokens_css_text: str):
    assert f"{token}:" in tokens_css_text, (
        f"tokens.css must declare {token}"
    )


@pytest.mark.parametrize("token", REQUIRED_RADIUS_TOKENS + REQUIRED_SHADOW_TOKENS
                         + REQUIRED_MOTION_TOKENS + REQUIRED_Z_TOKENS)
def test_tokens_file_declares_layout_tokens(token: str, tokens_css_text: str):
    assert f"{token}:" in tokens_css_text, (
        f"tokens.css must declare {token}"
    )


def test_color_tokens_do_not_wrap_rgb(tokens_css_text: str):
    """--color-* tokens must be raw triples — Pitfall 2."""
    in_root = False
    for line in tokens_css_text.splitlines():
        if line.strip().startswith(":root"):
            in_root = True
            continue
        if in_root and line.strip() == "}":
            break
        if in_root and line.lstrip().startswith("--color-"):
            # --color-foo: value; OR --color-foo: var(--color-bar);
            _, _, raw = line.partition(":")
            value = raw.strip().rstrip(";").strip()
            assert not value.startswith("rgb("), (
                f"Color token wraps rgb(): {line.strip()}"
            )
            assert "," not in value or value.startswith("var("), (
                f"Color token uses commas (expect space-separated triples): {line.strip()}"
            )


def test_reduced_motion_block_present(tokens_css_text: str):
    assert "@media (prefers-reduced-motion: reduce)" in tokens_css_text


def test_app_css_imports_tokens(app_css_text: str = None):
    text = _APP_CSS.read_text(encoding="utf-8")
    first_line = text.splitlines()[0]
    assert first_line.strip() == "@import url('./tokens.css');", (
        f"app.css first line must import tokens.css, got: {first_line!r}"
    )


@pytest.mark.parametrize("token", REQUIRED_SEMANTIC_COLOR_TOKENS[:6])
def test_tailwind_config_references_tokens(token: str, tailwind_js_text: str):
    pattern = f"var({token})"
    assert pattern in tailwind_js_text, (
        f"tailwind.config.js must reference {token}"
    )


def test_tailwind_uses_alpha_value_pattern(tailwind_js_text: str):
    # At least one use of the Tailwind alpha pattern (sanity).
    assert "/ <alpha-value>" in tailwind_js_text


def test_branding_md_has_tokens_section(branding_md_text: str):
    assert "## Tokens (Phase 8)" in branding_md_text
    assert "60 / 30 / 10" in branding_md_text
    assert "`--color-primary`" in branding_md_text
```

Both files must satisfy the repository's `ruff format` and `ruff check` rules
(the test file lives under `tests/`, which is excluded from the mypy scope
per `pyproject.toml`, but ruff still applies).
  </action>
  <acceptance_criteria>
    - `grep -F "## Tokens (Phase 8)" docs/BRANDING.md` returns one line.
    - `grep -F "60 / 30 / 10" docs/BRANDING.md` returns one line.
    - `test -f tests/infra/test_tokens_css.py` returns 0.
    - `pytest tests/infra/test_tokens_css.py -x -q` returns exit code 0 and reports more than 30 tests passing.
    - `ruff check tests/infra/test_tokens_css.py` returns exit code 0.
    - `ruff format --check tests/infra/test_tokens_css.py` returns exit code 0.
  </acceptance_criteria>
  <verify>
    <automated>ruff check tests/infra/test_tokens_css.py &amp;&amp; ruff format --check tests/infra/test_tokens_css.py &amp;&amp; pytest tests/infra/test_tokens_css.py -x -q</automated>
  </verify>
  <done>BRANDING.md has the Tokens (Phase 8) section. Regression test suite passes and would fail if any required token disappears or breaks format.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Static asset pipeline | `tokens.css` served by Caddy + FastAPI `StaticFiles`; cache-controlled. No user input crosses. |
| Dev tooling | `tailwind.config.js` evaluated client-side by Tailwind CDN. Anyone with a browser can read it, but it contains no secrets — only design tokens. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-08-01-01 | Tampering | `static/css/tokens.css` (if an attacker serves a modified version) | accept | Purely cosmetic tokens; no auth/secrets. Static asset integrity is the existing Caddy/nginx concern, not this plan's scope. |
| T-08-01-02 | Information Disclosure | `docs/BRANDING.md` | accept | Already public-by-design (branding doc). No PII, no credentials. |
| T-08-01-03 | Denial of Service | Large `tokens.css` blocking first paint | mitigate | File is <10 KB minified; `@import` in `app.css` keeps it in the critical path alongside existing styles (no network overhead — same-origin). |
</threat_model>

<verification>
Whole phase sanity (this plan):

1. `pytest tests/ -x -q` (full suite) exits 0 — Phase 5 + Phase 7 regression.
2. `ruff check api/ nexo/` and `ruff format --check api/ nexo/` pass.
3. Manual: open the app locally (`make dev`). The app looks UNCHANGED — the Tailwind config is still inline in base.html; the new tokens.css is loaded via app.css @import but no template consumes the new tokens yet. If anything visually drifts, revert and investigate.

(Nothing else changes in behaviour — that is the point of a foundation plan.)
</verification>

<success_criteria>
- Wave 1 substrate is green: tokens + Tailwind config + doc + test all shipped.
- Plan 08-02 can depend on `static/css/tokens.css` and
  `static/js/tailwind.config.js` without touching either.
- No visible regression in the running app (screenshots, if the operator
  checks, look identical to pre-plan state).
- `pytest tests/infra/test_tokens_css.py -x -q` runs locally and in CI
  green.
</success_criteria>

<output>
After completion, create `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-01-SUMMARY.md` with:

- Token inventory (listed by section: raw / semantic / motion / z-index).
- Files added: `static/css/tokens.css`, `static/js/tailwind.config.js`,
  `tests/infra/test_tokens_css.py`.
- Files modified: `static/css/app.css` (added `@import`), `docs/BRANDING.md`
  (appended `## Tokens (Phase 8)`).
- Test count added.
- Deviations (should be none).
- Handoff note for Plan 08-02: base.html can now load tokens.css + the
  new Tailwind config.
</output>