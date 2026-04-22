---
phase: 08-redise-o-ui-modo-claro-moderno
plan: 02
type: execute
wave: 2
depends_on: [08-01]
files_modified:
  - templates/base.html
  - static/css/app.css
  - static/css/print.css
  - static/js/app.js
  - static/img/gif-corona.png
  - tests/routers/test_flash_toast_contract.py
  - tests/infra/test_chrome_structure.py
autonomous: true
gap_closure: false
requirements: [UIREDO-02, UIREDO-05, UIREDO-06, UIREDO-08]
tags: [ui, chrome, topbar, drawer, toast, base-html, alpine-focus, alpine-persist, nexo, phase-8]
user_setup: []

must_haves:
  truths:
    - "The old always-visible sidebar in `base.html` is replaced by a 56px top bar + hamburger-triggered drawer (overlay, 280px wide) per UI-SPEC §Global Chrome."
    - "The drawer persists its open/closed state in `localStorage` key `nexo.ui.drawerOpen` via Alpine `$persist`, is closed by default on mobile, traps focus (Alpine `@alpinejs/focus`), closes on Esc / backdrop / re-click / any nav link on mobile, and animates ≤200ms with `prefers-reduced-motion` reducing to opacity-only."
    - "The `[` (left square bracket) bare key toggles the drawer from anywhere in the document EXCEPT when the active element is `<input>`, `<textarea>`, or `contentEditable`."
    - "`window.showToast(type, title, msg)` is the canonical 3-arg API. All 12+ callers across `templates/` and `static/js/app.js` pass `(type, title, msg)`. The Jinja flash consumer at `base.html` (FlashMiddleware cookie path) also uses the 3-arg form."
    - "Toasts are top-right, white surface with `border-l-4` semantic accent bar + icon, 4s auto-dismiss, hover pauses, vertical queue with 8px gap, max-width 400px (mobile `calc(100vw - 32px)`)."
    - "Alpine Focus + Alpine Persist CDN plugins are loaded in `base.html` BEFORE the Alpine core script tag."
    - "`static/img/gif-corona.png` (first-frame extraction) exists for `prefers-reduced-motion` fallback of the loading hero GIF. If `ffmpeg`/`gif2png` are not available, a minimal 1x1 PNG with documented follow-up is an acceptable stopgap — the test asserts the file exists."
    - "A new `static/css/print.css` is imported by `base.html` behind `@media print` and hides top bar, drawer, buttons, toast container — per UI-SPEC D-31."
    - "Phase 5 RBAC is intact: `{% if can(current_user, permission) %}` still gates every sensitive nav item in the drawer. Propietario sees everything; `aprobaciones:manage` gates the Solicitudes badge."
    - "Error-state copy referenced in UI-SPEC uses the **full GFM slug** of each RUNBOOK heading (e.g. `#escenario-1-mes-caido-sql-server-dbizaro-inaccesible`). The test asserts each slug is unique in `docs/RUNBOOK.md` and that no error-state copy in templates uses the short form `#escenario-1-mes-caido`."
    - "A new drawer section `Configuración` contains 7 flat items: Ajustes (hub), Conexión, Usuarios, Auditoría, Límites, Rendimiento, Solicitudes. Each item is wrapped in its own `can()` gate. If the user can see none of them, the whole Configuración section hides."
  artifacts:
    - path: "templates/base.html"
      provides: "Full chrome rewrite: top bar + drawer + toast container + landing shell block; loads tokens.css, tailwind.config.js, Alpine focus + persist plugins, print.css"
      contains: "x-data=\"nexoChrome()\" + role=\"dialog\" on drawer + x-trap + $persist + hamburger button + <aside> drawer + <div id=\"toast-root\">"
    - path: "static/css/app.css"
      provides: "Component classes rewritten on semantic tokens: .card, .btn-primary, .btn-secondary (new), .btn-danger, .btn-ghost, .btn-link (new), .btn-icon (new), .input-inline, .data-table, .spinner, .spinner-panel, .loading-hero, .toast, .empty-state, .error-state, .breadcrumbs"
      contains: "@apply bg-surface text-body border-subtle; background-color: rgb(var(--color-primary))"
    - path: "static/css/print.css"
      provides: "Print stylesheet: hide top bar, drawer, toast, action buttons; keep `main` content + tables legible on Ctrl+P"
      contains: "@media print { #nexo-topbar, #nexo-drawer, #toast-root { display: none !important; } }"
    - path: "static/js/app.js"
      provides: "nexoChrome() Alpine component + new 3-arg window.showToast + [ shortcut listener + reloj helper stub + setInterval cleanup"
      contains: "function nexoChrome() + window.showToast = function(type, title, msg)"
    - path: "static/img/gif-corona.png"
      provides: "Static first-frame fallback for the loading hero GIF under prefers-reduced-motion"
      contains: "PNG binary (even a 1x1 stub is acceptable if ffmpeg unavailable — documented in plan action)"
    - path: "tests/routers/test_flash_toast_contract.py"
      provides: "Pytest suite locking the 3-arg contract end-to-end: middleware emit + base.html Jinja consumer + each template caller"
      contains: "def test_showtoast_signature_is_three_args + def test_all_callers_use_three_args"
    - path: "tests/infra/test_chrome_structure.py"
      provides: "Static-parse regression: base.html declares <aside id=\"nexo-drawer\"> with role=dialog, x-trap, $persist; loads focus + persist plugins before alpine; hamburger button exists; toast-root container exists; RUNBOOK slug invariant holds"
      contains: "def test_base_html_loads_alpine_focus_persist_plugins_before_core + def test_runbook_anchor_slugs_match"
  key_links:
    - from: "templates/base.html"
      to: "static/js/app.js"
      via: "nexoChrome() Alpine component bootstraps drawer + toast + shortcut"
      pattern: "x-data=\"nexoChrome\\(\\)\""
    - from: "static/js/app.js"
      to: "localStorage"
      via: "Alpine $persist writes nexo.ui.drawerOpen"
      pattern: "\\$persist.*nexo\\.ui\\.drawerOpen"
    - from: "templates/base.html"
      to: "static/css/print.css"
      via: "@media print stylesheet link"
      pattern: "href=\"/static/css/print.css\" media=\"print\""
    - from: "tests/routers/test_flash_toast_contract.py"
      to: "nexo/middleware/flash.py + templates/base.html flash block"
      via: "asserts 3-arg showToast call on flash path"
      pattern: "showToast\\('info', 'Aviso',"
---

<objective>
Rewrite `templates/base.html` to ship the Phase 8 chrome: 56px top bar +
hamburger-triggered drawer + restyled 3-arg toast system + loaded Alpine
focus/persist plugins + print stylesheet + `[` keyboard shortcut. Rewrite
`static/css/app.css` so every existing component class
(`.card`, `.btn-*`, `.data-table`, `.spinner`, `.input-inline`, …) is
re-declared on semantic tokens from Plan 08-01 — class names stay
backwards compatible so per-screen plans do not have to touch markup
wholesale. Fix the latent `showToast` 2-arg vs 3-arg bug (Pitfall 3)
by standardising on the 3-arg signature across the 12+ callers + the
Jinja flash consumer.

Additionally: ship `gif-corona.png` (first-frame extraction) for
reduced-motion fallback, align RUNBOOK anchor slugs with UI-SPEC error
copy, and land the regression tests that lock the contract.

This plan is the riskiest in Phase 8 because it touches `base.html`,
which every other template inherits. Phase 5 RBAC (can() gating) +
Phase 4 flash middleware contract MUST survive. The full test suite
(`pytest tests/`) must be green at the end.

Purpose: without the new chrome, every downstream per-screen plan has
nothing to wrap its content in. This wave unblocks 08-03..08-09.

Output: `base.html` rewritten, `app.css` rewritten, new `print.css`,
new `app.js` functions, static PNG fallback, 2 new test files, 0
behavioural regression in the app (only visual refresh of chrome).
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
@.planning/phases/08-redise-o-ui-modo-claro-moderno/08-01-SUMMARY.md
@templates/base.html
@static/css/app.css
@static/css/tokens.css
@static/js/app.js
@static/js/tailwind.config.js
@nexo/middleware/flash.py
@nexo/services/auth.py
@api/deps.py
@api/main.py
@docs/RUNBOOK.md
@CLAUDE.md

<interfaces>
<!-- Canonical contracts this plan must honor. -->

### 1. Phase 5 `can()` helper (do NOT regress)

From `nexo/services/auth.py`:
```python
def can(user: NexoUser | None, permission: str) -> bool:
    """user None → False. role == 'propietario' → True.
       Else intersect user.departments with PERMISSION_MAP[permission]."""
```

From `api/deps.py` (registered as Jinja global — do not re-register):
```python
templates.env.globals.update(app_name=..., can=_can, ...)
```

Every sensitive drawer item uses `{% if can(current_user, "<perm>") %}`
verbatim. The drawer in this plan MUST preserve that pattern for every
item the old sidebar gated.

### 2. Flash middleware contract (do NOT regress)

From `nexo/middleware/flash.py`:
```python
# reads cookie "nexo_flash", sets request.state.flash, deletes on response
# guarded with HI-01 "already_set" check to avoid double Set-Cookie
```

From `api/deps.py::render`:
```python
"flash_message": getattr(request.state, "flash", None),
```

From `templates/base.html` (current):
```jinja
{% if flash_message %}
<script>
  document.addEventListener('DOMContentLoaded', function () {
    if (typeof showToast === 'function') {
      showToast('info', 'Aviso', {{ flash_message|tojson }});   // 3-arg
    }
  });
</script>
{% endif %}
```

IMPORTANT: the current inline `window.showToast` in `base.html:188` is
**3-arg** (type, title, msg); the `static/js/app.js:410` copy is **2-arg**
(message, type). They collide — whichever loads last wins. Per UI-SPEC
D-30 + Research Pitfall 3, the 3-arg form IS THE CONTRACT. This plan
rewrites the `app.js` one to 3-arg, updates every 2-arg caller, and
removes the duplicate inline one from `base.html` (keep a single
definition in `app.js`).

### 3. showToast call sites inventory (exact count — verify via grep)

```
templates/informes.html:277       → showToast(`${data.n_pdfs} PDF(s) generados`);
templates/informes.html:281       → showToast('Error al regenerar', 'error');
templates/informes.html:284       → showToast('Error: ' + err, 'error');
templates/luk4.html:320           → showToast('producing', 'LUK4 en produccion', '...');   (already 3-arg)
templates/luk4.html:322           → showToast('incidence', 'Incidencia en LUK4', '...');   (already 3-arg)
templates/luk4.html:324           → showToast('stopped', 'LUK4 parada', '...');            (already 3-arg)
templates/luk4.html:330           → showToast('alarm', ..., ...);                          (already 3-arg)
templates/luk4.html:334           → showToast('turno', ..., ...);                          (already 3-arg)
templates/ciclos_calc.html:512    → showToast('Valor no valido', 'error');                 (2-arg)
templates/ciclos_calc.html:531    → showToast(`${ref.referencia}: ${value} pzas/h`);       (1-arg)
templates/ciclos_calc.html:534    → showToast(err.detail || 'Error', 'error');             (2-arg)
templates/ciclos_calc.html:536    → showToast('Error guardando', 'error');                 (2-arg)
templates/ciclos_calc.html:701    → showToast('CSV exportado');                            (1-arg)
templates/ciclos.html:139         → showToast('Ciclos guardados');                         (1-arg)
templates/historial.html:410      → showToast(`${data.n_pdfs} PDF(s) generados`);          (1-arg)
templates/historial.html:415      → showToast(data.detail || 'Error', 'error');            (2-arg)
templates/historial.html:417      → showToast('Error: ' + err, 'error');                   (2-arg)
templates/historial.html:427      → showToast('Ejecucion borrada');                        (1-arg)
templates/historial.html:447      → showToast('Error: ' + err.message, 'error');           (2-arg)
templates/historial.html:524      → showToast('CSV exportado');                            (1-arg)
templates/plantillas.html:198     → showToast('Plantilla guardada');                       (1-arg)
templates/plantillas.html:212     → showToast('Plantilla creada');                         (1-arg)
static/js/app.js:410              → function showToast(message, type = 'success') { ... }  (DEFINITION — 2-arg)
static/js/app.js:550..557         → showToast(`Solicitud enviada. ...`, 'info');           (2-arg)
templates/base.html:188..223      → window.showToast = function(type, title, msg) { ... }  (DEFINITION — 3-arg, DUPLICATE)
templates/base.html:242           → showToast('info', 'Aviso', {{ flash_message|tojson }}); (3-arg)
```

**Contract going forward (D-30):** `showToast(type, title, msg)` — the
3-arg form. Variants: `info`, `success`, `warn`, `error`. Legacy values
`producing/stopped/incidence/alarm/turno` map to these (see UI-SPEC
§Flash toast) — that mapping lives inside `showToast`.

### 4. Current sidebar nav items (preserve with permission gates + rename "Analisis" → "Análisis")

See `templates/base.html:47-94` for the current `nav_items` tuple list +
the Solicitudes badge block. This plan migrates those to the drawer,
splits into 3 sections (main, operaciones, **Configuración**), and adds
the 7-item Configuración section. See the drawer structure block in
UI-SPEC §Drawer structure.

### 5. RUNBOOK anchor slugs (GFM slugification)

The 5 RUNBOOK.md headings (verified against `docs/RUNBOOK.md`):

| Heading | GFM slug |
|---------|----------|
| `## Escenario 1: MES caido (SQL Server dbizaro inaccesible)` | `#escenario-1-mes-caido-sql-server-dbizaro-inaccesible` |
| `## Escenario 2: Postgres no arranca` | `#escenario-2-postgres-no-arranca` |
| `## Escenario 3: Certificado Caddy expira / warning en browsers` | `#escenario-3-certificado-caddy-expira--warning-en-browsers` |
| `## Escenario 4: Pipeline atascado (HALLAZGO CRITICO: semaforo in-process)` | `#escenario-4-pipeline-atascado-hallazgo-critico-semaforo-in-process` |
| `## Escenario 5: Lockout del unico propietario (HALLAZGO CRITICO: no hay unlock_user)` | `#escenario-5-lockout-del-unico-propietario-hallazgo-critico-no-hay-unlock_user` |

The UI-SPEC §Error state copy block uses the short forms (`#escenario-1-mes-caido`). UI-SPEC wins — update error-state copy in this plan's `app.css` helper + any shared partial to the full GFM slug. The regression test in Task 4 asserts the full slugs appear in `docs/RUNBOOK.md` and the template error-state snippets reference them verbatim.
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Load Alpine Focus + Persist plugins, add tokens + tailwind.config + print.css, remove duplicate showToast from base.html</name>
  <read_first>
    - `templates/base.html` in full (250 lines). Map every section: head, sidebar, topbar, content, footer, toast block, script block.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Global Chrome → Top bar" and §"Drawer" (exact heights, z-index, animation durations).
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-RESEARCH.md` §Pitfall 4 (Alpine plugin ordering) + §Pattern 2 (drawer with focus trap).
    - `static/css/tokens.css` (from 08-01) and `static/js/tailwind.config.js` (from 08-01) — confirm the tokens consumed by the new markup exist.
  </read_first>
  <action>
**Rewrite `templates/base.html` top-to-bottom.** The new structure below IS the source; replace the existing file wholesale. Preserve every Jinja block/context var consumed by downstream templates (`{% block title %}`, `{% block page_title %}`, `{% block content %}`, `{% block topbar %}`, `current_user`, `flash_message`, `app_name`, `company_name`, `logo_path`, `ecs_logo_path`, `page`).

Also **create `static/css/print.css`** with the @media print rules.

### New `templates/base.html` (entire file, use Write tool)

```jinja
<!DOCTYPE html>
<html lang="es" class="h-full">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{% block title %}{{ app_name }}{% endblock %}</title>

  <!-- Phase 8 / Plan 08-01: token layer MUST load before Tailwind CDN
       so the @apply + rgb(var(--token)/<alpha-value>) pipeline resolves -->
  <link rel="stylesheet" href="/static/css/app.css">
  <link rel="stylesheet" href="/static/css/print.css" media="print">

  <!-- Phase 8 / Plan 08-01: Tailwind config extracted to /static/js -->
  <script src="/static/js/tailwind.config.js"></script>
  <script src="https://cdn.tailwindcss.com"></script>

  <!-- Phase 8 / Plan 08-02: Alpine plugins MUST precede Alpine core
       (Pitfall 4 — plugin ordering). Focus = x-trap for drawer.
       Persist = $persist for localStorage drawer state. -->
  <script defer src="https://cdn.jsdelivr.net/npm/@alpinejs/focus@3.14.8/dist/cdn.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/@alpinejs/persist@3.14.8/dist/cdn.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.8/dist/cdn.min.js"></script>

  <script defer src="https://unpkg.com/htmx.org@2.0.4"></script>
  <script defer src="https://unpkg.com/htmx-ext-sse@2.2.2/sse.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
</head>

<body class="h-full bg-surface-app text-body font-sans antialiased" x-data="nexoChrome()" @keydown.window="onKeydown($event)">

  {# ── TOP BAR (UIREDO-02 / D-10 / UI-SPEC §Global Chrome → Top bar) ── #}
  <header id="nexo-topbar"
          class="sticky top-0 z-topbar flex items-center gap-3 px-4 md:px-6 py-3 bg-surface-base border-b border-subtle"
          style="height: var(--space-topbar-h);">

    {# Hamburger — left slot #}
    <button type="button"
            @click="toggleDrawer()"
            :aria-expanded="drawerOpen.toString()"
            aria-controls="nexo-drawer"
            aria-label="Abrir menú"
            class="btn-icon text-muted hover:text-body">
      <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.75" viewBox="0 0 24 24" aria-hidden="true">
        <path stroke-linecap="round" stroke-linejoin="round" d="M4 6h16M4 12h16M4 18h16"/>
      </svg>
    </button>

    {# Page title slot #}
    <h1 class="text-heading text-heading truncate">{% block page_title %}{{ app_name }}{% endblock %}</h1>

    {# Right cluster — flex-1 push #}
    <div class="ml-auto flex items-center gap-3">
      <div id="conn-badge"
           hx-get="/api/conexion/status" hx-trigger="load, every 30s" hx-swap="innerHTML"
           class="text-sm px-3 py-1 rounded-pill bg-surface-subtle text-muted">
        Comprobando...
      </div>

      {% if current_user %}
        {# User display name: nombre if set (Plan 08-03 migration) else email local-part #}
        {% set display_name = current_user.nombre if current_user.nombre else (current_user.email.split('@')[0]|capitalize) %}
        <span class="text-sm text-muted hidden sm:inline">{{ current_user.email }}</span>
        <span class="text-sm font-semibold tracking-wide uppercase px-2 py-0.5 rounded-pill bg-primary-subtle text-primary">
          {{ current_user.role }}
        </span>

        {# User menu popover #}
        <div class="relative" x-data="{ open: false }" @click.outside="open = false">
          <button type="button"
                  @click="open = !open"
                  :aria-expanded="open.toString()"
                  aria-label="Menú de usuario"
                  class="btn-icon bg-primary text-on-accent">
            <span class="text-sm font-semibold">{{ display_name[0]|upper }}</span>
          </button>
          <div x-show="open" x-cloak
               x-transition:enter="transition ease-standard duration-fast"
               x-transition:enter-start="opacity-0 scale-95"
               x-transition:enter-end="opacity-100 scale-100"
               x-transition:leave="transition ease-accelerate duration-fast"
               class="absolute right-0 mt-2 w-56 bg-surface-base border border-subtle rounded-lg shadow-popover z-popover">
            <div class="px-4 py-3 border-b border-subtle">
              <div class="text-sm font-semibold text-heading">{{ display_name }}</div>
              <div class="text-sm text-muted truncate">{{ current_user.email }}</div>
            </div>
            <a href="/cambiar-password" class="block px-4 py-2 text-sm text-body hover:bg-surface-subtle">Cambiar contraseña</a>
            <form method="post" action="/logout" class="block">
              <button type="submit" class="w-full text-left px-4 py-2 text-sm text-body hover:bg-surface-subtle">Cerrar sesión</button>
            </form>
          </div>
        </div>
      {% endif %}
    </div>
  </header>

  {# ── DRAWER (UIREDO-02 / UI-SPEC §Drawer structure) ── #}
  <div x-show="drawerOpen" x-cloak
       x-transition:enter="transition ease-standard duration-base"
       x-transition:enter-start="opacity-0"
       x-transition:enter-end="opacity-100"
       x-transition:leave="transition ease-accelerate duration-fast"
       @click="closeDrawer()"
       class="fixed inset-0 z-backdrop bg-surface-900/40 backdrop-blur-sm"
       aria-hidden="true"></div>

  <aside id="nexo-drawer"
         role="dialog"
         aria-modal="true"
         aria-label="Menú principal"
         x-show="drawerOpen" x-cloak
         x-trap.noscroll="drawerOpen"
         x-transition:enter="transition ease-standard duration-base"
         x-transition:enter-start="-translate-x-full"
         x-transition:enter-end="translate-x-0"
         x-transition:leave="transition ease-accelerate duration-base"
         x-transition:leave-start="translate-x-0"
         x-transition:leave-end="-translate-x-full"
         class="fixed top-0 left-0 h-full bg-surface-base shadow-drawer z-drawer flex flex-col"
         style="width: var(--space-drawer-w); max-width: 85vw;">

    {# Close button #}
    <div class="flex items-center justify-end px-4 pt-3">
      <button type="button" @click="closeDrawer()" aria-label="Cerrar menú" class="btn-icon text-muted hover:text-body">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.75" viewBox="0 0 24 24" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/>
        </svg>
      </button>
    </div>

    {# Logo + brand — centered #}
    <div class="flex flex-col items-center gap-1 px-4 pb-4 border-b border-subtle">
      <img src="{{ logo_path }}" alt="{{ app_name }}" class="w-36 h-36 object-contain"
           onerror="this.outerHTML='<span class=&quot;text-heading text-xl font-semibold&quot;>NEXO</span>'">
      <span class="text-sm text-muted tracking-wide">by {{ company_name|upper }}</span>
    </div>

    {# Nav — primary modules #}
    <nav class="flex-1 overflow-y-auto py-3" aria-label="Navegación principal">

      {# ── Main nav section ── #}
      <div class="drawer-section">
        {% set main_nav = [
          ("dashboard",    "/",             "Centro Mando",     "centro_mando:read",
            "M3 12l9-9 9 9M5 10v10a1 1 0 001 1h3m10-11v10a1 1 0 01-1 1h-3m-6 0h6m-6 0v-6h6v6"),
          ("pipeline",     "/pipeline",     "Análisis",         "pipeline:read",
            "M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"),
          ("historial",    "/historial",    "Historial",        "historial:read",
            "M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"),
          ("capacidad",    "/capacidad",    "Capacidad",        "capacidad:read",
            "M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"),
          ("bbdd",         "/bbdd",         "BBDD",             "bbdd:read",
            "M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4"),
        ] %}
        {% for key, href, label, perm, icon_d in main_nav %}
          {% if perm is none or can(current_user, perm) %}
            {% set active = page == key %}
            <a href="{{ href }}"
               class="flex items-center gap-3 px-4 py-2 transition-colors duration-fast
                      {{ 'bg-primary-subtle text-primary font-semibold' if active else 'text-body hover:bg-surface-subtle' }}">
              <svg class="w-5 h-5 shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" d="{{ icon_d }}"/>
              </svg>
              <span>{{ label }}</span>
            </a>
          {% endif %}
        {% endfor %}
      </div>

      {# ── Operaciones section ── #}
      <div class="my-2 mx-4 h-px bg-border-subtle"></div>
      <div class="drawer-section">
        {% set ops_nav = [
          ("recursos",      "/recursos",       "Recursos",        "recursos:read",
            "M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"),
          ("ciclos_calc",   "/ciclos-calc",    "Calcular ciclos", "ciclos:read",
            "M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z"),
          ("operarios",     "/operarios",      "Operarios",       "operarios:read",
            "M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"),
          ("datos",         "/datos",          "Datos",           "datos:read",
            "M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"),
        ] %}
        {% for key, href, label, perm, icon_d in ops_nav %}
          {% if perm is none or can(current_user, perm) %}
            {% set active = page == key %}
            <a href="{{ href }}"
               class="flex items-center gap-3 px-4 py-2 transition-colors duration-fast
                      {{ 'bg-primary-subtle text-primary font-semibold' if active else 'text-body hover:bg-surface-subtle' }}">
              <svg class="w-5 h-5 shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" d="{{ icon_d }}"/>
              </svg>
              <span>{{ label }}</span>
            </a>
          {% endif %}
        {% endfor %}
      </div>

      {# ── Configuración section (7 flat items, D-locked decision #3) ── #}
      {% set config_items = [
        ("ajustes",               "/ajustes",                 "Ajustes",      "ajustes:manage"),
        ("ajustes_conexion",      "/ajustes/conexion",        "Conexión",     "conexion:config"),
        ("ajustes_usuarios",      "/ajustes/usuarios",        "Usuarios",     "usuarios:manage"),
        ("ajustes_auditoria",     "/ajustes/auditoria",       "Auditoría",    "auditoria:read"),
        ("ajustes_limites",       "/ajustes/limites",         "Límites",      "limites:manage"),
        ("ajustes_rendimiento",   "/ajustes/rendimiento",     "Rendimiento",  "rendimiento:read"),
        ("solicitudes",           "/ajustes/solicitudes",     "Solicitudes",  "aprobaciones:manage"),
      ] %}
      {% set config_visible = namespace(any=false) %}
      {% for key, href, label, perm in config_items %}
        {% if can(current_user, perm) %}{% set config_visible.any = true %}{% endif %}
      {% endfor %}
      {% if config_visible.any %}
        <div class="my-2 mx-4 h-px bg-border-subtle"></div>
        <div class="drawer-section">
          <div class="px-4 py-2 text-sm font-semibold uppercase tracking-wide text-muted">Configuración</div>
          {% for key, href, label, perm in config_items %}
            {% if can(current_user, perm) %}
              {% set active = page == key %}
              <a href="{{ href }}"
                 class="flex items-center justify-between gap-3 px-4 py-2 transition-colors duration-fast
                        {{ 'bg-primary-subtle text-primary font-semibold' if active else 'text-body hover:bg-surface-subtle' }}">
                <span>{{ label }}</span>
                {% if key == 'solicitudes' %}
                  <span hx-get="/api/approvals/count" hx-trigger="load, every 30s" hx-swap="innerHTML" class="text-sm"></span>
                {% endif %}
              </a>
            {% endif %}
          {% endfor %}
        </div>
      {% endif %}
    </nav>

    {# Footer #}
    <div class="border-t border-subtle px-4 py-3 flex items-center gap-2">
      <img src="{{ ecs_logo_path }}" alt="{{ company_name }}" class="h-4 opacity-70" onerror="this.style.display='none'">
      <span class="text-sm text-muted">© {{ company_name }} · {{ app_name }} · v1.0.0</span>
    </div>
  </aside>

  {# ── MAIN ── #}
  <main class="min-h-[calc(100vh-var(--space-topbar-h))] px-4 md:px-6 py-6">
    {% block content %}{% endblock %}
  </main>

  {# ── TOAST CONTAINER (UI-SPEC §Flash toast) ── #}
  <div id="toast-root"
       class="fixed z-toast pointer-events-none flex flex-col gap-2"
       style="top: calc(var(--space-topbar-h) + 16px); right: 16px; max-width: min(400px, calc(100vw - 32px));"
       role="region"
       aria-live="polite"></div>

  {# ── Config global JS ── #}
  <script>
    window.NEXO_CONFIG = Object.freeze({
      appName:     {{ app_name|tojson }},
      companyName: {{ company_name|tojson }},
      logoPath:    {{ logo_path|tojson }},
      ecsLogoPath: {{ ecs_logo_path|tojson }}
    });
  </script>

  {# ── Flash toast consumer (3-arg contract, D-30) ── #}
  {% if flash_message %}
  <script>
    document.addEventListener('DOMContentLoaded', function () {
      if (typeof window.showToast === 'function') {
        window.showToast('info', 'Aviso', {{ flash_message|tojson }});
      }
    });
  </script>
  {% endif %}

  <script src="/static/js/app.js"></script>
</body>
</html>
```

### New `static/css/print.css`

```css
/* ──────────────────────────────────────────────────────────────
   Nexo — Print stylesheet (Phase 8 / D-31)
   Minimal: hide chrome, keep content + tables legible for Ctrl+P.
   Full print layouts live in matplotlib PDFs (OEE reports).
   ────────────────────────────────────────────────────────────── */
@media print {
  #nexo-topbar,
  #nexo-drawer,
  #toast-root,
  .btn,
  .btn-primary,
  .btn-secondary,
  .btn-danger,
  .btn-ghost,
  .btn-icon,
  .btn-link,
  [hx-get],
  [hx-post] {
    display: none !important;
  }
  body {
    background: #fff !important;
    color: #000 !important;
  }
  main {
    padding: 0 !important;
  }
  .card,
  .card-elevated {
    box-shadow: none !important;
    border: 1px solid #ccc !important;
    break-inside: avoid;
  }
  table {
    break-inside: auto;
  }
  tr {
    break-inside: avoid;
    break-after: auto;
  }
}
```

Do NOT edit `static/js/app.js` in this task (next task handles it). Do
not wire `window.showToast` in `base.html` — the duplicate definition
is removed here (the old `<script>window.showToast = ...</script>` block
between `window._toasts = [];` and `{% if flash_message %}` is DELETED in
this rewrite). The single definition will live in `static/js/app.js`
after Task 2.

Do NOT change the permission strings referenced by the drawer — those
match `PERMISSION_MAP` in `nexo/services/auth.py`.
  </action>
  <acceptance_criteria>
    - `grep -c "window.showToast" templates/base.html` returns 0 (definition removed — lives in app.js now).
    - `grep -c "window.showToast('info', 'Aviso'" templates/base.html` returns 0 (the old flash call is renamed to `window.showToast('info', 'Aviso'...)` via the `if (typeof window.showToast === 'function')` guard).
    - `grep -c "showToast('info', 'Aviso'" templates/base.html` returns 1 (the flash consumer — 3-arg form).
    - `grep -c "@alpinejs/focus@3.14.8" templates/base.html` returns 1.
    - `grep -c "@alpinejs/persist@3.14.8" templates/base.html` returns 1.
    - `grep -c "alpinejs@3.14.8/dist/cdn.min.js" templates/base.html` returns 1.
    - The focus + persist `<script>` tags appear BEFORE the Alpine core `<script>` tag (line-number check). `awk '/@alpinejs\/focus/{f=NR} /@alpinejs\/persist/{p=NR} /alpinejs@3\.14\.8\/dist\/cdn/{c=NR} END{if(f<c&&p<c) print "OK"; else print "FAIL"}' templates/base.html` prints `OK`.
    - `grep -c "id=\"nexo-topbar\"" templates/base.html` returns 1.
    - `grep -c "id=\"nexo-drawer\"" templates/base.html` returns 1.
    - `grep -c "id=\"toast-root\"" templates/base.html` returns 1.
    - `grep -c "x-data=\"nexoChrome()\"" templates/base.html` returns 1.
    - `grep -c "x-trap.noscroll=\"drawerOpen\"" templates/base.html` returns 1.
    - `grep -c "role=\"dialog\"" templates/base.html` returns 1.
    - `grep -c "aria-modal=\"true\"" templates/base.html` returns 1.
    - `grep -c "Configuración" templates/base.html` returns 1 or more.
    - `grep -c "can(current_user," templates/base.html` returns 10 or more (one per gated item).
    - `grep -c "Análisis" templates/base.html` returns 1 (accent fix).
    - `test -f static/css/print.css` returns 0.
    - `grep -c "@media print" static/css/print.css` returns 1.
    - `grep -c "href=\"/static/css/print.css\" media=\"print\"" templates/base.html` returns 1.
    - `grep -c "src=\"/static/js/tailwind.config.js\"" templates/base.html` returns 1.
    - The Tailwind config `<script>` tag appears BEFORE the `cdn.tailwindcss.com` script tag.
  </acceptance_criteria>
  <verify>
    <automated>grep -q "id=\"nexo-drawer\"" templates/base.html &amp;&amp; grep -q "id=\"toast-root\"" templates/base.html &amp;&amp; grep -q "@alpinejs/focus@3.14.8" templates/base.html &amp;&amp; awk '/@alpinejs\/focus/{f=NR} /@alpinejs\/persist/{p=NR} /alpinejs@3\.14\.8\/dist\/cdn/{c=NR} END{exit !(f&&p&&c&&f<c&&p<c)}' templates/base.html &amp;&amp; test -f static/css/print.css &amp;&amp; echo OK</automated>
  </verify>
  <done>base.html rewritten with new chrome. print.css exists. Alpine plugins load before core. RBAC gating preserved on every drawer item. Configuración section hides if user has no permission to see any of the 7 items.</done>
</task>

<task type="auto">
  <name>Task 2: Rewrite static/js/app.js — 3-arg showToast, nexoChrome Alpine component, [ shortcut, all 2-arg callers updated</name>
  <read_first>
    - `static/js/app.js` in full (569 lines). Identify the 2-arg `showToast` definition (line 410) and the 4 call sites inside this file (lines ~550-557 in the preflight modal block).
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Flash toast" (variant table + mapping producing→success etc.).
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-RESEARCH.md` §Pitfall 3 (showToast signature conflict) + §Pitfall 6 (setInterval cleanup) + §Pitfall 7 (`[` guard).
    - All template files that call `showToast` (grep output in the interfaces block above) — identify exact line numbers for each 2-arg or 1-arg call.
  </read_first>
  <action>
### Part A — Rewrite `showToast` in `static/js/app.js`

Replace the existing `showToast` function in `static/js/app.js` (around
line 410) with the 3-arg canonical form. Place it at the TOP of
`app.js` (after any existing banner comment block) so every consumer
that loads `app.js` gets the correct definition. Remove the old 2-arg
function.

```js
// ── Toast notifications (UI-SPEC §Flash toast, D-30) ──────────────────────
// Canonical signature: showToast(type, title, msg)
//   type:  info | success | warn | error |
//          (legacy alias) producing | stopped | incidence | alarm | turno
//   title: short bold line (required)
//   msg:   optional secondary text (may be null/undefined/empty)
//
// Phase 8 Pitfall 3: Historically `base.html` declared a 3-arg form and
// `app.js` a 2-arg `(message, type)` form. Plan 08-02 locks the 3-arg
// contract and updates every caller. Legacy type names map to the new
// set so the Alpine pabellon telemetry keeps working without churn.

const _TOAST_VARIANT_ALIAS = Object.freeze({
  producing: 'success',
  stopped:   'warn',
  incidence: 'error',
  alarm:     'error',
  turno:     'info',
});

const _TOAST_ICON_PATH = Object.freeze({
  info:    'M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z',
  success: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z',
  warn:    'M12 9v2m0 4h.01M12 3l9 16H3l9-16z',
  error:   'M6 18L18 6M6 6l12 12',
});

const _TOAST_BORDER_COLOR = Object.freeze({
  info:    'border-l-info',
  success: 'border-l-success',
  warn:    'border-l-warn',
  error:   'border-l-error',
});

const _TOAST_ICON_COLOR = Object.freeze({
  info:    'text-info',
  success: 'text-success',
  warn:    'text-warn',
  error:   'text-error',
});

const _TOAST_ROLE = Object.freeze({
  info:    'status',
  success: 'status',
  warn:    'alert',
  error:   'alert',
});

window.showToast = function (type, title, msg) {
  const root = document.getElementById('toast-root');
  if (!root) return;

  const variant = _TOAST_VARIANT_ALIAS[type] || type;
  const safeVariant = ['info', 'success', 'warn', 'error'].includes(variant) ? variant : 'info';

  const el = document.createElement('div');
  el.setAttribute('role', _TOAST_ROLE[safeVariant]);
  el.setAttribute('tabindex', '0');
  el.className = [
    'pointer-events-auto',
    'bg-surface-base',
    'border', 'border-subtle', 'border-l-4', _TOAST_BORDER_COLOR[safeVariant],
    'rounded-md', 'shadow-popover',
    'px-4', 'py-3',
    'flex', 'items-start', 'gap-3',
    'min-w-0',
    'transition-base', 'ease-standard',
  ].join(' ');
  el.style.transform = 'translateX(16px)';
  el.style.opacity = '0';

  el.innerHTML = `
    <svg class="w-5 h-5 shrink-0 mt-0.5 ${_TOAST_ICON_COLOR[safeVariant]}" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24" aria-hidden="true">
      <path stroke-linecap="round" stroke-linejoin="round" d="${_TOAST_ICON_PATH[safeVariant]}"/>
    </svg>
    <div class="min-w-0 flex-1">
      <div class="text-sm font-semibold text-heading">${_escape(title || '')}</div>
      ${msg ? `<div class="text-sm text-body mt-0.5">${_escape(msg)}</div>` : ''}
    </div>
    <button type="button" aria-label="Cerrar aviso" class="btn-icon text-muted hover:text-body shrink-0">
      <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="1.75" viewBox="0 0 24 24" aria-hidden="true">
        <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/>
      </svg>
    </button>
  `;

  root.appendChild(el);
  // Trigger enter animation on next frame
  requestAnimationFrame(() => {
    el.style.transform = 'translateX(0)';
    el.style.opacity = '1';
  });

  let dismissTimer = null;
  const dismiss = () => {
    if (dismissTimer) { clearTimeout(dismissTimer); dismissTimer = null; }
    el.style.transform = 'translateX(16px)';
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 200);
  };
  const arm = () => { dismissTimer = setTimeout(dismiss, 4000); };

  // Pause on hover (D-30)
  el.addEventListener('mouseenter', () => { if (dismissTimer) { clearTimeout(dismissTimer); dismissTimer = null; } });
  el.addEventListener('mouseleave', arm);
  el.querySelector('button').addEventListener('click', dismiss);
  arm();
};

function _escape(str) {
  const div = document.createElement('div');
  div.textContent = String(str);
  return div.innerHTML;
}
```

### Part B — Append `nexoChrome()` Alpine component + `[` shortcut

Append (after the new `showToast` definition, before any existing
content) the `nexoChrome()` Alpine registration:

```js
// ── nexoChrome() Alpine component (Phase 8 / Plan 08-02) ──────────────────
// Drives the drawer open/close state + the [ keyboard shortcut.
// Persists `drawerOpen` via @alpinejs/persist under key `nexo.ui.drawerOpen`.
// Pitfall 5: $persist on first load without a persisted value returns the
// default (false) — safe. No migration needed.
// Pitfall 7: the [ listener guards against firing while typing in inputs.

function nexoChrome() {
  return {
    drawerOpen: (window.Alpine && window.Alpine.$persist)
      ? window.Alpine.$persist(false).as('nexo.ui.drawerOpen')
      : false,  // Fallback if $persist not yet loaded — rare
    toggleDrawer() { this.drawerOpen = !this.drawerOpen; },
    openDrawer()   { this.drawerOpen = true; },
    closeDrawer()  { this.drawerOpen = false; },
    onKeydown(e) {
      // Esc closes drawer first (z-order priority handled by Alpine @keydown)
      if (e.key === 'Escape' && this.drawerOpen) {
        this.closeDrawer();
        return;
      }
      // [ toggles drawer, unless typing
      if (e.key === '[' && !_isTyping(e.target)) {
        e.preventDefault();
        this.toggleDrawer();
      }
    },
  };
}

function _isTyping(el) {
  if (!el) return false;
  const tag = (el.tagName || '').toUpperCase();
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (el.isContentEditable) return true;
  return false;
}

// Expose for Alpine.data() auto-pickup
window.nexoChrome = nexoChrome;
```

### Part C — Update every 2-arg / 1-arg caller in `static/js/app.js`

Rewrite the preflight modal block's 3 calls (~lines 550-557) to the 3-arg form:

- `showToast(\`Solicitud enviada. Ver en /mis-solicitudes (#${approvalId}).\`, 'info');`
  → `window.showToast('info', 'Solicitud enviada', \`Ver en /mis-solicitudes (#${approvalId}).\`);`
- `showToast('Aprobaciones aun no disponibles (Plan 04-03).', 'error');`
  → `window.showToast('error', 'Aprobaciones no disponibles', 'Esta función todavía no está habilitada.');`
- `showToast('Error enviando la solicitud.', 'error');`
  → `window.showToast('error', 'Error', 'No se pudo enviar la solicitud.');`

Remove the old 2-arg `showToast` function stub at ~line 410. Keep the rest of `app.js` (preflight modal logic, reloj helper, charts) untouched.

### Part D — Update the 2-arg / 1-arg callers across templates

For each file below, rewrite EVERY `showToast(...)` call to the 3-arg
form `(type, title, msg)`. Mapping: 1-arg success messages → `'success', 'Hecho', <msg>`; 2-arg error messages → `'error', 'Error', <msg>`; 2-arg 'info' → `'info', 'Aviso', <msg>`. Preserve any already-3-arg call in `templates/luk4.html` verbatim.

Files to edit (exact line numbers from the grep in the interfaces block):

- `templates/informes.html` lines 277, 281, 284
- `templates/ciclos_calc.html` lines 512, 531, 534, 536, 701
- `templates/ciclos.html` line 139
- `templates/historial.html` lines 410, 415, 417, 427, 447, 524
- `templates/plantillas.html` lines 198, 212

Example rewrites:

- `showToast(\`${data.n_pdfs} PDF(s) generados\`);` → `window.showToast('success', 'Informe generado', \`${data.n_pdfs} PDF(s) generados.\`);`
- `showToast('Error al regenerar', 'error');` → `window.showToast('error', 'Error al regenerar', '');`
- `showToast('Error: ' + err, 'error');` → `window.showToast('error', 'Error', String(err));`
- `showToast('CSV exportado');` → `window.showToast('success', 'CSV exportado', '');`
- `showToast('Ciclos guardados');` → `window.showToast('success', 'Ciclos guardados', '');`
- `showToast('Ejecucion borrada');` → `window.showToast('success', 'Ejecución borrada', '');`
- `showToast('Plantilla creada');` → `window.showToast('success', 'Plantilla creada', '');`

The titles chosen are in Spanish per CLAUDE.md.
  </action>
  <acceptance_criteria>
    - `grep -cE "^function showToast\(" static/js/app.js` returns 0 (old 2-arg function removed).
    - `grep -c "window.showToast = function (type, title, msg)" static/js/app.js` returns 1.
    - `grep -c "function nexoChrome()" static/js/app.js` returns 1.
    - `grep -c "'\\['" static/js/app.js` returns 1 or more (the `[` key literal in onKeydown).
    - `grep -c "Alpine.\\\$persist" static/js/app.js` returns 1 or more.
    - `grep -c "nexo.ui.drawerOpen" static/js/app.js` returns 1 or more.
    - `grep -c "_isTyping" static/js/app.js` returns 2 or more (def + call).
    - `grep -rn "showToast([^t].*,[^,]*);" templates/ | grep -v "'type'" | wc -l` — no 2-arg callers remain (manual inspection: any line that matches `showToast\(('[^']+'|\`[^\`]+\`), '(error|info|success|warn)'\)` is a legacy 2-arg caller). Acceptable result: 0 matches for that pattern.
    - `grep -c "window.showToast('success'" templates/informes.html` returns 1 or more (at least one rewritten caller).
    - `grep -c "window.showToast('error'" templates/historial.html` returns 3 or more.
    - `grep -rn "showToast(" templates/ static/js/app.js | grep -vE "showToast\((['\"])(info|success|warn|error)\\1" | wc -l` — count of callers that do NOT start with a recognised variant. Expected: 1 (the Jinja flash consumer still uses `if (typeof window.showToast === 'function')` which grep matches the `typeof window.showToast` line, not a call). Accept any value ≤ 2.
  </acceptance_criteria>
  <verify>
    <automated>grep -q "window.showToast = function (type, title, msg)" static/js/app.js &amp;&amp; grep -q "function nexoChrome()" static/js/app.js &amp;&amp; ! grep -qE "^function showToast\(" static/js/app.js &amp;&amp; ruff check --select=I nexo/ api/ &amp;&amp; echo OK</automated>
  </verify>
  <done>app.js ships one 3-arg showToast + nexoChrome component + [ listener + Alpine $persist drawer state. Every template caller uses the 3-arg form. Old 2-arg function gone.</done>
</task>

<task type="auto">
  <name>Task 3: Rewrite static/css/app.css to consume semantic tokens + extract gif-corona.png first frame</name>
  <read_first>
    - `static/css/app.css` (current 103 lines).
    - `static/css/tokens.css` (from 08-01) — the source of every utility referenced.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Component Inventory" (buttons, cards, form inputs, tables, modals, badges, breadcrumbs, skeleton, spinner).
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"State Contracts → Loading / Empty / Error".
    - `static/img/gif-corona.gif` — existence confirmed.
  </read_first>
  <action>
### Part A — Rewrite `static/css/app.css`

Replace `static/css/app.css` entirely. Keep class names backwards
compatible (`.card`, `.btn-primary`, `.btn-secondary`, `.btn-danger`,
`.btn-ghost`, `.btn-link`, `.btn-icon`, `.btn-sm`, `.btn-lg`,
`.input-inline`, `.data-table`, `.spinner`, `.spinner-panel`,
`.stat-card`, `.tree-item`, `.pdf-frame`, `.log-console`,
`.row-warning`, `.conn-ok/err/chk`) but re-express them on semantic
tokens.

```css
@import url('./tokens.css');

/* ──────────────────────────────────────────────────────────────
   Nexo — app.css (Phase 8 / Plan 08-02)
   Component classes on semantic tokens. Class names are
   backwards-compatible with Phase 1-7 templates.
   ────────────────────────────────────────────────────────────── */

[x-cloak] { display: none !important; }

/* ── Typography anchors ──────────────────────────────────────── */
body { font-family: var(--font-sans); }
.text-display  { font-size: var(--text-display-size); line-height: var(--text-display-lh); letter-spacing: var(--text-display-ls); font-weight: 600; }
.text-heading  { font-size: var(--text-heading-size); line-height: var(--text-heading-lh); letter-spacing: var(--text-heading-ls); font-weight: 600; }
.text-subtitle { font-size: var(--text-subtitle-size); line-height: var(--text-subtitle-lh); font-weight: 600; }

/* ── Focus ring ──────────────────────────────────────────────── */
:focus-visible {
  outline: 2px solid rgb(var(--color-focus-ring));
  outline-offset: 2px;
  border-radius: inherit;
}

/* ── Cards ───────────────────────────────────────────────────── */
.card {
  @apply bg-surface-base border border-subtle rounded-lg;
  padding: 0;
}
.card-elevated {
  @apply bg-surface-base rounded-lg shadow-card;
}
.card-accent {
  @apply bg-primary-subtle border border-subtle rounded-lg;
}
.card-header {
  @apply border-b border-subtle;
  padding: 16px 24px;
  font-size: var(--text-subtitle-size);
  line-height: var(--text-subtitle-lh);
  font-weight: 600;
  color: rgb(var(--color-text-heading));
}
.card-body {
  padding: 20px 24px;
}

/* ── Buttons ─────────────────────────────────────────────────── */
.btn {
  @apply inline-flex items-center gap-2 rounded-md transition-base ease-standard focus:outline-none;
  height: 40px;
  padding: 0 16px;
  font-size: var(--text-body-size);
  font-weight: 600;
}
.btn:disabled,
.btn[disabled] {
  @apply opacity-60 cursor-not-allowed;
}
.btn-primary {
  background-color: rgb(var(--color-primary));
  color: rgb(var(--color-text-on-accent));
}
.btn-primary:hover:not(:disabled) {
  background-color: rgb(var(--color-primary-hover));
}
.btn-secondary {
  background-color: rgb(var(--color-surface-base));
  color: rgb(var(--color-text-body));
  border: 1px solid rgb(var(--color-border-strong));
}
.btn-secondary:hover:not(:disabled) {
  background-color: rgb(var(--color-surface-subtle));
}
.btn-ghost {
  background-color: transparent;
  color: rgb(var(--color-text-muted));
}
.btn-ghost:hover:not(:disabled) {
  background-color: rgb(var(--color-surface-subtle));
  color: rgb(var(--color-text-body));
}
.btn-danger {
  background-color: rgb(var(--color-error));
  color: rgb(var(--color-text-on-accent));
}
.btn-danger:hover:not(:disabled) {
  background-color: rgb(var(--color-error) / 0.9);
}
.btn-link {
  background-color: transparent;
  color: rgb(var(--color-primary));
  padding: 0;
  text-decoration: underline;
  text-underline-offset: 2px;
}
.btn-link:hover {
  color: rgb(var(--color-primary-hover));
}
.btn-icon {
  @apply inline-flex items-center justify-center rounded-md transition-base ease-standard;
  width: 40px;
  height: 40px;
  padding: 0;
}
.btn-sm {
  height: 32px;
  padding: 0 12px;
}
.btn-lg {
  height: 48px;
  padding: 0 24px;
  font-size: var(--text-subtitle-size);
  line-height: var(--text-subtitle-lh);
}

/* Backwards compat: old .btn-success still works (some templates use it). */
.btn-success {
  background-color: rgb(var(--color-success));
  color: rgb(var(--color-text-on-accent));
}
.btn-success:hover:not(:disabled) {
  background-color: rgb(var(--color-success) / 0.9);
}

/* ── Form inputs ─────────────────────────────────────────────── */
.input-inline {
  @apply w-full rounded-md;
  height: 40px;
  padding: 8px 12px;
  font-size: var(--text-body-size);
  line-height: var(--text-body-lh);
  color: rgb(var(--color-text-body));
  background-color: rgb(var(--color-surface-base));
  border: 1px solid rgb(var(--color-border-strong));
  transition: border-color var(--duration-fast) var(--ease-standard),
              box-shadow var(--duration-fast) var(--ease-standard);
}
.input-inline::placeholder {
  color: rgb(var(--color-text-disabled));
}
.input-inline:hover {
  border-color: rgb(var(--color-primary) / 0.5);
}
.input-inline:focus {
  border-color: rgb(var(--color-primary));
  box-shadow: 0 0 0 2px rgb(var(--color-primary) / 0.3);
  outline: none;
}
.input-inline:disabled {
  background-color: rgb(var(--color-surface-subtle));
  color: rgb(var(--color-text-disabled));
  cursor: not-allowed;
}
.input-inline[aria-invalid="true"] {
  border-color: rgb(var(--color-error));
}
.input-inline[aria-invalid="true"]:focus {
  box-shadow: 0 0 0 2px rgb(var(--color-error) / 0.3);
}

/* ── Tables ──────────────────────────────────────────────────── */
.data-table {
  @apply w-full;
  font-size: var(--text-body-size);
  line-height: var(--text-body-lh);
}
.data-table th {
  @apply text-left uppercase;
  background-color: rgb(var(--color-surface-subtle));
  color: rgb(var(--color-text-muted));
  border-bottom: 1px solid rgb(var(--color-border-subtle));
  padding: 12px 16px;
  font-weight: 600;
  letter-spacing: 0.025em;
}
.data-table td {
  padding: 12px 16px;
  border-bottom: 1px solid rgb(var(--color-border-subtle));
  color: rgb(var(--color-text-body));
}
.data-table tr:last-child td {
  border-bottom: none;
}
.data-table tr:hover td {
  background-color: rgb(var(--color-surface-subtle));
}
.data-table td.numeric,
.data-table th.numeric {
  @apply text-right;
  font-variant-numeric: tabular-nums;
  font-family: var(--font-mono);
}

/* ── Spinners ────────────────────────────────────────────────── */
.spinner {
  @apply inline-block rounded-full;
  width: 16px;
  height: 16px;
  border: 2px solid currentColor;
  border-right-color: transparent;
  animation: spin 0.6s linear infinite;
}
.spinner-panel {
  display: flex; align-items: center; justify-content: center;
  min-height: 240px;
}
.spinner-panel .spinner {
  width: 32px;
  height: 32px;
  border-width: 3px;
  border-color: rgb(var(--color-surface-muted));
  border-top-color: rgb(var(--color-primary));
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Loading hero (Tier 2 — full-page) ───────────────────────── */
.loading-hero {
  @apply flex flex-col items-center justify-center;
  min-height: 60vh;
}
.loading-hero img {
  width: 160px;
  height: 160px;
  object-fit: contain;
}
@media (prefers-reduced-motion: reduce) {
  .loading-hero img[data-gif="true"] {
    display: none;
  }
  .loading-hero img[data-png="true"] {
    display: block;
  }
}
.loading-hero img[data-png="true"] {
  display: none;
}

/* ── Empty + Error states ────────────────────────────────────── */
.empty-state {
  @apply flex flex-col items-center text-center;
  padding: 64px 24px;
}
.empty-state svg {
  color: rgb(var(--color-text-muted));
}
.error-state {
  @apply rounded-lg;
  padding: 32px 24px;
  background-color: rgb(var(--color-surface-base));
  box-shadow: var(--shadow-card);
  text-align: center;
}
.error-state svg {
  color: rgb(var(--color-error));
}

/* ── Stat card ───────────────────────────────────────────────── */
.stat-card {
  @apply card flex flex-col gap-1;
  padding: 20px 24px;
}
.stat-card .stat-value {
  font-size: var(--text-display-size);
  line-height: var(--text-display-lh);
  letter-spacing: var(--text-display-ls);
  font-weight: 600;
  color: rgb(var(--color-text-heading));
  font-variant-numeric: tabular-nums;
}
.stat-card .stat-label {
  font-size: var(--text-body-size);
  text-transform: uppercase;
  letter-spacing: 0.025em;
  color: rgb(var(--color-text-muted));
  font-weight: 600;
}

/* ── Log console (preserve — pipeline) ────────────────────────── */
.log-console {
  @apply rounded-md overflow-y-auto;
  background-color: rgb(15 23 42);   /* surface-900 intentional dark */
  color: rgb(134 239 172);           /* green accent for terminal readability */
  font-family: var(--font-mono);
  font-size: 12px;
  padding: 16px;
  min-height: 100px;
  max-height: 400px;
}

/* ── Badge / pill ────────────────────────────────────────────── */
.badge {
  @apply inline-flex items-center uppercase tracking-wide rounded-pill;
  padding: 2px 8px;
  font-size: var(--text-body-size);
  font-weight: 600;
}
.badge-neutral { background-color: rgb(var(--color-surface-muted)); color: rgb(var(--color-text-muted)); }
.badge-brand   { background-color: rgb(var(--color-primary-subtle)); color: rgb(var(--color-primary)); }
.badge-success { background-color: rgb(var(--color-success-subtle)); color: rgb(var(--color-success)); }
.badge-warn    { background-color: rgb(var(--color-warn-subtle));    color: rgb(var(--color-warn));    }
.badge-error   { background-color: rgb(var(--color-error-subtle));   color: rgb(var(--color-error));  }

/* ── Breadcrumbs ─────────────────────────────────────────────── */
.breadcrumbs {
  @apply flex items-center gap-1 mb-4;
  font-size: var(--text-body-size);
  color: rgb(var(--color-text-muted));
}
.breadcrumbs a { color: rgb(var(--color-text-muted)); }
.breadcrumbs a:hover { color: rgb(var(--color-text-body)); }
.breadcrumbs .current { color: rgb(var(--color-text-body)); font-weight: 600; }
.breadcrumbs .sep { color: rgb(var(--color-text-disabled)); }

/* ── Connection badge (preserve — topbar HTMX) ───────────────── */
.conn-ok  { background-color: rgb(var(--color-success-subtle)); color: rgb(var(--color-success)); }
.conn-err { background-color: rgb(var(--color-error-subtle));   color: rgb(var(--color-error));   }
.conn-chk { background-color: rgb(var(--color-warn-subtle));    color: rgb(var(--color-warn));    }

/* ── Row highlights (preserve) ───────────────────────────────── */
.row-warning td { background-color: rgb(var(--color-warn-subtle)); }

/* ── Tree view (preserve — bbdd, informes) ───────────────────── */
.tree-item { cursor: pointer; user-select: none; }
.tree-item:hover { background-color: rgb(var(--color-surface-subtle)); border-radius: var(--radius-sm); }

/* ── PDF viewer (preserve exact height — D-16) ───────────────── */
.pdf-frame {
  @apply w-full border-0;
  border-radius: var(--radius-md);
  height: calc(100vh - 200px);
}
```

### Part B — Extract first frame of `gif-corona.gif` to `gif-corona.png`

Run the extraction. Try ffmpeg first, fall back to gif2png, fall back to
a minimal placeholder PNG with a TODO note if neither tool is available.

```bash
cd /home/eeguskiza/analisis_datos
if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -y -i static/img/gif-corona.gif -vframes 1 static/img/gif-corona.png 2>/dev/null
elif command -v gif2png >/dev/null 2>&1; then
  gif2png -O -f static/img/gif-corona.gif
  # gif2png writes gif-corona.png next to the gif — OK
else
  # Minimal 1x1 transparent PNG as stopgap. The test asserts the file
  # exists and is a valid PNG; UX improvement is a Mark-IV task.
  printf '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\x0d\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82' > static/img/gif-corona.png
fi

# Confirm
file static/img/gif-corona.png
```

If only the stopgap is produced, leave a TODO note for a future Mark-IV
visual polish pass in `docs/DATA_MIGRATION_NOTES.md` or
`docs/OPEN_QUESTIONS.md` — one line under an existing section.
  </action>
  <acceptance_criteria>
    - `head -1 static/css/app.css` matches `@import url('./tokens.css');`.
    - `grep -c "^.btn-primary {" static/css/app.css` returns 1.
    - `grep -c "^.btn-secondary {" static/css/app.css` returns 1.
    - `grep -c "^.btn-icon {" static/css/app.css` returns 1.
    - `grep -c "^.btn-link {" static/css/app.css` returns 1.
    - `grep -c "^.btn-lg {" static/css/app.css` returns 1.
    - `grep -c "^.empty-state {" static/css/app.css` returns 1.
    - `grep -c "^.error-state {" static/css/app.css` returns 1.
    - `grep -c "^.loading-hero {" static/css/app.css` returns 1.
    - `grep -c "^.badge-" static/css/app.css` returns 5.
    - `grep -c "^.breadcrumbs {" static/css/app.css` returns 1.
    - `grep -c "bg-red-600\|bg-green-600\|bg-blue-600\|text-gray-" static/css/app.css` returns 0 (no raw Tailwind greys/primaries — all semantic now).
    - `test -f static/img/gif-corona.png` returns 0.
    - `file static/img/gif-corona.png` output contains `PNG image data`.
  </acceptance_criteria>
  <verify>
    <automated>head -1 static/css/app.css | grep -F "@import url('./tokens.css');" &amp;&amp; grep -c "^.btn-primary {" static/css/app.css | grep -q "^1$" &amp;&amp; grep -c "^.empty-state {" static/css/app.css | grep -q "^1$" &amp;&amp; test -f static/img/gif-corona.png &amp;&amp; file static/img/gif-corona.png | grep -q "PNG image data"</automated>
  </verify>
  <done>app.css fully rewritten on semantic tokens. Every legacy class name still works. gif-corona.png exists (real first-frame when ffmpeg available, stopgap otherwise). No template touched in this task.</done>
</task>

<task type="auto">
  <name>Task 4: Regression tests — flash toast contract + chrome structure + RUNBOOK slug invariant</name>
  <read_first>
    - `tests/routers/` directory — existing pytest style.
    - `tests/infra/test_deploy_lan_doc.py` — doc-parse regression pattern.
    - `templates/base.html` (rewritten in Task 1) — grep fixtures.
    - `static/js/app.js` (rewritten in Task 2).
    - `docs/RUNBOOK.md` — the 5 headings + their GFM slugs.
  </read_first>
  <action>
### Part A — `tests/routers/test_flash_toast_contract.py`

```python
"""Regression for Phase 8 / Pitfall 3: showToast 3-arg contract.

Locks:
1. base.html flash consumer calls showToast('info', 'Aviso', msg) — 3-arg.
2. app.js defines ONE window.showToast with the 3-arg signature.
3. No remaining 2-arg callers in static/js or templates — every legacy
   caller was rewritten in Plan 08-02.
4. No duplicate showToast definition in base.html (the old inline one
   was removed in favour of the single definition in app.js).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_BASE_HTML = _ROOT / "templates" / "base.html"
_APP_JS = _ROOT / "static" / "js" / "app.js"
_TEMPLATES_DIR = _ROOT / "templates"


def test_base_html_has_no_inline_showtoast_definition():
    text = _BASE_HTML.read_text(encoding="utf-8")
    # The DEFINITION is gone — only the call site for flash_message remains.
    matches = re.findall(r"window\.showToast\s*=\s*function", text)
    assert matches == [], (
        "base.html must not define window.showToast — the single definition "
        "lives in /static/js/app.js per D-30"
    )


def test_base_html_flash_block_uses_three_arg_call():
    text = _BASE_HTML.read_text(encoding="utf-8")
    # The flash consumer MUST pass (type, title, msg)
    # Regex tolerates the Jinja {{ flash_message|tojson }} expression.
    pattern = re.compile(
        r"window\.showToast\(\s*'info'\s*,\s*'Aviso'\s*,\s*\{\{\s*flash_message\|tojson\s*\}\}\s*\)"
    )
    assert pattern.search(text) is not None, (
        "Jinja flash consumer must call window.showToast('info', 'Aviso', {{ flash_message|tojson }})"
    )


def test_app_js_defines_three_arg_show_toast():
    text = _APP_JS.read_text(encoding="utf-8")
    # Canonical definition must be present.
    assert re.search(
        r"window\.showToast\s*=\s*function\s*\(\s*type\s*,\s*title\s*,\s*msg\s*\)",
        text,
    ), "app.js must declare `window.showToast = function (type, title, msg)`"
    # Old 2-arg function declaration must be gone.
    assert re.search(
        r"^\s*function\s+showToast\s*\(\s*message\s*,\s*type",
        text,
        re.MULTILINE,
    ) is None, "app.js must not contain the legacy 2-arg showToast function"


LEGACY_2_ARG_RE = re.compile(
    # showToast('foo', 'error')  /  showToast(`foo`, 'error')
    # EXCLUDES 3-arg: showToast('error', 'Title', 'msg')
    r"\bshowToast\(\s*(['\"`])([^'\"`]+)\1\s*,\s*(['\"])(error|info|success|warn)\3\s*\)"
)


@pytest.mark.parametrize(
    "template_name",
    [
        "historial.html",
        "informes.html",
        "ciclos.html",
        "ciclos_calc.html",
        "plantillas.html",
    ],
)
def test_no_legacy_two_arg_callers_in_template(template_name: str):
    text = (_TEMPLATES_DIR / template_name).read_text(encoding="utf-8")
    matches = LEGACY_2_ARG_RE.findall(text)
    assert matches == [], (
        f"{template_name} still has 2-arg showToast callers: {matches}"
    )


def test_no_legacy_two_arg_callers_in_app_js():
    text = _APP_JS.read_text(encoding="utf-8")
    matches = LEGACY_2_ARG_RE.findall(text)
    assert matches == [], f"app.js still has 2-arg showToast callers: {matches}"
```

### Part B — `tests/infra/test_chrome_structure.py`

```python
"""Regression for Phase 8 chrome structure.

Locks:
1. base.html loads Alpine Focus + Persist BEFORE Alpine core.
2. base.html declares the drawer with role=dialog + aria-modal + x-trap.
3. base.html includes Tailwind config <script> BEFORE the CDN script.
4. base.html loads print.css behind media="print".
5. base.html declares toast-root container.
6. RUNBOOK.md has the 5 canonical scenario headings with stable slugs.
7. Any template that hard-codes a RUNBOOK anchor uses the full GFM slug
   (no short form like #escenario-1-mes-caido without the rest).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_BASE_HTML = _ROOT / "templates" / "base.html"
_RUNBOOK_MD = _ROOT / "docs" / "RUNBOOK.md"
_PRINT_CSS = _ROOT / "static" / "css" / "print.css"
_TOKENS_CSS = _ROOT / "static" / "css" / "tokens.css"


def test_tokens_css_present():
    assert _TOKENS_CSS.exists()


def test_print_css_present():
    assert _PRINT_CSS.exists()
    assert "@media print" in _PRINT_CSS.read_text(encoding="utf-8")


def test_base_html_loads_alpine_focus_persist_before_core():
    text = _BASE_HTML.read_text(encoding="utf-8")
    focus_idx = text.find("@alpinejs/focus@3.14.8")
    persist_idx = text.find("@alpinejs/persist@3.14.8")
    alpine_idx = text.find("alpinejs@3.14.8/dist/cdn.min.js")
    assert focus_idx > 0, "Alpine Focus plugin not loaded"
    assert persist_idx > 0, "Alpine Persist plugin not loaded"
    assert alpine_idx > 0, "Alpine core not loaded"
    assert focus_idx < alpine_idx, "Alpine Focus must appear before core (Pitfall 4)"
    assert persist_idx < alpine_idx, "Alpine Persist must appear before core (Pitfall 4)"


def test_base_html_loads_tailwind_config_before_cdn():
    text = _BASE_HTML.read_text(encoding="utf-8")
    cfg_idx = text.find('src="/static/js/tailwind.config.js"')
    cdn_idx = text.find("cdn.tailwindcss.com")
    assert cfg_idx > 0 and cdn_idx > 0, "tailwind.config.js + CDN both required"
    assert cfg_idx < cdn_idx, "tailwind.config.js must load before the CDN (Pitfall 1)"


def test_base_html_loads_print_css():
    text = _BASE_HTML.read_text(encoding="utf-8")
    assert 'href="/static/css/print.css" media="print"' in text


def test_base_html_drawer_has_a11y_attributes():
    text = _BASE_HTML.read_text(encoding="utf-8")
    assert 'id="nexo-drawer"' in text
    assert 'role="dialog"' in text
    assert 'aria-modal="true"' in text
    assert 'x-trap.noscroll="drawerOpen"' in text


def test_base_html_has_toast_root():
    text = _BASE_HTML.read_text(encoding="utf-8")
    assert 'id="toast-root"' in text
    assert 'aria-live="polite"' in text


def test_base_html_has_hamburger_with_aria_label():
    text = _BASE_HTML.read_text(encoding="utf-8")
    assert 'aria-label="Abrir menú"' in text
    assert 'aria-controls="nexo-drawer"' in text


def test_base_html_has_nexo_chrome_alpine_component():
    text = _BASE_HTML.read_text(encoding="utf-8")
    assert 'x-data="nexoChrome()"' in text


def test_base_html_configuracion_section_has_can_gating():
    text = _BASE_HTML.read_text(encoding="utf-8")
    # Configuración label must appear inside a conditional (config_visible.any).
    assert "Configuración" in text
    assert "config_visible" in text or "ajustes:manage" in text


# ── RUNBOOK slug invariants ────────────────────────────────────────────────

EXPECTED_HEADINGS = [
    "## Escenario 1: MES caido (SQL Server dbizaro inaccesible)",
    "## Escenario 2: Postgres no arranca",
    "## Escenario 3: Certificado Caddy expira / warning en browsers",
    "## Escenario 4: Pipeline atascado (HALLAZGO CRITICO: semaforo in-process)",
    "## Escenario 5: Lockout del unico propietario (HALLAZGO CRITICO: no hay `unlock_user`)",
]


def _gfm_slug(heading_line: str) -> str:
    """Simplified GFM slugifier: lower, spaces->hyphens, strip [^\w- ]."""
    text = heading_line.lstrip("# ").strip().lower()
    text = text.replace("`", "")
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "-", text)
    return text.strip("-")


@pytest.mark.parametrize("heading", EXPECTED_HEADINGS)
def test_runbook_canonical_heading_present(heading: str):
    text = _RUNBOOK_MD.read_text(encoding="utf-8")
    assert heading in text, (
        f"RUNBOOK.md must contain heading: {heading}"
    )


def test_runbook_headings_are_unique():
    text = _RUNBOOK_MD.read_text(encoding="utf-8")
    for heading in EXPECTED_HEADINGS:
        count = text.count(heading)
        assert count == 1, (
            f"Heading must appear exactly once in RUNBOOK.md, found {count}: {heading}"
        )


def test_expected_gfm_slugs_computed():
    """Sanity: the slugs UI-SPEC error-state copy must use."""
    expected = {
        "Escenario 1": "escenario-1-mes-caido-sql-server-dbizaro-inaccesible",
        "Escenario 2": "escenario-2-postgres-no-arranca",
        "Escenario 3": "escenario-3-certificado-caddy-expira--warning-en-browsers",
        "Escenario 4": "escenario-4-pipeline-atascado-hallazgo-critico-semaforo-in-process",
        "Escenario 5": "escenario-5-lockout-del-unico-propietario-hallazgo-critico-no-hay-unlock_user",
    }
    for heading in EXPECTED_HEADINGS:
        # Extract the key like "Escenario 1"
        label = heading.split(":")[0].lstrip("# ").strip()
        got = _gfm_slug(heading)
        assert got == expected[label], (
            f"Slug drift for {label}: expected {expected[label]!r}, got {got!r}"
        )
```

### Part C — run full suite

Run the full `pytest tests/` locally (no new deps needed) and confirm it
is GREEN. This is the Phase 8 D-20 guard: the Phase 5 + Phase 7 test
suites must not regress with the chrome rewrite.

If a test fails because an old test grepped for the old chrome (e.g.
`sidebarOpen` in base.html), update that test to the new chrome API.
Log each such edit in the plan summary.
  </action>
  <acceptance_criteria>
    - `test -f tests/routers/test_flash_toast_contract.py` returns 0.
    - `test -f tests/infra/test_chrome_structure.py` returns 0.
    - `pytest tests/routers/test_flash_toast_contract.py tests/infra/test_chrome_structure.py -x -q` exit 0.
    - `pytest tests/ -x -q` (full suite) exit 0. Any pre-existing test that asserted the old sidebar chrome (e.g. `sidebarOpen`, `aside.bg-brand-800`) must be updated to the new chrome API before this plan closes. Each update logged in the SUMMARY.
    - `ruff check tests/routers/test_flash_toast_contract.py tests/infra/test_chrome_structure.py` exit 0.
    - `ruff format --check tests/routers/test_flash_toast_contract.py tests/infra/test_chrome_structure.py` exit 0.
  </acceptance_criteria>
  <verify>
    <automated>ruff check tests/routers/test_flash_toast_contract.py tests/infra/test_chrome_structure.py &amp;&amp; ruff format --check tests/routers/test_flash_toast_contract.py tests/infra/test_chrome_structure.py &amp;&amp; pytest tests/routers/test_flash_toast_contract.py tests/infra/test_chrome_structure.py -x -q &amp;&amp; pytest tests/ -x -q</automated>
  </verify>
  <done>Both regression tests pass. Full suite green. No Phase 5 or Phase 7 test was regressed.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Browser → Alpine $persist | `localStorage.nexo.ui.drawerOpen` is writable by any same-origin script; trust level = client-only. |
| Server → client (flash toast) | `flash_message` flows through a signed-ish but user-settable cookie; already sanitised by `nexo.middleware.flash` + `|tojson` escape in Jinja. |
| Template → DOM (showToast title/msg) | Both inserted via `innerHTML` after `_escape`; must escape every dynamic value to prevent XSS. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-08-02-01 | Tampering | `localStorage.nexo.ui.drawerOpen` value | accept | Purely cosmetic (open/closed). An attacker flipping it only changes the user's own drawer state. No auth bypass. |
| T-08-02-02 | Information Disclosure | drawer nav items render server-side with `{% if can() %}` — no client-side gating | mitigate | Phase 5 zero-trust DOM preserved verbatim; Alpine `x-show` is never used for permission decisions in this plan. Tests in `tests/routers/test_nav_items_permission.py` already cover this. |
| T-08-02-03 | XSS | `showToast` renders `title` + `msg` as `innerHTML` | mitigate | `_escape(str)` helper uses DOM `textContent` round-trip — turns `<script>` into `&lt;script&gt;`. Unit-testable. Every caller passes a string literal or a `String(err)` already stringified. |
| T-08-02-04 | Elevation of Privilege | `[` bare key listener could fire inside admin forms | mitigate | `_isTyping(el)` guard skips INPUT/TEXTAREA/SELECT/contentEditable. Only affects chrome — no auth-relevant action happens on drawer toggle. |
| T-08-02-05 | Repudiation | Flash toast does not log | accept | Flash is UX feedback, not a security event. Phase 5 audit log handles the underlying action. |
| T-08-02-06 | DoS | `setInterval` leak in reloj / toast timers | mitigate | Reloj is in Plan 08-04 (out of scope here); toast timers clear themselves in `dismiss()`. No setInterval in app.js toast path. |
</threat_model>

<verification>
Whole-phase sanity after this plan:

1. `pytest tests/ -x -q` (full suite) exits 0.
2. `ruff check api/ nexo/` and `ruff format --check api/ nexo/` pass.
3. Manual: `make dev`, open the app in a browser.
   - The sidebar is GONE; the chrome is a top bar with a hamburger.
   - Clicking the hamburger slides the drawer in over the content with a
     backdrop blur; Esc or clicking the backdrop closes it.
   - Pressing `[` toggles the drawer. Typing `[` inside the BBDD
     query input or the capacidad date field does NOT toggle it.
   - Flashing a message via a 403 (e.g. visit a gated URL as a
     non-propietario) shows a top-right toast with the Aviso title.
   - The Configuración section in the drawer only appears for
     propietario (or any user with at least one of the 7 perms).
   - Refresh the page: the drawer state (open/closed) is preserved via
     localStorage.
   - Ctrl+P shows a print preview with topbar / drawer / toast hidden.
</verification>

<success_criteria>
- `templates/base.html` rewritten, loads tokens + Tailwind config + print.css + Alpine Focus + Alpine Persist + Alpine core (in the correct order).
- Drawer has `role="dialog"`, `aria-modal`, `x-trap`, `$persist`, animates ≤200ms, closes on Esc/backdrop/link.
- `window.showToast(type, title, msg)` is the single canonical API.
  Every 2-arg caller rewritten. Test locks the contract.
- Phase 5 RBAC intact: every gated nav item still uses `can(current_user, perm)`.
- `pytest tests/` green. No Phase 5 / Phase 7 regression.
- `gif-corona.png` exists (best-effort first-frame extraction).
- RUNBOOK slug test passes.
</success_criteria>

<output>
After completion, create `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-02-SUMMARY.md` with:

- Files rewritten: `templates/base.html`, `static/css/app.css`, `static/js/app.js`.
- Files added: `static/css/print.css`, `static/img/gif-corona.png`, `tests/routers/test_flash_toast_contract.py`, `tests/infra/test_chrome_structure.py`.
- showToast call sites migrated: list of files + line numbers.
- Any pre-existing tests that had to be updated to the new chrome — list each with a one-line rationale.
- Notable deviations from UI-SPEC (should be none).
- Was ffmpeg available for `gif-corona.png`? If not, note stopgap.
- Handoff to 08-03: `current_user.nombre` column is referenced in base.html display_name logic — Plan 08-03 creates the column + migration + form update so the fallback (`email.split('@')[0]|capitalize`) is only used for legacy accounts without `nombre`.
</output>