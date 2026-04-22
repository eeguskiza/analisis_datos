---
phase: 08
slug: redise-o-ui-modo-claro-moderno
status: draft
shadcn_initialized: false
preset: none
created: 2026-04-22
requirements: [UIREDO-01, UIREDO-02, UIREDO-03, UIREDO-04, UIREDO-05, UIREDO-06, UIREDO-07, UIREDO-08]
source_decisions: 31   # CONTEXT.md D-01..D-31 (D-23 moved into Section F)
---

# Phase 08 — UI Design Contract (Nexo, modo claro moderno)

> Visual and interaction contract for the Mark-III UI redesign. Locks every
> token, component variant, state, copy string, interaction budget and
> per-screen adaptation needed by downstream agents (`gsd-ui-checker`,
> `gsd-planner`, `gsd-executor`, `gsd-ui-auditor`).
>
> Non-negotiables carried from CLAUDE.md + CONTEXT.md + Phase 5:
>
> - No emojis anywhere (code, templates, copy, commits, docs).
> - User-facing text in Spanish; technical identifiers in English.
> - Stack locked: Tailwind CDN + Alpine 3.14 + HTMX 2.0.4 + Jinja2. No
>   React/Vue/build step/Google Fonts.
> - Light theme only (D-04). No dark hooks.
> - RBAC intact: `can()` Jinja helper + `{% if can() %}` gating + zero-trust
>   DOM. The flash middleware contract (Phase 5 D-07/D-08) is preserved —
>   only the toast surface is restyled.
> - Centro de Mando interaction is LOCKED (D-16): plano de fondo +
>   máquinas editables. Only tokens, typography, spacing and chrome around
>   the pabellón may change; no behavioural edits to `_partials/mapa_pabellon.html`.

---

## Design System

| Property | Value |
|----------|-------|
| Tool | none (shadcn gate N/A — stack is Jinja/Alpine, not React) |
| Preset | not applicable |
| Component library | none (custom utility classes in `static/css/app.css` + Tailwind CDN) |
| Icon library | Heroicons outline (inline SVG, stroke-width 1.5, 24x24 viewBox) |
| Font | Tailwind `font-sans` system stack — SF / Segoe UI / system-ui (D-06) |
| Theme | Light only. No `[data-theme='dark']`, no `prefers-color-scheme` (D-04) |
| Tokens file | `static/css/tokens.css` (two-layer: raw + semantic, D-01) |
| Tailwind config | extracted from `base.html` to `static/js/tailwind.config.js`, consumed via `theme.extend` referencing CSS vars as `rgb(var(--token) / <alpha-value>)` (D-02) |
| Loading hero asset | `static/img/gif-corona.gif` — used for full-page / long-running loads only (see State Contracts) |

---

## Spacing Scale

All values are multiples of 4. Tailwind's default spacing scale is the
substrate; the semantic aliases below sit on top of it and are used when a
field is meaningfully named (card padding vs ad-hoc 16px gap).

### Raw scale (Tailwind defaults, explicitly enumerated)

| Token | Value | Tailwind utility | Usage |
|-------|-------|------------------|-------|
| xs | 4px | `p-1 / gap-1 / m-1` | Icon gaps, inline padding |
| sm | 8px | `p-2 / gap-2 / m-2` | Compact element spacing, small pills |
| md | 16px | `p-4 / gap-4 / m-4` | Default element spacing, form gaps |
| lg | 24px | `p-6 / gap-6 / m-6` | Card padding, section padding |
| xl | 32px | `p-8 / gap-8 / m-8` | Layout gaps, page gutters |
| 2xl | 48px | `p-12 / gap-12` | Major section breaks |
| 3xl | 64px | `p-16 / gap-16` | Page-level hero spacing |

### Semantic aliases (declared in `tokens.css`, D-05)

| Token | Value | Intent |
|-------|-------|--------|
| `--space-field-gap` | 16px | Vertical gap between form fields |
| `--space-card-padding` | 24px | Inner padding of `.card` |
| `--space-card-gap` | 16px | Gap between stacked cards |
| `--space-section-gap` | 32px | Gap between top-level sections of a page |
| `--space-page-gutter-x` | 24px | Page horizontal gutter on desktop |
| `--space-page-gutter-x-mobile` | 16px | Page horizontal gutter on mobile |
| `--space-topbar-h` | 56px | Fixed top bar height on desktop |
| `--space-topbar-h-mobile` | 52px | Top bar height on mobile |
| `--space-drawer-w` | 280px | Drawer width (desktop + mobile overlay) |
| `--space-drawer-padding` | 16px | Inner padding of drawer items |

### Exceptions

| Element | Value | Rationale |
|---------|-------|-----------|
| Icon-only buttons | 40x40px (hit target) | 36px visual target falls below touch min; keep visual square on 16px grid (10x scale) |
| Mobile nav touch targets | min 44x44px | WCAG 2.1 AA §Target Size (Enhanced) for drawer items on mobile overlay |
| `_partials/mapa_pabellon.html` zone editor arrows | 32x32px | Legacy editor; preserved verbatim because interaction is LOCKED (D-16). Visible in editor mode only. |
| `.pdf-frame` height | `calc(100vh - 200px)` | Existing rule in `app.css` — keep. Viewer needs full viewport. |

---

## Typography

Four sizes, two weights. Line-heights fixed per role.

| Role | Size | Weight | Line Height | Letter Spacing | CSS var |
|------|------|--------|-------------|----------------|---------|
| Body (labels, inputs, table cells, nav items, paragraphs, captions, badges, footnotes) | 14px | 400 (regular) | 1.5 | 0 | `--text-body` |
| Subtitle (card headers, section titles, form legends) | 16px | 600 (semibold) | 1.4 | 0 | `--text-subtitle` |
| Heading (page title in top bar, modal title, primary CTA in landing) | 20px | 600 (semibold) | 1.3 | -0.01em | `--text-heading` |
| Display (landing greeting, large KPI digits) | 32px | 600 (semibold) | 1.2 | -0.02em | `--text-display` |

Weights in use: **400 (regular)** and **600 (semibold)**. That is the full
set — no other weights are allowed anywhere in the system. 400 carries body
copy, table cells, inputs, paragraphs, captions and footnote text. 600
carries Subtitle/Heading/Display, primary CTA labels, active nav items,
badge labels, table header cells and any inline emphasis that used to be
"medium". The Display role intentionally stops at 600: impact comes from
the 32px size + `-0.02em` letter-spacing, not from a heavier weight.

Line-height policy:

- Body text (14px) uses 1.5 — WCAG 1.4.12 (Text Spacing) compliant.
- Subtitle / Heading (16–20px) use 1.3–1.4 — tight, modern.
- Display (32px) uses 1.2 — posters / hero.

### Migration notes (from the pre-consolidation spec)

The earlier draft declared 5 sizes × 4 weights (Micro 12 + 500, Display
700). Dimension 4 caps the system at 4 sizes × 2 weights, so the extra
size and extra weights are dropped and their roles migrated:

| Former role | Former token | Migrates to |
|-------------|--------------|-------------|
| Micro / badge label (12px, 500, uppercase) | `--text-micro` | `--text-body` at weight 600, still uppercase + `tracking-wide` when in a badge context |
| Micro / caption, footnote, `(opcional)` marker (12px, 400) | `--text-micro` | `--text-body` at weight 400 with `text-muted` (so the caption still reads as secondary without leaning on a smaller size) |
| Table header cell (12px, 500, uppercase) | `--text-micro` | `--text-body` at weight 600, uppercase + `tracking-wide` |
| Button default label (14px, 500) | Body at 500 | `--text-body` at weight 600 (CTAs deserve the emphasis anyway) |
| Button small label (12px, 500) | `--text-micro` | `--text-body` at weight 600 (the button stays compact via 32px height + `px-3`, not via a smaller type size) |
| Breadcrumb "current" segment (14px, 500) | Body at 500 | `--text-body` at weight 600 |
| Form label (`font-medium`) | weight 500 | `font-semibold` (weight 600) |
| Display (32px, 700) | `--text-display` at 700 | `--text-display` at 600 — size + letter-spacing carry impact |

Wherever the rest of this document previously referred to "Micro 12/500",
"Micro 12/400", "Body 14/500" or "Display 32/700", the text now reads
"Body 14/600", "Body 14/400 text-muted", "Body 14/600" or "Display 32/600"
respectively. The `--text-micro` CSS variable is **not** declared in
`tokens.css` at all — it has been removed from the type scale.

Font stack (`--font-sans`, D-06):

```
-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif
```

Monospace (for timestamps, data tables, KPI digits, code cells — existing
`font-mono` utility already used in `_partials/mapa_pabellon.html`):

```
ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace
```

---

## Color

### 60 / 30 / 10 split (UIREDO-01)

| Role | Value | % | Usage |
|------|-------|---|-------|
| Dominant (60%) | `--color-surface-50` = `#f8fafc` | 60% | Page background (`body`), main content area |
| Secondary (30%) | `--color-surface-0` = `#ffffff` | 30% | Cards, drawer, top bar, modals, form fields |
| Accent (10%) | `--color-brand-600` = `#1a3a5c` | 10% | Primary CTA, active nav item, focus ring, links |
| Destructive | `--color-error-600` = `#dc2626` | — | Destructive actions and error states only |

Text sits on surface:

- `--color-text-body` = `#0f2236` (brand.800) on surface 0/50 → contrast 14.2:1 — AAA.
- `--color-text-muted` = `#64748b` (gray-500) on surface 50 → contrast 5.1:1 — AA.
- `--color-text-disabled` = `#94a3b8` (gray-400) on surface 50 → contrast 3.1:1 — AA for 18px+ only (applies to disabled buttons at Body 14px via opacity-60, see note below).
- `--color-border` = `#e2e8f0` (surface.200) — dividers, input borders, card borders.
- `--color-border-strong` = `#cbd5e1` (surface.300) — focus outlines on neutrals, table header bottom border.

Accent is reserved for, and only for:

1. Primary buttons (CTA) — `bg-brand-600` → `bg-brand-700` hover.
2. Active nav item in drawer — `bg-brand-600 text-white`.
3. Focus ring on any interactive element — `ring-2 ring-brand-500 ring-offset-2`.
4. Text links within paragraphs — `text-brand-600 underline-offset-2 hover:text-brand-700 hover:underline`.
5. Selected state in controls (radio, checkbox, active tab).

Accent is **not** used for: card borders, hover-only backgrounds, decorative
rules, plain headings, info toast chrome, tooltip arrows, badge pills that
aren't calls to action.

### Raw palette (layer 1, `tokens.css`)

Preserve the current brand + surface scales (already in `base.html:18-19`).
Add the semantic-state rows and fill in surface 400–900.

| Group | Steps | Notes |
|-------|-------|-------|
| `brand` | 50 `#eef5ff`, 100 `#d9e8ff`, 200 `#bcd5ff`, 300 `#8ebbff`, 400 `#5995ff`, 500 `#336dff`, **600 `#1a3a5c`**, **700 `#142d47`**, 800 `#0f2236`, 900 `#0a1724` | Kept verbatim. 600 is accent, 700 is hover, 800 is body text. |
| `surface` | 0 `#ffffff` (new alias), 50 `#f8fafc`, 100 `#f1f5f9`, 200 `#e2e8f0`, 300 `#cbd5e1`, **400 `#94a3b8`**, **500 `#64748b`**, **600 `#475569`**, **700 `#334155`**, **800 `#1e293b`**, **900 `#0f172a`** | 400–900 new. 500 is muted text. |
| `success` | 50 `#ecfdf5`, 100 `#d1fae5`, 500 `#10b981`, 600 `#059669`, 700 `#047857` | Toast success, status "produciendo", validation success marker. |
| `warn` | 50 `#fffbeb`, 100 `#fef3c7`, 500 `#f59e0b`, 600 `#d97706`, 700 `#b45309` | Preflight amber, status "parada", warn toast. |
| `error` | 50 `#fef2f2`, 100 `#fee2e2`, 500 `#ef4444`, 600 `#dc2626`, 700 `#b91c1c` | Destructive buttons, 5xx error state, validation errors, alarm toast. |
| `info` | 50 `#eff6ff`, 100 `#dbeafe`, 500 `#3b82f6`, 600 `#2563eb`, 700 `#1d4ed8` | Info toast (non-destructive notifications distinct from brand). |

Color-blind safety: state colors use green (success), amber (warn), red
(error), blue (info) in chroma-distinct hues. Every semantic colour is
reinforced by an icon in toasts and inline validation messages — colour is
never the sole signal.

### Semantic tokens (layer 2, `tokens.css`)

Every consumer reads **only** from this list. Tailwind `theme.extend` maps
utilities (`bg-primary`, `text-muted`, `border-subtle`) to these vars using
the `rgb(var(--token) / <alpha-value>)` pattern so alpha utilities
(`bg-primary/20`) keep working.

```css
:root {
  /* Base surfaces */
  --color-surface-base:    255 255 255;  /* cards, modals, drawer */
  --color-surface-app:     248 250 252;  /* page background */
  --color-surface-subtle:  241 245 249;  /* hover on rows, sidebar rail */
  --color-surface-muted:   226 232 240;  /* dividers bg, tag backgrounds */

  /* Text */
  --color-text-body:       15 34 54;     /* brand.800 */
  --color-text-heading:    10 23 36;     /* brand.900 */
  --color-text-muted:      100 116 139;  /* surface.500 */
  --color-text-disabled:   148 163 184;  /* surface.400 */
  --color-text-on-accent:  255 255 255;

  /* Border */
  --color-border-subtle:   226 232 240;
  --color-border-strong:   203 213 225;

  /* Accent (brand) */
  --color-primary:         26 58 92;     /* brand.600 */
  --color-primary-hover:   20 45 71;     /* brand.700 */
  --color-primary-active:  15 34 54;     /* brand.800 */
  --color-primary-subtle:  238 245 255;  /* brand.50 — soft chip bg */
  --color-focus-ring:      51 109 255;   /* brand.500 */

  /* Semantic states */
  --color-success:         5 150 105;    /* success.600 */
  --color-success-subtle:  236 253 245;  /* success.50 */
  --color-warn:            217 119 6;    /* warn.600 */
  --color-warn-subtle:     255 251 235;  /* warn.50 */
  --color-error:           220 38 38;    /* error.600 */
  --color-error-subtle:    254 242 242;  /* error.50 */
  --color-info:            37 99 235;    /* info.600 */
  --color-info-subtle:     239 246 255;  /* info.50 */

  /* Elevation (shadow) */
  --shadow-card:     0 1px 2px 0 rgb(15 34 54 / 0.04),
                     0 1px 3px 0 rgb(15 34 54 / 0.06);
  --shadow-popover:  0 4px 6px -1px rgb(15 34 54 / 0.08),
                     0 2px 4px -2px rgb(15 34 54 / 0.06);
  --shadow-modal:    0 20px 25px -5px rgb(15 34 54 / 0.12),
                     0 8px 10px -6px rgb(15 34 54 / 0.08);
  --shadow-drawer:   4px 0 24px -4px rgb(15 34 54 / 0.12);

  /* Radius */
  --radius-sm: 6px;   /* pills, badges, inline inputs */
  --radius-md: 10px;  /* buttons, form inputs */
  --radius-lg: 16px;  /* cards, modals */
  --radius-xl: 24px;  /* landing hero surfaces */
  --radius-pill: 9999px;

  /* Motion (D-05) */
  --duration-fast: 150ms;    /* hover, focus, tooltip */
  --duration-base: 200ms;    /* drawer open/close, toast enter, panel swap */
  --duration-slow: 300ms;    /* modal enter, large transitions — hard cap */
  --ease-standard:    cubic-bezier(0.2, 0, 0, 1);  /* entering: decelerate */
  --ease-emphasized:  cubic-bezier(0.3, 0, 0, 1);  /* large modals */
  --ease-accelerate:  cubic-bezier(0.3, 0, 1, 1);  /* exits */

  /* Z-index scale (Claude's Discretion → locked here) */
  --z-base:     0;
  --z-sticky:   10;   /* sticky table headers */
  --z-topbar:   30;   /* fixed top bar */
  --z-backdrop: 40;   /* drawer / modal backdrop */
  --z-drawer:   50;   /* drawer overlay */
  --z-modal:    60;   /* dialog / confirmation modal */
  --z-popover:  70;   /* dropdown, user menu */
  --z-toast:    90;   /* flash toast — above everything */
  --z-reserved: 100;  /* emergency escape hatch, must not be used in app code */
}
```

### Tailwind mapping (Plan 08-01 target)

| Utility namespace | Maps to | Rationale |
|-------------------|---------|-----------|
| `bg-surface`, `bg-surface-app`, `bg-surface-subtle`, `bg-surface-muted` | `--color-surface-*` | 60% dominant + 30% secondary split readable from markup |
| `text-body`, `text-muted`, `text-heading`, `text-disabled`, `text-on-accent` | `--color-text-*` | Replaces ad-hoc `text-gray-500` / `text-brand-800` across templates |
| `bg-primary`, `text-primary`, `border-primary`, `ring-primary`, `hover:bg-primary-hover` | `--color-primary*` | Accent utilities |
| `bg-success`, `text-success`, `bg-warn`, `text-warn`, `bg-error`, `text-error`, `bg-info`, `text-info` (and their `-subtle` siblings) | `--color-<state>*` | Semantic state classes — never used decoratively |
| `border-subtle`, `border-strong` | `--color-border-*` | Divider / input border |
| `shadow-card`, `shadow-popover`, `shadow-modal`, `shadow-drawer` | `--shadow-*` | Elevation |
| `rounded-sm/md/lg/xl/pill` | `--radius-*` | Override Tailwind defaults to the scale above |
| `transition-fast`, `transition-base`, `transition-slow` | duration + easing presets via `theme.extend.transitionDuration` and `transitionTimingFunction` | Keeps motion budget visible in markup |
| `z-topbar`, `z-backdrop`, `z-drawer`, `z-modal`, `z-popover`, `z-toast` | `--z-*` | Named z-index utilities |

Legacy scales `brand.*` and `surface.*` (numeric 50-900) remain available
for edge cases that still need an exact shade (e.g. `mapa_pabellon`
editor), but **new templates must consume only semantic utilities**. This
is the guardrail `gsd-ui-auditor` checks in Phase 9+.

---

## Copywriting Contract

### Primary CTAs (verb + noun, Spanish)

| Screen / context | CTA label |
|------------------|-----------|
| Login (`login.html`) | `Entrar` |
| Cambio de password (`cambiar_password.html`) | `Guardar contraseña` |
| Landing (`/bienvenida`) | `Ir a Centro de Mando` |
| Pipeline run | `Ejecutar análisis` |
| BBDD query | `Lanzar consulta` |
| Ajustes — nuevo usuario | `Crear usuario` |
| Ajustes — editar umbral | `Guardar cambios` |
| Ajustes — nueva solicitud aprobada | `Aprobar solicitud` |
| Centro de Mando — editor de zonas | `Guardar` / `Cancelar` (literales del partial, preserve verbatim — D-16) |
| Ciclos-calc — recalcular | `Calcular ciclos` |

Secondary action labels: `Cancelar`, `Cerrar`, `Volver`, `Descartar`,
`Reintentar`. Never `OK`, never `Aceptar` alone.

### Empty state copy (per family)

Empty state block = icon (Heroicons outline, 48px, `text-muted`) +
headline (Subtitle 16/600) + body (Body 14/400, `text-muted`) + CTA
(Primary button, gated by `can()` — D-28).

| Family | Icon | Headline | Body | CTA (conditional) |
|--------|------|----------|------|-------------------|
| Historial vacío | `archive-box` | `Sin ejecuciones todavía` | `Cuando lances un análisis aparecerá aquí con su informe PDF.` | `Ir a Análisis` → `/pipeline` (if `pipeline:run`) |
| Solicitudes vacías (`/mis-solicitudes`) | `paper-airplane` | `No tienes solicitudes pendientes` | `Las aprobaciones que pidas para consultas largas aparecerán aquí.` | (none) |
| Ajustes / solicitudes del propietario | `inbox` | `Bandeja limpia` | `No hay solicitudes pendientes de aprobar.` | (none) |
| BBDD sin resultado | `magnifying-glass` | `Sin resultados para esta consulta` | `Ajusta los filtros de fecha o recurso y vuelve a lanzar la consulta.` | `Cambiar filtros` (scrolls to filter form) |
| Informes sin PDF | `document` | `Aún no hay informes generados` | `Lanza un análisis desde la página de Análisis.` | `Ir a Análisis` → `/pipeline` (if `pipeline:run`) |
| Capacidad / operarios sin datos en rango | `calendar` | `Sin datos en el rango seleccionado` | `Prueba con un rango distinto o revisa la conexión con MES.` | (none) |
| Usuarios (`/ajustes/usuarios` — propietario) | `users` | `Aún no hay usuarios además de ti` | `Crea el primer usuario para empezar a delegar.` | `Crear usuario` |
| Alarmas turno (en `_partials/mapa_pabellon`) | LOCKED (partial owns its own empty state; preserve verbatim) | — | — | — |

### Error state copy (per scenario, D-29)

Error state block = icon (`exclamation-triangle`, 48px, `text-error`) +
headline (Subtitle 16/600) + body (Body 14/400) + **Reintentar** button
(secondary) + contextual runbook link. The five RUNBOOK scenarios
(`docs/RUNBOOK.md`) each get their own headline + link anchor.

| Trigger | Headline | Body | Action buttons | Runbook anchor |
|---------|----------|------|----------------|----------------|
| 5xx del servidor (genérico) | `Algo ha ido mal` | `El servidor ha devuelto un error inesperado. Vuelve a intentarlo en unos segundos.` | `Reintentar` · link `Consultar guía` | `#escenario-1-mes-caido` (if module consumes MES) else no link |
| MES caído (`/bbdd`, `/pipeline`, `/capacidad` timeout o 503) | `Sin conexión con MES` | `La base de datos de producción no responde. Puedes reintentar; si persiste, avisa al equipo.` | `Reintentar` · link `Ver guía de incidencia` | `docs/RUNBOOK.md#escenario-1-mes-caido` |
| Postgres caído (health 503) | `Servicio temporalmente no disponible` | `No podemos leer la base interna. Se ha avisado al administrador.` | `Reintentar` · link `Ver guía de incidencia` | `docs/RUNBOOK.md#escenario-2-postgres-no-arranca` |
| HTTPS / cert warning (no opera desde la UI, pero enlazamos desde panel de ajustes) | `Certificado próximo a caducar` | `El certificado HTTPS expira pronto. Renuévalo antes de la fecha indicada.` | link `Ver guía de renovación` | `docs/RUNBOOK.md#escenario-3-certificado-caddy-expira` |
| Pipeline atascado (timeout 900s, cola saturada) | `Análisis detenido` | `El análisis lleva demasiado tiempo ejecutándose. Puedes cancelarlo o esperar a que el sistema lo libere.` | `Reintentar más tarde` · link `Ver guía de incidencia` | `docs/RUNBOOK.md#escenario-4-pipeline-atascado` |
| Lockout propietario (pantalla de login tras 5 fallos) | `Acceso bloqueado` | `Has superado el número de intentos. Espera 15 minutos o contacta con el administrador.` | (no retry button, tiempo transcurrido solo) · link `Ver guía de incidencia` | `docs/RUNBOOK.md#escenario-5-lockout-del-unico-propietario` |

All runbook links open in the same tab (`target="_self"`) — this is an
internal doc, not a third-party reference. The anchor slug format is the
Markdown slugification of the scenario heading; authors verify each link
during Plan 08-0N by opening `docs/RUNBOOK.md#...` locally.

### Destructive actions (inventory for this phase + confirmation pattern)

Every destructive action opens a modal (not a native `confirm()`). Modal
copy explicitly names the entity being destroyed.

| Action | Trigger location | Modal title | Body | Confirm label | Permission gate |
|--------|------------------|-------------|------|---------------|-----------------|
| Borrar ejecución (histórico) | `historial.html` row | `Borrar ejecución` | `Se borrará la ejecución #{id} del {fecha} y sus PDFs asociados. Esta acción es irreversible.` | `Borrar` (red) | `informes:delete` |
| Borrar usuario | `ajustes_usuarios.html` | `Borrar usuario` | `Se desactivará {email} y se invalidarán sus sesiones. La fila queda en audit log.` | `Borrar` (red) | `usuarios:manage` (propietario) |
| Cancelar solicitud de aprobación | `mis_solicitudes.html` | `Cancelar solicitud` | `Se descartará la solicitud de {endpoint} enviada el {fecha}. Tendrás que volver a solicitarla si la necesitas.` | `Cancelar solicitud` (red) | owner of the request |
| Rechazar aprobación | `ajustes_solicitudes.html` | `Rechazar solicitud` | `Se notificará al usuario y la solicitud quedará marcada como rechazada en audit log.` | `Rechazar` (red) | `aprobaciones:manage` (propietario) |
| Borrar zona del pabellón | `_partials/mapa_pabellon.html` (editor) | (LOCKED — partial uses inline red button; preserve behaviour) | — | — | — |

---

## Global Chrome (UIREDO-02, D-07..D-10)

### Top bar

Persistent, sits at the very top of every authenticated page (not on
`/login` nor `/cambiar-password` — those have their own shell, see per-
screen adaptations).

| Property | Value |
|----------|-------|
| Height | `--space-topbar-h` (56px desktop, 52px mobile) |
| Background | `bg-surface-base` (white) |
| Border-bottom | `1px solid var(--color-border-subtle)` |
| Shadow | none (border provides separation) |
| Padding | `px-6 py-3` desktop / `px-4 py-3` mobile |
| Z-index | `--z-topbar` (30) |
| Sticky | `position: sticky; top: 0;` |
| Slots (left→right) | 1. Hamburger (drawer toggle). 2. `{{ page_title }}` (Heading 20/600). 3. Right cluster: conn-badge (HTMX 30s poll) · user email (Body 14/400, muted, `hidden sm:inline`) · role pill (`bg-primary-subtle text-primary`, Body 14/600 uppercase) · user menu trigger (avatar / initial circle, opens popover). |
| User menu popover | Align right. Contains: `{{ user.email }}`, divider, `Cambiar contraseña` (→ `/cambiar-password`), `Cerrar sesión` (POST `/logout`). `shadow-popover`, `rounded-lg`, `z-popover`. |
| Responsive | Below `md` (768px): hide user email text; keep role pill + avatar. Page title truncates with `text-overflow: ellipsis`. |

The brand (Nexo logo) lives **inside the drawer**, not on the top bar —
the top bar is functional, not identitary (D-10).

### Drawer

Hamburger-hidden on desktop by default (D-07). Mobile is always drawer
(overlay). Persist via `localStorage` key `nexo.ui.drawerOpen` (D-08).

| Property | Desktop (≥md) | Mobile (<md) |
|----------|---------------|--------------|
| Trigger | Hamburger button in top bar left slot | Same |
| Width | `--space-drawer-w` (280px) | 280px (capped at 85vw for small devices) |
| Position | Slides in from `left: 0`, pushes content right when open (or overlays — see variant below) | Always overlay |
| Behaviour on open | Default = **overlay** over content with backdrop. Rationale: users toggle drawer to navigate, close to focus on content; don't shift layout. | Overlay with backdrop |
| Background | `bg-surface-base` (white) | Same |
| Shadow | `shadow-drawer` | Same |
| Backdrop | `bg-surface-900/40` with `backdrop-blur-sm`, `z-backdrop` | Same |
| Enter animation | `transform: translateX(-100%) → 0` over `--duration-base` (200ms), `--ease-standard` | Same |
| Exit animation | Reverse, `--ease-accelerate`, `--duration-base` | Same |
| `prefers-reduced-motion` | Animation reduced to 0s; opacity snap instead | Same |
| Close triggers | Click backdrop · Esc · re-click hamburger · click any nav link (mobile only — desktop keeps it open) | Same except always close on link click |
| Focus trap | Yes (see keyboard spec below). First focusable = close button inside drawer, last = active nav link | Same |
| Scroll lock | When drawer is open on mobile, `document.body` scroll locks. Desktop keeps body scroll. | Scroll lock applies |
| z-index | `--z-drawer` (50) | Same |
| Persistence | Alpine reads/writes `localStorage.getItem('nexo.ui.drawerOpen')` on init/mutate. Ignored on mobile (always closed by default). | Ignored on mobile |

### Drawer structure

```
[Close icon button · top-right, 40x40 target, aria-label="Cerrar menú"]
[Logo Nexo (NEXO_LOGO_PATH) + "by ECS MOBILITY" line, centered, padding-top 24px]
[Divider: border-subtle]
[Nav section — primary modules]
  · Centro Mando       /             centro_mando:read
  · Análisis           /pipeline     pipeline:read
  · Historial          /historial    historial:read
  · Capacidad          /capacidad    capacidad:read
  · BBDD               /bbdd         bbdd:read
[Divider]
[Nav section — operaciones]
  · Recursos           /recursos     recursos:read
  · Calcular ciclos    /ciclos-calc  ciclos:read
  · Operarios          /operarios    operarios:read
  · Datos              /datos        datos:read
[Divider]
[Nav section — administración — only if can(ajustes:manage)]
  · Ajustes            /ajustes      ajustes:manage
  · Solicitudes (+ HTMX badge)  /ajustes/solicitudes   aprobaciones:manage
[Spacer · flex-1]
[Footer: ECS logo + "© ECS Mobility · Nexo · v1.0.0" · Body 14/400 text-muted]
```

Each nav item is a Jinja `{% if can(current_user, permission) or permission is none %}` wrapper — Phase 5 D-01 preserved verbatim. Active state:
`bg-primary-subtle text-primary font-semibold` (rest: `text-body hover:bg-surface-subtle`). Hover transition: `transition-fast`.

Icon slot: 20x20 Heroicons outline, stroke-width 1.5, `text-muted` (inactive) / `text-primary` (active).

The "Análisis" rename: current sidebar label is "Analisis" (no accent).
The redesign uses "Análisis" with the acute accent — Spanish-correct.
This is a content change, not a route change.

### Keyboard navigation (UIREDO-08)

| Key | Action |
|-----|--------|
| Tab / Shift+Tab | Normal focus order. When drawer is open, focus is trapped inside drawer (first focusable = close button, last = footer link). |
| Esc | Closes any open overlay in z-order priority: toast → popover → modal → drawer. Returns focus to the trigger element. |
| Enter / Space | Activates focused interactive element. |
| Arrow Up / Down | Navigates within a popover menu (user menu, select dropdowns). |
| `[` (left square bracket) | **Keyboard shortcut for drawer toggle** (D-09 resolution). Rationale: no modifier (fast), unused by browser, mnemonic for "panel on left". `Ctrl+[` is Back in Chrome → we use bare `[` and register on `keydown` with `e.target.tagName` guard to avoid intercepting typing in inputs/textareas. |

Alternative shortcuts evaluated and rejected:

- `\` (backslash): conflicts with user muscle memory for escape sequences.
- `gm` (two-key Gmail-style): requires a key-sequence handler in Alpine; overkill.
- `Ctrl+\`: acceptable fallback but collides with some IDE-style apps the operator runs locally.

Plan 08-02 implements the `[` listener in `static/js/app.js` with a 50ms
debounce and the input-guard described. If the operator rejects `[`
during plan review, the fallback is `Alt+S` (`S` for sidebar) — leave
this as an open question, see Open Questions.

### Flash toast (Phase 5 D-08 contract preserved, restyled — D-30)

| Property | Value |
|----------|-------|
| Position | Fixed, top-right (`top: calc(var(--space-topbar-h) + 16px); right: 16px;`) |
| Max width | 400px (mobile: `calc(100vw - 32px)`) |
| Stack | Vertical queue, 8px gap between toasts |
| Enter animation | slide-in from right, 16px, + fade, `--duration-base`, `--ease-standard` |
| Exit animation | slide-out to right + fade, `--duration-fast`, `--ease-accelerate` |
| Auto-dismiss | 4000ms. Hover pauses the timer (resume on mouseleave). |
| Focus | Focusable (tabindex=0 on the toast container) so screen readers announce it. Role `status` for info/success, `alert` for warn/error. |
| `prefers-reduced-motion` | Swap slide for opacity fade only; same duration. |
| Variants | `info`, `success`, `warn`, `error` (4 only). Legacy `producing/stopped/incidence/alarm/turno` map to these: producing→success, stopped→warn, incidence/alarm→error, turno→info. The mapping lives in `static/js/app.js`. |
| Backward compatibility | `window.showToast(type, title, msg)` signature **preserved** — the flash middleware, Phase 4 preflight modal and the pabellón telemetry all call it. Only the DOM + styles change. |

Variant styles:

| Variant | Background | Border | Icon color | Title color | Body color | Icon |
|---------|-----------|--------|------------|-------------|------------|------|
| info | `bg-surface-base` | `border-subtle` with `border-l-4 border-l-info` | `text-info` | `text-heading` | `text-body` | `information-circle` |
| success | same | `border-l-4 border-l-success` | `text-success` | `text-heading` | `text-body` | `check-circle` |
| warn | same | `border-l-4 border-l-warn` | `text-warn` | `text-heading` | `text-body` | `exclamation-triangle` |
| error | same | `border-l-4 border-l-error` | `text-error` | `text-heading` | `text-body` | `x-circle` |

All toasts use the same white surface (never saturated fills like the
current `bg-red-600 text-white` ones). The left accent bar + icon carry
the semantic weight. Close button (16x16 `x-mark`, top-right of the
toast) dismisses manually.

---

## Landing Screen (`/bienvenida`, UIREDO-02 extension via D-23)

New route, new template `templates/bienvenida.html`. Post-login redirect
target instead of `/` (Plan 08-03 updates `api/routers/auth.py`).

### Layout

```
┌──────────────────────── Top bar (shared) ─────────────────────┐
│ [≡]  Bienvenida                       conn-badge · user menu  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│         [Saludo, {nombre}]            ← Display 32/600       │
│         [Es {día de la semana en castellano}, {fecha}]        │
│                                      ← Body 14/400 text-muted│
│         [HH:MM:SS, 24h]              ← Display 32/600 mono   │
│                                                              │
│         ┌──────────────────────────┐                          │
│         │  Ir a Centro de Mando → │  ← Primary button, large │
│         └──────────────────────────┘                          │
│                                                              │
│         (Drawer accesible vía ≡ en top bar para otras rutas) │
└──────────────────────────────────────────────────────────────┘
```

### Saludo rules (hour bands, D-23)

| Band | Saludo |
|------|--------|
| 06:00 – 11:59 | `Buenos días, {nombre}` |
| 12:00 – 20:59 | `Buenas tardes, {nombre}` |
| 21:00 – 05:59 | `Buenas noches, {nombre}` |

Server-side rendering: Jinja resolves `hora_saludo(now)` (template
filter added in Plan 08-03) based on server local time
(`Europe/Madrid`). Client-side doesn't re-compute — the server render
is authoritative. If the client's clock drifts past a band, the next
navigation corrects it.

`{nombre}` resolution: `current_user.nombre` if the user record has a
first-name field populated; fall back to the local-part of the email
(`e.eguskiza@ecsmobility.com` → `Erik`). If neither is feasible, use
empty → `Buenos días` (no trailing comma).

### Reloj en tiempo real

- Format: `HH:MM:SS` (24-hour, zero-padded). Example: `14:07:32`.
- Font: `--font-mono`, Display 32/600, `tabular-nums` (CSS
  `font-variant-numeric: tabular-nums`) so digits don't jitter.
- Tick via Alpine `setInterval(1000)` in the `bienvenidaPage()` component.
- Cleared in the Alpine `destroy()` hook to avoid leaks on SPA-style
  navigations (HTMX partial loads).
- Uses `Date.now()` client-side (no network calls); day-of-week and
  date come from the initial Jinja render.

### Primary CTA

`Ir a Centro de Mando` — primary button, size `lg`, chevron-right icon
trailing. Tabs to it after the page loads (autofocus). Click navigates to
`/` via plain anchor (no `history.replaceState` needed — standard nav).

### Deferred (Mark-IV, D-23)

Widgets configurables per-user: NOT in this spec. The landing in Mark-III
is static. Leave a `{# TODO Mark-IV: widgets configurables #}` comment in
`bienvenida.html` so the next phase finds it.

### Responsive

- Desktop ≥lg: center the greeting block in a max-width 640px column,
  vertically centered in viewport.
- Mobile: same layout, padding reduced to 16px, Display typography
  scales to 28px (override via `@media (max-width: 640px)`).

---

## Component Inventory

All components live in `static/css/app.css` as utility-class compositions
(using the existing `@apply` pattern) or inline in templates via
Tailwind utilities. Naming stays backward-compatible: the current
`.card`, `.btn`, `.btn-primary`, `.input-inline`, `.data-table`,
`.spinner`, `.stat-card` classes continue to work, their styles are
rewritten to consume semantic tokens.

### Buttons

| Variant | Background | Text | Border | Hover | Focus ring | Disabled | Loading |
|---------|-----------|------|--------|-------|------------|----------|---------|
| Primary (`.btn-primary`) | `bg-primary` | `text-on-accent` | none | `bg-primary-hover` | `ring-2 ring-primary ring-offset-2` | `opacity-60 cursor-not-allowed` | Same visual + inline spinner (12px) + label stays (don't hide it) |
| Secondary (`.btn-secondary`, new) | `bg-surface-base` | `text-body` | `border border-strong` | `bg-surface-subtle` | same | same | same |
| Ghost (`.btn-ghost`) | `bg-transparent` | `text-muted` | none | `bg-surface-subtle text-body` | same | same | same |
| Danger (`.btn-danger`) | `bg-error` | `text-on-accent` | none | `bg-error/90` (alpha) | `ring-2 ring-error ring-offset-2` | same | same |
| Link (`.btn-link`, new) | `bg-transparent` | `text-primary underline-offset-2` | none | `text-primary-hover underline` | `ring-2 ring-primary ring-offset-2 rounded-sm` | same | — |

Sizes: `.btn` (40px height, `px-4`, Body 14/600) default. `.btn-sm`
(32px, `px-3`, Body 14/600). `.btn-lg` new (48px, `px-6`, Subtitle
16/600) used for landing CTA and primary confirmation modal actions.

Icon-only buttons use `.btn .btn-icon` (40x40 square, padding 0, icon
centered). Must have `aria-label`. Rounded `--radius-md`.

### Cards

| Variant | Background | Border | Shadow | Radius | Padding |
|---------|-----------|--------|--------|--------|---------|
| Flat (`.card`) | `bg-surface-base` | `border border-subtle` | none | `--radius-lg` (16px) | `--space-card-padding` (24px) |
| Elevated (`.card-elevated`, new) | `bg-surface-base` | none | `shadow-card` | `--radius-lg` | 24px |
| Accent (`.card-accent`, new) | `bg-primary-subtle` | `border border-primary/20` | none | `--radius-lg` | 24px |

Card header (`.card-header`) remains but restyled: `border-b
border-subtle`, `px-6 py-4`, Subtitle 16/600 text-heading. Card body
(`.card-body`) = `px-6 py-5`.

### Form inputs

| Property | Value |
|----------|-------|
| Height | 40px |
| Padding | `px-3 py-2` |
| Background | `bg-surface-base` |
| Border | `1px solid var(--color-border-strong)` |
| Radius | `--radius-md` (10px) |
| Typography | Body 14/400 |
| Hover border | `border-primary/50` |
| Focus | `border-primary`, `ring-2 ring-primary/30`, `outline: none` |
| Invalid | `border-error`, `ring-2 ring-error/30` on focus |
| Disabled | `bg-surface-subtle text-disabled cursor-not-allowed` |
| Placeholder | `text-disabled` |

Label pattern (D-24 — top-aligned / stacked):

```
<label class="block text-sm font-semibold text-body mb-1">
  Email <span class="text-muted font-normal">(opcional)</span>
</label>
<input ...>
<p class="mt-1 text-sm text-error" role="alert">{{ errors.email }}</p>
```

`(opcional)` marker (D-26) is Body 14/400 in `text-muted`, placed
inline after the label. **Required is default** — do not mark required
fields with asterisks.

Validation message slot: `<p class="mt-1 text-sm text-error"
role="alert">` — always present in DOM (empty when no error, populated
by Jinja on re-render after server-side validation, D-25). Screen
readers announce it via `role="alert"`.

Select: same chrome as text input, chevron-down icon on the right
(16px, `text-muted`), native `<select>` (no custom dropdown in Mark-III).

Checkbox / Radio: 16x16 box, `border-strong`, `rounded-sm` (checkbox)
or `rounded-pill` (radio). Checked fill `bg-primary` with
`text-on-accent` check icon (14x14). Focus ring same as inputs.

Textarea: same as text input, `min-height: 96px`, resize-y only.

### Tables (`.data-table`)

| Region | Style |
|--------|-------|
| Container | `.card` wrapper, overflow-x auto on mobile |
| Header cell (`th`) | `bg-surface-subtle`, `text-muted`, Body 14/600 uppercase, `tracking-wide`, `border-b border-subtle`, `px-4 py-3`, sticky when inside a scroll container (`sticky top-0 z-sticky`) |
| Row (`tr`) | `border-b border-subtle` |
| Last row | no border-b |
| Cell (`td`) | `px-4 py-3`, Body 14/400, `text-body` |
| Row hover | `bg-surface-subtle` |
| Row interactive (clickable) | cursor-pointer, hover reveals chevron-right (`text-muted`) in the rightmost cell |
| Zebra | NOT used (hover + border is enough) |
| Numeric column | `text-right`, `font-mono`, `tabular-nums` |
| Warning row (`.row-warning`) | `bg-warn-subtle` — reserved for zero-cycle diagnostic rows |
| Destructive row action | `.btn-icon` with `x-mark` icon, `text-muted hover:text-error` |

### Modals / Dialogs

| Property | Value |
|----------|-------|
| Backdrop | `bg-surface-900/40 backdrop-blur-sm`, `z-backdrop`, click-to-close |
| Container | `bg-surface-base`, `shadow-modal`, `rounded-lg` (16px), max-width varies (sm:480, md:640, lg:800) |
| Enter animation | opacity 0 → 1 + scale 0.97 → 1, `--duration-slow` (300ms, hard cap per UIREDO-05), `--ease-emphasized` |
| Exit | reverse, `--duration-base` (200ms), `--ease-accelerate` |
| `prefers-reduced-motion` | opacity only, 200ms |
| z-index | `--z-modal` (60) |
| Header | `px-6 py-4`, Subtitle 16/600, close button (`x-mark` 24x24) top-right, `border-b border-subtle` |
| Body | `px-6 py-5`, Body 14/400 |
| Footer | `px-6 py-4`, `border-t border-subtle`, action buttons right-aligned, secondary left of primary, `gap-3` |
| Scroll | Body scrolls internally when content exceeds `90vh`; `max-height: calc(90vh - footer - header)` |
| Focus | Focus trap active, first focusable element focused on open, focus returns to trigger on close |
| ESC | Closes unless action is in-flight (loading state) — then ESC is ignored |

Confirmation modal (specialization for destructive actions): headline
= action + entity ("Borrar usuario"), body = consequences in Body 14,
destructive action on the right (`.btn-danger`), `Cancelar` left
(`.btn-secondary`). Destructive button is never auto-focused.

### Badges / Pills

| Variant | Background | Text | Radius | Padding | Typography |
|---------|-----------|------|--------|---------|------------|
| Neutral | `bg-surface-muted` | `text-muted` | `--radius-pill` | `px-2 py-0.5` | Body 14/600 uppercase |
| Brand | `bg-primary-subtle` | `text-primary` | `--radius-pill` | `px-2 py-0.5` | Body 14/600 uppercase |
| Success | `bg-success-subtle` | `text-success` | same | same | same |
| Warn | `bg-warn-subtle` | `text-warn` | same | same | same |
| Error | `bg-error-subtle` | `text-error` | same | same | same |

Role pill in top bar uses **brand** variant. Preflight amber/red modals
map to warn/error pill in the modal header.

### Breadcrumbs (only on ajustes sub-pages)

Used inside `ajustes_*.html` to show hub → sub-page path. Layout:

```
Ajustes  /  Usuarios
 └ link muted         └ text-body (current)
```

- Separator: `/` with `px-2` padding, `text-disabled`.
- Link color: `text-muted hover:text-body`.
- Current: `text-body font-semibold`, no link.
- Typography: Body 14/400, placed above the page heading, `mb-4`.

### Skeleton / loading placeholders — NOT USED

Per D-27, loading pattern is a centered spinner, not skeleton shimmer.
See State Contracts below.

---

## State Contracts

### Loading

Two tiers:

#### Tier 1 — Panel-level (D-27, default)

Short operations (<1s) and HTMX panel refreshes use a single
centred spinner inside the panel card.

```
.spinner-panel {
  display: flex; align-items: center; justify-content: center;
  min-height: 240px;
}
.spinner-panel .spinner {
  width: 32px; height: 32px;
  border: 3px solid var(--color-surface-muted);
  border-top-color: var(--color-primary);
  border-radius: 50%;
  animation: spin var(--duration-slow) linear infinite;
}
@media (prefers-reduced-motion: reduce) {
  .spinner-panel .spinner { animation-duration: 1.2s; }
}
```

No label by default. If the load exceeds 3s, reveal a muted
`Cargando…` label below (server sends via SSE or HTMX progress hook).

#### Tier 2 — Full-page / long-running (DISCOVERED-TOKEN: `gif-corona.gif`)

Full-page or long-running loads (initial auth check, pipeline
execution >2s, PDF regeneration, cold boot of the drawer state)
use the brand-sanctioned crown GIF discovered in `static/img/`.

Layout:

```
<div class="loading-hero">
  <img src="/static/img/gif-corona.gif"
       alt="Cargando"
       class="w-40 h-40 object-contain"
       style="image-rendering: auto;">
  <p class="mt-4 text-subtitle text-heading">{{ tagline }}</p>
  <p class="mt-1 text-body text-muted">{{ sub_tagline }}</p>
</div>
```

Taglines (rotate deterministically by context):

| Context | Tagline | Sub-tagline |
|---------|---------|-------------|
| Pipeline ejecutando | `Procesando análisis` | `Esto puede tardar unos minutos` |
| Informe generándose | `Preparando informe` | `Compilando PDFs` |
| Post-login bootstrap | `Cargando Nexo` | — |
| Recalibrando factor | `Calculando factor de aprendizaje` | `Leyendo histórico de ejecuciones` |

Centred horizontally + vertically in the available content area.
The GIF plays automatically (no controls). Respects
`prefers-reduced-motion` by swapping for a static first frame
(`<img>` with `onplaying="this.pause()"` heuristic is insufficient
— Plan 08-01 extracts the first frame to `gif-corona.png` and
serves it via `<picture>` with `prefers-reduced-motion: reduce`
media query).

**Selection rule:** The default is Tier 1 (panel spinner). Tier 2
activates when either (a) the operation is whole-screen by nature
(auth bootstrap, pipeline run full-page shell), or (b) the panel
spinner has been visible for >2000ms without receiving any
progress event.

### Empty state

Follows the empty state family table in Copywriting Contract. Layout:

```
<div class="empty-state">
  <svg class="w-12 h-12 text-muted" ...heroicon.../>
  <h3 class="mt-4 text-subtitle text-heading">{{ headline }}</h3>
  <p class="mt-1 text-body text-muted max-w-md">{{ body }}</p>
  {% if cta and can(current_user, cta.permission) %}
    <a href="{{ cta.href }}" class="btn btn-primary mt-6">{{ cta.label }}</a>
  {% endif %}
</div>
```

Padding: `py-16`, centered. Icon `text-muted`. CTA visibility gated by
`can()` so empty states respect Phase 5 RBAC.

### Error state (D-29)

Follows the error state table in Copywriting Contract. Layout:

```
<div class="error-state card card-elevated">
  <svg class="w-12 h-12 text-error" ...exclamation-triangle.../>
  <h3 class="mt-4 text-subtitle text-heading">{{ headline }}</h3>
  <p class="mt-1 text-body text-body max-w-md">{{ body }}</p>
  <div class="mt-6 flex gap-3">
    <button class="btn btn-secondary" @click="retry()">Reintentar</button>
    <a href="{{ runbook_anchor }}" class="btn btn-link">Ver guía de incidencia</a>
  </div>
</div>
```

Lives **inside** the panel where the failed operation was. Never
full-page (the top bar + drawer must remain usable so the user can
navigate elsewhere). Exception: if the failure is the auth check
itself, the login page handles it.

---

## Accessibility (UIREDO-08)

### Contrast (WCAG 2.1 AA)

| Pair | Ratio | Requirement | Status |
|------|-------|-------------|--------|
| text-body on surface-app (`#0f2236` on `#f8fafc`) | 14.2:1 | 4.5:1 (body) | PASS (AAA) |
| text-body on surface-base (`#0f2236` on `#ffffff`) | 14.7:1 | 4.5:1 | PASS (AAA) |
| text-muted on surface-app (`#64748b` on `#f8fafc`) | 5.1:1 | 4.5:1 | PASS (AA) |
| text-disabled on surface-base (`#94a3b8` on `#ffffff`) | 3.1:1 | 3.0:1 large only | PASS (large text / disabled state) |
| on-accent on primary (`#ffffff` on `#1a3a5c`) | 10.3:1 | 4.5:1 | PASS |
| primary on primary-subtle (`#1a3a5c` on `#eef5ff`) | 11.4:1 | 4.5:1 | PASS |
| success on surface-base (`#059669` on `#ffffff`) | 4.7:1 | 4.5:1 | PASS (tight — do not lighten) |
| warn on surface-base (`#d97706` on `#ffffff`) | 4.5:1 | 4.5:1 | PASS (tight — do not lighten) |
| error on surface-base (`#dc2626` on `#ffffff`) | 4.9:1 | 4.5:1 | PASS |
| focus-ring on surface-app (`#336dff` on `#f8fafc`) | 4.0:1 | 3.0:1 (non-text UI) | PASS |

`pa11y-ci` (D-21) runs these checks automatically in CI. Any future
token adjustment must re-run the matrix.

### Focus ring

All interactive elements receive the same focus treatment:

```css
:focus-visible {
  outline: 2px solid var(--color-focus-ring);
  outline-offset: 2px;
  border-radius: inherit;
}
```

Fallback (non-`:focus-visible` browsers): plain `:focus`. Never
`outline: none` without replacement.

### Keyboard navigation map (summary)

See Global Chrome § Keyboard navigation for the detailed table. Rules:

- Tab order follows DOM order; skip links not required (no page has
  >10 items above main content).
- Every icon-only button has `aria-label` in Spanish.
- Every form input has an associated `<label for="...">`.
- Every SVG used as information icon has `role="img" aria-label="..."`.
  SVGs used purely decoratively have `aria-hidden="true"`.
- Dialog `<div role="dialog" aria-modal="true" aria-labelledby="...">`.
- Toast container `<div role="region" aria-live="polite">` for info,
  `aria-live="assertive"` for error/warn.

### Reduced motion

`@media (prefers-reduced-motion: reduce)` rules in `tokens.css`:

```css
@media (prefers-reduced-motion: reduce) {
  * { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; }
  .spinner, .loading-hero img { animation-duration: 1.5s !important; }
  /* Spinners + loading GIF continue animating — they signal progress. */
}
```

Drawer, modal, toast all swap slide/scale for opacity-only transitions
at the same duration.

---

## Motion

Duration budget (hard-capped by UIREDO-05 and D-05):

| Interaction class | Duration | Easing | Example |
|-------------------|----------|--------|---------|
| Micro (hover, focus glow) | `--duration-fast` (150ms) | `--ease-standard` | Button hover background, link underline in |
| Drawer open/close | `--duration-base` (200ms, hard cap from UIREDO-02) | `--ease-standard` in, `--ease-accelerate` out | Drawer slide, backdrop fade |
| Panel swap (HTMX) | `--duration-base` | `--ease-standard` | HTMX content swap fades over 200ms |
| Toast enter | `--duration-base` | `--ease-standard` | Slide + fade |
| Toast exit | `--duration-fast` | `--ease-accelerate` | Slide out fast |
| Modal enter | `--duration-slow` (300ms, hard cap) | `--ease-emphasized` | Scale + fade |
| Modal exit | `--duration-base` | `--ease-accelerate` | Reverse |

(The "Micro" label in the first row is a motion-class name — duration of
short micro-interactions — and is unrelated to the dropped typography
role of the same name.)

Animations that DO NOT exist in this system: parallax, Ken Burns,
bounce, spring overshoot, continuous loops (except the spinner and the
`gif-corona.gif` loading hero). Any contributor proposing an animation
outside this list must justify it in the plan.

---

## Per-Screen Adaptations

Per UIREDO-04, each template ships its own sketch+implementation plan
cycle (Plans 08-05 onwards). This spec provides the **starting point**
for the sketch brief: which system components compose the screen, and
which concerns each screen has that are not covered by the global
chrome.

Legend: `G` = uses global chrome · `P` = this screen has permission
gating worth reiterating · `L` = LOCKED interaction per D-16 · `F` =
new in this phase

| Screen (template) | Plan order | Scope of redesign | Notable concerns |
|-------------------|------------|-------------------|------------------|
| `login.html` | 08-02b (alongside chrome — high-impact first) | Full redesign: centered card on `surface-app` background, brand-subtle (no `bg-brand-800` saturation). Logo at top, card with email + password inputs, primary CTA. Error banner uses error-subtle card variant. Lockout message uses Error state contract copy. | Public route, does NOT have top bar or drawer. Autofocus on email. `autocomplete="username"` + `autocomplete="current-password"`. |
| `bienvenida.html` (F) | 08-03 | New screen. Spec section "Landing Screen" is the source of truth. | Server-rendered greeting; client-side clock tick. |
| `index.html` / `centro_mando.html` (L) | 08-04 | Chrome only — tokens, typography, spacing, card wrapping. Plano + máquinas editor = untouched. `_partials/mapa_pabellon.html` consumes the new tokens via `app.css` but its Alpine component + HTML structure is preserved verbatim. | Zone editor CSS classes (`bg-violet-*`, `bg-emerald-*`) are LEGACY-ALLOWED during Mark-III. Replacement by tokens is Mark-IV work. |
| `pipeline.html` | 08-05 | Filter form (stacked labels, D-24), Run button (primary-lg), log console (preserve `.log-console` dark), PDF list (cards grid), OEE dashboard panel. Preflight amber/red modal restyles to new Modal spec. | Server-Sent Events streaming into log console. `preflightModal` Alpine component (in `app.js`) preserves API; only the modal chrome re-skins. Loading Tier 2 (`gif-corona.gif`) appears while pipeline runs. |
| `historial.html` | 08-06 | Table-first layout. Filters sidebar becomes collapsible top bar of filters above the table. Row actions (view PDFs, delete) gated by `informes:delete`. Empty state per family table. | Destructive delete → modal confirmation. Click row → expand details inline (not navigate). |
| `bbdd.html` | 08-07 | Query form top (stacked fields), run button triggers preflight, result table below (or empty state). Query history in a sidebar card. | Preflight amber/red integration identical to pipeline.html. Pre-existing modals restyle to new Modal spec. |
| `capacidad.html` | 08-08 | Filters (date range, section), KPI cards row, data table. | Preflight amber/red for ranges > 90 días. |
| `operarios.html` | 08-09 | Filters, table, export CSV button gated by `operarios:export`. | Preflight amber/red for ranges > 90 días. |
| `recursos.html` | 08-10 | CRUD list: table with inline edit, "Nuevo recurso" button gated by `recursos:edit`. Sync button gated by `recursos:edit`. | Each edit opens a dialog (Modal) rather than inline-editing to reduce accidental edits. |
| `ciclos.html` | 08-11 | Same shape as recursos. | — |
| `ciclos_calc.html` | 08-12 | Form + computed table output. Heavy user — redesign focuses on layout clarity: input panel top (card), results panel below (card). | Calculation runs client-side — no preflight. |
| `datos.html` | 08-13 | Simplest screen: a single "Refrescar datos" card with last-refresh timestamp and primary CTA. | Minimal. |
| `informes.html` | 08-14 | Grid of PDF cards. Each card shows: icon, filename, date, size, download + view actions. | Empty state "Informes sin PDF". |
| `luk4.html` | 08-15 | Chrome only — LUK4 viewer is a specialized dashboard. Preserve structure, swap tokens. | Similar lock to Centro de Mando but softer — LUK4 owns its own interaction style. Plan 08-15 decides visual scope during sketch. |
| `plantillas.html` | 08-16 | Read-only list (CRUD is Mark-IV per REF-04). | Read-only table + "Información" banner pointing to Mark-IV deferral. |
| `mis_solicitudes.html` | 08-17 | Table of my approval requests with status pills (pending/approved/rejected/cancelled). Cancel action → confirmation modal. | `?approval_id=<N>` URL param triggers preflight auto-retry (existing behaviour). |
| `cambiar_password.html` | 08-18 | Minimal card, three inputs (current, new, confirm), primary button. Redirects to login on success (flash toast). | Publicly accessible to logged-in users only; no drawer needed (uses top bar only). |
| `ajustes.html` (hub) | 08-19 | Grid of 6 cards (Conexión, Usuarios, Auditoría, Límites, Rendimiento, Solicitudes). Each card: icon + title + 1-sentence description + chevron. Visible to propietario only (Phase 5 D-05). | Card list filtered by `can()` — all cards are gated even though hub is propietario-only (defensive consistency). |
| `ajustes_conexion.html` | 08-20 | Two cards: "Conexión actual" (read-only status) + "Editar credenciales" (form, propietario-only per `conexion:config`). | Form changes trigger confirmation modal before saving. |
| `ajustes_usuarios.html` | 08-21 | Table of users, "Crear usuario" CTA, row actions (edit, deactivate). | Each row edit opens a modal. Deactivate = soft delete (confirmation modal). |
| `ajustes_auditoria.html` | 08-22 | Filters sidebar (user, fecha, path, status) + paginated table + CSV export button. | Heavy table — sticky header, tabular-nums on timestamps. |
| `ajustes_limites.html` | 08-23 | Table of thresholds (endpoint, warn_ms, block_ms, last update) with inline edit. "Recalcular factor" button at top. | Edit triggers save button per row (simple pattern, no modal). |
| `ajustes_rendimiento.html` | 08-24 | Filters + Chart.js canvas + summary table. Chart.js color palette re-tokenized (series colors pull from `--color-primary`, `--color-info`, `--color-warn`, `--color-success`). | Chart.js integration unchanged API-wise; only the `backgroundColor`/`borderColor` defaults swap to token values. |
| `ajustes_solicitudes.html` | 08-25 | Table of pending approvals with approve/reject actions. Row expand shows request details. | Both actions open confirmation modal. |
| `_partials/mapa_pabellon.html` (L) | Not sketched — LOCKED by D-16 | No changes to HTML/Alpine. Only `static/css/app.css` rules that indirectly style it (via `.card`, `.spinner`) re-tokenize. | Audit: after tokens land, visually diff `/centro-mando` against pre-Phase-8 screenshot. Any regression = partial is un-re-tokenizable → revert. |

---

## Sketch Workflow Integration

Per UIREDO-04 + D-11/D-12/D-13/D-14/D-15, each screen runs through
`/gsd-sketch` before implementation. The sketch prompt template below
is what Plan 08-04+ plans embed in their task list. This ensures all
sketches produce output consumable by the subsequent implementation
plan.

### Sketch prompt template

```
You are sketching Phase 8 variant proposals for the Nexo screen:
{{ screen_template }}

Inputs you MUST read first:
- .planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md (this file)
- .planning/phases/08-redise-o-ui-modo-claro-moderno/08-CONTEXT.md
- The current {{ screen_template }} for structural / feature parity.
- nexo/services/auth.py PERMISSION_MAP for permission gates referenced by this screen.

Constraints (non-negotiable):
- Apply ONLY tokens declared in 08-UI-SPEC.md. No hex values, no ad-hoc rgba.
- Use semantic utilities (bg-surface, text-body, border-subtle, etc.).
- Typography: 4 sizes only (Body 14, Subtitle 16, Heading 20, Display 32)
  and 2 weights only (400 regular, 600 semibold). No other sizes or
  weights anywhere in the sketch — this is the Dimension 4 contract.
- Preserve Phase 5 RBAC gating: {% if can(current_user, "<perm>") %} wraps every sensitive button.
- Chrome is provided by base.html (top bar + drawer) — do NOT re-invent.
- No emojis. Spanish user-facing copy.
- Motion budget: ≤200ms for drawer-class, ≤300ms for modal-class.
- Light theme only.
- Respect prefers-reduced-motion.

Output:
- N self-contained HTML files in .planning/sketches/08-{{ screen }}-variant-{1..N}.html.
- Each variant fully standalone (inline CSS, inline Alpine bootstrap, demo data).
- Each variant includes a header comment block:
    /* Variant X — {{ short_name }}
       Rationale: <one sentence>
       Token inventory: <list of tokens used>
       Component inventory: <list of components used>
       Deviations from 08-UI-SPEC: <list or "none"> */
- Do not commit; leave files as drafts for the user to review in-browser.

Minimum N per screen:
- Simple screens (datos, cambiar_password, plantillas): 2 variants
- Medium screens (informes, ciclos, recursos, operarios, capacidad,
  ciclos_calc, mis_solicitudes, ajustes_*): 3 variants
- Complex screens (pipeline, bbdd, historial, ajustes hub, centro_mando,
  login, bienvenida, luk4): 4 variants

Ask the user to confirm the N before generating. Do not auto-decide.
```

### Post-selection workflow

Per D-13, after the user picks a variant:

1. `/gsd-sketch-wrap-up` runs and produces
   `.claude/skills/sketch-findings-{screen}/SKILL.md` with: chosen
   variant index, token inventory used, component variants used,
   rationale, landmines encountered.
2. The implementation plan (`08-0N-PLAN.md`) reads the skill first
   and treats the chosen sketch as the design baseline.
3. Plan executor's success criteria include: token inventory from
   skill matches tokens used in the final template (prevents drift).

---

## Registry Safety

| Registry | Blocks Used | Safety Gate |
|----------|-------------|-------------|
| shadcn official | (none — stack is not React) | not applicable |
| shadcn-style third-party blocks | (none) | not applicable |

No third-party component registry is in scope. All components are
hand-written within `static/css/app.css` (utility compositions) +
Tailwind utilities in templates. The only external runtime dependencies
are the ones already loaded via CDN in `base.html` (Tailwind 3 CDN,
Alpine 3.14.8, HTMX 2.0.4, Chart.js 4.4.7). These are not UI-registry
in the shadcn sense — they are JS libraries, already vetted for
Phase 1-7 use.

If a future plan wants to introduce a headless component (e.g. a
Radix-style popover library because native `<select>` isn't enough),
the registry vetting gate from the UI researcher playbook applies and
UI-SPEC must be amended via `/gsd-refresh-ui-spec`.

---

## Open Questions

Items flagged as Claude's Discretion that warrant final operator sign-off
before Plan 08-0N picks them up. Each has a proposed default; the
operator can accept-by-silence or override per-plan.

1. **Keyboard shortcut for drawer toggle** (D-09). Proposed default: `[`.
   Fallback: `Alt+S`. Operator may prefer `Ctrl+\`. Decision deadline:
   Plan 08-02 review.
2. **Sub-nav treatment for `/ajustes`** in the drawer (D-discretion).
   Proposed default: treat `Ajustes` as a single top-level entry in the
   drawer; inside the `/ajustes` hub, show a 6-card grid that acts as
   the navigation. No in-drawer expandable group. Alternative (if user
   finds this too clicky): in-drawer expandable section showing all 6
   sub-pages. Decision deadline: Plan 08-19 review.
3. **Loading hero tagline rotation**: current table (see State
   Contracts § Loading) is deterministic. If the operator wants a
   playful random rotation per load, swap to `Math.random()` index in
   `app.js`. Default: deterministic. Decision deadline: Plan 08-03.
4. **Role pill wording**: current backend stores `propietario` /
   `directivo` / `usuario` lowercased. Pill displays uppercase via
   `tracking-wide`. Alternative: capitalize (`Propietario`,
   `Directivo`, `Usuario`) and drop uppercase. Default: uppercase
   (matches existing `base.html` behaviour). Decision deadline: Plan
   08-02 review.
5. **`bienvenida.html` saludo first-name source**: operator may prefer
   always using email local-part (more deterministic — the user's
   name field might be empty for legacy accounts). Default: prefer
   `current_user.nombre`, fall back to email local-part. Decision
   deadline: Plan 08-03 review.

---

## Checker Sign-Off

- [ ] Dimension 1 Copywriting: PASS when empty/error/destructive copy tables
      are present and every state maps to a family.
- [ ] Dimension 2 Visuals: PASS when spacing, radius, elevation tables
      are declared and the global chrome has exact heights.
- [ ] Dimension 3 Color: PASS when the 60/30/10 split is declared with
      hex values and the accent reserved-for list has at least 3
      entries and no "all interactive elements".
- [ ] Dimension 4 Typography: PASS when no more than 4 font sizes and no
      more than 2 font weights are declared, each with an assigned
      role and line-height.
- [ ] Dimension 5 Spacing: PASS when the raw scale is multiples of 4
      and exceptions are listed individually.
- [ ] Dimension 6 Registry Safety: PASS when registry section states
      not applicable with rationale (stack is not React).
