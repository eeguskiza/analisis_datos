---
phase: 08-redise-o-ui-modo-claro-moderno
type: context
status: ready-for-research
created: 2026-04-22
mode: discuss
requirements: [UIREDO-01, UIREDO-02, UIREDO-03, UIREDO-04, UIREDO-05, UIREDO-06, UIREDO-07, UIREDO-08]
decisions_count: 23
---

# Phase 8 Context — Rediseño UI (modo claro moderno)

<domain>
## Phase Boundary

**Goal (from ROADMAP.md):** Rediseño visual completo de la aplicación manteniendo el Centro de
Mando como ancla. Look&feel moderno en tema claro, fácil de leer, no sobrecargado. Sidebar
colapsable tipo drawer/popup en lugar de barra lateral siempre visible. Animaciones sutiles
donde aporten. Discussion window-by-window con propuestas visuales antes de tocar código.

**Depends on:** Phase 5 (permisos y filtrado estables — el rediseño NO puede regresionar la
capa de autorización), Phase 6 (despliegue existente para validar en LAN antes/después),
Phase 7 (DevEx para iterar rápido durante el rediseño).

**Requirements:** UIREDO-01..UIREDO-08 (ver `.planning/REQUIREMENTS.md`).

**Success Criteria:** ver ROADMAP.md §"Phase 8" — 8 criterios, todos relevantes para Phase 8.

**Mark-III closure:** Phase 8 es la última phase de Mark-III. Su cierre dispara el release
checklist de `docs/RELEASE.md` (tag `v1.0.0`) y el deploy LAN de `docs/DEPLOY_LAN.md`.
</domain>

<decisions>
## Implementation Decisions

### A. Token architecture + palette (UIREDO-01)

- **D-01 — Two-layer token structure.** `static/css/tokens.css` declara dos capas: raw
  (scales `--color-blue-500`, `--color-gray-400`, etc.) + semántica (`--color-primary`,
  `--color-surface`, `--color-text-body`, `--color-text-muted`, `--color-border`, …).
  Templates y `app.css` consumen **sólo** la capa semántica. Renombrar una paleta =
  editar la capa raw. Documentar cada valor en `docs/BRANDING.md`.

- **D-02 — Tailwind consume tokens vía `theme.extend`.** La config de Tailwind
  (actualmente inline en `base.html`, se moverá a archivo dedicado en Plan 08-01) declara
  `colors: { primary: 'rgb(var(--color-primary) / <alpha-value>)', … }` con los tokens en
  formato `R G B` (space-separated RGB) en `:root`. Preserva la DX de utilidades
  (`bg-primary`, `text-muted`). Sin `bg-[var(...)]` en templates.

- **D-03 — Palette scope = current brand + fill gaps.** Se conserva la escala `brand.50..900`
  (azul oscuro) y `surface.50..300` (neutros claros) ya en `base.html`. Se completan:
  `surface.400..900`, tokens semánticos `success` / `warn` / `error` / `info` desde cero, y
  derivaciones de text/border. Sin rebranding; sin nueva paleta.

- **D-04 — Light-only, sin hooks para dark mode.** Nada de `:root[data-theme='dark']`,
  nada de `@media (prefers-color-scheme: dark)`. YAGNI estricto para Mark-III. Mark-IV
  decide si/cómo añadir dark desde cero.

- **D-05 — tokens.css cubre más que colores.** Incluye:
  - **Typography** — escala `--text-xs..2xl`, line-height scale, weight tokens.
  - **Spacing** — aliases semánticos (`--space-card-padding`, `--space-section-gap`,
    `--space-field-gap`) encima de la scale raw de Tailwind.
  - **Radius + elevation** — `--radius-sm/md/lg` + `--shadow-card/popover/modal`.
  - **Motion** — `--duration-fast: 150ms`, `--duration-base: 200ms`, `--duration-slow: 300ms`,
    `--ease-standard`. Centraliza el presupuesto ≤200ms drawer + ≤300ms animación total de
    UIREDO-02/05.

- **D-06 — Font family = Tailwind system stack.** `font-sans` default (San Francisco /
  Segoe UI / system UI). Sin Google Fonts CDN, sin self-hosted. LAN-friendly, zero cost,
  no añade red externa al stack.

### B. Sidebar drawer behavior (UIREDO-02)

- **D-07 — Desktop default = fully hidden behind hamburger.** Sin rail permanente. El
  sidebar actual (`w-56`/`w-16` siempre visible en `base.html`) se reemplaza por un drawer
  completamente oculto que se abre al clicar un hamburger en el top bar. Maximiza espacio
  útil de la pantalla.

- **D-08 — Estado del drawer persiste vía `localStorage`.** Key `nexo.ui.drawerOpen`.
  El estado sobrevive navegación entre páginas, no se envía al servidor (no cookie), no
  cruza dispositivos. Una sola fuente de verdad: Alpine lee/escribe el storage.

- **D-09 — Comportamiento mobile + teclado.** Cuatro requisitos simultáneos:
  1. **Mobile** (`< md:` = 768px breakpoint): drawer se vuelve overlay full-viewport que
     desliza desde la izquierda + backdrop dim + click en backdrop cierra.
  2. **Esc cierra el drawer** en cualquier breakpoint; focus vuelve al toggle.
  3. **Focus trap** dentro del drawer mientras está abierto — Tab cicla sólo por los items
     del nav (implementación vía pequeño helper Alpine o `inert` en el resto del body).
  4. **Keyboard shortcut** para togglear el drawer — tecla exacta se decide en Plan 08-02.
     Opciones a investigar: `\`, `gm`, `Ctrl+\`.

- **D-10 — Top bar mínima + hamburger + user menu.** El redesign introduce un top bar
  global (nuevo, no existe hoy) en `base.html` con tres slots: `[hamburger]`
  `[{{ page_title }}]` `[{{ user.email }} + logout]`. El user menu emigra del sidebar al
  top bar. La marca (logo Nexo) vive dentro del drawer, no en el top bar (el top bar es
  funcional, no identitario).

### C. Per-screen proposal workflow (UIREDO-04)

- **D-11 — Propuestas visuales vía `/gsd-sketch`.** Cada screen spawnea un
  `/gsd-sketch` que genera N variantes HTML auto-contenidas en `.planning/sketches/`.
  El usuario las abre en el navegador y decide.

- **D-12 — Sign-off del usuario en todas las propuestas.** El usuario revisa y aprueba
  **todas** las propuestas antes de implementar — no sólo Centro de Mando. Claude
  genera las opciones, el usuario pica. (Nota: mensaje literal del usuario: "yo quiero
  ver las propuestas antes de aceptarlas".)

- **D-13 — Tras la elección, `/gsd-sketch-wrap-up` → skill por screen.** Cada sketch
  ganador se empaqueta en una skill `.claude/skills/sketch-findings-{screen}/` que
  documenta: variante escogida, tokens usados, variantes de componente, rationale,
  landmines detectadas. `/gsd-plan-phase 8` lee estas skills antes de producir plans.

- **D-14 — Número de propuestas por screen = variable, Claude decide.** Screens simples
  (ej. `datos.html`) pueden tener 2; screens complejas (pipeline, ajustes hubs) pueden
  tener 4+. Claude propone el N por screen y el usuario confirma al abrir el sketch.
  Mínimo absoluto = 2 (literal de UIREDO-04).

- **D-15 — Unidad de screen = un template Jinja.** Cada archivo en `templates/*.html`
  (y el partial `_partials/mapa_pabellon.html`) es su propio ciclo de propuesta. ~24
  ciclos totales. No se agrupan por feature.

- **D-16 — Centro de Mando: interacción LOCKED, visual abierto.** El patrón de
  plano-de-fondo + máquinas editables por el usuario (pattern actual de `/`) es un
  **hard must** — las propuestas para Centro de Mando conservan esa interacción sin
  excepción. Sólo los tokens, typography, spacing, containers y chrome alrededor son
  susceptibles de rediseño. Literal del usuario: "el funcionamiento principal que ya
  hay de un plano de fondo y máquinas encima y que yo genero y edito es must".

### D. Execution order + rollout (UIREDO-06, UIREDO-08)

- **D-17 — Orden: chrome primero, Centro de Mando segundo, resto después.**
  Secuencia fija:
  1. **Plan 08-01**: `tokens.css` + `tailwind.config` (extraída de `base.html`).
  2. **Plan 08-02**: nuevo top bar + drawer en `base.html` (RBAC `can()` preservado).
  3. **Plan 08-03**: `/bienvenida` landing (ver D-23).
  4. **Plan 08-04**: Centro de Mando (`index.html` / `centro_mando.html`) — sólo visual.
  5. **Plan 08-05..08-NN**: un plan por template restante (ajustes hub + sub-pages,
     pipeline, historial, bbdd, capacidad, operarios, recursos, ciclos, datos, login,
     cambiar_password, mis_solicitudes, ciclos_calc, informes, luk4, plantillas, …).

- **D-18 — Branch = `feature/Mark-III` directo.** No se abre branch dedicado para el
  rediseño. Los plans commitean en orden en la misma rama Mark-III. Pre-commit hooks de
  Phase 7 garantizan calidad por commit. El tag `v1.0.0` se corta al cierre.

- **D-19 — Replace immediately, sin coexistencia.** Cada plan sustituye la UI vieja de
  la screen que toca. Sin feature flag `NEXO_UI_V2`, sin toggle per-user. Durante el
  rediseño algunas screens se ven nuevas y otras viejas — asumido. No hay rollback
  granular; un commit específico se revierte con `git revert` si algo regresiona.

- **D-20 — Full Phase 5 test suite GREEN en cada plan.** Cada plan de Phase 8 incluye
  en su success criteria: `tests/auth + tests/routers + tests/middleware` pasan completo
  (no sólo smoke). Igual que Phase 7. Además, los tests per-screen nuevos
  (`tests/infra/test_ui_*.py` o `tests/routers/test_*_template.py`) si se añaden.

- **D-21 — CI a11y tool = `pa11y-ci` sobre el smoke job de compose.** Extiende el job
  `smoke` de Phase 7 (`.github/workflows/ci.yml`): después del health-check, instala
  `pa11y-ci`, itera sobre una lista de URLs (login público + login-ed fixture hitting
  centro de mando / pipeline / ajustes / una sub-page) y falla el job si hay issues de
  contraste AA (WCAG 2.1).

- **D-22 — Mark-III cierra con tag `v1.0.0` + `docs/RELEASE.md`.** Phase 8 verificada ⇒
  bump `CHANGELOG.md` `[Unreleased] → [1.0.0 — 2026-MM-DD]`, tag `v1.0.0`, deploy
  productivo LAN por `docs/DEPLOY_LAN.md`. Mark-IV se abre en un milestone nuevo.

### E. Forms, states, notifications (UIREDO-05, UIREDO-08)

- **D-24 — Form labels = top-aligned (stacked).** Label encima del input, jerarquía
  clara. Sin floating labels. Sin labels horizontales. Estándar en todas las forms
  (login, ajustes_*, filtros).

- **D-25 — Validación server-side autoritativa + inline error + flash toast.** FastAPI
  sigue siendo la validación autoritativa — devuelve 400 con `errors: {field: msg}`
  que Jinja pinta bajo cada campo. Flash toast (Phase 5 D-07/D-08) muestra el resumen.
  Sin client-side validation con Alpine salvo casos explícitos. Sin HTML5 `required`
  como única defensa.

- **D-26 — Marcador `(opcional)` en fields opcionales; required = default.** La mayoría
  de los fields en Nexo son obligatorios. Marcar la minoría opcional mantiene labels
  limpias y evita asteriscos rojos decorativos.

- **D-27 — Loading pattern = spinner centrado en el panel.** Single spinner sobre el
  panel mientras carga. Sin skeleton shimmer. Simpler, no requiere esqueletos por
  screen.

- **D-28 — Empty state = icon + headline + next action button.** Bloque centrado:
  icono Heroicons neutro (ej. `inbox`, `search`, `exclamation-triangle`), headline
  corta, mensaje, botón CTA condicionado a `can()` para que RBAC aplique también en
  empty states.

- **D-29 — Error state = in-panel block + retry button + link a RUNBOOK.** Cuando un
  endpoint devuelve 5xx / timeout / connection lost, el panel se reemplaza por un card
  de error con: icono, mensaje, botón "Reintentar", link a
  `docs/RUNBOOK.md#{scenario}` (los 5 escenarios de Phase 7). Hace accionable el
  fallo.

- **D-30 — Flash toast = top-right, auto-dismiss 4s, queue vertical, pause on hover.**
  Toast slide-in desde la derecha en la esquina superior. Auto-dismiss 4s
  configurable por nivel. Múltiples toasts se apilan. Hover pausa el auto-dismiss
  (a11y contra dismiss hostil). Phase 5 flash middleware contract preservado — sólo
  cambia el estilo.

- **D-31 — Print stylesheet mínimo `@media print`.** Una regla pequeña en
  `static/css/print.css` (~20 líneas): oculta top bar + drawer + botones de acción en
  `@media print`. Permite `Ctrl+P` limpio en pantallas como `/historial`. No se crean
  layouts print per-screen (eso sigue siendo trabajo de matplotlib para reports
  oficiales).

### F. Landing post-login (nueva capacidad añadida durante discuss)

- **D-23 — Nuevo screen `/bienvenida` post-login.** Al hacer login exitoso, redirect
  va a `/bienvenida` (antes iba a `/` = Centro de Mando). El landing muestra:
  - Saludo dinámico grande: "Buenos días / Buenas tardes / Buenas noches, {user.name}"
    (reglas exactas de franja horaria: TBD en Plan 08-03, propuesta inicial:
    06–12 / 12–21 / 21–06).
  - Reloj en tiempo real (tick por segundo vía Alpine `setInterval`).
  - Botón primario "Ir a Centro de Mando".
  - Drawer disponible — el usuario puede navegar a cualquier otra screen si prefiere.

  **Out of scope Phase 8 — deferred a Mark-IV o phase dedicada:** widgets
  configurables por usuario (favoritos, resumen de turno, KPIs personales,
  accesos directos recientes). Requerirá schema per-user en Postgres y API de
  config. Ver Deferred Ideas.

### G. Claude's Discretion

- **Iconografía concreta por empty-state / error** (Heroicons outline por defecto,
  variante específica por contexto) — Claude elige en cada sketch.
- **Z-index scale para drawer / modals / toasts** — Claude propone en Plan 08-01 con
  tokens.css.
- **Exact breakpoint list más allá de `md: 768px`** (mantener Tailwind defaults
  `sm/md/lg/xl/2xl` probablemente).
- **Sub-nav treatment dentro del drawer para `/ajustes`** (grupo expandible, sección
  separada, o items individuales) — decidir en Plan 08-02 o en el sketch de
  `ajustes.html`.
- **Keyboard shortcut exacto para togglear drawer** (D-09) — Claude propone 2–3
  opciones en Plan 08-02.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents (phase-researcher, planner, executor) MUST read these before
planning or implementing.**

### Requirements + Project scope
- `.planning/REQUIREMENTS.md` §UIREDO-01..UIREDO-08 — los 8 requisitos locked.
- `.planning/ROADMAP.md` §"Phase 8: Rediseño UI (modo claro moderno)" — goal +
  success criteria + dependencies.
- `.planning/PROJECT.md` §Out of Scope — SMTP / LDAP / 2FA / exposición internet
  siguen fuera; Phase 8 no los abre.

### Phase 5 contracts (NON-negociables — el rediseño no puede regresionarlos)
- `.planning/phases/05-ui-por-roles/05-CONTEXT.md` D-01 — sidebar filtering fino
  con `PERMISSION_MAP` + `can()` helper.
- `.planning/phases/05-ui-por-roles/05-CONTEXT.md` D-02 — button-level gating vía
  `{% if can() %}` Jinja, sin `x-show` JS para permisos (zero-trust DOM).
- `.planning/phases/05-ui-por-roles/05-CONTEXT.md` D-03 — `can(user, perm)` como
  Jinja global, misma lógica que `require_permission`.
- `.planning/phases/05-ui-por-roles/05-CONTEXT.md` D-07/D-08 — flash toast
  middleware y API; UIREDO-02 conserva contract.

### Phase 7 artefactos que Phase 8 extiende o consume
- `docs/ARCHITECTURE.md` — mapa técnico de los 3 engines + middleware stack;
  Phase 8 no toca la capa de datos.
- `docs/RUNBOOK.md` — los 5 escenarios; el error state (D-29) enlaza a este doc
  como link-to-runbook.
- `docs/RELEASE.md` — el checklist que Phase 8 dispara al cerrarse (D-22).
- `docs/DEPLOY_LAN.md` — deploy productivo post-tag `v1.0.0`.
- `pyproject.toml` / `.pre-commit-config.yaml` — hooks scope `^(api|nexo)/`; el
  ruff backfill no tocó templates ni CSS, siguen fuera del scope hook salvo
  decisión explícita en Plan 08-01.
- `.github/workflows/ci.yml` §job `smoke` — Phase 8 extiende este job con
  `pa11y-ci` (D-21).
- `Makefile` — targets `test`/`lint`/`format` siguen válidos; `make ui-check` o
  similar podría añadirse en Plan 08-01 para correr pa11y localmente.

### Branding
- `docs/BRANDING.md` — paleta actual + convenciones de marca. Plan 08-01 extiende
  este doc con tokens documentados.

### Código base relevante
- `templates/base.html` — sidebar + layout actual que el redesign sustituye.
- `templates/_partials/mapa_pabellon.html` — partial consumida por varios screens.
- `static/css/app.css` — CSS custom actual (inspeccionar para identificar reglas
  que migren a tokens vs se queden inline).
- `static/js/app.js` — Alpine + HTMX glue; el redesign puede extenderlo para
  helpers de focus trap / keyboard shortcut / clock.
- `nexo/services/auth.py` — `can()` + `PERMISSION_MAP` (sidebar / button gating).
- `api/deps.py::render_html` — helper que inyecta `current_user`,
  `app_name`, `logo_path`, `company_name` en templates.
- `api/routers/auth.py` — login flow (D-23 cambia el redirect post-login a
  `/bienvenida`).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Alpine.js 3.14 ya cargado** en `base.html` — suficiente para drawer,
  reloj en tiempo real, focus trap helper. No añadir nuevos frameworks.
- **HTMX 2.0.4 ya cargado** — suficiente para refresh incremental de paneles
  sin full reload. Aprovechar para los loading states (D-27).
- **`can()` Jinja global** — ya registrado en `Jinja2Templates.env.globals`
  desde Phase 5. Disponible en todos los templates. Reusa en drawer nav + empty
  state CTAs (D-28).
- **Flash middleware** (`nexo/middleware/flash.py`) — contract Phase 5 D-07/D-08
  preservado en D-30. Sólo cambia estilo del toast, no API.
- **Heroicons SVGs inline** ya presentes en `base.html` nav_items — tomar como
  fuente canónica para iconografía del redesign.

### Established Patterns
- **Tailwind inline config en `<script>`** dentro de `base.html` — Plan 08-01
  extrae esto a un archivo dedicado (probablemente `static/js/tailwind.config.js`
  o inline reducido referenciando CSS vars en `tokens.css`).
- **Sin build step para CSS/JS** — Tailwind vía CDN, scripts inline. El redesign
  mantiene zero build step si es posible.
- **Templates flat en `templates/*.html`** — sin subfolders; una subfolder por
  screen grande se podría introducir en Plan 08-05+ pero no se fuerza.
- **Forms usan POST + server redirect + flash toast** — Phase 2/4 pattern; D-25
  lo mantiene.

### Integration Points
- `base.html` es el punto de entrada de toda la UI; el redesign de chrome
  (Plan 08-02) lo reescribe.
- `api/deps.py::render_html` es el único helper que inyecta contexto de render a
  templates — si se añade un context var nuevo (ej. `drawer_open`) aquí.
- `api/routers/auth.py` login flow + `api/routers/pages.py` rutas HTML —
  `/bienvenida` (D-23) se añade en `pages.py`.

</code_context>

<specifics>
## Specific Ideas

- **Centro de Mando interaction LOCKED** (D-16): el plano-de-fondo + máquinas
  editables son must. Ninguna propuesta visual puede cambiar ese pattern.
- **Landing screen `/bienvenida`** (D-23): saludo dinámico por hora, reloj en
  tiempo real, botón a Centro de Mando. Minimum viable — la customización
  per-user (widgets, favoritos, KPIs personales) se defiere a Mark-IV o phase
  dedicada.
- **Hamburger-hidden drawer** (D-07): referencia operativa = Notion, Linear,
  Figma — drawer / top bar / content-first layouts, sin rail permanente.
- **Light theme "modo claro moderno"** (D-03 + D-05): neutros existentes
  (surface.50-300) + azul brand existente, no re-branding. El "moderno" viene
  de typography scale, spacing breathable, radius/shadow sutiles, animaciones
  <300ms — no de un cambio de paleta.

</specifics>

<deferred>
## Deferred Ideas

### Post-Mark-III (Mark-IV o phase dedicada)

- **Landing customization per-user** — widgets configurables en `/bienvenida`:
  favoritos, resumen de turno activo, KPIs personales, accesos directos
  recientes. Requiere schema per-user en `nexo.users` o tabla
  `nexo.user_dashboard_config`, API de persistencia, componentes de widget.
  Referencia literal del usuario: "y luego iremos viendo cómo customizarla
  según el usuario".

- **Dark theme** — tokens.css sólo carga light; D-04 congela la decisión.
  Mark-IV decide si / cómo añadir dark, y lo hace desde cero con una phase
  dedicada.

- **Per-screen print layouts** — D-31 ships sólo un `@media print` mínimo.
  Layouts detallados (headers/footers/page breaks per screen) siguen siendo
  trabajo de `matplotlib` para reports oficiales. Si Mark-IV los necesita,
  phase nueva.

- **Axe-core via Playwright** — D-21 adopta `pa11y-ci` por simplicidad. Si
  el rigor de axe-core se considera necesario (auditoría externa, cliente
  pide proof), Mark-IV puede migrar. `pa11y-ci` cubre AA baseline suficiente
  para UIREDO-08.

- **Visual diff screenshots** — la regression strategy D-20 se limita a
  tests funcionales (Phase 5 suite GREEN). Visual regression (Playwright
  screenshots vs baseline) se considera si se detecta drift repetido;
  default Mark-III es no.

### Scope creep evitado durante discusión

- Ninguno — la discusión añadió una capacidad nueva (D-23 landing) pero el
  usuario lo aceptó explícitamente como parte de Phase 8 y se acotó el
  alcance (customization = deferred). Sin otros ideas fuera de scope.

</deferred>

---

*Phase: 08-redise-o-ui-modo-claro-moderno*
*Context gathered: 2026-04-22*
