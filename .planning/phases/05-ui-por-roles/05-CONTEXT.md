---
phase: 05-ui-por-roles
type: context
status: ready-for-research
created: 2026-04-20
mode: discuss
requirements: [UIROL-01, UIROL-02, UIROL-03, UIROL-04, UIROL-05]
decisions_count: 9
---

# Phase 5 Context — UI por roles

## Phase Boundary (from ROADMAP.md)

**Goal:** Sidebar y páginas muestran sólo lo que el rol del usuario puede ver;
split de `ajustes.html`.

**Depends on:** Phase 2 (roles + auth), Phase 4 (páginas de ajustes existen).

**Requirements:** UIROL-01..UIROL-05.

**Success Criteria** (must be TRUE):

1. Login como propietario muestra todas las entradas del sidebar.
2. Login como usuario de "producción" muestra pipeline/historial/capacidad/
   recursos/ciclos; oculta `/bbdd` y `/ajustes/*`.
3. Login como directivo de "ingeniería" muestra los módulos de su
   departamento.
4. `/ajustes` hub lleva a 6 sub-páginas separadas (conexión, SMTP, usuarios,
   auditoría, límites, solicitudes).
5. Botones "Ejecutar pipeline", "Borrar ejecución", "Sincronizar recursos"
   ocultos si el user no tiene permiso.

## Carrying forward from prior phases

- **AUTH_MODEL locked** (Phase 2): roles propietario/directivo/usuario +
  departments rrhh/comercial/ingenieria/produccion/gerencia; propietario
  tiene bypass global.
- **PERMISSION_MAP existe** en `nexo/services/auth.py` — source of truth
  granular por `modulo:accion` → `list[department_code]`. Lista vacía en
  el mapa = "solo propietario" (convención enforced by `require_permission`).
- **`current_user` ya inyectado** en templates vía `api/deps.py::render_html`
  (helper usado por routers HTML). UIROL-01 prácticamente listo; Phase 5
  solo formaliza el patrón.
- **Middleware auth** (`api/middleware/auth.py`, Phase 2) pobla
  `request.state.user` con departments eager-loaded antes de cerrar la
  sesión ORM — base para todo lo demás.
- **5 de 6 sub-páginas de `/ajustes` ya creadas** en Phase 4: auditoria
  (02-04), usuarios (02-04), limites (04-04), rendimiento (04-04),
  solicitudes (04-03). Falta `ajustes_conexion.html` nuevo; SMTP está
  explícitamente diferido a Mark-IV (CLAUDE.md §No hacer).
- **Out of Scope Mark-III confirmado**: SMTP operativo, LDAP, 2FA,
  delegación inter-departamentos.

## Decisions (9 locked)

### D-01 — Sidebar filtering: fino con PERMISSION_MAP

Cada item del sidebar se asocia a un permiso (p.ej. `/pipeline` →
`pipeline:read`, `/bbdd` → `bbdd:read`). Jinja llama a `can(user, perm)` y
oculta si la intersección (user.departments ∩ PERMISSION_MAP[perm]) es
vacía. Propietario ve todo (bypass).

**Rationale:** UI y backend usan la misma fuente de verdad. Imposible que el
sidebar muestre un link que devuelva 403 al hacer click. Refactor de ~11
items en el `nav_items` array de `base.html`.

**Rejected alternatives:**
- **Grueso por rol + override**: dos fuentes de verdad → des-sincronización
  garantizada cuando cambie PERMISSION_MAP.
- **Híbrido (permiso para módulos, hardcode /ajustes)**: inconsistente;
  `/ajustes` también tiene su permiso (`ajustes:manage`), no hay razón
  para excepcionarlo.

### D-02 — Button-level filtering: Jinja-only `{% if can() %}`

Botones sensibles envueltos en `{% if can(current_user, "pipeline:run") %}
<button>Ejecutar</button>{% endif %}`. Nada de `x-show` en JS para
permisos.

**Rationale:** Zero-trust en el DOM — el botón "Ejecutar pipeline" ni
siquiera aparece en el HTML si el user no lo puede usar. No hay forma de
des-ocultarlo con devtools. Una sola función `can()` compartida con D-01.

**Rejected alternatives:**
- **`window.__USER__.permissions`**: deja los botones en el DOM (aunque
  `x-show=false`) — inspectores pueden hacer `querySelector + .click()`.
  Backend rechazaría, pero el principio "don't leak UI state" se viola.
  También duplica lógica Python↔JS. Sobre-ingeniería para Mark-III donde
  los permisos no cambian en sesión.
- **`/api/me/permissions` fetch**: +1 round-trip por página, botones
  parpadean hasta hidratar, overkill.

### D-03 — Helper de permisos: función Jinja `can(user, perm)`

Exportar `can(user: NexoUser | None, perm: str) -> bool` como función
Jinja global. Implementación **reutiliza la lógica de `require_permission`**
(misma intersección `user.departments ∩ PERMISSION_MAP[perm]` + bypass
propietario). Ubicación: `nexo/services/auth.py` (exportada) + registrada
en `Jinja2Templates.env.globals` al arrancar la app.

**Rationale:** Una función = una fuente de verdad. El dependency de backend
y el helper de template llaman al mismo core.

### D-04 — `/ajustes` SMTP: omitir del hub + no crear page

Quitar la card SMTP completamente del `ajustes.html`. No crear
`ajustes_smtp.html`. UIROL-03 queda funcionalmente como "hub + 5 sub-pages
+ nota de SMTP diferido en docs".

**Rationale:** SMTP está explícitamente Out of Scope Mark-III
(CLAUDE.md §No hacer). Crear un stub "Próximamente" da falsa expectativa y
requiere mantenimiento.

**Deferred:** Mark-IV abrirá Phase X dedicada a configuración email +
relay interno; entonces se añadirá la card.

**Rejected alternatives:**
- **Stub "Próximamente"**: page que no hace nada, card clickable pero
  inútil.
- **Card con modal explicativo**: patrón ad-hoc, ensucia el hub.

### D-05 — `/ajustes` hub: propietario-only

Ruta `/ajustes` requiere permiso `ajustes:manage` (lista vacía en
PERMISSION_MAP → solo propietario vía bypass). Non-propietario no ve el
link en sidebar (D-01) y si teclea la URL entra al flujo de D-07.

**Rationale:** Ninguna sub-página de ajustes es delegable en Mark-III
(todas tienen lista vacía en PERMISSION_MAP). Coherente con la realidad.

**Rejected alternative:** Hub visible con cards filtradas por permiso
individual — preparado para Mark-IV pero hoy renderizaría 0 cards para
non-propietarios → UX peor (página vacía con "Sin opciones").

### D-06 — `/ajustes/conexion` nueva página

Crear `ajustes_conexion.html` que absorbe la sección "Conexión a BBDD"
que hoy vive embebida en `ajustes.html`. Permiso: `conexion:config` (lista
vacía → propietario-only, consistente con el estado actual).

**Rationale:** El split de UIROL-03 es real para esta sub-página. La
pantalla de "probar conexión / editar credenciales" es suficientemente
distinta para justificar página propia.

### D-07 — Forbidden UX (HTML): redirect + flash toast por Accept header

El `global_exception_handler` (existente desde Phase 1 NAMING-07) se
amplía para capturar `HTTPException(status_code=403)` y actuar según
`Accept` header:

- **HTML** (`Accept: text/html`): `Response 302 Location: /` con cookie
  `nexo_flash` set → dashboard lee y muestra toast "No tienes permiso
  para acceder a {modulo}" arriba.
- **JSON/API** (`Accept: application/json` o path `/api/*`): mantiene el
  comportamiento actual → `{"detail": "Permiso requerido: {perm}"}` con
  status 403. Sin toast, sin redirect. Contract estable para HTMX y
  clients.

**Rationale:** UX limpio para navegación humana sin romper clients
existentes. El 403 JSON es el contract que HTMX, el badge sidebar y
cualquier fetch interno esperan.

**Rejected alternatives:**
- **Página 403 dedicada**: friction extra ("volver al inicio" click);
  innecesario cuando el sidebar ya ocultó el link.
- **Redirect silencioso sin mensaje**: el user piensa que es un bug.

### D-08 — Flash toast: cookie de sesión `nexo_flash`

Implementación del toast de D-07:

- Middleware nuevo (o extensión de AuthMiddleware) lee cookie `nexo_flash`
  al entrar, la borra en la respuesta, inyecta el mensaje en
  `request.state.flash`.
- `api/deps.py::render_html` pasa `flash_message = request.state.flash` a
  los templates; `base.html` muestra un toast en el top-right (4-5s,
  auto-dismiss) cuando `flash_message` existe.
- Cookie: `HttpOnly=True`, `Secure=True` (prod), `SameSite=Lax`, TTL de
  un solo request (se borra al leerla).
- Valor del flash para D-07: string simple "No tienes permiso para
  acceder a {nombre_amigable_modulo}".

**Rationale:** Patrón estándar Flask/Starlette. Funciona con redirects
(el 302 preserva la cookie). Sin dependencias nuevas. Ventaja futura: el
mismo mecanismo sirve para mensajes post-acción ("Usuario creado OK",
"Threshold actualizado").

**Rejected alternatives:**
- **Query param `?denied=bbdd`**: URLs feas tras recargar; acumulables
  si el user navega.
- **Redirect silencioso**: el user no entiende qué pasó.

### D-09 — UIROL-01 formalization: `can` + `current_user` como Jinja globals

Aunque `current_user` ya llega a templates vía `api/deps.py::render_html`,
Phase 5 formaliza:

- Registrar `current_user` y `can` como `Jinja2Templates.env.globals` en
  `api/main.py` (o en un nuevo módulo `api/template_globals.py`).
- Resultado: cualquier template — incluso los que NO pasen por
  `render_html` — accede a `{{ current_user }}` y `{% if can(...) %}`.
  Elimina el riesgo de olvidarse de pasar el contexto.

**Rationale:** Defensivo. Si mañana alguien crea un template nuevo y
olvida usar `render_html`, las vistas siguen funcionando.

## Canonical refs

- `nexo/services/auth.py` — PERMISSION_MAP + `require_permission` (core
  compartido con D-01/D-02/D-03).
- `api/middleware/auth.py` — puebla `request.state.user` con departments
  eager-loaded.
- `api/deps.py::render_html` — helper actual de inyección de
  `current_user` en templates.
- `templates/base.html:48-95` — `nav_items` actual con `visible_to`;
  target principal de refactor D-01.
- `templates/ajustes.html` — hub 208 LOC a split (D-04/D-05/D-06).
- `docs/AUTH_MODEL.md` — modelo de roles/departments/permisos (referencia
  autoritativa para downstream agents).
- `docs/OPEN_QUESTIONS.md` + `CLAUDE.md` §No hacer — SMTP diferido
  (contexto para D-04).

## Out of scope (explicit)

- **SMTP operativo** — diferido a Mark-IV (D-04).
- **LDAP / Active Directory** — diferido Mark-IV.
- **Nuevos roles o permisos por fila de `nexo.roles`** — el modelo
  actual (PERMISSION_MAP en código) es suficiente para Mark-III.
- **Delegación inter-departamental** (directivo de X da acceso puntual a
  directivo de Y) — diferido Mark-IV.
- **Rol `directivo` con controles distintos al `usuario` dentro del mismo
  módulo** — PERMISSION_MAP actual no distingue rol dentro de un
  departamento; cualquier refinamiento aquí abre una línea de trabajo
  propia y se difiere a Mark-IV si el producto lo pide.
- **Refactor de routes `/api/*` para negociar Accept header** — D-07
  mantiene el contract JSON actual; solo el handler HTML se amplía.
- **Cambios en la tabla `nexo.permissions`** — es un snapshot del mapa
  Python, no se toca en Phase 5.

## Deferred ideas

- **Mensajes flash para éxito de acciones** — el mecanismo de D-08 es
  reutilizable; oportunidad de añadir mensajes "Usuario creado OK" tras
  Phase 5 cuando el patrón esté probado.
- **Badge visible para "tienes aprobaciones pendientes como requester"**
  — Phase 4 añadió badge para propietario; el usuario-solicitante no
  tiene señal visual hoy. Candidato a Phase 5.x si la smoke de Phase 4
  destapa que lo necesitan.
- **Pagina 404 custom** — cuando una ruta no existe (vs no autorizada).
  Fuera de scope Phase 5, pero el patrón del Accept header de D-07 sirve
  también aquí.

## Success metric for research phase

El `05-RESEARCH.md` debe responder:

1. **Cómo extraer `can()` de `require_permission` sin duplicar lógica**
   (¿factoring helper o trampolín?).
2. **Cómo registrar Jinja globals** en `Jinja2Templates` (patrón
   documentado de FastAPI + Starlette).
3. **Cómo el middleware lee-y-borra la cookie flash** (cookie policy,
   SameSite, interacción con `Set-Cookie` tras redirect 302).
4. **Catálogo de botones/acciones sensibles** en los templates actuales
   — enumeración para no olvidarse ninguno (grep de `<button>` +
   `@click=` por módulo).
5. **Patrón de extensión de `global_exception_handler`** para capturar
   403 sin romper el contract actual de JSON (el NAMING-07 handler ya
   existe y no debería regresionarse).

## Ready for

`/gsd-research-phase 5` (o `/gsd-plan-phase 5` si no requiere
investigación técnica adicional — D-01..D-09 son suficientemente
concretas).
