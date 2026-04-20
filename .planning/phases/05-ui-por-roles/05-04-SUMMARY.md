---
phase: 05-ui-por-roles
plan: 04
subsystem: ui
tags: [templates, jinja2, alpine, rbac, fastapi, ajustes, phase-5]

requires:
  - phase: 05-ui-por-roles
    provides: "can(current_user, perm) registered as Jinja global + require_permission trampoline (Plan 05-01)"
  - phase: 05-ui-por-roles
    provides: "FlashMiddleware + StarletteHTTPException handler for HTML-GET 403 → 302+flash cookie (Plan 05-03)"

provides:
  - "templates/ajustes_conexion.html dedicated sub-page for Conexion SQL Server"
  - "GET /ajustes/conexion HTML route gated by conexion:config"
  - "templates/ajustes.html refactored: 6 cards with per-card can() gates, no SMTP, no inline conexion block"
  - "Alpine scope isolation: hub is static shell, conexion form owns ajustesConexionPage()"

affects: [05-05-ui-por-roles-html-get-button-gating]

tech-stack:
  added: []
  patterns:
    - "Hub → sub-page split for ajustes module (following ajustes_limites.html pattern)"
    - "Per-card can(current_user, perm) gates replace wrapper role==propietario in hub templates"
    - "Alpine scope isolation via function rename (ajustesPage → ajustesConexionPage) to prevent cross-template collision"

key-files:
  created:
    - "templates/ajustes_conexion.html"
    - "tests/routers/test_ajustes_split.py"
  modified:
    - "templates/ajustes.html"
    - "api/routers/pages.py"
    - ".planning/phases/05-ui-por-roles/deferred-items.md"

key-decisions:
  - "D-04 reinterpretacion UIROL-03: el plan literal menciona 6 sub-paginas incluyendo ajustes_smtp.html. D-04 (SMTP Out of Scope Mark-III) lo supersede. Hub envia 6 cards SIN SMTP: conexion, usuarios, auditoria, solicitudes, limites, rendimiento."
  - "Hub es shell puro: eliminamos wrapper Alpine (I-05) porque el unico consumidor del componente era la card inline de Conexion, que ahora vive en ajustes_conexion.html."
  - "Function rename ajustesPage → ajustesConexionPage: evita colision de scope Alpine si ambos templates coexistieran durante desarrollo y sirve como marca para detectar regresion en tests (assert 'ajustesConexionPage' in body)."

patterns-established:
  - "Hub/sub-page split seguira este patron para cualquier futura refactorizacion de cards de /ajustes que crezcan mas alla de un enlace (movable a ajustes_X.html propio, gateado por su propio permiso, hub queda como menu)."
  - "Asserciones anti-SMTP (y en general anti-feature-deferred) usan patron anclado href=\"/path\" en lugar de substring, para evitar falsos positivos cuando el copy del UI incluye la palabra en contexto legitimo (I-02)."

requirements-completed: [UIROL-03]

duration: 8min
completed: 2026-04-20
---

# Phase 5 Plan 04: Ajustes hub split + /ajustes/conexion sub-page Summary

**Hub /ajustes reducido a shell estatico con 6 cards gateadas por can(); bloque inline de Conexion extraido a /ajustes/conexion propio (propietario-only), sin card SMTP (D-04).**

## Performance

- **Duration:** 8 min
- **Started:** 2026-04-20T18:34:35Z
- **Completed:** 2026-04-20T18:42:40Z
- **Tasks:** 4
- **Files modified/created:** 5 (2 templates, 1 router, 1 test, 1 deferred-items update)
- **LOC delta en ajustes.html:** +32/-137 (−105 LOC netos, supera el objetivo de −60 LOC del plan)

## Accomplishments

- Nueva pagina `/ajustes/conexion` (propietario-only via `conexion:config` = `[]` en PERMISSION_MAP) que renderiza `templates/ajustes_conexion.html` (152 LOC) con el formulario completo de conexion SQL Server + info sistema + componente Alpine `ajustesConexionPage()`.
- Hub `templates/ajustes.html` (103 LOC, era 208) refactorizado: 6 cards enlazadas, cada una gateada individualmente con `{% if can(current_user, "<perm>") %}`. Orden: Conexion → Usuarios → Auditoria → Solicitudes → Limites → Rendimiento.
- Eliminado wrapper Alpine del hub (era `<div x-data="ajustesPage()" x-init="loadConfig()">`). Justificacion I-05: el unico consumidor del estado era el formulario de Conexion, que se mudo. Decision documentada en body del commit `f0d984c`.
- Endpoints backend sin modificar: `/api/conexion/config` (GET/PUT) y `/api/conexion/status` siguen en `api/routers/conexion.py` con sus gates existentes (Open Q4).
- 6 integration tests nuevos en `tests/routers/test_ajustes_split.py` cubren: propietario ve 6 cards y ninguna SMTP, propietario GET /ajustes/conexion renderiza, non-propietario HTML → 302, non-propietario JSON → 403, hub sigue propietario-only via require_permission.

## Task Commits

1. **Task 1:** `feat(templates): add ajustes_conexion.html sub-page (D-06, UIROL-03)` — `bee3e23`
2. **Task 2:** `feat(routers): add GET /ajustes/conexion guarded by conexion:config (D-06)` — `757cac3`
3. **Task 3:** `refactor(ajustes): extract conexion sub-page, per-card can() gates, drop SMTP (D-04, D-05, D-06)` — `f0d984c`
4. **Task 4:** `test(routers): integration tests for ajustes hub split + conexion sub-page` — `800bdfb`

## Files Created/Modified

- `templates/ajustes_conexion.html` (created, 152 LOC) — Sub-pagina dedicada para Conexion SQL Server. Extiende `base.html`, `x-data="ajustesConexionPage()"`, breadcrumb "← Volver a Ajustes", tarjeta con form completo (servidor/puerto/DB/user/password/driver/encrypt/trust/uf_code), semaforo de estado, botones Guardar/Probar conexion, info sistema (API Docs + /bbdd + v2.0.0), script Alpine al final del template con el componente renombrado.
- `templates/ajustes.html` (modified, 103 LOC; era 208) — Hub reducido a grid de 6 cards. Ningun bloque inline. Ningun componente Alpine. Ningun ref a SMTP.
- `api/routers/pages.py` (modified, +13 LOC) — Nueva ruta `GET /ajustes/conexion` con `dependencies=[Depends(require_permission("conexion:config"))]`. Comentario inline explica el gate y el acoplamiento con Pitfall 6.
- `tests/routers/test_ajustes_split.py` (created, 272 LOC, 6 tests) — Integration tests con TestClient + seed/purge de usuarios sintéticos en `@ajustes-split-test.local`, reset de slowapi, assertions ancladas para hrefs y SMTP.
- `.planning/phases/05-ui-por-roles/deferred-items.md` (modified, +5 LOC) — Anade el 3er test de `test_thresholds_crud.py` (`test_recalibrate_insufficient_data_returns_400`) al registro DEF-05-01-A de contaminación de sample-size.

## Pre-smoke snapshot (manual verification)

Live `GET /ajustes` como `propietario` via TestClient dentro de `docker compose exec web`. Hrefs observados en el body (ordenados alfabeticamente):

```
/ajustes                  (sidebar link — viene de base.html, no del hub)
/ajustes/auditoria
/ajustes/conexion
/ajustes/limites
/ajustes/rendimiento
/ajustes/solicitudes
/ajustes/usuarios
```

`"SMTP" in body`: **False**. `href="/ajustes/smtp" in body`: **False**. Cumplido D-04.

## Decisions Made

- **Hub sin Alpine:** el antiguo `x-data="ajustesPage()" x-init="loadConfig()"` servia exclusivamente al formulario inline. Tras mover ese formulario a `/ajustes/conexion`, el hub no consume estado compartido — las 6 cards son `<a>` estaticos. Decision I-05 del plan; documentada en commit `f0d984c`.
- **Orden de cards en la grid:** Conexion primero por ser la que introduce el nuevo flujo; resto mantiene el orden previo (Usuarios, Auditoria, Solicitudes, Limites, Rendimiento).
- **Reuso verbatim del componente Alpine:** el cuerpo de `ajustesConexionPage()` es identico al de `ajustesPage()` original — solo cambia el nombre. Esto preserva el contrato con los endpoints backend (misma forma de `cfg`, mismos fetch calls, mismo flujo loadConfig/guardar/testConnection).

## UIROL-03 reinterpretacion

El literal de UIROL-03 en `REQUIREMENTS.md` dice:

> `templates/ajustes.html` dividido en `ajustes_conexion.html`, `ajustes_smtp.html`, `ajustes_usuarios.html`, `ajustes_auditoria.html`, `ajustes_limites.html`, `ajustes_solicitudes.html`, + hub `ajustes.html`

**Este plan (05-04) cumple 1 de los 6 splits (Conexion).** Los otros splits (`ajustes_usuarios.html`, `ajustes_auditoria.html`, `ajustes_limites.html`, `ajustes_solicitudes.html`) ya se habían creado en Phase 2 (02-04) y Phase 4 (04-03, 04-04). La excepcion es `ajustes_smtp.html`: **NO se crea** porque D-04 (docs/OPEN_QUESTIONS.md) declara SMTP como Out of Scope Mark-III.

**Hub card set final (6, SIN SMTP):** conexion + usuarios + auditoria + solicitudes + limites + rendimiento. El hub tiene una card extra respecto al literal original (`rendimiento`, añadida en Plan 04-04 / D-11) y una menos (`smtp`).

El plan-checker (W-01 mitigation) avisó de esta reinterpretacion antes de arrancar; se documenta aqui para trazabilidad.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Typo en `{% block page_title %}` durante Write inicial de ajustes.html**
- **Found during:** Task 3 (refactor hub)
- **Issue:** El Write inicial incluyo por descuido un `{% endml %}` malformado concatenado con el correcto `{% endblock %}`, produciendo `{% block page_title %}Ajustes{% endml %}{% block page_title %}Ajustes{% endblock %}`. Jinja hubiera fallado al parsear con un error de tag desconocido.
- **Fix:** Edit inmediato para dejar una linea limpia `{% block page_title %}Ajustes{% endblock %}` antes del commit.
- **Verification:** `python3 -c "from jinja2 import Environment, FileSystemLoader; e = Environment(loader=FileSystemLoader('templates')); e.parse(open('templates/ajustes.html').read())"` → OK.
- **Committed in:** `f0d984c` (commit de Task 3, el typo nunca llegó a main).

**2. [Rule 1 - Bug] Comentario HTML con la palabra "SMTP" sobreviviria al refactor y romperia `assert 'SMTP' not in body`**
- **Found during:** Task 3 (verification post-refactor)
- **Issue:** Deje un comentario inline `<!-- sin card SMTP — D-04 ... -->` explicando el gate que falta. Aunque el test del plan usa patron anclado (I-02 mitigation: `href="/ajustes/smtp" not in body`), una revision por substring fallaria. Para preservar defensiva-en-profundidad del test, el comentario no debe contener el string SMTP.
- **Fix:** Reformule el comentario a hablar de "email" en lugar de "SMTP" (mismo significado para el lector).
- **Verification:** `grep -ci 'smtp' templates/ajustes.html` → 0.
- **Committed in:** `f0d984c` (mismo commit de Task 3).

**3. [Rule 1 - Bug] CAUSADA POR MI: `rm -f /app/tests/routers/test_ajustes_split.py` dentro del container borro el archivo del host via bind-mount `./tests:/app/tests`**
- **Found during:** Task 4 verification (post-regresion sweep)
- **Issue:** Al intentar reproducir las 3 fallas de `test_thresholds_crud.py` contra el baseline `HEAD~3`, ejecute `docker compose exec web rm -f /app/tests/...` para limpiar el test nuevo temporalmente. No recorde que `docker-compose.yml` tiene `./tests:/app/tests` como bind-mount, asi que el delete se propago al host y `git status` mostró el archivo como untracked perdido.
- **Fix:** Re-escribi `tests/routers/test_ajustes_split.py` verbatim desde el contenido que ya tenia en memoria (el Write tool aun tenia el contenido previo en el turno anterior). Re-corri los 6 tests: 6 passed. Commit `800bdfb` incluye el archivo correcto.
- **Verification:** `pytest tests/routers/test_ajustes_split.py -q` → 6 passed.
- **Committed in:** `800bdfb` (archivo restaurado antes del commit de Task 4; el delete temporal nunca llegó a main).
- **Lesson:** El bind-mount `./tests:/app/tests` existe para que los PDFs baseline del gate de regresión Plan 03-02 sobrevivan a `docker compose up --build`. Toda operación de borrado/modificación dentro del container sobre `/app/tests/*` impacta el working tree del host. Nunca mas.

---

**Total deviations:** 3 auto-fixed (3 Rule 1 bugs). Ninguna Rule 4 (no cambios arquitectónicos). Ninguna Rule 2/3.
**Impact on plan:** Todas las fixes fueron defensivas (typo jinja que hubiera roto el render, string SMTP residual en comentario, bind-mount gotcha que casi borra la suite de tests). Ninguna cambio el alcance del plan.

## Issues Encountered

- **3 tests preexistentes en `tests/routers/test_thresholds_crud.py` siguen en rojo.** Confirmé ejecutando los 3 tests contra `HEAD~3` (antes de cualquier commit de Plan 05-04): mismo 3/3 fail. Plan 05-04 no los introduce. Documentado en `.planning/phases/05-ui-por-roles/deferred-items.md` / DEF-05-01-A.
- **`ruff` no instalado en la imagen Docker.** El container web no tiene ruff ni en `$PATH` ni como modulo Python. No bloqueante (CI en host ya corre ruff — mencionado en STATE.md como pendiente Sprint 6 / Phase 7).

## User Setup Required

None — cambios 100% template/routing, sin nuevas env vars, sin nuevos servicios externos.

## Next Phase Readiness

**Ready for Plan 05-05 / Wave 5 (button gating + HTML-GET hardening):**

- El patrón hub/sub-page esta establecido; si Plan 05-05 necesita refactorizar otros hubs similares (no hay candidatos obvios pendientes), el shell de `ajustes.html` sirve de referencia.
- El refactor a `can()` per-card en hubs esta completo para `/ajustes`. Plan 05-05 se centra en botones dentro de paginas y en gatear GETs HTML que hoy se apoyan solo en AuthMiddleware — área distinta, sin colisión con este plan.
- `_PERMISSION_LABELS` en `api/main.py` ya cubre `conexion:config` y los 10 permisos HTML-guarded que Plan 05-05 enumera (confirmado por `test_flash_label_coverage` en `test_forbidden_redirect.py`).

## Self-Check

- Files exist:
  - `templates/ajustes_conexion.html` — FOUND
  - `templates/ajustes.html` — FOUND (modified)
  - `api/routers/pages.py` — FOUND (modified)
  - `tests/routers/test_ajustes_split.py` — FOUND
- Commits exist:
  - `bee3e23` feat(templates): add ajustes_conexion.html sub-page — FOUND
  - `757cac3` feat(routers): add GET /ajustes/conexion — FOUND
  - `f0d984c` refactor(ajustes): extract conexion sub-page — FOUND
  - `800bdfb` test(routers): integration tests for ajustes hub split — FOUND

## Self-Check: PASSED

---
*Phase: 05-ui-por-roles*
*Plan: 04*
*Completed: 2026-04-20*
