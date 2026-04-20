---
phase: 05-ui-por-roles
type: verification
status: human_needed
verified: 2026-04-20
must_haves_total: 5
must_haves_verified: 5
requirements_verified: [UIROL-01, UIROL-02, UIROL-03, UIROL-04, UIROL-05]
requirements_unverified: []
code_review: .planning/phases/05-ui-por-roles/05-REVIEW.md (8 findings; 1 HIGH fixed, 7 MEDIUM/LOW deferred)
---

# Phase 5 — Verification (UI por roles)

Goal check: **Sidebar y páginas muestran sólo lo que el rol del usuario
puede ver; split de ajustes.html.** ✅ Achieved.

5 plans cerrados (05-01 → 05-05) + 1 code review + 1 HI-01 fix. 222 tests
green / 11 skip / 3 fails pre-existentes DEF-05-01-A (scope fuera de
Phase 5). Zero regresiones introducidas por Phase 5.

## Success Criteria Evidence

| # | Criterion | Evidence | Status |
|---|-----------|----------|--------|
| 1 | Propietario ve todas las entradas del sidebar | `templates/base.html` nav_items usa `can(current_user, perm)`; `nexo/services/auth.py::can` propietario → bypass sin consultar PERMISSION_MAP; `tests/routers/test_sidebar_filtering.py::test_propietario_sees_all_nav` | ✅ PASS |
| 2 | Usuario producción ve pipeline/historial/capacidad/recursos/ciclos; oculta /bbdd y /ajustes/* | `can()` + departments intersect con PERMISSION_MAP[perm]; `test_sidebar_filtering::test_produccion_usuario_excludes_bbdd_and_ajustes` green; URL directa a /bbdd → 302 + flash via D-07 handler | ✅ PASS |
| 3 | Directivo ingeniería ve módulos de su departamento | PERMISSION_MAP enforza por department (role no distingue directivo/usuario en Mark-III por diseño); `test_sidebar_filtering::test_directivo_ingenieria_sees_ingeneria_modules` green | ✅ PASS |
| 4 | /ajustes hub lleva a sub-páginas separadas | `templates/ajustes.html` refactored: 6 cards con per-card `{% if can() %}`; nueva `templates/ajustes_conexion.html`; `api/routers/pages.py:140` GET /ajustes/conexion con require_permission("conexion:config") | ✅ PASS* |
| 5 | Botones "Ejecutar pipeline", "Borrar ejecución", "Sincronizar recursos" ocultos si el user no tiene permiso | 11 botones wrapped con `{% if can() %}` en 4 templates; `test_button_gating.py` 10 tests green | ✅ PASS |

**Score: 5/5 must-haves verified via code inspection + automated tests.**

*SC-4 note: UIROL-03 literal menciona 6 sub-pages incluyendo `ajustes_smtp.html`.
D-04 (CONTEXT.md) supersedes: SMTP diferido a Mark-IV. Card set shipping:
conexion + usuarios + auditoria + limites + rendimiento + solicitudes (6 items
sin SMTP). Reinterpretación documentada en 05-04-PLAN.md + 05-04-SUMMARY.md.

## Requirement Traceability

| ID | Phase Deliverable | Complete | Evidence |
|----|-------------------|----------|----------|
| UIROL-01 | `request.state.user` en templates vía Jinja globals / render() | ✅ Plan 05-01 | `api/deps.py:11,31` — `can` + `templates.env.globals`; `render()` inyecta current_user ya desde Phase 4 |
| UIROL-02 | `base.html` filtra nav_items por permisos | ✅ Plan 05-02 + 05-03 | Sidebar refactor (05-02) + forbidden UX 403→302+flash (05-03); redireccion cierra el loop URL-typing |
| UIROL-03 | `ajustes.html` dividido en sub-pages | ✅ Plan 05-04 | `ajustes_conexion.html` nuevo + hub refactored + SMTP omitido (D-04) |
| UIROL-04 | Botones sensibles ocultos según permiso | ✅ Plan 05-05 | 11 botones wrapped exactamente una vez cada uno con permiso correcto de PERMISSION_MAP |
| UIROL-05 | Verificación manual E2E | 🟡 Human pending | Checkpoint auto-aprobado; 5 smoke items documentados en 05-05-SUMMARY §Human verification |

**Traceability: 4/5 Complete via código + 1/5 pending human smoke.**

## Code Review Follow-up

Ran `/gsd-code-review 5` → 0 CRITICAL, 1 HIGH, 4 MEDIUM, 3 LOW.
Ran `/gsd-code-review-fix 5` sobre HIGH only:

| Finding | Severity | Fix Commit | Status |
|---------|----------|------------|--------|
| HI-01 flash cookie race | HIGH | `f821e46` | ✅ fixed |
| MD-01 recursos drawer inputs no gated | MEDIUM | — | ⏭ Phase 5.1 polish |
| MD-02 nav separator guard implícito | MEDIUM | — | ⏭ Phase 5.1 polish |
| MD-03 403 handler parsea exc.detail con startswith | MEDIUM | — | ⏭ Phase 5.1 polish |
| MD-04 ajustes_conexion.html refs stale en comments | MEDIUM | — | ⏭ Phase 5.1 polish |
| LO-01 showToast con 1 arg | LOW | — | ⏭ Phase 5.1 polish |
| LO-02 docstring plan IDs stale | LOW | — | ⏭ Phase 5.1 polish |
| LO-03 convención sin tildes no documentada | LOW | — | ⏭ Phase 5.1 polish |

MEDIUM (4) + LOW (3) documentados en `deferred-items.md` como Phase 5.1
polish candidates — ninguno bloquea producción ni seguridad.

## Key Decisions Honored (from 05-CONTEXT.md)

D-01 (sidebar fino con PERMISSION_MAP), D-02 (Jinja-only button gating),
D-03 (helper `can` reutiliza lógica de require_permission), D-04 (SMTP
omitido), D-05 (hub propietario-only), D-06 (`ajustes_conexion.html`
nueva), D-07 (Accept-header 403 → HTML redirect + flash, JSON preserva
contract), D-08 (flash cookie `nexo_flash` read-and-clear middleware),
D-09 (`can` registrado como Jinja global import-time, NO `current_user`
global por race condition).

## Human Verification Items (post-return smoke)

Cubiertos estructuralmente por tests automáticos pero requieren eyeball
humano antes de considerar Phase 5 100% cerrada. Status `human_needed`
hasta confirmación.

1. **Login propietario** — sidebar muestra 11 items + divider + Solicitudes con badge (si hay pending). Todos los botones sensibles visibles (Ejecutar pipeline, Regenerar, Borrar informe, Detectar recursos, Add máquina/sección, Delete, Guardar ciclos).
2. **Login directivo ingeniería** (sin producción) — sidebar muestra: Dashboard, Datos, Analisis, Historial, Capacidad, Recursos, Ciclos, BBDD. Sin Operarios. Sin Ajustes. Sin Solicitudes. Botón "Add máquina" visible en /recursos. Botón "Guardar ciclos" visible en /ciclos-calc.
3. **Login usuario rrhh** — sidebar muestra: Dashboard, Datos, Analisis, Historial, Operarios. Sin Pipeline-ejecutable buttons. Sin Recursos ni Ciclos ni BBDD ni Ajustes. Teclear `/bbdd` en URL → 302 redirect a / con toast "No tienes permiso para acceder al explorador de BBDD".
4. **JSON contract preserved** — `curl -H "Accept: application/json" localhost:8001/bbdd` devuelve 302 (HTML-like path) PERO `curl -H "Accept: application/json" localhost:8001/api/algo-protegido` sigue devolviendo 403 JSON con `{detail:...}` (si existe endpoint); `curl -H "HX-Request: true" /bbdd` también devuelve 403 JSON.
5. **Sidebar badge Solicitudes** como propietario — crear una approval pending como usuario → badge muestra (1) en ≤30s sin recargar (HTMX polling).

## Regression Check

- Plan 04 tests preservados: 153 tests core Postgres-only sin cambios.
- Plan 05 tests añadidos: 69 nuevos (27 can_helper + 5 sidebar + 4 flash middleware + 9 forbidden redirect + 6 ajustes split + 10 button gating + 11 HTML GET guarded — incluye test de W-06 label coverage).
- Total full suite: 222 passed / 11 skipped / 3 pre-existing failures (DEF-05-01-A, Phase 4 test isolation en test_recalibrate_*).
- **Zero regresiones introducidas por Phase 5.**

## Deferred Items

- 7 findings MEDIUM/LOW de code review → `deferred-items.md` Phase 5.1.
- UIROL-05 manual smoke (5 items arriba) → pending operator execution.
- DEF-05-01-A (test_recalibrate_* isolation) heredado de Phase 4.

## Verdict

**Phase 5 passed automated verification (5/5 must-haves + 4/5 requirements Complete + HI-01 fixed).**

Status `human_needed` hasta que el operador haga el smoke manual de los
5 items listados arriba. Phase 5 está funcionalmente lista; la
validación humana es confirmatoria, no bloqueante para planificar
Phase 6 (Despliegue LAN HTTPS).
