---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: "Phase 04 casi cerrada. Plan 04-03 ✓ 2026-04-20 — 5 commits del plan. Approval flow end-to-end: service layer (6 funciones) + router (7 endpoints) + 2 templates + badge sidebar HTMX + scheduler asyncio + job cleanup Mon 03:05 + TTL 7d. PC-04-07 xfail removido (test_run_red_with_valid_approval_executes ahora green end-to-end). Tests: 174 pass / 18 skip / 0 xfail / 0 fail (+23 vs baseline 151). Siguiente: Plan 04-04 (observability UI + LISTEN/NOTIFY + learning)."
last_updated: "2026-04-20T10:30:00.000Z"
last_activity: 2026-04-20 -- Plan 04-03 completo (approval flow + cleanup scheduler + badge HTMX + PC-04-07 xfail removido)
progress:
  total_phases: 7
  completed_phases: 3
  total_plans: 12
  completed_plans: 11
  percent: 92
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-18)

**Core value:** Consultar datos reales de producción (MES IZARO) y generar informes OEE fiables por máquina/turno/sección sin bloquear al operario y sin filtrar información entre departamentos.
**Current focus:** Phase 04 — consultas-pesadas

## Current Position

Phase: 04 (consultas-pesadas) — EXECUTING
Plan: 4 of 4 (04-01 ✓ 2026-04-20, 04-02 ✓ 2026-04-20, 04-03 ✓ 2026-04-20, 04-04 next)
Status: Wave 3 complete — 04-03 approval flow end-to-end + scheduler asyncio + badge HTMX. 174 tests green (+23 nuevos), 0 xfails. Import-guard _APPROVALS_AVAILABLE=True activado en 4 routers. Siguiente: Plan 04-04 (observability UI + LISTEN/NOTIFY real + learning).
Last activity: 2026-04-20 -- Plan 04-03 cerrado (approvals + scheduler + badge + PC-04-07 xfail removido)

Progress: [██████████] 43% (3/7 phases completas) — Phase 4 en curso (3/4 plans)

## Plans de Phase 2 (estado)

- [x] 02-01 — schema-engine-bootstrap ✓ 2026-04-18
- [x] 02-02 — auth-middleware-login ✓ 2026-04-19
- [x] 02-03 — rbac-retrofit-routers ✓ 2026-04-19
- [x] 02-04 — audit-middleware-ajustes-ui ✓ 2026-04-19

## Performance Metrics

**Velocity:**

- Total plans completed in current session: 2 (Plan 04-02, Plan 04-03)
- Average duration: ~55 min (execute-plan sequential agent)
- Total execution time Plan 04-03: ~55 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Naming + Higiene + CI | 1/1 | ~3h | ~3h |
| 3. Capa de datos (03-03 APP+NEXO) | 1/1 | ~17 min | ~17 min |
| 4. Consultas pesadas (04-02 preflight+middleware+asyncio) | 1/1 | ~60 min | ~60 min |
| 4. Consultas pesadas (04-03 approvals + scheduler + badge) | 1/1 | ~55 min | ~55 min |

**Recent Trend:**

- Last 5 plans: [Phase 3 / Plan 03-03 ~17 min, Phase 4 / Plan 04-01 ~3h, Phase 4 / Plan 04-02 ~60 min, Phase 4 / Plan 04-03 ~55 min]
- Trend: Phase 4 plans promedio ~90 min; foundation (04-01) la más larga por Wave 0 tests + DB bootstrap; plans secundarios ~55-60 min.

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisiones clave en `.planning/PROJECT.md` (sección "Key Decisions"). 5 decisiones
bloqueantes de `docs/OPEN_QUESTIONS.md` resueltas + modelo de auth definido +
reorden de sprints confirmado + **rotación de password SA ejecutada** durante Sprint 0.

Ver `docs/SECURITY_AUDIT.md` para el cierre de H1/H2.

**Plan 04-02 decisions implementadas (2026-04-20):**

- D-01..D-04: umbrales consumidos desde cache (seed de 04-01).
- D-05/D-06/D-07: modales AMBER/RED bloqueantes con textos literales en `templates/pipeline.html` + `templates/bbdd.html`.
- D-17: slow status cuando actual_ms > warn_ms*1.5 + log.warning con ratio.
- D-18: semáforo(3) + timeout 900s soft + landmine documentado.
- Bug fixed: middleware leía request.state antes de `await call_next` (siempre None). Fix: leer state DESPUÉS de call_next. Patrón documentado como comentario explícito en query_timing.py.

**Plan 04-03 decisions implementadas (2026-04-20):**

- D-06: modal RED → POST /api/approvals (endpoint ahora existe, frontend 04-02 se activa).
- D-13: badge HTMX sidebar 30s (hx-get /api/approvals/count) sin email/banner.
- D-14: TTL 7d + Monday 03:05 UTC cleanup via cleanup_scheduler asyncio + histórico 30d.
- D-15: CAS single-use con `_canonical_json(sort_keys=True)` garantiza equality params_json; consume_approval traduce None en HTTPException(403) con 5 mensajes diagnósticos específicos.
- D-16: owner-cancel con ownership check server-side (ApprovalRepo.cancel devuelve False si user != dueño).
- PC-04-02: link "Ejecutar ahora" en /mis-solicitudes con `?approval_id={{ s.id }}` por endpoint.
- PC-04-07: xfail removido de test_run_red_with_valid_approval_executes; test extendido con create+approve+run end-to-end.
- Deviation: rate limiter slowapi 20/min en /login acumulaba 429 entre módulos; fix con `limiter.reset()` en autouse fixtures de los 3 test suites que hacen login.

### Pending Todos

None yet. (Ideas futuras que surjan durante la ejecución se capturan con `/gsd-add-todo`.)

### Blockers/Concerns

- **SMTP roto**: `data/db_config.json` no tiene sección `smtp`; envío por email no funciona hoy. No bloquea Mark-III (SMTP es Out of Scope); sí bloquea si alguien intenta usar email durante ejecución.
- **`pandas==3.0.2` pineado**: la mayor 3.x tiene cambios de API frente a 2.x. `api/services/pipeline.py` y módulos OEE no se han re-validado explícitamente con 3.x. Verificar en Sprint 2 o al primer run del CI.
- **`ruff check .` sin `pyproject.toml`** configurado: CI usa defaults. Configurar reglas en Sprint 6.
- **Tests pytest no ejecutados** durante el sprint. CI los corre pero no bloquean. Primera ejecución real tras push.

## Deferred Items

Items explícitamente diferidos o pendientes de decisión posterior:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Security | Rotación de credenciales SQL Server | ✓ Ejecutada 2026-04-18 (durante Sprint 0) | - |
| Security | Reescritura de historial git (`filter-repo`) | Deferred — credencial muerta, limpieza cosmética | 2026-04-18 |
| Infra | Configuración SMTP operativa | Deferred (Out of Scope Mark-III) | 2026-04-18 |
| Infra | Compra de dominio público / exposición internet | Out of Scope (decisión de producto) | 2026-04-18 |
| Auth | 2FA | Deferred Mark-IV | 2026-04-18 |
| Auth | LDAP / Active Directory | Deferred Mark-IV | 2026-04-18 |
| Refactor | Rename carpeta `OEE/` → `modules/oee/` | Deferred Sprint 2 (Phase 3) | 2026-04-18 |
| Refactor | Rename repo GitHub `analisis_datos` | Deferred (coste alto, beneficio cosmético) | 2026-04-18 |
| Refactor | Unificar `data/ecs-logo.png` con `static/img/brand/ecs/logo.png` | Deferred Sprint 2 (Phase 3) | 2026-04-18 |
| Tooling | Comando GSD para regenerar `.planning/` desde `docs/` | Sin comando nativo; conversacional via CLAUDE.md | 2026-04-18 |

## Deferred Verifications

Verificaciones bloqueantes que se ejecutan fuera de la sesión donde se cerró el plan. Cada entrada tiene deadline; vencido el plazo sin ejecución se re-abre el plan.

| Plan | Verification | Deadline | Operador | Comando |
|------|--------------|----------|----------|---------|
| 03-02 | PDF regression baseline check (success criterion #5) | 2026-04-26 (7d) | e.eguskiza@ecsmobility.com | `docker compose up -d --build web && docker compose exec -T web python scripts/gen_pdf_reference.py --fecha=2026-03-15 && docker compose exec -T web python scripts/pdf_regression_check.py --fecha=2026-03-15`. Aceptación: exit 0 (idéntico) o exit 1 + visual check OK. Bind-mount `./tests:/app/tests` añadido en `docker-compose.yml` para que el baseline persista. |

## Session Continuity

Last session: 2026-04-20 — `/gsd-execute-phase 4` (Plan 04-03 cerrado sequential autonomous; Wave 3 completa).
Stopped at: Phase 4 casi cerrada (3/4 plans). Plan 04-03 entregó approval flow end-to-end: service (6 funciones) + router (7 endpoints) + templates (ajustes_solicitudes + mis_solicitudes) + badge HTMX 30s + cleanup_scheduler asyncio (Mon 03:05 UTC) + job approvals_cleanup + TTL 7d. 174 tests green / 18 skip / 0 xfail / 0 fail (+23 vs baseline 151). PC-04-07 xfail removido; test_run_red_with_valid_approval_executes ahora ejerce el flujo completo end-to-end.
Resume file: `.planning/phases/04-consultas-pesadas/04-03-SUMMARY.md`. Siguientes pasos recomendados: (a) `/gsd-execute-phase 4` para Plan 04-04 (último de la phase: observability UI /ajustes/rendimiento + LISTEN/NOTIFY listener real + /ajustes/limites CRUD + factor_auto_refresh + query_log cleanup); (b) smoke manual pendiente del Task 6 de 04-03 (auto-aprobado): 2-browser owner+user concurrent flow + verificación ocular de badge HTMX refresh.
