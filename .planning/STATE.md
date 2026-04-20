---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: "Phase 04 en curso. Plan 04-02 ✓ 2026-04-20 — 6 commits del plan. Preflight + middleware + asyncio.to_thread operativos; 4 routers (pipeline/bbdd/capacidad/operarios) gatean por level. Modal AMBER/RED en pipeline.html y bbdd.html. Bug resuelto: middleware leía request.state antes de call_next → fix para leer después. Tests: 151 pass / 22 skip / 1 xfail (04-03) / 0 fail. Siguiente: Plan 04-03 (approvals flow) ó Plan 04-04 (observability UI)."
last_updated: "2026-04-20T07:45:00.000Z"
last_activity: 2026-04-20 -- Plan 04-02 completo (preflight + middleware + asyncio.to_thread + modal frontend)
progress:
  total_phases: 7
  completed_phases: 3
  total_plans: 12
  completed_plans: 10
  percent: 83
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-18)

**Core value:** Consultar datos reales de producción (MES IZARO) y generar informes OEE fiables por máquina/turno/sección sin bloquear al operario y sin filtrar información entre departamentos.
**Current focus:** Phase 04 — consultas-pesadas

## Current Position

Phase: 04 (consultas-pesadas) — EXECUTING
Plan: 3 of 4 (04-01 ✓ 2026-04-20, 04-02 ✓ 2026-04-20, 04-03 next)
Status: Wave 2 complete — 04-02 preflight + middleware + asyncio.to_thread + modal frontend operativos. 151 tests green (28 nuevos), 1 xfail aguardando Plan 04-03 consume_approval.
Last activity: 2026-04-20 -- Plan 04-02 cerrado (preflight + middleware + modal frontend)

Progress: [██████████] 43% (3/7 phases completas) — Phase 4 en curso (2/4 plans)

## Plans de Phase 2 (estado)

- [x] 02-01 — schema-engine-bootstrap ✓ 2026-04-18
- [x] 02-02 — auth-middleware-login ✓ 2026-04-19
- [x] 02-03 — rbac-retrofit-routers ✓ 2026-04-19
- [x] 02-04 — audit-middleware-ajustes-ui ✓ 2026-04-19

## Performance Metrics

**Velocity:**

- Total plans completed in current session: 1 (Plan 04-02)
- Average duration: ~1h (execute-plan sequential agent)
- Total execution time Plan 04-02: ~60 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Naming + Higiene + CI | 1/1 | ~3h | ~3h |
| 3. Capa de datos (03-03 APP+NEXO) | 1/1 | ~17 min | ~17 min |
| 4. Consultas pesadas (04-02 preflight+middleware+asyncio) | 1/1 | ~60 min | ~60 min |

**Recent Trend:**

- Last 5 plans: [Phase 1 / Plan 01-01 ~3h, Phase 3 / Plan 03-03 ~17 min, Phase 4 / Plan 04-01 ~3h, Phase 4 / Plan 04-02 ~60 min]
- Trend: Phase 4 plans promedio ~2h incluyendo Wave 0 tests + DB bootstrap

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

Last session: 2026-04-20 — `/gsd-execute-phase 4` (Plan 04-02 cerrado sequential autonomous; Wave 2 completa).
Stopped at: Phase 4 en curso. Plans 04-01 + 04-02 completos. Plan 04-02 entregó preflight + middleware + asyncio.to_thread + modal frontend en 6 commits. 151 tests green / 22 skip / 1 xfail (04-03 awaiting) / 0 fail. Siguiente: Plan 04-03 (approval flow, nexo/services/approvals.py + /api/approvals/* + /mis-solicitudes + /ajustes/solicitudes) en paralelo, luego Plan 04-04 (observability UI + LISTEN/NOTIFY + learning).
Resume file: `.planning/phases/04-consultas-pesadas/04-02-SUMMARY.md`. Siguientes pasos recomendados: (a) `/gsd-execute-phase 4` para Plan 04-03 approvals flow (puede arrancar en paralelo con 04-02 mergeado), (b) smoke manual pendiente de Task 7 (checkpoint auto-aprobado durante ejecución no-supervisada): modal amber E2E, UI no-congela, URL approval_id re-dispatch.
