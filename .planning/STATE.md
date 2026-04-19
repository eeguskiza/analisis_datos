# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-18)

**Core value:** Consultar datos reales de producción (MES IZARO) y generar informes OEE fiables por máquina/turno/sección sin bloquear al operario y sin filtrar información entre departamentos.
**Current focus:** Phase 3 — Capa de datos (Sprint 2 de Mark-III) — CONTEXT.md gathered, ready for /gsd-plan-phase 3.

## Current Position

Phase: 3 of 7 — Capa de datos (Wave 1 ✓; Wave 2 pendiente)
Plan: (Phase 2) 4/4 complete ✓; (Phase 3) 1/3 — ✓ 03-01 foundation; ☐ 03-02 capa MES; ☐ 03-03 capa APP+NEXO
Status: 2026-04-19 — Plan 03-01 ejecutado autonomously (4 commits atómicos + 1 docs). Hard gate 11/11 PASS: engines+shim, schema_guard wired en lifespan, loader con lru_cache, DTOs frozen base, fixtures `db_nexo`/`db_app`/`engine_mes_mock`, Makefile `test-data`. `pytest tests/data/` 16 passed; `pytest tests/auth/` 11 passed/1 skipped (IDENT-06 verde). Requirements cubiertos: DATA-01, DATA-05, DATA-06, DATA-08, DATA-10, DATA-11. Wave 2 (03-02 MES + 03-03 APP+NEXO) pausado por decisión del operador — necesita acceso a SQL Server preprod para PDF baseline en 03-02 antes de continuar.
Last activity: 2026-04-19 — commits `4325943` (engines), `bb6cd62` (loader+schema_guard+DTO), `9c81227` (lifespan+deps+Makefile), `5c70f0b` (tests/data), `1403da5` (SUMMARY).

Progress: [█████░░░░░] 31% (2/7 phases completas + Phase 3 Wave 1 ✓; Wave 2 pendiente)

## Plans de Phase 2 (estado)

- [x] 02-01 — schema-engine-bootstrap ✓ 2026-04-18
- [x] 02-02 — auth-middleware-login ✓ 2026-04-19
- [x] 02-03 — rbac-retrofit-routers ✓ 2026-04-19
- [x] 02-04 — audit-middleware-ajustes-ui ✓ 2026-04-19

## Performance Metrics

**Velocity:**
- Total plans completed: 1
- Average duration: ~3h (sesión única, interactive mode)
- Total execution time: ~3.0 horas

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Naming + Higiene + CI | 1/1 | ~3h | ~3h |

**Recent Trend:**
- Last 5 plans: [Phase 1 / Plan 01-01 ~3h]
- Trend: n/a (1 sola muestra)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisiones clave en `.planning/PROJECT.md` (sección "Key Decisions"). 5 decisiones
bloqueantes de `docs/OPEN_QUESTIONS.md` resueltas + modelo de auth definido +
reorden de sprints confirmado + **rotación de password SA ejecutada** durante Sprint 0.

Ver `docs/SECURITY_AUDIT.md` para el cierre de H1/H2.

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

## Session Continuity

Last session: 2026-04-19 — `/gsd-execute-phase 3 --auto --no-transition` (Wave 1 only por elección del operador).
Stopped at: Plan 03-01 cerrado con 5 commits y SUMMARY. `nexo/data/` package operativo con engines (3), loader, schema_guard wired en lifespan, DTOs base, fixtures de tests. Wave 1 verde. Wave 2 (03-02 + 03-03) en reposo hasta que el operador tenga acceso a SQL Server preprod para grabar PDF baseline (gate del PDF regression check, success criterion #5).
Resume file: `.planning/phases/03-capa-de-datos/03-02-PLAN.md`. Siguiente: `/gsd-execute-phase 3 --wave 2` cuando estés en preprod (ejecuta 03-02 y 03-03 en paralelo).
