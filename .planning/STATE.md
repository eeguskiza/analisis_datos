# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-18)

**Core value:** Consultar datos reales de producción (MES IZARO) y generar informes OEE fiables por máquina/turno/sección sin bloquear al operario y sin filtrar información entre departamentos.
**Current focus:** Phase 3 — Capa de datos (Sprint 2 de Mark-III) — CONTEXT.md gathered, ready for /gsd-plan-phase 3.

## Current Position

Phase: 3 of 7 — Capa de datos (CONTEXT.md gathered, PLAN.md pendiente)
Plan: (Phase 2) 4/4 complete ✓; (Phase 3) 0/3 previstos (foundation + capa MES + capa APP+NEXO)
Status: 2026-04-19 — /gsd-discuss-phase 3 ejecutado. 4 gray areas aceptadas en bloque por el operador ("elige tu"): SQL loader + `.sql` layout (text+`:named`, un `.sql` por método, `bindparam` expanding), repository shape (sesión inyectada, DTOs `*Row` frozen en `nexo/data/dto/`, repos sin transacción), plan breakdown (03-01 foundation → 03-02 MES ∥ 03-03 APP+NEXO), pipeline+bbdd (wrapper delgado sobre `OEE/db/connector.py` + whitelist queda en `bbdd.py`). 11 REQ-IDs mapeados a los 3 plans.
Last activity: 2026-04-19 — commit `5111cfe` con `03-CONTEXT.md` + `03-DISCUSSION-LOG.md`.

Progress: [█████░░░░░] 29% (2/7 phases completas + CONTEXT.md de Phase 3 listo)

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

Last session: 2026-04-19 — `/gsd-discuss-phase 3` (CONTEXT.md para capa de datos).
Stopped at: Phase 3 CONTEXT gathered. 4 gray areas aceptadas en bloque. 11 REQ-IDs DATA-01..DATA-11 mapeados a plans 03-01 (foundation), 03-02 (MES), 03-03 (APP+NEXO). Commit: `5111cfe`.
Resume file: `.planning/phases/03-capa-de-datos/03-CONTEXT.md`. Siguiente: `/gsd-plan-phase 3`.
