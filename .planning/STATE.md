# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-18)

**Core value:** Consultar datos reales de producción (MES IZARO) y generar informes OEE fiables por máquina/turno/sección sin bloquear al operario y sin filtrar información entre departamentos.
**Current focus:** Phase 1 — Naming + Higiene + CI (Sprint 0 de Mark-III)

## Current Position

Phase: 1 of 7 (Naming + Higiene + CI)
Plan: 0 of 1 in current phase
Status: Ready to execute (PLAN.md a generar en `/gsd-execute-phase 1` o manualmente desde el brief del Sprint 0)
Last activity: 2026-04-18 — Scaffolding de `.planning/` derivado de `docs/` completado (Modo C)

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: n/a
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: n/a
- Trend: n/a (no runs yet)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisiones clave en `.planning/PROJECT.md` (sección "Key Decisions"). 5 decisiones
bloqueantes de `docs/OPEN_QUESTIONS.md` resueltas + modelo de auth definido +
reorden de sprints confirmado.

Lagunas conocidas (ver bloque "Session Continuity" y la respuesta de sesión):
- MCP server en compose de producción: default = mantener, pendiente confirmación explícita
- Timeline / hito duro: no definido; operador marca ritmo (10-12 semanas naturales estimadas)

### Pending Todos

None yet. (Ideas futuras que surjan durante la ejecución se capturan con `/gsd-add-todo`.)

### Blockers/Concerns

- **SMTP roto**: `data/db_config.json` no tiene sección `smtp`; envío por email no funciona hoy. No bloquea Mark-III (SMTP es Out of Scope); sí bloquea si alguien intenta usar email durante ejecución.
- **`data/oee.db` SQLite residual**: decisión pendiente en Phase 1 (snapshot o borrar). Parte del commit 4 del Sprint 0.

## Deferred Items

Items explícitamente diferidos o pendientes de decisión posterior:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Security | Rotación de credenciales SQL Server | Pending post-audit | 2026-04-18 (Sprint 0 brief) |
| Security | Reescritura de historial git (`filter-repo`) | Pending operator review | 2026-04-18 (Sprint 0 brief) |
| Infra | Configuración SMTP operativa | Deferred (Out of Scope Mark-III) | 2026-04-18 |
| Infra | Compra de dominio público / exposición internet | Out of Scope (decisión de producto) | 2026-04-18 |
| Auth | 2FA | Deferred Mark-IV | 2026-04-18 |
| Auth | LDAP / Active Directory | Deferred Mark-IV | 2026-04-18 |
| Refactor | Rename carpeta `OEE/` → `modules/oee/` | Deferred Sprint 2 (Phase 3) | 2026-04-18 |
| Refactor | Rename repo GitHub `analisis_datos` | Deferred (coste alto, beneficio cosmético) | 2026-04-18 |
| Tooling | Comando GSD para regenerar `.planning/` desde `docs/` | Sin comando nativo; gap documentado en CLAUDE.md | 2026-04-18 |

## Session Continuity

Last session: 2026-04-18 14:XX
Stopped at: Scaffolding de `.planning/` completado (PROJECT.md, REQUIREMENTS.md, ROADMAP.md, STATE.md). Siguiente paso: commit del scaffolding, luego arranque de Sprint 0 vía `/gsd-execute-phase 1`.
Resume file: None (sesión activa)
