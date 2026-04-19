# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-18)

**Core value:** Consultar datos reales de producción (MES IZARO) y generar informes OEE fiables por máquina/turno/sección sin bloquear al operario y sin filtrar información entre departamentos.
**Current focus:** Phase 3 — Capa de datos (Sprint 2 de Mark-III) — CONTEXT.md gathered, ready for /gsd-plan-phase 3.

## Current Position

Phase: 3 of 7 — Capa de datos (Wave 1 ✓; 03-02 ✓ con gate diferido; 03-03 pendiente)
Plan: (Phase 2) 4/4 complete ✓; (Phase 3) 2/3 — ✓ 03-01 foundation; 🟡 03-02 capa MES (PDF regression diferido a preprod, deadline 2026-04-26); ☐ 03-03 capa APP+NEXO
Status: 2026-04-19 — Plan 03-02 ejecutado en `--wave 2`. 9 commits del plan + 1 fix out-of-plan (script PDF). Refactor mecánico de 5 routers MES (centro_mando, capacidad, operarios, luk4, bbdd). 12 archivos `.sql` versionados en `nexo/data/sql/mes/`. Cero `import pyodbc` en routers refactorizados (bbdd mantiene pyodbc solo para metadata ops, justificado per D-05). Cero `dbizaro.admuser.*` 3-part names en `api/`. `MesRepository` con 5 métodos como wrapper delgado sobre `OEE/db/connector.py` (D-04). `api/services/pipeline.py` PRISTINO (handoff atómico a 03-03). `pytest tests/data/` 43 passed + tests/auth/ 11 passed/1 skipped. PDF regression check diferido a preprod (baseline se grabó pero se perdió al rebuild; bind-mount `./tests:/app/tests` añadido en compose para persistencia futura). 03-03 pendiente.
Last activity: 2026-04-19 — commits `2d48d99`..`b255c8f` (Plan 03-02), bind-mount + SUMMARY pendiente de commit final.

Progress: [██████░░░░] 35% (2/7 phases completas + Phase 3 ⅔ plans cerrados; 03-03 pendiente)

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

## Deferred Verifications

Verificaciones bloqueantes que se ejecutan fuera de la sesión donde se cerró el plan. Cada entrada tiene deadline; vencido el plazo sin ejecución se re-abre el plan.

| Plan | Verification | Deadline | Operador | Comando |
|------|--------------|----------|----------|---------|
| 03-02 | PDF regression baseline check (success criterion #5) | 2026-04-26 (7d) | e.eguskiza@ecsmobility.com | `docker compose up -d --build web && docker compose exec -T web python scripts/gen_pdf_reference.py --fecha=2026-03-15 && docker compose exec -T web python scripts/pdf_regression_check.py --fecha=2026-03-15`. Aceptación: exit 0 (idéntico) o exit 1 + visual check OK. Bind-mount `./tests:/app/tests` añadido en `docker-compose.yml` para que el baseline persista. |

## Session Continuity

Last session: 2026-04-19 — `/gsd-execute-phase 3 --wave 2` (Plan 03-02 cerrado con gate diferido).
Stopped at: Plan 03-02 ✓ con 9 commits del plan + 1 fix out-of-plan + bind-mount + SUMMARY (commit final pendiente). 5 routers MES refactorizados, 12 `.sql` versionados, MesRepository operativo. 03-03 pendiente — refactor de 3 routers APP (`historial`, `recursos`, `ciclos`), repos APP (RecursoRepo, CicloRepo, EjecucionRepo, MetricaRepo, LukRepo, ContactoRepo) + repos NEXO (UserRepo, RoleRepo, AuditRepo), migración de `nexo/services/auth.py` + `api/routers/{auditoria,usuarios}.py`, edit atómico de `api/services/pipeline.py` (Task 4.7).
Resume file: `.planning/phases/03-capa-de-datos/03-03-PLAN.md`. Siguiente opciones: (a) `/gsd-execute-phase 3 --wave 2` retoma con 03-03 (autonomous, sin checkpoints humanos), o (b) ejecutar regression check del 03-02 pendiente cuando estés en preprod (ver "Deferred Verifications").
