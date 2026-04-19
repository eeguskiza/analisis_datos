# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-18)

**Core value:** Consultar datos reales de producción (MES IZARO) y generar informes OEE fiables por máquina/turno/sección sin bloquear al operario y sin filtrar información entre departamentos.
**Current focus:** Phase 3 CERRADA (Sprint 2 Mark-III completo) — ready for `/gsd-verify-work 3` + `/gsd-plan-phase 4`.

## Current Position

Phase: 3 of 7 — Capa de datos ✓ CERRADA (los 3 plans cumplidos)
Plan: (Phase 2) 4/4 ✓; (Phase 3) 3/3 ✓ — 03-01 foundation + 03-02 capa MES + 03-03 capa APP+NEXO
Status: 2026-04-19 — Plan 03-03 ejecutado autonomous. 12 commits atómicos del plan + 1 fix Rule-1 (SQL Server compat RecursoRepo.list_activos + EjecucionRow date coercion). 9 repos creados (6 APP + 3 NEXO), 13 DTOs frozen (9 APP + 4 NEXO), 5 routers refactorizados (ciclos, recursos, historial, auditoria, usuarios), auth.py delegado a UserRepo, pipeline.py edit atómico Opción B (swap imports ORM al path canónico nexo.data.models_app). Modelos ORM migrados a `nexo/data/models_{app,nexo}.py` con shims transicionales (`api/database.py` re-exporta; `nexo/db/models.py` shim). `pytest tests/data/` + `tests/auth/` + OEE safety net = 109 passed / 11 skipped. AST-based meta-test detecta 0 `import pyodbc` en routers (bbdd.py documentado como excepción D-05). DATA-05 D-01 compliance: 0 archivos `.sql` APP/NEXO (ORM puro = representación canónica). 
Last activity: 2026-04-19 — commits `d4a2a5f`..`2ccec36` (Plan 03-03), SUMMARY + este STATE pendientes de commit final.

Progress: [██████████] 43% (3/7 phases completas) — Phase 3 CERRADA

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
| 3. Capa de datos (03-03 APP+NEXO) | 1/1 | ~17 min | ~17 min |

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

Last session: 2026-04-19 — `/gsd-execute-phase 3` (Plan 03-03 cerrado autonomous; Phase 3 CERRADA).
Stopped at: Phase 3 completa. 3 plans ejecutados (03-01 ✓ 2026-04-18, 03-02 ✓ 2026-04-19 con gate diferido, 03-03 ✓ 2026-04-19). Total 29 commits en Phase 3 (5 + 10 + 13 + 1 closing). 9 repos operativos (6 APP + 3 NEXO). 5 routers APP/NEXO refactorizados a repos. `api/services/pipeline.py` tocado una sola vez en Task 4.7 (atómico, Opción B). SUMMARY + STATE + ROADMAP + final commit pendientes. Blocker abierto: solo el PDF regression check de 03-02 (deadline 2026-04-26 en preprod).
Resume file: `.planning/phases/03-capa-de-datos/03-03-SUMMARY.md`. Siguientes pasos recomendados: (a) `/gsd-verify-work 3` para validación integral de Phase 3, (b) `/gsd-plan-phase 4` para abrir Sprint 3 (consultas pesadas — preflight/postflight/umbrales), (c) ejecutar PDF regression check del 03-02 cuando estés en preprod (ver "Deferred Verifications").
