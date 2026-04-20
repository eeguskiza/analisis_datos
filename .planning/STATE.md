---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: "Phase 04 CERRADA. Plan 04-04 ✓ 2026-04-20 — 5 commits del plan. Observability UI + LISTEN/NOTIFY + learning: listener real (_blocking_listen_forever + start_listener en lifespan) + /ajustes/limites CRUD (PUT + recalibrate con factor_learning helper + NOTIFY propagation) + /ajustes/rendimiento (tabla summary + Chart.js timeseries + fallback) + query_log_cleanup (Mon 03:00 + NEXO_QUERY_LOG_RETENTION_DAYS) + factor_auto_refresh (1er Mon del mes 03:10 + NEXO_AUTO_REFRESH_STALE_DAYS) + 2 env vars + 19 tests nuevos (9 thresholds_cache + 9 thresholds_crud + 1 listen_notify E2E). Tests: 173 pass / 28 skip / 0 fail (+4 deselected SQL Server infra pre-existing). Fix Rule 1: notify_changed ahora usa psycopg2 dedicado (no pool SQLAlchemy) para evitar polución de isolation_level. Siguiente: Phase 5 (UI por roles)."
last_updated: "2026-04-20T15:00:00.000Z"
last_activity: 2026-04-20 -- Plan 04-04 completo (LISTEN/NOTIFY real + UI limites/rendimiento + cleanup jobs + Phase 4 cerrada)
progress:
  total_phases: 7
  completed_phases: 4
  total_plans: 12
  completed_plans: 12
  percent: 57
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-18)

**Core value:** Consultar datos reales de producción (MES IZARO) y generar informes OEE fiables por máquina/turno/sección sin bloquear al operario y sin filtrar información entre departamentos.
**Current focus:** Phase 04 — consultas-pesadas

## Current Position

Phase: 04 (consultas-pesadas) — COMPLETE ✓ 2026-04-20
Plan: 4 of 4 ALL COMPLETE (04-01 ✓, 04-02 ✓, 04-03 ✓, 04-04 ✓ — todos 2026-04-20)
Status: **Phase 4 cerrada**. Wave 4 entregó observability UI + LISTEN/NOTIFY real + learning + Phase 4 complete. 173 tests green (+19 nuevos vs 04-03) / 28 skipped / 4 SQL Server infra pre-existing deselected. Siguiente: Phase 5 (UI por roles) — no planificada aún, requiere `/gsd-plan-phase 5`.
Last activity: 2026-04-20 -- Plan 04-04 cerrado (LISTEN/NOTIFY listener + /ajustes/limites + /ajustes/rendimiento + cleanup jobs + factor_auto_refresh)

Progress: [████████░░] 57% (4/7 phases completas) — Phase 4 cerrada con 4/4 plans

## Plans de Phase 2 (estado)

- [x] 02-01 — schema-engine-bootstrap ✓ 2026-04-18
- [x] 02-02 — auth-middleware-login ✓ 2026-04-19
- [x] 02-03 — rbac-retrofit-routers ✓ 2026-04-19
- [x] 02-04 — audit-middleware-ajustes-ui ✓ 2026-04-19

## Performance Metrics

**Velocity:**

- Total plans completed in current session: 3 (Plan 04-02, Plan 04-03, Plan 04-04)
- Average duration: ~75 min (execute-plan sequential agent)
- Total execution time Plan 04-04: ~110 min (listener + 2 routers + 2 templates + 2 services + 19 tests + infra bug fix)

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Naming + Higiene + CI | 1/1 | ~3h | ~3h |
| 3. Capa de datos (03-03 APP+NEXO) | 1/1 | ~17 min | ~17 min |
| 4. Consultas pesadas (04-02 preflight+middleware+asyncio) | 1/1 | ~60 min | ~60 min |
| 4. Consultas pesadas (04-03 approvals + scheduler + badge) | 1/1 | ~55 min | ~55 min |
| 4. Consultas pesadas (04-04 observability + LISTEN/NOTIFY + learning) | 1/1 | ~110 min | ~110 min |

**Recent Trend:**

- Last 5 plans: [Phase 3 / Plan 03-03 ~17 min, Phase 4 / Plan 04-01 ~3h, Phase 4 / Plan 04-02 ~60 min, Phase 4 / Plan 04-03 ~55 min, Phase 4 / Plan 04-04 ~110 min]
- Trend: Phase 4 plans promedio ~90 min; foundation (04-01) y último (04-04, observability + tests E2E) más largos; plans intermedios ~55-60 min.

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

**Plan 04-04 decisions implementadas (2026-04-20):**

- D-04: botón manual "Recalcular" en /ajustes/limites + helper compartido `nexo/services/factor_learning.compute_factor` (median con outlier filter >500ms, min 10 samples). Para pipeline/run usa `actual_ms / (n_recursos × n_dias)`; para otros usa `median(actual_ms)`.
- D-10: query_log_cleanup Monday 03:00 UTC + NEXO_QUERY_LOG_RETENTION_DAYS=90 (0=forever). Audit log con path='__cleanup_query_log__'.
- D-11: /ajustes/rendimiento página dedicada (propietario-only) con filtros endpoint/user/status/rango + tabla summary (divergencia color-coded) + Chart.js canvas.
- D-12: Chart.js vía CDN + fallback tabla si CDN cae (`typeof Chart === 'undefined'` check inline) + `maintainAspectRatio: false` responsive.
- D-19 complete: `_blocking_listen_forever` con psycopg2 conn dedicado AUTOCOMMIT + `LISTEN nexo_thresholds_changed` + `select()` loop 5s. Wrappeado en `asyncio.to_thread` dentro del lifespan. Reconexión automática backoff 5s. Safety-net 5min de 04-01 sigue operativo como 2a defensa.
- D-20: factor_auto_refresh 1er Monday del mes 03:10 UTC (filter `now.day <= 7`) si factor_updated_at > NEXO_AUTO_REFRESH_STALE_DAYS=60 días. Reusa compute_factor (DRY con recalibrate manual). Audit log con path='__auto_refresh__'.
- Rule 1 fix: `thresholds_cache.notify_changed` usaba `engine_nexo.raw_connection()` + `set_isolation_level(AUTOCOMMIT)` que devolvía la conexión al pool SQLAlchemy con nivel alterado → siguientes tests con `yield_per=500` fallaban con "can't use a named cursor outside of transactions". Fix: psycopg2.connect() dedicado, descartado tras NOTIFY. No toca el pool.

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

Last session: 2026-04-20 — `/gsd-execute-phase 4` (Plan 04-04 cerrado sequential autonomous; Wave 4 completa + Phase 4 cerrada).
Stopped at: **Phase 4 CERRADA (4/4 plans)**. Plan 04-04 entregó observability + hot-reload completo: listener LISTEN/NOTIFY real (`_blocking_listen_forever` + `start_listener` cableado en lifespan) + `/ajustes/limites` CRUD (PUT + recalibrate manual con factor_learning helper + NOTIFY propagation) + `/ajustes/rendimiento` (filtros + tabla summary + Chart.js timeseries + fallback CDN) + `query_log_cleanup` (Mon 03:00 UTC) + `factor_auto_refresh` (1er Mon del mes 03:10 UTC, filter `day <= 7`) + 2 env vars nuevas (NEXO_QUERY_LOG_RETENTION_DAYS=90, NEXO_AUTO_REFRESH_STALE_DAYS=60) + 19 tests nuevos (9 thresholds_cache integration + 9 thresholds_crud + 1 listen_notify E2E).
Tests: 173 pass / 28 skip / 0 fail (+4 deselected SQL Server infra pre-existing). Rule 1 fix: `notify_changed` usaba `engine_nexo.raw_connection()` lo que polucionaba el pool SQLAlchemy con isolation_level=AUTOCOMMIT; fix con psycopg2.connect() dedicado que no toca el pool.
Resume file: `.planning/phases/04-consultas-pesadas/04-04-SUMMARY.md`. Siguientes pasos recomendados: (a) `/gsd-verify-work 4` — verificación final de Phase 4; (b) `/gsd-plan-phase 5` — planificar Phase 5 (UI por roles, Sprint 4); (c) smoke manual pendiente del Task 6 de 04-04 (auto-aprobado): Chart.js render + LISTEN <1s (2 pestañas) + modal red E2E completo + scheduler 3 jobs al arrancar.
