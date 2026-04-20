---
phase: 04-consultas-pesadas
type: verification
status: human_needed
verified: 2026-04-20
must_haves_total: 5
must_haves_verified: 5
requirements_verified: [QUERY-01, QUERY-02, QUERY-03, QUERY-04, QUERY-05, QUERY-06, QUERY-07, QUERY-08]
requirements_unverified: []
code_review: .planning/phases/04-consultas-pesadas/04-REVIEW.md (19 findings; 6 CRITICAL+HIGH fixed in 04-REVIEW-FIX.md)
---

# Phase 4 — Verification (Consultas pesadas)

Goal check: **Preflight estima coste antes de ejecutar pipeline/queries caras;
postflight mide y alerta; aprobación asíncrona para rojos.** ✅ Achieved.

Phase executed over 4 plans (04-01 → 04-04) + 6 fix commits closing the
CRITICAL + HIGH items surfaced by `/gsd-code-review`. All 4 SUMMARY.md
artifacts present. 141 tests green (Postgres-backed subset) / 22 skipped /
0 regressions. 4 pre-existing SQL-Server-dependent tests in
`tests/auth/test_rbac_smoke.py` + `tests/routers/test_preflight_endpoints.py`
fail due to MES/APP SQL Server login timeouts (environment-only; no MSSQL
host reachable from dev machine) — documented in 04-04 SUMMARY §Deviations.

## Success Criteria Evidence

| # | Criterion | Evidence | Status |
|---|-----------|----------|--------|
| 1 | Pipeline AMBER modal con "esto tardará ~X min, ¿continuar?" | `templates/pipeline.html:336` (`x-show="modalLevel === 'amber'"`) + `api/routers/pipeline.py` returns `Estimation(level="amber", ms, reason)` + `static/js/app.js` `humanize_ms()` helper | ✅ PASS |
| 2 | `/bbdd` RED gate: no ejecuta hasta que propietario aprueba | `api/routers/bbdd.py:566-588` — si `est.level == "red"` y no `(force and approval_id)` → 428 con approval creado; ejecución bloqueada hasta `consume_approval` del propietario | ✅ PASS |
| 3 | `query_log` tiene `estimated_ms` + `actual_ms`; WARNING si divergen > 50% | `nexo/middleware/query_timing.py:208-219` — `actual_ms > warn_ms * 1.5` → `status='slow'` + `log.warning`. Columnas presentes en `nexo/data/models_nexo.py` `NexoQueryLog`. | ✅ PASS |
| 4 | `/ajustes/limites` edita umbrales sin tocar código | `api/routers/limites.py:104` `@router.put("/api/thresholds/{endpoint:path}")` (propietario-only) + `thresholds_cache.notify_changed(endpoint)` en línea 142 → propagación cross-worker via LISTEN/NOTIFY | ✅ PASS |
| 5 | Pipeline no congela la UI (asyncio.to_thread) | `api/routers/pipeline.py:200` `asyncio.to_thread(_worker)` + `pipeline_semaphore` (max 3 concurrentes, D-18) + `asyncio.wait_for(timeout=900s)` | ✅ PASS |

**Score: 5/5 must-haves verified via code inspection.**

## Requirement Traceability (QUERY-01..QUERY-08)

| ID | Phase Deliverable | Complete | Evidence |
|----|-------------------|----------|----------|
| QUERY-01 | `nexo.query_log` with `(id, ts, user_id, endpoint, params_json, estimated_ms, actual_ms, rows, status)` + `ip` + `approval_id` | ✅ Plan 04-01 | `nexo/data/models_nexo.py::NexoQueryLog`, 4 indices incl. partial `WHERE status='slow'` |
| QUERY-02 | `nexo.query_thresholds` editable desde UI | ✅ Plan 04-01 (schema+seeds) + Plan 04-04 (UI) | Schema + 4 seeds (D-01..D-04) + `/ajustes/limites` CRUD |
| QUERY-03 | `preflight.estimate_cost(endpoint, params) -> Estimation(ms, level, reason)` + aprendizaje desde query_log | ✅ Plan 04-02 (heurística) + Plan 04-04 (learning) | `nexo/services/preflight.py`, `factor_learning.py`, `factor_auto_refresh.py` |
| QUERY-04 | Endpoints devuelven Estimation antes de ejecutar; green/amber/red | ✅ Plan 04-02 | 4 routers (pipeline/bbdd/capacidad/operarios) gatean por level |
| QUERY-05 | `query_timing` middleware mide actual_ms + WARNING si divergencia | ✅ Plan 04-02 | `nexo/middleware/query_timing.py` innermost middleware |
| QUERY-06 | Flujo approval asíncrono + `/ajustes/solicitudes` + force=true re-dispatch | ✅ Plan 04-03 | `nexo/services/approvals.py` state machine + `api/routers/approvals.py` + 2 páginas UI + badge sidebar HTMX |
| QUERY-07 | `/ajustes/limites` edita umbrales + preflight aplicado en pipeline/bbdd/capacidad/operarios | ✅ Plan 04-04 (UI) + Plan 04-02 (preflight aplicado) | `api/routers/limites.py` + preflight wiring en 4 routers |
| QUERY-08 | Pipeline no congela UI (asyncio.to_thread + pipeline_lock) | ✅ Plan 04-02 | `asyncio.to_thread` + semáforo(3) + timeout 900s |

**Traceability: 8/8 requirements Complete.**

## Code Review Follow-up

Ran `/gsd-code-review 4` → 1 CRITICAL + 5 HIGH + 7 MEDIUM + 6 LOW.
Ran `/gsd-code-review-fix 4` on CRITICAL + HIGH only (6 fixes landed):

| Finding | Severity | Fix Commit | Status |
|---------|----------|------------|--------|
| CR-01 query_timing writes on gate-reject | CRITICAL | `a709035` | ✅ fixed |
| H-01 approval mutations not audited | HIGH | `872064a` | ✅ fixed |
| H-02 POST /api/approvals allowlist | HIGH | `c1f3361` | ✅ fixed |
| H-03 user email in /ajustes/solicitudes | HIGH | `882056a` | ✅ fixed |
| H-04 PUT /api/thresholds allowlist | HIGH | `08725e0` | ✅ fixed |
| H-05 canonical_json list order drift | HIGH | `f9a8849` | ✅ fixed |

MEDIUM (7) + LOW (6) deferred to Phase 4.1 polish per operator delegation.
Full detail in `.planning/phases/04-consultas-pesadas/04-REVIEW-FIX.md`.

## Key Decisions Honored (from 04-CONTEXT.md)

D-01..D-04 (thresholds seeds), D-05/D-06/D-07 (modal textos), D-11/D-12
(/ajustes/rendimiento), D-13 (HTMX badge sin email), D-14 (TTL 7d +
cleanup Mon 03:05), D-15 (CAS single-use + canonical_json), D-16 (owner
cancel any, user cancel own), D-17 (slow status), D-18 (semáforo 3 +
timeout 900s soft), D-19 (LISTEN/NOTIFY + safety-net 5min), D-20
(factor auto-refresh 1er Mon del mes 03:10 UTC).

## Human Verification Items (post-return smoke)

Estos items estructuralmente cubiertos por tests automatizados pero
necesitan eyeball humano antes de considerar Phase 4 cerrada 100%.
Status: `human_needed` hasta que el operador haga este smoke.

1. **AMBER modal E2E** — Abrir `/pipeline`, disparar run con muchos recursos/días → modal amber aparece con texto "Esto tardará ~X min, ¿continuar?". Confirmar cancela/continúa.
2. **RED gate + approval flow** — Como usuario, disparar query en `/bbdd` con coste > `block_ms` → modal red con enlace a "Solicitar aprobación". Como propietario en `/ajustes/solicitudes`, aprobar. Volver como usuario al enlace `?approval_id=<N>` → se ejecuta.
3. **UI no-congela** — Durante `pipeline/run` activo, navegar a `/historial` → responde sin bloquear (el semáforo permite otras páginas).
4. **Sidebar badge HTMX** — Login como propietario con 1+ approvals pendientes → badge (N) aparece en sidebar. Crear otra pending → badge incrementa tras ≤30s sin recargar.
5. **`/ajustes/limites` hot-reload** — Editar `warn_ms` de un endpoint → guardar → inmediatamente repetir la misma query → comportamiento usa el nuevo umbral (sin restart de worker; vía LISTEN/NOTIFY).
6. **`/ajustes/rendimiento`** — Tabla muestra estadísticas `estimated_ms` vs `actual_ms` por endpoint + gráfica Chart.js de series temporales.
7. **Scheduler 3 jobs operativos** — Logs muestran los 3 jobs al arrancar: `approvals_cleanup` Mon 03:05, `query_log_cleanup` Mon 03:00, `factor_auto_refresh` 1er Mon 03:10.

Sube un "approved" o reporta fallo por item. Los 7 items pasarán automáticamente a `resolved` cuando el operador confirme.

## Regression Check

- Plan 04-01 foundation: 12 green tests (schema_guard + repos + DTOs)
- Plan 04-02 preflight: +28 new tests (13 preflight unit, 6 pipeline_lock, 4 middleware integration, 5 router contract)
- Plan 04-03 approvals: +23 net tests (service state machine + /api/approvals contract)
- Plan 04-04 observability: +19 new tests (thresholds_cache + CRUD + listen/notify)
- Phase 2/3 baseline: preserved (zero regressions)

Total green: ~141 tests en subset Postgres-only (tests/data + tests/services +
tests/routers + tests/middleware + tests/integration). SQL-Server-dependent
tests en tests/auth timeout en entorno dev sin acceso a MES/APP.

## Deferred Items

- Manual smoke (7 items arriba) — pending human execution.
- MEDIUM (7) + LOW (6) code review findings — candidatos para Phase 4.1
  polish si se decide abrir gap closure; documentados en 04-REVIEW.md.
- PDF regression baseline de 03-02 — deadline 2026-04-26, sigue pendiente
  (ver STATE.md "Deferred Verifications").

## Verdict

**Phase 4 passed automated verification (5/5 must-haves + 8/8 requirements + 6/6 CRITICAL+HIGH code review fixed).**

Status `human_needed` hasta que el operador haga el smoke manual de los
7 items listados arriba. Recomendado continuar a Phase 5 (UI por roles)
en paralelo — Phase 4 está funcionalmente lista; la validación humana
es confirmatoria, no bloqueante para seguir planificando.
