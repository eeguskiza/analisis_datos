---
phase: 4
slug: consultas-pesadas
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-19
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Wave 0 mapping derived from `04-RESEARCH.md §Validation Architecture`.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pytest.ini` (existing, extends Phase 3 config) |
| **Quick run command** | `pytest tests/services/ tests/data/ -x --tb=short` |
| **Full suite command** | `make test-data && pytest tests/services/ tests/middleware/ tests/routers/` |
| **Estimated runtime** | ~45 seconds (unit + integration con docker compose up db) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/services/ tests/data/ -x` (quick)
- **After every plan wave:** Run full suite
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Requirement | Test Coverage | Command | File Exists | Status |
|-------------|---------------|---------|-------------|--------|
| QUERY-01 (query_log tabla + índices) | unit (schema shape) | `pytest tests/data/test_schema_query_log.py -x` | ❌ Wave 0 | ⬜ pending |
| QUERY-01 (schema_guard incluye query_log) | unit | `pytest tests/data/test_schema_guard_extended.py -x` | ❌ Wave 0 | ⬜ pending |
| QUERY-02 (query_thresholds CRUD + NOTIFY emitted) | integration (Postgres real) | `pytest tests/services/test_thresholds_cache.py -x` | ❌ Wave 0 | ⬜ pending |
| QUERY-03 (preflight.estimate_cost green/amber/red pipeline) | unit puro | `pytest tests/services/test_preflight.py::test_pipeline_classify -x` | ❌ Wave 0 | ⬜ pending |
| QUERY-03 (learning factor desde últimos 30 runs) | unit | `pytest tests/services/test_preflight.py::test_factor_recalc -x` | ❌ Wave 0 | ⬜ pending |
| QUERY-04 (endpoints devuelven Estimation antes de ejecutar) | contract | `pytest tests/routers/test_preflight_endpoints.py -x` | ❌ Wave 0 | ⬜ pending |
| QUERY-05 (query_timing middleware escribe actual_ms + alerta slow) | integration | `pytest tests/middleware/test_query_timing.py -x` | ❌ Wave 0 | ⬜ pending |
| QUERY-06 (approval CAS single-use) | integration | `pytest tests/services/test_approvals.py -x` | ❌ Wave 0 | ⬜ pending |
| QUERY-06 (usuario cancela solicitud propia) | contract | `pytest tests/routers/test_approvals_api.py::test_user_cancel -x` | ❌ Wave 0 | ⬜ pending |
| QUERY-07 (preflight aplicado a pipeline/bbdd/capacidad/operarios >90d) | contract | `pytest tests/routers/test_preflight_endpoints.py::test_preflight_scope -x` | ❌ Wave 0 | ⬜ pending |
| QUERY-08 (pipeline en asyncio.to_thread, semáforo, timeout) | integration | `pytest tests/services/test_pipeline_lock.py -x` | ❌ Wave 0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Tests files a crear antes de que los plans de Phase 4 ejecuten (bloqueo duro):

- [ ] `tests/data/test_schema_query_log.py` — stubs para QUERY-01 (verifica tabla existe, columnas correctas, índices)
- [ ] `tests/data/test_schema_guard_extended.py` — stubs para schema_guard extendido (incluye query_log, query_thresholds, query_approvals)
- [ ] `tests/services/test_thresholds_cache.py` — stubs para cache + LISTEN/NOTIFY roundtrip (QUERY-02, D-19)
- [ ] `tests/services/test_preflight.py` — stubs para estimate_cost + factor learning (QUERY-03)
- [ ] `tests/services/test_approvals.py` — stubs para CAS approval consumption (QUERY-06, D-15)
- [ ] `tests/services/test_pipeline_lock.py` — stubs para asyncio.to_thread + semáforo + timeout (QUERY-08, D-18)
- [ ] `tests/middleware/test_query_timing.py` — stubs para middleware timing (QUERY-05)
- [ ] `tests/routers/test_preflight_endpoints.py` — stubs para contracts /preflight (QUERY-04, QUERY-07)
- [ ] `tests/routers/test_approvals_api.py` — stubs para `/api/approvals/*` (QUERY-06, D-16)
- [ ] `tests/conftest.py` — fixtures compartidos (extiende fixtures existentes de Phase 3: `db_nexo`, añade `thresholds_cache`, `approval_factory`, `mock_pipeline`)
- [ ] Instalar `pytest-asyncio` (si no está ya) para tests de asyncio.to_thread y LISTEN/NOTIFY

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Modal amber aparece con [Continuar]/[Cancelar] al disparar pipeline >120s | Success Criterion #1 | Frontend testing no automatizado en Mark-III (no Playwright en CI) | Login como usuario; POST /pipeline/preflight con fecha_desde=2025-01-01, fecha_hasta=2025-12-31, recursos=[todos]; verificar estimated_ms > 120000; disparar /pipeline/run en UI; modal debe aparecer con duración, breakdown, botones. Presionar Continuar ejecuta; Cancelar no. |
| Modal red con [Solicitar aprobación] para queries >600s | Success Criterion #2 | Frontend + approval flow multi-usuario | Login como usuario; disparar pipeline de 1 año × 100 recursos; modal rojo aparece; click [Solicitar aprobación] crea fila en query_approvals pending; badge sidebar del propietario muestra "(1)"; login como propietario; aprobar en /ajustes/solicitudes; usuario re-dispara pipeline con approval_id; ejecuta. |
| Chart.js línea estimated vs actual en /ajustes/rendimiento | D-11, D-12 | Visualización requiere inspección ocular | Login propietario; ir a /ajustes/rendimiento; filtrar endpoint=pipeline/run, rango=30d; verificar que tabla muestra n_runs, avg_est, avg_actual, divergencia%, p95; verificar que gráfica renderiza con 2 líneas (estimated azul, actual naranja); hover muestra tooltip con timestamp + ms. |
| UI no congela al ejecutar pipeline (asyncio.to_thread) | Success Criterion #5 | Requiere 2 pestañas + inspección visual | Pestaña A: POST /pipeline/run con payload grande. Pestaña B (simultáneo): navegar a /recursos, /historial, /capacidad — deben responder <500ms sin bloqueo. Toast/feedback en pestaña A durante ejecución. |
| Usuario puede cancelar su solicitud pending en /mis-solicitudes | D-16 | Flujo inter-usuario | Login usuario; disparar pipeline red; solicitar aprobación; ir a /mis-solicitudes; verificar fila pending propia; click Cancelar; confirmar; estado → cancelled; badge propietario no cuenta la cancelled. |
| Threshold edit se propaga <1s via LISTEN/NOTIFY | D-19 | Requiere 2 workers o timing manual | En UI: editar warn_ms de pipeline en /ajustes/límites de 120s a 60s; simultáneamente en otra pestaña hacer POST /pipeline/preflight; verificar que el nuevo threshold se aplica. Si workers uvicorn=1, trivial. Si workers>1, verificar con logs. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (10 archivos nuevos + 1 conftest extension)
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter (cuando todos los tests pasen)

**Approval:** pending
