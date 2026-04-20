---
phase: 04-consultas-pesadas
plan: 02
subsystem: preflight/postflight/middleware/async
type: summary
wave: 2
status: complete
requirements: [QUERY-03, QUERY-04, QUERY-05, QUERY-07, QUERY-08]
tags: [nexo, phase-4, preflight, middleware, asyncio, semaphore, modal-amber, modal-red]
executed: 2026-04-20
duration_minutes: 60
task_count: 7
file_count: 15
commits:
  - 48e3875 feat(04-02): add preflight service + pipeline_lock semaphore/timeout
  - 67838c0 feat(04-02): add QueryTimingMiddleware + wire as innermost middleware
  - cdad52e refactor(04-02): async pipeline router with preflight + force/approval gate
  - 97d09bc refactor(04-02): inject preflight in bbdd/capacidad/operarios routers
  - e119f4e feat(04-02): preflight modal (AMBER/RED) + humanize_ms + env vars
  - ec340d4 test(04-02): convert Wave 0 stubs to real tests + fix state read timing
self_check: PASSED
metrics:
  tests_before: 123 passed / 25 skipped / 0 fail
  tests_after: 151 passed / 22 skipped / 1 xfailed / 0 fail
  new_tests: 28 (13 preflight unit + 6 pipeline_lock + 4 middleware + 5 routers + 1 xfail 04-03)
  files_created: 7
  files_modified: 8
key-files:
  created:
    - nexo/services/preflight.py
    - nexo/services/pipeline_lock.py
    - nexo/middleware/__init__.py
    - nexo/middleware/query_timing.py
  modified:
    - api/main.py
    - api/routers/pipeline.py
    - api/routers/bbdd.py
    - api/routers/capacidad.py
    - api/routers/operarios.py
    - templates/pipeline.html
    - templates/bbdd.html
    - static/js/app.js
    - .env.example
    - tests/services/test_preflight.py
    - tests/services/test_pipeline_lock.py
    - tests/middleware/test_query_timing.py
    - tests/routers/test_preflight_endpoints.py
decisions-implemented:
  - D-01 pipeline warn=120s block=600s (cached from 04-01 seed)
  - D-02 bbdd warn=3s block=30s (cached from 04-01 seed)
  - D-03 capacidad/operarios preflight sólo rango > 90d
  - D-04 factor fallback 2000ms (pipeline) / 1000ms (bbdd) / 50ms (rango)
  - D-05 modal AMBER bloqueante [Cancelar][Continuar]
  - D-06 modal RED bloqueante [Cancelar][Solicitar aprobación]
  - D-07 texto jerárquico 3 líneas (duración + breakdown + reassurance)
  - D-08 per-request sin sticky
  - D-09 params_json con whitelist (sql + database + user_provided)
  - D-15 semántica force+approval_id (import-guard hasta 04-03)
  - D-17 slow status cuando actual_ms > warn_ms * 1.5 + log.warning
  - D-18 semáforo(3) + timeout 900s soft
---

# Phase 04 / Plan 04-02 — Preflight + Middleware + asyncio.to_thread

**Preflight + postflight + asyncio.to_thread + modal frontend** — cierra el 80% funcional de Phase 4: los 4 endpoints caros (pipeline/run, bbdd/query, capacidad, operarios) ahora estiman coste antes de ejecutar, bloquean amber/red según umbrales editables, y el pipeline OEE corre en un thread bajo semáforo que no congela la UI.

## Objective delivered

- **`nexo/services/preflight.py`**: función pura `estimate_cost(endpoint, params) -> Estimation`. Heurística por endpoint:
  - pipeline/run → `n_recursos × n_dias × factor` (default 2000ms).
  - bbdd/query → `factor` baseline (default 1000ms, sin EXPLAIN).
  - capacidad/operarios → `n_dias × factor` (default 50ms/día) **sólo si rango>90d**.
  - Fallback green defensivo cuando threshold ausente o endpoint desconocido.
- **`nexo/services/pipeline_lock.py`**: `asyncio.Semaphore(MAX_CONCURRENT=3)` módulo-level + `PIPELINE_TIMEOUT_SEC=900` + helper `run_with_lock(fn)`.
  - Landmine explícito en docstring: `asyncio.wait_for` NO mata el thread subyacente (soft timeout).
- **`nexo/middleware/query_timing.py`**: `BaseHTTPMiddleware` registrado como innermost en api/main.py. Escribe fila en `nexo.query_log` con `status ∈ {ok, slow, error, timeout}` + emite `log.warning` cuando `status=slow`.
- **Refactor routers**:
  - `api/routers/pipeline.py`: nuevo `POST /preflight` + `POST /run` async con gate (green ejecuta, amber requiere `force=true`, red requiere `force+approval_id`).
  - `api/routers/bbdd.py`: preflight en `POST /query` **ANTES** de `_validate_sql` (UX primero, security después).
  - `api/routers/capacidad.py` + `api/routers/operarios.py`: preflight condicional a `rango_dias > 90` (D-03).
- **Frontend**: Alpine component `preflightModal` en `static/js/app.js` + modales AMBER/RED en `templates/pipeline.html` y `templates/bbdd.html`. Helper `humanize_ms()`. URL re-dispatch (`?approval_id=<N>` auto-ejecuta).
- **Env vars**: `NEXO_PIPELINE_MAX_CONCURRENT=3` + `NEXO_PIPELINE_TIMEOUT_SEC=900` documentadas en `.env.example` con comentario del landmine.

## Key files — contracts

### `nexo/services/preflight.py` (76 sloc)

```python
def estimate_cost(endpoint: str, params: dict) -> Estimation:
    """Pure dispatch. No I/O except thresholds_cache.get().
    Returns green/amber/red + estimated_ms + breakdown + warn_ms/block_ms.
    """
```

Dispatcher: `pipeline/run | bbdd/query | capacidad | operarios`. Endpoints no reconocidos → green defensivo. Thresholds ausentes → green defensivo. Clasificación strict less-than: `warn <= x < block → amber`, `x >= block → red` (600000 exacto → red).

### `nexo/services/pipeline_lock.py` (35 sloc)

```python
MAX_CONCURRENT = int(os.environ.get("NEXO_PIPELINE_MAX_CONCURRENT", "3"))
PIPELINE_TIMEOUT_SEC = float(os.environ.get("NEXO_PIPELINE_TIMEOUT_SEC", "900"))
pipeline_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

async def run_with_lock(fn, /, *args, **kwargs):
    async with pipeline_semaphore:
        return await asyncio.wait_for(
            asyncio.to_thread(fn, *args, **kwargs),
            timeout=PIPELINE_TIMEOUT_SEC,
        )
```

Landmine documentado (grep "wait_for does NOT kill" → 1 hit).

### `nexo/middleware/query_timing.py` (~175 sloc)

- `_TIMED_PATHS`: 4 rutas (pipeline/run, bbdd/query, capacidad, operarios).
- `_EXCLUDED`: `/api/health`, `/api/approvals/count`.
- Dispatch short-circuits:
  1. Path no en `_TIMED_PATHS` ni excluido.
  2. `user=None` (request sin autenticar).
  3. `capacidad`/`operarios` sin `estimated_ms` en state (rango<=90d).
- Lee `request.state` **DESPUÉS de `call_next`** (fix crítico — ver sección Deviations).
- `status` classification: 504→timeout, >=400→error, actual_ms>warn_ms*1.5→slow, else ok.
- Best-effort INSERT: fallos de DB NO tumban la response (patrón AuditMiddleware).

### `api/main.py` middleware chain

```
Request → Auth → Audit → QueryTiming → handler
                                      ↓
Response ← Auth ← Audit ← QueryTiming ← (handler)
```

Registro:
```python
app.add_middleware(QueryTimingMiddleware)  # innermost
app.add_middleware(AuditMiddleware)
app.add_middleware(AuthMiddleware)         # outermost
```

### `api/routers/pipeline.py` endpoints

- `POST /api/pipeline/preflight` → `Estimation` JSON (response_model).
- `POST /api/pipeline/run` async:
  - `_build_pipeline_params(req)` → dict normalizado.
  - Estima + pobla `request.state`.
  - Gate por level: green directo, amber sin force → 428, red sin approval → 403, red+force+approval → `consume_approval` (via import-guard 04-03) + ejecuta.
  - `async with pipeline_semaphore: await asyncio.wait_for(asyncio.to_thread(_worker), timeout=900)`.
  - Opción A research: colecta todos los messages en el thread + re-emite via SSE al final (pierde progreso real, no congela event loop).
  - Timeout → `HTTPException(504, f"Pipeline timeout ({PIPELINE_TIMEOUT_SEC}s)")`.

### `api/routers/bbdd.py` POST /query

Orden: preflight → gate level → `_validate_sql` → ejecución. `user_provided=True` flag en params_json para distinguir del ciclo batch del cron.

### `api/routers/capacidad.py` + `api/routers/operarios.py`

Preflight condicional `if rango_dias > 90:` con gate idéntico al pipeline. Rango ≤90d pasa directo sin tocar `request.state`.

### Frontend — `static/js/app.js::preflightModal`

Registered via `document.addEventListener('alpine:init', ...)`. Estado: `modalLevel | estimation | currentEndpoint | currentParams | currentExecuteFn | pendingApprovalId`.

Métodos:
- `init()` — lee `?approval_id=<N>` de URL.
- `attempt(endpoint, params, executeFn)` — POST preflight → green ejecuta / amber|red abre modal. Si `pendingApprovalId && level=red` → auto-executeFn(true, aid).
- `confirmRun()` — amber → executeFn(true, null).
- `requestApproval()` — red → POST /api/approvals (stub hasta 04-03, muestra toast).
- `cancel()` — cierra modal.

### `humanize_ms(ms)` (pure JS)

```
500    -> "~500ms"
12000  -> "~12s"
125000 -> "~2 min 5s"
4000000-> "~1h 6 min"
```

## Decisions honored

| ID | Decision | How honored |
|----|----------|-------------|
| D-01 | pipeline warn=2min block=10min | Consumido desde cache (seed de 04-01) |
| D-02 | bbdd warn=3s block=30s | Consumido desde cache |
| D-03 | preflight solo rango>90d en capacidad/operarios | Gate explícito `if rango_dias > 90:` en ambos routers |
| D-04 | factor fallback 2000ms pipeline | Constante `_FALLBACK_PIPELINE_FACTOR_MS` + preferencia `t.factor_ms if t.factor_ms is not None` |
| D-05 | modal AMBER bloqueante [Continuar][Cancelar] | Markup Alpine `x-show="modalLevel === 'amber'"` + textos literales |
| D-06 | modal RED bloqueante [Solicitar aprobación][Cancelar] | Markup + `requestApproval()` POST /api/approvals |
| D-07 | texto 3 líneas (duración bold, breakdown gris, "UI seguirá respondiendo...") | Tres `<p>` jerárquicos en ambos modales |
| D-08 | per-request sin sticky | No se usa cookie/sessionStorage; `cancel()` limpia estado |
| D-09 | params_json con SQL completo + database + user_provided:true | `_build_pipeline_params` + `{"sql": ..., "database": ..., "user_provided": True}` en bbdd |
| D-15 | single-use approval (status=approved, user match, params match) | Wrapper `consume_approval` con import-guard hasta Plan 04-03 |
| D-17 | slow status + log.warning cuando actual>warn*1.5 | `_classify_status` + `logger.warning(..., ratio)` en dispatch |
| D-18 | semáforo(3) + timeout 900s via env vars | `pipeline_lock.py` + `.env.example` + landmine documentado |

## Commits (6 total)

| Hash | Type | Message |
|------|------|---------|
| 48e3875 | feat | preflight service + pipeline_lock semaphore/timeout |
| 67838c0 | feat | QueryTimingMiddleware + wire as innermost middleware |
| cdad52e | refactor | async pipeline router with preflight + force/approval gate |
| 97d09bc | refactor | inject preflight in bbdd/capacidad/operarios routers |
| e119f4e | feat | preflight modal (AMBER/RED) + humanize_ms + env vars |
| ec340d4 | test | convert Wave 0 stubs to real tests + fix state read timing |

## Test coverage

| Suite | Tests | Status |
|-------|-------|--------|
| `tests/services/test_preflight.py` | 13 green + 1 skipped | unit, no DB |
| `tests/services/test_pipeline_lock.py` | 6 green | unit (pytest-asyncio) |
| `tests/middleware/test_query_timing.py` | 4 green | integration (Postgres) |
| `tests/routers/test_preflight_endpoints.py` | 5 green + 1 xfail | integration (Postgres) |
| **TOTAL Plan 04-02** | **28 new tests** | **0 fail** |

Full suite: **151 passed / 22 skipped / 1 xfailed / 0 failed** (up from baseline 123 passed). Zero regressions in `tests/auth/`, `tests/data/`, `tests/test_oee_*`.

xfail: `test_run_red_with_valid_approval_executes` — espera `nexo/services/approvals.consume_approval` de Plan 04-03.

## Deviations from plan

### 1. [Rule 1 — Bug] `request.state` read timing in QueryTimingMiddleware

- **Found during:** Task 6 (integration test `test_timing_writes_row_for_bbdd_query`).
- **Issue:** La implementación inicial leía `request.state.estimated_ms` **ANTES** de `await call_next(request)`. El router pobla ese campo DURANTE la ejecución del handler, por lo que la lectura previa siempre devolvía `None`. Las filas en `nexo.query_log` se escribían con `estimated_ms=NULL` pese a que el router había poblado state correctamente.
- **Fix:** Mover la lectura (`estimated_ms`, `params_json`, `approval_id`) **dentro del try** después de `call_next`. Añadir re-lectura defensiva en el except path vía `getattr(request.state, ..., None)` por si `call_next` levantó antes de la asignación de locals. Short-circuit de rango<=90d movido después de call_next para aprovechar el mismo patrón.
- **Files modified:** `nexo/middleware/query_timing.py`.
- **Commit:** ec340d4 (dentro del commit de tests; el fix fue prerequisito para que los tests pasaran).
- **Impact on other plans:** Documentado; patrón correcto para cualquier futuro middleware que mida state poblado por el handler.

Ninguna otra desviación funcional. Los 6 commits encajan exactamente con las 6 tasks auto+tdd del plan. Task 7 (checkpoint human-verify) auto-aprobada per delegación explícita del operador (ver sección siguiente).

## Task 7 — Manual smoke (auto-approved)

El operador delegó las tareas Wave 1→4 sin supervisión interactiva. El checkpoint manual de Task 7 (modal amber E2E + UI no-congela + URL re-dispatch) queda **pendiente de verificación humana** cuando el operador regrese. Los 3 comportamientos están cubiertos por:

1. **Modal amber E2E** — cubierto por `test_run_amber_no_force_returns_428` (contract). Smoke visual queda para Task 7 humano.
2. **UI no-congela** — cubierto por el patrón `asyncio.to_thread` + `asyncio.Semaphore(3)` (estructural); no testeable en automatizado Mark-III sin Playwright. `test_semaphore_limits_to_max` verifica la concurrencia real.
3. **URL re-dispatch** — cubierto parcialmente por el test `test_run_red_with_valid_approval_executes` (xfail hasta 04-03). Smoke visual requiere Plan 04-03 mergeado.

Log del checkpoint auto-approved:
```
⚡ Auto-approved: Task 7 (manual smoke) — Modal amber E2E + UI non-blocking + URL re-dispatch. User delegated Waves 1-4 execution via "vuelvo en unas horas vale? no hagas preguntas". Covered structurally by automated tests; visual checks deferred to next interactive session or after Plan 04-03 lands.
```

## Landmines documented in code

1. **`nexo/services/pipeline_lock.py`** (module docstring + code comment):
   > asyncio.wait_for does NOT kill the underlying thread. Timeout is UX-soft — matplotlib keeps running until it completes. (ES) NO mata el thread subyacente.

2. **`api/routers/pipeline.py`** (import-guard):
   ```python
   try:
       from nexo.services.approvals import consume_approval
       _APPROVALS_AVAILABLE = True
   except ImportError:
       _APPROVALS_AVAILABLE = False
   ```
   Cuando Plan 04-03 aterrice, `_APPROVALS_AVAILABLE=True` y la ruta red+force+approval_id funciona completa. Mientras tanto responde 503 explícito (no silent-pass que bypasearía security).

3. **`nexo/middleware/query_timing.py`** (comentario explícito):
   > IMPORTANTE: leemos request.state DESPUÉS de call_next. El router pobla estimated_ms/approval_id/params_json durante la ejecución del handler (dentro de call_next). Leer antes siempre da None para esos campos.

## Requirements traceability

| Requirement | Status | Covered by |
|-------------|--------|------------|
| QUERY-03 | Complete | `nexo/services/preflight.py` + 13 unit tests |
| QUERY-04 | Complete | 4 routers gatean por level + 428/403 con Estimation detail |
| QUERY-05 | Complete | `nexo/middleware/query_timing.py` + 4 integration tests |
| QUERY-07 | Complete | Preflight aplicado a los 4 endpoints; rango>90d en capacidad/operarios |
| QUERY-08 | Complete | `pipeline_lock.py` + `asyncio.to_thread` wrapping en pipeline/run |

QUERY-06 (approvals flow) queda intencionalmente fuera de este plan → Plan 04-03.

## Discoveries for downstream plans

### For Plan 04-03 (approvals)

- Los 4 routers (pipeline, bbdd, capacidad, operarios) tienen el mismo **import-guard** de `consume_approval`. Cuando 04-03 entregue `nexo/services/approvals.py`, los 4 se activarán automáticamente al siguiente reload de uvicorn.
- El front-end ya POSTea a `/api/approvals` desde `requestApproval()`. El endpoint no existe hasta 04-03 — el fetch devuelve 404 y el toast dice "Aprobaciones aún no disponibles (Plan 04-03)".
- URL query `?approval_id=<N>` es leído por el modal en `init()`. Cuando 04-03 entregue `/mis-solicitudes` con link "Ejecutar ahora", la URL destino `/pipeline?approval_id=<N>` activará el auto-re-dispatch.

### For Plan 04-04 (observability + LISTEN/NOTIFY + learning)

- `nexo.query_log` ya se puebla correctamente. `/ajustes/rendimiento` puede consultar desde día 1.
- `thresholds_cache` con safety-net de 5min funciona; Plan 04-04 sólo añade `listen_loop` para reducir latencia a <1s.
- El factor_ms recalculado manual puede persistir en `nexo.query_thresholds.factor_ms`; el preflight lo consumirá desde el cache automáticamente tras `full_reload()`.

### For verifier

- Tests unitarios NO requieren Postgres (`tests/services/test_preflight.py`, `test_pipeline_lock.py`).
- Tests de integración SÍ requieren Postgres (`tests/middleware/`, `tests/routers/`). Skipif si Postgres down.
- Smoke manual pendiente: ver Task 7 arriba.

## Next

Plan 04-03 puede arrancar **en paralelo** con Plan 04-02 (no tiene dependencia de código). Plan 04-04 depende de 04-02 + 04-03 merged.

## Self-Check: PASSED

- ✓ `nexo/services/preflight.py` exists (FOUND)
- ✓ `nexo/services/pipeline_lock.py` exists (FOUND)
- ✓ `nexo/middleware/__init__.py` exists (FOUND)
- ✓ `nexo/middleware/query_timing.py` exists (FOUND)
- ✓ Commit 48e3875 present (FOUND)
- ✓ Commit 67838c0 present (FOUND)
- ✓ Commit cdad52e present (FOUND)
- ✓ Commit 97d09bc present (FOUND)
- ✓ Commit e119f4e present (FOUND)
- ✓ Commit ec340d4 present (FOUND)
- ✓ Full suite: 151 passed / 22 skipped / 1 xfailed / 0 failed
