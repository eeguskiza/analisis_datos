# Phase 4: Consultas pesadas — Research

**Researched:** 2026-04-19
**Domain:** Preflight/postflight de queries caras + middleware de timing + aprobación asíncrona + umbrales editables + `asyncio.to_thread` para pipeline
**Confidence:** HIGH (stack verified, project context fully loaded, official docs consulted)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions (4 explícitas del usuario)

- **D-05 Modal AMBER bloqueante** `[CITED: 04-CONTEXT.md D-05]`
  - Overlay oscuro Alpine.js con título "Esta operación puede tardar" + duración estimada + breakdown gris + botones `[Continuar]` + `[Cancelar]`.
  - Per-request, sin sticky.
  - Cancelar = cierra modal, no ejecuta. Continuar = dispara request original con `force=true` + payload.
  - Texto de reassurance: "La UI seguirá respondiendo mientras la operación corre".

- **D-11 Página `/ajustes/rendimiento`** `[CITED: 04-CONTEXT.md D-11]`
  - Nueva sub-página bajo `/ajustes/*`, acceso sólo `propietario` (mismo patrón que `/ajustes/auditoria`).
  - Filtros: user (dropdown), endpoint (dropdown de los 4), estado (green/amber/red/slow/timeout/approved_run), rango temporal (7d/30d/90d/custom).
  - Tabla superior: `endpoint | n_runs | avg_estimated_ms | avg_actual_ms | divergencia_% | p95_actual_ms | n_slow`.
  - Gráfica inferior: line chart con 2 series (estimated, actual) vía Chart.js CDN.
  - Endpoints API: `GET /api/rendimiento/summary` + `GET /api/rendimiento/timeseries`.
  - Comparte dataset con botón "Recalcular factor" de `/ajustes/límites` (D-04).

- **D-16 Usuario puede cancelar solicitud propia** `[CITED: 04-CONTEXT.md D-16]`
  - Página `/mis-solicitudes` accesible a todos los usuarios autenticados.
  - Lista `WHERE user_id = :current_user` con estados pending/approved/rejected/expired/cancelled/consumed.
  - Botón `[Cancelar]` sobre filas `pending`.
  - POST `/api/approvals/<id>/cancel`: verifica ownership, cambia `status='cancelled'`, graba `cancelled_at`.
  - Cancelled visible en histórico 30d.

- **D-19 Cache thresholds + invalidate on edit (LISTEN/NOTIFY)** `[CITED: 04-CONTEXT.md D-19]`
  - `nexo/services/thresholds_cache.py` in-memory dict `{endpoint: (warn_ms, block_ms, factor, updated_at)}`.
  - Al arrancar worker: carga todas las filas de `nexo.query_thresholds` en cache.
  - Al guardar desde `/ajustes/límites`: UPDATE + `NOTIFY nexo_thresholds_changed, '<endpoint>'`.
  - Listener background (async task) con `LISTEN nexo_thresholds_changed`: al recibir, recarga fila específica en cache.
  - **Fallback**: si `now() - updated_at > 5min`, re-lee BD forzosamente. Double safety net.

### Claude's Discretion (delegadas al planner con defaults documentados)

- **D-01 Umbrales pipeline**: `warn_ms=120_000` (2 min), `block_ms=600_000` (10 min). Editables.
- **D-02 Umbrales bbdd/query**: `warn_ms=3_000`, `block_ms=30_000`. Editables.
- **D-03 Preflight capacidad/operarios**: sólo si `rango_días > 90`. Umbrales default iguales a bbdd.
- **D-04 factor_por_modulo inicial pipeline**: seed=2000ms por (recurso × día). Botón "Recalcular desde últimos 30 runs": `factor_nuevo = median(actual_ms / (n_recursos × n_días))` con `status IN ('ok','slow')`.
- **D-06 Modal RED**: bloqueante rojo, botones `[Solicitar aprobación]` + `[Cancelar]`. POST `/api/approvals` → toast verde → link a `/mis-solicitudes`. No redirect.
- **D-07 Texto del modal**: jerarquía 3 líneas (estimación bold, breakdown gris, reassurance gris).
- **D-08 Persistencia modal**: per-request siempre, sin sticky.
- **D-09 Contenido `params_json`**: SQL completo + params para bbdd; `{fecha_desde, fecha_hasta, recursos, n_dias, n_recursos}` para pipeline; `{fecha_desde, fecha_hasta, rango_dias, filters_applied}` para capacidad/operarios. Sanitización con whitelist del audit_middleware.
- **D-10 Retención query_log**: 90 días default. Env var `NEXO_QUERY_LOG_RETENTION_DAYS` (0 = forever). Job al arrancar + Monday 03:00.
- **D-12 Chart library**: Chart.js vía CDN (ya cargado en `base.html@4.4.7`). Fallback tabla sin gráfica.
- **D-13 Notificación approvals**: badge sidebar HTMX `hx-get="/api/approvals/count" hx-trigger="every 30s"`. Sin email, sin banner global.
- **D-14 TTL approvals**: 7 días default (env `NEXO_APPROVAL_TTL_DAYS`). Job Monday 03:05 marca expired. Histórico 30d más.
- **D-15 Semántica force+approval_id**: verifica 5 condiciones (existe, status=approved, user_id match, consumed_at NULL, params_json equality). Marca `consumed_at=now()` + `consumed_run_id`. Single-use.
- **D-17 Postflight divergence alert**: `actual_ms > warn_ms × 1.5` → `logging.warning` + `status='slow'` en query_log. Sin push notification.
- **D-18 Matplotlib concurrency**: `asyncio.Semaphore(NEXO_PIPELINE_MAX_CONCURRENT=3)` + `asyncio.wait_for(asyncio.to_thread(...), timeout=NEXO_PIPELINE_TIMEOUT_SEC=900)`.
- **D-20 Preflight learning**: botón manual + cron mensual fallback (`factor_auto_refresh.py` 1er Monday 03:10, recalcula si `factor_updated_at > 60d`).

### Deferred Ideas (OUT OF SCOPE — copy verbatim)

- Email notificación approvals → Mark-IV (depende de SMTP).
- Banner global top de "Solicitudes pendientes" → reconsiderar si badge resulta poco visible.
- Dashboard streaming (websockets) → Mark-IV.
- Rate limiting fino por endpoint → Mark-IV (preflight + login rate-limit cubren el 80%).
- Sustituir matplotlib → sólo si >20% de pipelines caen en `status='timeout'` tras 30d en prod.
- Paginación/virtualización de `/ajustes/rendimiento` → Mark-IV si volumen >10k filas/página.
- Filtros por departamento en `/ajustes/solicitudes` → Phase 5 (UIROL).
- Alertas por Slack/Telegram → Mark-IV.
- Forecast predictivo de coste (ML) → Mark-V.
- Aprobación delegada (propietario delega en directivo) → Mark-IV.
- Exportar CSV de query_log desde `/ajustes/rendimiento` → nice-to-have, planner decide si trivial.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| QUERY-01 | Tabla `nexo.query_log` con `(id, ts, user_id, endpoint, params_json, estimated_ms, actual_ms, rows, status)` | §Schema Design; §Plan 04-01 |
| QUERY-02 | Tabla `nexo.query_thresholds` con `(endpoint, warn_ms, block_ms, updated_at, updated_by)` editable desde UI | §Schema Design; §Plan 04-01 + 04-04 |
| QUERY-03 | `nexo/services/preflight.py` con `estimate_cost(endpoint, params) -> Estimation` + heurística `n_recursos × n_días × factor` + aprendizaje desde `query_log` | §Preflight Service (Pattern 2); §Factor Learning Algorithm |
| QUERY-04 | Endpoints devuelven `Estimation` antes de ejecutar: green=directo, amber=confirmación UI, red=aprobación | §Modal Amber/Red Flow; §Plan 04-02 |
| QUERY-05 | `nexo/middleware/query_timing.py` mide `time.monotonic()` y escribe `actual_ms`; alerta WARNING si `actual > warn_ms × 1.5` | §QueryTimingMiddleware (Pattern 1); §Middleware Order |
| QUERY-06 | Flujo approval asíncrono: `/ajustes/solicitudes` + `force=true` + approval_id single-use | §Approval Flow (Pattern 4); §Single-use CAS |
| QUERY-07 | `/ajustes/limites` edita umbrales; preflight aplicado en pipeline/run, bbdd/query, capacidad (>90d), operarios (>90d) | §Thresholds UI; §Endpoint Retrofit |
| QUERY-08 | Pipeline OEE en `asyncio.to_thread` para no bloquear worker uvicorn | §asyncio.to_thread + Semaphore + Timeout (Pattern 3); §Matplotlib GIL |
</phase_requirements>

## Summary

Phase 4 introduce una capa transversal de observabilidad y control de coste sobre 4 endpoints caros. El work se divide en **4 planes secuenciales** (04-01 foundation → 04-02 preflight+middleware → 04-03 approval flow → 04-04 observability UI), con 04-02 y 04-03 potencialmente paralelizables tras 04-01 si el equipo puede manejar dos hilos de trabajo.

La arquitectura propuesta es pragmática: **middleware Starlette para timing** (patrón estándar, integración limpia con la cadena auth→audit existente), **servicio puro en `nexo/services/preflight.py`** para la heurística (testeable unitariamente sin DB), **tabla `query_approvals` con CAS atómico vía `UPDATE ... WHERE consumed_at IS NULL RETURNING id`** para el flujo single-use, y **`asyncio.to_thread` + semáforo + `wait_for`** para desacoplar matplotlib del event loop.

**Las tres landmines más peligrosas** identificadas:
1. **`asyncio.wait_for` NO cancela el thread subyacente** — cuando el timeout dispara, la corrutina recibe `CancelledError` pero el pipeline sigue corriendo en background hasta que matplotlib termine por su cuenta. Impacto: PDFs parciales en `informes_dir`, memoria no liberada, slots del semáforo zombies. Mitigación obligatoria: marcador `is_cancelled` + cleanup tras `shutil.rmtree(tmp_root)` + documentar que `NEXO_PIPELINE_TIMEOUT_SEC` es un soft timeout a nivel de UX, no un kill hard.
2. **LISTEN/NOTIFY es best-effort** — cualquier desconexión del listener pierde mensajes irrecuperablemente (Postgres no los encola). D-19 ya lo anticipa con el safety net de 5min; research confirma que es la única estrategia robusta.
3. **psycopg2 síncrono bloquea el worker** si se usa `LISTEN` directo — la implementación recomendada es wrappear el listen-loop en `asyncio.to_thread` dentro del `lifespan`, o migrar a `psycopg` (v3) / `asyncpg` para el listener. Migración completa descartada (ripple effect inaceptable en Mark-III); el wrap es suficiente.

**Primary recommendation:** 4 planes atómicos, orden secuencial recomendado (04-01 → 04-02 → 04-03 → 04-04). Usar SQLAlchemy ORM puro para las 3 tablas nuevas (evita archivos `.sql` redundantes según D-01 del 03-CONTEXT). No introducir APScheduler — `asyncio.create_task` con loop simple en `lifespan` cubre los 2-3 jobs necesarios sin dep nueva.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Middleware de timing | API / Backend | — | Transversal a todas las requests; Starlette middleware ejecuta pre/post handler con `request.state` ya poblado por auth→audit |
| Preflight estimation | API / Backend (service layer) | Browser (modal display) | Cálculo puro sin I/O de dominio: cabe en service puro testeable |
| Persistencia query_log / thresholds / approvals | Database / Storage | — | Postgres (`nexo.*`) — mismo engine y schema que Phase 2/3 |
| Cache in-memory de thresholds | API / Backend (proceso worker) | Database (LISTEN/NOTIFY) | Cache local a cada worker uvicorn; invalidación cross-worker vía pub/sub Postgres |
| Modal amber/red | Browser / Client | API (estimation endpoint) | Alpine.js componente UI; datos vienen de `/api/.../preflight` |
| Badge sidebar HTMX | Browser / Client | API (count endpoint) | HTMX polling 30s; endpoint `/api/approvals/count` devuelve HTML fragment |
| Chart.js timeseries | Browser / Client | API (`/api/rendimiento/timeseries`) | Render client-side; datos JSON serializados por endpoint |
| `asyncio.to_thread` pipeline | API / Backend (async route) | OS thread pool | Matplotlib es CPU-bound + sincrónico; thread pool evita bloquear event loop |
| Semáforo de concurrencia pipeline | API / Backend (async) | Memoria del proceso | `asyncio.Semaphore(3)` local al proceso worker; multi-worker requeriría distributed lock (Mark-IV) |
| Cleanup jobs (query_log retention, approvals TTL, factor auto-refresh) | API / Backend (scheduler) | Database | `asyncio.create_task` en lifespan + loop simple con sleep hasta próximo Monday 03:00 |
| LISTEN/NOTIFY listener | API / Backend (background task) | Database | `asyncio.create_task` en lifespan; thread wrapper sobre psycopg2 sync LISTEN |

**Verificación de tier:** no hay capabilities misasignadas respecto al estado actual de Phase 3. El único punto delicado es el semáforo — funciona dentro de **1 proceso uvicorn**; si en Mark-IV se escala a multi-worker requerirá un lock distribuido (Redis / Postgres advisory lock). Se documenta como trade-off aceptado (default `workers=1` en compose actual).

## Standard Stack

### Core (ya presentes en el repo — verificados vía `requirements.txt`)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| fastapi | 0.135.3 | HTTP framework | `[VERIFIED: requirements.txt]` — ya en uso, 3+ phases de uso sin fricción |
| uvicorn[standard] | 0.44.0 | ASGI server | `[VERIFIED: requirements.txt]` — stock del proyecto |
| sqlalchemy | 2.0.49 | ORM + core | `[VERIFIED: requirements.txt]` — Phase 3 estableció patrón multi-engine |
| psycopg2-binary | 2.9.11 | Postgres driver (sync) | `[VERIFIED: requirements.txt]` — driver usado por `engine_nexo` |
| pydantic | 2.x (via pydantic-settings 2.13.1) | DTOs + validación | `[VERIFIED: requirements.txt]` — patrón para `Estimation` |
| jinja2 | 3.1.6 | Templates | `[VERIFIED: requirements.txt]` — pages en `/ajustes/*` |
| matplotlib | 3.10.8 | PDF generation | `[VERIFIED: requirements.txt]` — explícitamente NO se sustituye en Mark-III |
| slowapi | 0.1.9 | Rate limiting | `[VERIFIED: requirements.txt]` — ya en uso para `/login` |

### Supporting (front-end CDN — verificado en `templates/base.html`)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Chart.js | 4.4.7 (CDN jsdelivr) | Timeseries line chart | `[VERIFIED: templates/base.html line 9]` + `[VERIFIED: npm view chart.js → 4.5.1 latest, 4.4.7 ya cargado]` — PERFECT: no requiere añadir CDN nueva |
| Alpine.js | 3.14.8 | Modal state, reactive UI | `[VERIFIED: templates/base.html]` — patrón ya usado en sidebar |
| HTMX | 2.0.4 | Polling del badge | `[VERIFIED: templates/base.html]` — patrón ya usado para `/ajustes/auditoria` auto-refresh |
| HTMX SSE extension | 2.2.2 | Pipeline SSE | `[VERIFIED: templates/base.html]` — ya cargado, pipeline lo usa |
| Tailwind | CDN | Styling | `[VERIFIED: templates/base.html]` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| psycopg2 + thread-wrapped LISTEN | `psycopg` (v3) con async LISTEN nativo | `[ASSUMED]` Mejor performance pero requiere añadir dep + posible conflicto con `engine_nexo`. Descartado: ripple effect inaceptable. |
| psycopg2 + thread-wrapped LISTEN | `asyncpg` | `[CITED: FastAPI discussion #13732]` Best-in-class async Postgres driver pero SQLAlchemy 2.0.49 requiere setup doble (engine sync + engine async). Descartado: complejidad no justificada para 1 listener. |
| `asyncio.create_task` loop para cleanup | APScheduler (`AsyncIOScheduler`) | `[CITED: apscheduler.readthedocs.io]` APScheduler da cron triggers más expresivos, misfire handling, pero añade dep. **Recomendación: NO añadir APScheduler** — sólo 2-3 jobs simples (Monday 03:00), el loop async es trivial. |
| Chart.js | ApexCharts | `[CITED: flowbite.com]` Flowbite lo usa; más moderno. **Descartado**: Chart.js ya cargado, cambiar rompe nada pero añade 50KB sin beneficio. |
| `asyncio.to_thread` pipeline | `ProcessPoolExecutor` | `[CITED: runebook.dev/asyncio.to_thread]` ProcessPool bypasea GIL pero matplotlib + serialización de data_rows vía pickle = overhead + complejidad. Descartado: ver §GIL Analysis. |
| UPDATE ... RETURNING para consume approval | Advisory locks Postgres | `[CITED: cybertec-postgresql]` Advisory locks son alternativa pero UPDATE ... RETURNING es más simple y atómico por naturaleza. **Elegido: UPDATE ... RETURNING**. |

**Installation:**
```bash
# NO requiere instalación nueva. Todos los deps ya en requirements.txt.
# Verificado:
# - Chart.js CDN ya cargado en base.html
# - psycopg2-binary 2.9.11 ya disponible para LISTEN/NOTIFY
# - slowapi ya disponible (rate limit opcional para /api/approvals/*)
```

**Version verification (ejecutado 2026-04-19):**
- `chart.js`: latest stable `4.5.1` (published 2025-10-13); project loads `4.4.7` via CDN (Dec 2024). Gap: 1 minor. Decisión: **mantener 4.4.7** — no hay breaking changes relevantes entre 4.4.7 y 4.5.1, upgrade deferido para no introducir churn.
- `fastapi`: `0.135.3` pineado. Verificado en `requirements.txt`.
- `sqlalchemy`: `2.0.49`. Verificado.
- `psycopg2-binary`: `2.9.11`. Verificado.

## Architecture Patterns

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BROWSER / CLIENT                                │
│                                                                              │
│  User dispara run → fetch /api/pipeline/preflight ─────────┐                 │
│                                                             │                 │
│  Estimation recibido:                                       │                 │
│    green → fetch /api/pipeline/run (directo)                │                 │
│    amber → Alpine.js modal [Continuar] [Cancelar]           │                 │
│            └─ Continuar → fetch /api/pipeline/run + force=true                │
│    red   → Alpine.js modal [Solicitar aprob] [Cancelar]     │                 │
│            └─ Solicitar → POST /api/approvals → toast ──────│                 │
│                                                             │                 │
│  Badge sidebar (solo propietario):                          │                 │
│    HTMX hx-get /api/approvals/count every 30s ──────────────│                 │
│                                                             │                 │
│  Chart.js timeseries en /ajustes/rendimiento                │                 │
│    fetch /api/rendimiento/timeseries?endpoint=X&days=N ─────│                 │
└─────────────────────────────────────────────────────────────│─────────────────┘
                                                              │
                                    ┌─────────────────────────▼─────────────────┐
                                    │   STARLETTE MIDDLEWARE CHAIN              │
                                    │                                           │
                                    │   request ──▶ AuthMiddleware (outer)      │
                                    │              ├─ populates request.state.user
                                    │              └─ 401/redirect if no session │
                                    │                                           │
                                    │              AuditMiddleware              │
                                    │              ├─ writes nexo.audit_log     │
                                    │              └─ sanitizes body            │
                                    │                                           │
                                    │              QueryTimingMiddleware (NEW)  │
                                    │              ├─ t0 = monotonic()          │
                                    │              ├─ await call_next           │
                                    │              ├─ t1 = monotonic()          │
                                    │              ├─ if path in TIMED_PATHS:   │
                                    │              │     write nexo.query_log   │
                                    │              │     with actual_ms         │
                                    │              └─ emit WARNING if slow      │
                                    └──────────────┬────────────────────────────┘
                                                   │
                                    ┌──────────────▼────────────────────────────┐
                                    │   ROUTERS (4 con preflight)               │
                                    │                                           │
                                    │   pipeline.py                             │
                                    │   ├─ POST /preflight ───────────┐         │
                                    │   └─ POST /run                  │         │
                                    │      ├─ verify approval_id (si  │         │
                                    │      │   force=true)            │         │
                                    │      └─ async with semaphore:   │         │
                                    │         await asyncio.wait_for( │         │
                                    │           asyncio.to_thread(    │         │
                                    │             run_pipeline_sync)) │         │
                                    │                                 │         │
                                    │   bbdd.py /query                │         │
                                    │   ├─ _validate_sql (existente)  │         │
                                    │   ├─ preflight.estimate_cost ◀──┤         │
                                    │   └─ execute or reject          │         │
                                    │                                 │         │
                                    │   capacidad.py, operarios.py    │         │
                                    │   └─ preflight SÓLO si rango>90d│         │
                                    └─────────────────┬───────────────┘         │
                                                      │                         │
                                    ┌─────────────────▼───────────────────────┐ │
                                    │   SERVICES                              │ │
                                    │                                         │ │
                                    │   preflight.py                          │ │
                                    │   └─ estimate_cost(endpoint, params)    │ │
                                    │       ├─ load factor from cache ◀───────┼─┤
                                    │       ├─ compute estimated_ms           │ │
                                    │       └─ classify green/amber/red       │ │
                                    │                                         │ │
                                    │   thresholds_cache.py  (in-memory)      │ │
                                    │   ├─ dict {endpoint: (warn, block, ... )│ │
                                    │   ├─ reload_one(endpoint)               │ │
                                    │   └─ full_reload()                      │ │
                                    │                                         │ │
                                    │   approvals.py                          │ │
                                    │   ├─ create_approval                    │ │
                                    │   ├─ consume_approval (CAS)             │ │
                                    │   └─ cancel/reject/expire               │ │
                                    │                                         │ │
                                    │   pipeline_lock.py                      │ │
                                    │   └─ _pipeline_semaphore (global)       │ │
                                    └─────────────────┬───────────────────────┘ │
                                                      │                         │
                                    ┌─────────────────▼───────────────────────┐ │
                                    │   BACKGROUND TASKS (lifespan)           │ │
                                    │                                         │ │
                                    │   listen_thresholds_loop                │ │
                                    │   └─ await to_thread(                   │ │
                                    │        blocking_listen_forever)         │ │
                                    │        psycopg2 LISTEN nexo_thresholds  │ │
                                    │        on NOTIFY: schedule reload       │ │
                                    │                                         │ │
                                    │   cleanup_scheduler                     │ │
                                    │   ├─ query_log_cleanup (Monday 03:00)   │ │
                                    │   ├─ approvals_expire (Monday 03:05)    │ │
                                    │   └─ factor_auto_refresh (1st Mon 03:10)│ │
                                    └─────────────────┬───────────────────────┘ │
                                                      │                         │
                                    ┌─────────────────▼───────────────────────┐ │
                                    │   POSTGRES (nexo.*)                     │ │
                                    │                                         │ │
                                    │   nexo.query_log       (retention 90d)  │ │
                                    │   nexo.query_thresholds (editable UI)   │ │
                                    │   nexo.query_approvals (single-use CAS) │ │
                                    │                                         │ │
                                    │   NOTIFY nexo_thresholds_changed ───────┼─┘
                                    └─────────────────────────────────────────┘
```

**Flow de un POST /api/pipeline/run con `force=true` (amber aprobado):**

1. Browser → AuthMiddleware (outer) valida cookie → popula `request.state.user`.
2. AuditMiddleware lee body (POST) → sanitiza → graba fila en `nexo.audit_log`.
3. QueryTimingMiddleware guarda `t0 = time.monotonic()`.
4. Router `pipeline.run`:
   - Si `req.force and req.approval_id`: verifica approval (CAS atomic UPDATE).
   - Si no force: llama `preflight.estimate_cost(...)`. Si level ≠ green sin force/approval → `HTTPException(403)`.
   - Entra al `async with semaphore:` + `await asyncio.wait_for(asyncio.to_thread(run_pipeline_sync(...)), timeout=900)`.
5. `run_pipeline_sync` (ex-generator ahora wrappeado — ver §Pipeline Wrapping) corre matplotlib en thread pool → devuelve lista de PDFs.
6. QueryTimingMiddleware calcula `actual_ms = (monotonic() - t0) * 1000`.
7. Graba fila en `nexo.query_log` con `endpoint='pipeline/run'`, `estimated_ms`, `actual_ms`, `status` (`ok`/`slow`/`timeout`/`error`), `approval_id` si aplica.
8. Response SSE al browser.

### Recommended Project Structure

```
nexo/
├── middleware/              # NUEVO — no existe hoy
│   ├── __init__.py
│   └── query_timing.py      # QueryTimingMiddleware
├── services/
│   ├── auth.py              # (existente) — no se toca
│   ├── preflight.py         # NUEVO — estimate_cost
│   ├── thresholds_cache.py  # NUEVO — cache + LISTEN listener
│   ├── approvals.py         # NUEVO — create / consume / cancel / expire
│   ├── pipeline_lock.py     # NUEVO — _pipeline_semaphore
│   ├── query_log_cleanup.py # NUEVO — job retención
│   ├── approvals_cleanup.py # NUEVO — job TTL
│   └── factor_auto_refresh.py # NUEVO — job recálculo
├── data/
│   ├── models_nexo.py       # (existente) — AÑADIR 3 modelos ORM
│   ├── engines.py           # (existente) — no se toca
│   └── repositories/
│       └── nexo.py          # (existente) — AÑADIR QueryLogRepo, ThresholdRepo, ApprovalRepo
api/
├── main.py                  # MODIFICAR — add middleware, lifespan tasks
├── deps.py                  # (existente) — AÑADIR get_thresholds cache dep (opcional)
├── models.py                # (existente) — AÑADIR Estimation, ApprovalRequest, etc.
└── routers/
    ├── pipeline.py          # MODIFICAR — añadir /preflight, aceptar force+approval_id
    ├── bbdd.py              # MODIFICAR — inyectar preflight en /query
    ├── capacidad.py         # MODIFICAR — preflight si rango>90d
    ├── operarios.py         # MODIFICAR — preflight si rango>90d
    ├── approvals.py         # NUEVO — /api/approvals/*
    ├── rendimiento.py       # NUEVO — /api/rendimiento/summary, /timeseries
    └── ajustes_extras.py    # NUEVO (o extender pages.py) — GET /ajustes/{limites,solicitudes,rendimiento}, /mis-solicitudes
templates/
├── ajustes_limites.html     # NUEVO
├── ajustes_solicitudes.html # NUEVO
├── ajustes_rendimiento.html # NUEVO
└── mis_solicitudes.html     # NUEVO
static/js/
└── app.js                   # MODIFICAR — modal amber/red Alpine component + humanize_ms
tests/
├── services/                # NUEVO subdirectorio
│   ├── test_preflight.py
│   ├── test_approvals_cas.py
│   └── test_thresholds_cache.py
└── middleware/
    └── test_query_timing.py
```

### Pattern 1: QueryTimingMiddleware

**What:** Starlette BaseHTTPMiddleware que mide `time.monotonic()` pre/post handler y graba fila en `nexo.query_log` para los 4 endpoints relevantes.

**When to use:** Exactamente en esta fase — middleware se activa sólo para paths de `_TIMED_PATHS`; las demás requests lo atraviesan sin overhead (early return).

**Critical order:** `app.add_middleware` en Starlette es **LIFO** — último registrado = primero en procesar (outer). El orden actual del proyecto es:

```python
# api/main.py (estado actual):
app.add_middleware(AuditMiddleware)    # inner — 2º en ejecutar
app.add_middleware(AuthMiddleware)     # outer — 1º en ejecutar
```

Para insertar `QueryTimingMiddleware` **entre audit y router**:

```python
# api/main.py (tras Phase 4):
app.add_middleware(QueryTimingMiddleware)  # inner-most (más cercano al handler)
app.add_middleware(AuditMiddleware)
app.add_middleware(AuthMiddleware)          # outer
```

Cadena efectiva: `request → Auth → Audit → QueryTiming → handler → QueryTiming → Audit → Auth → response`.

`request.state.user` ya poblado por Auth cuando llega a Timing ✓.

**Example:**
```python
# nexo/middleware/query_timing.py
# Source: [CITED: fastapi.tiangolo.com/tutorial/middleware/] + Phase 3 patrón audit
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import ClassVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from nexo.data.engines import SessionLocalNexo
from nexo.data.models_nexo import NexoQueryLog  # NEW en Plan 04-01
from nexo.services import thresholds_cache

logger = logging.getLogger("nexo.query_timing")

# Mapa path → endpoint_key (coincide con nexo.query_thresholds.endpoint).
# Los endpoints NO listados pasan por el middleware sin escribir en query_log.
_TIMED_PATHS: ClassVar[dict[str, str]] = {
    "/api/pipeline/run":  "pipeline/run",
    "/api/bbdd/query":    "bbdd/query",
    "/api/capacidad":     "capacidad",
    "/api/operarios":     "operarios",
}

# Paths EXCLUIDOS explícitamente del timing (para evitar ruido en query_log
# si algún día aparecen en _TIMED_PATHS por error):
_EXCLUDED = frozenset({"/api/health", "/api/approvals/count"})


class QueryTimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        endpoint_key = _TIMED_PATHS.get(path)

        # Short-circuit: paths no auditados pasan sin overhead.
        if endpoint_key is None or path in _EXCLUDED:
            return await call_next(request)

        user = getattr(request.state, "user", None)
        estimated_ms = getattr(request.state, "estimated_ms", None)
        approval_id = getattr(request.state, "approval_id", None)
        params_snapshot = getattr(request.state, "params_json", None)

        t0 = time.monotonic()
        status_code = 500
        error_name: str | None = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            actual_ms = int((time.monotonic() - t0) * 1000)
            query_status = _classify_status(
                actual_ms=actual_ms,
                endpoint_key=endpoint_key,
                http_status=status_code,
            )
            _persist(
                user_id=user.id if user else None,
                endpoint=endpoint_key,
                params_json=params_snapshot,
                estimated_ms=estimated_ms,
                actual_ms=actual_ms,
                rows=None,
                status=query_status,
                approval_id=approval_id,
                ip=request.client.host if request.client else "unknown",
            )
            if query_status == "slow":
                warn_ms = thresholds_cache.get(endpoint_key).warn_ms
                ratio = actual_ms / warn_ms if warn_ms else 0
                logger.warning(
                    "slow_query endpoint=%s user=%s estimated=%s actual=%s ratio=%.2f",
                    endpoint_key,
                    user.id if user else None,
                    estimated_ms,
                    actual_ms,
                    ratio,
                )
            return response
        except Exception as exc:
            actual_ms = int((time.monotonic() - t0) * 1000)
            error_name = type(exc).__name__
            _persist(
                user_id=user.id if user else None,
                endpoint=endpoint_key,
                params_json=params_snapshot,
                estimated_ms=estimated_ms,
                actual_ms=actual_ms,
                rows=None,
                status="error",
                approval_id=approval_id,
                ip=request.client.host if request.client else "unknown",
            )
            raise


def _classify_status(*, actual_ms: int, endpoint_key: str, http_status: int) -> str:
    """green/amber/red/slow/timeout — slow se detecta vs warn_ms*1.5."""
    if http_status == 504:
        return "timeout"
    if http_status >= 400:
        return "error"
    t = thresholds_cache.get(endpoint_key)
    if t is None:
        return "ok"
    if actual_ms > t.warn_ms * 1.5:
        return "slow"
    return "ok"


def _persist(**fields) -> None:
    """INSERT fila en nexo.query_log. Errores de escritura NO tumban response."""
    try:
        db = SessionLocalNexo()
        try:
            db.add(NexoQueryLog(ts=datetime.now(timezone.utc), **fields))
            db.commit()
        finally:
            db.close()
    except Exception:
        logger.exception("Error escribiendo nexo.query_log endpoint=%s", fields.get("endpoint"))
```

**Per-endpoint config via `request.state`**: el router que hace preflight puebla `request.state.estimated_ms`, `request.state.approval_id`, `request.state.params_json` **antes** de ejecutar la lógica. El middleware los lee tras `call_next`. Esto acopla levemente el router al middleware pero es el patrón más simple — alternativa (headers custom) es más hacky.

### Pattern 2: Preflight Service (pure, testeable)

**What:** Servicio puro en `nexo/services/preflight.py` que calcula `Estimation` sin side effects (salvo leer cache thresholds que ya está in-memory).

**When to use:** Llamado desde:
- `POST /api/pipeline/preflight` (nuevo endpoint que NO ejecuta, sólo estima).
- `POST /api/pipeline/run` al inicio: decide ejecutar o rechazar.
- `POST /api/bbdd/query` tras whitelist, antes de ejecutar.
- `GET /api/capacidad` y `GET /api/operarios` si `rango_dias > 90`.

**Example:**
```python
# nexo/services/preflight.py
# Source: [CITED: 04-CONTEXT.md D-03, D-04] — heurística decidida
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from nexo.services import thresholds_cache


Level = Literal["green", "amber", "red"]


@dataclass(frozen=True)
class Estimation:
    estimated_ms: int
    level: Level
    reason: str
    breakdown: str          # "4 recursos × 15 días × ~2.0s/run"
    factor_used: float | None  # ms per (recurso × día) para pipeline; None para otros


def estimate_cost(endpoint: str, params: dict) -> Estimation:
    """Estima coste ANTES de ejecutar. No ejecuta nada.

    endpoint: 'pipeline/run' | 'bbdd/query' | 'capacidad' | 'operarios'
    params: dict específico por endpoint (ver D-09).
    """
    t = thresholds_cache.get(endpoint)
    if t is None:
        # Thresholds no configuradas → fallback green (defensivo).
        return Estimation(
            estimated_ms=0,
            level="green",
            reason="threshold no configurado — default green",
            breakdown="",
            factor_used=None,
        )

    if endpoint == "pipeline/run":
        return _estimate_pipeline(params, t)
    if endpoint == "bbdd/query":
        return _estimate_bbdd(params, t)
    if endpoint in ("capacidad", "operarios"):
        return _estimate_rango(params, t, endpoint)

    # endpoint no reconocido → green defensivo
    return Estimation(
        estimated_ms=0,
        level="green",
        reason=f"endpoint {endpoint} no soportado",
        breakdown="",
        factor_used=None,
    )


def _estimate_pipeline(params: dict, t) -> Estimation:
    n_recursos = int(params.get("n_recursos", 0) or len(params.get("recursos", []) or []))
    n_dias = int(params.get("n_dias", 0) or _calc_days(params))
    factor = t.factor or 2000.0  # seed inicial D-04

    estimated = int(n_recursos * n_dias * factor)
    level = _classify(estimated, t)
    return Estimation(
        estimated_ms=estimated,
        level=level,
        reason=_reason_for(level, estimated, t),
        breakdown=f"{n_recursos} recursos × {n_dias} días × ~{factor/1000:.1f}s/run",
        factor_used=factor,
    )


def _estimate_bbdd(params: dict, t) -> Estimation:
    # Sin EXPLAIN (demasiado frágil por D-03). Fallback: ms medio vs baseline
    # del factor (si está calibrado) o 1000ms default.
    factor = t.factor or 1000.0
    estimated = int(factor)
    level = _classify(estimated, t)
    return Estimation(
        estimated_ms=estimated,
        level=level,
        reason=_reason_for(level, estimated, t),
        breakdown=f"baseline ~{factor/1000:.1f}s (sin EXPLAIN)",
        factor_used=factor,
    )


def _estimate_rango(params: dict, t, endpoint: str) -> Estimation:
    n_dias = int(params.get("rango_dias", 0) or _calc_days(params))
    # Sólo se activa si > 90d (D-03); el router ya filtra, defensivo aquí:
    if n_dias <= 90:
        return Estimation(
            estimated_ms=0, level="green",
            reason="rango ≤ 90d — preflight desactivado",
            breakdown="", factor_used=None,
        )
    factor = t.factor or 50.0  # ms por día default
    estimated = int(n_dias * factor)
    level = _classify(estimated, t)
    return Estimation(
        estimated_ms=estimated,
        level=level,
        reason=_reason_for(level, estimated, t),
        breakdown=f"{n_dias} días × ~{factor}ms/día",
        factor_used=factor,
    )


def _classify(estimated_ms: int, t) -> Level:
    if estimated_ms < t.warn_ms:
        return "green"
    if estimated_ms < t.block_ms:
        return "amber"
    return "red"


def _reason_for(level: Level, estimated_ms: int, t) -> str:
    if level == "green":
        return "Ejecución estándar"
    if level == "amber":
        return f"Supera umbral de aviso ({t.warn_ms}ms)"
    return f"Excede límite configurado de {t.block_ms/1000/60:.0f} min"


def _calc_days(params: dict) -> int:
    fi = params.get("fecha_desde") or params.get("fecha_inicio")
    ff = params.get("fecha_hasta") or params.get("fecha_fin")
    if not fi or not ff:
        return 0
    if isinstance(fi, str):
        fi = date.fromisoformat(fi)
    if isinstance(ff, str):
        ff = date.fromisoformat(ff)
    return (ff - fi).days + 1
```

**Testeable unitariamente** sin DB: pasas thresholds mockeados vía `thresholds_cache` (monkeypatch o fixture).

### Pattern 3: asyncio.to_thread + Semaphore + Timeout (QUERY-08)

**What:** Desacoplar pipeline síncrono (matplotlib) del event loop; limitar concurrencia a 3; timeout 15min.

**Key finding CRÍTICO** `[CITED: github.com/python/cpython/issues/87185]` + `[CITED: runebook.dev/asyncio.to_thread]`:

> When using `asyncio.wait_for()` or `asyncio.timeout()` with `asyncio.to_thread()`, the thread itself continues running the synchronous task until it finishes, but the await is cancelled. Killing the actual thread/process is much harder.

**Implicaciones:**
- Timeout dispara `TimeoutError` (subclass de `CancelledError`) en la corrutina.
- Thread sigue ejecutando `run_pipeline_sync` hasta que matplotlib termina.
- El slot del semáforo **se libera al salir del `async with`** (Python semantics), pero el thread sigue consumiendo CPU/RAM hasta que el código síncrono acabe.
- PDFs parciales en `tmp_root` se cleanan por el `finally: shutil.rmtree(tmp_root)` del `run_pipeline` — **siempre que el thread termine**. Si matplotlib se cuelga, el cleanup nunca corre.

**Mitigación recomendada:**
1. El pipeline YA tiene `finally: shutil.rmtree(tmp_root, ignore_errors=True)` — se ejecuta dentro del thread, no en la corrutina. OK.
2. Documentar que el timeout es "soft" a nivel UX: el usuario ve 504, la fila en `query_log` se marca `status='timeout'`, pero el thread sigue 30-60s más hasta que matplotlib cierra figs.
3. **NO usar `ProcessPoolExecutor`** aunque bypaseara GIL: `data_rows` contiene objetos no trivialmente picklable (fechas, floats grandes) + 600MB de matplotlib pickle roundtrip = peor que el thread.
4. Env var `NEXO_PIPELINE_MAX_CONCURRENT=3` documentada en `.env.example` con comentario: "Memoria RAM pico ~200MB por slot; ajustar según disponibilidad de host".

**Pipeline wrapping (la parte tricky)**:

El `run_pipeline` actual es un **generator** (`yield "msg"` para SSE). `asyncio.to_thread` no wrappea generators — sólo funciones regulares. Dos opciones:

**Opción A (recomendada): recoger todos los msgs, devolverlos al final.**
Funciona porque el pipeline ya acumula `log_lines`. El SSE se pierde parcialmente durante la ejecución (el usuario ve spinner, no progreso en tiempo real), pero a cambio el worker no se bloquea. Trade-off aceptable porque ya hoy la UI no es crítica en el progreso (los mensajes son info para el operador, no bloqueantes).

**Opción B: SSE con queue thread-safe + worker que consume.**
Más complejo pero preserva SSE en tiempo real:

```python
# api/routers/pipeline.py — Opción B
import asyncio
from queue import Queue, Empty
from typing import AsyncGenerator

from nexo.services.pipeline_lock import pipeline_semaphore, PIPELINE_TIMEOUT_SEC


async def _run_pipeline_async(req) -> AsyncGenerator[str, None]:
    """Wrappea run_pipeline (sync generator) en thread con queue bridge."""
    q: Queue[str | None] = Queue()
    _SENTINEL = None

    def worker():
        try:
            from api.services.pipeline import run_pipeline
            for msg in run_pipeline(
                fecha_inicio=req.fecha_inicio, fecha_fin=req.fecha_fin,
                modulos=req.modulos, source=req.source, recursos=req.recursos,
            ):
                q.put(msg)
        finally:
            q.put(_SENTINEL)

    async with pipeline_semaphore:
        # Lanza worker en thread pool
        loop = asyncio.get_running_loop()
        worker_task = loop.run_in_executor(None, worker)

        start = loop.time()
        try:
            while True:
                # Drain queue con timeout cooperativo
                if loop.time() - start > PIPELINE_TIMEOUT_SEC:
                    # Marcar en state para middleware; NO matar thread
                    from fastapi import HTTPException
                    raise HTTPException(504, "Pipeline timeout (15 min)")
                try:
                    msg = q.get_nowait()
                except Empty:
                    await asyncio.sleep(0.1)
                    continue
                if msg is _SENTINEL:
                    break
                yield msg
        finally:
            # Esperar que el thread termine su finally (cleanup tmp_root)
            # Pero SIN bloquear la response: schedule cancelación soft
            # y reportar al operador que el thread sigue corriendo.
            # worker_task es Future; await o cancel aquí según semántica.
            pass
```

**Recomendación planner**: elegir Opción A por simplicidad (pipeline ya funciona ~correctamente sin SSE en tiempo real si el operador ve "Generando..." spinner). Opción B queda documentada para Mark-IV si el UX requiere progreso real.

**pipeline_lock.py:**
```python
# nexo/services/pipeline_lock.py
# Source: [CITED: 04-CONTEXT.md D-18]
import asyncio
import os

MAX_CONCURRENT = int(os.environ.get("NEXO_PIPELINE_MAX_CONCURRENT", "3"))
PIPELINE_TIMEOUT_SEC = int(os.environ.get("NEXO_PIPELINE_TIMEOUT_SEC", "900"))

# Semáforo global — una sola instancia por proceso worker uvicorn.
# Workers adicionales (uvicorn --workers N) tendrían N semáforos independientes
# — trade-off aceptado en Mark-III (default workers=1).
pipeline_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
```

### Pattern 4: Approval Flow (single-use, atomic CAS)

**What:** Tabla `nexo.query_approvals` + flujo pending → approved/rejected/cancelled/expired/consumed.

**CAS atómico** `[CITED: postgresql.org/docs/current/sql-update.html]` + `[CITED: cybertec-postgresql.com]`:

```sql
-- Consumo atómico: sólo un consumer puede ganar
UPDATE nexo.query_approvals
SET consumed_at = now(),
    consumed_run_id = :run_id
WHERE id = :approval_id
  AND status = 'approved'
  AND user_id = :user_id
  AND consumed_at IS NULL
RETURNING id, params_json;
```

Si `RETURNING` devuelve 0 filas → `HTTPException(403, "Invalid or expired approval")`.
Si devuelve 1 fila → approval válido, comparar `params_json` contra request actual (D-15 punto 5) en Python (SQLAlchemy `text(...).bindparams(...)`).

**Schema ORM:**
```python
# nexo/data/models_nexo.py — AÑADIR (Plan 04-01)
class NexoQueryLog(NexoBase):
    __tablename__ = "query_log"
    __table_args__ = (
        Index("ix_query_log_ts", "ts"),
        Index("ix_query_log_endpoint_ts", "endpoint", "ts"),
        Index("ix_query_log_user_ts", "user_id", "ts"),
        Index("ix_query_log_status_slow", "ts", postgresql_where=text("status = 'slow'")),
        {"schema": NEXO_SCHEMA},
    )

    id = Column(Integer, primary_key=True)
    ts = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    user_id = Column(Integer, ForeignKey("nexo.users.id", ondelete="SET NULL"), nullable=True)
    endpoint = Column(String(100), nullable=False)  # "pipeline/run", "bbdd/query", ...
    params_json = Column(Text, nullable=True)       # sanitized D-09
    estimated_ms = Column(Integer, nullable=True)
    actual_ms = Column(Integer, nullable=False)
    rows = Column(Integer, nullable=True)           # nº filas devueltas (para bbdd/query)
    status = Column(String(20), nullable=False)     # ok | slow | timeout | error | approved_run
    approval_id = Column(Integer, ForeignKey("nexo.query_approvals.id", ondelete="SET NULL"), nullable=True)
    ip = Column(String(64), nullable=True)


class NexoQueryThreshold(NexoBase):
    __tablename__ = "query_thresholds"
    __table_args__ = {"schema": NEXO_SCHEMA}

    endpoint = Column(String(100), primary_key=True)  # PK natural
    warn_ms = Column(Integer, nullable=False)
    block_ms = Column(Integer, nullable=False)
    factor = Column(Float, nullable=True)  # ms por unidad (recurso×día para pipeline)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_by = Column(Integer, ForeignKey("nexo.users.id", ondelete="SET NULL"), nullable=True)
    factor_updated_at = Column(DateTime(timezone=True), nullable=True)  # D-20 cron fallback


class NexoQueryApproval(NexoBase):
    __tablename__ = "query_approvals"
    __table_args__ = (
        Index("ix_approvals_status", "status"),
        Index("ix_approvals_user_status", "user_id", "status"),
        Index("ix_approvals_created_at", "created_at"),
        {"schema": NEXO_SCHEMA},
    )

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("nexo.users.id", ondelete="CASCADE"), nullable=False)
    endpoint = Column(String(100), nullable=False)
    params_json = Column(Text, nullable=False)
    estimated_ms = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    # pending | approved | rejected | cancelled | expired | consumed
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    ttl_days = Column(Integer, nullable=False, default=7)

    approved_at = Column(DateTime(timezone=True), nullable=True)
    approved_by = Column(Integer, ForeignKey("nexo.users.id", ondelete="SET NULL"), nullable=True)
    rejected_at = Column(DateTime(timezone=True), nullable=True)
    cancelled_at = Column(DateTime(timezone=True), nullable=True)
    expired_at = Column(DateTime(timezone=True), nullable=True)

    consumed_at = Column(DateTime(timezone=True), nullable=True)
    consumed_run_id = Column(Integer, nullable=True)  # FK a query_log.id — ciclo (no fuerzo FK)
```

**State machine:**
```
        ┌─ approved (propietario aprobó) ──▶ consumed (usuario ejecutó)
        │                                ╲
pending ┤                                 └─▶ expired (TTL 7d sin consumir — poco usual)
        ├─ rejected (propietario rechazó)
        ├─ cancelled (usuario canceló antes de aprobar — D-16)
        └─ expired (TTL 7d sin acción)
```

**approvals.py service:**
```python
# nexo/services/approvals.py
from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from nexo.data.models_nexo import NexoQueryApproval


def create_approval(
    db: Session, *, user_id: int, endpoint: str, params: dict, estimated_ms: int, ttl_days: int = 7,
) -> NexoQueryApproval:
    """Crea una solicitud pending. Devuelve el row con id poblado."""
    row = NexoQueryApproval(
        user_id=user_id,
        endpoint=endpoint,
        params_json=json.dumps(params, sort_keys=True, ensure_ascii=False),
        estimated_ms=estimated_ms,
        status="pending",
        ttl_days=ttl_days,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def consume_approval(
    db: Session, *, approval_id: int, user_id: int, current_params: dict,
) -> NexoQueryApproval:
    """CAS atomic: UPDATE ... WHERE ... RETURNING id.

    Raises HTTPException(403) si el approval no es consumible.
    """
    current_json = json.dumps(current_params, sort_keys=True, ensure_ascii=False)

    # UPDATE atómico — sólo un caller gana cuando hay carrera.
    stmt = text("""
        UPDATE nexo.query_approvals
        SET consumed_at = now() AT TIME ZONE 'UTC',
            status = 'consumed'
        WHERE id = :id
          AND user_id = :user_id
          AND status = 'approved'
          AND consumed_at IS NULL
          AND params_json = :params_json
        RETURNING id, endpoint, params_json, estimated_ms
    """)
    result = db.execute(
        stmt,
        {"id": approval_id, "user_id": user_id, "params_json": current_json},
    ).first()
    db.commit()

    if result is None:
        # Diagnóstico: distinguir "no existe", "ya consumido", "params cambiaron"
        row = db.get(NexoQueryApproval, approval_id)
        if row is None:
            raise HTTPException(403, "Approval no existe")
        if row.user_id != user_id:
            raise HTTPException(403, "Approval pertenece a otro usuario")
        if row.status != "approved":
            raise HTTPException(403, f"Approval está en estado {row.status}")
        if row.consumed_at is not None:
            raise HTTPException(403, "Approval ya fue consumido")
        # params_json difieren:
        raise HTTPException(403, "Parámetros cambiaron respecto a la solicitud aprobada")

    # Devolver el row recargado
    db.refresh(db.get(NexoQueryApproval, approval_id))
    return db.get(NexoQueryApproval, approval_id)
```

### Pattern 5: Thresholds Cache + LISTEN/NOTIFY

**What:** In-memory dict sincronizado entre workers via Postgres pub/sub.

**psycopg2 LISTEN es SÍNCRONO** — no hay variant async. Opciones `[CITED: github.com/fastapi/fastapi/issues/5015]` + `[CITED: medium.com/@diwasb54]`:

**Opción A (recomendada en Mark-III):** Thread-wrap el LISTEN loop.

```python
# nexo/services/thresholds_cache.py
from __future__ import annotations

import asyncio
import logging
import select
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Optional

import psycopg2
import psycopg2.extensions

from api.config import settings
from nexo.data.engines import SessionLocalNexo, engine_nexo
from nexo.data.models_nexo import NexoQueryThreshold

log = logging.getLogger("nexo.thresholds_cache")

FALLBACK_REFRESH_SECONDS = 300  # 5 min — D-19 safety net


@dataclass(frozen=True)
class ThresholdEntry:
    endpoint: str
    warn_ms: int
    block_ms: int
    factor: float | None
    loaded_at: datetime


_cache: dict[str, ThresholdEntry] = {}
_cache_lock = Lock()
_loaded_at_global: datetime | None = None


def full_reload() -> None:
    """Lee todas las filas de nexo.query_thresholds y repuebla el cache."""
    global _loaded_at_global
    db = SessionLocalNexo()
    try:
        rows = db.query(NexoQueryThreshold).all()
        now = datetime.now(timezone.utc)
        with _cache_lock:
            _cache.clear()
            for r in rows:
                _cache[r.endpoint] = ThresholdEntry(
                    endpoint=r.endpoint,
                    warn_ms=r.warn_ms,
                    block_ms=r.block_ms,
                    factor=r.factor,
                    loaded_at=now,
                )
            _loaded_at_global = now
        log.info("thresholds_cache reloaded %d entries", len(rows))
    finally:
        db.close()


def reload_one(endpoint: str) -> None:
    """Recarga una sola fila tras NOTIFY."""
    db = SessionLocalNexo()
    try:
        r = db.query(NexoQueryThreshold).filter_by(endpoint=endpoint).first()
        if r is None:
            with _cache_lock:
                _cache.pop(endpoint, None)
            return
        entry = ThresholdEntry(
            endpoint=r.endpoint, warn_ms=r.warn_ms, block_ms=r.block_ms,
            factor=r.factor, loaded_at=datetime.now(timezone.utc),
        )
        with _cache_lock:
            _cache[endpoint] = entry
        log.info("thresholds_cache reload_one endpoint=%s", endpoint)
    finally:
        db.close()


def get(endpoint: str) -> Optional[ThresholdEntry]:
    """Devuelve entry; si stale (>5min) fuerza full reload (safety net D-19)."""
    global _loaded_at_global
    now = datetime.now(timezone.utc)
    if _loaded_at_global is None or (now - _loaded_at_global).total_seconds() > FALLBACK_REFRESH_SECONDS:
        log.warning("thresholds_cache stale — forzando full_reload (fallback LISTEN)")
        full_reload()
    with _cache_lock:
        return _cache.get(endpoint)


def notify_changed(endpoint: str) -> None:
    """Emite NOTIFY tras UPDATE de thresholds. Llamar desde el CRUD endpoint."""
    # SQLAlchemy: NOTIFY necesita conexión dedicada (no session pool) o session.execute
    # con autocommit. psycopg2 requiere AUTOCOMMIT para NOTIFY.
    with engine_nexo.connect() as conn:
        conn.execute(text("NOTIFY nexo_thresholds_changed, :endpoint"), {"endpoint": endpoint})
        conn.commit()


# ── Background LISTEN loop ─────────────────────────────────────────────
# psycopg2 LISTEN es sync; lo corremos en thread via asyncio.to_thread.

def _blocking_listen_forever(stop_event) -> None:
    """Worker síncrono que hace LISTEN y dispara callbacks.

    Ejecutado dentro de asyncio.to_thread(...). El stop_event es
    threading.Event — en shutdown se pone a True y el loop sale.
    """
    while not stop_event.is_set():
        try:
            conn = psycopg2.connect(
                host=settings.pg_host, port=settings.pg_port,
                user=settings.effective_pg_user, password=settings.effective_pg_password,
                dbname=settings.pg_db,
            )
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            cur = conn.cursor()
            cur.execute("LISTEN nexo_thresholds_changed")
            log.info("thresholds_cache LISTEN activo")

            while not stop_event.is_set():
                if select.select([conn], [], [], 5) == ([], [], []):
                    continue  # timeout — revisa stop_event y vuelve a esperar
                conn.poll()
                while conn.notifies:
                    notify = conn.notifies.pop(0)
                    endpoint = notify.payload or ""
                    log.info("NOTIFY recibido endpoint=%s", endpoint)
                    if endpoint:
                        try:
                            reload_one(endpoint)
                        except Exception:
                            log.exception("Error en reload_one(%s)", endpoint)
                        else:
                            # Fallback full_reload si payload está vacío
                            try:
                                full_reload()
                            except Exception:
                                log.exception("Error en full_reload")
        except Exception:
            log.exception("LISTEN loop caído — reintentando en 5s")
            # stop_event.wait con timeout actúa como sleep interrumpible
            stop_event.wait(timeout=5.0)


async def start_listener(stop_event) -> None:
    """Lanza el worker síncrono en thread pool. Llamar desde lifespan."""
    await asyncio.to_thread(_blocking_listen_forever, stop_event)
```

**Lifespan integration (api/main.py):**
```python
import threading

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Existing:
    schema_guard.verify(engine_nexo)
    init_db()

    # NEW Phase 4:
    from nexo.services import thresholds_cache
    from nexo.services.query_log_cleanup import cleanup_loop
    thresholds_cache.full_reload()

    stop_event = threading.Event()
    listener_task = asyncio.create_task(thresholds_cache.start_listener(stop_event))
    cleanup_task = asyncio.create_task(cleanup_loop())

    try:
        yield
    finally:
        stop_event.set()
        cleanup_task.cancel()
        listener_task.cancel()
        try:
            await asyncio.gather(listener_task, cleanup_task, return_exceptions=True)
        except Exception:
            pass
```

**Opción B (descartada):** migrar todo a `asyncpg` para LISTEN async nativo. Pros: cleaner. Contras: `engine_nexo` requiere refactor global + test regression fullset + mezcla de drivers sync+async. Mark-III descarta esto.

### Pattern 6: Cleanup Scheduler (sin APScheduler)

**What:** Loop `asyncio.create_task` que duerme hasta el próximo Monday 03:00 y corre los jobs.

**Example:**
```python
# nexo/services/cleanup_scheduler.py
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timedelta, timezone

log = logging.getLogger("nexo.cleanup_scheduler")


def _seconds_until(target_time: time, dow: int | None = None) -> float:
    """Segundos desde ahora hasta next target_time (UTC). Si dow dado, al día indicado."""
    now = datetime.now(timezone.utc)
    target = now.replace(hour=target_time.hour, minute=target_time.minute, second=0, microsecond=0)
    if dow is not None:
        days_ahead = (dow - now.weekday()) % 7
        if days_ahead == 0 and target <= now:
            days_ahead = 7
        target = target + timedelta(days=days_ahead)
    elif target <= now:
        target = target + timedelta(days=1)
    return (target - now).total_seconds()


async def cleanup_loop() -> None:
    """Un job por iteración; prioriza el más próximo."""
    while True:
        # Monday = 0 en weekday()
        sec_qlog = _seconds_until(time(3, 0), dow=0)   # query_log purge
        sec_appr = _seconds_until(time(3, 5), dow=0)   # approvals expire
        sec_fact = _seconds_until(time(3, 10), dow=0)  # factor auto-refresh (1st Mon filter)

        next_sec = min(sec_qlog, sec_appr, sec_fact)
        log.info("cleanup_scheduler next run in %.0fs", next_sec)
        try:
            await asyncio.sleep(next_sec)
        except asyncio.CancelledError:
            log.info("cleanup_scheduler cancelled")
            raise

        now = datetime.now(timezone.utc)
        # Disparar todos los que toquen (tolerancia ±1 min)
        try:
            if abs(sec_qlog - next_sec) < 60:
                from nexo.services.query_log_cleanup import run as qlog_run
                await asyncio.to_thread(qlog_run)
            if abs(sec_appr - next_sec) < 60:
                from nexo.services.approvals_cleanup import run as appr_run
                await asyncio.to_thread(appr_run)
            if abs(sec_fact - next_sec) < 60 and now.day <= 7:  # 1er Monday
                from nexo.services.factor_auto_refresh import run as fact_run
                await asyncio.to_thread(fact_run)
        except Exception:
            log.exception("cleanup_scheduler job error — continúo")
```

### Anti-Patterns to Avoid

- **Correr LISTEN en el mismo conexión que `engine_nexo`**: LISTEN mantiene la conexión ocupada → pool starvation. Siempre conexión dedicada.
- **Usar `request.state` para datos multi-request**: `request.state` es per-request. Para cache cross-request usa module-level dict (como `thresholds_cache._cache`).
- **`asyncio.run(coroutine)` dentro de lifespan o middleware**: rompe el loop activo. Siempre `await` o `asyncio.create_task` dentro de un loop running.
- **Capturar body 2 veces (audit + timing)**: `await request.body()` es cacheado desde FastAPI 0.135+, pero timing middleware NO necesita el body — sólo lee `request.state`. Nunca leerlo.
- **Intentar matar thread tras timeout**: Python no permite `thread.kill()` sin `ctypes` hacky. Documentar como limitación, no pelearla.
- **Usar `float` para ms**: `Integer` es suficiente (ms nunca fraccional a nivel práctico).
- **Loggear query_log writes a `logger.info`**: genera ruido — sólo warn en slow, error en fallo.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Pub/sub cross-worker | In-memory event bus propio | Postgres LISTEN/NOTIFY | Postgres ya está ahí; evita Redis. `[CITED: neon.com pub-sub-listen-notify]` |
| CAS atómico de approvals | `SELECT ... FOR UPDATE` + `UPDATE` en 2 pasos | `UPDATE ... WHERE ... RETURNING` 1 paso | Más simple, atómico por naturaleza. `[CITED: postgresql.org docs sql-update]` |
| Cron scheduler | Thread propio con `time.sleep` | `asyncio.create_task` + `asyncio.sleep` | Integra con lifespan, cancelable, sin dep extra |
| Chart rendering | Canvas API raw / D3 custom | Chart.js (ya cargado) | 0 código extra, CDN estable |
| Modal bloqueante | Bootstrap modal / lib externa | Alpine.js `x-data` + `x-show` | Alpine ya cargado, 20 LOC |
| Password hashing | Custom bcrypt | Argon2id (Phase 2) | Ya resuelto |
| Rate limiting | Custom middleware | slowapi (ya en requirements) | Usarlo si Mark-IV lo requiere; en Mark-III innecesario para approvals |
| Session management | JWT propio | itsdangerous + tabla sessions (Phase 2) | Ya resuelto |
| SQL injection protection | String concat | SQLAlchemy `text()` + bindparams | Phase 3 ya fijó el patrón |
| Timeout thread | Kill thread via ctypes | Soft timeout + logging | No es posible matar thread Python; documentar |
| Date arithmetic humanize | Custom `"%d min %d s"` builder | Función helper simple `humanize_ms` (~10 LOC) | Sí es hand-rolled, aceptable por ser trivial |

**Key insight:** Todas las primitivas robustas (pub/sub, CAS, async) ya están en el stack. Phase 4 compone, no inventa.

## Runtime State Inventory

> N/A — Phase 4 es greenfield (no rename/refactor). No hay strings heredadas que buscar ni state migration. Las 3 tablas nuevas se crean con `NEXO_AUTO_MIGRATE=true` o manualmente vía `make nexo-init` extendido.

**Excepción**: decisión consciente de **extender `schema_guard.py`** para incluir las 3 nuevas tablas en `CRITICAL_TABLES` (hoy son 8; tras Phase 4 pasan a 11). Si esto se olvida, el schema_guard no detecta drift de las nuevas tablas.

## Common Pitfalls

### Pitfall 1: `asyncio.wait_for` NO cancela el thread subyacente
**What goes wrong:** El timeout de 15 min dispara `TimeoutError`, se responde 504 al browser, pero matplotlib sigue consumiendo RAM y tiempo CPU en background. El siguiente pipeline arranca sin que el primero haya terminado — memoria se acumula.
**Why it happens:** Python no permite killar threads limpiamente (la CPython API no lo expone). `asyncio.wait_for` sólo cancela la corrutina que espera.
**How to avoid:**
- Documentar que el timeout es "soft" a nivel UX.
- El semáforo **se libera al salir del `async with`**, lo cual puede permitir 4º pipeline mientras el 1º sigue agonizando.
- Opción mitigación: wrappear con `threading.Event` que el pipeline checka en cada iteración (requiere modificar `run_pipeline` para cooperar). Diferido a Mark-IV.
**Warning signs:** Memoria del container creciendo de forma monotónica después de timeouts repetidos.
**Source:** `[CITED: docs.python.org/3/library/asyncio-task.html]` + `[CITED: github.com/python/cpython/issues/87185]`

### Pitfall 2: LISTEN/NOTIFY pierde mensajes si la conexión cae
**What goes wrong:** Worker uvicorn pierde brevemente conexión a Postgres (network blip, DB restart). Mensajes NOTIFY emitidos durante ese periodo se pierden para siempre — Postgres NO los encola.
**Why it happens:** LISTEN/NOTIFY es fire-and-forget por diseño. `[CITED: postgresql.org/docs/current/sql-notify.html]`
**How to avoid:**
- Safety net D-19: `get(endpoint)` fuerza `full_reload()` si `now - _loaded_at_global > 5min`. Incluso si todos los NOTIFY fallan, el cache se re-sincroniza cada 5 min.
- El LISTEN loop reconecta automáticamente (`while True: try: connect + listen except: sleep(5)`).
- Test manual: hacer `docker compose restart db` mientras hay cambios de thresholds pendientes y verificar que se propagan en <5 min.
**Warning signs:** Logs `thresholds_cache stale — forzando full_reload` muy frecuentes indican LISTEN caído.

### Pitfall 3: Middleware order silenciosamente roto
**What goes wrong:** Si `QueryTimingMiddleware` se registra ANTES de `AuditMiddleware`, el timing no ve `request.state.user`. Si se registra DESPUÉS de los routers (imposible con `add_middleware`, pero fácil de romper si se mueve a `@app.middleware("http")` decorator), la cadena se invierte.
**Why it happens:** Starlette usa LIFO para `add_middleware` — último registrado es outer-most. Documentación oficial no lo hace evidente.
**How to avoid:**
- Comentario explícito en `api/main.py` explicando el orden (ya existe para auth/audit; añadir nota para timing).
- Test en `tests/middleware/test_query_timing.py` que verifica `request.state.user` es legible en el middleware (FastAPI TestClient con cookie de sesión válida).
**Source:** `[CITED: fastapi.tiangolo.com/tutorial/middleware/]`

### Pitfall 4: Race condition en consume_approval
**What goes wrong:** Usuario clickea "Ejecutar con aprobación" 2 veces rápidamente. 2 requests simultáneas con mismo `approval_id`. Sin CAS atómico, ambas ejecutan el pipeline → doble coste.
**Why it happens:** ORM fetch + update en 2 queries separadas no es atómico.
**How to avoid:**
- Patrón `UPDATE ... WHERE consumed_at IS NULL RETURNING id` (Pattern 4 arriba). Una sola query SQL.
- Si `RETURNING` devuelve 0 filas → el otro request ya lo consumió → 403.
**Source:** `[CITED: postgresql.org docs update]` + `[CITED: cybertec-postgresql.com]`

### Pitfall 5: psycopg2 LISTEN bloquea el worker si se llama sync
**What goes wrong:** `conn.poll()` + `conn.notifies` son síncronos. Si se llaman desde el event loop directamente, bloquean a uvicorn.
**Why it happens:** psycopg2 no tiene variant async nativa.
**How to avoid:**
- Wrappear en `asyncio.to_thread` (Pattern 5). El thread hace el `select.select([conn], [], [], 5)` blocante; el event loop sigue libre.
**Source:** `[CITED: the-fonz.gitlab.io/posts/postgres-notify]` + `[CITED: github.com/fastapi/fastapi/issues/5015]`

### Pitfall 6: Factor de aprendizaje amplifica outliers
**What goes wrong:** Usuario arranca pipeline de prueba con 1 recurso × 1 día. Tarda 200ms (caché warm, sin PDFs grandes). Siguiente auto-refresh del factor pone `factor = 200ms`. Próximo pipeline real (10 × 30) estima 60s, pero tarda 10 min. Toda la UI queda "green" cuando debería ser "red".
**Why it happens:** Median sobre muestras pequeñas es volátil.
**How to avoid:**
- `query_log_cleanup` y `factor_auto_refresh` filtran por `status IN ('ok', 'slow')` — excluye `timeout` y `error`.
- Añadir filtro mínimo: `WHERE actual_ms > 500` (excluye test runs triviales).
- Requerir mínimo N=10 muestras antes de permitir recalc (botón UI deshabilitado si hay <10 runs); mostrar "factor no calibrado" si `factor_updated_at IS NULL`.
**Warning signs:** `/ajustes/rendimiento` muestra divergencia_% >200% de forma sostenida → factor desalineado.

### Pitfall 7: Cookie/session expira durante flujo aprobación largo
**What goes wrong:** Usuario solicita aprobación. Se va de fin de semana. Vuelve lunes, propietario aprueba miércoles. Usuario clickea "Ejecutar" el jueves. Su sesión expiró (TTL 12h). Redirige a /login → pierde contexto.
**Why it happens:** Sesión HttpOnly + sliding 12h.
**How to avoid:**
- Esto es un caso aceptable — UX estándar. El usuario logea, va a `/mis-solicitudes`, ve su aprobación, clickea "Ejecutar ahora".
- Documentar en texto del modal red: "Cuando el propietario apruebe, podrás ejecutar desde 'Mis solicitudes'".
- Log audit_log cuando un approval queda `approved` más de 24h sin consumir (low priority).

### Pitfall 8: /ajustes/rendimiento lento con 90d × 4 endpoints
**What goes wrong:** Propietario abre `/ajustes/rendimiento` con filtro `90d + todos endpoints + custom range`. Query hace `SELECT ts, endpoint, estimated_ms, actual_ms FROM query_log WHERE ts > now() - '90 days'` → ~1000 filas/día × 90 = 90k filas → frontend lento.
**Why it happens:** Sin paginación ni agregación server-side.
**How to avoid:**
- `/api/rendimiento/summary`: agregación SQL (`SELECT endpoint, COUNT(*), AVG, PERCENTILE_CONT(...) GROUP BY endpoint`) — devuelve 4 filas.
- `/api/rendimiento/timeseries`: bucketing por hora/día según rango (`date_trunc('hour', ts)` para 7d, `date_trunc('day', ts)` para 90d). Chart.js recibe ~200 puntos máximo.
- Índice compuesto `(endpoint, ts DESC)` acelera ambas.
**Warning signs:** `SELECT ... FROM query_log ...` > 500ms en logs del DB.

### Pitfall 9: Audit middleware lee body 2 veces
**What goes wrong:** Si Audit y Timing ambos leen `await request.body()`, y el body stream no está cacheado (versión FastAPI antigua), el segundo `read()` devuelve bytes vacíos → router recibe body vacío → 400.
**Why it happens:** Por defecto Starlette request body es stream consumible una vez.
**How to avoid:**
- FastAPI 0.135.3 (en requirements) ya cachea body tras primera lectura. Audit lo lee, Timing NO lo necesita (lee `request.state.params_json` que el router pobló antes).
- Test explícito en `tests/middleware/test_timing_doesnt_consume_body.py`.
**Source:** `[CITED: fastapi.tiangolo.com/tutorial/middleware/]`

### Pitfall 10: Pipeline tmp_root leak tras timeout
**What goes wrong:** Timeout 15 min dispara. Thread sigue. El `finally: shutil.rmtree(tmp_root)` SÍ se ejecuta eventualmente cuando matplotlib termina, pero si matplotlib se cuelga infinitamente, el tmp dir queda en `/tmp/oee_*/`.
**Why it happens:** Limpieza depende de que el thread termine.
**How to avoid:**
- Garbage collector periódico de `/tmp/oee_*` con `mtime > 2h`. Añadir a `cleanup_scheduler` o cron del host.
- Baja prioridad — impacto es disco, no funcionalidad.

## Code Examples

### Endpoint preflight (nuevo)

```python
# api/routers/pipeline.py — AÑADIR
from fastapi import Depends
from api.deps import DbNexo
from nexo.services.preflight import estimate_cost, Estimation
from nexo.services.auth import require_permission


@router.post("/preflight")
def preflight(
    req: PipelineRequest,
    user=Depends(require_permission("pipeline:run")),
) -> Estimation:
    """Estima coste SIN ejecutar. Frontend decide green/amber/red."""
    params = {
        "fecha_inicio": req.fecha_inicio,
        "fecha_fin": req.fecha_fin,
        "recursos": req.recursos or [],
        "n_recursos": len(req.recursos or []),
        "n_dias": (req.fecha_fin - req.fecha_inicio).days + 1,
    }
    return estimate_cost("pipeline/run", params)
```

### Endpoint run con force + approval_id

```python
# api/routers/pipeline.py — MODIFICAR el /run existente
from typing import Optional

@router.post("/run", dependencies=[Depends(require_permission("pipeline:run"))])
async def run(
    req: PipelineRequest,
    request: Request,
    db: DbNexo,
    force: bool = False,
    approval_id: Optional[int] = None,
    user=Depends(require_permission("pipeline:run")),
):
    params = {
        "fecha_inicio": req.fecha_inicio.isoformat(),
        "fecha_fin": req.fecha_fin.isoformat(),
        "recursos": req.recursos or [],
        "n_recursos": len(req.recursos or []),
        "n_dias": (req.fecha_fin - req.fecha_inicio).days + 1,
    }

    # Preflight siempre
    est = estimate_cost("pipeline/run", params)
    request.state.estimated_ms = est.estimated_ms
    request.state.params_json = json.dumps(params, sort_keys=True)

    # Gate: amber/red requieren force + approval_id
    if est.level == "red":
        if not (force and approval_id):
            raise HTTPException(403, f"{est.reason}. Solicita aprobación.")
        approval = consume_approval(
            db, approval_id=approval_id, user_id=user.id, current_params=params,
        )
        request.state.approval_id = approval.id
    elif est.level == "amber":
        if not force:
            raise HTTPException(
                status_code=428,  # Precondition Required
                detail={"estimation": est.__dict__, "action": "confirm_amber"},
            )

    # Ejecutar con semáforo + to_thread
    from nexo.services.pipeline_lock import pipeline_semaphore, PIPELINE_TIMEOUT_SEC
    async with pipeline_semaphore:
        # Opción A: collect all messages
        def worker() -> list[str]:
            return list(run_pipeline(
                fecha_inicio=req.fecha_inicio, fecha_fin=req.fecha_fin,
                modulos=req.modulos, source=req.source, recursos=req.recursos,
            ))
        try:
            messages = await asyncio.wait_for(
                asyncio.to_thread(worker), timeout=PIPELINE_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            raise HTTPException(504, f"Pipeline timeout ({PIPELINE_TIMEOUT_SEC}s)")

    # Stream replay (el usuario ve todos los mensajes de golpe al final)
    def event_stream():
        for msg in messages:
            yield f"data: {msg}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

### Alpine modal amber component

```html
<!-- templates/pipeline.html — AÑADIR -->
<!-- Source: Alpine.js 3.14.8 ya cargado en base.html -->
<div x-data="pipelineRunner()" x-cloak>
  <button @click="attempt()" class="btn-primary">Ejecutar pipeline</button>

  <!-- Modal AMBER -->
  <div x-show="modalLevel === 'amber'"
       x-transition.opacity
       class="fixed inset-0 bg-black/60 flex items-center justify-center z-50"
       @keydown.escape.window="cancel()">
    <div class="bg-white rounded-lg p-6 max-w-md">
      <h2 class="text-xl font-bold">Esta operación puede tardar</h2>
      <p class="text-2xl font-bold my-2" x-text="humanize(estimation.estimated_ms)"></p>
      <p class="text-sm text-gray-500" x-text="estimation.breakdown"></p>
      <p class="text-sm text-gray-500 mt-2">
        La UI seguirá respondiendo mientras la operación corre.
      </p>
      <div class="flex gap-2 justify-end mt-4">
        <button @click="cancel()" class="btn-ghost">Cancelar</button>
        <button @click="confirmRun()" class="btn-primary">Continuar</button>
      </div>
    </div>
  </div>

  <!-- Modal RED -->
  <div x-show="modalLevel === 'red'" x-transition.opacity
       class="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
    <div class="bg-white rounded-lg p-6 max-w-md border-t-4 border-red-500">
      <h2 class="text-xl font-bold text-red-700">Operación requiere aprobación</h2>
      <p class="text-2xl font-bold my-2" x-text="humanize(estimation.estimated_ms)"></p>
      <p class="text-sm text-gray-500" x-text="estimation.reason"></p>
      <div class="flex gap-2 justify-end mt-4">
        <button @click="cancel()" class="btn-ghost">Cancelar</button>
        <button @click="requestApproval()" class="btn-primary">Solicitar aprobación</button>
      </div>
    </div>
  </div>
</div>

<script>
function pipelineRunner() {
  return {
    modalLevel: null,
    estimation: null,
    currentParams: null,

    async attempt() {
      this.currentParams = this.collectParams();  // impl específica del form
      const resp = await fetch('/api/pipeline/preflight', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(this.currentParams),
      });
      this.estimation = await resp.json();
      if (this.estimation.level === 'green') return this.doRun(false);
      this.modalLevel = this.estimation.level;
    },

    async confirmRun() {
      this.modalLevel = null;
      await this.doRun(true);
    },

    async doRun(force, approvalId = null) {
      const url = new URL('/api/pipeline/run', window.location.origin);
      if (force) url.searchParams.set('force', 'true');
      if (approvalId) url.searchParams.set('approval_id', approvalId);
      // SSE stream handling...
    },

    async requestApproval() {
      const resp = await fetch('/api/approvals', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          endpoint: 'pipeline/run',
          params: this.currentParams,
          estimated_ms: this.estimation.estimated_ms,
        }),
      });
      const data = await resp.json();
      this.modalLevel = null;
      window.showToast({
        type: 'success',
        msg: `Solicitud enviada. Ver en Mis Solicitudes (#${data.approval_id}).`,
        link: '/mis-solicitudes',
      });
    },

    cancel() {
      this.modalLevel = null;
      this.estimation = null;
    },

    humanize(ms) {
      if (ms < 1000) return `~${ms}ms`;
      const s = Math.round(ms / 1000);
      if (s < 60) return `~${s}s`;
      const m = Math.floor(s / 60);
      const sr = s % 60;
      return `~${m} min ${sr}s`;
    },

    collectParams() { /* read from form */ return {}; },
  };
}
</script>
```

### Chart.js timeseries snippet

```javascript
// templates/ajustes_rendimiento.html inline JS
// Source: [CITED: chartjs.org/docs/latest/charts/line.html]
async function loadChart() {
  const resp = await fetch('/api/rendimiento/timeseries?endpoint=pipeline/run&days=30');
  const data = await resp.json();  // {points: [{ts, estimated, actual}, ...]}
  const ctx = document.getElementById('perfChart').getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.points.map(p => p.ts),
      datasets: [
        {label: 'Estimado (ms)', data: data.points.map(p => p.estimated), borderColor: '#5995ff'},
        {label: 'Real (ms)',     data: data.points.map(p => p.actual),    borderColor: '#dc2626'},
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {x: {type: 'time'}, y: {beginAtZero: true}},
    },
  });
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| JWT stateless sessions | itsdangerous signed cookies + sessions table | Phase 2 | No se replica — consistente |
| `@app.middleware("http")` decorator | `BaseHTTPMiddleware` class subclassing | Starlette 0.27+ / FastAPI 0.100+ | Decorador tiene limitations con streaming responses (StreamingResponse + middleware decorator puede consumir body early). Usar `BaseHTTPMiddleware` class-based. `[CITED: fastapi.tiangolo.com/advanced/middleware/]` |
| `datetime.utcnow()` | `datetime.now(timezone.utc)` | Python 3.12+ | utcnow deprecated. Ya aplicado en Phase 2/3. |
| APScheduler clásico | `asyncio.create_task` loop | — | En Mark-III se prefiere no añadir deps; APScheduler queda para Mark-IV si hace falta cron más sofisticado. |
| Chart.js 3.x | Chart.js 4.x | 4.0 released Oct 2022 | Breaking changes minimales; 4.4.7 ya en base.html. |

**Deprecated/outdated:**
- `asyncio.get_event_loop()` dentro de corutinas — usar `asyncio.get_running_loop()`. Aplicar en código nuevo.
- `async for` sobre generators síncronos — no funciona; usar queue bridge (Pattern 3 Opción B).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Opción A (collect messages) del pipeline wrapping es UX aceptable (el operador no requiere SSE en tiempo real durante el run) | Pattern 3 | Medio: si el operador necesita feedback continuo, Opción B es mandatory. Planner debe verificar con usuario en plan 04-02. |
| A2 | Factor seed inicial `2000ms/recurso×día` es del orden correcto (±50%) | D-04 / Pattern 2 | Bajo-Medio: si está lejos del real, primeras semanas muestran amber/red falsos. Mitigado por botón manual de recalc tras 30 runs. Sin datos históricos de matplotlib + hardware LAN, hay que empezar en algún punto. |
| A3 | Workers uvicorn = 1 en LAN (no se escala a multi-worker en Mark-III) | §Arquitectura + §Pipeline Lock | Medio: si se escala, el semáforo por-proceso permite 3×N concurrencia en vez de 3 global. Aceptado por D-18 defaults. |
| A4 | PostgreSQL 16 soporta NOTIFY con payload hasta 8000 bytes (default) | Pattern 5 | Bajo: payload es `endpoint` string (~50 bytes). No hay risk. `[ASSUMED]` — verificar en docs si se decide enviar JSON payloads grandes. |
| A5 | `status='approved_run'` no es necesario como status separado en query_log | Pattern 4 | Bajo: el campo `approval_id` ya discrimina. Planner puede ajustar si UX lo requiere. |
| A6 | No hay requirement de atomic transaction que envuelva `consume_approval` + `run_pipeline` | Pattern 4 | Medio: si el pipeline falla después de consumir el approval, el usuario necesita solicitar una nueva. Aceptado en D-15 (single-use = 1 intento). |
| A7 | `params_json` equality comparison funciona con `json.dumps(sort_keys=True)` | Pattern 4 | Bajo: sort_keys normaliza orden; espacios/encoding pueden variar. Mitigación: definir helper `_canonical_json(obj)` en approvals.py. |
| A8 | El endpoint `/api/rendimiento/timeseries` puede bucketizar con `date_trunc('hour'|'day', ts)` según rango | Pitfall 8 | Bajo: estándar Postgres. |

**If this table is empty:** N/A — hay 8 assumptions. Las A1 y A2 son las más relevantes para el planner.

## Open Questions

1. **¿Opción A vs B para pipeline wrapping?**
   - What we know: A es trivial (5 LOC cambio en pipeline.run), B mantiene SSE en tiempo real (~60 LOC extra + testing complejo).
   - What's unclear: ¿el operador del taller tolera UI sin progreso durante 5-15 min?
   - Recommendation: plan 04-02 arranca con Opción A, planner puede mover a B en un commit separado si feedback operador lo pide.

2. **¿TTL de la cookie extendida durante flujo approval largo?**
   - What we know: Cookie sliding 12h. Flujo approval puede tomar días.
   - What's unclear: ¿auto-re-login desde `/mis-solicitudes` o requerir relogin?
   - Recommendation: aceptar comportamiento estándar (expira → relogin → de vuelta a `/mis-solicitudes`). Link desde email NO aplica porque no hay SMTP.

3. **¿Persistir `factor` individual por recurso o global por endpoint?**
   - What we know: D-04 dice global por endpoint. Pero algunos recursos son más lentos (más datos).
   - What's unclear: Mark-IV podría querer granularidad por recurso.
   - Recommendation: mantener global Mark-III; añadir columna `recurso_factor_json` en `query_thresholds` en Mark-IV si hace falta.

4. **¿Cómo tratar `params_json` en Chart.js timeseries?**
   - What we know: El params_json puede ser grande (SQL completo). En timeseries no se muestra.
   - What's unclear: ¿filtrar en server-side no devolver params_json en summary/timeseries endpoint?
   - Recommendation: NO devolver params_json en summary/timeseries; sólo ids. Detalle en click en punto (modal opcional Mark-IV).

5. **¿Qué pasa si el propietario aprueba un request pero al consumirlo se descubre que el params equality falló?**
   - What we know: Consume devuelve 403 "Parámetros cambiaron". Usuario tiene que solicitar nueva aprobación.
   - What's unclear: ¿avisar al propietario que hubo un attempt con params distintos?
   - Recommendation: simplemente loggear a audit_log; propietario lo ve en `/ajustes/auditoria` filtrando por path=/api/pipeline/run con status=403.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12+ asyncio | Middleware, to_thread, LISTEN wrapper | ✓ | 3.12 (Docker base) | — |
| PostgreSQL 16 + NOTIFY/LISTEN | Thresholds cache pub/sub | ✓ | 16 (docker-compose.yml) | — (sin fallback viable) |
| psycopg2-binary | LISTEN loop, engine_nexo | ✓ | 2.9.11 | — |
| fastapi | Middleware, routers | ✓ | 0.135.3 | — |
| sqlalchemy | ORM modelos | ✓ | 2.0.49 | — |
| matplotlib | Pipeline (ya en uso) | ✓ | 3.10.8 | — |
| Chart.js CDN | `/ajustes/rendimiento` gráfica | ✓ | 4.4.7 (loaded in base.html) | Tabla sin gráfica si CDN falla |
| Alpine.js CDN | Modal amber/red | ✓ | 3.14.8 (base.html) | — (CDN es crítico, sin fallback) |
| HTMX CDN | Badge sidebar polling | ✓ | 2.0.4 (base.html) | — |
| argon2-cffi | Auth (Phase 2) | ✓ | 25.1.0 | — |
| itsdangerous | Sessions (Phase 2) | ✓ | 2.2.0 | — |
| slowapi | Rate limiting (opcional en approvals) | ✓ | 0.1.9 | — |
| docker compose | Tests con Postgres real | ✓ | — | — |

**Missing dependencies with no fallback:** Ninguna — todas las deps requeridas ya están en el stack tras Phase 3.

**Missing dependencies with fallback:** Ninguna.

**Nuevas deps opcionales (descartadas):**
- APScheduler: descartado, `asyncio.create_task` suffices.
- asyncpg: descartado, psycopg2 + thread wrap suffices.
- sqlalchemy[async]: descartado, sync ORM suffices.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest ≥8 (ya en `requirements-dev.txt`) |
| Config file | `pytest.ini` (existe) |
| Quick run command | `pytest tests/services/test_preflight.py -x` |
| Full suite command | `pytest tests/` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| QUERY-01 | query_log tabla + índices | unit (schema) | `pytest tests/data/test_schema_query_log.py -x` | ❌ Wave 0 |
| QUERY-01 | schema_guard incluye query_log | unit | `pytest tests/data/test_schema_guard_extended.py -x` | ❌ Wave 0 |
| QUERY-02 | query_thresholds CRUD + NOTIFY emitted | integration (Postgres real) | `pytest tests/services/test_thresholds_cache.py -x` | ❌ Wave 0 |
| QUERY-03 | preflight.estimate_cost green/amber/red para pipeline | unit (puro) | `pytest tests/services/test_preflight.py::test_pipeline_classify -x` | ❌ Wave 0 |
| QUERY-03 | factor learning algorithm (median con filtro status) | unit (puro) | `pytest tests/services/test_factor_learning.py -x` | ❌ Wave 0 |
| QUERY-04 | /preflight endpoint devuelve Estimation JSON | contract (TestClient) | `pytest tests/routers/test_pipeline_preflight.py -x` | ❌ Wave 0 |
| QUERY-04 | Endpoint devuelve 428 para amber sin force | contract | mismo archivo | ❌ Wave 0 |
| QUERY-05 | QueryTimingMiddleware escribe fila en query_log | integration | `pytest tests/middleware/test_query_timing.py -x` | ❌ Wave 0 |
| QUERY-05 | `actual_ms > warn_ms * 1.5` → status='slow' + warning log | integration | mismo archivo | ❌ Wave 0 |
| QUERY-05 | Middleware NO escribe en paths no listados | unit | `pytest tests/middleware/test_query_timing.py::test_excluded_paths -x` | ❌ Wave 0 |
| QUERY-06 | consume_approval CAS atomic (race test) | integration | `pytest tests/services/test_approvals_cas.py -x` | ❌ Wave 0 |
| QUERY-06 | cancel approval ownership check | integration | mismo archivo | ❌ Wave 0 |
| QUERY-06 | TTL expiry job marca approvals expired | integration | `pytest tests/services/test_approvals_cleanup.py -x` | ❌ Wave 0 |
| QUERY-07 | Umbrales editables vía `/api/thresholds` | contract | `pytest tests/routers/test_thresholds_crud.py -x` | ❌ Wave 0 |
| QUERY-07 | Los 4 endpoints invocan preflight correctamente | contract smoke | `pytest tests/routers/test_preflight_integration.py -x` | ❌ Wave 0 |
| QUERY-08 | Pipeline dentro de semáforo (max 3 concurrentes) | integration | `pytest tests/routers/test_pipeline_concurrency.py -x` (skipped en CI, manual) | ❌ Wave 0 |
| QUERY-08 | Timeout dispara 504 + query_log status='timeout' | integration | mismo archivo | ❌ Wave 0 |
| — | Modal amber/red UI | E2E manual | manual browser | N/A |
| — | Chart.js rendering | E2E manual | manual browser | N/A |
| — | LISTEN/NOTIFY cache invalidation <1s | integration docker | `tests/integration/test_listen_notify.py` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `pytest tests/services/ tests/middleware/ -x -q` (subconjunto rápido sin integration DB)
- **Per wave merge:** `pytest tests/ -x` (full suite, incluye integration contra Postgres real vía docker compose)
- **Phase gate:** Full suite green + manual E2E de modal amber + manual E2E de flujo red con approval antes de `/gsd-verify-work 4`.

### Wave 0 Gaps

- [ ] `tests/services/test_preflight.py` — unit tests de `estimate_cost` para los 4 endpoints (QUERY-03, QUERY-04)
- [ ] `tests/services/test_factor_learning.py` — median cálculo con filtros (QUERY-03)
- [ ] `tests/services/test_thresholds_cache.py` — cache + LISTEN/NOTIFY (QUERY-02, D-19)
- [ ] `tests/services/test_approvals_cas.py` — consume_approval CAS atomic con race (QUERY-06)
- [ ] `tests/services/test_approvals_cleanup.py` — TTL expiry job (D-14)
- [ ] `tests/middleware/test_query_timing.py` — middleware escribe query_log correctamente (QUERY-05)
- [ ] `tests/middleware/test_timing_doesnt_consume_body.py` — audit+timing coexisten (Pitfall 9)
- [ ] `tests/routers/test_pipeline_preflight.py` — endpoint /preflight returns Estimation (QUERY-04)
- [ ] `tests/routers/test_thresholds_crud.py` — PUT/GET /api/thresholds emite NOTIFY (QUERY-02)
- [ ] `tests/routers/test_preflight_integration.py` — 4 endpoints invocan preflight (QUERY-07)
- [ ] `tests/routers/test_pipeline_concurrency.py` — semáforo + timeout (QUERY-08), skip CI, run dev/preprod
- [ ] `tests/data/test_schema_query_log.py` — tabla + índices creados (QUERY-01)
- [ ] `tests/data/test_schema_guard_extended.py` — guard incluye las 3 tablas nuevas
- [ ] `tests/integration/test_listen_notify.py` — cache invalidation <1s (D-19)
- [ ] `tests/conftest.py` extensión: fixture `thresholds_cache_empty` para isolation entre tests
- [ ] No requiere nuevo framework — pytest ya configurado

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No (Phase 2 resuelto) | — |
| V3 Session Management | No (Phase 2 resuelto) | — |
| V4 Access Control | yes | `require_permission(...)` ya existe; aplicar a los nuevos endpoints: `/api/rendimiento/*` (propietario), `/api/approvals` (user crea, propietario aprueba), `/api/thresholds` (propietario), `/mis-solicitudes` (self-read only) |
| V5 Input Validation | yes | pydantic models para bodies; whitelist ya existe en `bbdd._validate_sql` |
| V6 Cryptography | No (no nuevos secretos) | — |
| V7 Error Handling | yes | `global_exception_handler` ya devuelve UUID sin traceback (Sprint 0) — verificar que nuevos endpoints no regresan |
| V8 Data Protection | yes | `params_json` puede contener SQL + datos sensibles. D-09 aplica whitelist redact; D-10 retention 90d. |
| V9 Communication | No (LAN-only) | — |
| V10 Malicious Code | No | — |
| V11 Business Logic | yes | CAS atomic de approval consume previene double-spend; params equality previene request tampering |
| V12 File and Resources | Parcial | Pipeline crea tmp_root en /tmp; cleanup en finally |

### Known Threat Patterns for FastAPI + Postgres Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| SQL injection via params_json | Tampering | Logged as-is (text column); no se re-evalúa. SQLAlchemy `text()` + bindparams para cualquier query que use estos datos. |
| Approval ID enumeration | Information Disclosure | Integer ID auto-increment no leak más que la cardinalidad; alternativa UUID si se requiere. **Decisión**: integer OK para Mark-III, /mis-solicitudes filtra por user_id. |
| Approval tampering (user modifica params entre aprobación y consumo) | Tampering | D-15 punto 5: equality check de `params_json` en `consume_approval`. |
| Approval double-consume (race) | Repudiation | `UPDATE ... WHERE consumed_at IS NULL RETURNING id` (CAS atomic). |
| Preflight bypass (request directo al /run sin pasar por /preflight) | Elevation of Privilege | `/run` SIEMPRE ejecuta `estimate_cost` internamente; el `/preflight` público es informativo, no gatekeeper. |
| Denial of Service via múltiples pipeline concurrentes | Denial of Service | Semáforo(3); 4º en cola (UI muestra wait message). |
| Log injection en query_log via params | Tampering | JSON se serializa, no se inyecta en path/endpoint. Text column en Postgres = seguro. |
| Threshold modification sin audit | Repudiation | Endpoint `PUT /api/thresholds/:endpoint` graba `updated_by` + audit middleware lo graba automáticamente. |

**Principal preocupación**: `params_json` puede contener 500-5000 chars de SQL. No es ejecutado nunca después de grabarse — sólo es informativo. Text column Postgres lo acepta sin escape issues. Retention 90d (D-10) limita exposición histórica.

## Project Constraints (from CLAUDE.md)

- **Naming**: prefijo `NEXO_*` para nuevas env vars: `NEXO_PIPELINE_MAX_CONCURRENT`, `NEXO_PIPELINE_TIMEOUT_SEC`, `NEXO_APPROVAL_TTL_DAYS`, `NEXO_QUERY_LOG_RETENTION_DAYS`, `NEXO_AUTO_REFRESH_STALE_DAYS`. Documentar en `.env.example`.
- **Schemas**: todas las tablas nuevas en `nexo.*` (Postgres). NO mover a SQL Server.
- **Carpetas**: `nexo/middleware/` nueva, `nexo/services/` extendida. NO renombrar `OEE/`.
- **No bypass de hooks**: commits siguen pre-commit. En Phase 7 se añadirá ruff/black; aquí NO (aún no configurado).
- **Commit format**: Conventional Commits inglés título, body español. Tipo `feat` para nuevas features, `refactor` para modificar pipeline.py, `test` para tests nuevos, `docs` para README/BRANDING.
- **Atomicidad**: 1 commit = 1 cambio coherente. Lista orientativa de commits por plan en §Plan breakdown.
- **SMTP**: NO configurar. D-13 ya lo respeta (badge sidebar + polling, sin email).
- **No 2FA/LDAP**: respetado (no afecta a Phase 4).
- **Matplotlib**: NO sustituir. D-18 respeta.
- **Tests contra Postgres**: docker compose db up (Phase 3 established pattern).

## Plan Breakdown (Propuesta del Researcher)

**4 planes, orden recomendado secuencial** (paralelización posible 04-02 ∥ 04-03 tras 04-01 si equipo maneja dos hilos):

### Plan 04-01 — Foundation (bloqueante para 02/03/04)

**Scope:**
- 3 modelos ORM nuevos en `nexo/data/models_nexo.py` (`NexoQueryLog`, `NexoQueryThreshold`, `NexoQueryApproval`) + índices.
- `nexo.data.repositories.nexo`: añadir `QueryLogRepo`, `ThresholdRepo`, `ApprovalRepo` (métodos skeleton).
- Extender `schema_guard.CRITICAL_TABLES` de 8 a 11.
- `scripts/init_nexo_schema.py` extendido: crea nuevas tablas + seeds de `query_thresholds` con defaults D-01/D-02/D-03.
- `nexo/services/thresholds_cache.py` skeleton (dict + `full_reload` + `get`; sin LISTEN todavía).
- Pydantic `Estimation` en `api/models.py`.
- Tests: `test_schema_query_log.py`, `test_schema_guard_extended.py`, `test_thresholds_cache_basic.py` (sin LISTEN).

**Requirements:** QUERY-01 (partial), QUERY-02 (partial — cache skeleton).

**Exit gate:** schema_guard verifica 11 tablas, tests pasan, `/ajustes/rendimiento` NO existe todavía pero DB tiene las tablas con seeds.

**Estimación:** 1 día focalizado.

### Plan 04-02 — Preflight + Middleware (depende de 04-01)

**Scope:**
- `nexo/services/preflight.py`: `estimate_cost` completo para los 4 endpoints.
- `nexo/middleware/query_timing.py`: `QueryTimingMiddleware` + wiring en `api/main.py`.
- `nexo/services/pipeline_lock.py`: semáforo + timeout constants.
- Refactor de `api/routers/pipeline.py`: añadir `POST /preflight` + modificar `POST /run` para aceptar `force` + `approval_id` + semáforo + `asyncio.to_thread`.
- Refactor de `api/routers/bbdd.py` `/query`: inyectar preflight tras `_validate_sql`.
- Refactor de `api/routers/capacidad.py` y `operarios.py`: preflight si rango > 90d.
- Modal amber/red en `templates/pipeline.html` + `templates/bbdd.html` + Alpine component en `static/js/app.js`.
- Tests: `test_preflight.py`, `test_query_timing.py`, `test_pipeline_preflight.py`, `test_preflight_integration.py`, `test_pipeline_concurrency.py` (skip CI).

**Requirements:** QUERY-03, QUERY-04, QUERY-05, QUERY-07 (parcial), QUERY-08.

**Exit gate:** arranco pipeline amber → veo modal → Continuar ejecuta → query_log tiene fila con estimated+actual.

**Estimación:** 2 días focalizados.

### Plan 04-03 — Approval Flow (depende de 04-01; puede paralelizar con 04-02)

**Scope:**
- `nexo/services/approvals.py`: `create_approval`, `consume_approval` (CAS atomic), `cancel`, `reject`, `expire`.
- `api/routers/approvals.py`: `POST /api/approvals`, `GET /api/approvals/:id`, `POST /api/approvals/:id/approve`, `POST /api/approvals/:id/reject`, `POST /api/approvals/:id/cancel`, `GET /api/approvals/count`.
- `/mis-solicitudes` page: template + route en pages.py (o nuevo router).
- `/ajustes/solicitudes` page: template + route (propietario only).
- Badge sidebar HTMX: modificar `base.html` para incluir badge en item "Ajustes" (condicional rol=propietario).
- `nexo/services/approvals_cleanup.py`: job TTL 7d expire.
- `nexo/services/cleanup_scheduler.py`: loop asyncio + wire en lifespan.
- Tests: `test_approvals_cas.py`, `test_approvals_cleanup.py`.

**Requirements:** QUERY-06.

**Exit gate:** modal red → solicito aprobación → propietario en /ajustes/solicitudes aprueba → vuelvo a /mis-solicitudes → "Ejecutar" dispara pipeline. Cancelled/Rejected flows también OK.

**Estimación:** 1.5-2 días focalizados.

### Plan 04-04 — Observability UI + LISTEN/NOTIFY (depende de 04-02 para tener datos; y 04-03 para approvals visibles)

**Scope:**
- `api/routers/rendimiento.py`: `GET /api/rendimiento/summary`, `GET /api/rendimiento/timeseries`.
- `/ajustes/rendimiento` page: template + route + Chart.js integration + filtros dropdown.
- `/ajustes/limites` page: template + route + CRUD thresholds + botón "Recalcular factor".
- `nexo/services/factor_auto_refresh.py`: job mensual.
- `nexo/services/query_log_cleanup.py`: job retention 90d.
- `nexo/services/thresholds_cache.py`: completar LISTEN/NOTIFY (start_listener + _blocking_listen_forever).
- `PUT /api/thresholds/:endpoint`: CRUD + emit NOTIFY.
- Tests: integration `test_listen_notify.py`, `test_thresholds_crud.py`, manual E2E del dashboard.
- docs: `.env.example` con env vars nuevas; `docs/BRANDING.md` o nueva sección README con comportamiento de retention y learning.

**Requirements:** QUERY-02 (complete), QUERY-07 (complete — UI límites), D-19, D-20.

**Exit gate:** edito threshold en UI → el otro worker reconoce en <1s → `/ajustes/rendimiento` muestra summary + gráfica. `/ajustes/limites` muestra botón recalc con preview.

**Estimación:** 2 días focalizados.

**Total estimación Phase 4:** 6.5-7 días focalizados (consistente con `docs/MARK_III_PLAN.md` Sprint 3 = 4 días — la research revela que el scope real exige +2-3 días por approval flow + LISTEN/NOTIFY).

## Sources

### Primary (HIGH confidence)

- `[CITED: fastapi.tiangolo.com/tutorial/middleware/]` — FastAPI middleware patrón canónico.
- `[CITED: fastapi.tiangolo.com/advanced/middleware/]` — `BaseHTTPMiddleware` class-based vs decorator.
- `[CITED: postgresql.org/docs/current/sql-notify.html]` — NOTIFY semántica oficial y entrega best-effort.
- `[CITED: postgresql.org/docs/current/sql-update.html]` — UPDATE...RETURNING atomicidad.
- `[CITED: docs.python.org/3/library/asyncio-task.html]` — `asyncio.to_thread`, `wait_for` semántica.
- `[CITED: github.com/python/cpython/issues/87185]` — "wait_for(to_thread) does not kill thread" — bug tracker oficial.
- `[VERIFIED: requirements.txt]` — todas las versiones confirmadas.
- `[VERIFIED: templates/base.html lines 6-10]` — CDN loadings de Chart.js 4.4.7, Alpine 3.14.8, HTMX 2.0.4.
- `[VERIFIED: api/main.py lines 138-139]` — orden actual middleware confirmed.
- `[VERIFIED: npm view chart.js 4.5.1 latest]` — versión actual (gap minor vs 4.4.7 loaded).

### Secondary (MEDIUM confidence, WebSearch verified)

- `[CITED: github.com/fastapi/fastapi/issues/5015]` — Permanently running background task listening to Postgres notifications.
- `[CITED: medium.com/@diwasb54]` — Real-Time Communication with PostgreSQL LISTEN/NOTIFY and FastAPI.
- `[CITED: the-fonz.gitlab.io/posts/postgres-notify]` — Playing with Postgres NOTIFY/LISTEN using Python asyncio and psycopg2.
- `[CITED: enterprisedb.com — listening-postgres-listen-notify]` — Delivery guarantees.
- `[CITED: cybertec-postgresql.com — transaction-anomalies-select-for-update]` — CAS patterns.
- `[CITED: apscheduler.readthedocs.io]` — APScheduler reference (descartado para Mark-III).
- `[CITED: runebook.dev/asyncio.to_thread]` — GIL behavior con to_thread.
- `[CITED: labs.quansight.org — scaling-asyncio-free-threaded-python]` — Python 3.14 cambios.
- `[CITED: chartjs.org/docs/latest/charts/line.html]` — Chart.js line chart API.

### Tertiary (LOW confidence, WebSearch only — flagged)

- `[ASSUMED]` Performance asumida de `monotonic()` vs `perf_counter()` para sub-ms precision — ambos son aceptables para ms resolution.
- `[ASSUMED]` 200MB RAM por pipeline matplotlib — estimación basada en experiencia típica, no medida.
- `[ASSUMED]` Seed factor 2000ms/recurso×día — D-04 lo decide, no está medido.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — todas las deps verificadas en requirements.txt y base.html.
- Middleware pattern: HIGH — patrón oficial FastAPI, implementación análoga a Phase 2 auth+audit.
- LISTEN/NOTIFY wrapper: MEDIUM-HIGH — patrón establecido en la comunidad, sin API oficial clean en psycopg2, probado en ejemplos citados.
- asyncio.to_thread semántica: HIGH — bug tracker oficial + docs Python.
- Pipeline Opción A vs B: MEDIUM — Opción A es segura pero pierde SSE; recomendación queda sujeta a feedback operador.
- Preflight heurística: MEDIUM — sin datos históricos, el seed factor es best-guess.
- Approval CAS: HIGH — patrón canónico Postgres.
- Chart.js: HIGH — ya cargado en el proyecto.
- Schema design: HIGH — derivado de REQUIREMENTS QUERY-01/02 + D-14/15/16.
- Cleanup scheduler: HIGH — `asyncio.create_task` es simple y estándar.
- Plan breakdown: HIGH — 4 planes alineados con QUERY-01..08 + CONTEXT decisiones.

**Research date:** 2026-04-19
**Valid until:** 2026-05-19 (30 días — stack estable, sin upgrades forzados esperados)

---

*Phase: 04-consultas-pesadas*
*Research completed: 2026-04-19*
