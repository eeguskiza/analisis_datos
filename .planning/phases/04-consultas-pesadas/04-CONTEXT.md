# Phase 4: Consultas pesadas — Context

**Gathered:** 2026-04-19
**Status:** Ready for research
**Source:** síntesis de `docs/MARK_III_PLAN.md` §Sprint 3, `.planning/REQUIREMENTS.md` QUERY-01..QUERY-08, `.planning/ROADMAP.md` Phase 4 + /gsd-discuss-phase sesión 2026-04-19 (8 gray areas, 3 decisiones explícitas del usuario, 17 delegadas al planner con defaults documentados).

<domain>
## Phase Boundary

**Objetivo**: introducir preflight que estima coste antes de ejecutar queries/pipelines caros, postflight que mide y alerta, tabla de umbrales editable desde UI, flujo de aprobación asíncrona para casos rojos, y `asyncio.to_thread` para que el pipeline OEE no congele la UI.

**Entregable end-to-end**:
- Tablas Postgres `nexo.query_log`, `nexo.query_thresholds`, `nexo.query_approvals` (esta última no está en REQUIREMENTS pero surge de D-15: necesaria para ligar approval_id al consumo single-use).
- Módulo `nexo/services/preflight.py` con `estimate_cost(endpoint, params) -> Estimation(ms, level, reason, breakdown)` (heurística `n_recursos × n_días × factor` para pipeline; rango de días >90 para capacidad/operarios; ms medio vs baseline para bbdd/query).
- Middleware `nexo/middleware/query_timing.py` que envuelve endpoints con preflight y postflight; escribe filas en `nexo.query_log`; emite `log.WARNING` + marca `status='slow'` cuando `actual_ms > warn_ms × 1.5`.
- 4 endpoints refactorizados con preflight: `pipeline/run`, `bbdd/query`, `capacidad` (rango >90d), `operarios` (rango >90d).
- UI `/ajustes/límites` (editor de thresholds + factor_por_modulo + botón "Recalcular desde últimos 30 runs"), `/ajustes/solicitudes` (gestión de approvals para propietario), `/ajustes/rendimiento` (dashboard estimated vs actual), `/mis-solicitudes` (usuario ve y cancela propias).
- Modal amber + modal red en frontend (Alpine.js + Tailwind) disparados por respuesta del preflight antes de ejecutar.
- `asyncio.to_thread` + semáforo global(3) + timeout 15 min para el pipeline OEE.
- Cache de thresholds con invalidation on edit (Postgres LISTEN/NOTIFY o in-memory pub/sub).
- Hooks de learning: botón manual "Recalcular factor" + cron mensual fallback si factor no se actualizó en 60d.
- Tests: unit (preflight heurística), integration (middleware timing escribe query_log), E2E (flujo amber+red manual desde UI).

**Qué NO entra en esta phase** (scope guard):
- Email de notificación de approvals — depende de SMTP (Out of Scope Mark-III).
- Dashboards avanzados / streaming — Mark-IV.
- Rate limiting por endpoint (distinto de preflight) — Mark-IV.
- Sustituir matplotlib — sólo si durante Phase 4 preflight demuestra que es inviable (ver MARK_III_PLAN.md).
- Paginación / virtualización de `/ajustes/rendimiento` — Mark-IV si el volumen crece.
- Filtros por departamento en las páginas de ajustes — Phase 5 (UIROL).

</domain>

<decisions>
## Implementation Decisions

### D-01 — Umbrales pipeline (delegado)
- **warn_ms** = 120_000 (2 min); **block_ms** = 600_000 (10 min).
- Defecto medio-conservador para pipeline OEE en hardware LAN típico.
- Editables desde `/ajustes/límites` sin tocar código (QUERY-02 + QUERY-07).
- Justificación: sin baseline real, mejor empezar conservador y relajar tras ver mediciones en `query_log`.

### D-02 — Umbrales bbdd/query (delegado)
- **warn_ms** = 3_000 (3s); **block_ms** = 30_000 (30s).
- Defecto medio para SQL libre sobre MES.
- Editables desde `/ajustes/límites`.
- Justificación: un SELECT típico retorna <500ms; amber da margen para JOINs razonables, red bloquea queries verdaderamente problemáticas.

### D-03 — Preflight capacidad/operarios (delegado)
- Preflight se dispara **sólo si `rango_días > 90`** (fiel a QUERY-07).
- Umbrales amber/red iguales a bbdd/query como default (warn 3s, block 30s), editables.
- Queries con rango ≤90d no pasan por preflight (no escriben en query_log tampoco — el middleware de timing se salta).

### D-04 — factor_por_modulo inicial pipeline (delegado)
- **Seed inicial = 2000ms** por (recurso × día). Estimate para pipeline = `n_recursos × n_días × 2000ms`.
- UI `/ajustes/límites` muestra el factor actual con botón **"Recalcular desde últimos 30 runs"**.
- El botón calcula `factor_nuevo = median(actual_ms / (n_recursos × n_días))` sobre las últimas 30 filas de `query_log` con `endpoint='pipeline/run'` y `status IN ('ok','slow')`; pide confirmación antes de persistir; graba `updated_by=<user>` y `updated_at=now()` en `query_thresholds`.
- Self-tuning **no automático**: el propietario dispara el recálculo. Cron mensual como fallback (ver D-20).

### D-05 — Modal AMBER (explícito)
- Modal **bloqueante** (Alpine.js overlay oscuro sobre la página) con título "Esta operación puede tardar" + duración estimada destacada + breakdown en texto gris + botones [**Continuar**] + [Cancelar].
- Per-request, sin sticky (ver D-08).
- Cancelar = cierra modal y no ejecuta; Continuar = dispara el request original con `force=true` + el payload.
- Reasegura UX: "La UI seguirá respondiendo mientras la operación corre" (alivia ansiedad del operador, aplicable por D-18).

### D-06 — Modal RED (delegado)
- Modal **bloqueante** rojo con título "Operación requiere aprobación" + duración estimada + motivo ("Excede el límite configurado de 10 min") + botones [**Solicitar aprobación**] + [Cancelar].
- [Solicitar aprobación]:
  - POST `/api/approvals` con `{endpoint, params, estimated_ms}`.
  - Backend crea fila en `nexo.query_approvals` con `status='pending'`, `user_id`, `created_at`, `ttl_days=7`, `params_json`, `estimated_ms`.
  - Responde `{approval_id, status: 'pending'}`.
  - Modal cierra y muestra toast verde "Solicitud enviada. Te avisaremos cuando el propietario apruebe." (link: "Ver mis solicitudes" → `/mis-solicitudes`).
- NO redirect (usuario sigue en la página donde estaba).

### D-07 — Texto del modal (delegado)
- Formato jerárquico:
  - **Línea 1** (grande, bold): "Estimación: ~3 min 20s" (duración humanizada desde ms).
  - **Línea 2** (pequeña, gris): "(4 recursos × 15 días × ~3.3s/run)" — breakdown para debug operativo.
  - **Línea 3** (pequeña, gris): "La UI seguirá respondiendo. Los demás módulos no se bloquearán." (reassurance por D-18).

### D-08 — Persistencia modal (delegado)
- **Per-request siempre**, sin cookie/sessionStorage sticky.
- Cada request que caiga en amber muestra el modal; Continuar dispara ese request y reset.
- Trade-off aceptado: más fricción pero más observabilidad; si en uso real molesta, se itera en Mark-IV.

### D-09 — Contenido de `params_json` (delegado)
- **SQL completo + params** para `bbdd/query`: `{sql: "SELECT ...", database: "dbizaro", user_provided: true}`.
- **Para `pipeline/run`**: `{fecha_desde, fecha_hasta, recursos: [...], n_dias, n_recursos}`.
- **Para `capacidad`/`operarios`**: `{fecha_desde, fecha_hasta, rango_dias, filters_applied}`.
- **Sanitización mínima**: campos sensibles (passwords de conexión en endpoints de ajustes, tokens) van con valor `"[REDACTED]"` — misma whitelist que ya existe en audit_middleware (Phase 2).
- Protección principal vs PII: retención corta (D-10). Decisión consciente de priorizar debug-ability sobre zero-retention.

### D-10 — Retención `nexo.query_log` (delegado)
- **90 días** por defecto con purga automática.
- Job programado `nexo/services/query_log_cleanup.py`: al arrancar app + cada Monday 03:00 ejecuta `DELETE FROM nexo.query_log WHERE ts < now() - interval '<retention_days> days'`.
- Loggea filas borradas en `audit_log` con `path='__cleanup__'`, `method='DELETE'`, `details_json={rows_deleted: N, cutoff_ts}`.
- Override vía env var `NEXO_QUERY_LOG_RETENTION_DAYS`:
  - Default: `90`.
  - `0` = forever (sin purga).
  - Otro entero N = retención de N días.
- Documentado en `.env.example` + `docs/BRANDING.md` o sección nueva en README.

### D-11 — Página `/ajustes/rendimiento` (explícito)
- **Nueva sub-página** bajo `/ajustes/*` con acceso restringido a `propietario` (mismo patrón que `/ajustes/auditoria`).
- Filtros: user (dropdown de usuarios Nexo), endpoint (dropdown con los 4 con preflight), estado (green/amber/red/slow/timeout/approved_run), rango temporal (7d/30d/90d/custom).
- Tabla superior con una fila por endpoint filtrado: `endpoint | n_runs | avg_estimated_ms | avg_actual_ms | divergencia_% | p95_actual_ms | n_slow`.
- Gráfica inferior: línea estimated vs actual por timestamp (Chart.js CDN) — muestra timeseries del endpoint y rango seleccionado.
- Template: `templates/ajustes_rendimiento.html`. Endpoint API: `GET /api/rendimiento/summary` + `GET /api/rendimiento/timeseries`.
- Integración con D-04: la página `/ajustes/rendimiento` comparte dataset con el botón "Recalcular factor" de `/ajustes/límites` — ambos consultan el mismo subset de `query_log`.

### D-12 — Chart library (delegado)
- **Chart.js** vía CDN (consistente con política de Nexo de usar CDNs para JS libs de UI — ver tailwind.config y alpine.js).
- Tipo: line chart con 2 series (estimated, actual), eje X = timestamp, eje Y = ms.
- Responsive con `maintain-aspect-ratio: false` + container con altura fija.
- Fallback si CDN no carga: tabla sin gráfica (no bloqueante).

### D-13 — Notificación de approvals al propietario (delegado)
- **Badge en sidebar**: item `/ajustes/solicitudes` muestra `Solicitudes (N)` cuando hay pendientes.
- Polling HTMX `hx-get="/api/approvals/count" hx-trigger="every 30s" hx-swap="innerHTML"` en el nav item del propietario.
- Endpoint `/api/approvals/count` devuelve `<span>(3)</span>` o string vacío si 0.
- Página `/ajustes/solicitudes` hace auto-refresh cada 30s (`hx-trigger="every 30s"` en la tabla de pending).
- **No banner global**, **no email** (email diferido a Mark-IV cuando SMTP esté operativo).

### D-14 — TTL de approvals pendientes (delegado)
- **TTL = 7 días** por defecto (env var `NEXO_APPROVAL_TTL_DAYS=7`).
- Tras 7d sin acción (approve/reject/cancel), la solicitud pasa a `status='expired'`.
- Job semanal `nexo/services/approvals_cleanup.py` corre Monday 03:05 (después de query_log cleanup) y marca expired.
- Expired visibles en `/ajustes/solicitudes` bajo sección "Histórico" durante 30 días más (por auditoría); tras 37d totales, se purgan definitivamente.

### D-15 — Semántica `force=true` + approval_id (delegado)
- Endpoints que aceptan `force=true` (pipeline/run, bbdd/query, capacidad, operarios) requieren **también** `approval_id=<uuid>` en el request body (JSON) o query string.
- Backend verifica:
  1. `approval_id` existe en `nexo.query_approvals`.
  2. `status = 'approved'`.
  3. `user_id` coincide con el user autenticado.
  4. `consumed_at` es NULL.
  5. `params_json` del approval coincide con el request actual (equality check — si el usuario cambia parámetros debe solicitar aprobación nueva).
- Tras verificación: marca `consumed_at = now()` y `consumed_run_id = <query_log.id>` en `query_approvals`; ejecuta el endpoint; escribe fila en `query_log` con `approval_id` poblado.
- Approvals son **single-use**: cada aprobación = 1 ejecución. Nuevo run del mismo tipo requiere nuevo approval.
- Si verificación falla: `HTTPException(403, "Invalid or expired approval")`.

### D-16 — Cancelación por usuario (explícito)
- Página **`/mis-solicitudes`** (nueva, accesible a todos los usuarios autenticados).
- Muestra las solicitudes propias del usuario (`WHERE user_id = :current_user`) con estado pending/approved/rejected/expired/cancelled/consumed.
- Botón [**Cancelar**] sobre cada fila `pending` — UX amable.
- POST `/api/approvals/<id>/cancel`: verifica ownership, cambia `status='cancelled'`, graba `cancelled_at`.
- Cancelled queda visible en histórico 30d (para auditoría) y luego purga.
- Reduce ruido para el propietario — el operador retracta solicitudes que ya no necesita.

### D-17 — Postflight divergence alert (delegado)
- Cuando `actual_ms > warn_ms × 1.5` tras un run, el middleware `query_timing`:
  1. Emite `logging.warning(f"slow_query endpoint={endpoint} user={user} estimated={est}ms actual={act}ms ratio={act/est:.2f}")`.
  2. Graba la fila en `query_log` con `status='slow'` (en lugar de `'ok'`).
- Propietario ve divergencias en `/ajustes/rendimiento` filtrando por `status='slow'`.
- **Sin push notification** (no badge, no banner) — pull model via la página.
- Rationale: si pusheamos, spam en planta con queries lentas regulares; propietario revisa cuando quiere afinar.

### D-18 — Matplotlib concurrency (delegado)
- **Semáforo global** en `nexo/services/pipeline_lock.py`: `_pipeline_semaphore = asyncio.Semaphore(NEXO_PIPELINE_MAX_CONCURRENT)` (default 3).
- `pipeline.run` hace `async with _pipeline_semaphore:` + `await asyncio.to_thread(run_pipeline_sync, ...)`.
- Si los 3 slots están ocupados, el 4º+ usuario ve modal "Pipeline en cola (3 ejecutándose). Tu ejecución empezará cuando termine una."
- **Timeout duro**: `await asyncio.wait_for(asyncio.to_thread(...), timeout=NEXO_PIPELINE_TIMEOUT_SEC)` (default 900s = 15 min).
- Si timeout dispara: cancela el thread (matplotlib cierra figs, PDF parcial se descarta), graba `status='timeout'` + `actual_ms=timeout_ms` en query_log, devuelve HTTPException(504).
- Env vars configurables:
  - `NEXO_PIPELINE_MAX_CONCURRENT=3` (rango recomendado: 1-5, dependiendo de RAM).
  - `NEXO_PIPELINE_TIMEOUT_SEC=900` (rango recomendado: 300-1800).

### D-19 — Hot-reload de thresholds (explícito)
- **Cache + invalidate on edit** (decisión explícita, más robusta que TTL).
- `nexo/services/thresholds_cache.py` mantiene dict in-memory `{endpoint: (warn_ms, block_ms, factor, updated_at)}`.
- Al arrancar el worker: carga todas las filas de `nexo.query_thresholds` en cache.
- Al guardar desde `/ajustes/límites`: POST `/api/thresholds/<endpoint>` hace UPDATE en BD + emite `NOTIFY nexo_thresholds_changed, '<endpoint>'`.
- Listener background (async task en cada worker uvicorn) con `LISTEN nexo_thresholds_changed`: al recibir, recarga la fila específica en cache.
- **Todos los workers quedan coherentes en <1 segundo** tras la edición.
- Fallback si LISTEN/NOTIFY falla (error Postgres, caída): cache tiene `updated_at` propia; al acceder, si `now() - updated_at > 5min`, re-lee BD forzosamente. Double safety net.

### D-20 — Preflight learning (delegado)
- **Botón manual** en `/ajustes/límites` (ver D-04): propietario dispara recálculo, ve preview, confirma.
- **Cron mensual fallback**: job `nexo/services/factor_auto_refresh.py` corre 1er Monday de cada mes 03:10; para cada endpoint, comprueba `query_thresholds.factor_updated_at`:
  - Si `> 60 días`: recalcula factor automáticamente y actualiza; graba entry en audit_log con `path='__auto_refresh__'`, `details_json={endpoint, old_factor, new_factor, sample_size, reason='stale_60d'}`.
  - Si `≤ 60 días`: no-op.
- Doble safety net: operador controla cuándo; el sistema evita factores desactualizados si se olvida.
- Env var `NEXO_AUTO_REFRESH_STALE_DAYS=60` para tuning.

### Claude's Discretion
- Schema exacto de `nexo.query_approvals` (columnas, tipos, índices) — planner decide basándose en requisitos de D-14, D-15, D-16.
- Schema exacto de `nexo.query_log` — columnas ya en QUERY-01, pero tipos e índices (sugeridos: BTREE en `ts`, `endpoint`, `user_id`, `status`) quedan al planner.
- Orden de rollout dentro del plan (pipeline primero? bbdd primero? middleware global vs per-router?) — planner decide tras research.
- Cómo lidiar con CI sin Postgres: tests de middleware pueden usar mocks o docker compose up db en CI — mantener patrón de Phase 3 (D-08 del CONTEXT 03).
- Modelo Pydantic `Estimation` — campos exactos (ms, level, reason, breakdown, factor_used) a concretar en research/planner.
- Integración con auth: reuse `require_permission` (Phase 2) para `/ajustes/rendimiento:read`, `/ajustes/límites:write`, `/ajustes/solicitudes:write` (propietario), `/mis-solicitudes:read` (todos), `/api/approvals/*` según corresponda.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Contratos de producto
- `docs/MARK_III_PLAN.md` §Sprint 3 — plan detallado de preflight+postflight con riesgos y estimaciones.
- `docs/AUTH_MODEL.md` — contrato de auth para `require_permission(...)` en los nuevos endpoints.
- `docs/GLOSSARY.md` — términos preflight/postflight/green/amber/red usados en la UI y backend.
- `CLAUDE.md` — reglas del repo (convención naming Nexo, NO renombrar OEE/ ni repo, asyncio.to_thread NO sustituye matplotlib en Mark-III).

### Requisitos y roadmap
- `.planning/REQUIREMENTS.md` §QUERY-01..QUERY-08 — 8 requisitos trazables que esta phase debe cumplir.
- `.planning/ROADMAP.md` §Phase 4 — goal + 5 success criteria + dependencias.
- `.planning/PROJECT.md` §Key Decisions — constraints de producto (Postgres nexo.*, SQL Server ecs_mobility intacto).

### Estado del codebase (proyección)
- `.planning/codebase/ARCHITECTURE.md` — layers actuales, middleware chain.
- `.planning/codebase/STRUCTURE.md` — directorios; `nexo/middleware/` no existe aún (se crea en Phase 4).
- `.planning/codebase/CONVENTIONS.md` — estilo del proyecto (naming, async patterns).

### Precondiciones de Phase 3 (aún pipeline activo)
- `.planning/phases/03-capa-de-datos/03-CONTEXT.md` — repos + DTOs + engines disponibles.
- `.planning/phases/03-capa-de-datos/03-03-SUMMARY.md` — repos APP/NEXO operativos; pipeline.py ya usa canonical ORM path.
- `nexo/data/repositories/` — `MesRepository`, `RecursoRepo`, etc. ya exponen métodos consumibles.

### Precondiciones de Phase 2 (auth + audit)
- `.planning/phases/02-identidad-auth-rbac-audit/02-CONTEXT.md` — middleware chain auth→audit; require_permission semantics.
- `nexo/services/auth.py` — `require_permission("modulo:accion")` reutilizable en nuevos endpoints.
- `nexo.audit_log` — middleware audit ya escribe todas las requests; query_log es complementario (métricas de coste, no auditoría).

### Código que se toca
- `api/routers/pipeline.py` — añadir endpoint `POST /preflight` + hacer `POST /run` aceptar `force` + `approval_id`.
- `api/routers/bbdd.py` — inyectar preflight antes del whitelist anti-DDL (preflight primero, rechazo si red sin approval).
- `api/routers/capacidad.py`, `operarios.py` — preflight condicional a rango >90d.
- `api/services/pipeline.py` — envolver `run_pipeline` generator en `asyncio.to_thread` + semáforo.
- `api/main.py` — añadir query_timing middleware después de audit middleware; arrancar listener LISTEN/NOTIFY para thresholds cache; registrar cleanup jobs.
- `templates/ajustes.html` — hub existente añade enlaces a `/ajustes/límites`, `/ajustes/solicitudes`, `/ajustes/rendimiento`.
- `templates/` — crear `ajustes_limites.html`, `ajustes_solicitudes.html`, `ajustes_rendimiento.html`, `mis_solicitudes.html`.
- `static/js/app.js` — modal amber + modal red (Alpine.js component); utility `humanize_ms(ms)`.
- `docs/GLOSSARY.md` — añadir términos: `estimated_ms`, `actual_ms`, `approval_id`, `slow_query`.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `nexo/services/auth.py:require_permission(...)` — dependency ya operativa, se reusa tal cual para los 4 nuevos endpoints de `/ajustes/*`.
- `nexo/data/repositories/nexo.py:UserRepo`, `AuditRepo` — expuestos en Phase 3; AuditRepo.append reusado por cleanup jobs y approval events.
- `api/middleware/audit.py` (del Phase 2) — patrón de middleware que inyecta `user_id` y graba request: `query_timing.py` se monta después siguiendo el mismo shape.
- `nexo/db/engine.py` + `engine_nexo` — conexión Postgres disponible; se reusa para LISTEN/NOTIFY desde un worker thread separado.
- Chart.js — no está hoy; se añade CDN en `base.html` o en las templates concretas de `/ajustes/rendimiento`.
- Alpine.js — ya operativo en Nexo; modal amber+red usa `x-data`, `x-show`, `x-transition` (sin librería modal externa).
- HTMX — ya operativo; badge sidebar + auto-refresh de /ajustes/solicitudes/rendimiento usan `hx-get` + `hx-trigger="every 30s"`.

### Established Patterns
- SQLAlchemy `text()` + bindparams — ya usado para queries dinámicas en Phase 3; se replica para queries de query_log y approvals (aunque ORM puro es preferible por D-01 del CONTEXT 03: no generar `.sql` redundante para queries ORM simples).
- Jinja2 template inheritance con `base.html` — todas las nuevas páginas extienden `base.html`.
- `Depends(get_db_nexo)` — ya expuesto en `api/deps.py` tras Phase 3; todos los nuevos endpoints lo usan.
- Middleware chain actual: `request → auth_middleware → audit_middleware → router`. Phase 4 inserta `query_timing_middleware` entre audit y router (para que timing vea user_id).

### Integration Points
- `api/main.py:lifespan()` — ya tiene `schema_guard.verify()`. Se añade:
  1. Carga inicial de thresholds cache.
  2. Arranque del listener LISTEN/NOTIFY en background task.
  3. Arranque del scheduler (simple `asyncio.create_task(cleanup_loop())` o si hay muchos jobs, `apscheduler` via pip — planner decide).
- `api/deps.py` — posiblemente se añade `Depends(get_thresholds)` para inyectar cache a los endpoints que hacen preflight.
- `docker-compose.yml` — Postgres 16 ya expuesto; LISTEN/NOTIFY funciona sin config extra.

### Riesgos específicos
- **Middleware order**: `query_timing` DEBE ir después de `audit` (para tener `user_id`) y antes del router. Phase 2 ya definió el orden; verificar que no hay regresión.
- **Thread safety del cache**: los workers uvicorn son procesos independientes (no threads). Cada proceso tiene su propio cache; LISTEN/NOTIFY es el mecanismo de coherencia. Si uvicorn workers=1 (actual default), no hay problema; si se escala a multi-worker, LISTEN listener debe montarse en CADA worker.
- **Matplotlib memory**: un pipeline genera ~10-20 PDFs por run. Semáforo(3) × ~200MB/pipeline = ~600MB pico. Ubuntu 16GB aguanta fácilmente pero hay que documentar.
- **LISTEN/NOTIFY vs polling fallback**: si LISTEN falla silenciosamente, cache queda desactualizada sin error visible. Por eso D-19 añade safety net (re-lee si updated_at >5min).
- **Query_log write overhead**: cada request + postflight inserta fila. Con 1000 req/día y 90d retención = 90k filas, despreciable. Con índice en `ts` descendente, purge es rápida.
- **PDF regression check de 03-02**: aún pending (deadline 2026-04-26). Phase 4 toca pipeline.py vía semáforo. **Si el PDF regression check descubre divergencia entre Mark-II y Mark-III**, debemos investigar antes de cerrar Phase 4.
- **Factor drift**: si el seed inicial (2000ms) está muy lejos del real, los primeros días todos los pipelines caerán en amber/red innecesariamente. Mitigación: el operador recalcula manualmente tras las primeras 30 ejecuciones (D-04) o se diferencia en la UI "factor no calibrado — recalibra cuando tengas 30+ runs".

</code_context>

<specifics>
## Requisitos trazables (REQUIREMENTS.md)

- **QUERY-01** → Plan A (foundation): tabla `nexo.query_log` + schema_guard + model + migración.
- **QUERY-02** → Plan A (foundation): tabla `nexo.query_thresholds` + seed inicial + cache + CRUD vía `/ajustes/límites`.
- **QUERY-03** → Plan B (preflight): `nexo/services/preflight.py` con `estimate_cost` + heurística + learning (D-04/D-20).
- **QUERY-04** → Plan B (preflight): niveles green/amber/red + modal UX (D-05/D-06/D-07).
- **QUERY-05** → Plan C (postflight): middleware `query_timing.py` + alert (D-17).
- **QUERY-06** → Plan D (approval): tabla `nexo.query_approvals` + `/ajustes/solicitudes` + `/mis-solicitudes` + `force=true` + approval_id (D-13/D-14/D-15/D-16).
- **QUERY-07** → Plans B+C aplicados a los 4 endpoints (pipeline, bbdd, capacidad, operarios) + UI `/ajustes/límites`.
- **QUERY-08** → Plan B (preflight) y/o dedicated: `asyncio.to_thread` + semáforo + timeout en `pipeline/run` (D-18).

## Success criteria (ROADMAP.md Phase 4)

1. Disparar un pipeline con muchos recursos/días muestra toast amber "esto tardará ~X min, ¿continuar?" → cubierto por D-04+D-05+D-07.
2. Una query en `/bbdd` con coste estimado > `block_ms` abre flujo de aprobación → cubierto por D-02+D-06+D-13+D-15.
3. `nexo.query_log` contiene `estimated_ms` y `actual_ms` para cada ejecución; alertas WARNING si divergen > 50% → cubierto por D-09+D-17.
4. `/ajustes/limites` permite editar umbrales por endpoint sin tocar código → cubierto por D-01+D-02+D-19.
5. Pipeline no congela la UI → cubierto por D-18 (asyncio.to_thread + semáforo).

## Decisiones explícitas del usuario (no delegadas)

Tres decisiones donde el usuario rechazó "delego" y eligió opción concreta:

1. **D-05** Modal AMBER bloqueante con [Continuar]+[Cancelar]. Rechazó toast no-bloqueante y sticky.
2. **D-11** Página dedicada `/ajustes/rendimiento` con filtros + gráfica. Rechazó "sin página, consulta vía /bbdd".
3. **D-16** Usuario puede cancelar solicitud propia (pantalla `/mis-solicitudes`). Rechazó "sólo propietario decide".
4. **D-19** Cache + invalidate on edit (LISTEN/NOTIFY). Rechazó cache TTL 60s.

Estas 4 decisiones son LOCKED. Todo lo demás es delegación con default documentado que el planner puede ajustar si research revela problemas.

</specifics>

<deferred>
## Deferred Ideas

- **Email de notificación de approvals** — depende de SMTP (Out of Scope Mark-III). Mark-IV cuando SMTP esté operativo.
- **Banner global top de "Solicitudes pendientes"** — D-13 rechazó en favor de badge sidebar. Reconsiderar si badge resulta poco visible.
- **Dashboard streaming (websockets) de rendimiento** — Mark-IV; polling HTMX suficiente para LAN.
- **Rate limiting fino por endpoint** (además de preflight) — Mark-IV; preflight + login rate-limit cubren el 80%.
- **Sustituir matplotlib** — sólo si Phase 4 demuestra inviable. Decision gate: si >20% de pipelines caen en `status='timeout'` tras 30 días en prod, abrir ticket Mark-IV.
- **Paginación/virtualización de `/ajustes/rendimiento`** — Mark-IV si el volumen crece (>10k filas/página).
- **Filtros por departamento en `/ajustes/solicitudes`** — Phase 5 (UIROL).
- **Alertas por Slack/Telegram** — Mark-IV con SMTP o webhooks.
- **Forecast predictivo de coste** (ML sobre query_log) — Mark-V; por ahora heurística lineal + learning manual es suficiente.
- **Aprobación delegada** (propietario delega en directivo) — Mark-IV; en Mark-III sólo propietario aprueba.
- **Exportar CSV de query_log desde `/ajustes/rendimiento`** — nice-to-have; planner incluye si es trivial, difiere si requiere >1h.

</deferred>

---

*Phase: 04-consultas-pesadas*
*Context gathered: 2026-04-19 via /gsd-discuss-phase (8 gray areas; 4 LOCKED por preferencia explícita del usuario; 16 delegadas al planner con defaults documentados).*
