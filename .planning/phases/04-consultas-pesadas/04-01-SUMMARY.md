---
phase: 04-consultas-pesadas
plan: 01
type: summary
wave: 1
status: complete
requirements: [QUERY-01, QUERY-02]
executed: 2026-04-20
commits:
  - 50a404b feat(04-01): add 3 ORM models for query_log/thresholds/approvals
  - 1ae376b feat(04-01): add query DTOs + Estimation + extend PipelineRequest
  - bdfbe6a feat(04-01): add QueryLogRepo + ThresholdRepo + ApprovalRepo
  - ffb069d feat(04-01): extend schema_guard to 11 tables + seed query_thresholds
  - f67e7fd feat(04-01): add thresholds_cache skeleton (in-memory dict + safety-net)
  - 1e34771 test(04-01): add Wave 0 test contracts + 2 green suites
  - f91706d fix(dev): point uvicorn to Postgres on localhost:5433 and auto-start db container
  - 0a9588d refactor(tests): elevate db_nexo/db_app/engine_mes_mock fixtures to root conftest
  - caa7aa9 fix(bootstrap): init_nexo_schema siempre usa owner creds + añadir nexo-init-dev
self_check: PASSED
---

# Plan 04-01 — Foundation (QUERY-01, QUERY-02)

Wave 1 cierra con Postgres real verificado contra HEAD. 11 tablas en
`nexo.*`, 4 seeds iniciales de `query_thresholds`, cache skeleton,
y Wave 0 completo (12 tests green + 13 stubs SKIPPED, cero FAILED).

## Objective delivered

Foundation bloqueante para toda Phase 4. Nada de los siguientes plans
compila sin esto:

- **3 tablas Postgres** (`query_log`, `query_thresholds`,
  `query_approvals`) con ORM, DTOs, repositorios, y cobertura en
  schema_guard.
- **4 seeds iniciales** de `query_thresholds` materializando
  decisiones D-01 (pipeline/run 2min/10min), D-02 (bbdd/query 3s/30s),
  D-03 (capacidad/operarios 3s/30s) y D-04 (factor_ms para
  pipeline/run).
- **thresholds_cache skeleton** con D-19 safety-net de 5 min;
  listener LISTEN/NOTIFY aterriza en Plan 04-04.
- **Wave 0 test contracts** — 10 archivos nuevos, 2 suites green, 5
  stubs apuntando a plans 04-02/03/04, dependencias (pytest-asyncio)
  añadidas.

## Key files created / modified

### Data layer

- `nexo/data/models_nexo.py` — 3 ORM models en orden de dependencia
  (NexoQueryApproval → NexoQueryThreshold → NexoQueryLog), con
  índices explícitos:
  - `query_log`: 4 índices (`ts`, `endpoint+ts`, `user+ts`,
    **partial `WHERE status='slow'`**).
  - `query_approvals`: 3 índices (`status`, `user+status`,
    `created_at`).
  - `query_thresholds`: PK en `endpoint` (4 filas esperadas).
- `nexo/data/dto/query.py` + `nexo/data/dto/__init__.py` — DTOs
  frozen + `Estimation` con `Literal["green","amber","red"]`.
- `nexo/data/repositories/nexo.py` — 3 repos:
  - `QueryLogRepo.append()` commit-free (parity con AuditRepo).
  - `ApprovalRepo.consume()` usa `text()` UPDATE con
    `WHERE consumed_at IS NULL AND params_json = :pj RETURNING *` para
    CAS atómico (mitigación D-15 / T-04-01-03).
  - `ApprovalRepo.list_recent_non_pending()` preparado para el
    histórico 30d de Plan 04-03.
- `nexo/data/schema_guard.py` — extendido de 8 → 11 tablas
  críticas (append-only; no rompe tests Phase 2/3).

### Services layer

- `nexo/services/thresholds_cache.py` — skeleton in-memory dict con
  `full_reload` / `reload_one(endpoint)` / `get(endpoint)` (con
  fallback safety-net 5 min por D-19) / `notify_changed(endpoint)`
  (AUTOCOMMIT raw conn para `NOTIFY`). `listen_loop` /
  `start_listener` **no** implementados (deferred a Plan 04-04).

### API layer

- `api/models.py` — `PipelineRequest` backwards-compatible:
  `force: bool = False`, `approval_id: Optional[int] = None`.

### Bootstrap / scripts

- `scripts/init_nexo_schema.py` — seeds de `query_thresholds`
  idempotentes via `text()` + bindparams + `ON CONFLICT (endpoint)
  DO NOTHING`. Fix: engine dedicado al bootstrap con `pg_user` directo
  (bypass `effective_pg_user`) — evita `permission denied` cuando
  `NEXO_PG_APP_USER=nexo_app` está en `.env`.

### Tests

- `tests/conftest.py` — fixtures elevados a raíz:
  - `db_nexo`, `db_app`, `engine_mes_mock` (movidos desde
    `tests/data/conftest.py`).
  - `_postgres_reachable`, `_mssql_reachable` idem.
  - `thresholds_cache_empty` (Phase 4 / Plan 04-01 exclusivo).
- `tests/data/conftest.py` — thin shim que re-exporta para no romper
  `from tests.data.conftest import _postgres_reachable`.
- `tests/data/test_schema_query_log.py` — 8 tests green.
- `tests/data/test_schema_guard_extended.py` — 4 tests green.
- `tests/services/test_thresholds_cache.py` — 5 green + 1 SKIPPED.
- `tests/services/test_approvals.py` — 2 green (create defaults +
  TTL) + 3 SKIPPED (CAS race / cancel / expire → Plan 04-03).
- `tests/services/test_preflight.py`, `test_pipeline_lock.py`,
  `tests/middleware/test_query_timing.py`,
  `tests/routers/test_preflight_endpoints.py`,
  `tests/routers/test_approvals_api.py` — módulo SKIP con reason
  apuntando a plans 04-02/03/04.
- `requirements-dev.txt` — añadido `pytest-asyncio==0.24.0`.

### Dev ergonomics (fixes colaterales)

- `Makefile`:
  - `dev`: auto-start del container `db` + wait `pg_isready` +
    export `NEXO_PG_HOST=localhost NEXO_PG_PORT=${NEXO_PG_HOST_PORT:-5433}`.
    Corrige el gotcha pre-existente donde `make dev` fallaba con
    `could not translate host name "db"` por comentario obsoleto.
  - `nexo-init-dev` nuevo: corre el init script desde el host contra
    `localhost:5433` (para workflow `make dev`, sin container web).

## Manual smoke verification (2026-04-20)

Ejecutado end-to-end contra Postgres real arrancado por `make dev`:

```
$ docker compose exec db psql -U oee -d oee_planta -c "\dt nexo.*"
 nexo | audit_log        | table | oee
 nexo | departments      | table | oee
 nexo | login_attempts   | table | oee
 nexo | permissions      | table | oee
 nexo | query_approvals  | table | oee  ← Plan 04-01
 nexo | query_log        | table | oee  ← Plan 04-01
 nexo | query_thresholds | table | oee  ← Plan 04-01
 nexo | roles            | table | oee
 nexo | sessions         | table | oee
 nexo | user_departments | table | oee
 nexo | users            | table | oee
(11 rows)

$ ... SELECT endpoint, warn_ms, block_ms, factor_ms
      FROM nexo.query_thresholds ORDER BY endpoint;
 bbdd/query   |    3000 |    30000 |    1000  ← D-02
 capacidad    |    3000 |    30000 |      50  ← D-03
 operarios    |    3000 |    30000 |      50  ← D-03
 pipeline/run |  120000 |   600000 |    2000  ← D-01 + D-04

$ pytest tests/
123 passed, 25 skipped, 0 errors in 4.54s
```

Segunda pasada de `init_nexo_schema.py` (idempotencia): 0 filas
duplicadas, exit 0.

## Decisions honored (from 04-CONTEXT.md)

| ID | Decision | How honored |
|----|----------|-------------|
| D-01 | pipeline/run warn=2min, block=10min | Seed exacto en `query_thresholds` |
| D-02 | bbdd/query warn=3s, block=30s | Seed exacto |
| D-03 | capacidad/operarios warn=3s, block=30s, factor=50ms/día | Seeds |
| D-04 | pipeline/run factor=2000ms por recurso·día | Seed |
| D-14 | Approval TTL default 7d | `ApprovalRepo.create` + test verifica |
| D-15 | Approval status machine (pending/approved/rejected/cancelled/expired/consumed) | ORM columns + CAS single-use UPDATE |
| D-16 | `nexo.query_log` indices (incl. partial slow) | 4 índices en ORM |
| D-19 | thresholds_cache safety-net cuando listener muere | `get()` fallback a `full_reload` si `loaded_at_global` > 5 min |

## Deviations from plan

Ninguna funcional. Los **3 fixes colaterales** surgieron del smoke:

1. **`make dev`** fallaba con DNS error (`could not translate host
   name "db"`) porque el comentario "sin Docker, SQLite" llevaba
   obsoleto desde Phase 3 y el target no exportaba `NEXO_PG_HOST`
   overrides. Fix en commit `f91706d` — auto-arranca `db` y apunta
   uvicorn a `localhost:5433`.
2. **`db_nexo` fixture no visible** para `tests/services/` y
   `tests/routers/` porque `tests/data/conftest.py` solo se
   auto-descubre para hermanos de `tests/data/`. Fix en commit
   `0a9588d` — elevar fixtures al root conftest con shim de
   re-exportación.
3. **`init_nexo_schema.py`** fallaba con `permission denied for
   database oee_planta` cuando `.env` tenía
   `NEXO_PG_APP_USER=nexo_app` configurado (rol con GRANTs
   limitados no puede CREATE SCHEMA). Fix en commit `caa7aa9` —
   engine dedicado al bootstrap usa `pg_user`/`pg_password` directo.

Los tres son bugs pre-existentes o consecuencias del cambio Plan
02-04 (split owner / app role), no regresiones introducidas por 04-01.

## Test coverage snapshot

| Area | Green | Skipped | Notes |
|------|-------|---------|-------|
| Plan 04-01 específicos | 19 | 9 | stubs apuntan a 04-02/03/04 |
| Regression Phase 2/3 | 104 | 16 | sin cambios |
| **Full suite** | **123** | **25** | **0 FAILED** |

## Next

Wave 2 (Plan 04-02) preflight + postflight + asyncio.to_thread +
modal frontend. Depende de `NexoQueryThreshold`, `NexoQueryLog`,
`ApprovalRepo`, `Estimation`, `PipelineRequest.force`, y
`thresholds_cache` ya aterrizados aquí.
