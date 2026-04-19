---
phase: 03-capa-de-datos
plan: 01
slug: foundation-engines-loader-schema-guard
subsystem: data-layer
tags: [data-layer, engines, schema-guard, test-fixtures, foundation]
status: complete
wave: 1
completed: 2026-04-19
duration: "7 min"
requires: [02-04]
provides:
  - "nexo.data.engines (engine_nexo, engine_app, engine_mes)"
  - "nexo.data.sql.loader.load_sql"
  - "nexo.data.schema_guard.verify"
  - "nexo.data.dto.base.ROW_CONFIG"
  - "api.deps.get_db_app / get_db_nexo / get_engine_mes + Annotated aliases"
  - "tests/data/ fixtures (db_nexo, db_app, engine_mes_mock)"
  - "Makefile target test-data"
  - "NEXO_AUTO_MIGRATE feature flag"
affects:
  - "nexo/db/engine.py (ahora shim)"
  - "api/main.py (lifespan llama schema_guard antes de init_db)"
  - "api/deps.py (añade 3 deps + 3 Annotated aliases)"
  - "Makefile (.PHONY + test-data target)"
  - ".env.example (+ NEXO_AUTO_MIGRATE=false)"
tech-stack:
  added:
    - "importlib.resources (lru_cache-backed SQL loader)"
    - "pydantic v2 ConfigDict(frozen=True, from_attributes=True) — ROW_CONFIG"
  patterns:
    - "shim module re-export (nexo/db/engine.py → nexo/data/engines.py)"
    - "startup validator en FastAPI lifespan (aborta si schema drift)"
    - "kwarg injection en vez de module-level monkeypatch (critical_tables)"
key-files:
  created:
    - "nexo/data/__init__.py"
    - "nexo/data/engines.py"
    - "nexo/data/sql/__init__.py"
    - "nexo/data/sql/loader.py"
    - "nexo/data/dto/__init__.py"
    - "nexo/data/dto/base.py"
    - "nexo/data/schema_guard.py"
    - "tests/data/__init__.py"
    - "tests/data/conftest.py"
    - "tests/data/test_engines.py"
    - "tests/data/test_sql_loader.py"
    - "tests/data/test_schema_guard.py"
    - "tests/data/test_dto_immutable.py"
    - "tests/data/test_fixtures.py"
  modified:
    - "nexo/db/engine.py"
    - "api/main.py"
    - "api/deps.py"
    - "Makefile"
    - ".env.example"
key-decisions:
  - "schema_guard.verify acepta critical_tables: tuple[str, ...] = CRITICAL_TABLES como kwarg para testabilidad sin monkeypatch del módulo (más robusto si CRITICAL_TABLES migra a frozenset)."
  - "engine_app re-exportado desde api/database.py en vez de duplicar _mssql_creator (Landmine #3)."
  - "engine_nexo usa settings.effective_pg_user / effective_pg_password (Landmine #12 — preserva gate IDENT-06 de 02-04)."
  - "nexo/db/engine.py queda como shim de ≤10 líneas efectivas; se elimina al final de 03-03."
  - "schema_guard.verify corre FUERA del try/except de init_db en lifespan — si falla, la app NO arranca (comportamiento deseado para detectar drift)."
  - "dto/base.py expone ROW_CONFIG compartible — DTOs concretos aterrizan en 03-02 y 03-03."
requirements-completed: [DATA-01, DATA-05, DATA-06, DATA-08, DATA-10, DATA-11]
metrics:
  tasks: 3
  commits: 4
  files_created: 14
  files_modified: 5
  tests_added: 16
  tests_pass: 16
  auth_regression: "11 passed, 1 skipped"
---

# Phase 3 Plan 1: Foundation (engines + loader + schema_guard + fixtures) Summary

Base de la capa de datos aterrizada sin tocar routers: 3 engines SQLAlchemy (`engine_nexo`, `engine_app`, `engine_mes`) coexisten en runtime con pool DATA-11 y DSN `DATABASE=dbizaro` que habilita DATA-09 en 03-02; loader `lru_cache + importlib.resources` listo para los `.sql` versionados; `schema_guard.verify` cableado en `lifespan` aborta arranque si faltan tablas críticas y auto-migra con `NEXO_AUTO_MIGRATE=true`; shim en `nexo/db/engine.py` mantiene a los consumidores Phase 2 funcionales sin cambios; `tests/data/` con 16 tests verdes + fixtures; gate IDENT-06 intacto (11 passed / 1 skipped).

## Ejecución

- **Duración**: 7 min (455 s)
- **Inicio**: 2026-04-19T14:44:51Z
- **Fin**: 2026-04-19T14:52:26Z
- **Tasks ejecutadas**: 3 (Task 1 Foundation + Task 2 Wiring + Task 3 Tests)
- **Commits atómicos**: 4
- **Archivos nuevos**: 14
- **Archivos modificados**: 5

## Commits

| # | Hash      | Type     | Scope   | Message |
|---|-----------|----------|---------|---------|
| 1 | `4325943` | refactor | 03-01   | move engine_nexo to nexo/data/engines.py + add engine_mes + shim |
| 2 | `bb6cd62` | feat     | 03-01   | add sql loader + schema_guard (critical_tables kwarg) + dto skeleton |
| 3 | `9c81227` | feat     | 03-01   | wire schema_guard in lifespan + api/deps.py + Makefile test-data |
| 4 | `5c70f0b` | test     | 03-01   | add tests/data/ with fixtures + 5 wave-0 validations |

## Hard gate — resultados

| # | Check | Resultado |
|---|-------|-----------|
| 1 | `from nexo.data.engines import engine_nexo, engine_app, engine_mes, SessionLocalNexo, SessionLocalApp; shim is engine_nexo` | PASS — `engines+shim OK` |
| 2 | `engine_mes.pool._recycle==3600` + `pool_pre_ping is True` + `url.database == settings.mes_db` | PASS — `DATA-11 + DATA-09 prep OK` (DB=`dbizaro`) |
| 3 | `load_sql.cache_info` + `len(CRITICAL_TABLES)==8` + `critical_tables` en signature de `verify` | PASS — `loader + schema_guard OK` |
| 4 | `get_db_app`, `get_db_nexo`, `get_engine_mes`, `DbApp`, `DbNexo`, `EngineMes` importables | PASS — `deps OK` |
| 5 | `inspect.getsource(lifespan)` contiene `schema_guard.verify(engine_nexo)` | PASS — `lifespan wired` |
| 6 | `TestClient(app).get('/login')` → 200 (app arranca end-to-end) | PASS — `app arranca, schema_guard verde` |
| 7a | `pytest tests/data/ -q` | PASS — **16 passed** (16 / 0 fail) |
| 7b | `pytest tests/auth/ -q` | PASS — **11 passed, 1 skipped** (regression IDENT-06 intacta) |
| 8a | `grep -qE "^test-data:" Makefile` | PASS — línea 77 |
| 8b | `grep -q "NEXO_AUTO_MIGRATE" .env.example` | PASS — bloque Phase 3 añadido |
| 9 | `make test-data` → exit 0 | PASS — 16 passed |

## Decisiones tomadas en ejecución

1. **`schema_guard.verify` acepta `critical_tables` kwarg** (default `CRITICAL_TABLES`). Los tests inyectan `("users", "__nonexistent_table__")` sin `monkeypatch.setattr(schema_guard, "CRITICAL_TABLES", ...)`. Rationale: más robusto si algún día `CRITICAL_TABLES` migra a `frozenset` o se computa al importar. Patrón explícitamente pedido por el plan (validation sign-off 2026-04-19 punto 3).
2. **`engine_app` re-exportado desde `api/database.py`** con `from api.database import engine as engine_app` — NO se duplica el `_mssql_creator` pyodbc. Landmine #3 del RESEARCH. Si algún día duplicamos la connection string nos arriesgamos a divergencia silenciosa entre el engine del pipeline OEE y el que usan los repos Nexo.
3. **`engine_nexo` preserva `settings.effective_pg_user` / `effective_pg_password`**. Si se usase `settings.pg_user` a secas, el runtime conectaría como owner y `tests/auth/test_audit_append_only.py` dejaría de capturar el gate IDENT-06 (el test pasaría porque el owner SÍ puede UPDATE/DELETE audit_log). Landmine #12.
4. **`schema_guard.verify(engine_nexo)` va FUERA del try/except de `init_db()`** en lifespan. Si schema_guard lanza `RuntimeError`, uvicorn propaga y la app no arranca — drift de schema detectado antes del primer request. `init_db()` sigue best-effort (los schemas cfg/oee/luk4 ya existen en SQL Server, sólo hace bootstrap idempotente de CSVs).
5. **Orden LIFO de middlewares intacto** (`api/main.py:123-124` — `AuditMiddleware` luego `AuthMiddleware`). Landmine #11. Todo el wiring de schema_guard ocurre en `lifespan`, no toca `add_middleware`.
6. **`nexo/db/engine.py` reducido a shim de 10 líneas** que re-exporta `engine_nexo` y `SessionLocalNexo`. Los consumidores Phase 2 (`nexo/services/auth.py`, `api/routers/auditoria.py`, `api/routers/usuarios.py`, `scripts/init_nexo_schema.py`, `scripts/create_propietario.py`, `tests/auth/*`) NO se modifican en 03-01. El shim se elimina al final de 03-03.
7. **`nexo/data/dto/base.py` expone `ROW_CONFIG`** — `ConfigDict(frozen=True, from_attributes=True)` compartible por los DTOs concretos que aterrizan en 03-02 / 03-03. Los tests usan un `_RowDummy(BaseModel)` ad-hoc para validar el contrato sin depender de DTOs reales.

## Archivos creados (14)

- `nexo/data/__init__.py` — paquete marker + docstring
- `nexo/data/engines.py` — 3 engines + 2 session factories + builders `_build_pg_dsn` y `_build_mes_dsn`
- `nexo/data/sql/__init__.py` — paquete marker (necesario para `importlib.resources`)
- `nexo/data/sql/loader.py` — `load_sql(name)` con `@lru_cache(128)` e `importlib.resources.files`
- `nexo/data/dto/__init__.py` — paquete marker + docstring
- `nexo/data/dto/base.py` — `ROW_CONFIG` (Pydantic v2 frozen + from_attributes)
- `nexo/data/schema_guard.py` — `verify(engine, critical_tables=CRITICAL_TABLES)` + `CRITICAL_TABLES`
- `tests/data/__init__.py` — test marker
- `tests/data/conftest.py` — fixtures `db_nexo` / `db_app` / `engine_mes_mock` + helpers de reachability
- `tests/data/test_engines.py` — 5 tests (DATA-01, DATA-09 prep, DATA-11)
- `tests/data/test_sql_loader.py` — 3 tests (DATA-05)
- `tests/data/test_schema_guard.py` — 3 tests (DATA-06, todos `integration`)
- `tests/data/test_dto_immutable.py` — 2 tests (DATA-08)
- `tests/data/test_fixtures.py` — 3 tests (DATA-10 meta)

## Archivos modificados (5)

- `nexo/db/engine.py` — reducido a shim (re-export de `engine_nexo` + `SessionLocalNexo`)
- `api/main.py` — `lifespan` llama `schema_guard.verify(engine_nexo)` antes de `init_db()`
- `api/deps.py` — añade `get_db_app`, `get_db_nexo`, `get_engine_mes` + `DbApp`, `DbNexo`, `EngineMes`
- `Makefile` — `test-data` añadido a `.PHONY` + target que arranca compose db y corre `pytest tests/data/`
- `.env.example` — bloque "Schema guard (Phase 3)" con `NEXO_AUTO_MIGRATE=false` y comentario "SOLO dev"

## Deviations from Plan

None — plan executed exactly as written.

## Authentication Gates

None — 03-01 es foundation pura, sin auth gates.

## Tests — detalle

Resultado de `docker compose exec web pytest tests/data/ -v`:

- **test_engines.py** (5 tests): `test_three_engines_exported`, `test_two_session_factories_exported`, `test_mes_pool_config`, `test_mes_url_has_mes_db_as_default_catalog`, `test_engine_mes_database_is_dbizaro` (integration, pasa contra SQL Server real en este entorno).
- **test_sql_loader.py** (3 tests): `test_load_sql_has_cache`, `test_load_sql_normalizes_extension`, `test_load_sql_missing_raises`.
- **test_schema_guard.py** (3 tests, integration): `test_verify_ok_with_all_tables`, `test_verify_raises_when_table_missing`, `test_auto_migrate_creates_missing`.
- **test_dto_immutable.py** (2 tests): `test_frozen_dto_rejects_mutation`, `test_frozen_dto_roundtrips_from_attributes`.
- **test_fixtures.py** (3 tests): `test_db_nexo_fixture_yields_session`, `test_engine_mes_mock_is_magicmock`, `test_db_app_fixture_optional`.

**Total: 16 passed in 0.76s**. Sólo 9 son "unit (`not integration`)"; los 7 restantes requieren Postgres. En CI sin Postgres, los 7 integration se skipean y el exit code sigue siendo 0.

## Regression IDENT-06

`docker compose exec web pytest tests/auth/ -q` → **11 passed, 1 skipped** (mismo resultado que al arrancar 03-01). El skip viene de `test_rbac_smoke.py` cuando el rol `nexo_app` no está configurado en el entorno — comportamiento pre-existente, no introducido por 03-01.

## Hallazgos para 03-02 y 03-03

- **`engine_mes.url.database == 'dbizaro'`** verificado en runtime → DATA-09 queda habilitado. En 03-02, al escribir `nexo/data/sql/mes/centro_mando_fmesmic.sql`, las queries pueden usar `FROM admuser.fmesmic` (2-part name) sin `dbizaro.` prefix.
- **Shim `nexo/db/engine.py` funciona** con todos los consumidores Phase 2. Cualquier import de `from nexo.db.engine import engine_nexo, SessionLocalNexo` sigue operativo. 03-03 puede migrar los consumidores de uno en uno sin prisa; al final del plan se elimina el shim.
- **`lru_cache` del loader** no se invalida entre tests (esperado: los `.sql` son estáticos). Los tests usan `load_sql.cache_clear()` explícitamente donde relevante; no hay fugas entre tests.
- **Fixture `engine_mes_mock`** monkeypatchea `nexo.data.engines.engine_mes` globalmente para la duración del test — 03-02 puede consumirla directamente en `tests/data/test_mes_repository.py` sin necesidad de ajustes.
- **Paralelización autorizada**: 03-02 y 03-03 tocan repos/routers disjuntos. Ambos dependen de esta foundation, que queda sólidamente instalada. No hay blockers abiertos.
- **Riesgos residuales**: ninguno. El único warning en logs es el `DeprecationWarning` de `starlette.testclient` sobre `cookies=` per-request — pre-existente, no afecta runtime.

## Self-Check: PASSED

Archivos creados/modificados verificados:

- `nexo/data/engines.py` → FOUND (exporta 5 símbolos, engine_mes.pool_recycle=3600, url.database=dbizaro).
- `nexo/data/sql/loader.py` → FOUND (load_sql tiene cache_info).
- `nexo/data/schema_guard.py` → FOUND (verify tiene kwarg critical_tables, CRITICAL_TABLES=8 tablas).
- `nexo/data/dto/base.py` → FOUND (ROW_CONFIG importable).
- `nexo/db/engine.py` → FOUND (shim, 10 líneas efectivas).
- `api/main.py` → MODIFIED (lifespan contiene `schema_guard.verify(engine_nexo)`).
- `api/deps.py` → MODIFIED (exporta 3 deps + 3 aliases).
- `Makefile` → MODIFIED (test-data target en línea 77, .PHONY actualizado).
- `.env.example` → MODIFIED (NEXO_AUTO_MIGRATE=false presente).
- `tests/data/` (7 archivos) → FOUND.

Commits verificados:

- `4325943` engines move + shim → FOUND en `git log`.
- `bb6cd62` loader + schema_guard + dto → FOUND.
- `9c81227` lifespan wiring + deps + Makefile + .env.example → FOUND.
- `5c70f0b` tests/data/ → FOUND.

Gate results: **todos PASS**. Plan 03-01 COMPLETE. Ready for 03-02 y 03-03 (paralelización autorizada).
