# Phase 3: Capa de datos — Research

**Researched:** 2026-04-19
**Domain:** SQLAlchemy 2.0 multi-engine repositorios + FastAPI DI + Pydantic v2 DTOs + Postgres test fixtures + legacy MES wrapper
**Confidence:** HIGH en patrones SQLAlchemy/Pydantic/FastAPI; MEDIUM en pandas+SQLAlchemy 2.0 interaction; HIGH en idiomática del propio codebase (revisado in-situ).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-01 — SQL loader**
- Placeholders: SQLAlchemy `text()` con params `:named`. DATA-05 menciona `?` pero la implementación usa `:named` (uniforme pyodbc/SQL Server + psycopg2/Postgres).
- Layout: `nexo/data/sql/<engine>/<method_name>.sql` — un archivo = una query canónica.
- Loader: `nexo/data/sql/loader.py:load_sql(name: str) -> str` con `@lru_cache`, lee vía `importlib.resources`.
- IN-clauses dinámicos: `text(...).bindparams(bindparam("codes", expanding=True))`. Sin Jinja.
- Branching: si una query necesita dos formas → dos archivos. `.sql` puros.
- Comentarios preservados: el filtro T3 cruza medianoche va como header `-- NOTA T3: ...`.

**D-02 — Repository shape**
- Sesión inyectada, repos sin transacción. `Session` para `engine_app`/`engine_nexo`, `Engine` para `engine_mes` (read-only).
- Repos NO hacen `commit()` ni `rollback()`. Transacción = caller.
- Nuevas dependencies `api/deps.py`: `get_db_app()`, `get_db_nexo()`, `get_engine_mes()`.
- Retornos: DTOs Pydantic **frozen** en `nexo/data/dto/` para lo que cruza HTTP. ORM sólo intra-`nexo/services/`. Routers nunca reciben ORM.
- Naming DTO: sufijo `Row` (`ProduccionRow`, `RecursoRow`, `CapacidadRow`, `AuditLogRow`, `CicloRow`, `OperarioRow`, `EstadoMaquinaRow`).
- `AuditRepo.append(...)` solo INSERT; caller orquesta transacción (coherente con gate IDENT-06).

**D-03 — Plan breakdown** (3 plans)
- 03-01 foundation: mueve `engine_nexo` a `nexo/data/engines.py`, crea `engine_mes`, re-exporta `engine_app`, loader, DTO base, schema_guard wired en lifespan, fixtures pytest. Sin tocar routers ni `nexo/services/auth.py`.
- 03-02 capa MES (paralelo con 03-03 tras 03-01): `MesRepository` con 5 métodos (wrappers delgados sobre `OEE/db/connector.py`), refactor 5 routers (`bbdd`, `capacidad`, `operarios`, `centro_mando`, `luk4`). Tests con `engine_mes` mockeado.
- 03-03 capa APP+NEXO (paralelo): `RecursoRepo`, `CicloRepo`, `EjecucionRepo`, `MetricaRepo`, `LukRepo`, `ContactoRepo` + `UserRepo`, `RoleRepo`, `AuditRepo`. Refactor `historial`, `recursos`, `ciclos`, `nexo/services/auth.py`, `api/routers/auditoria.py`, `api/routers/usuarios.py`. Tests contra Postgres real del compose.
- Cada plan = commit atómico, revert isolado.

**D-04 — Pipeline OEE (wrapper superficial)**
- `OEE/db/connector.py` (847 LOC) NO se reescribe. Se mantienen `extraer_datos`, `detectar_recursos`, `calcular_ciclos_reales`, `estado_maquina_live`, `_build_connection_string`, `load_config`, `save_config`, `detectar_driver`.
- `MesRepository.extraer_datos_produccion(...)` → delega a `OEE.db.connector.extraer_datos(...)`. Idem para los otros 3 métodos.
- `api/services/pipeline.py` y `api/services/db.py` cambian el import (`OEE.db.connector` → `nexo.data.repositories.mes.MesRepository`). Lógica intacta.
- Regression gate (success criterion #5): antes del refactor, generar PDF con fecha conocida y hashear. Tras refactor, repetir con mismos inputs. Si no hay fecha reproducible → count/total páginas + bytes.

**D-05 — `/bbdd` explorer**
- Whitelist anti-DDL/DML se queda en `api/routers/bbdd.py` (validación del transporte).
- `MesRepository.consulta_readonly(sql: str, database: str) -> list[dict]` recibe SQL ya validada.
- `list_databases`, `list_tables`, `list_columns`, `preview` quedan inline en `bbdd.py` — específicos de UI, no justifican repo.
- El check anti-forbidden (`re.match`) se conserva y se testea con SQL malicioso.

**D-06 — Retrofit routers**
- Orden del plan: de más simple a más complejo.
  - 03-02: `centro_mando.py` (2 q) → `capacidad.py` (3 q) → `operarios.py` (6 q) → `luk4.py` → `bbdd.py` (10+ q + SQL libre).
  - 03-03: `ciclos.py` → `recursos.py` → `historial.py` → `nexo/services/auth.py` + `auditoria.py` + `usuarios.py`.
- Cada router pasa smoke HTTP antes de continuar.

**D-07 — schema_guard**
- Valida existencia de tablas críticas schema `nexo`. NO valida columnas/tipos (gap aceptable Mark-III).
- `NEXO_AUTO_MIGRATE=false` (default): falta tabla → `RuntimeError`, lifespan aborta.
- `NEXO_AUTO_MIGRATE=true`: `NexoBase.metadata.create_all(bind=engine_nexo)` + log WARN. Solo dev/primer deploy.
- Scope solo schema `nexo`. No toca `ecs_mobility.cfg.*` ni `dbizaro`.

**D-08 — Tests DB strategy**
- Postgres real vía `docker compose up db` (APP + NEXO). Fixture sesión con rollback (no truncate).
- Mocks para `engine_mes`. Queries MES se testean con fixtures de respuesta grabadas.
- `tests/data/conftest.py` provee `db_nexo`, `db_app`, `engine_mes_mock`.
- CI: `make test-data` arranca compose → pytest → apaga. GitHub Actions futuro (Phase 7).

### Claude's Discretion

- Orden exacto de refactor dentro de 03-02 y 03-03 (empezar por simple; planner concreta).
- Forma de los DTOs (campos, validadores) — copiar shape actual de los routers.
- Naming interno del loader (`load_sql` vs `sql`) — planner decide.
- Si `lru_cache` del loader se invalida en test — probablemente no, `.sql` son estáticos.
- Si se graba PDF de referencia para hash guard de D-04 — planner elige fecha histórica con datos estables.

### Deferred Ideas (OUT OF SCOPE)

- Validación columnas + tipos en `schema_guard` — Mark-IV (drift detector).
- `engine_mes` con usuario read-only real a nivel DB (hoy solo código) — Mark-IV.
- Rename `OEE/` → `modules/oee/` — Mark-IV (CLAUDE.md lo confirma).
- Refactor interno 4 módulos OEE (`disponibilidad`, `rendimiento`, `calidad`, `oee_secciones`) — fuera de scope Mark-III por diseño.
- Mover `cfg.*` / `oee.*` / `luk4.*` a Postgres — Power BI + IoT lo impiden. Mark-V+.
- Migración Alembic para `schema_guard` — Mark-IV DevEx.
- Sustituir matplotlib — solo si Phase 4 preflight demuestra inviabilidad.
- CI con GitHub Actions ejecutando `make test-data` — Phase 7.
- Unificar `data/ecs-logo.png` con `static/img/brand/ecs/logo.png` — cleanup oportunista en 03-01 si sale.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DATA-01 | `nexo/data/engines.py` con `engine_mes` + `engine_app` + `engine_nexo` | §SQLAlchemy 2.0 Multi-Engine Patterns — 3 engines coexisten (ya hay 2 en runtime); pattern de re-export `engine_app` |
| DATA-02 | `nexo/data/repositories/mes.py` `MesRepository` con 5 métodos | §Repository + DTO Patterns + §Wrapping OEE/db/connector.py |
| DATA-03 | `nexo/data/repositories/app.py` con 6 repos (Recurso/Ciclo/Ejecucion/Metrica/Luk/Contacto) | §Repository + DTO Patterns (Session injectada, DTOs frozen) |
| DATA-04 | `nexo/data/repositories/nexo.py` con `UserRepo`, `RoleRepo`, `AuditRepo` | §Repository + DTO Patterns (caller orquesta commit, AuditRepo solo INSERT) |
| DATA-05 | `nexo/data/sql/` con `.sql` versionados + loader + comentario T3 preservado | §Implementation Patterns §Pattern 2 (loader con `lru_cache` + `importlib.resources`) |
| DATA-06 | `schema_guard` en lifespan; `NEXO_AUTO_MIGRATE=true` crea tablas | §Lifespan + schema_guard Implementation |
| DATA-07 | Routers consumen repos, 0 `pyodbc` directo, 0 SQL inline en los 8 routers | §Repository + DTO Patterns + orden D-06 |
| DATA-08 | DTOs Pydantic `*Row` en `nexo/data/dto/` | §Pydantic v2 DTO Recipe |
| DATA-09 | Eliminar 3-part names `dbizaro.admuser.*` | §DATA-09: Killing 3-part Names |
| DATA-10 | Tests `tests/data/` por repo; Postgres real + mocks MES | §Postgres Test Fixtures + §MES Mocking Strategy |
| DATA-11 | `engine_mes` con `pool_recycle=3600`, `pool_pre_ping=True`, timeout 15s | §SQLAlchemy 2.0 Multi-Engine Patterns §Pool Configuration |

</phase_requirements>

## Research Summary

Phase 3 no inventa tecnología: reordena código existente detrás de una capa de repositorios. Los tres engines que propone D-01/DATA-01 (`engine_mes` dbizaro read-only, `engine_app` ecs_mobility, `engine_nexo` Postgres) coexisten trivialmente en SQLAlchemy 2.0 — el codebase ya corre dos (`api/database.py:engine` + `nexo/db/engine.py:engine_nexo`) sin fricción. Los patrones de repository + DTO inyectados vía `Depends` son el mainstream 2024-2025 para FastAPI síncrono; no hay trampa de async aquí porque Nexo es síncrono de punta a punta (FastAPI + SQLAlchemy síncrono + slowapi + itsdangerous).

**Riesgo principal:** success criterion #5 (PDFs idénticos a Mark-II). El wrapper delgado sobre `OEE/db/connector.py` minimiza la superficie de cambio, pero el hash de regresión tiene que materializarse como tarea explícita en el plan 03-02, no como afterthought. Secundario: `pandas==3.0.2` con `pyodbc.connect()` directo (no SQLAlchemy engine) en `extraer_datos` — pandas 2.x+ emite `UserWarning` si recibe DBAPI connection que no es sqlite3, pero la llamada sigue funcionando y devuelve el DataFrame correcto. No hay que cambiar nada aquí, pero el warning puede contaminar logs.

**Lo que está resuelto:** el patrón de engines múltiples (Phase 2 ya introdujo un segundo engine limpio); el loader con `lru_cache` + `importlib.resources` es canónico; Pydantic v2 `model_config = ConfigDict(frozen=True, from_attributes=True)` es la forma explícita para DTOs inmutables hidratables desde ORM; `inspect(engine).has_table("users", schema="nexo")` es la API 2.0 correcta para `schema_guard`; el gate IDENT-06 ya demostró que el rol `nexo_app` bloquea UPDATE/DELETE en `audit_log` (tests pasan).

**Primary recommendation:** ejecutar 03-01 como foundation puro (engines + loader + DTOs base + schema_guard + fixtures) con commit atómico verde antes de arrancar 03-02/03-03. Gate IDENT-06 debe seguir verde tras la reorganización (el motor sigue siendo el mismo, sólo cambia de archivo). El hash guard de D-04 se debe implementar como una *task* explícita al inicio de 03-02, no al final: grabar PDF pre-refactor inmediatamente tras `git checkout` de la rama de trabajo.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| SQL query definitions | `nexo/data/sql/<engine>/*.sql` | — | Versionadas en disco, una por método repo (D-01) |
| SQL loader (cache) | `nexo/data/sql/loader.py` | — | `lru_cache` + `importlib.resources`, nadie más toca archivos `.sql` |
| Engine management | `nexo/data/engines.py` | `api/config.py` (credenciales) | 3 engines declarados en un solo lugar, config viene de `settings` |
| Repository (MES) | `nexo/data/repositories/mes.py` | `OEE/db/connector.py` (delega) | Wrapper delgado; la lógica MES no se reescribe |
| Repository (APP/NEXO) | `nexo/data/repositories/{app,nexo}.py` | ORM models en `nexo/data/models_{app,nexo}.py` | Patrón repo + session inyectada |
| DTOs de transporte | `nexo/data/dto/*.py` | — | Pydantic frozen, todo lo que sale a router |
| Schema validation | `nexo/data/schema_guard.py` | `api/main.py:lifespan()` | Arranca el check al inicio, aborta o auto-migrate |
| Transacción | Router / `nexo/services/*` | Repo (nunca) | Repos exponen operaciones, caller commit/rollback |
| Session injection | `api/deps.py` | FastAPI `Depends` | `get_db_app`, `get_db_nexo`, `get_engine_mes` |
| Validation anti-DDL (`/bbdd`) | `api/routers/bbdd.py` | `MesRepository.consulta_readonly` (acepta SQL ya validada) | Whitelist es concern de transporte, no de repo (D-05) |
| Fixture Postgres (tests) | `tests/data/conftest.py` | `docker compose db` | Postgres real del compose, no testcontainers |
| Mock MES (tests) | `tests/data/conftest.py` | `unittest.mock` | CI no tiene SQL Server; mocks sobre `Engine.connect()` |

## Standard Stack

### Core — ya en requirements.txt, versiones verificadas

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| sqlalchemy | 2.0.49 | Engine, text(), bindparam(expanding=True), inspect() | [VERIFIED: requirements.txt] 2.0.x es la línea con statement cache + compat pyodbc/psycopg2 |
| psycopg2-binary | 2.9.11 | Driver Postgres (engine_nexo) | [VERIFIED: requirements.txt] binary evita compilación; OK en contenedor |
| pyodbc | 5.3.0 | Driver SQL Server (engine_mes + engine_app) | [VERIFIED: requirements.txt] consumido por `_build_connection_string` en OEE y `_mssql_creator` en api/database.py |
| pydantic | 2.x (via pydantic-settings 2.13.1) | DTOs frozen `*Row` | [VERIFIED: requirements.txt pydantic-settings==2.13.1 trae pydantic 2.x] |
| fastapi | 0.135.3 | lifespan, Depends | [VERIFIED: requirements.txt] lifespan pattern `@asynccontextmanager` establecido |
| pandas | 3.0.2 | `read_sql` en `OEE/db/connector.extraer_datos` | [VERIFIED: requirements.txt] se mantiene (wrapper no toca la función) |

### Supporting — ya en dev

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | 8.3.4 | Framework tests | [VERIFIED: requirements-dev.txt] |
| pytest-cov | 6.0.0 | Coverage para evaluar gap en tests data | [VERIFIED: requirements-dev.txt] |
| httpx | 0.28.1 | `TestClient(app)` en smokes HTTP | [VERIFIED: requirements-dev.txt] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `docker compose db` real | testcontainers-python | testcontainers da aislamiento por container ephemeral pero añade dependencia (~50MB) + orquesta docker desde pytest. CONTEXT.md D-08 lo descarta: compose ya está, devs ya lo conocen. |
| Mocks MES con `MagicMock` | SQLite in-memory como stand-in | SQLite no habla T-SQL (CONVERT, RTRIM, TOP). Romper queries con dialecto cruzado. `MagicMock` en `Engine.connect()` + grabar respuestas es más fiel. |
| `session.rollback()` por test | `TRUNCATE` entre tests | Rollback es más rápido y no requiere permisos de TRUNCATE (nexo_app no los tiene, precisamente). CONTEXT.md D-08 lo fija. |
| `lru_cache` del loader | No-cache, `open()` cada vez | `.sql` son estáticos en runtime; overhead I/O innecesario. Invalidación en test no es problema si fixtures no mutan archivos. |
| Jinja para SQL dinámico | Branching con archivos `.sql` + IN-clauses con `expanding=True` | Jinja añade dependencia + superficie de injection. D-01 lo descarta explícitamente. |

**Version verification:**
```bash
# Todas verificadas contra requirements.txt ya pineado (Phase 1)
grep -E "^(sqlalchemy|psycopg2|pyodbc|pydantic|fastapi|pandas)" requirements.txt requirements-dev.txt
```
No hace falta `npm view` / `pip index versions` — las versiones están pineadas en el repo y no se tocan en Phase 3.

## Architecture Patterns

### System Architecture Diagram — Phase 3 post-refactor

```
                     ┌─────────────────────────┐
      HTTP ──────────►   FastAPI Router        │
                     │  (bbdd/capacidad/...)   │
                     └──────┬──────────┬───────┘
                            │          │
                            │          ▼
                            │   api/deps.py
                            │   get_db_app / get_db_nexo / get_engine_mes
                            │          │
                            ▼          ▼
                     ┌─────────────────────────┐
                     │   nexo/data/repositories │
                     │   ┌────────┬───────┬───┐ │
                     │   │ mes.py │app.py │nx │ │
                     │   └────┬───┴───┬───┴─┬─┘ │
                     └────────┼───────┼─────┼───┘
                              │       │     │
                              ▼       ▼     ▼
                         ┌──────┐ ┌──────┐ ┌──────┐
                         │engine│ │engine│ │engine│
                         │ _mes │ │ _app │ │_nexo │
                         └───┬──┘ └───┬──┘ └───┬──┘
                             │        │        │
                             │        │        │
                      ┌──────▼──┐ ┌───▼───┐ ┌──▼───┐
                      │OEE/db/  │ │ MSSQL │ │  PG  │
                      │connector│ │ecs_mob│ │ nexo.│
                      │ (wrap)  │ └───────┘ └──────┘
                      └────┬────┘
                           ▼
                      dbizaro MES (read-only)

  Lifespan (api/main.py):
    1. schema_guard.verify(engine_nexo)  ◄── NEW: abort or auto-migrate
    2. init_db()                         ◄── existing (ecs_mobility bootstrap)

  DTOs frozen (nexo/data/dto/) cruzan el borde router↔repo.
  ORM entities (NexoUser, Recurso, Ciclo, ...) solo intra-services.
  `.sql` files loaded by nexo/data/sql/loader.load_sql(name) [lru_cache]
```

### Recommended Project Structure — post Phase 3

```
nexo/
├── data/
│   ├── __init__.py
│   ├── engines.py                 # 3 engines + 3 session factories
│   ├── schema_guard.py            # verify() + auto_migrate()
│   ├── models_app.py              # NEW: ORM ecs_mobility (migrated from api/database.py)
│   ├── models_nexo.py             # NEW: ORM nexo.* (migrated from nexo/db/models.py)
│   ├── dto/
│   │   ├── __init__.py
│   │   ├── mes.py                 # ProduccionRow, EstadoMaquinaRow, CicloRealRow
│   │   ├── app.py                 # RecursoRow, CicloRow, EjecucionRow, MetricaRow, LukRow, ContactoRow
│   │   └── nexo.py                # UserRow, RoleRow, AuditLogRow
│   ├── repositories/
│   │   ├── __init__.py
│   │   ├── mes.py                 # MesRepository (5 métodos)
│   │   ├── app.py                 # RecursoRepo/CicloRepo/EjecucionRepo/MetricaRepo/LukRepo/ContactoRepo
│   │   └── nexo.py                # UserRepo/RoleRepo/AuditRepo
│   ├── sql/
│   │   ├── loader.py              # load_sql(name) + @lru_cache
│   │   ├── mes/
│   │   │   ├── extraer_datos_produccion.sql
│   │   │   ├── estado_maquina_live.sql
│   │   │   ├── calcular_ciclos_reales.sql
│   │   │   ├── capacidad_piezas_linea.sql
│   │   │   ├── capacidad_ciclos_p10_180d.sql
│   │   │   ├── centro_mando_fmesmic.sql
│   │   │   ├── operarios_listar.sql
│   │   │   └── ... (uno por método repo)
│   │   ├── app/                   # (solo si hay SQL específicas; la mayoría son ORM queries)
│   │   └── nexo/                  # (solo si hay SQL específicas; casi todas ORM)
│   └── __init__.py
├── db/                            # SHIM durante Phase 3
│   ├── engine.py                  # re-export from nexo.data.engines
│   └── models.py                  # re-export from nexo.data.models_nexo
└── services/
    └── auth.py                    # (unchanged in 03-01; migrado a UserRepo en 03-03)
```

### Pattern 1: Sesión inyectada + repo sin transacción

**What:** Cada repo recibe `Session` (APP/NEXO) o `Engine` (MES) en `__init__`. Métodos no comitean. Caller (router o service) controla la transacción.

**When to use:** Siempre en Phase 3. Coherente con patrón existente en `api/routers/auditoria.py:get_nexo_db()` y gate IDENT-06 (AuditRepo solo INSERT, commit del caller).

**Example:**
```python
# nexo/data/repositories/nexo.py
from sqlalchemy import select
from sqlalchemy.orm import Session
from nexo.data.models_nexo import NexoUser, NexoAuditLog
from nexo.data.dto.nexo import UserRow, AuditLogRow

class UserRepo:
    def __init__(self, db: Session):
        self._db = db

    def get_by_email(self, email: str) -> UserRow | None:
        row = self._db.execute(
            select(NexoUser).where(NexoUser.email == email, NexoUser.active.is_(True))
        ).scalar_one_or_none()
        return UserRow.model_validate(row) if row else None

class AuditRepo:
    def __init__(self, db: Session):
        self._db = db

    def append(self, *, user_id: int | None, ip: str, method: str,
               path: str, status: int, details_json: str | None) -> None:
        """INSERT solo. Caller comitea. IDENT-06: rol nexo_app bloquea UPDATE/DELETE."""
        self._db.add(NexoAuditLog(
            user_id=user_id, ip=ip, method=method,
            path=path, status=status, details_json=details_json,
        ))
        # NO commit here — caller owns the transaction
```

```python
# api/deps.py
from typing import Iterator
from sqlalchemy.orm import Session
from nexo.data.engines import SessionLocalApp, SessionLocalNexo, engine_mes

def get_db_app() -> Iterator[Session]:
    db = SessionLocalApp()
    try:
        yield db
    finally:
        db.close()

def get_db_nexo() -> Iterator[Session]:
    db = SessionLocalNexo()
    try:
        yield db
    finally:
        db.close()

def get_engine_mes():
    """MES es read-only; entregamos el Engine directamente — no hay session stateful."""
    return engine_mes
```

### Pattern 2: `.sql` loader con `lru_cache` + `importlib.resources`

**What:** Un archivo `.sql` por método de repo. Loader central que los lee una vez y cachea.

**Example:**
```python
# nexo/data/sql/loader.py
from __future__ import annotations
from functools import lru_cache
from importlib.resources import files

_PACKAGE = "nexo.data.sql"

@lru_cache(maxsize=128)
def load_sql(name: str) -> str:
    """Carga 'mes/extraer_datos_produccion.sql' desde el paquete.

    El nombre incluye el subdirectorio (engine) y extensión opcional.
    Normaliza: 'mes/estado' → 'mes/estado.sql'.
    """
    if not name.endswith(".sql"):
        name = f"{name}.sql"
    # files() devuelve Traversable. Para subdirs usa / operator.
    # Ej: files("nexo.data.sql") / "mes" / "estado_maquina_live.sql"
    subpath = name.replace("/", "__PSEP__").split("__PSEP__")
    ref = files(_PACKAGE)
    for part in subpath:
        ref = ref / part
    return ref.read_text(encoding="utf-8")
```

```python
# Uso en repo:
from sqlalchemy import bindparam, text
from nexo.data.sql.loader import load_sql

class MesRepository:
    def __init__(self, engine):
        self._engine = engine

    def consulta_readonly(self, sql: str, database: str) -> list[dict]:
        # SQL recibida ya validada por el router (D-05)
        with self._engine.connect() as conn:
            rows = conn.execute(text(sql)).mappings().all()
            return [dict(r) for r in rows]

    def centro_mando_fmesmic(self, ct_codes: list[int]) -> list[dict]:
        stmt = text(load_sql("mes/centro_mando_fmesmic")).bindparams(
            bindparam("codes", expanding=True)
        )
        with self._engine.connect() as conn:
            rows = conn.execute(stmt, {"codes": [str(c) for c in ct_codes]}).mappings().all()
            return [dict(r) for r in rows]
```

```sql
-- nexo/data/sql/mes/centro_mando_fmesmic.sql
-- NOTA: tras Phase 3 DATA-09, engine_mes tiene DATABASE=dbizaro en connection string,
-- por eso usamos 2-part names (admuser.fmesmic) en lugar de 3-part (dbizaro.admuser.fmesmic).
SELECT
    CAST(RTRIM(mi020) AS INT)    AS ct,
    COUNT(*)                     AS piezas_hoy,
    MAX(CAST(mi100 AS TIME))     AS ultimo_evento,
    (SELECT TOP 1 RTRIM(m2.mi060)
     FROM admuser.fmesmic m2
     WHERE RTRIM(m2.mi020) = RTRIM(m.mi020)
       AND CONVERT(DATE, m2.mi090) = CONVERT(DATE, GETDATE())
     ORDER BY m2.mi050 DESC)     AS referencia
FROM admuser.fmesmic m
WHERE CONVERT(DATE, mi090) = CONVERT(DATE, GETDATE())
  AND RTRIM(mi020) IN :codes
GROUP BY RTRIM(mi020)
```

### Pattern 3: Multi-engine coexistence

**What:** Tres `create_engine` en un solo módulo con tres `sessionmaker` correspondientes. Pools independientes.

**Example:**
```python
# nexo/data/engines.py
from __future__ import annotations
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from api.config import settings

# ── engine_mes (dbizaro, read-only de facto; DATA-11 pool settings) ─────
def _build_mes_dsn() -> str:
    # pyodbc DSN-as-URL-query siguiendo patrón existente (api/database.py).
    # Hoy MES y APP comparten instancia; en el futuro split es solo cambio de env.
    pwd = settings.mes_password.replace("+", "%2B")
    return (
        f"mssql+pyodbc://{settings.mes_user}:{pwd}"
        f"@{settings.mes_server}:{settings.mes_port}/{settings.mes_db}"
        "?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes&Encrypt=yes"
    )

engine_mes: Engine = create_engine(
    _build_mes_dsn(),
    pool_pre_ping=True,     # DATA-11
    pool_recycle=3600,      # DATA-11
    pool_size=3,
    max_overflow=2,
    pool_timeout=15,        # DATA-11 — connection timeout
    connect_args={"timeout": 15},  # pyodbc-level timeout
)

# ── engine_app (ecs_mobility) — re-export del existente en api/database.py ─
# Durante Phase 3 re-exportamos. En Phase 4+ movemos `_mssql_creator` aquí.
from api.database import engine as engine_app  # noqa: E402

SessionLocalApp = sessionmaker(bind=engine_app, autoflush=False,
                               autocommit=False, expire_on_commit=False)

# ── engine_nexo (Postgres) — movido desde nexo/db/engine.py en 03-01 ──────
def _build_nexo_dsn() -> str:
    user = settings.effective_pg_user
    pwd = settings.effective_pg_password
    return f"postgresql+psycopg2://{user}:{pwd}@{settings.pg_host}:{settings.pg_port}/{settings.pg_db}"

engine_nexo: Engine = create_engine(
    _build_nexo_dsn(),
    pool_size=5, max_overflow=5,
    pool_timeout=10, pool_recycle=1800,
    pool_pre_ping=True,
    echo=settings.debug,
)

SessionLocalNexo = sessionmaker(bind=engine_nexo, autoflush=False,
                                autocommit=False, expire_on_commit=False)
```

**Shim durante transición (03-01):**
```python
# nexo/db/engine.py — SHIM, eliminar en 03-03 cuando auth.py migre
from nexo.data.engines import engine_nexo, SessionLocalNexo  # noqa: F401
```

### Anti-Patterns to Avoid

- **Repo que haga `commit()`**: rompe el modelo de transacción del caller; IDENT-06 asume caller controla.
- **Router que reciba ORM entity**: acopla router a schema ORM. Siempre DTO frozen.
- **Jinja en SQL**: introduce interpolación insegura + dependencia innecesaria. D-01 prohibido.
- **Hacer `create_all` en runtime sin flag**: `schema_guard` con `NEXO_AUTO_MIGRATE=true` es opt-in por diseño; default `false` para prod.
- **SQLite como stand-in de SQL Server en tests**: dialectos incompatibles (T-SQL `CONVERT`, `TOP`, `RTRIM` con semánticas distintas). Mock el engine o graba respuestas.
- **Re-exportar el engine_nexo del shim eternamente**: el shim es temporal (03-01 → 03-03). Eliminar al migrar `nexo/services/auth.py`.

## Repository + DTO Patterns (FastAPI)

### Shape canónica por repo

Cada repo sigue este contrato:

```python
class FooRepo:
    """Sesión inyectada, sin transacción interna."""

    def __init__(self, db: Session): ...

    # Listado → list[DTO]
    def list_all(self, *, limit: int = 100) -> list[FooRow]: ...

    # Lookup → DTO | None
    def get_by_id(self, foo_id: int) -> FooRow | None: ...

    # Mutación → DTO del recurso afectado (pero NO commit)
    def create(self, *, field1, field2) -> FooRow: ...
    def update(self, foo_id: int, *, field1) -> FooRow | None: ...
    def delete(self, foo_id: int) -> bool: ...
```

### Dependency factories en `api/deps.py`

```python
# api/deps.py — añadir al final
from typing import Annotated
from fastapi import Depends
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine

from nexo.data.engines import SessionLocalApp, SessionLocalNexo, engine_mes as _engine_mes

def get_db_app():
    db = SessionLocalApp()
    try: yield db
    finally: db.close()

def get_db_nexo():
    db = SessionLocalNexo()
    try: yield db
    finally: db.close()

def get_engine_mes() -> Engine:
    return _engine_mes

# Alias typed para legibilidad en routers (PEP 593 Annotated, ya en stack)
DbApp = Annotated[Session, Depends(get_db_app)]
DbNexo = Annotated[Session, Depends(get_db_nexo)]
EngineMes = Annotated[Engine, Depends(get_engine_mes)]
```

### Uso en router (post-refactor)

```python
# api/routers/historial.py — ejemplo post 03-03
from fastapi import APIRouter, Depends
from api.deps import DbApp
from nexo.data.repositories.app import EjecucionRepo
from nexo.data.dto.app import EjecucionRow
from nexo.services.auth import require_permission

router = APIRouter(
    prefix="/historial",
    dependencies=[Depends(require_permission("historial:read"))],
)

@router.get("")
def listar(db: DbApp, limit: int = 50) -> dict:
    repo = EjecucionRepo(db)
    rows: list[EjecucionRow] = repo.list_recent(limit=limit)
    return {"ejecuciones": [r.model_dump(mode="json") for r in rows]}
```

### Mocking en tests — Protocol vs Fake

**Opción recomendada (más simple):** fake repo concreto que implementa la misma API.

```python
# tests/data/fakes.py
class FakeEjecucionRepo:
    def __init__(self, rows: list[EjecucionRow]):
        self._rows = rows
    def list_recent(self, *, limit: int = 50) -> list[EjecucionRow]:
        return self._rows[:limit]
```

Tests pasan instancia del fake directamente al código testeado. No hace falta `Protocol` en Mark-III — la granularidad de repos es pequeña y Python duck-typea bien sin él. Si en Mark-IV se multiplican los consumidores, migrar a `Protocol` cuando duela.

## Pydantic v2 DTO Recipe

[CITED: docs.pydantic.dev/latest/concepts/models/] Pydantic v2 cambia `Config` por `model_config = ConfigDict(...)`. Las opciones relevantes:

```python
# nexo/data/dto/app.py
from datetime import date, datetime
from decimal import Decimal
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

class RecursoRow(BaseModel):
    model_config = ConfigDict(
        frozen=True,           # Inmutable. __hash__ auto-generado si todos los fields son hashable.
        from_attributes=True,  # Permite .model_validate(orm_instance) (antes orm_mode=True).
    )

    id: int
    centro_trabajo: int
    nombre: str
    seccion: str = "GENERAL"
    activo: bool = True

class MetricaRow(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    ejecucion_id: int
    seccion: Optional[str] = None
    recurso: Optional[str] = None
    fecha: Optional[date] = None
    turno: Optional[str] = None
    disponibilidad_pct: float = 0.0
    rendimiento_pct: float = 0.0
    calidad_pct: float = 0.0
    oee_pct: float = 0.0
    # ... otros 16 campos (ver api/database.py MetricaOEE)
```

**Hidratación desde ORM:**
```python
# En el repo:
from nexo.data.models_app import MetricaOEE
from nexo.data.dto.app import MetricaRow

row = session.get(MetricaOEE, 42)
dto = MetricaRow.model_validate(row)  # usa from_attributes=True
```

**Hidratación desde dict (raw MES output):**
```python
# MES no tiene ORM — devolvemos DTOs desde dicts:
from nexo.data.dto.mes import ProduccionRow
raw_rows: list[dict] = mes_repo.extraer_datos_produccion(fi, ff)
dtos = [ProduccionRow.model_validate(r) for r in raw_rows]
```

**Validators para conversiones defensivas:**
```python
from pydantic import field_validator

class ProduccionRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    recurso: str
    seccion: str
    fecha: date
    h_ini: str
    h_fin: str
    tiempo: float
    proceso: str
    incidencia: str = ""
    cantidad: float = 0
    malas: float = 0
    recuperadas: float = 0
    referencia: str = ""

    @field_validator("fecha", mode="before")
    @classmethod
    def _coerce_fecha(cls, v):
        # pandas a veces devuelve Timestamp; pyodbc datetime. Normalizar a date.
        if hasattr(v, "date") and not isinstance(v, date):
            return v.date()
        return v
```

**Convenciones Mark-III:**
- Sufijo `Row` (no `DTO`, no `Model`) para distinguir de ORM models y de Pydantic request schemas (`api/models.py`).
- Todos `frozen=True`. Mutaciones hacen new instance (filosofía de CLAUDE.md + rules/common/coding-style.md).
- `from_attributes=True` solo en DTOs que hidratan desde ORM (APP + NEXO). MES DTOs pueden omitirlo.
- Nullability explícita con `Optional[...]` + `None` default, no campos `str = ""` silenciosos (excepto cuando el shape actual del router ya lo hace así — copiar comportamiento).

## Lifespan + schema_guard Implementation

```python
# nexo/data/schema_guard.py
from __future__ import annotations
import logging
import os
from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from nexo.data.models_nexo import NexoBase, NEXO_SCHEMA

log = logging.getLogger("nexo.schema_guard")

# Tablas que DEBEN existir para que la app arranque. Subset del metadata
# (no validamos columnas en Mark-III por decisión D-07).
CRITICAL_TABLES = (
    "users", "roles", "departments", "user_departments",
    "permissions", "sessions", "login_attempts", "audit_log",
)


def verify(engine: Engine) -> None:
    """Chequea existencia de tablas nexo.*. Aborta o auto-migra.

    Comportamiento:
    - default (NEXO_AUTO_MIGRATE=false): si falta tabla → RuntimeError con
      mensaje actionable. El lifespan aborta y el usuario corre `make nexo-init`.
    - NEXO_AUTO_MIGRATE=true (opt-in dev): crea las que faltan con
      NexoBase.metadata.create_all(bind=engine) y loggea WARN.

    Scope: solo schema nexo. No toca ecs_mobility ni dbizaro.
    """
    insp = inspect(engine)
    existing = set(insp.get_table_names(schema=NEXO_SCHEMA))
    missing = [t for t in CRITICAL_TABLES if t not in existing]

    if not missing:
        log.info("schema_guard OK — %d tablas nexo presentes", len(CRITICAL_TABLES))
        return

    auto = os.environ.get("NEXO_AUTO_MIGRATE", "").lower() in {"1", "true", "yes"}
    if not auto:
        raise RuntimeError(
            f"Schema guard: faltan tablas en nexo.* → {missing}. "
            f"Ejecuta `make nexo-init` o define NEXO_AUTO_MIGRATE=true "
            f"(solo dev) para crearlas automáticamente."
        )

    log.warning("NEXO_AUTO_MIGRATE activo — creando %d tablas faltantes: %s",
                len(missing), missing)
    NexoBase.metadata.create_all(bind=engine)
    log.warning("auto-migración completada. NO usar en producción.")
```

**Wiring en lifespan:**
```python
# api/main.py — lifespan existente + hook schema_guard
from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.database import init_db
from nexo.data.engines import engine_nexo
from nexo.data import schema_guard

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Schema guard ANTES de init_db — si falla aquí, abortamos claro.
    schema_guard.verify(engine_nexo)

    # 2. init_db existente (bootstrap ecs_mobility)
    try:
        init_db()
        logger.info("Base de datos inicializada OK")
    except Exception as exc:
        logger.error(f"Error inicializando BD: {exc}")
        logger.error(traceback.format_exc())

    yield
```

**Notas importantes:**
- Orden: `schema_guard.verify()` corre ANTES que `init_db()`. Si Postgres está caído o tablas faltan, el error es claro y precoz.
- Si `verify()` lanza, el `yield` no se ejecuta y FastAPI aborta el arranque con traceback completo en stdout. Docker compose lo mostrará como `nexo-web exit 3`.
- Si estamos en modo dev y faltan tablas, `NEXO_AUTO_MIGRATE=true` en `.env` hace que arranque solo. En prod nunca.

[CITED: fastapi.tiangolo.com/advanced/events/] — `@asynccontextmanager` + función async con `yield` es el patrón oficial; el código antes del yield corre al startup.

[CITED: docs.sqlalchemy.org/en/20/core/reflection.html] — `inspect(engine).get_table_names(schema='nexo')` es la API 2.0 correcta para listar tablas de un schema sin traer metadata. Devuelve `list[str]`.

## Postgres Test Fixtures (CI strategy)

**Estrategia fijada por D-08:** Postgres real vía `docker compose up db` (ya levantado por los devs). Mocks para MES.

### Fixtures mínimos para `tests/data/conftest.py`

```python
# tests/data/conftest.py
from __future__ import annotations
from typing import Iterator
from unittest.mock import MagicMock

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from nexo.data.engines import (
    SessionLocalApp, SessionLocalNexo, engine_app, engine_nexo, engine_mes,
)


def _postgres_reachable() -> bool:
    try:
        with engine_nexo.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def _mssql_reachable() -> bool:
    try:
        with engine_app.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


# ── Postgres real (schema nexo) ─────────────────────────────────────────

@pytest.fixture
def db_nexo() -> Iterator[Session]:
    """Sesión con rollback al final — no contamina datos."""
    if not _postgres_reachable():
        pytest.skip("Postgres no arriba: `docker compose up -d db`")
    db = SessionLocalNexo()
    trans = db.begin_nested()  # SAVEPOINT por si el código hace commit intermedio
    try:
        yield db
    finally:
        db.rollback()
        db.close()


# ── SQL Server ecs_mobility ─────────────────────────────────────────────
# Disponible en dev / preprod contra la instancia real. En CI skippear.

@pytest.fixture
def db_app() -> Iterator[Session]:
    if not _mssql_reachable():
        pytest.skip("SQL Server ecs_mobility no arriba")
    db = SessionLocalApp()
    try:
        yield db
    finally:
        db.rollback()
        db.close()


# ── MES engine mockeado ─────────────────────────────────────────────────

@pytest.fixture
def engine_mes_mock(monkeypatch):
    """Engine MES con conexiones mockeadas. Los tests configuran el retorno
    de cursor.execute().fetchall() / mappings().all() según necesidad."""
    mock_engine = MagicMock()
    # Helper: mock_engine.connect().__enter__().execute(...).mappings().all()
    #   devuelve lo que el test programe.
    monkeypatch.setattr("nexo.data.engines.engine_mes", mock_engine)
    return mock_engine
```

### Patrón de rollback por test (D-08)

```python
# tests/data/test_nexo_repository.py
import pytest
from sqlalchemy import select
from nexo.data.repositories.nexo import AuditRepo
from nexo.data.models_nexo import NexoAuditLog

@pytest.mark.integration
def test_audit_repo_append_inserts_row(db_nexo):
    repo = AuditRepo(db_nexo)
    repo.append(user_id=None, ip="127.0.0.1", method="GET",
                path="/_test_repo", status=200, details_json=None)
    db_nexo.commit()  # caller controla commit

    rows = db_nexo.execute(
        select(NexoAuditLog).where(NexoAuditLog.path == "/_test_repo")
    ).scalars().all()
    assert len(rows) == 1
    # Cleanup gracias a rollback — pero cuidado: commit() arriba
    # consolida la fila. El fixture hace rollback de lo NO committed.
    # Si queremos limpiar en tests que hacen commit, usar DELETE explícito
    # al final o un mecanismo de cleanup con conexión owner (ver
    # tests/auth/test_audit_append_only.py para el patrón).
```

**Gotcha crítico:** `session.rollback()` no deshace filas ya comiteadas. El fixture `db_nexo` con `begin_nested()` crea un SAVEPOINT; si el código bajo test llama `commit()`, el savepoint se cierra pero la transacción outer puede seguir. Para tests que llaman `commit()`:

**Opción A (simple):** usar datos de test con path/email únicos + DELETE en teardown (owner engine, ver `tests/auth/test_audit_append_only.py:_owner_engine()`).

**Opción B (más limpia):** envolver la sesión en `connection.begin()` + pasar la connection al sessionmaker. SQLAlchemy rollbackea la outer transaction cuando el scope termina, incluso si el código interno commitea. Patrón "nested transactions" documentado en SQLAlchemy.

Para Phase 3 basta con Opción A por coherencia con el patrón existente (`test_audit_append_only.py`).

### Makefile target

```makefile
test-data: ## Arranca compose, corre tests tests/data/ y apaga compose
	docker compose up -d db
	@echo "Esperando Postgres healthy..."
	@until docker compose ps --format json db | grep -q '"Health":"healthy"'; do sleep 1; done
	docker compose exec -T web pytest tests/data/ -q -m "not integration or integration"
	# Nota: no apagamos compose aquí; dev normalmente lo quiere arriba.
	# Si quieres teardown, añadir `docker compose down` al final.
```

### Speed considerations

- `TestClient(app)` arranca el app completo (lifespan incluido). Si muchos tests lo hacen, considera `@pytest.fixture(scope="module")` como hace `test_rbac_smoke.py`.
- `lru_cache` del loader no causa problemas: los `.sql` no cambian durante un test run.
- Fixture `db_nexo` es function-scope por defecto (seguro); si CI es lento, evaluar session-scope + cleanup explícito, pero no optimizar hasta medir.

## MES Mocking Strategy

**Decisión D-08:** no hay SQL Server en CI, mock el engine.

### Patrón 1 — MagicMock + context manager

```python
# tests/data/test_mes_repository.py
from unittest.mock import MagicMock
from nexo.data.repositories.mes import MesRepository

def test_centro_mando_formatea_rows_correctamente():
    # 1. Montar mock: engine.connect() devuelve context manager cuyo
    #    .execute(stmt, params).mappings().all() devuelve lo fijado.
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.mappings.return_value.all.return_value = [
        {"ct": 101, "piezas_hoy": 42, "ultimo_evento": "14:30:00", "referencia": "REF-X"},
        {"ct": 102, "piezas_hoy": 5,  "ultimo_evento": "09:15:00", "referencia": ""},
    ]

    # 2. Instanciar repo con mock
    repo = MesRepository(engine=mock_engine)
    result = repo.centro_mando_fmesmic(ct_codes=[101, 102])

    # 3. Afirmar shape + parámetros enviados
    assert len(result) == 2
    assert result[0]["ct"] == 101
    # Aserción sobre los params reales enviados a .execute():
    args, kwargs = mock_conn.execute.call_args
    assert kwargs.get("codes") == ["101", "102"] or args[1].get("codes") == ["101", "102"]
```

### Patrón 2 — wrapper sobre `OEE.db.connector` (D-04)

Para los 4 métodos de `MesRepository` que delegan al connector (`extraer_datos_produccion`, `detectar_recursos`, `calcular_ciclos_reales`, `estado_maquina_live`), mockeamos a nivel de connector, no de engine:

```python
from unittest.mock import patch
from nexo.data.repositories.mes import MesRepository

@patch("nexo.data.repositories.mes.extraer_datos")  # import local en mes.py
def test_extraer_datos_produccion_delega_a_connector(mock_extraer):
    mock_extraer.return_value = [
        {"recurso": "luk1", "seccion": "LINEAS", "fecha": "2026-04-01",
         "h_ini": "06:00", "h_fin": "14:00", "tiempo": 480, "proceso": "Producción",
         "incidencia": "", "cantidad": 100, "malas": 2, "recuperadas": 0,
         "referencia": "REF-A"},
    ]
    repo = MesRepository(engine=MagicMock())
    from datetime import date
    rows = repo.extraer_datos_produccion(
        fecha_inicio=date(2026,4,1), fecha_fin=date(2026,4,1), recursos=["luk1"]
    )
    assert len(rows) == 1
    mock_extraer.assert_called_once()
```

**Ventaja del patrón 2:** no necesitamos reimplementar el mock de pyodbc cursor que hace el connector. Testeamos que `MesRepository` delega bien, no que la lógica MES funcione (eso es responsabilidad de `OEE/db/connector.py`, que no cambia).

### Cuando usar cada uno

| Método del repo | Patrón mock |
|-----------------|-------------|
| `extraer_datos_produccion` | Patrón 2 (delega a `OEE.db.connector.extraer_datos`) |
| `detectar_recursos` | Patrón 2 (delega) |
| `calcular_ciclos_reales` | Patrón 2 (delega) |
| `estado_maquina_live` | Patrón 2 (delega) |
| `consulta_readonly` | Patrón 1 (engine + `.execute(text(sql))`) |
| `centro_mando_fmesmic` (o nombre equivalente) | Patrón 1 (SQL carga desde loader + `bindparam(expanding=True)`) |
| Queries de `luk4`, `capacidad`, `operarios` (si se mueven al repo) | Patrón 1 |

### Grabación de respuestas (opcional, recomendado para queries complejas)

Para la query T3 cruza-medianoche y para `calcular_ciclos_reales` (que devuelve estructuras complejas con histograma + por_dia), considera grabar una vez la respuesta real y serializarla a JSON en `tests/data/fixtures/mes_*.json`. Los tests hacen `load_fixture("mes_extraer_datos_luk1_2026-04-01.json")` y hacen assert contra el contrato. Si el shape cambia en el futuro, un único sitio para actualizar.

## Wrapping OEE/db/connector.py Safely

**Problema:** `OEE/db/connector.py` lee `data/db_config.json` directamente (líneas 84-102 de `connector.py`) y construye su propia connection string. El nuevo `MesRepository` tiene que convivir con ese flujo sin duplicar lógica.

### Estrategia — wrapper sin reimportar credenciales

```python
# nexo/data/repositories/mes.py
from __future__ import annotations
from datetime import date
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

# Imports del connector legacy — UN SOLO LUGAR los concentra.
# El connector sigue leyendo data/db_config.json internamente; no lo tocamos.
from OEE.db.connector import (
    extraer_datos as _legacy_extraer_datos,
    detectar_recursos as _legacy_detectar_recursos,
    calcular_ciclos_reales as _legacy_calcular_ciclos_reales,
    estado_maquina_live as _legacy_estado_maquina_live,
)

# get_config del servicio existente (lee .env + db_config.json).
# Durante Phase 3 mantenemos la delegación al servicio viejo para no
# duplicar lectura de config. Phase 4+ centraliza en settings.
from api.services.db import get_config as _get_mes_config

from nexo.data.sql.loader import load_sql


class MesRepository:
    """Acceso a dbizaro (MES). Read-only de facto.

    - engine_mes recibido en __init__ (inyectado vía Depends(get_engine_mes)).
    - Los 4 métodos de alto nivel delegan al connector legacy, que
      mantiene las 847 LOC de lógica de pipeline OEE sin cambios.
    - consulta_readonly + queries directas (centro_mando_fmesmic, etc.)
      usan el engine nuevo con .sql versionados.
    """

    def __init__(self, engine: Engine):
        self._engine = engine

    # ── Delgados (D-04): delegan al connector ──────────────────────────

    def extraer_datos_produccion(
        self, fecha_inicio: date, fecha_fin: date, recursos: list[str] | None = None
    ) -> list[dict]:
        """Thin wrapper sobre OEE.db.connector.extraer_datos.

        Firma homogénea para el resto de Nexo. No cambia el connector.
        """
        cfg = _get_mes_config()
        if recursos:
            cfg = {**cfg, "recursos": [
                r for r in cfg.get("recursos", [])
                if r.get("nombre") in recursos
            ]}
        return _legacy_extraer_datos(cfg, fecha_inicio, fecha_fin)

    def detectar_recursos(self) -> list[dict]:
        return _legacy_detectar_recursos(_get_mes_config())

    def calcular_ciclos_reales(
        self, centro_trabajo: int, dias_atras: int = 30
    ) -> tuple[list[dict], str]:
        return _legacy_calcular_ciclos_reales(_get_mes_config(), centro_trabajo, dias_atras)

    def estado_maquina_live(
        self, centro_trabajo: int, umbral_activo_seg: int = 600
    ) -> dict:
        return _legacy_estado_maquina_live(_get_mes_config(), centro_trabajo, umbral_activo_seg)

    # ── Nuevos (usan engine_mes + .sql loader) ─────────────────────────

    def consulta_readonly(self, sql: str, database: str = "dbizaro") -> dict:
        """Ejecuta SQL ya validada por el router bbdd (D-05).

        engine_mes apunta a DATABASE=dbizaro; si database != dbizaro
        construimos engine one-shot — en Mark-III dejamos solo dbizaro
        (consistent con uso real del explorer).
        """
        with self._engine.connect() as conn:
            cursor = conn.execute(text(sql))
            columns = list(cursor.keys())
            rows = [list(r) for r in cursor.fetchall()]
        return {"columns": columns, "rows": rows, "n_rows": len(rows)}

    def centro_mando_fmesmic(self, ct_codes: list[int]) -> list[dict]:
        if not ct_codes:
            return []
        stmt = text(load_sql("mes/centro_mando_fmesmic")).bindparams(
            bindparam("codes", expanding=True)
        )
        with self._engine.connect() as conn:
            rows = conn.execute(stmt, {"codes": [str(c) for c in ct_codes]}).mappings().all()
            return [dict(r) for r in rows]
```

### Import cycles — ¿hay riesgo?

**Orden de imports:**
- `nexo/data/engines.py` → importa `api.config`. No importa nada de `nexo.data.*`. Seguro.
- `nexo/data/repositories/mes.py` → importa `nexo.data.sql.loader` + `OEE.db.connector` + `api.services.db`. No importa de `nexo.data.repositories` ni de `nexo.services`. Seguro.
- `nexo/services/auth.py` → importa `nexo.data.repositories.nexo` (en 03-03). No hay ciclo porque `repositories/nexo.py` no importa `services/auth.py`.
- `api/services/pipeline.py` → cambiará de `OEE.db.connector` a `nexo.data.repositories.mes.MesRepository`. No es ciclo (pipeline no es importado por mes).

**Shim crítico:** mientras `nexo/services/auth.py` siga importando de `nexo.db.engine`, ese módulo debe re-exportar:
```python
# nexo/db/engine.py — SHIM hasta 03-03
from nexo.data.engines import engine_nexo, SessionLocalNexo  # noqa: F401
```
En 03-03, cuando `auth.py` migre a `nexo.data.engines`, el shim se elimina.

### ¿Por qué NO reescribir `OEE/db/connector.py` en Phase 3?

1. **Success criterion #5** (PDFs idénticos) se convierte en problema abierto si se reescribe: la superficie de bug aumenta.
2. 847 LOC de lógica T-SQL delicada (filtros T3, cross-database joins, KDE mode, IQR filter) que funcionan hoy.
3. `CLAUDE.md` lo documenta explícitamente: "No refactorizar los 4 módulos de `OEE/`".
4. Phase 3 ya carga suficiente (3 engines + 10 repos + 8 routers + schema_guard + tests). Ampliar scope es receta para stall.

## DATA-09: Killing 3-part Names (centro_mando case study)

### El problema concreto

`api/routers/centro_mando.py` línea 56-64 (ya leído):
```python
FROM dbizaro.admuser.fmesmic m2   -- 3-part name: database.schema.table
WHERE RTRIM(m2.mi020) = RTRIM(m.mi020)
  ...
FROM dbizaro.admuser.fmesmic m    -- 3-part name outer
```

El engine usado es `api.database.engine` con `DATABASE=ecs_mobility` en la connection string. Por eso necesita 3-part names para cross-database SELECT — está literalmente saltando de `ecs_mobility` a `dbizaro`.

### La solución post DATA-09

`engine_mes` con `DATABASE=dbizaro` en connection string:
```python
# nexo/data/engines.py
engine_mes = create_engine(
    "mssql+pyodbc://USER:PWD@192.168.0.4:1433/dbizaro?driver=...",
    # ...
)
```

Al ejecutar contra `engine_mes`, SQL Server interpreta `admuser.fmesmic` como `dbizaro.admuser.fmesmic` implícitamente:
```sql
-- nexo/data/sql/mes/centro_mando_fmesmic.sql (post DATA-09)
FROM admuser.fmesmic m2   -- 2-part name: schema.table
```

### Gotchas con SQL Server y cross-database

[CITED: learn.microsoft.com/en-us/sql/t-sql/functions/servername-transact-sql] SQL Server resuelve nombres no calificados contra la DB del contexto actual. Con DATABASE=dbizaro en la connection string, todas las queries están en contexto dbizaro por defecto.

**Gotcha 1: USE statements.** No los emitimos — connection pool SQLAlchemy reutiliza conexiones, y un `USE` sticky entre queries contamina. Confiar en la DATABASE del connection string.

**Gotcha 2: GETDATE() y tempdb.** Funciones built-in siguen disponibles en cualquier contexto de DB. No hay cambio.

**Gotcha 3: Permisos del usuario.** El login (hoy `sa` o `dbizaro`, según qué credencial exacta uses) tiene que tener SELECT en `dbizaro.admuser.*`. En prod lo tiene (el código actual ya ejecuta ahí con 3-part names); no es regresión.

**Gotcha 4: ejecutar la query post-refactor contra la MISMA DB no contra master.** Confirmar en el smoke test de 03-02 que `SELECT DB_NAME()` devuelve `dbizaro` desde `engine_mes`.

### Smoke test de DATA-09

Tras el refactor de `centro_mando`:
```python
# tests/data/test_mes_engine_context.py
import pytest
from sqlalchemy import text
from nexo.data.engines import engine_mes

@pytest.mark.integration
def test_engine_mes_default_database_is_dbizaro():
    with engine_mes.connect() as conn:
        db = conn.execute(text("SELECT DB_NAME()")).scalar()
    assert db == "dbizaro", f"engine_mes debe apuntar a dbizaro, apunta a {db}"

@pytest.mark.integration
def test_centro_mando_query_sin_three_part_names_devuelve_rows():
    # La query real con 2-part names contra engine_mes
    from nexo.data.sql.loader import load_sql
    sql = load_sql("mes/centro_mando_fmesmic")
    assert "dbizaro." not in sql.lower(), "La SQL sigue usando 3-part names"
    # Ejecutar una consulta mínima (TOP 1) para confirmar que el engine la resuelve
    from sqlalchemy import bindparam
    stmt = text(sql).bindparams(bindparam("codes", expanding=True))
    with engine_mes.connect() as conn:
        # Pasamos un CT cualquiera existente; basta con que la query no falle
        result = conn.execute(stmt, {"codes": ["101"]}).fetchall()
    # No afirmamos count; solo que no explote
```

### Auditoría de 3-part names en el codebase

Grep previo a 03-02 debe listar:
```bash
grep -rn "dbizaro\." api/routers/ api/services/ --include="*.py"
grep -rn "dbizaro\." OEE/ --include="*.py"
```

Cada match en `api/` DEBE desaparecer tras 03-02. Matches en `OEE/db/connector.py` quedan (el connector no cambia por D-04), pero las queries ahí ya usan 2-part names (`admuser.fmesdtc`, no `dbizaro.admuser.fmesdtc`) — verificado leyendo `connector.py`.

## Runtime State Inventory

(Phase 3 NO es un rename/refactor/migración de strings. Es refactor estructural con creación de módulos nuevos y reorganización de imports. Este inventario no aplica en el sentido estricto de D-2.5 del protocolo, pero documentamos aquí lo que SÍ tiene estado runtime persistente para que el planner no lo olvide.)

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | Postgres `nexo.*` ya inicializado (Phase 2). `ecs_mobility.*` ya inicializado. `dbizaro.admuser.*` read-only. | Ninguna — schema_guard sólo valida, no muta (excepto NEXO_AUTO_MIGRATE=true). |
| Live service config | `data/db_config.json` contiene credenciales MES leídas por `OEE/db/connector.py`. | Ninguna — el connector no se toca. El MesRepository re-lee vía `api.services.db.get_config()` existente. |
| OS-registered state | Nada. Nexo no registra tasks OS-level. | Ninguna. |
| Secrets/env vars | `.env` ya tiene `NEXO_MES_*`, `NEXO_APP_*`, `NEXO_PG_*`, `NEXO_PG_APP_*`. Phase 3 no añade nuevas env vars (excepto `NEXO_AUTO_MIGRATE` opcional). | Añadir `NEXO_AUTO_MIGRATE` a `.env.example` con valor `false` por default y un comentario de "solo dev". |
| Build artifacts / installed packages | `Dockerfile` copia `nexo/` y `scripts/` ya. La nueva estructura `nexo/data/*` se copia automáticamente porque `nexo/` está en el COPY. | Ninguna. Verificar en smoke post-03-01: `docker compose exec web python -c "from nexo.data.engines import engine_nexo"` no falla. |

**Nada crítico oculto:** Phase 3 NO renombra strings operacionales (ni user_ids de Mem0, ni workflow names, ni SOPS keys).

## Common Pitfalls

### Pitfall 1: Statement caching con `bindparam(expanding=True)`

**What goes wrong:** Sin `expanding=True`, pasar una lista como parámetro falla (SQLAlchemy envía la lista literal al DBAPI que no sabe expandir).

**Why it happens:** Para poder cachear statements, SQLAlchemy compila la query una vez. Con IN(:codes) y `codes=[1,2,3]`, la forma exacta del SQL depende del tamaño de la lista. `expanding=True` le dice "renderiza esto en el último momento con N placeholders según el tamaño".

**How to avoid:** SIEMPRE `bindparam("codes", expanding=True)` para IN-clauses dinámicos. Es el único patrón 2.0-canónico.

**Warning signs:** Error `CompileError: Bind parameter 'codes' ... expected a non-list value` o ejecución que devuelve 0 filas silenciosa (peor).

### Pitfall 2: pandas 3.0 + SQLAlchemy 2.0 + pyodbc raw connection

**What goes wrong:** `OEE/db/connector.extraer_datos` hace `pd.read_sql(sql, conn)` donde `conn` es un `pyodbc.Connection` crudo. pandas 2.2+ emite `UserWarning: pandas only supports SQLAlchemy connectable (engine/connection) or database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 objects are not tested. Please consider using SQLAlchemy.` Con pandas 3.0.2 el warning sigue apareciendo pero la ejecución no falla [VERIFIED: pandas 3.0.2 docs del parameter `con` acepta "ADBC/SQLAlchemy/str/sqlite3"; pyodbc cae en "other DBAPI2" → warning pero funcional].

**Why it happens:** pandas prefiere adapters conocidos. pyodbc no está en la whitelist oficial pero funciona en la práctica (DBAPI2 compliant).

**How to avoid:** NO TOCAR. El connector no se reescribe (D-04). El warning es ruido cosmético; silenciar con `warnings.filterwarnings` solo si molesta los logs del CI. Alternativa: envolver `conn` en `engine_mes.connect()` — pero cambia semántica del pool y el connector maneja timeouts propios (15s). Diferido a Mark-IV.

**Warning signs:** Warning en logs del pipeline; si pandas decide en una versión futura elevar a error, romperemos. Monitorizar changelog de pandas.

### Pitfall 3: `init_db()` existente intenta importar `OEE.db.connector` durante lifespan

Ver `api/database.py:_import_recursos_json`:
```python
from OEE.db.connector import load_config as _load_cfg
cfg = _load_cfg()
```

**What goes wrong:** Si `schema_guard.verify(engine_nexo)` falla y aborta el lifespan, `init_db()` nunca corre y los recursos no se cargan. No es pérdida de datos (ya están en BD), pero el efecto visible es: la primera vez que alguien levanta sin Postgres, no entiende por qué falta el init.

**How to avoid:** el mensaje de error de `schema_guard` debe ser claro: "ejecuta `make nexo-init`". Ya contemplado en la implementación propuesta.

### Pitfall 4: `docker compose restart web` no recarga `.env`

**What goes wrong:** Si el planner añade `NEXO_AUTO_MIGRATE=true` a `.env` y hace `docker compose restart web`, el container no lo ve.

**Why it happens:** `restart` mantiene el container original con sus env vars iniciales. Documentado en SUMMARY 02-04.

**How to avoid:** Siempre `docker compose up -d --force-recreate web` cuando se toca `.env`. Documentar en el plan de 03-01.

### Pitfall 5: Shim `nexo/db/engine.py` re-export rompe si alguien importa `SessionLocalNexo` como alias

Durante 03-01 dejamos:
```python
# nexo/db/engine.py — shim
from nexo.data.engines import engine_nexo, SessionLocalNexo
```

**What goes wrong:** Si algún lugar hace `from nexo.db.engine import SessionLocalNexo as Foo` y luego monkeypatcheamos `nexo.data.engines.SessionLocalNexo` en tests, el monkeypatch no afecta al alias.

**How to avoid:** No alias a nivel módulo. Importar directamente en cada sitio.

### Pitfall 6: FastAPI `TestClient(app)` ejecuta lifespan completo

**What goes wrong:** Cada test que usa `TestClient(app)` arranca schema_guard + init_db. Si Postgres no está, el lifespan falla (o warna) — los tests unitarios sin DB fallan ruidosamente.

**How to avoid:** Tests de repo con `pytestmark = [pytest.mark.integration, skipif(not _postgres_reachable())]` (patrón establecido en `tests/auth/test_rbac_smoke.py`). Tests puros de lógica DTO/loader no importan `app`, no necesitan lifespan.

### Pitfall 7: `AuditRepo.append` sin commit y nexo_app sin UPDATE

**What goes wrong:** si el caller hace `db.add(row)` y no hace `db.commit()`, la fila no se persiste. El middleware de audit actual (`api/middleware/audit.py`) hace commit implícito (hay que verificar al refactorizar).

**How to avoid:** documentar en el docstring de `AuditRepo.append`: "caller DEBE comitear". Smoke test que pase por el middleware completo y verifique que la fila está en BD tras request.

### Pitfall 8: `lru_cache` del loader mantiene contenido viejo entre test runs en el mismo proceso

**What goes wrong:** Un test modifica un `.sql` en disco (ej: fixture que genera SQL dinámico) y espera recargar.

**How to avoid:** Tests NO modifican archivos `.sql` en disco. Si algún test muy específico lo necesita, llamar `load_sql.cache_clear()` en el setup. Los `.sql` son de solo-lectura en Mark-III.

## Code Examples

Todos los ejemplos ya en §Architecture Patterns y §Implementation Patterns Library abajo.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `orm_mode = True` en config | `model_config = ConfigDict(from_attributes=True)` | Pydantic 2.0 (2023) | [CITED: docs.pydantic.dev/latest/migration/] Sintaxis explícita. |
| `allow_mutation = False` | `model_config = ConfigDict(frozen=True)` | Pydantic 2.0 | Mismo efecto, nombre corregido. |
| `@validator(..., pre=True)` | `@field_validator(..., mode="before")` | Pydantic 2.0 | [CITED: docs.pydantic.dev] Nueva API. |
| `text("SELECT ... IN ({})".format(",".join(["?"]*n)))` con string fmt | `text(...).bindparams(bindparam("codes", expanding=True))` | SQLAlchemy 1.4+ | [CITED: docs.sqlalchemy.org/en/20/faq/sqlexpressions.html] Statement cacheable + seguro. |
| `@app.on_event("startup")` | `@asynccontextmanager` + `lifespan` param | FastAPI 0.100+ (2023) | [CITED: fastapi.tiangolo.com/advanced/events/] `on_event` deprecated. |
| `inspect(engine).has_table("users", schema="nexo")` | Sigue válido en 2.0 | — | API estable. Alternativa: `.get_table_names(schema="nexo")`. |

**Deprecated/outdated en este codebase:**
- `datetime.utcnow()` aparece en `api/database.py` (7 ocurrencias) — deprecated en Python 3.12 (`nexo/db/models.py` ya migró a `datetime.now(timezone.utc)` en Phase 2). Oportunidad de cleanup en 03-03 al mover ORM de `api/database.py` → `nexo/data/models_app.py`. No es bloqueante.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | pandas 3.0.2 + pyodbc raw connection sigue funcionando con solo UserWarning en el pipeline | Pitfall 2 | Si pandas 3.x ya elevó a error, `OEE.db.connector.extraer_datos` fallará — bloquea pipeline. Mitigación: smoke test pre-refactor (basta correr `extract_2025.py` u otro script que use el connector) antes de arrancar 03-02. |
| A2 | El hash guard de PDFs (success #5) se puede reproducir con la misma fecha histórica generando bytes idénticos | D-04 wrapper | matplotlib puede incluir timestamp en metadata del PDF → hash varía entre runs del mismo día. Fallback: comparar count de páginas + size en bytes ± tolerancia. |
| A3 | El rol `nexo_app` (creado en 02-04) tiene INSERT en `nexo.audit_log`, y esto NO cambia por el refactor | §Repository + DTO Patterns AuditRepo | Si al refactorizar a `AuditRepo` alguien introduce un UPDATE accidental (p.ej. upsert), los tests del gate IDENT-06 de 02-04 deben detectarlo. Correr `tests/auth/test_audit_append_only.py` post-refactor. |
| A4 | `engine_app` re-exportado desde `api/database.py` no crea conflicto de imports en lifespan | §Architecture Patterns Pattern 3 | Si `api/database.py` acaba importando de `nexo/data/engines.py` (vía `init_db()` o similar), ciclo. Verificación: grep `from nexo.data` en `api/database.py` tras refactor. |
| A5 | `from api.services.db import get_config` en `MesRepository` no crea ciclo | §Wrapping OEE/db/connector.py | `api.services.db` importa `OEE.db.connector` (nivel legacy), no `nexo.data.repositories.mes`. No hay ciclo. Verificado manualmente (api/services/db.py líneas 10-20). |
| A6 | `schema_guard` con `get_table_names(schema="nexo")` funciona con el rol `nexo_app` (no requiere OWNER) | §Lifespan Implementation | psycopg2 inspector lee de `information_schema`, que es visible al rol app (tiene GRANT USAGE + SELECT). Verificación: smoke test con engine_nexo real. |
| A7 | `TestClient(app)` corre lifespan completo SIEMPRE (no se puede skippear) | Pitfall 6 | FastAPI >=0.100 corre lifespan en TestClient. Si algún test unitario puro no quiere lifespan, tiene que NO usar TestClient (usar repos directamente con db_nexo fixture). Documentado en plan. |

**If this table is non-empty:** Sí lo está (7 assumptions). El plan debe validar A1 (smoke pre-refactor) y A6 (smoke schema_guard con rol nexo_app) como tareas explícitas, no dejarlos implícitos.

## Open Questions

1. **Cómo gestionar `consulta_readonly` cuando el explorer `/bbdd` cambia de `database` dinámicamente.**
   - What we know: Hoy `bbdd.py` construye conn strings on-the-fly para cada DB distinta (master, dbizaro, ecs_mobility).
   - What's unclear: `MesRepository.consulta_readonly(sql, database)` — ¿un engine por DB? ¿Un engine_mes fijo y ignorar el param database?
   - Recommendation: en Mark-III, `consulta_readonly` usa `engine_mes` (fijo a dbizaro); las operaciones de metadata del explorer (`list_databases`, `preview` a otras DBs) quedan inline en `bbdd.py` con sus engines temporales (D-05 ya lo deja así). Documentar en el docstring que `consulta_readonly` solo cubre dbizaro.

2. **¿El wrapper `MesRepository` debe exponer la config MES para tests o esconderla?**
   - What we know: `MesRepository.extraer_datos_produccion` lee `_get_mes_config()` cada vez — ineficiente pero aislado.
   - What's unclear: ¿inyectar `config` como atributo del repo para facilitar mocking?
   - Recommendation: en Mark-III mantener `_get_mes_config()` como call interno; los tests mockean con `@patch("nexo.data.repositories.mes._legacy_extraer_datos")` (Patrón 2). Si en Mark-IV se rediseña la config, inyectarla entonces.

3. **¿El schema_guard debe validar también `ecs_mobility.cfg.recursos` y `ecs_mobility.oee.*`?**
   - What we know: D-07 dice "solo schema nexo". Arquitectura dice `cfg.*` está fuera de control Nexo (Power BI los usa).
   - What's unclear: si alguien borra `cfg.recursos` accidentalmente, la app arranca pero falla la primera request a `/api/recursos`.
   - Recommendation: mantener D-07 (solo nexo). Si en runtime falla una query a `cfg.*`, el error es claro (SQLAlchemy `NoSuchTableError`). No worth adding validation overhead ni alcance cruzado en Mark-III.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Postgres | engine_nexo, tests data (NEXO) | ✓ | 16-alpine (container `db`) | `make nexo-init` required; compose up -d db |
| SQL Server `ecs_mobility` | engine_app, tests data (APP) en dev | ✓ (según .env) | - | Tests APP saltan en CI (skipif no reachable) |
| SQL Server `dbizaro` (MES) | engine_mes, queries directas | ✓ (según .env) | - | Tests MES usan mocks SIEMPRE |
| Docker compose | `make test-data`, `make up` | ✓ | - | - |
| pytest, httpx | Tests | ✓ | 8.3.4, 0.28.1 | - |
| ODBC Driver 18 | pyodbc → engine_mes/app | ✓ en container web | - | Fallback ODBC 17 ya en `detectar_driver()` |
| psycopg2 | engine_nexo | ✓ | 2.9.11 | - |

**Missing dependencies with no fallback:** Ninguna para el alcance Phase 3. Todo lo necesario ya está en el entorno del operador (compose arriba, .env con credenciales reales MES + APP).

**Missing dependencies with fallback:** CI no tiene SQL Server — tests MES siempre mockeados (D-08); tests APP con skipif.

## Validation Architecture (Nyquist Dimension 8)

**Framework confirmado:** pytest 8.3.4 + pytest-cov 6.0.0 + httpx 0.28.1 (TestClient). `tests/conftest.py` registra marker `integration`. Sin `pytest.ini` ni `pyproject.toml` con config de pytest — defaults.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.3.4 |
| Config file | ninguno (usa defaults + `tests/conftest.py`) |
| Quick run command | `docker compose exec web pytest tests/data/ -q -m "not integration"` |
| Full suite command | `docker compose exec web pytest tests/ -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | 3 engines con pool configurado correcto | unit | `pytest tests/data/test_engines.py -x` | ❌ Wave 0 |
| DATA-01 | `engine_mes` apunta a `dbizaro` (DB_NAME()) | integration | `pytest tests/data/test_engines.py::test_engine_mes_database_is_dbizaro -x` | ❌ Wave 0 |
| DATA-02 | `MesRepository` con 5 métodos (firma + wrap delegation) | unit | `pytest tests/data/test_mes_repository.py -x` | ❌ Wave 0 |
| DATA-02 | `extraer_datos_produccion` delega a `OEE.db.connector.extraer_datos` | unit (mock) | `pytest tests/data/test_mes_repository.py::test_extraer_datos_delega -x` | ❌ Wave 0 |
| DATA-03 | 6 repos APP con contrato session-inyectada | integration | `pytest tests/data/test_app_repository.py -x` | ❌ Wave 0 |
| DATA-04 | 3 repos NEXO (`UserRepo`, `RoleRepo`, `AuditRepo`) | integration | `pytest tests/data/test_nexo_repository.py -x` | ❌ Wave 0 |
| DATA-04 | `AuditRepo.append` no comitea internamente | integration | `pytest tests/data/test_nexo_repository.py::test_audit_repo_no_commits -x` | ❌ Wave 0 |
| DATA-05 | Loader carga `.sql` con `lru_cache` | unit | `pytest tests/data/test_sql_loader.py -x` | ❌ Wave 0 |
| DATA-05 | Comentario T3 presente en `extraer_datos_produccion.sql` | unit | `pytest tests/data/test_sql_loader.py::test_t3_comment_preserved -x` | ❌ Wave 0 |
| DATA-06 | `schema_guard.verify` falla con tabla faltante | integration | `pytest tests/data/test_schema_guard.py::test_verify_raises_missing -x` | ❌ Wave 0 |
| DATA-06 | `NEXO_AUTO_MIGRATE=true` auto-crea | integration | `pytest tests/data/test_schema_guard.py::test_auto_migrate_creates -x` | ❌ Wave 0 |
| DATA-07 | `grep -rn "import pyodbc" api/routers/` = 0 | smoke | `pytest tests/data/test_no_raw_pyodbc_in_routers.py -x` (meta-test) | ❌ Wave 0 |
| DATA-07 | `grep -rn "dbizaro\\." api/` = 0 | smoke | `pytest tests/data/test_no_three_part_names.py -x` | ❌ Wave 0 |
| DATA-07 | Los 8 routers responden 200 bajo auth válida (smoke) | integration | `pytest tests/data/test_routers_smoke.py -x` | ❌ Wave 0 |
| DATA-08 | DTOs son frozen y fail con mutation | unit | `pytest tests/data/test_dto_immutable.py -x` | ❌ Wave 0 |
| DATA-09 | Query `centro_mando` sin `dbizaro.` | integration | `pytest tests/data/test_mes_engine_context.py -x` | ❌ Wave 0 |
| DATA-10 | Fixtures `db_nexo`, `db_app`, `engine_mes_mock` funcionan | integration | `pytest tests/data/test_fixtures.py -x` | ❌ Wave 0 |
| DATA-11 | `engine_mes` pool = pool_recycle 3600, pool_pre_ping True | unit | `pytest tests/data/test_engines.py::test_mes_pool_config -x` | ❌ Wave 0 |

**Success criterion #5 (PDFs idénticos Mark-II):** test especial de regresión, no unitario.

| Behavior | Test Type | Automated Command | Location |
|----------|-----------|-------------------|----------|
| PDF generation pre/post refactor → hash o count match | regression gate | `python scripts/pdf_regression_check.py` (nuevo script) | ❌ Wave 0 (script nuevo en 03-02) |

### Sampling Rate

- **Per task commit:** Quick run — solo unit tests sin integration marker. `pytest tests/data/ -m "not integration" -q`.
- **Per wave merge:** Full suite de la fase. `pytest tests/data/ tests/auth/ -q` (auth se mantiene verde = no regresión del gate IDENT-06).
- **Phase gate:** Full suite + `scripts/pdf_regression_check.py` antes de `/gsd-verify-work`.

**Muestreo entre routers (D-06):** NO necesario sacrificar cobertura. Cada router refactorizado corre su propio smoke HTTP en tests/data/test_routers_smoke.py. No usar "test 1 of 5 en lugar de todos" — la superficie es pequeña y TestClient es rápido. Pero SÍ usar:
- Parametrización de pytest para los 5 routers de 03-02 (mismo test shape con ct de ruta distinto).
- Un único fixture `client` (`scope="module"`) para evitar lifespan overhead.

**Gate de "PDFs idénticos" (success #5):**
1. Antes de arrancar 03-02: commit actual (hash conocido) + `python scripts/gen_pdf_reference.py --fecha=2026-03-15` (o fecha estable con datos) → guarda `tests/data/reference/pipeline_2026-03-15.pdf` + `tests/data/reference/pipeline_2026-03-15.sha256`.
2. Tras 03-02 completo: `python scripts/pdf_regression_check.py --fecha=2026-03-15` → regenera PDF con el MesRepository wrapper, compara hash.
3. Si hash no coincide (p.ej. matplotlib metadata cambia): fallback a comparación de páginas + bytes ± 5%.
4. Test CI futuro: `pytest tests/data/test_pipeline_regression.py -m integration` marcado slow.

### Wave 0 Gaps

- [ ] `tests/data/__init__.py` — empty marker
- [ ] `tests/data/conftest.py` — fixtures `db_nexo`, `db_app`, `engine_mes_mock` + skipifs
- [ ] `tests/data/test_engines.py` — DATA-01 + DATA-11
- [ ] `tests/data/test_sql_loader.py` — DATA-05
- [ ] `tests/data/test_schema_guard.py` — DATA-06
- [ ] `tests/data/test_dto_immutable.py` — DATA-08
- [ ] `tests/data/test_mes_repository.py` — DATA-02 + DATA-09
- [ ] `tests/data/test_mes_engine_context.py` — DATA-09 smoke
- [ ] `tests/data/test_app_repository.py` — DATA-03
- [ ] `tests/data/test_nexo_repository.py` — DATA-04
- [ ] `tests/data/test_routers_smoke.py` — DATA-07 por los 8 routers
- [ ] `tests/data/test_no_raw_pyodbc_in_routers.py` — meta-test (grep-based)
- [ ] `tests/data/test_no_three_part_names.py` — meta-test (grep-based)
- [ ] `scripts/gen_pdf_reference.py` — genera PDF referencia para hash guard
- [ ] `scripts/pdf_regression_check.py` — compara post-refactor
- [ ] `tests/data/reference/` — directorio para fixtures binarias (gitignored probablemente)
- [ ] `Makefile` target `test-data` — arranca compose db + pytest tests/data/
- [ ] `.env.example` entry `NEXO_AUTO_MIGRATE=false` con comentario de "solo dev"

**Framework install:** nada nuevo. pytest, httpx, pytest-cov ya en `requirements-dev.txt`.

**Nota meta-tests:** `test_no_raw_pyodbc_in_routers.py` y `test_no_three_part_names.py` son chequeos grep sobre filesystem. No son idiomáticos pero son la forma más barata de gate para DATA-07/DATA-09. Ejemplo:
```python
# tests/data/test_no_raw_pyodbc_in_routers.py
import pathlib
import pytest

ROUTERS_DIR = pathlib.Path(__file__).resolve().parents[2] / "api" / "routers"

def test_ningun_router_importa_pyodbc_directamente():
    infringes = []
    for py in ROUTERS_DIR.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        if "import pyodbc" in text:
            infringes.append(py.name)
    assert not infringes, f"Routers con pyodbc directo: {infringes}"
```

## Landmines & Pitfalls (codebase-specific)

1. **`OEE/db/connector.py` lee `data/db_config.json` con path absoluto al arrancar.** Si `MesRepository` corre en un entorno sin ese archivo (test en CI), el connector falla tarde. Mitigación: mocks Patrón 2 en tests; en prod/dev el archivo existe siempre.

2. **`api/database.py:init_db()` importa `OEE.db.connector.load_config` dentro de la función.** Import lazy — si `OEE/db/connector.py` se rompe, no lo notamos hasta el primer `init_db()`. Smoke test post-refactor debe correr `init_db()` explícitamente o arrancar el app completo.

3. **El engine SQL Server usa `_mssql_creator` con pyodbc directo, NO string URL de SQLAlchemy.** Ver `api/database.py` líneas 19-31. Cualquier código nuevo que use `engine_app` sigue el mismo approach (está pensado así por fiabilidad con ODBC 18). Si tratas de `create_engine("mssql+pyodbc://...")` con string URL para engine_app, puede romper diferente.

4. **`SessionLocalNexo` expira objetos en commit (no por default).** `expire_on_commit=False` ya configurado en `nexo/db/engine.py`. Repos que comiteen y luego accedan a atributos seguirán funcionando. Mantener en los nuevos `SessionLocalApp` / `SessionLocalNexo` de `nexo/data/engines.py`.

5. **`require_permission("xxx")` en routers viene de `nexo.services.auth` importado directamente, no inyectado.** Cuando 03-03 migra `auth.py` a `UserRepo`, tener cuidado de que los imports en routers no rompan (siguen siendo `from nexo.services.auth import require_permission`).

6. **Tests del gate IDENT-06 (`test_audit_append_only.py`) usan `engine_nexo` importado de `nexo.db.engine`.** Cuando 03-01 mueva engine a `nexo.data.engines`, el shim `nexo/db/engine.py` los sigue satisfaciendo. En 03-03, cuando se elimine el shim, actualizar los imports de esos tests a `from nexo.data.engines import engine_nexo`.

7. **`docker compose restart web` no refresca `.env`.** Ya documentado en SUMMARY 02-04. Plan debe incluir nota al introducir `NEXO_AUTO_MIGRATE`: usar `--force-recreate`.

8. **`pandas==3.0.2` emite `UserWarning` con pyodbc raw connection.** Cosmético, no bloqueante. Si el equipo quiere silenciarlo, añadir `warnings.filterwarnings("ignore", message=".*DBAPI2.*")` en `OEE/db/connector.py` — pero esto altera el módulo que D-04 dice no tocar. Preferible: dejar el warning en Mark-III.

9. **`api/services/pipeline.py:_save_metrics_to_db` accede directamente a `MetricaOEE`, `ReferenciaStats`, `IncidenciaResumen` del módulo `api.database`.** Tras mover esos modelos a `nexo/data/models_app.py` en 03-03, actualizar imports. Hay ~15 puntos en `pipeline.py` que los tocan.

10. **`.env` con `NEXO_AUTO_MIGRATE` puede ser peligroso en prod si alguien lo activa sin pensar.** El warning del log (`"NO usar en producción"`) es la única barrera. Plan debe incluir mención clara en `.env.example` y en el target `make test-data` + `make nexo-init`.

11. **El orden de middlewares es LIFO.** Si 03-03 toca algo de auth, no mover el orden de `app.add_middleware(AuditMiddleware); app.add_middleware(AuthMiddleware)` en `api/main.py`. Gate IDENT-06 depende de ese orden.

12. **`settings.effective_pg_user` / `effective_pg_password`** usan `pg_app_*` preferente con fallback a `pg_*` owner. `nexo/data/engines.py` DEBE usar `effective_*`, no `pg_*` directamente — si no, la app arranca como owner (privilegios de UPDATE/DELETE en audit_log) y el gate IDENT-06 se rompe silencioso.

## Implementation Patterns Library

Referencias concretas que el planner puede citar directamente en las tareas.

### Ref 1: Definición del engine_mes (DATA-01, DATA-11)

```python
# Ubicación: nexo/data/engines.py
engine_mes = create_engine(
    _build_mes_dsn(),
    pool_pre_ping=True,       # DATA-11
    pool_recycle=3600,        # DATA-11
    pool_size=3, max_overflow=2,
    pool_timeout=15,          # DATA-11
    connect_args={"timeout": 15},  # pyodbc timeout
)
```

### Ref 2: Loader con lru_cache (DATA-05)

```python
# Ubicación: nexo/data/sql/loader.py
from functools import lru_cache
from importlib.resources import files

@lru_cache(maxsize=128)
def load_sql(name: str) -> str:
    if not name.endswith(".sql"):
        name = f"{name}.sql"
    ref = files("nexo.data.sql")
    for part in name.split("/"):
        ref = ref / part
    return ref.read_text(encoding="utf-8")
```

### Ref 3: schema_guard + lifespan (DATA-06)

```python
# Ubicación: nexo/data/schema_guard.py
CRITICAL_TABLES = ("users", "roles", "departments", "user_departments",
                   "permissions", "sessions", "login_attempts", "audit_log")

def verify(engine):
    insp = inspect(engine)
    existing = set(insp.get_table_names(schema="nexo"))
    missing = [t for t in CRITICAL_TABLES if t not in existing]
    if not missing: return
    if os.environ.get("NEXO_AUTO_MIGRATE", "").lower() in {"1","true","yes"}:
        NexoBase.metadata.create_all(bind=engine)
        return
    raise RuntimeError(f"Schema guard: faltan tablas nexo.* → {missing}. ...")
```

```python
# Ubicación: api/main.py (modificación del lifespan existente)
@asynccontextmanager
async def lifespan(app: FastAPI):
    schema_guard.verify(engine_nexo)   # NEW — antes de init_db
    init_db()                          # existing
    yield
```

### Ref 4: Repo DTO frozen (DATA-08)

```python
# Ubicación: nexo/data/dto/app.py
from pydantic import BaseModel, ConfigDict

class RecursoRow(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    id: int
    centro_trabajo: int
    nombre: str
    seccion: str = "GENERAL"
    activo: bool = True
```

### Ref 5: Repo con sesión inyectada (DATA-03, DATA-04)

```python
# Ubicación: nexo/data/repositories/app.py
class RecursoRepo:
    def __init__(self, db: Session):
        self._db = db

    def list_activos(self) -> list[RecursoRow]:
        rows = self._db.execute(
            select(Recurso).where(Recurso.activo.is_(True))
                           .order_by(Recurso.seccion, Recurso.nombre)
        ).scalars().all()
        return [RecursoRow.model_validate(r) for r in rows]
```

### Ref 6: Router post-refactor (DATA-07)

```python
# Ubicación: api/routers/centro_mando.py (post 03-02)
from fastapi import APIRouter, Depends
from api.deps import DbApp, EngineMes
from nexo.data.repositories.app import RecursoRepo
from nexo.data.repositories.mes import MesRepository
from nexo.services.auth import require_permission

router = APIRouter(
    prefix="/dashboard",
    dependencies=[Depends(require_permission("centro_mando:read"))],
)

@router.get("/summary")
def summary(db: DbApp, engine_mes: EngineMes):
    recurso_repo = RecursoRepo(db)
    mes_repo = MesRepository(engine=engine_mes)
    recursos = recurso_repo.list_activos()
    mic_data = mes_repo.centro_mando_fmesmic([r.centro_trabajo for r in recursos])
    # ... lógica de cacheo y render (preservada tal cual del actual)
```

### Ref 7: IN-clause con bindparam expanding (DATA-05)

```python
# Fragmento dentro de un método de MesRepository
from sqlalchemy import bindparam, text
from nexo.data.sql.loader import load_sql

def centro_mando_fmesmic(self, ct_codes: list[int]) -> list[dict]:
    stmt = text(load_sql("mes/centro_mando_fmesmic")).bindparams(
        bindparam("codes", expanding=True)
    )
    with self._engine.connect() as conn:
        return [dict(r) for r in conn.execute(
            stmt, {"codes": [str(c) for c in ct_codes]}
        ).mappings().all()]
```

[CITED: docs.sqlalchemy.org/en/20/faq/sqlexpressions.html] — `bindparam(name, expanding=True)` renderiza placeholders dinámicamente y permite statement caching simultáneo. Es el único patrón canónico 2.0 para IN(:list).

### Ref 8: Fixture Postgres real con rollback (DATA-10)

```python
# Ubicación: tests/data/conftest.py
@pytest.fixture
def db_nexo():
    if not _postgres_reachable():
        pytest.skip("Postgres no arriba")
    db = SessionLocalNexo()
    db.begin_nested()  # SAVEPOINT
    try:
        yield db
    finally:
        db.rollback()
        db.close()
```

### Ref 9: Mock MES engine (DATA-10)

```python
# Ubicación: tests/data/test_mes_repository.py
def test_consulta_readonly_devuelve_dict_shape(engine_mes_mock):
    engine_mes_mock.connect.return_value.__enter__.return_value.execute.return_value.keys.return_value = ["col1"]
    engine_mes_mock.connect.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [(42,)]

    from nexo.data.repositories.mes import MesRepository
    repo = MesRepository(engine=engine_mes_mock)
    result = repo.consulta_readonly("SELECT 42 AS col1", "dbizaro")
    assert result["columns"] == ["col1"]
    assert result["rows"] == [[42]]
```

### Ref 10: PDF regression guard (Success #5)

```python
# Ubicación: scripts/pdf_regression_check.py (nuevo, 03-02)
"""Compara PDF generado tras refactor con baseline pre-refactor.

Uso:
    python scripts/pdf_regression_check.py --fecha=2026-03-15

Exit codes:
    0 → OK (hash match)
    1 → hash distinto pero fallback (count + bytes) pasa → WARN
    2 → regresión real detectada
"""
import hashlib, sys, argparse
from pathlib import Path

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fecha", required=True)
    args = ap.parse_args()

    ref_dir = Path("tests/data/reference")
    baseline = ref_dir / f"pipeline_{args.fecha}.pdf"
    baseline_hash_file = ref_dir / f"pipeline_{args.fecha}.sha256"
    if not baseline.exists():
        sys.exit("No baseline — correr gen_pdf_reference.py primero")

    # Regenerar PDF con el código actual (post-refactor)
    from api.services.pipeline import run_pipeline
    from datetime import date
    fecha = date.fromisoformat(args.fecha)
    pdfs = []
    for msg in run_pipeline(fecha, fecha):
        if msg.startswith("DONE:"):
            pdfs = msg.split(":", 3)[3].split("|")

    # Localizar el PDF equivalente al baseline (mismo nombre)
    candidate = Path("informes") / pdfs[0]
    new_hash = sha256(candidate)
    ref_hash = baseline_hash_file.read_text().strip()

    if new_hash == ref_hash:
        print(f"OK — hashes match ({new_hash[:16]}...)")
        sys.exit(0)

    # Fallback: comparar bytes ± 5%
    baseline_size = baseline.stat().st_size
    candidate_size = candidate.stat().st_size
    diff_pct = abs(candidate_size - baseline_size) / baseline_size
    if diff_pct <= 0.05:
        print(f"WARN — hash distinto pero size within 5% ({diff_pct:.2%})")
        sys.exit(1)

    print(f"REGRESSION — size diff {diff_pct:.2%}, investigar")
    sys.exit(2)

if __name__ == "__main__":
    main()
```

## Project Constraints (from CLAUDE.md)

Extracto de directivas actionables de `/home/eeguskiza/analisis_datos/CLAUDE.md` que el planner DEBE cumplir:

- **`OEE/` folder no renamed en Phase 3** (diferido a Mark-IV). `D-04` wrapping pattern preserva esta regla.
- **`.planning/` es proyección, no editar a mano.** Este RESEARCH.md va en `.planning/phases/03-capa-de-datos/`, que es runtime de GSD → OK.
- **Conventional commits** (en inglés en el título). Ejemplos válidos: `feat(03-01): add engine_mes + sql loader + schema_guard`.
- **Formato idioma:** código y docstrings en español como el resto del repo; commit messages en inglés.
- **No `filter-repo`, no `--force`.** N/A en Phase 3.
- **No rotar credenciales SQL Server.** N/A en Phase 3.
- **No sustituir matplotlib.** N/A — explícitamente descartado por D-04.
- **`cfg.*` permanece en `ecs_mobility.cfg.*`.** Respetado por D-03 (schema_guard solo nexo).
- **`make up` NO arranca `mcp`.** Respetar el profile al modificar compose.
- **`.env` permisos 600 en producción.** No tocamos `.env` salvo añadir `NEXO_AUTO_MIGRATE` al `.example`.
- **Engine `nexo_app` (Phase 2) preserva gate IDENT-06 — UPDATE/DELETE en `audit_log` bloqueados.** `AuditRepo.append` NO debe hacer nada más que INSERT.
- **No refactorizar módulos `OEE/{disponibilidad,rendimiento,calidad,oee_secciones}`.** Fuera de scope Mark-III.
- **Postgres 16 = casa de `nexo.*`**; `cfg.*` y `oee.*` se quedan en SQL Server. Coherente con D-03/D-07.
- **Env var naming:** todas las nuevas con prefijo `NEXO_*` (ej: `NEXO_AUTO_MIGRATE`). Respetado.

## Sources

### Primary (HIGH confidence)
- Repo local: `nexo/db/engine.py`, `nexo/db/models.py`, `api/database.py`, `OEE/db/connector.py`, `api/routers/centro_mando.py`, `api/routers/bbdd.py`, `api/routers/auditoria.py`, `api/routers/usuarios.py`, `nexo/services/auth.py`, `api/main.py`, `api/deps.py`, `api/services/db.py`, `api/services/pipeline.py`, `tests/auth/test_audit_append_only.py`, `tests/auth/test_rbac_smoke.py`, `tests/conftest.py`, `requirements.txt`, `requirements-dev.txt`, `docker-compose.yml`, `Makefile`, `api/config.py`, `CLAUDE.md`.
- [.planning/phases/03-capa-de-datos/03-CONTEXT.md](/home/eeguskiza/analisis_datos/.planning/phases/03-capa-de-datos/03-CONTEXT.md) — decisiones D-01..D-08 locked.
- [.planning/phases/02-identidad-auth-rbac-audit/02-01-SUMMARY.md](/home/eeguskiza/analisis_datos/.planning/phases/02-identidad-auth-rbac-audit/02-01-SUMMARY.md), [02-04-SUMMARY.md](/home/eeguskiza/analisis_datos/.planning/phases/02-identidad-auth-rbac-audit/02-04-SUMMARY.md) — precondiciones (engine_nexo + nexo_app role).
- [.planning/REQUIREMENTS.md](/home/eeguskiza/analisis_datos/.planning/REQUIREMENTS.md) DATA-01..DATA-11.
- [.planning/codebase/ARCHITECTURE.md](/home/eeguskiza/analisis_datos/.planning/codebase/ARCHITECTURE.md), [STRUCTURE.md](/home/eeguskiza/analisis_datos/.planning/codebase/STRUCTURE.md) — layout actual.

### Secondary (MEDIUM confidence)
- [SQLAlchemy 2.0 SQL Expressions FAQ](https://docs.sqlalchemy.org/en/20/faq/sqlexpressions.html) — `bindparam(expanding=True)` patrón canónico.
- [SQLAlchemy 2.0 Column Elements](https://docs.sqlalchemy.org/en/20/core/sqlelement.html) — API de `bindparam`.
- [Pydantic V2 Models](https://docs.pydantic.dev/latest/concepts/models/) — `ConfigDict(frozen=True, from_attributes=True)`.
- [Pydantic V2 Migration](https://docs.pydantic.dev/latest/migration/) — `orm_mode → from_attributes`, `allow_mutation → frozen`.
- [FastAPI Lifespan Events](https://fastapi.tiangolo.com/advanced/events/) — `@asynccontextmanager` patrón oficial.
- [Patterns and Practices for SQLAlchemy 2.0 with FastAPI](https://chaoticengineer.hashnode.dev/fastapi-sqlalchemy) — multi-engine + DI.
- [Testcontainers Python](https://testcontainers-python.readthedocs.io/) — alternativa no adoptada (D-08 fija compose).

### Tertiary (LOW confidence — marked for validation)
- [pandas 3.0.2 read_sql docs](https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html) — aceptación de pyodbc raw connection sigue siendo "other DBAPI2" con UserWarning. A validar con smoke del pipeline.
- [GitHub pandas/pandas#57053](https://github.com/pandas-dev/pandas/issues/57053) — pandas ≥2.2 + SQLAlchemy interaction issues. No afecta el uso actual (pyodbc raw) pero monitorizar.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — todas las versiones verificadas en `requirements.txt` del repo.
- Architecture: HIGH — patrones leídos del código existente (engines múltiples ya en runtime, DI ya en uso, DTOs frozen ya consumidos en schema Pydantic).
- Pitfalls: HIGH — pitfalls 3/6/7/11/12 verificados en el código; pitfall 2 (pandas+pyodbc) MEDIUM pending smoke pre-refactor.
- Test strategy: HIGH — pattern establecido por `tests/auth/test_rbac_smoke.py` + `test_audit_append_only.py`.
- PDF regression guard: MEDIUM — A2 assumption (matplotlib metadata timestamp) a validar con primer run.

**Research date:** 2026-04-19
**Valid until:** 2026-05-19 (30 días — stack estable, pineado)

---

## RESEARCH COMPLETE

**Phase:** 3 - Capa de datos
**Confidence:** HIGH

### Key Findings

1. **Zero tecnología nueva:** los 3 engines coexisten trivialmente en SQLAlchemy 2.0 (ya hay 2 en runtime). Patrón DI con `Depends` está establecido en el repo. `bindparam(expanding=True)` para IN-clauses es canónico y uniforme para pyodbc + psycopg2.
2. **`OEE/db/connector.py` NO se reescribe (D-04):** MesRepository es wrapper delgado para los 4 métodos legacy; nuevos métodos (`consulta_readonly`, `centro_mando_fmesmic`) usan engine_mes + loader. Minimiza superficie de regresión vs success criterion #5 (PDFs idénticos).
3. **`schema_guard` es simple:** `inspect(engine_nexo).get_table_names(schema="nexo")` + list diff. Flag `NEXO_AUTO_MIGRATE=true` opcional para dev. Integra en lifespan existente antes de `init_db()`.
4. **Test strategy fija:** Postgres real del compose + fixtures rollback; mocks para MES (unittest.mock sobre `Engine.connect` o `@patch` sobre funciones del connector). Patrón ya establecido en `tests/auth/`.
5. **DATA-09 es mecánico:** `engine_mes` con `DATABASE=dbizaro` en conn string + queries con 2-part names (`admuser.fmesmic`). Un único smoke test de `DB_NAME()` + grep-based meta-test valida el gate.
6. **Success criterion #5 requiere task explícita:** grabar PDF baseline PRE-refactor + script de comparación (hash con fallback a size ±5%). No es afterthought — debe ser la primera task de 03-02.
7. **Wave 0 gap es grande pero acotado:** ~13 archivos de tests + 2 scripts + 1 fixture dir + 1 Makefile target + 1 entry en `.env.example`. Sin instalación de framework nuevo.

### File Created

`/home/eeguskiza/analisis_datos/.planning/phases/03-capa-de-datos/03-RESEARCH.md`

### Confidence Assessment

| Area | Level | Reason |
|------|-------|--------|
| Standard Stack | HIGH | Versiones pineadas en repo, nada nuevo que añadir |
| Architecture | HIGH | Patrones ya consumidos en Phase 2 (engine_nexo + Depends + DTOs); 03 extiende |
| Pitfalls | HIGH | 12 pitfalls verificados en código; solo A1 (pandas+pyodbc) pending smoke |
| DATA-09 mechanics | HIGH | Mecánica SQL Server verificada; gate concreto con `SELECT DB_NAME()` |
| PDF hash regression | MEDIUM | A2 (matplotlib metadata) necesita primer run para confirmar estrategia |
| Test fixtures | HIGH | Patrón identico al de `tests/auth/` ya funcionando |
| MES mocking | HIGH | Dos patrones con ejemplos; elección por método clara |
| Schema guard | HIGH | API inspect() documentada; comportamiento simple (dos ramas) |

### Open Questions (escaladas a planner)

1. `consulta_readonly` y engines dinámicos para DBs != dbizaro — recomendación: Mark-III solo dbizaro, metadata ops del explorer quedan inline.
2. `MesRepository` carga config por method o por constructor — recomendación: por method (consistency con connector legacy).
3. `schema_guard` no valida `ecs_mobility.cfg.*` (D-07) — aceptar el gap.
4. A1/A2 assumptions (pandas+pyodbc runtime + matplotlib PDF hash) → convertir en smoke tasks explícitas al inicio de 03-02.

### Ready for Planning

Research completo. El planner puede ahora:
- Crear `.planning/phases/03-capa-de-datos/03-01-PLAN.md` (foundation).
- Crear `.planning/phases/03-capa-de-datos/03-02-PLAN.md` (MES + 5 routers).
- Crear `.planning/phases/03-capa-de-datos/03-03-PLAN.md` (APP+NEXO + 3 routers + auth/auditoria/usuarios migration).
- Los 13 archivos de tests + 2 scripts del Wave 0 van al arranque de cada plan respectivo.
- El hash guard de PDFs es la primera task obligatoria del plan 03-02.
