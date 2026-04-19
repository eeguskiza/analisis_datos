# Phase 3: Capa de datos — Pattern Map

**Mapped:** 2026-04-19
**Files analyzed:** 47 (new + modified)
**Analogs found:** 44 / 47

Upstream sources: `.planning/phases/03-capa-de-datos/03-CONTEXT.md` (D-01..D-08), `.planning/phases/03-capa-de-datos/03-RESEARCH.md` (§Architecture Patterns, §Wrapping OEE/db/connector.py, §DATA-09, §Implementation Patterns Library).

This map groups files by the three plans (03-01 foundation, 03-02 MES, 03-03 APP+NEXO) and provides concrete analog files + line-referenced excerpts the planner will cite verbatim in each task's "action" section.

---

## File Classification

### Plan 03-01 — Foundation (engines + loader + DTOs base + schema_guard + fixtures)

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `nexo/data/__init__.py` | package marker | n/a | `nexo/db/__init__.py` (empty) | exact |
| `nexo/data/engines.py` | infra (engine factory) | connection pooling | `nexo/db/engine.py` (engine_nexo) + `api/database.py:19-41` (engine SQL Server) | exact (dual source) |
| `nexo/data/sql/__init__.py` | package marker | n/a | `nexo/__init__.py` | exact |
| `nexo/data/sql/loader.py` | utility (resource reader) | file-I/O (cached) | no exact analog — RESEARCH §Pattern 2 | no analog |
| `nexo/data/dto/__init__.py` | package marker | n/a | `nexo/db/__init__.py` | exact |
| `nexo/data/dto/app.py`, `mes.py`, `nexo.py` | DTO (Pydantic frozen) | transform (ORM→DTO) | `api/models.py` (Pydantic request models) | role-match (different direction) |
| `nexo/data/schema_guard.py` | startup validator | introspection | `api/database.py:203-209` (`init_db()`) + `api/middleware/audit.py` startup check patterns | partial |
| `nexo/db/engine.py` (modified → SHIM) | re-export shim | n/a | no analog — new pattern for Phase 3 | no analog |
| `api/deps.py` (modified — add `get_db_app`, `get_db_nexo`, `get_engine_mes`) | DI factory | request-response | `api/database.py:281-287` (`get_db()`) + `api/routers/auditoria.py:46-51` (`get_nexo_db()`) | exact |
| `api/main.py` (modified — wire `schema_guard.verify()`) | lifespan wiring | event-driven (startup) | `api/main.py:25-34` (existing `init_db()` hook in `lifespan`) | exact |
| `.env.example` (modified — add `NEXO_AUTO_MIGRATE`) | config docs | n/a | existing `.env.example` entries with comments | exact |
| `tests/data/__init__.py` | test marker | n/a | `tests/auth/__init__.py` | exact |
| `tests/data/conftest.py` | test fixtures | test setup | `tests/auth/conftest.py` (not present — only `tests/conftest.py:1-16` + `tests/auth/test_rbac_smoke.py:37-60` for `_postgres_reachable` + skipif) | role-match |
| `tests/data/test_engines.py` | unit (pool config assertions) | introspection | `tests/auth/test_audit_append_only.py` (integration style) | partial |
| `tests/data/test_sql_loader.py` | unit (resource reads) | file-I/O | no analog | no analog |
| `tests/data/test_schema_guard.py` | integration (startup check) | introspection | `tests/auth/test_audit_append_only.py:99-179` (rol-level DB probing) | partial |
| `tests/data/test_dto_immutable.py` | unit (Pydantic frozen) | n/a | `api/models.py` (shape reference) | partial |
| `tests/data/test_fixtures.py` | meta-test (fixtures work) | test setup | `tests/auth/test_rbac_smoke.py:37-60` | partial |

### Plan 03-02 — MES layer (MesRepository + 5 router refactors + PDF regression gate)

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `nexo/data/repositories/__init__.py` | package marker | n/a | `nexo/db/__init__.py` | exact |
| `nexo/data/repositories/mes.py` | repository (read-only) | request-response (wrap legacy) | `api/services/db.py` (existing thin wrapper over `OEE.db.connector`) | exact |
| `nexo/data/sql/mes/extraer_datos_produccion.sql` | SQL artifact | n/a | `OEE/db/connector.py` extraer_datos inline SQL (not read — delegated) | n/a |
| `nexo/data/sql/mes/estado_maquina_live.sql` | SQL artifact | n/a | delegated to connector | n/a |
| `nexo/data/sql/mes/calcular_ciclos_reales.sql` | SQL artifact | n/a | delegated to connector | n/a |
| `nexo/data/sql/mes/centro_mando_fmesmic.sql` | SQL artifact (new 2-part-name version) | n/a | `api/routers/centro_mando.py:50-64` (current 3-part-name SQL) | exact (rewrite target) |
| `nexo/data/sql/mes/capacidad_*.sql` | SQL artifact | n/a | `api/routers/capacidad.py:60-87` (inline SQL to extract) | exact |
| `nexo/data/sql/mes/operarios_*.sql` | SQL artifact | n/a | `api/routers/operarios.py:34-53` | exact |
| `nexo/data/sql/mes/luk4_*.sql` | SQL artifact | n/a | `api/routers/luk4.py:47-54,79-80+` | exact |
| `api/routers/centro_mando.py` (modified) | router | request-response | itself pre-refactor (lines 38-88, 117) | self (before/after) |
| `api/routers/capacidad.py` (modified) | router | request-response | itself pre-refactor (lines 36-88+) | self |
| `api/routers/operarios.py` (modified) | router | request-response | itself pre-refactor (lines 18-22, 31-56) | self |
| `api/routers/luk4.py` (modified) | router | request-response | itself pre-refactor (lines 44-80+) | self |
| `api/routers/bbdd.py` (modified) | router (metadata explorer) | request-response | itself (lines 17-52, 150+) + keep anti-DDL whitelist | self |
| `api/services/db.py` (modified — repoint imports) | service shim | request-response | itself (lines 10-20) | self |
| `api/services/pipeline.py` (modified — repoint imports) | service | batch | itself (import block) | self |
| `tests/data/test_mes_repository.py` | unit (mock engine) | test | `tests/auth/test_audit_append_only.py` pattern + RESEARCH §Patrón 1+2 | partial |
| `tests/data/test_mes_engine_context.py` | integration (DB_NAME()) | introspection | `tests/auth/test_audit_append_only.py:32-54` (SELECT current_user check) | exact (pattern analog) |
| `tests/data/test_no_three_part_names.py` | meta-test (grep) | file-I/O | no analog — grep-based | no analog |
| `scripts/gen_pdf_reference.py` | standalone script | batch | `scripts/extract_2025.py` + `scripts/init_nexo_schema.py` | role-match |
| `scripts/pdf_regression_check.py` | standalone script | batch | `scripts/init_nexo_schema.py` | role-match |

### Plan 03-03 — APP + NEXO layer (repos + 3 router refactors + auth.py/auditoria.py/usuarios.py refactor)

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `nexo/data/repositories/app.py` | repository (ORM CRUD) | CRUD | `nexo/services/auth.py:175-178` (get_user_by_email), `api/routers/recursos.py:24-31,52-65` (inline ORM) | role-match (pattern extraction) |
| `nexo/data/repositories/nexo.py` | repository (ORM CRUD + auth) | CRUD | `nexo/services/auth.py:95-178` (all ORM queries on Nexo*) | exact |
| `nexo/data/models_app.py` (new) | ORM models | n/a | `api/database.py:44-190` (existing models) | exact (migration target) |
| `nexo/data/models_nexo.py` (new) | ORM models | n/a | `nexo/db/models.py:40-173` (full file) | exact (migration target) |
| `nexo/db/engine.py` (delete shim at end of 03-03) | cleanup | n/a | self | self |
| `nexo/db/models.py` (shim → delete at end of 03-03) | re-export shim | n/a | no analog | no analog |
| `api/database.py` (modified — re-export only) | infra legacy | n/a | self | self |
| `api/routers/historial.py` (modified) | router | CRUD | itself (lines 18-46, 49-97, 100-134, 137-148) | self |
| `api/routers/recursos.py` (modified) | router | CRUD | itself (lines 1-100+) | self |
| `api/routers/ciclos.py` (modified) | router | CRUD | itself (lines 1-167) | self |
| `api/routers/auditoria.py` (modified) | router | CRUD (reads + export) | itself (lines 30-220) | self |
| `api/routers/usuarios.py` (modified) | router | CRUD | itself (lines 27-295) | self |
| `nexo/services/auth.py` (modified — consume repos) | service | CRUD (via repos) | itself (lines 95-178) | self |
| `tests/data/test_app_repository.py` | integration | CRUD | `tests/auth/test_rbac_smoke.py:37-60` + RESEARCH §Ref 8 | partial |
| `tests/data/test_nexo_repository.py` | integration | CRUD | `tests/auth/test_audit_append_only.py` (IDENT-06 gate) | exact |
| `tests/data/test_routers_smoke.py` | integration (8 routers HTTP) | request-response | `tests/auth/test_rbac_smoke.py:69-74` (TestClient pattern) | exact |
| `tests/data/test_no_raw_pyodbc_in_routers.py` | meta-test (grep) | file-I/O | no analog — grep-based | no analog |
| `Makefile` (modified — add `test-data` target) | build tool | n/a | `Makefile:53-60` (existing `nexo-init`, `nexo-owner` targets) | exact |

---

## Pattern Assignments

### `nexo/data/engines.py` (infra, connection pooling)

**Dual analog:** `nexo/db/engine.py` (full file, Phase 2 analog for Postgres) + `api/database.py:17-41` (legacy analog for SQL Server with pyodbc creator).

**Imports pattern** (copy from `nexo/db/engine.py:11-17`):
```python
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from api.config import settings
```

**Postgres DSN builder + engine_nexo** (copy verbatim from `nexo/db/engine.py:20-54`, note use of `settings.effective_pg_user` / `effective_pg_password` — landmine #12 in RESEARCH: must stay as `effective_*` or the IDENT-06 gate silently breaks):
```python
def _build_dsn() -> str:
    user = settings.effective_pg_user
    password = settings.effective_pg_password
    return (
        f"postgresql+psycopg2://{user}:{password}"
        f"@{settings.pg_host}:{settings.pg_port}/{settings.pg_db}"
    )

engine_nexo: Engine = create_engine(
    _build_dsn(),
    pool_size=5,
    max_overflow=5,
    pool_timeout=10,
    pool_recycle=1800,
    pool_pre_ping=True,
    echo=settings.debug,
)

SessionLocalNexo = sessionmaker(
    bind=engine_nexo,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)
```

**SQL Server re-export pattern** (copy `engine_app` re-export from `api/database.py:19-41`, but **do not duplicate** — import it). The legacy engine uses `creator=_mssql_creator` (pyodbc raw) not URL — Landmine #3 in RESEARCH requires this preservation:
```python
# ── engine_app (ecs_mobility) — re-export desde api/database.py ─
# Durante Phase 3 se re-exporta. No duplicar _mssql_creator:
# RESEARCH §Landmine 3 exige preservar el creator pyodbc directo.
from api.database import engine as engine_app  # noqa: E402

SessionLocalApp = sessionmaker(
    bind=engine_app,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)
```

**New engine_mes with DATABASE=dbizaro** (RESEARCH §Ref 1 + DATA-11). This is the key piece of DATA-09 — baking `dbizaro` into the connection string kills 3-part names:
```python
def _build_mes_dsn() -> str:
    pwd = settings.mes_password.replace("+", "%2B")
    return (
        f"mssql+pyodbc://{settings.mes_user}:{pwd}"
        f"@{settings.mes_server}:{settings.mes_port}/{settings.mes_db}"
        "?driver=ODBC+Driver+18+for+SQL+Server"
        "&TrustServerCertificate=yes&Encrypt=yes"
    )

engine_mes: Engine = create_engine(
    _build_mes_dsn(),
    pool_pre_ping=True,          # DATA-11
    pool_recycle=3600,           # DATA-11
    pool_size=3, max_overflow=2,
    pool_timeout=15,             # DATA-11
    connect_args={"timeout": 15},
)
```

---

### `nexo/db/engine.py` (modified → SHIM for Phase 3)

**Replace entire file** (after saving original content in `nexo/data/engines.py`). Shim pattern from RESEARCH §Architecture Pattern 3 §Shim:
```python
"""SHIM — moved to nexo/data/engines.py in Plan 03-01.
Eliminar en 03-03 cuando nexo/services/auth.py migre a nexo.data.engines."""
from nexo.data.engines import engine_nexo, SessionLocalNexo  # noqa: F401
```

No alias beyond the re-export (Pitfall 5 in RESEARCH).

---

### `nexo/data/sql/loader.py` (utility, file-I/O cached)

**No codebase analog.** Copy verbatim from RESEARCH §Ref 2 / §Pattern 2 (lines 326-348):
```python
from __future__ import annotations
from functools import lru_cache
from importlib.resources import files

_PACKAGE = "nexo.data.sql"

@lru_cache(maxsize=128)
def load_sql(name: str) -> str:
    """Carga 'mes/extraer_datos_produccion.sql' desde el paquete."""
    if not name.endswith(".sql"):
        name = f"{name}.sql"
    ref = files(_PACKAGE)
    for part in name.split("/"):
        ref = ref / part
    return ref.read_text(encoding="utf-8")
```

**Consumer pattern** (RESEARCH §Ref 7):
```python
from sqlalchemy import bindparam, text
from nexo.data.sql.loader import load_sql

stmt = text(load_sql("mes/centro_mando_fmesmic")).bindparams(
    bindparam("codes", expanding=True)
)
```

---

### `nexo/data/schema_guard.py` (startup validator)

**Partial analog:** `api/database.py:203-209` (`init_db()`) shows the "run at startup, log errors" pattern. The new file follows RESEARCH §Lifespan + schema_guard + §Ref 3 (lines 652-703):

**Module pattern** (copy from RESEARCH §Ref 3 lines 1470-1484 and §Lifespan Implementation lines 662-703):
```python
from __future__ import annotations
import logging
import os
from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from nexo.data.models_nexo import NexoBase, NEXO_SCHEMA

log = logging.getLogger("nexo.schema_guard")

CRITICAL_TABLES = (
    "users", "roles", "departments", "user_departments",
    "permissions", "sessions", "login_attempts", "audit_log",
)

def verify(engine: Engine) -> None:
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

**Import `NEXO_SCHEMA`** — constant already defined at `nexo/db/models.py:37` → will be re-exported from `nexo/data/models_nexo.py` in 03-03. During 03-01 it still lives in `nexo/db/models.py`; import as `from nexo.db.models import NexoBase, NEXO_SCHEMA`.

---

### `api/main.py` (modified — wire schema_guard)

**Exact analog:** itself (lines 25-34). The existing `lifespan` body has the shape we need:

**Current (api/main.py:25-34)**:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa la BBDD al arrancar."""
    try:
        init_db()
        logger.info("Base de datos inicializada OK")
    except Exception as exc:
        logger.error(f"Error inicializando BD: {exc}")
        logger.error(traceback.format_exc())
    yield
```

**Modification** (RESEARCH §Ref 3 lines 1486-1493 + §Lifespan Implementation lines 706-728) — `schema_guard.verify()` runs **before** `init_db()` and is **not wrapped** in the existing try/except. If guard raises, lifespan must abort:
```python
from nexo.data.engines import engine_nexo
from nexo.data import schema_guard

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Schema guard ANTES de init_db — si falla aquí, abortamos claro.
    schema_guard.verify(engine_nexo)  # raises RuntimeError si falta tabla y auto-migrate=false

    # 2. init_db existente (bootstrap ecs_mobility) — se mantiene el try/except.
    try:
        init_db()
        logger.info("Base de datos inicializada OK")
    except Exception as exc:
        logger.error(f"Error inicializando BD: {exc}")
        logger.error(traceback.format_exc())
    yield
```

Ordering is critical — `schema_guard` must fail loud, whereas `init_db()` is best-effort (the legacy behavior). Landmine #11 in RESEARCH: do **not** touch middleware registration order (`app.add_middleware(AuditMiddleware); app.add_middleware(AuthMiddleware)` at lines 123-124).

---

### `api/deps.py` (modified — add DB/engine dependencies)

**Exact analogs:**
- `api/database.py:281-287` (existing `get_db()` yield-style generator):
```python
def get_db():
    """Dependency para FastAPI: yield session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```
- `api/routers/auditoria.py:46-51` (existing `get_nexo_db()` — literally the pattern we generalize):
```python
def get_nexo_db():
    db = SessionLocalNexo()
    try:
        yield db
    finally:
        db.close()
```

**New additions to `api/deps.py`** (RESEARCH §Pattern 1 + §Ref 5 lines 295-318, 494-520). Current file has only `templates`/`render` — append at the bottom:
```python
from typing import Annotated, Iterator

from fastapi import Depends
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from nexo.data.engines import (
    SessionLocalApp,
    SessionLocalNexo,
    engine_mes as _engine_mes,
)


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


def get_engine_mes() -> Engine:
    """MES es read-only; entregamos el Engine directamente."""
    return _engine_mes


# Annotated aliases (PEP 593) for cleaner router signatures
DbApp = Annotated[Session, Depends(get_db_app)]
DbNexo = Annotated[Session, Depends(get_db_nexo)]
EngineMes = Annotated[Engine, Depends(get_engine_mes)]
```

The `get_nexo_db()` helpers currently duplicated in `api/routers/auditoria.py:46-51` and `api/routers/usuarios.py:50-55` are **replaced by `get_db_nexo`** during 03-03 router refactor.

---

### `nexo/data/dto/*.py` (Pydantic frozen DTOs)

**Partial analog:** `api/models.py` (existing Pydantic request models — shape reference only; those are mutable request DTOs, new `*Row` DTOs are frozen response DTOs).

**DTO pattern** (copy from RESEARCH §Pydantic v2 DTO Recipe lines 564-595 + §Ref 4 lines 1497-1507):
```python
# nexo/data/dto/app.py
from datetime import date
from typing import Optional
from pydantic import BaseModel, ConfigDict

class RecursoRow(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    id: int
    centro_trabajo: int
    nombre: str
    seccion: str = "GENERAL"
    activo: bool = True
```

**Validator pattern for MES DTOs** (RESEARCH lines 619-641) — only when hydrating from pandas/pyodbc raw dicts:
```python
from pydantic import field_validator

class ProduccionRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    recurso: str
    seccion: str
    fecha: date
    # ...

    @field_validator("fecha", mode="before")
    @classmethod
    def _coerce_fecha(cls, v):
        if hasattr(v, "date") and not isinstance(v, date):
            return v.date()
        return v
```

**DTOs to define per file** (field shapes extracted from router response dicts + ORM model columns):
- `nexo/data/dto/mes.py` — `ProduccionRow` (fields from `api/routers/centro_mando.py:139-146` + `OEE/db/connector.py` extraer_datos output), `EstadoMaquinaRow`, `CicloRealRow`, `CapacidadRow` (fields from `api/routers/capacidad.py:98-105`), `OperarioRow` (fields from `api/routers/operarios.py:62-69`).
- `nexo/data/dto/app.py` — `RecursoRow`, `CicloRow`, `EjecucionRow`, `MetricaRow` (fields from `api/database.py:138-164`), `ReferenciaRow` (fields from `api/database.py:167-178`), `IncidenciaRow`, `InformeRow` (fields from `api/database.py:91-103`), `ContactoRow` (from `api/database.py:126-133`), `LukRow`.
- `nexo/data/dto/nexo.py` — `UserRow` (from `nexo/db/models.py:61-83`), `RoleRow` (from `nexo/db/models.py:88-95`), `AuditLogRow` (from `nexo/db/models.py:162-173`), shape matches `api/routers/auditoria.py:130-141` serialization.

---

### `nexo/data/repositories/mes.py` (repository, wrap legacy)

**Exact analog:** `api/services/db.py` full file. It **already is** a thin wrapper over `OEE.db.connector`. The new `MesRepository` is a class-shaped version of the same idea.

**Existing analog (`api/services/db.py:10-20`) — import block to copy**:
```python
from OEE.db.connector import (
    calcular_ciclos_reales,
    datos_a_csvs,
    detectar_recursos,
    estado_maquina_live,
    explorar_columnas_fmesdtc,
    extraer_datos,
    load_config,
    save_config,
    test_conexion,
)
```

**Existing analog (`api/services/db.py:23-75`) — thin-wrapper delegation pattern**:
```python
def get_config() -> dict:
    cfg = load_config()
    if not cfg.get("server") or cfg["server"] == "":
        cfg["server"] = settings.db_server
    # ... (merges env settings)
    return cfg


def discover_resources() -> List[dict]:
    cfg = get_config()
    return detectar_recursos(cfg)


def extract_data(fecha_inicio: date, fecha_fin: date, recursos: list[str] | None = None) -> List[dict]:
    cfg = get_config()
    if recursos:
        cfg = {**cfg, "recursos": [
            r for r in cfg.get("recursos", [])
            if r.get("nombre") in recursos
        ]}
    return extraer_datos(cfg, fecha_inicio, fecha_fin)
```

**New `MesRepository` class** — copy verbatim from RESEARCH §Wrapping OEE/db/connector.py lines 984-1051. Key points:
- `__init__(self, engine: Engine)` — engine injected.
- The 4 legacy-wrapping methods (`extraer_datos_produccion`, `detectar_recursos`, `calcular_ciclos_reales`, `estado_maquina_live`) **reuse `api.services.db.get_config()`** rather than reimplementing config loading (Landmine #1, A5). Import local: `from api.services.db import get_config as _get_mes_config`.
- `consulta_readonly(sql, database)` uses `self._engine.connect()` directly and runs already-validated SQL (D-05, whitelist stays in `bbdd.py`).
- `centro_mando_fmesmic(ct_codes)` and other "new" methods use `load_sql(name)` + `bindparam("codes", expanding=True)`.

---

### `api/routers/centro_mando.py` (modified — DATA-09 target)

**Self analog — line-by-line before/after ready for planner:**

**BEFORE — `api/routers/centro_mando.py:12` (import block)**:
```python
from api.database import Recurso, engine, get_db
```

**AFTER**:
```python
from api.deps import DbApp, EngineMes
from nexo.data.repositories.app import RecursoRepo
from nexo.data.repositories.mes import MesRepository
```

**BEFORE — `api/routers/centro_mando.py:38-88` (_query_fmesmic inline SQL with 3-part names)**:
```python
def _query_fmesmic(ct_codes: list[int]) -> dict[int, dict]:
    if not ct_codes:
        return {}

    placeholders = ",".join(f":ct{i}" for i in range(len(ct_codes)))
    params = {f"ct{i}": str(ct) for i, ct in enumerate(ct_codes)}

    sql = text(f"""
        SELECT
            CAST(RTRIM(mi020) AS INT)          AS ct,
            COUNT(*)                            AS piezas_hoy,
            MAX(CAST(mi100 AS TIME))            AS ultimo_evento,
            (SELECT TOP 1 RTRIM(m2.mi060)
             FROM dbizaro.admuser.fmesmic m2             -- ← 3-part name
             WHERE RTRIM(m2.mi020) = RTRIM(m.mi020)
               AND CONVERT(DATE, m2.mi090) = CONVERT(DATE, GETDATE())
             ORDER BY m2.mi050 DESC)            AS referencia
        FROM dbizaro.admuser.fmesmic m                   -- ← 3-part name
        WHERE CONVERT(DATE, mi090) = CONVERT(DATE, GETDATE())
          AND RTRIM(mi020) IN ({placeholders})            -- ← string-interpolated IN
        GROUP BY RTRIM(mi020)
    """)

    result = {}
    with engine.connect() as conn:                        -- ← engine_app (wrong engine)
        rows = conn.execute(sql, params).fetchall()
        for r in rows:
            # ... tuple indexing r[0], r[1], r[2], r[3]
```

**AFTER** — extract to `nexo/data/sql/mes/centro_mando_fmesmic.sql` (2-part names, RESEARCH §D-09 lines 1075-1105):
```sql
-- NOTA: engine_mes tiene DATABASE=dbizaro en connection string → 2-part names.
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

Then call via `MesRepository.centro_mando_fmesmic(ct_codes)`. The post-refactor shape is RESEARCH §Ref 6 lines 1527-1547.

**BEFORE — `api/routers/centro_mando.py:117-126` (router handler signature)**:
```python
@router.get("/summary")
def summary(db: Session = Depends(get_db)):
    hoy = _fecha_produccion()
    all_recursos = (
        db.query(Recurso)
        .filter_by(activo=True)
        .order_by(Recurso.seccion, Recurso.nombre)
        .all()
    )
```

**AFTER**:
```python
@router.get("/summary")
def summary(db: DbApp, engine_mes: EngineMes):
    hoy = _fecha_produccion()
    recurso_repo = RecursoRepo(db)
    mes_repo = MesRepository(engine=engine_mes)
    recursos = recurso_repo.list_activos()  # returns list[RecursoRow]
```

Cache wrapping (`_get_cached_data`, `_cache`) is preserved — it's transport concern, not repo.

---

### `api/routers/bbdd.py` (modified — preserve anti-DDL whitelist, D-05)

**Self analog + D-05 constraint:** Lines 17-52 (pyodbc conn-string construction for `master`) and the `list_databases`, `list_tables`, `list_columns`, `preview` metadata ops (lines 55-149+) stay inline (D-05 explicit decision — specific to UI, no justification for repo method).

**What changes:**
- Current `_get_conn_string()` (lines 17-38) and `_connect*()` helpers (lines 41-52) stay for metadata ops.
- User-submitted SQL (`POST /bbdd/consulta` handler further down the file) moves to `MesRepository.consulta_readonly(sql, database="dbizaro")`.
- Whitelist regex (not shown in my read but documented in D-05 and at the POST handler) stays in `bbdd.py` — transport concern.
- Test `tests/data/test_bbdd_whitelist.py` (implicit in D-05) exercises the regex with malicious SQL.

---

### `api/routers/capacidad.py` (modified)

**Self analog — lines 17-22 (router decl preserved), lines 44-51 (current connect pattern to replace):**
```python
import pyodbc                                            # ← REMOVE
from api.services.db import get_config                    # ← REMOVE
from OEE.db.connector import _build_connection_string     # ← REMOVE

cfg = {**get_config(), "database": "dbizaro"}
conn = pyodbc.connect(_build_connection_string(cfg), timeout=30)
cursor = conn.cursor()
```

**Replacement pattern:** inject `EngineMes` dep + call `MesRepository`. SQL at lines 60-87 (production query) and 119+ (cycle-teorico query) extracted to `nexo/data/sql/mes/capacidad_piezas_linea.sql` and `nexo/data/sql/mes/capacidad_ciclos_p10_180d.sql` (names from RESEARCH structure, lines 243-244).

The dynamic `OR`-clause generation at `api/routers/capacidad.py:110-114` (cartesian product of `(ref, ct)` tuples) needs rewriting with `bindparam(..., expanding=True)` on a VALUES table or composite IN — planner should evaluate whether it's worth extracting to SQL file or keeping as a repo method that builds the `text()` with composite bindparams.

---

### `api/routers/operarios.py` (modified)

**Self analog — lines 18-22 (to replace)**:
```python
def _connect():
    from OEE.db.connector import _build_connection_string
    import pyodbc
    cfg = mes_service.get_config()
    return pyodbc.connect(_build_connection_string(cfg), timeout=15)
```

**Replacement:** inject `EngineMes` and use repo methods. SQL at lines 34-53 extracts to `nexo/data/sql/mes/operarios_listar.sql`. Current router has 6 queries — extract each to its own `.sql` file following D-05 ("un archivo = una query canónica").

---

### `api/routers/luk4.py` (modified)

**Self analog — lines 13 (to remove), 45-80 (queries using `engine` SQL Server directly)**:
```python
from api.database import engine              # ← REMOVE — uses engine_app, should use engine_mes
...
with engine.connect() as conn:
    est = conn.execute(text("""
        SELECT TOP 1 e.timestamp, ...
        FROM luk4.estado e
        LEFT JOIN luk4.alarmas a ON a.codigo = e.codigo_error
        ORDER BY e.idcelula_estado DESC
    """)).fetchone()
```

**Note:** `luk4.estado` lives in `ecs_mobility.luk4.*` (the APP database), not `dbizaro`. Careful: the router name is misleading — LUK4 telemetry is in APP, not MES. The planner must decide whether `luk4_*` queries go through `MesRepository` (no — they're in APP) or through a new repo method wrapped in `engine_app`. RESEARCH classifies LUK4 under MES in the SQL layout (line 243), but **reading the code, `luk4.*` tables are in APP**. Propose: `LukRepo` in `nexo/data/repositories/app.py` (DATA-03 already lists `LukRepo`) consuming SQL from `nexo/data/sql/app/luk4_*.sql`. Planner confirms in 03-03.

---

### `api/services/db.py` (modified — import repoint only)

**Self analog — current (lines 10-20):**
```python
from OEE.db.connector import (
    calcular_ciclos_reales,
    datos_a_csvs,
    detectar_recursos,
    ...
)
```

**After 03-02:** the file stays as a facade (service-layer helpers still used by `api/services/pipeline.py`), but can optionally import `MesRepository` for the repo-wrapped methods while keeping direct `OEE.db.connector` imports for `datos_a_csvs`, `test_conexion`, `explorar_columnas_fmesdtc` (not wrapped by `MesRepository` per D-04). Planner decides minimum diff.

---

### `api/services/pipeline.py` (modified — import repoint + model references)

**Landmine #9 in RESEARCH:** `api/services/pipeline.py:_save_metrics_to_db` has ~15 direct references to `MetricaOEE`, `ReferenciaStats`, `IncidenciaResumen` from `api.database`. When Plan 03-03 moves those models to `nexo/data/models_app.py`, the imports must update.

Strategy: keep a re-export in `api/database.py` (the file itself stays for the legacy engine + `init_db()`). Planner to write a shim at end of 03-03:
```python
# api/database.py after 03-03 (tail)
from nexo.data.models_app import (  # noqa: F401
    Recurso, Ciclo, Ejecucion, InformeMeta, DatosProduccion,
    Contacto, MetricaOEE, ReferenciaStats, IncidenciaResumen,
)
```

---

### `nexo/data/repositories/nexo.py` (repository, ORM queries for auth + audit)

**Exact analog:** `nexo/services/auth.py:95-178` — the entire session-management + lockout + user-lookup surface is already there. Extract per-entity groups into `UserRepo`, `RoleRepo`, `AuditRepo`.

**`get_user_by_email` pattern from `nexo/services/auth.py:175-178`**:
```python
def get_user_by_email(db: Session, email: str) -> Optional[NexoUser]:
    return db.execute(
        select(NexoUser).where(NexoUser.email == email, NexoUser.active.is_(True))
    ).scalar_one_or_none()
```

**Becomes (in 03-03):**
```python
class UserRepo:
    def __init__(self, db: Session):
        self._db = db

    def get_by_email(self, email: str) -> UserRow | None:
        row = self._db.execute(
            select(NexoUser).where(
                NexoUser.email == email, NexoUser.active.is_(True)
            )
        ).scalar_one_or_none()
        return UserRow.model_validate(row) if row else None
```

**Session helpers from `nexo/services/auth.py:95-131`** stay in `auth.py` — `create_session`, `get_session`, `extend_session`, `revoke_session`, `revoke_all_sessions` are behavior (tied to `secrets.token_urlsafe` + `datetime.now(timezone.utc)` business logic). They're not pure CRUD, so they remain services, not repo methods.

**Lockout helpers (`nexo/services/auth.py:140-170`)** stay in `auth.py` too (policy logic, not CRUD).

**AuditRepo.append pattern** (RESEARCH §Pattern 1 lines 281-293) — **NO commit**. Caller owns transaction. Gate IDENT-06 guarantees `nexo_app` rol only has INSERT on audit_log, so any accidental UPDATE/DELETE would fail at DB level (test in `tests/auth/test_audit_append_only.py` enforces):
```python
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
        # NO commit aquí — caller owns the transaction
```

**Consumer update — `nexo/services/auth.py`** (Plan 03-03 task): `get_user_by_email` stays as module function but calls `UserRepo(db).get_by_email(...)` internally. Middleware/`auditoria.py` routers use repos directly.

---

### `nexo/data/repositories/app.py` (repository, ORM queries for APP)

**Analogs — inline ORM in current routers, one method per pattern:**

**Source (`api/routers/recursos.py:24-31`) — `RecursoRepo.list_all`**:
```python
rows = db.query(Recurso).order_by(Recurso.seccion, Recurso.nombre).all()
return {"recursos": [
    {"id": r.id, "centro_trabajo": r.centro_trabajo, "nombre": r.nombre,
     "seccion": r.seccion, "activo": r.activo}
    for r in rows
]}
```

**Repo target (RESEARCH §Ref 5 lines 1510-1523)**:
```python
class RecursoRepo:
    def __init__(self, db: Session):
        self._db = db

    def list_all(self) -> list[RecursoRow]:
        rows = self._db.execute(
            select(Recurso).order_by(Recurso.seccion, Recurso.nombre)
        ).scalars().all()
        return [RecursoRow.model_validate(r) for r in rows]

    def list_activos(self) -> list[RecursoRow]:
        rows = self._db.execute(
            select(Recurso).where(Recurso.activo.is_(True))
                           .order_by(Recurso.seccion, Recurso.nombre)
        ).scalars().all()
        return [RecursoRow.model_validate(r) for r in rows]
```

**Source (`api/routers/ciclos.py:22-45`) — `CicloRepo.list_all`**: current router returns both flat and grouped shapes; the grouping is transport-layer logic. Repo returns the flat list; router does the grouping.

**Source (`api/routers/historial.py:18-46`) — `EjecucionRepo.list_recent`**: current router does a join-with-count on `DatosProduccion`. Repo exposes `list_recent(limit) -> list[EjecucionRow]` and a separate `count_rows_by_ejecucion(ids) -> dict[int, int]`. Router composes.

**Source (`api/routers/historial.py:49-97`) — `MetricaRepo.tendencias_recurso`**: filter + dedup logic moves to repo.

**Source (`api/routers/historial.py:100-134`) — `InformeRepo.list_by_ejecucion` + `EjecucionRepo.get_detail`**.

**ORM import pattern — when models move to `nexo/data/models_app.py`**, keep `api/database.py` re-exporting for the ~15 `api/services/pipeline.py` references (Landmine #9).

---

### `nexo/data/models_app.py` (new — migration target for ORM models)

**Exact analog:** `api/database.py:44-190` (entire ORM block — `class Ciclo`, `Recurso`, `Ejecucion`, `InformeMeta`, `DatosProduccion`, `Contacto`, `MetricaOEE`, `ReferenciaStats`, `IncidenciaResumen`).

**Migration in 03-03:**
1. Copy the `class Base(DeclarativeBase)` declaration (line 44-45) + all models (lines 50-190) verbatim into `nexo/data/models_app.py`.
2. Optional cleanup: migrate `datetime.utcnow` to `datetime.now(timezone.utc)` (RESEARCH §State of the Art lines 1258-1260 — 7 occurrences in `api/database.py`). Non-blocking — can stay if behavior is identical.
3. In `api/database.py`, replace the model classes with a re-export block (`from nexo.data.models_app import ...`).

---

### `nexo/data/models_nexo.py` (new — migration target)

**Exact analog:** `nexo/db/models.py` entire file (lines 1-173). Move verbatim. Keep `NEXO_SCHEMA = "nexo"` constant at the top (line 37). Update `nexo/db/models.py` to shim:
```python
# nexo/db/models.py — SHIM (03-03), eliminar cuando consumidores migren
from nexo.data.models_nexo import (  # noqa: F401
    NEXO_SCHEMA, NexoBase,
    NexoUser, NexoRole, NexoDepartment, NexoPermission,
    NexoSession, NexoLoginAttempt, NexoAuditLog,
    user_departments,
)
```

Existing consumers: `nexo/services/auth.py:35`, `api/routers/auditoria.py:33`, `api/routers/usuarios.py:29`, `tests/auth/test_rbac_smoke.py:28-33`, `scripts/init_nexo_schema.py`. During 03-03 their imports update one-by-one; the shim covers the transition.

---

### `tests/data/conftest.py` (fixtures)

**Exact analog:** `tests/auth/test_rbac_smoke.py:37-60` — `_postgres_reachable()` helper + `pytestmark skipif` pattern. Copy into `tests/data/conftest.py`:

**From `tests/auth/test_rbac_smoke.py:37-48`**:
```python
def _postgres_reachable() -> bool:
    """True si el Postgres del compose esta arriba y aceptando queries."""
    try:
        db = SessionLocalNexo()
        try:
            db.execute(text("SELECT 1"))
            return True
        finally:
            db.close()
    except Exception:
        return False
```

**Full `conftest.py` pattern from RESEARCH §Postgres Test Fixtures lines 745-820**:
```python
from __future__ import annotations
from typing import Iterator
from unittest.mock import MagicMock

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from nexo.data.engines import (
    SessionLocalApp, SessionLocalNexo,
    engine_app, engine_nexo, engine_mes,
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

@pytest.fixture
def db_nexo() -> Iterator[Session]:
    if not _postgres_reachable():
        pytest.skip("Postgres no arriba: `docker compose up -d db`")
    db = SessionLocalNexo()
    db.begin_nested()  # SAVEPOINT
    try:
        yield db
    finally:
        db.rollback()
        db.close()

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

@pytest.fixture
def engine_mes_mock(monkeypatch):
    """Engine MES con conexiones mockeadas."""
    mock_engine = MagicMock()
    monkeypatch.setattr("nexo.data.engines.engine_mes", mock_engine)
    return mock_engine
```

**For commit-then-cleanup test pattern**, reuse `tests/auth/test_audit_append_only.py:60-97` (`_owner_engine()` + `_clean_test_rows` autouse fixture). Tests that commit (like `test_nexo_repository.py::test_audit_repo_append_inserts_row`) must use the owner-engine cleanup because `nexo_app` can't DELETE on audit_log (IDENT-06 gate).

---

### `tests/data/test_mes_repository.py` (MES repo unit tests)

**Analog patterns from RESEARCH §MES Mocking Strategy lines 881-946**:

**Patrón 2 (delegation to legacy connector) — for the 4 wrapping methods**:
```python
from unittest.mock import patch, MagicMock
from datetime import date
from nexo.data.repositories.mes import MesRepository

@patch("nexo.data.repositories.mes._legacy_extraer_datos")
def test_extraer_datos_produccion_delega_a_connector(mock_extraer):
    mock_extraer.return_value = [{"recurso": "luk1", "fecha": "2026-04-01", ...}]
    repo = MesRepository(engine=MagicMock())
    rows = repo.extraer_datos_produccion(
        fecha_inicio=date(2026,4,1), fecha_fin=date(2026,4,1), recursos=["luk1"]
    )
    assert len(rows) == 1
    mock_extraer.assert_called_once()
```

**Patrón 1 (engine + mock cursor) — for `consulta_readonly` and new SQL-loader methods (RESEARCH §Ref 9 lines 1589-1598)**:
```python
def test_consulta_readonly_devuelve_dict_shape(engine_mes_mock):
    mc = engine_mes_mock.connect.return_value.__enter__.return_value
    mc.execute.return_value.keys.return_value = ["col1"]
    mc.execute.return_value.fetchall.return_value = [(42,)]

    repo = MesRepository(engine=engine_mes_mock)
    result = repo.consulta_readonly("SELECT 42 AS col1", "dbizaro")
    assert result["columns"] == ["col1"]
    assert result["rows"] == [[42]]
```

---

### `tests/data/test_mes_engine_context.py` (DATA-09 integration gate)

**Exact pattern analog:** `tests/auth/test_audit_append_only.py:32-54` — runs a `SELECT current_user`-style probe against the engine, gates with `pytestmark` skipif.

**From RESEARCH §DATA-09 Smoke test lines 1121-1146**:
```python
import pytest
from sqlalchemy import text, bindparam
from nexo.data.engines import engine_mes

@pytest.mark.integration
def test_engine_mes_default_database_is_dbizaro():
    with engine_mes.connect() as conn:
        db = conn.execute(text("SELECT DB_NAME()")).scalar()
    assert db == "dbizaro", f"engine_mes debe apuntar a dbizaro, apunta a {db}"

@pytest.mark.integration
def test_centro_mando_query_sin_three_part_names():
    from nexo.data.sql.loader import load_sql
    sql = load_sql("mes/centro_mando_fmesmic")
    assert "dbizaro." not in sql.lower(), "La SQL sigue usando 3-part names"
```

---

### `tests/data/test_no_raw_pyodbc_in_routers.py` + `tests/data/test_no_three_part_names.py` (meta-tests)

**No codebase analog — grep-based meta-tests (RESEARCH §Validation Architecture lines 1390-1404):**
```python
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

Same shape for `dbizaro\.` substring match.

---

### `scripts/gen_pdf_reference.py` + `scripts/pdf_regression_check.py` (regression gate, Success #5)

**Analog:** `scripts/init_nexo_schema.py` + `scripts/extract_2025.py` (existing standalone scripts with argparse + direct imports from `api.services.pipeline`).

**Copy pattern from RESEARCH §Ref 10 lines 1604-1663**:
```python
"""Compara PDF generado tras refactor con baseline pre-refactor."""
import hashlib, sys, argparse
from pathlib import Path

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fecha", required=True)
    args = ap.parse_args()
    # ... compara vs baseline, fallback a size ±5%
```

**Execution sequence in plan 03-02 (RESEARCH §Sampling Rate lines 1361-1365):**
1. Before 03-02: `python scripts/gen_pdf_reference.py --fecha=2026-03-15` → commits `tests/data/reference/pipeline_2026-03-15.pdf` + `.sha256`.
2. After 03-02: `python scripts/pdf_regression_check.py --fecha=2026-03-15` → exit 0 = OK / 1 = WARN (hash differs, size ±5%) / 2 = REGRESSION.

---

### `Makefile` (modified — `test-data` target)

**Exact analog:** existing `nexo-init`, `nexo-owner` targets at `Makefile:55-59`:
```makefile
nexo-init: ## Crea schema nexo + 8 tablas + seed (idempotente)
	docker compose exec web python scripts/init_nexo_schema.py

nexo-owner: ## Crea el primer usuario 'propietario' (interactivo)
	docker compose exec -it web python scripts/create_propietario.py
```

**New target pattern from RESEARCH §Makefile lines 860-867**:
```makefile
test-data: ## Arranca compose, corre tests tests/data/ y apaga compose
	docker compose up -d db
	@echo "Esperando Postgres healthy..."
	@until docker compose ps --format json db | grep -q '"Health":"healthy"'; do sleep 1; done
	docker compose exec -T web pytest tests/data/ -q
	# No apagamos compose; dev normalmente lo quiere arriba.
```

Update `.PHONY` line at `Makefile:1` to include `test-data`.

---

### `.env.example` (modified — add `NEXO_AUTO_MIGRATE`)

**Analog:** existing `.env.example` entries (already modified in `git status`). Append:
```
# Schema guard: si está en true y al arrancar faltan tablas en nexo.*,
# las crea automáticamente (NexoBase.metadata.create_all).
# SOLO dev / primer deploy. NUNCA en producción.
NEXO_AUTO_MIGRATE=false
```

**Landmine #4 in RESEARCH:** docs must include "when you change .env, use `docker compose up -d --force-recreate web`" — add as comment or in the commit message.

---

## Shared Patterns

### Authentication / Permission gate
**Source:** `nexo/services/auth.py:225-261` (`require_permission(permission)`)
**Apply to:** All modified routers (every router already uses it — pattern preserved after refactor)
```python
router = APIRouter(
    prefix="/capacidad",
    tags=["capacidad"],
    dependencies=[Depends(require_permission("capacidad:read"))],
)

_edit = [Depends(require_permission("capacidad:edit"))]  # for mutating endpoints
```
Mutating endpoints layer `_edit` in their decorator: `@router.put("", dependencies=_edit)` (existing pattern at `api/routers/recursos.py:34`, `api/routers/ciclos.py:48`).

### Error handling (router level)
**Source:** `api/routers/recursos.py:82-85`, `api/routers/ciclos.py:128-131`, `api/routers/bbdd.py:79-80`
**Apply to:** All modified routers with outbound DB calls that can fail externally (MES reachability, etc.)
```python
try:
    maquinas = mes_service.discover_resources()
except Exception as exc:
    raise HTTPException(502, f"Error conectando a IZARO: {exc}")
```
Global fallback in `api/main.py:66-94` (`global_exception_handler`) swallows unexpected exceptions → error_id in response, full traceback in logs.

### Repository shape (session injected, no commit, DTO out)
**Source:** RESEARCH §Pattern 1 + existing `nexo/services/auth.py:95-171` (shows session-injected pattern without repo wrapper)
**Apply to:** All repos in `nexo/data/repositories/{mes,app,nexo}.py`
```python
class FooRepo:
    def __init__(self, db: Session):   # or Engine for MES
        self._db = db

    # Reads return DTO or list[DTO]
    def get_by_id(self, id: int) -> FooRow | None: ...
    def list_all(self) -> list[FooRow]: ...

    # Writes return DTO — caller commits
    def create(self, *, f1, f2) -> FooRow: ...
```

### Dependency injection with `Depends` + `Annotated` alias
**Source:** `api/routers/historial.py:19` (`db: Session = Depends(get_db)`) and RESEARCH §Repository + DTO Patterns §Dependency factories lines 494-520
**Apply to:** All modified routers via new `api/deps.py` aliases `DbApp`, `DbNexo`, `EngineMes`

### Sessions generator pattern
**Source:** `api/database.py:281-287` (get_db), `api/routers/auditoria.py:46-51` (get_nexo_db)
**Apply to:** `api/deps.py` new factories — same structure with `try/yield/finally: db.close()`

### Test skipif pattern for Postgres-dependent tests
**Source:** `tests/auth/test_rbac_smoke.py:50-60`, `tests/auth/test_audit_append_only.py:44-54`
**Apply to:** All `tests/data/test_*_repository.py` and `test_schema_guard.py`
```python
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _postgres_reachable(),
        reason="Postgres no disponible — requiere `docker compose up -d db`",
    ),
]
```

### Test cleanup with owner engine (IDENT-06 compatibility)
**Source:** `tests/auth/test_audit_append_only.py:60-97` (`_owner_engine()` + `_clean_test_rows` fixture)
**Apply to:** `tests/data/test_nexo_repository.py` tests that exercise `AuditRepo.append` and need to clean rows. `nexo_app` cannot DELETE on audit_log — requires owner engine for cleanup.

### Marker registration
**Source:** `tests/conftest.py:9-15`
**Apply to:** Inherits automatically — `tests/data/` tests use `@pytest.mark.integration` already registered globally.

### Pydantic v2 frozen DTO config
**Source:** RESEARCH §Pydantic v2 DTO Recipe lines 571-595
**Apply to:** All files in `nexo/data/dto/`
```python
model_config = ConfigDict(frozen=True, from_attributes=True)
```

### Shim pattern for Phase 3 transition
**Source:** none in repo (new); RESEARCH §Shim lines 453-457, §Pitfall 5 lines 1214-1224
**Apply to:** `nexo/db/engine.py` (replaced in 03-01) and `nexo/db/models.py` (replaced in 03-03)
```python
"""SHIM — see nexo/data/X for real location. Remove in ..."""
from nexo.data.X import A, B  # noqa: F401
```
No module-level aliases beyond the re-export (breaks monkeypatch tests otherwise).

### Settings access
**Source:** `api/config.py:1-120` (Settings class with AliasChoices for NEXO_*/OEE_* dual envs)
**Apply to:** New `nexo/data/engines.py` — reads `settings.mes_*`, `settings.app_*`, `settings.pg_*`, and critically `settings.effective_pg_user`/`effective_pg_password` (Landmine #12).

---

## No Analog Found

Files with no close codebase match — planner uses RESEARCH.md patterns directly:

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `nexo/data/sql/loader.py` | utility | file-I/O cached | `lru_cache` + `importlib.resources` pattern is new for this repo (RESEARCH §Pattern 2 / §Ref 2) |
| `tests/data/test_sql_loader.py` | unit | file-I/O | No test exercises `importlib.resources` today |
| `tests/data/test_no_raw_pyodbc_in_routers.py` | meta-test | filesystem grep | New pattern (RESEARCH §Validation Architecture lines 1390-1404) |
| `tests/data/test_no_three_part_names.py` | meta-test | filesystem grep | Same as above |
| `nexo/db/engine.py` (shim) + `nexo/db/models.py` (shim) | re-export shim | n/a | No existing shim in the repo |
| `scripts/gen_pdf_reference.py` + `scripts/pdf_regression_check.py` | standalone scripts | batch | PDF hashing gate has no precedent; RESEARCH §Ref 10 lines 1604-1663 is the template |
| `nexo/data/dto/app.py`, `mes.py`, `nexo.py` | Pydantic frozen DTOs | transform | `api/models.py` uses Pydantic but for request bodies, not frozen row DTOs — different direction. Follow RESEARCH §Pydantic v2 DTO Recipe lines 561-648 |

---

## Metadata

**Analog search scope:**
- `nexo/` (engine.py, models.py, services/auth.py) — Phase 2 foundation
- `api/database.py`, `api/main.py`, `api/deps.py`, `api/config.py` — core infra
- `api/routers/{centro_mando,capacidad,operarios,luk4,bbdd,recursos,ciclos,historial,auditoria,usuarios}.py` — refactor targets
- `api/services/{db.py,pipeline.py}` — legacy service layer
- `tests/auth/{conftest-less,test_audit_append_only,test_rbac_smoke}.py` — test fixture patterns
- `tests/conftest.py` — global marker registration
- `Makefile`, `scripts/*.py`, `.env.example` — build + config surfaces
- `OEE/db/connector.py` (header read) — confirmed shape of wrapped module for D-04

**Files scanned:** ~25 source files + 3 test files + RESEARCH + CONTEXT

**Pattern extraction date:** 2026-04-19

---

## PATTERN MAPPING COMPLETE

**Phase:** 03 — Capa de datos
**Files classified:** 47
**Analogs found:** 44 / 47

### Coverage
- Files with exact analog: 31 (routers-to-self, engine extensions, ORM migrations, shared infra patterns)
- Files with role-match / partial analog: 13 (repositories from inline ORM, DTOs from request schemas, tests from auth tests)
- Files with no analog: 7 (SQL loader, grep-based meta-tests, shims, PDF regression scripts)

### Key Patterns Identified
- Repository = session injected, no commit, DTO out — pure extraction from `nexo/services/auth.py` and inline router ORM (recursos/ciclos/historial).
- `engine_mes` with `DATABASE=dbizaro` in DSN is the DATA-09 mechanic — 2-part SQL names (`admuser.fmesmic`) work automatically once the engine's connection default is `dbizaro`.
- Shim pattern for transitional Phase 3 imports: `nexo/db/engine.py` and `nexo/db/models.py` become thin re-exports; consumers migrate at their own pace within 03-03.
- Test skipif pattern established by `tests/auth/*` (Postgres reachable probe + `pytestmark`) transfers verbatim to `tests/data/conftest.py`.
- `AuditRepo.append` MUST NOT commit — IDENT-06 gate (tested in `tests/auth/test_audit_append_only.py`) guarantees DB-level defense if the pattern slips.
- SQL loader + `bindparam(expanding=True)` is the only canonical 2.0 pattern for IN-clauses — replaces the string-interpolation `ct0,ct1,...` in `api/routers/centro_mando.py:47-48`.
- `effective_pg_user`/`effective_pg_password` (not `pg_user`) is mandatory in the new `engine_nexo` or IDENT-06 breaks silently (Landmine #12).

### File Created
`/home/eeguskiza/analisis_datos/.planning/phases/03-capa-de-datos/03-PATTERNS.md`

### Ready for Planning
Pattern mapping complete. Planner can cite the concrete analog file + line range per task action, reference the DATA-09 before/after diff for `centro_mando.py`, and reuse shared patterns (auth gate, session DI, test skipif) without reinventing them.
