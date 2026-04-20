"""3 engines + 2 session factories (Phase 3, plan 03-01).

- ``engine_nexo``: Postgres schema ``nexo`` (movido desde
  ``nexo/db/engine.py``). Usa ``settings.effective_pg_user`` /
  ``effective_pg_password`` — rol ``nexo_app`` cuando está configurado
  (Plan 02-04 / gate IDENT-06).
- ``engine_app``: SQL Server ``ecs_mobility`` (re-export desde
  ``api/database.py``; no se duplica el ``_mssql_creator`` pyodbc —
  Landmine #3 del RESEARCH).
- ``engine_mes``: SQL Server ``dbizaro`` read-only por convención
  (DATA-01, DATA-11). ``DATABASE=dbizaro`` en la connection string
  habilita el kill de 3-part names ``dbizaro.admuser.*`` (DATA-09 del
  plan siguiente).

Consumidores externos deben importar desde este módulo. El shim en
``nexo/db/engine.py`` re-exporta ``engine_nexo`` y ``SessionLocalNexo``
para compatibilidad con Phase 2.
"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from api.config import settings


# ── engine_nexo (Postgres, schema nexo) ──────────────────────────────────
# Copiado verbatim de nexo/db/engine.py. Mantener settings.effective_pg_user/
# effective_pg_password (Landmine #12 — rompe IDENT-06 si se usa pg_user a
# secas, el test tests/auth/test_audit_append_only.py fallaría silenciosamente).
def _build_pg_dsn() -> str:
    user = settings.effective_pg_user
    password = settings.effective_pg_password
    return (
        f"postgresql+psycopg2://{user}:{password}"
        f"@{settings.pg_host}:{settings.pg_port}/{settings.pg_db}"
    )


engine_nexo: Engine = create_engine(
    _build_pg_dsn(),
    pool_size=5,
    max_overflow=5,
    pool_timeout=10,
    pool_recycle=1800,
    pool_pre_ping=True,
    echo=settings.log_sql,
)

SessionLocalNexo = sessionmaker(
    bind=engine_nexo,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


# ── engine_app (SQL Server ecs_mobility) — RE-EXPORT ─────────────────────
# NO duplicar _mssql_creator. El engine legacy vive en api/database.py y el
# pipeline OEE lo consume tal cual; aquí solo lo exponemos con el nombre
# nuevo. Si alguna vez duplicamos la connection string nos arriesgamos a
# que ambas versiones diverjan silenciosamente.
from api.database import SessionLocal as SessionLocalApp  # noqa: E402
from api.database import engine as engine_app  # noqa: E402


# ── engine_mes (SQL Server dbizaro, read-only convención) ────────────────
# DATA-11 pool params; DATA-01 DSN dedicado. La clave de DATA-09 es que
# el ``DATABASE={settings.mes_db}`` fijado aquí permite que las queries
# escriban ``admuser.fmesmic`` (2-part name) en vez de
# ``dbizaro.admuser.fmesmic``.
def _build_mes_dsn() -> str:
    # Escape de '+' en el password (ODBC lo consume como encoded param).
    pwd = settings.mes_password.replace("+", "%2B")
    return (
        f"mssql+pyodbc://{settings.mes_user}:{pwd}"
        f"@{settings.mes_server}:{settings.mes_port}/{settings.mes_db}"
        "?driver=ODBC+Driver+18+for+SQL+Server"
        "&TrustServerCertificate=yes&Encrypt=yes"
    )


engine_mes: Engine = create_engine(
    _build_mes_dsn(),
    pool_pre_ping=True,        # DATA-11
    pool_recycle=3600,         # DATA-11
    pool_size=3,
    max_overflow=2,
    pool_timeout=15,
    connect_args={"timeout": 15},  # DATA-11 (connect_timeout segundos)
)


__all__ = [
    "engine_nexo",
    "engine_app",
    "engine_mes",
    "SessionLocalNexo",
    "SessionLocalApp",
]
