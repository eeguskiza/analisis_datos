"""SQLAlchemy engine + session factory para Postgres (schema ``nexo``).

Engine dedicado, separado del engine SQL Server de ``api/database.py``.
No importa ese módulo — coexisten como dos conexiones independientes.

Sprint 2 (Phase 3) refactorizará la capa de datos introduciendo
``engine_mes`` (dbizaro read-only) y ``engine_app`` (ecs_mobility)
explícitos. Hoy sólo existen ``engine_nexo`` (Postgres) y el engine
legacy de ``api/database.py`` (SQL Server).
"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from api.config import settings


def _build_dsn() -> str:
    """DSN para el engine que usa la app en runtime.

    Prefiere ``NEXO_PG_APP_USER`` / ``NEXO_PG_APP_PASSWORD`` (rol ``nexo_app``
    creado en Plan 02-04, con GRANTs limitados que impiden UPDATE/DELETE
    en ``nexo.audit_log``). Cae a ``NEXO_PG_USER``/``NEXO_PG_PASSWORD`` (el
    owner ``oee``) si el par APP no esta definido — backwards compat con
    deploys anteriores al plan 02-04.
    """
    user = settings.effective_pg_user
    password = settings.effective_pg_password
    return (
        f"postgresql+psycopg2://{user}:{password}"
        f"@{settings.pg_host}:{settings.pg_port}/{settings.pg_db}"
    )


# Parámetros de pool (research §Pattern 6). Coexiste con el engine de
# SQL Server sin conflicto (pools independientes por engine).
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
