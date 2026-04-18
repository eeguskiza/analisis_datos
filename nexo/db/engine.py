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
    return (
        f"postgresql+psycopg2://{settings.pg_user}:{settings.pg_password}"
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
