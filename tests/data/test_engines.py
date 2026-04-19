"""DATA-01 + DATA-11: engines + pool config.

Tests cubiertos:

- DATA-01: los 3 engines están exportados y son ``Engine``; las 2
  session factories existen.
- DATA-11: ``engine_mes`` tiene ``pool_recycle=3600`` y
  ``pool_pre_ping=True``.
- DATA-09 prep: DSN de ``engine_mes`` tiene ``DATABASE=dbizaro`` (kill
  de 3-part names en 03-02).
- DATA-09 integration: smoke contra SQL Server real que comprueba
  ``SELECT DB_NAME() = 'dbizaro'`` (marcado ``integration``, skip si no
  hay SQL Server disponible).
"""
from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

from api.config import settings
from nexo.data.engines import (
    SessionLocalApp,
    SessionLocalNexo,
    engine_app,
    engine_mes,
    engine_nexo,
)


def test_three_engines_exported():
    """DATA-01: ``engine_nexo``, ``engine_app``, ``engine_mes`` son ``Engine``."""
    for e in (engine_nexo, engine_app, engine_mes):
        assert isinstance(e, Engine)


def test_two_session_factories_exported():
    """DATA-01: session factories para nexo y app existen."""
    assert SessionLocalNexo is not None
    assert SessionLocalApp is not None


def test_mes_pool_config():
    """DATA-11: ``pool_recycle=3600``, ``pool_pre_ping=True``."""
    assert engine_mes.pool._recycle == 3600, (
        f"engine_mes pool_recycle esperado 3600, got {engine_mes.pool._recycle}"
    )
    assert engine_mes.pool._pre_ping is True, (
        "engine_mes pool_pre_ping debe ser True"
    )


def test_mes_url_has_mes_db_as_default_catalog():
    """DATA-09 prep: DSN tiene ``DATABASE={mes_db}`` (kill 3-part names en 03-02)."""
    assert engine_mes.url.database == settings.mes_db, (
        f"engine_mes debe apuntar a {settings.mes_db}, got {engine_mes.url.database}"
    )


@pytest.mark.integration
def test_engine_mes_database_is_dbizaro():
    """DATA-09: smoke real contra SQL Server — skip si no hay servidor."""
    try:
        with engine_mes.connect() as conn:
            db = conn.execute(text("SELECT DB_NAME()")).scalar()
    except Exception as exc:
        pytest.skip(f"SQL Server no alcanzable: {exc}")
    assert db == "dbizaro", f"engine_mes default catalog != dbizaro (got {db})"
