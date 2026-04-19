"""Fixtures comunes para ``tests/data/`` (Plan 03-01, DATA-10).

Expone tres fixtures:

- ``db_nexo``: ``Session`` de Postgres ``nexo.*`` con rollback al final.
  Skip automático si el Postgres del compose no está arriba (local dev
  sin ``make up``, CI sin servicio db).
- ``db_app``: ``Session`` SQL Server ``ecs_mobility``. Skip si SQL
  Server no es alcanzable (esperable en CI sin la instancia real).
- ``engine_mes_mock``: ``MagicMock`` que reemplaza
  ``nexo.data.engines.engine_mes`` vía ``monkeypatch.setattr``. Tests
  MES se ejecutan contra mocks en CI (no queremos dependencia del SQL
  Server real).

Los helpers ``_postgres_reachable`` y ``_mssql_reachable`` se exportan
para que ``test_schema_guard.py`` aplique el skipif a nivel módulo
(patrón copiado de ``tests/auth/test_rbac_smoke.py``).
"""
from __future__ import annotations

from typing import Iterator
from unittest.mock import MagicMock

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from nexo.data.engines import (
    SessionLocalApp,
    SessionLocalNexo,
    engine_app,
    engine_nexo,
)


def _postgres_reachable() -> bool:
    """True si el Postgres del compose está arriba y aceptando queries."""
    try:
        with engine_nexo.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def _mssql_reachable() -> bool:
    """True si el SQL Server ``ecs_mobility`` es alcanzable.

    En CI normalmente devuelve ``False`` (no hay SQL Server); los tests
    que lo consumen hacen ``pytest.skip`` entonces.
    """
    try:
        with engine_app.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.fixture
def db_nexo() -> Iterator[Session]:
    """Session Postgres ``nexo``. Rollback al final (no truncate).

    Skip si Postgres no está arriba (``docker compose up -d db``).
    """
    if not _postgres_reachable():
        pytest.skip("Postgres no arriba: docker compose up -d db")
    db = SessionLocalNexo()
    try:
        yield db
    finally:
        db.rollback()
        db.close()


@pytest.fixture
def db_app() -> Iterator[Session]:
    """Session SQL Server ``ecs_mobility``. Skip si no hay SQL Server (CI)."""
    if not _mssql_reachable():
        pytest.skip("SQL Server ecs_mobility no arriba (esperable en CI)")
    db = SessionLocalApp()
    try:
        yield db
    finally:
        db.rollback()
        db.close()


@pytest.fixture
def engine_mes_mock(monkeypatch):
    """Engine MES mockeado.

    Reemplaza ``nexo.data.engines.engine_mes`` con un ``MagicMock`` para
    tests que ejercitan código de repos MES sin necesitar SQL Server.
    """
    mock_engine = MagicMock()
    monkeypatch.setattr("nexo.data.engines.engine_mes", mock_engine)
    return mock_engine
