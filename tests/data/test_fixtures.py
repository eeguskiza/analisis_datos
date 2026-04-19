"""DATA-10: meta-test — las fixtures de ``conftest.py`` funcionan.

Verifica que:

- ``db_nexo`` yieldea una ``Session`` (o skipea si Postgres no está
  arriba).
- ``engine_mes_mock`` sustituye ``engine_mes`` con un ``MagicMock``.
- ``db_app`` yieldea una ``Session`` o skipea (esperable en CI sin
  SQL Server).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from sqlalchemy.orm import Session


def test_db_nexo_fixture_yields_session(db_nexo):
    """``db_nexo`` yieldea una ``Session`` (skip si Postgres no arriba)."""
    assert isinstance(db_nexo, Session)


def test_engine_mes_mock_is_magicmock(engine_mes_mock):
    """``engine_mes_mock`` sustituye ``nexo.data.engines.engine_mes``."""
    assert isinstance(engine_mes_mock, MagicMock)


def test_db_app_fixture_optional(db_app):
    """``db_app`` skip si no hay SQL Server (CI); ``Session`` si lo hay."""
    assert isinstance(db_app, Session)
