"""Fixtures compartidos para tests OEE / Nexo."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

# Asegurar que el proyecto esta en el path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexo.data.engines import (  # noqa: E402
    SessionLocalApp,
    SessionLocalNexo,
    engine_app,
    engine_nexo,
)


def pytest_configure(config):
    """Registra markers custom para evitar PytestUnknownMarkWarning."""
    config.addinivalue_line(
        "markers",
        "integration: tests que requieren infraestructura live (BD Postgres, "
        "servicios externos). Filtrables con `pytest -m 'not integration'`.",
    )


# ── Reachability helpers (Plan 03-01, extendidos en Plan 04-01) ───────────

def _postgres_reachable() -> bool:
    """True si el Postgres del compose esta arriba y aceptando queries.

    Movido desde ``tests/data/conftest.py`` en Plan 04-01 para que los
    suites de ``tests/services/`` y ``tests/routers/`` puedan aplicar el
    mismo skipif que ``tests/data/``. ``tests/data/conftest.py`` sigue
    re-exportando el simbolo para no romper los imports existentes.
    """
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


# ── DB session fixtures (Plan 03-01, elevados a root en Plan 04-01) ───────

@pytest.fixture
def db_nexo() -> Iterator[Session]:
    """Session Postgres ``nexo``. Rollback al final (no truncate).

    Skip si Postgres no esta arriba (``docker compose up -d db``).
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
    tests que ejercitan codigo de repos MES sin necesitar SQL Server.
    """
    mock_engine = MagicMock()
    monkeypatch.setattr("nexo.data.engines.engine_mes", mock_engine)
    return mock_engine


# ── Phase 4 / Plan 04-01 — thresholds_cache fixtures ──────────────────────

@pytest.fixture
def thresholds_cache_empty():
    """Resetea el cache global de ``nexo.services.thresholds_cache``.

    Util para tests que quieren asegurar que el cache arranca vacio o
    que una llamada a ``get`` fuerza ``full_reload`` por safety-net.

    Yield: el modulo thresholds_cache con cache vaciado.
    Teardown: deja el cache vaciado; el proximo ``get`` en otro test
    volvera a recargar via safety-net.
    """
    from nexo.services import thresholds_cache

    with thresholds_cache._cache_lock:
        thresholds_cache._cache.clear()
    # Reset global loaded_at_global — la unica forma sin referencia
    # es asignar al atributo modulo.
    thresholds_cache._loaded_at_global = None
    yield thresholds_cache
    with thresholds_cache._cache_lock:
        thresholds_cache._cache.clear()
    thresholds_cache._loaded_at_global = None
