"""Integration tests para schema_guard extendido a 11 tablas — Plan 04-01.

Phase 4 extiende ``CRITICAL_TABLES`` de 8 a 11 (append query_log,
query_thresholds, query_approvals). Este archivo verifica que:

1. ``verify()`` acepta las 11 tablas cuando todas existen.
2. ``verify()`` lanza ``RuntimeError`` si falta cualquiera de las 3 nuevas.

Estrategia de aislamiento: inyectamos ``critical_tables`` como kwarg
(mismo patron que test_schema_guard.py en Phase 3) — NO monkeypatch de
``schema_guard.CRITICAL_TABLES`` para no depender de como este
estructurada la constante.
"""
from __future__ import annotations

import pytest

from nexo.data import schema_guard
from nexo.data.engines import engine_nexo

from tests.data.conftest import _postgres_reachable


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _postgres_reachable(),
        reason="Postgres no arriba: docker compose up -d db",
    ),
]


def test_critical_tables_is_11_entries():
    """CRITICAL_TABLES tiene 11 entradas (8 Phase 2 + 3 Phase 4)."""
    assert len(schema_guard.CRITICAL_TABLES) == 11, (
        f"Esperado 11 tablas criticas, got {len(schema_guard.CRITICAL_TABLES)}: "
        f"{schema_guard.CRITICAL_TABLES}"
    )
    for name in ("query_log", "query_thresholds", "query_approvals"):
        assert name in schema_guard.CRITICAL_TABLES, (
            f"{name!r} debe estar en CRITICAL_TABLES"
        )


def test_verify_accepts_all_11_tables_when_present(caplog):
    """Cuando las 11 tablas existen (post init_nexo_schema), verify no lanza."""
    with caplog.at_level("INFO", logger="nexo.schema_guard"):
        schema_guard.verify(engine_nexo)
    assert any("OK" in r.message for r in caplog.records), (
        "schema_guard debe loguear INFO con 'OK' cuando las 11 tablas existen"
    )


def test_verify_raises_when_query_log_missing(monkeypatch):
    """Tabla ficticia con nombre query_log-like → RuntimeError mencionandola."""
    monkeypatch.delenv("NEXO_AUTO_MIGRATE", raising=False)
    with pytest.raises(RuntimeError) as exc_info:
        schema_guard.verify(
            engine_nexo,
            critical_tables=("users", "__missing_query_log_sim__"),
        )
    assert "__missing_query_log_sim__" in str(exc_info.value), (
        "El mensaje debe mencionar la tabla faltante"
    )


def test_verify_raises_when_query_approvals_missing(monkeypatch):
    """Mismo patron para query_approvals missing."""
    monkeypatch.delenv("NEXO_AUTO_MIGRATE", raising=False)
    with pytest.raises(RuntimeError) as exc_info:
        schema_guard.verify(
            engine_nexo,
            critical_tables=("users", "__missing_query_approvals_sim__"),
        )
    assert "__missing_query_approvals_sim__" in str(exc_info.value)
