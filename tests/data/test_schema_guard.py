"""DATA-06: ``schema_guard.verify`` — comportamiento.

Nota: inyectamos ``critical_tables`` como kwarg en lugar de
monkeypatchear la constante del módulo. Esto evita fragilidad si
``CRITICAL_TABLES`` se convierte en ``frozenset``/tuple computada en
import time.

Todos los tests son ``integration`` — requieren Postgres arriba. Se
skipean en CI sin ``docker compose up -d db``.
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


def test_verify_ok_with_all_tables(caplog):
    """Las 8 tablas críticas existen → None + log INFO 'OK'."""
    with caplog.at_level("INFO", logger="nexo.schema_guard"):
        schema_guard.verify(engine_nexo)
    # El log INFO debe mencionar "OK".
    assert any("OK" in r.message for r in caplog.records), (
        "schema_guard debe loguear INFO con 'OK' cuando todas las tablas existen"
    )


def test_verify_raises_when_table_missing(monkeypatch):
    """Tabla ficticia faltante + auto_migrate=false → ``RuntimeError``.

    Inyectamos ``critical_tables`` como kwarg (parámetro público). NO
    monkeypatch de ``schema_guard.CRITICAL_TABLES`` — eso es frágil si
    la constante se calcula al importar.
    """
    monkeypatch.delenv("NEXO_AUTO_MIGRATE", raising=False)
    with pytest.raises(RuntimeError) as exc_info:
        schema_guard.verify(
            engine_nexo,
            critical_tables=("users", "__nonexistent_table__"),
        )
    assert "__nonexistent_table__" in str(exc_info.value), (
        "El mensaje debe mencionar la tabla faltante"
    )


def test_auto_migrate_creates_missing(monkeypatch, caplog):
    """``NEXO_AUTO_MIGRATE=true`` + tabla faltante → ``create_all`` + log WARNING.

    Igual que el test anterior: ``critical_tables`` kwarg, no monkeypatch.
    Spyeamos ``NexoBase.metadata.create_all`` para verificar la llamada
    sin ejecutar DDL real.
    """
    monkeypatch.setenv("NEXO_AUTO_MIGRATE", "true")

    called = {"create_all": False}

    def fake_create_all(bind):  # noqa: ARG001
        called["create_all"] = True

    monkeypatch.setattr(
        "nexo.data.schema_guard.NexoBase.metadata.create_all",
        fake_create_all,
    )
    with caplog.at_level("WARNING", logger="nexo.schema_guard"):
        schema_guard.verify(
            engine_nexo,
            critical_tables=("users", "__nonexistent_table__"),
        )
    assert called["create_all"] is True, (
        "create_all debería haberse ejecutado cuando NEXO_AUTO_MIGRATE=true"
    )
    # Log WARNING debe mencionar auto-migrate / NEXO_AUTO_MIGRATE.
    assert any(
        "auto-migra" in r.message.lower() or "NEXO_AUTO_MIGRATE" in r.message
        for r in caplog.records
    ), "schema_guard debe loguear WARNING documentando la auto-migración"
