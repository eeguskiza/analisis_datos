"""Integration tests para la capa ORM/repo de Phase 4 — Plan 04-01.

Cubre QUERY-01 + QUERY-02 + QUERY-06 (schema + repos). Los tests
assume que ``scripts/init_nexo_schema.py`` ha corrido y las 3 tablas
nuevas + los 4 seeds de ``query_thresholds`` estan presentes.

Skipif automatico si Postgres no esta arriba (mismo patron que
``tests/data/test_nexo_repository.py``).

Contract test T-04-01-01 (paralelo a T-03-03-01 de Phase 3):
``QueryLogRepo.append`` NO debe contener ``.commit()`` ni ``.flush()``
en su source — caller orquesta la transaccion.
"""
from __future__ import annotations

import inspect

import pytest

from nexo.data.dto.query import (
    QueryApprovalRow,
    QueryLogRow,
    QueryThresholdRow,
)
from nexo.data.repositories.nexo import (
    ApprovalRepo,
    QueryLogRepo,
    ThresholdRepo,
)

from tests.data.conftest import _postgres_reachable


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _postgres_reachable(),
        reason="Postgres no arriba: docker compose up -d db",
    ),
]


# ── Contract tests (no requieren DB) ───────────────────────────────────────

def test_query_log_repo_append_source_has_no_commit():
    """Contract test T-04-01-01: QueryLogRepo.append NO comitea.

    Paralelo a test_audit_repo_append_source_has_no_commit (Phase 3).
    El caller (middleware query_timing — Plan 04-02) orquesta la
    transaccion.
    """
    src = inspect.getsource(QueryLogRepo.append)
    assert ".commit()" not in src, f"QueryLogRepo.append contiene .commit(): {src}"
    assert ".flush()" not in src, f"QueryLogRepo.append contiene .flush(): {src}"
    assert "db.commit" not in src, f"QueryLogRepo.append contiene db.commit: {src}"


# ── QueryLogRepo ────────────────────────────────────────────────────────────

def test_query_log_repo_append_adds_to_session(db_nexo):
    """append() añade fila a session.new sin comitear."""
    repo = QueryLogRepo(db_nexo)
    before_new = len(list(db_nexo.new))
    repo.append(
        user_id=None,
        ip="127.0.0.1",
        endpoint="pipeline/run",
        params_json='{"test": true}',
        estimated_ms=1000,
        actual_ms=1500,
        rows=None,
        status="ok",
        approval_id=None,
    )
    after_new = len(list(db_nexo.new))
    assert after_new == before_new + 1, (
        f"Esperado +1 en session.new, got before={before_new} after={after_new}"
    )
    assert db_nexo.in_transaction(), (
        "La sesion salio de la transaccion — QueryLogRepo.append comiteo implicitamente"
    )


# ── ThresholdRepo ───────────────────────────────────────────────────────────

def test_threshold_repo_list_all_returns_dtos(db_nexo):
    """list_all devuelve QueryThresholdRow e incluye los 4 seeds iniciales."""
    repo = ThresholdRepo(db_nexo)
    rows = repo.list_all()
    assert isinstance(rows, list)
    assert all(isinstance(r, QueryThresholdRow) for r in rows)
    endpoints = {r.endpoint for r in rows}
    # D-01/D-02/D-03: 4 seeds de init_nexo_schema.py
    assert endpoints.issuperset({
        "pipeline/run",
        "bbdd/query",
        "capacidad",
        "operarios",
    }), f"Esperados 4 seeds, got: {endpoints}"


def test_threshold_repo_get_returns_dto_or_none(db_nexo):
    """get devuelve QueryThresholdRow para endpoints seed, None para unknown."""
    repo = ThresholdRepo(db_nexo)
    # Endpoint valido del seed D-01
    row = repo.get("pipeline/run")
    assert isinstance(row, QueryThresholdRow)
    assert row.warn_ms == 120_000, f"Esperado warn_ms=120000 (D-01), got {row.warn_ms}"
    assert row.block_ms == 600_000, f"Esperado block_ms=600000 (D-01), got {row.block_ms}"
    # Endpoint desconocido
    unknown = repo.get("__no_existe__")
    assert unknown is None, f"Expected None for unknown endpoint, got {unknown!r}"


# ── ApprovalRepo ────────────────────────────────────────────────────────────

def test_approval_repo_create_returns_row_with_id(db_nexo):
    """create() devuelve ORM row con id poblado + status='pending'.

    Cleanup: el ``db_nexo`` fixture hace rollback al final, pero create()
    hace commit inline, asi que purgamos la fila explicitamente al
    terminar para no contaminar otros runs.
    """
    repo = ApprovalRepo(db_nexo)
    row = repo.create(
        user_id=1,  # Propietario suele ser id=1 tras bootstrap
        endpoint="pipeline/run",
        params_json='{"_test_04_01": true}',
        estimated_ms=5000,
        ttl_days=7,
    )
    try:
        assert row.id is not None, "id debe estar poblado tras commit/refresh"
        assert row.status == "pending", f"Status default debe ser pending, got {row.status}"
        assert row.ttl_days == 7, f"ttl_days default debe ser 7, got {row.ttl_days}"
    finally:
        # Cleanup: borramos la fila de test (bypaseamos el rollback
        # porque create() comiteo).
        from sqlalchemy import delete
        from nexo.data.models_nexo import NexoQueryApproval
        db_nexo.execute(
            delete(NexoQueryApproval).where(NexoQueryApproval.id == row.id)
        )
        db_nexo.commit()


def test_approval_repo_consume_returns_none_if_status_not_approved(db_nexo):
    """consume() con status='pending' devuelve None (CAS no matchea).

    Verifica el invariante "solo approved puede consumirse" (D-15 step 2).
    """
    repo = ApprovalRepo(db_nexo)
    row = repo.create(
        user_id=1,
        endpoint="pipeline/run",
        params_json='{"_test_consume_pending": true}',
        estimated_ms=5000,
        ttl_days=7,
    )
    try:
        result = repo.consume(
            approval_id=row.id,
            user_id=1,
            current_params_json='{"_test_consume_pending": true}',
        )
        assert result is None, (
            f"consume debe devolver None si status!='approved', got {result!r}"
        )
        # Sanity: la fila sigue pending (CAS no actualizo nada)
        refetch = repo.get(row.id)
        assert refetch is not None
        assert refetch.status == "pending", (
            f"Status debe seguir pending tras consume fallido, got {refetch.status}"
        )
    finally:
        from sqlalchemy import delete
        from nexo.data.models_nexo import NexoQueryApproval
        db_nexo.execute(
            delete(NexoQueryApproval).where(NexoQueryApproval.id == row.id)
        )
        db_nexo.commit()


# ── DTO roundtrip smoke test ────────────────────────────────────────────────

def test_query_log_row_dto_is_frozen():
    """QueryLogRow es frozen — mutation lanza ValidationError."""
    from datetime import datetime, timezone
    row = QueryLogRow(
        id=1,
        ts=datetime.now(timezone.utc),
        user_id=None,
        endpoint="pipeline/run",
        params_json=None,
        estimated_ms=100,
        actual_ms=150,
        rows=None,
        status="ok",
        approval_id=None,
        ip=None,
    )
    with pytest.raises(Exception):  # Pydantic v2: ValidationError
        row.endpoint = "other"  # type: ignore[misc]


def test_query_approval_row_dto_is_frozen():
    """QueryApprovalRow es frozen — mutation lanza ValidationError."""
    from datetime import datetime, timezone
    row = QueryApprovalRow(
        id=1,
        user_id=1,
        endpoint="pipeline/run",
        params_json="{}",
        estimated_ms=5000,
        status="pending",
        created_at=datetime.now(timezone.utc),
        ttl_days=7,
    )
    with pytest.raises(Exception):
        row.status = "approved"  # type: ignore[misc]
