"""Wave 0 — approvals service + repo state machine tests (QUERY-06).

Partial implementation en 04-01 (cubre creacion defaults). Los tests
de CAS / cancel / expire aterrizan en Plan 04-03 cuando el service
layer consume el ApprovalRepo via routers.
"""
from __future__ import annotations

import pytest

from nexo.data.repositories.nexo import ApprovalRepo

from tests.data.conftest import _postgres_reachable


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _postgres_reachable(),
        reason="Postgres no arriba: docker compose up -d db",
    ),
]


# ── Implemented in 04-01 ───────────────────────────────────────────────────

def test_create_sets_status_pending(db_nexo):
    """create() deja la fila en status='pending' (D-15 invariante)."""
    repo = ApprovalRepo(db_nexo)
    row = repo.create(
        user_id=1,
        endpoint="pipeline/run",
        params_json='{"_test_04_01_pending": true}',
        estimated_ms=5000,
        ttl_days=7,
    )
    try:
        assert row.status == "pending"
    finally:
        from sqlalchemy import delete
        from nexo.data.models_nexo import NexoQueryApproval
        db_nexo.execute(
            delete(NexoQueryApproval).where(NexoQueryApproval.id == row.id)
        )
        db_nexo.commit()


def test_create_sets_ttl_days_default_7(db_nexo):
    """D-14: TTL default = 7 dias cuando no se pasa explicitamente."""
    repo = ApprovalRepo(db_nexo)
    row = repo.create(
        user_id=1,
        endpoint="pipeline/run",
        params_json='{"_test_04_01_ttl": true}',
        estimated_ms=5000,
        # ttl_days omitido → default 7
    )
    try:
        assert row.ttl_days == 7, (
            f"TTL default debe ser 7 por D-14, got {row.ttl_days}"
        )
    finally:
        from sqlalchemy import delete
        from nexo.data.models_nexo import NexoQueryApproval
        db_nexo.execute(
            delete(NexoQueryApproval).where(NexoQueryApproval.id == row.id)
        )
        db_nexo.commit()


# ── Deferred to Plan 04-03 ─────────────────────────────────────────────────

@pytest.mark.skip(reason="CAS race test implementado en Plan 04-03")
def test_consume_cas_race(db_nexo):
    """Dos consume() concurrentes con mismo approval → solo uno gana (T-04-01-03)."""
    raise NotImplementedError("Plan 04-03")


@pytest.mark.skip(reason="Cancel ownership check implementado en Plan 04-03")
def test_cancel_wrong_user_returns_false(db_nexo):
    """cancel() devuelve False si user_id != row.user_id (D-16)."""
    raise NotImplementedError("Plan 04-03")


@pytest.mark.skip(reason="expire_stale test implementado en Plan 04-03")
def test_expire_stale_moves_pending_to_expired(db_nexo):
    """expire_stale(cutoff) marca pending old rows como expired (D-14)."""
    raise NotImplementedError("Plan 04-03")
