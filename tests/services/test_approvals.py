"""Approvals service tests (Plan 04-03 / QUERY-06 / D-15 / D-16).

Reemplaza los stubs de Plan 04-01 con cobertura funcional:

- create: status=pending, ttl_days default 7, ttl_days override.
- consume: happy path, double-consume, wrong user, wrong status,
  params tampering.
- cancel: owner-ok vs other-user vs wrong-status.
- expire_stale: filas viejas pasan a expired, recientes no.

Integration — requiere Postgres up. Los tests crean usuarios temporales
y los limpian en teardown (pattern de Plan 02-03).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterator

import pytest
from fastapi import HTTPException
from sqlalchemy import delete, select, text

from nexo.data.engines import SessionLocalNexo
from nexo.data.models_nexo import NexoQueryApproval, NexoUser
from nexo.data.repositories.nexo import ApprovalRepo
from nexo.services import approvals as svc
from nexo.services.auth import hash_password


def _postgres_reachable() -> bool:
    try:
        db = SessionLocalNexo()
        try:
            db.execute(text("SELECT 1"))
            return True
        finally:
            db.close()
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _postgres_reachable(),
        reason="Postgres no arriba: docker compose up -d db",
    ),
]


TEST_DOMAIN = "@approvals-svc-test.local"
TEST_PASSWORD = "approvalssvctest1234"


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def user_factory() -> Iterator[callable]:
    """Crea users temporales y los limpia al finalizar."""
    created_ids: list[int] = []

    def _create(email_suffix: str) -> NexoUser:
        email = f"{email_suffix}{TEST_DOMAIN}"
        db = SessionLocalNexo()
        try:
            # Reutilizar si existe (idempotente)
            existing = db.execute(
                select(NexoUser).where(NexoUser.email == email)
            ).scalar_one_or_none()
            if existing:
                created_ids.append(existing.id)
                return existing
            u = NexoUser(
                email=email,
                password_hash=hash_password(TEST_PASSWORD),
                role="usuario",
                active=True,
                must_change_password=False,
            )
            db.add(u)
            db.commit()
            db.refresh(u)
            created_ids.append(u.id)
            return u
        finally:
            db.close()

    yield _create

    # Teardown: borra approvals + users creados
    db = SessionLocalNexo()
    try:
        if created_ids:
            db.execute(
                delete(NexoQueryApproval).where(
                    NexoQueryApproval.user_id.in_(created_ids)
                )
            )
            db.execute(
                delete(NexoUser).where(NexoUser.id.in_(created_ids))
            )
            db.commit()
    finally:
        db.close()


# ── CREATE ────────────────────────────────────────────────────────────────

def test_create_sets_status_pending(db_nexo, user_factory):
    """create_approval() deja la fila en status='pending'."""
    u = user_factory("create-pending")
    approval_id = svc.create_approval(
        db_nexo,
        user_id=u.id,
        endpoint="pipeline/run",
        params={"a": 1, "b": 2},
        estimated_ms=5000,
    )
    row = db_nexo.get(NexoQueryApproval, approval_id)
    assert row is not None
    assert row.status == "pending"
    assert row.created_at is not None


def test_create_sets_ttl_days_default_7(db_nexo, user_factory):
    u = user_factory("create-ttl7")
    approval_id = svc.create_approval(
        db_nexo,
        user_id=u.id,
        endpoint="pipeline/run",
        params={"x": "y"},
        estimated_ms=1000,
    )
    row = db_nexo.get(NexoQueryApproval, approval_id)
    assert row.ttl_days == 7


def test_create_ttl_days_override(db_nexo, user_factory):
    u = user_factory("create-ttl3")
    approval_id = svc.create_approval(
        db_nexo,
        user_id=u.id,
        endpoint="pipeline/run",
        params={"x": "z"},
        estimated_ms=1000,
        ttl_days=3,
    )
    row = db_nexo.get(NexoQueryApproval, approval_id)
    assert row.ttl_days == 3


def test_create_canonicalizes_params(db_nexo, user_factory):
    """sort_keys=True en params_json — garantiza equality D-15."""
    u = user_factory("create-canon")
    # Mismo dict con keys en distinto orden → mismo params_json
    aid1 = svc.create_approval(
        db_nexo, user_id=u.id, endpoint="bbdd/query",
        params={"b": 2, "a": 1}, estimated_ms=100,
    )
    aid2 = svc.create_approval(
        db_nexo, user_id=u.id, endpoint="bbdd/query",
        params={"a": 1, "b": 2}, estimated_ms=100,
    )
    r1 = db_nexo.get(NexoQueryApproval, aid1)
    r2 = db_nexo.get(NexoQueryApproval, aid2)
    assert r1.params_json == r2.params_json == '{"a": 1, "b": 2}'


# ── CONSUME ───────────────────────────────────────────────────────────────

def _mk_approved(db, user_id: int, params: dict, endpoint: str = "bbdd/query") -> int:
    """Helper: crea approval + lo aprueba manualmente."""
    aid = svc.create_approval(
        db, user_id=user_id, endpoint=endpoint,
        params=params, estimated_ms=500,
    )
    ApprovalRepo(db).approve(aid, user_id)  # aprobado por el mismo user (shortcut test)
    return aid


def test_consume_success_marks_consumed(db_nexo, user_factory):
    u = user_factory("consume-ok")
    params = {"sql": "SELECT 1", "database": "dbizaro"}
    aid = _mk_approved(db_nexo, u.id, params)
    result = svc.consume_approval(
        db_nexo, approval_id=aid, user_id=u.id, current_params=params,
    )
    assert result is not None
    assert result.consumed_at is not None
    assert result.status == "consumed"


def test_consume_race_second_call_returns_403(db_nexo, user_factory):
    """Double-consume: segunda llamada → HTTPException(403) 'ya consumido'."""
    u = user_factory("consume-race")
    params = {"x": 1}
    aid = _mk_approved(db_nexo, u.id, params)
    svc.consume_approval(
        db_nexo, approval_id=aid, user_id=u.id, current_params=params,
    )
    # Segunda llamada
    with pytest.raises(HTTPException) as excinfo:
        svc.consume_approval(
            db_nexo, approval_id=aid, user_id=u.id, current_params=params,
        )
    assert excinfo.value.status_code == 403
    # Tras el primer consume, status='consumed' → diagnostic
    # "está en estado consumed" (prioritiza status check antes que
    # consumed_at para mensaje más accionable).
    detail_lower = excinfo.value.detail.lower()
    assert "consumed" in detail_lower or "consumido" in detail_lower


def test_consume_wrong_user_returns_403(db_nexo, user_factory):
    uA = user_factory("consume-ua")
    uB = user_factory("consume-ub")
    params = {"k": "v"}
    aid = _mk_approved(db_nexo, uA.id, params)
    with pytest.raises(HTTPException) as excinfo:
        svc.consume_approval(
            db_nexo, approval_id=aid, user_id=uB.id, current_params=params,
        )
    assert excinfo.value.status_code == 403
    assert "otro usuario" in excinfo.value.detail.lower()


def test_consume_wrong_status_returns_403(db_nexo, user_factory):
    """consume sobre status='pending' (no approved) → 403."""
    u = user_factory("consume-wstatus")
    params = {"y": 2}
    aid = svc.create_approval(
        db_nexo, user_id=u.id, endpoint="pipeline/run",
        params=params, estimated_ms=1000,
    )
    # status=pending (no lo aprobamos)
    with pytest.raises(HTTPException) as excinfo:
        svc.consume_approval(
            db_nexo, approval_id=aid, user_id=u.id, current_params=params,
        )
    assert excinfo.value.status_code == 403
    assert "pending" in excinfo.value.detail.lower()


def test_consume_params_tamper_returns_403(db_nexo, user_factory):
    """Params cambian entre solicitud y consumo → 403 'cambiaron'."""
    u = user_factory("consume-tamper")
    aid = _mk_approved(db_nexo, u.id, {"a": 1})
    with pytest.raises(HTTPException) as excinfo:
        svc.consume_approval(
            db_nexo, approval_id=aid, user_id=u.id, current_params={"a": 2},
        )
    assert excinfo.value.status_code == 403
    assert "cambiaron" in excinfo.value.detail.lower() or "parámetros" in excinfo.value.detail.lower()


def test_consume_nonexistent_returns_403(db_nexo, user_factory):
    u = user_factory("consume-404")
    with pytest.raises(HTTPException) as excinfo:
        svc.consume_approval(
            db_nexo, approval_id=99999999, user_id=u.id,
            current_params={"x": 1},
        )
    assert excinfo.value.status_code == 403
    assert "no existe" in excinfo.value.detail.lower()


# ── CANCEL ────────────────────────────────────────────────────────────────

def test_cancel_own_pending_returns_true(db_nexo, user_factory):
    u = user_factory("cancel-own")
    aid = svc.create_approval(
        db_nexo, user_id=u.id, endpoint="pipeline/run",
        params={"m": "n"}, estimated_ms=100,
    )
    ok = svc.cancel(db_nexo, aid, u.id)
    assert ok is True
    row = db_nexo.get(NexoQueryApproval, aid)
    # Refresh (repo commiteó ya)
    db_nexo.refresh(row)
    assert row.status == "cancelled"
    assert row.cancelled_at is not None


def test_cancel_others_returns_false(db_nexo, user_factory):
    uA = user_factory("cancel-a")
    uB = user_factory("cancel-b")
    aid = svc.create_approval(
        db_nexo, user_id=uA.id, endpoint="pipeline/run",
        params={"o": "p"}, estimated_ms=100,
    )
    ok = svc.cancel(db_nexo, aid, uB.id)
    assert ok is False
    row = db_nexo.get(NexoQueryApproval, aid)
    db_nexo.refresh(row)
    assert row.status == "pending"


def test_cancel_non_pending_returns_false(db_nexo, user_factory):
    """cancel sobre approval ya approved → False (no cambia status)."""
    u = user_factory("cancel-approved")
    params = {"t": "u"}
    aid = _mk_approved(db_nexo, u.id, params)
    ok = svc.cancel(db_nexo, aid, u.id)
    assert ok is False
    row = db_nexo.get(NexoQueryApproval, aid)
    db_nexo.refresh(row)
    assert row.status == "approved"


# ── EXPIRE ────────────────────────────────────────────────────────────────

def test_expire_stale_moves_pending_to_expired(db_nexo, user_factory):
    """Crea 3 approvals con created_at artificial; verifica que
    expire_stale(ttl_days=7) marca solo las >7d como expired.
    """
    u = user_factory("expire-stale")
    # Crear 3 approvals pending
    aid_recent = svc.create_approval(
        db_nexo, user_id=u.id, endpoint="pipeline/run",
        params={"age": "1d"}, estimated_ms=100,
    )
    aid_mid = svc.create_approval(
        db_nexo, user_id=u.id, endpoint="pipeline/run",
        params={"age": "5d"}, estimated_ms=100,
    )
    aid_old = svc.create_approval(
        db_nexo, user_id=u.id, endpoint="pipeline/run",
        params={"age": "10d"}, estimated_ms=100,
    )
    # Forzar created_at artificialmente vía UPDATE directo
    now = datetime.now(timezone.utc)
    db_nexo.execute(
        text("UPDATE nexo.query_approvals SET created_at = :ts WHERE id = :id"),
        {"ts": now - timedelta(days=1), "id": aid_recent},
    )
    db_nexo.execute(
        text("UPDATE nexo.query_approvals SET created_at = :ts WHERE id = :id"),
        {"ts": now - timedelta(days=5), "id": aid_mid},
    )
    db_nexo.execute(
        text("UPDATE nexo.query_approvals SET created_at = :ts WHERE id = :id"),
        {"ts": now - timedelta(days=10), "id": aid_old},
    )
    db_nexo.commit()

    n = svc.expire_stale(db_nexo, ttl_days=7)
    assert n == 1

    # Refresh from DB
    db_nexo.expire_all()
    r_recent = db_nexo.get(NexoQueryApproval, aid_recent)
    r_mid = db_nexo.get(NexoQueryApproval, aid_mid)
    r_old = db_nexo.get(NexoQueryApproval, aid_old)

    assert r_recent.status == "pending"
    assert r_mid.status == "pending"
    assert r_old.status == "expired"
    assert r_old.expired_at is not None
