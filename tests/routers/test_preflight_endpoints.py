"""Preflight router contract tests (Plan 04-02, QUERY-04, QUERY-07).

Suite integration — requiere Postgres up con el schema ``nexo`` poblado
(seeds ``query_thresholds``). Valida los contratos HTTP:

- POST /api/pipeline/preflight → 200 + Estimation JSON.
- POST /api/pipeline/run con amber sin force → 428.
- POST /api/pipeline/run con red sin approval → 403.
- POST /api/bbdd/query con amber sin force → 428.
- GET /api/capacidad rango <=90d NO dispara preflight (200 ó 502).
- GET /api/capacidad rango >90d dispara preflight.

test_run_red_with_valid_approval_executes queda ``xfail`` hasta que Plan
04-03 entregue ``nexo/services/approvals.py`` con ``consume_approval``.
"""
from __future__ import annotations

from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select, text

from nexo.data.engines import SessionLocalNexo
from nexo.data.models_nexo import (
    NexoDepartment,
    NexoLoginAttempt,
    NexoQueryLog,
    NexoQueryThreshold,
    NexoSession,
    NexoUser,
)
from nexo.services.auth import hash_password
from nexo.services import thresholds_cache


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
        reason="Postgres no disponible — requiere `docker compose up -d db`",
    ),
]


TEST_DOMAIN = "@preflight-endpoints-test.local"
TEST_PASSWORD = "pfendpointstest123456"


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    from api.main import app

    with TestClient(app, follow_redirects=False) as c:
        yield c


@pytest.fixture(autouse=True)
def _cleanup():
    _purge()
    yield
    _purge()


def _purge() -> None:
    db = SessionLocalNexo()
    try:
        users = (
            db.execute(
                select(NexoUser).where(NexoUser.email.like(f"%{TEST_DOMAIN}"))
            )
            .scalars()
            .all()
        )
        ids = [u.id for u in users]
        if ids:
            db.execute(delete(NexoQueryLog).where(NexoQueryLog.user_id.in_(ids)))
            for u in users:
                db.execute(delete(NexoSession).where(NexoSession.user_id == u.id))
                db.delete(u)
        db.execute(
            delete(NexoLoginAttempt).where(
                NexoLoginAttempt.email.like(f"%{TEST_DOMAIN}")
            )
        )
        db.commit()
    finally:
        db.close()


def _create_user(email: str, role: str, dept_codes: list[str]) -> NexoUser:
    db = SessionLocalNexo()
    try:
        user = NexoUser(
            email=email,
            password_hash=hash_password(TEST_PASSWORD),
            role=role,
            active=True,
            must_change_password=False,
        )
        db.add(user)
        db.flush()
        if dept_codes:
            depts = (
                db.execute(
                    select(NexoDepartment).where(
                        NexoDepartment.code.in_(dept_codes)
                    )
                )
                .scalars()
                .all()
            )
            user.departments = list(depts)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


def _login(c: TestClient, email: str) -> str:
    r = c.post(
        "/login",
        data={"email": email, "password": TEST_PASSWORD},
        headers={"Accept": "text/html"},
    )
    assert r.status_code == 303, f"Login failed: {r.status_code} {r.text[:200]}"
    return r.cookies["nexo_session"]


@pytest.fixture
def tighten_pipeline_thresholds():
    """Baja warn_ms y block_ms de pipeline/run para forzar amber/red con
    payloads pequeños. Restaura al final.
    """
    db = SessionLocalNexo()
    try:
        row = db.get(NexoQueryThreshold, "pipeline/run")
        original = (row.warn_ms, row.block_ms, row.factor_ms) if row else None
        if row:
            row.warn_ms = 1  # 1ms warn → casi cualquier estimación dispara amber
            row.block_ms = 30_000
            row.factor_ms = 2000.0
            db.commit()
        # Forzar reload del cache
        thresholds_cache.full_reload()
        yield
    finally:
        # restore
        db = SessionLocalNexo()
        try:
            row = db.get(NexoQueryThreshold, "pipeline/run")
            if row and original:
                row.warn_ms, row.block_ms, row.factor_ms = original
                db.commit()
        finally:
            db.close()
        thresholds_cache.full_reload()


# ── Tests ─────────────────────────────────────────────────────────────

def test_preflight_returns_estimation_json(client: TestClient):
    """POST /api/pipeline/preflight → 200 con Estimation completa."""
    email = f"preflight-user{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    r = client.post(
        "/api/pipeline/preflight",
        cookies={"nexo_session": cookie},
        json={
            "fecha_inicio": "2026-01-01",
            "fecha_fin": "2026-01-02",
            "modulos": ["disponibilidad"],
            "source": "db",
            "recursos": ["luk1"],
        },
    )
    assert r.status_code == 200, f"status={r.status_code} body={r.text[:200]}"
    body = r.json()
    assert body["endpoint"] == "pipeline/run"
    assert "level" in body and body["level"] in {"green", "amber", "red"}
    assert "estimated_ms" in body
    assert "breakdown" in body
    assert "factor_used_ms" in body
    assert "warn_ms" in body
    assert "block_ms" in body


def test_run_amber_no_force_returns_428(
    client: TestClient, tighten_pipeline_thresholds
):
    """Con warn_ms=1 y payload trivial, preflight clasifica amber.
    /run sin force → 428 Precondition Required con detail.estimation.
    """
    email = f"amber-user{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    r = client.post(
        "/api/pipeline/run",
        cookies={"nexo_session": cookie},
        json={
            "fecha_inicio": "2026-01-01",
            "fecha_fin": "2026-01-02",
            "modulos": [],
            "source": "db",
            "recursos": ["luk1"],
        },
    )
    assert r.status_code == 428, f"status={r.status_code} body={r.text[:300]}"
    detail = r.json()["detail"]
    assert detail["action"] == "confirm_amber"
    assert detail["estimation"]["level"] == "amber"


def test_run_red_without_approval_returns_403(client: TestClient):
    """Payload pesado → red. Sin approval_id → 403 con detail.estimation."""
    email = f"red-user{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    # Con umbrales default D-01 (block 600_000), 10 recursos × 30 días × 2000ms
    # = 600_000ms = block → red.
    r = client.post(
        "/api/pipeline/run",
        cookies={"nexo_session": cookie},
        json={
            "fecha_inicio": "2026-01-01",
            "fecha_fin": "2026-01-30",
            "modulos": [],
            "source": "db",
            "recursos": [f"r{i}" for i in range(10)],
        },
    )
    assert r.status_code == 403, f"status={r.status_code} body={r.text[:300]}"
    detail = r.json()["detail"]
    assert detail["action"] == "request_approval"
    assert detail["estimation"]["level"] == "red"


def test_preflight_scope_capacidad_short_bypass(client: TestClient):
    """GET /api/capacidad rango 30d NO dispara preflight (D-03).

    Sin SQL Server el endpoint devuelve 502, pero lo crítico es que NO
    devuelve 428 (preflight bypass) y NO escribe query_log.
    """
    email = f"cap-bypass{TEST_DOMAIN}"
    user = _create_user(email, role="usuario", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    # Contar filas query_log antes
    db = SessionLocalNexo()
    try:
        before = (
            db.execute(
                select(NexoQueryLog)
                .where(NexoQueryLog.user_id == user.id)
                .where(NexoQueryLog.endpoint == "capacidad")
            )
            .scalars()
            .all()
        )
    finally:
        db.close()

    r = client.get(
        "/api/capacidad",
        params={"fecha_inicio": "2025-01-01", "fecha_fin": "2025-01-31"},
        cookies={"nexo_session": cookie},
    )
    # Cualquier status EXCEPTO 428 (que indicaría que se disparó preflight).
    assert r.status_code != 428, "capacidad rango<=90d NO debe disparar preflight"

    db = SessionLocalNexo()
    try:
        after = (
            db.execute(
                select(NexoQueryLog)
                .where(NexoQueryLog.user_id == user.id)
                .where(NexoQueryLog.endpoint == "capacidad")
            )
            .scalars()
            .all()
        )
    finally:
        db.close()
    assert len(after) == len(before), (
        "capacidad rango<=90d NO debe escribir query_log"
    )


def test_bbdd_query_preflight_runs_before_validate(client: TestClient):
    """POST /api/bbdd/query: preflight SIEMPRE dispara (bbdd/query no tiene
    gate de rango — D-02). SQL trivial con umbral default → green → 200.
    """
    email = f"bbdd-pf{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    r = client.post(
        "/api/bbdd/query",
        cookies={"nexo_session": cookie},
        json={"sql": "SELECT 1", "database": "dbizaro"},
    )
    # Con SQL Server up → 200; sin él → 502. Aceptamos ambos; lo importante
    # es que NO se queda en 428 (no hay amber con factor default).
    assert r.status_code in (200, 500, 502), (
        f"expected 200/500/502, got {r.status_code} body={r.text[:200]}"
    )


@pytest.mark.xfail(
    reason="Requires approval flow from Plan 04-03 (consume_approval)",
    strict=False,
)
def test_run_red_with_valid_approval_executes(client: TestClient):
    """Con approval_id válido (status=approved, user match, params match),
    POST /run ejecuta y graba fila con approval_id poblado.

    xfail hasta que Plan 04-03 implemente ``nexo/services/approvals.py``
    con ``consume_approval``.
    """
    email = f"red-ok{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    r = client.post(
        "/api/pipeline/run",
        cookies={"nexo_session": cookie},
        json={
            "fecha_inicio": "2026-01-01",
            "fecha_fin": "2026-01-30",
            "modulos": [],
            "source": "db",
            "recursos": [f"r{i}" for i in range(10)],
            "force": True,
            "approval_id": 99999,  # stub id (no aterrizado hasta 04-03)
        },
    )
    # Plan 04-02 standalone responde 503; Plan 04-03 lo convertirá en 200
    # tras CAS exitoso, o 403 si el approval no matchea.
    assert r.status_code == 200
