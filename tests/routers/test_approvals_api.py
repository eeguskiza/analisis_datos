"""Approvals API router contract tests (Plan 04-03 / QUERY-06).

Cobertura de los 7 endpoints de ``api/routers/approvals.py``:

- POST /api/approvals (cualquier auth user)
- GET /api/approvals/count (propietario)
- POST /api/approvals/{id}/approve|reject (propietario)
- POST /api/approvals/{id}/cancel (owner)
- GET /ajustes/solicitudes (propietario)
- GET /mis-solicitudes (cualquier auth user)

Integration — requiere Postgres up + schema nexo inicializado. Sigue el
patrón de ``tests/auth/test_rbac_smoke.py`` (TestClient + users de
dominio @approvals-api-test.local + purga en teardown).
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
    NexoQueryApproval,
    NexoSession,
    NexoUser,
)
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


TEST_DOMAIN = "@approvals-api-test.local"
TEST_PASSWORD = "approvalsapitest12345"


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    from api.main import app
    with TestClient(app, follow_redirects=False) as c:
        yield c


@pytest.fixture(autouse=True)
def _cleanup():
    _purge()
    _reset_rate_limit()
    yield
    _purge()


def _reset_rate_limit() -> None:
    """Reset slowapi in-memory state para evitar 429 en /login durante la
    suite completa (cada test hace 1-2 logins; slowapi 20/min se acumula
    en runs concatenados).
    """
    try:
        from api.rate_limit import limiter
        limiter.reset()
    except Exception:
        pass


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
            db.execute(
                delete(NexoQueryApproval).where(
                    NexoQueryApproval.user_id.in_(ids)
                )
            )
            for u in users:
                db.execute(
                    delete(NexoSession).where(NexoSession.user_id == u.id)
                )
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


# ── Tests ────────────────────────────────────────────────────────────────

def test_create_approval_pending(client: TestClient):
    """POST /api/approvals crea fila pending + devuelve approval_id."""
    email = f"create-user{TEST_DOMAIN}"
    user = _create_user(email, role="usuario", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    r = client.post(
        "/api/approvals",
        cookies={"nexo_session": cookie},
        json={
            "endpoint": "pipeline/run",
            "params": {"fecha_desde": "2026-01-01", "fecha_hasta": "2026-01-30"},
            "estimated_ms": 700000,
        },
    )
    assert r.status_code == 200, f"status={r.status_code} body={r.text[:200]}"
    body = r.json()
    assert isinstance(body["approval_id"], int)
    assert body["status"] == "pending"

    # Verify row en DB
    db = SessionLocalNexo()
    try:
        row = db.get(NexoQueryApproval, body["approval_id"])
        assert row is not None
        assert row.user_id == user.id
        assert row.status == "pending"
        assert row.endpoint == "pipeline/run"
    finally:
        db.close()


def test_propietario_approve_flow(client: TestClient):
    """propietario POST /approve → 303 + status=approved + approved_by."""
    user_email = f"flow-user{TEST_DOMAIN}"
    owner_email = f"flow-owner{TEST_DOMAIN}"
    user = _create_user(user_email, role="usuario", dept_codes=["ingenieria"])
    owner = _create_user(owner_email, role="propietario", dept_codes=[])

    # User crea approval
    user_cookie = _login(client, user_email)
    r = client.post(
        "/api/approvals",
        cookies={"nexo_session": user_cookie},
        json={"endpoint": "pipeline/run", "params": {"a": 1}, "estimated_ms": 100},
    )
    aid = r.json()["approval_id"]

    # Propietario aprueba
    owner_cookie = _login(client, owner_email)
    r = client.post(
        f"/api/approvals/{aid}/approve",
        cookies={"nexo_session": owner_cookie},
    )
    assert r.status_code == 303, f"status={r.status_code} body={r.text[:200]}"
    assert r.headers["location"].startswith("/ajustes/solicitudes")

    db = SessionLocalNexo()
    try:
        row = db.get(NexoQueryApproval, aid)
        assert row.status == "approved"
        assert row.approved_by == owner.id
        assert row.approved_at is not None
    finally:
        db.close()


def test_propietario_reject_flow(client: TestClient):
    user_email = f"rej-user{TEST_DOMAIN}"
    owner_email = f"rej-owner{TEST_DOMAIN}"
    _create_user(user_email, role="usuario", dept_codes=["ingenieria"])
    _create_user(owner_email, role="propietario", dept_codes=[])

    user_cookie = _login(client, user_email)
    r = client.post(
        "/api/approvals",
        cookies={"nexo_session": user_cookie},
        json={"endpoint": "pipeline/run", "params": {"a": 2}, "estimated_ms": 100},
    )
    aid = r.json()["approval_id"]

    owner_cookie = _login(client, owner_email)
    r = client.post(
        f"/api/approvals/{aid}/reject",
        cookies={"nexo_session": owner_cookie},
    )
    assert r.status_code == 303

    db = SessionLocalNexo()
    try:
        row = db.get(NexoQueryApproval, aid)
        assert row.status == "rejected"
        assert row.rejected_at is not None
    finally:
        db.close()


def test_user_cancel_own_pending(client: TestClient):
    email = f"cancel-own{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    r = client.post(
        "/api/approvals",
        cookies={"nexo_session": cookie},
        json={"endpoint": "pipeline/run", "params": {"k": "v"}, "estimated_ms": 100},
    )
    aid = r.json()["approval_id"]

    r = client.post(
        f"/api/approvals/{aid}/cancel",
        cookies={"nexo_session": cookie},
    )
    assert r.status_code == 303
    assert r.headers["location"].startswith("/mis-solicitudes")

    db = SessionLocalNexo()
    try:
        row = db.get(NexoQueryApproval, aid)
        assert row.status == "cancelled"
    finally:
        db.close()


def test_user_cannot_cancel_others(client: TestClient):
    email_a = f"cancel-a{TEST_DOMAIN}"
    email_b = f"cancel-b{TEST_DOMAIN}"
    _create_user(email_a, role="usuario", dept_codes=["ingenieria"])
    _create_user(email_b, role="usuario", dept_codes=["ingenieria"])

    # A crea approval
    cookie_a = _login(client, email_a)
    r = client.post(
        "/api/approvals",
        cookies={"nexo_session": cookie_a},
        json={"endpoint": "pipeline/run", "params": {"q": 1}, "estimated_ms": 100},
    )
    aid = r.json()["approval_id"]

    # B intenta cancelar
    cookie_b = _login(client, email_b)
    r = client.post(
        f"/api/approvals/{aid}/cancel",
        cookies={"nexo_session": cookie_b},
    )
    assert r.status_code == 403
    db = SessionLocalNexo()
    try:
        row = db.get(NexoQueryApproval, aid)
        assert row.status == "pending"
    finally:
        db.close()


def test_count_badge_returns_html_fragment_empty(client: TestClient):
    owner_email = f"badge-owner-empty{TEST_DOMAIN}"
    _create_user(owner_email, role="propietario", dept_codes=[])
    cookie = _login(client, owner_email)

    r = client.get(
        "/api/approvals/count",
        cookies={"nexo_session": cookie},
    )
    assert r.status_code == 200
    assert r.text == ""


def test_count_badge_returns_html_fragment_with_count(client: TestClient):
    user_email = f"badge-user{TEST_DOMAIN}"
    owner_email = f"badge-owner{TEST_DOMAIN}"
    _create_user(user_email, role="usuario", dept_codes=["ingenieria"])
    _create_user(owner_email, role="propietario", dept_codes=[])

    # Usuario crea 3 solicitudes
    user_cookie = _login(client, user_email)
    for i in range(3):
        r = client.post(
            "/api/approvals",
            cookies={"nexo_session": user_cookie},
            json={
                "endpoint": "pipeline/run",
                "params": {"i": i},
                "estimated_ms": 100,
            },
        )
        assert r.status_code == 200

    # Propietario ve count
    owner_cookie = _login(client, owner_email)
    r = client.get(
        "/api/approvals/count",
        cookies={"nexo_session": owner_cookie},
    )
    assert r.status_code == 200
    assert "(3)" in r.text
    assert "<span" in r.text


def test_non_propietario_cannot_see_count(client: TestClient):
    email = f"nocount{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    r = client.get(
        "/api/approvals/count",
        cookies={"nexo_session": cookie},
    )
    assert r.status_code == 403


def test_mis_solicitudes_shows_only_own(client: TestClient):
    email_a = f"mis-a{TEST_DOMAIN}"
    email_b = f"mis-b{TEST_DOMAIN}"
    _create_user(email_a, role="usuario", dept_codes=["ingenieria"])
    _create_user(email_b, role="usuario", dept_codes=["ingenieria"])

    # A crea 2, B crea 1
    cookie_a = _login(client, email_a)
    for _ in range(2):
        client.post(
            "/api/approvals",
            cookies={"nexo_session": cookie_a},
            json={"endpoint": "pipeline/run", "params": {"a": 1}, "estimated_ms": 100},
        )
    cookie_b = _login(client, email_b)
    r_b_create = client.post(
        "/api/approvals",
        cookies={"nexo_session": cookie_b},
        json={"endpoint": "bbdd/query", "params": {"sql": "SELECT 1"}, "estimated_ms": 100},
    )
    aid_b = r_b_create.json()["approval_id"]

    # A solo ve 2 filas en /mis-solicitudes
    r = client.get("/mis-solicitudes", cookies={"nexo_session": cookie_a})
    assert r.status_code == 200
    # B solo ve 1 fila — y el id de B aparece en su página, no en la de A
    r_b = client.get("/mis-solicitudes", cookies={"nexo_session": cookie_b})
    assert r_b.status_code == 200
    assert f"#{aid_b}" in r_b.text
    assert f"#{aid_b}" not in r.text  # user A no ve el de B


def test_ajustes_solicitudes_is_propietario_only(client: TestClient):
    user_email = f"ajustes-user{TEST_DOMAIN}"
    owner_email = f"ajustes-owner{TEST_DOMAIN}"
    _create_user(user_email, role="usuario", dept_codes=["ingenieria"])
    _create_user(owner_email, role="propietario", dept_codes=[])

    # Usuario no-propietario:
    # - HTML GET (Accept: text/html) → 302 a "/" + cookie nexo_flash
    #   (Plan 05-03 / D-07). Antes devolvía 403; el handler
    #   ``http_exception_handler_403`` ahora redirige.
    # - JSON GET (Accept: application/json) → 403 JSON intacto.
    cookie = _login(client, user_email)
    r_html = client.get(
        "/ajustes/solicitudes",
        cookies={"nexo_session": cookie},
        headers={"Accept": "text/html"},
    )
    assert r_html.status_code == 302
    assert r_html.headers["location"] == "/"
    assert "nexo_flash=" in r_html.headers.get("set-cookie", "")

    r_json = client.get(
        "/ajustes/solicitudes",
        cookies={"nexo_session": cookie},
        headers={"Accept": "application/json"},
    )
    assert r_json.status_code == 403
    assert "nexo_flash" not in r_json.headers.get("set-cookie", "")

    # Propietario → 200
    owner_cookie = _login(client, owner_email)
    r = client.get("/ajustes/solicitudes", cookies={"nexo_session": owner_cookie})
    assert r.status_code == 200
    # Debe contener las secciones esperadas
    assert "Pendientes" in r.text
    assert "Histórico" in r.text
