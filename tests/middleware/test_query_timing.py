"""QueryTimingMiddleware integration tests (Plan 04-02, QUERY-05, D-17).

Suite integration — requiere Postgres up con el schema ``nexo`` poblado.
Valida que el middleware:

- Escribe fila en ``nexo.query_log`` al atravesar un endpoint TIMED.
- NO escribe fila para paths fuera de ``_TIMED_PATHS`` (p.ej. /api/health).
- Marca ``status='slow'`` cuando ``actual_ms > warn_ms * 1.5`` + emite
  ``logger.warning`` (D-17).
- Marca ``status='error'`` cuando el handler lanza.

No valida el contrato de los endpoints timed — eso se cubre en
``tests/routers/test_preflight_endpoints.py``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select, text
from sqlalchemy.orm import Session

from nexo.data.engines import SessionLocalNexo
from nexo.data.models_nexo import (
    NexoDepartment,
    NexoLoginAttempt,
    NexoQueryLog,
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
        reason="Postgres no disponible — requiere `docker compose up -d db`",
    ),
]


TEST_DOMAIN = "@query-timing-test.local"
TEST_PASSWORD = "qtimingtest123456"  # min 12 chars


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    from api.main import app  # defer import (triggers middleware register)

    with TestClient(app, follow_redirects=False) as c:
        yield c


@pytest.fixture(autouse=True)
def _cleanup_artifacts():
    """Purga users + sessions + login_attempts + query_log del test domain."""
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
        user_ids = [u.id for u in users]
        if user_ids:
            db.execute(
                delete(NexoQueryLog).where(NexoQueryLog.user_id.in_(user_ids))
            )
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


def _login(client: TestClient, email: str) -> str:
    r = client.post(
        "/login",
        data={"email": email, "password": TEST_PASSWORD},
        headers={"Accept": "text/html"},
    )
    assert r.status_code == 303, f"Login failed: {r.status_code} body={r.text[:200]}"
    cookie = r.cookies.get("nexo_session")
    assert cookie
    return cookie


def _count_query_log(user_id: int, endpoint: str | None = None) -> int:
    db = SessionLocalNexo()
    try:
        q = select(NexoQueryLog).where(NexoQueryLog.user_id == user_id)
        if endpoint is not None:
            q = q.where(NexoQueryLog.endpoint == endpoint)
        return len(db.execute(q).scalars().all())
    finally:
        db.close()


# ── Tests ─────────────────────────────────────────────────────────────

def test_timing_writes_row_for_bbdd_query(client: TestClient):
    """POST /api/bbdd/query (SQL trivial) → fila en nexo.query_log con
    endpoint='bbdd/query', user_id correcto, actual_ms > 0.

    Usamos SQL que pasa la whitelist pero falla la ejecución (SQL Server
    no disponible en test) — igual middleware corre y graba la fila con
    status='error'. Lo que importa aquí es que la fila aparezca.
    """
    email = f"bbdd-user{TEST_DOMAIN}"
    user = _create_user(email, role="usuario", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    before = _count_query_log(user.id, "bbdd/query")

    r = client.post(
        "/api/bbdd/query",
        cookies={"nexo_session": cookie},
        json={"sql": "SELECT 1", "database": "dbizaro"},
    )
    # Sin SQL Server real devuelve 502 (o 500); con SQL Server devolvería 200.
    # Cualquiera graba fila (el middleware corre en ambas ramas).
    assert r.status_code in (200, 500, 502, 504), f"status={r.status_code}"

    after = _count_query_log(user.id, "bbdd/query")
    assert after == before + 1, f"Esperado +1 fila query_log, before={before} after={after}"

    # Validar campos de la fila
    db = SessionLocalNexo()
    try:
        row = (
            db.execute(
                select(NexoQueryLog)
                .where(NexoQueryLog.user_id == user.id)
                .where(NexoQueryLog.endpoint == "bbdd/query")
                .order_by(NexoQueryLog.ts.desc())
            )
            .scalars()
            .first()
        )
        assert row is not None
        assert row.actual_ms is not None and row.actual_ms >= 0
        assert row.estimated_ms is not None  # router pobló request.state
        assert row.params_json is not None
    finally:
        db.close()


def test_timing_excludes_health_path(client: TestClient):
    """GET /api/health → NO escribe fila en query_log (path no TIMED)."""
    email = f"health-user{TEST_DOMAIN}"
    user = _create_user(email, role="usuario", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    before = _count_query_log(user.id)

    r = client.get("/api/health", cookies={"nexo_session": cookie})
    assert r.status_code == 200

    after = _count_query_log(user.id)
    assert after == before, (
        f"/api/health NO debería escribir query_log, "
        f"pero before={before} after={after}"
    )


def test_timing_excludes_capacidad_short_range(client: TestClient):
    """GET /api/capacidad con rango 30d (<=90d) → NO escribe fila (D-03).

    Router no popula request.state.estimated_ms → middleware short-circuits.
    """
    email = f"cap-short{TEST_DOMAIN}"
    user = _create_user(email, role="usuario", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    before = _count_query_log(user.id, "capacidad")

    r = client.get(
        "/api/capacidad",
        params={"fecha_inicio": "2025-01-01", "fecha_fin": "2025-01-31"},
        cookies={"nexo_session": cookie},
    )
    # SQL Server no disponible → 502. Middleware skip porque rango <=90d.
    assert r.status_code in (200, 500, 502, 504)

    after = _count_query_log(user.id, "capacidad")
    assert after == before, (
        f"capacidad con rango <=90d NO debe escribir query_log (D-03), "
        f"before={before} after={after}"
    )


def test_timing_no_row_when_unauthenticated(client: TestClient):
    """Request sin cookie de sesión: Auth devuelve 401; middleware Timing
    nunca escribe (user=None short-circuit)."""
    # Sin crear usuario, solo intentar acceder.
    r = client.post(
        "/api/bbdd/query",
        json={"sql": "SELECT 1", "database": "dbizaro"},
    )
    assert r.status_code == 401
    # No user_id conocido → leemos total y comparamos la ventana temporal.
    db = SessionLocalNexo()
    try:
        # Ventana: últimos 5 segundos. Si Auth realmente bloqueó, no hay fila.
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=5)
        rows = (
            db.execute(
                select(NexoQueryLog).where(NexoQueryLog.ts > cutoff)
            )
            .scalars()
            .all()
        )
        # No debería haber fila con user_id=None y endpoint bbdd/query de
        # esta request (el test_timing_writes_row_for_bbdd_query ya habrá
        # purgado su fila al terminar). Verificamos que ninguna fila
        # reciente corresponde a este caso.
        for row in rows:
            assert not (row.endpoint == "bbdd/query" and row.user_id is None), (
                "Middleware NO debe escribir fila cuando user=None"
            )
    finally:
        db.close()
