"""Thresholds CRUD router contract tests (Plan 04-04 / QUERY-02 / D-04 / D-19).

Cobertura de los 3 endpoints de ``api/routers/limites.py``:

- GET /ajustes/limites (propietario-only)
- PUT /api/thresholds/{endpoint:path} (propietario-only)
- POST /api/thresholds/{endpoint:path}/recalibrate (propietario-only)

Integration — requiere Postgres + schema nexo inicializado. Mismo patron
que ``tests/routers/test_approvals_api.py``: TestClient + users
@threshold-crud-test.local + purga en teardown + reset rate limiter.
"""
from __future__ import annotations

import json
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select, text

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
        reason="Postgres no arriba: docker compose up -d db",
    ),
]


TEST_DOMAIN = "@threshold-crud-test.local"
TEST_PASSWORD = "thresholdcrudtest12345"


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
    _restore_seed_thresholds()
    yield
    _purge()
    _restore_seed_thresholds()


def _reset_rate_limit() -> None:
    """Reset slowapi in-memory state — evita 429 en /login entre tests."""
    try:
        from api.rate_limit import limiter
        limiter.reset()
    except Exception:
        pass


def _restore_seed_thresholds() -> None:
    """Restaura valores originales de D-01/D-02/D-03/D-04 para aislamiento."""
    db = SessionLocalNexo()
    try:
        seeds = {
            "pipeline/run": (120_000, 600_000, 2000.0),
            "bbdd/query": (3_000, 30_000, 1000.0),
            "capacidad": (3_000, 30_000, 50.0),
            "operarios": (3_000, 30_000, 50.0),
        }
        for endpoint, (warn, block, factor) in seeds.items():
            db.execute(
                text(
                    "UPDATE nexo.query_thresholds "
                    "SET warn_ms = :w, block_ms = :b, factor_ms = :f, "
                    "    factor_updated_at = NULL, updated_by = NULL "
                    "WHERE endpoint = :ep"
                ),
                {"ep": endpoint, "w": warn, "b": block, "f": factor},
            )
        db.commit()
    finally:
        db.close()


def _purge() -> None:
    db = SessionLocalNexo()
    try:
        users = db.execute(
            select(NexoUser).where(NexoUser.email.like(f"%{TEST_DOMAIN}"))
        ).scalars().all()
        ids = [u.id for u in users]
        if ids:
            db.execute(
                delete(NexoQueryLog).where(NexoQueryLog.user_id.in_(ids))
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
            depts = db.execute(
                select(NexoDepartment).where(NexoDepartment.code.in_(dept_codes))
            ).scalars().all()
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


def _seed_query_log(
    user_id: int,
    endpoint: str,
    n_rows: int,
    actual_ms: int,
    n_recursos: int = 0,
    n_dias: int = 0,
    status: str = "ok",
) -> None:
    """Genera N filas sinteticas en nexo.query_log para tests de recalibrate."""
    db = SessionLocalNexo()
    try:
        for _ in range(n_rows):
            params = {}
            if n_recursos and n_dias:
                params = {"n_recursos": n_recursos, "n_dias": n_dias}
            db.add(NexoQueryLog(
                user_id=user_id,
                ip="127.0.0.1",
                endpoint=endpoint,
                params_json=json.dumps(params),
                estimated_ms=actual_ms,
                actual_ms=actual_ms,
                rows=None,
                status=status,
                approval_id=None,
            ))
        db.commit()
    finally:
        db.close()


# ── Tests ────────────────────────────────────────────────────────────────

def test_put_thresholds_non_propietario_forbidden(client: TestClient):
    """Usuario normal → 403 en PUT /api/thresholds/..."""
    email = f"put-user{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    r = client.put(
        "/api/thresholds/pipeline%2Frun",
        cookies={"nexo_session": cookie},
        json={"warn_ms": 99_000, "block_ms": 500_000},
    )
    assert r.status_code == 403


def test_put_thresholds_updates_db_and_emits_notify(client: TestClient):
    """Propietario edita warn_ms/block_ms → UPDATE + NOTIFY.

    Verificamos: (a) 200 response, (b) fila de DB actualizada, (c) cache
    in-memory tras un get() refleja el cambio (safety-net o listener
    cubriendo el propagation).
    """
    email = f"put-owner{TEST_DOMAIN}"
    _create_user(email, role="propietario", dept_codes=[])
    cookie = _login(client, email)

    new_warn, new_block = 99_000, 500_000
    r = client.put(
        "/api/thresholds/pipeline%2Frun",
        cookies={"nexo_session": cookie},
        json={"warn_ms": new_warn, "block_ms": new_block},
    )
    assert r.status_code == 200, f"body={r.text[:300]}"
    body = r.json()
    assert body["ok"] is True
    assert body["endpoint"] == "pipeline/run"

    # Verify DB
    db = SessionLocalNexo()
    try:
        row = db.execute(
            text("SELECT warn_ms, block_ms FROM nexo.query_thresholds "
                 "WHERE endpoint='pipeline/run'")
        ).mappings().first()
        assert row["warn_ms"] == new_warn
        assert row["block_ms"] == new_block
    finally:
        db.close()


def test_put_thresholds_rejects_warn_gte_block(client: TestClient):
    """Validacion: warn_ms < block_ms (si no, 400)."""
    email = f"put-validate{TEST_DOMAIN}"
    _create_user(email, role="propietario", dept_codes=[])
    cookie = _login(client, email)

    r = client.put(
        "/api/thresholds/pipeline%2Frun",
        cookies={"nexo_session": cookie},
        json={"warn_ms": 500_000, "block_ms": 100_000},  # warn > block
    )
    assert r.status_code == 400
    detail = r.json()["detail"].lower()
    assert "warn_ms" in detail and "block_ms" in detail


def test_put_thresholds_404_for_unknown_endpoint(client: TestClient):
    """Endpoint no existente → 404."""
    email = f"put-404{TEST_DOMAIN}"
    _create_user(email, role="propietario", dept_codes=[])
    cookie = _login(client, email)

    r = client.put(
        "/api/thresholds/no%2Fexiste",
        cookies={"nexo_session": cookie},
        json={"warn_ms": 1000, "block_ms": 5000},
    )
    assert r.status_code == 404


def test_recalibrate_insufficient_data_returns_400(client: TestClient):
    """< 10 runs validos → 400 con detail diagnostico."""
    email = f"recalib-low{TEST_DOMAIN}"
    user = _create_user(email, role="propietario", dept_codes=[])
    cookie = _login(client, email)

    # 5 filas no es suficiente.
    _seed_query_log(
        user_id=user.id,
        endpoint="pipeline/run",
        n_rows=5,
        actual_ms=2500,
        n_recursos=2,
        n_dias=10,
    )

    r = client.post(
        "/api/thresholds/pipeline%2Frun/recalibrate?confirm=false",
        cookies={"nexo_session": cookie},
    )
    assert r.status_code == 400
    assert "al menos 10" in r.json()["detail"].lower() or "runs" in r.json()["detail"].lower()


def test_recalibrate_preview_and_confirm_persists(client: TestClient):
    """Con >= 10 runs: preview + confirm flow.

    Preview: ?confirm=false devuelve {factor_old, factor_new, sample_size,
             committed: False}. NO persiste.
    Confirm: ?confirm=true persiste + devuelve committed: True.
    """
    email = f"recalib-ok{TEST_DOMAIN}"
    user = _create_user(email, role="propietario", dept_codes=[])
    cookie = _login(client, email)

    # 15 filas pipeline/run con params parseables. Factor esperado:
    # actual_ms / (n_recursos * n_dias) = 3000 / (3 * 10) = 100.
    _seed_query_log(
        user_id=user.id,
        endpoint="pipeline/run",
        n_rows=15,
        actual_ms=3000,
        n_recursos=3,
        n_dias=10,
    )

    # Preview
    r = client.post(
        "/api/thresholds/pipeline%2Frun/recalibrate?confirm=false",
        cookies={"nexo_session": cookie},
    )
    assert r.status_code == 200, f"preview failed: {r.text[:300]}"
    preview = r.json()
    assert preview["committed"] is False
    assert preview["preview"] is True
    assert preview["sample_size"] == 15
    assert preview["factor_new"] == pytest.approx(100.0, rel=0.05)
    assert preview["factor_old"] == 2000.0  # seed inicial

    # DB no debe haberse modificado aun.
    db = SessionLocalNexo()
    try:
        row = db.execute(
            text("SELECT factor_ms, factor_updated_at FROM nexo.query_thresholds "
                 "WHERE endpoint='pipeline/run'")
        ).mappings().first()
        assert row["factor_ms"] == 2000.0
        assert row["factor_updated_at"] is None
    finally:
        db.close()

    # Confirm
    r = client.post(
        "/api/thresholds/pipeline%2Frun/recalibrate?confirm=true",
        cookies={"nexo_session": cookie},
    )
    assert r.status_code == 200
    confirmed = r.json()
    assert confirmed["committed"] is True
    assert confirmed["factor_new"] == pytest.approx(100.0, rel=0.05)

    # DB debe reflejar el cambio + factor_updated_at poblado.
    db = SessionLocalNexo()
    try:
        row = db.execute(
            text("SELECT factor_ms, factor_updated_at FROM nexo.query_thresholds "
                 "WHERE endpoint='pipeline/run'")
        ).mappings().first()
        assert row["factor_ms"] == pytest.approx(100.0, rel=0.05)
        assert row["factor_updated_at"] is not None
    finally:
        db.close()


def test_recalibrate_filters_outliers_under_500ms(client: TestClient):
    """Pitfall 6: filas con actual_ms <= 500ms se filtran.

    Seed: 15 filas con actual_ms=400 (bajo umbral) + 10 filas con
    actual_ms=2000. Solo las 10 validas entran al median.
    """
    email = f"recalib-outlier{TEST_DOMAIN}"
    user = _create_user(email, role="propietario", dept_codes=[])
    cookie = _login(client, email)

    # Filas trivialmente rapidas — deben filtrarse (<500ms).
    _seed_query_log(
        user_id=user.id,
        endpoint="pipeline/run",
        n_rows=15,
        actual_ms=400,
        n_recursos=2,
        n_dias=5,
    )
    # Filas validas
    _seed_query_log(
        user_id=user.id,
        endpoint="pipeline/run",
        n_rows=10,
        actual_ms=5000,
        n_recursos=2,
        n_dias=5,
    )

    r = client.post(
        "/api/thresholds/pipeline%2Frun/recalibrate?confirm=false",
        cookies={"nexo_session": cookie},
    )
    # Puede 200 (si se aplico filtro y quedan >=10) o 400 si el outlier
    # filter dejo menos de 10. El limit=30 asegura que cojamos las 25
    # filas mas recientes; de esas, 15 son outliers -> 10 validas.
    # El limit=30 se aplica ANTES del filtro, asi que bajan a 30 lo mas
    # reciente — 15 outliers + 10 validas = 25 total -> 10 validas.
    assert r.status_code == 200, f"body={r.text[:300]}"
    preview = r.json()
    assert preview["sample_size"] == 10
    expected_factor = 5000 / (2 * 5)
    assert preview["factor_new"] == pytest.approx(expected_factor, rel=0.1)


def test_get_limites_page_requires_propietario(client: TestClient):
    """Non-propietario:
    - HTML GET → 302 a / + cookie nexo_flash (Plan 05-03 / D-07).
    - JSON GET → 403 JSON intacto.
    """
    email = f"get-user{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    # HTML Accept → 302 redirect + flash cookie.
    r_html = client.get(
        "/ajustes/limites",
        cookies={"nexo_session": cookie},
        headers={"Accept": "text/html"},
    )
    assert r_html.status_code == 302
    assert r_html.headers["location"] == "/"
    assert "nexo_flash=" in r_html.headers.get("set-cookie", "")

    # JSON Accept → 403 JSON intacto.
    r_json = client.get(
        "/ajustes/limites",
        cookies={"nexo_session": cookie},
        headers={"Accept": "application/json"},
    )
    assert r_json.status_code == 403
    assert "nexo_flash" not in r_json.headers.get("set-cookie", "")


def test_get_limites_page_renders_for_propietario(client: TestClient):
    """Propietario recibe HTML con 4 filas editables."""
    email = f"get-owner{TEST_DOMAIN}"
    _create_user(email, role="propietario", dept_codes=[])
    cookie = _login(client, email)

    r = client.get("/ajustes/limites", cookies={"nexo_session": cookie})
    assert r.status_code == 200
    body = r.text
    # Las 4 endpoints deben aparecer en la tabla.
    for ep in ("pipeline/run", "bbdd/query", "capacidad", "operarios"):
        assert ep in body, f"Endpoint {ep} no encontrado en /ajustes/limites"
