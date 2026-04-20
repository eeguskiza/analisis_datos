"""End-to-end LISTEN/NOTIFY integration test (Plan 04-04 / D-19).

Verifica el flujo completo:
  1. ``with TestClient(app) as client`` arranca el lifespan, que hidrata
     el cache y lanza ``start_listener`` en un asyncio.to_thread.
  2. Propietario hace ``PUT /api/thresholds/pipeline/run`` cambiando
     ``warn_ms``.
  3. El router emite ``NOTIFY nexo_thresholds_changed, 'pipeline/run'``.
  4. El listener (corriendo en thread pool) recibe el NOTIFY y llama
     ``reload_one`` sobre el mismo cache que el test observa.
  5. Poll ``thresholds_cache.get('pipeline/run').warn_ms`` hasta que
     refleje el nuevo valor — latencia esperada < 1.5s.

Integration — requiere Postgres up + schema nexo inicializado.
"""
from __future__ import annotations

import time
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select, text

from nexo.data.engines import SessionLocalNexo
from nexo.data.models_nexo import (
    NexoDepartment,
    NexoLoginAttempt,
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


TEST_DOMAIN = "@listen-notify-test.local"
TEST_PASSWORD = "listennotifytest12345"
LISTEN_LATENCY_DEADLINE_S = 3.0  # Margen generoso; objetivo real <1.5s.


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    from api.main import app
    # `with TestClient(app)` activa lifespan -> thresholds_cache.full_reload
    # + start_listener se lanzan.
    with TestClient(app, follow_redirects=False) as c:
        # Dar un pequeño respiro para que el listener entre en LISTEN.
        time.sleep(1.5)
        yield c


@pytest.fixture(autouse=True)
def _cleanup():
    _purge()
    _reset_rate_limit()
    _restore_seed()
    yield
    _purge()
    _restore_seed()


def _reset_rate_limit() -> None:
    try:
        from api.rate_limit import limiter
        limiter.reset()
    except Exception:
        pass


def _restore_seed() -> None:
    db = SessionLocalNexo()
    try:
        db.execute(
            text("UPDATE nexo.query_thresholds "
                 "SET warn_ms = 120000, block_ms = 600000, factor_ms = 2000.0, "
                 "    factor_updated_at = NULL, updated_by = NULL "
                 "WHERE endpoint = 'pipeline/run'")
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


def _create_user(email: str, role: str) -> NexoUser:
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
        if role == "propietario":
            pass  # propietario no necesita departamentos (bypass).
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


def test_end_to_end_put_threshold_propagates_via_listen_notify(client: TestClient):
    """PUT /api/thresholds → NOTIFY → listener → reload_one → cache updated <1.5s."""
    from nexo.services import thresholds_cache

    email = f"e2e-owner{TEST_DOMAIN}"
    _create_user(email, role="propietario")
    cookie = _login(client, email)

    # Baseline: asegura cache sincronizado con DB seed.
    thresholds_cache.full_reload()
    baseline = thresholds_cache.get("pipeline/run")
    assert baseline is not None
    assert baseline.warn_ms == 120_000
    baseline_loaded_at = baseline.loaded_at

    # PUT: cambiar warn_ms a un valor reconocible.
    new_warn = 77_777
    r = client.put(
        "/api/thresholds/pipeline%2Frun",
        cookies={"nexo_session": cookie},
        json={"warn_ms": new_warn, "block_ms": 600_000},
    )
    assert r.status_code == 200, f"body={r.text[:300]}"

    # Poll hasta ver el cache actualizado. El listener corre en el mismo
    # proceso (dentro de la app del TestClient) por lo que comparte el
    # dict _cache. Tras recibir NOTIFY, reload_one actualiza la entry.
    deadline = time.monotonic() + LISTEN_LATENCY_DEADLINE_S
    updated = False
    latency_s = None
    start = time.monotonic()
    while time.monotonic() < deadline:
        entry = thresholds_cache.get("pipeline/run")
        if entry is not None and entry.warn_ms == new_warn and entry.loaded_at > baseline_loaded_at:
            updated = True
            latency_s = time.monotonic() - start
            break
        time.sleep(0.05)

    assert updated, (
        f"Cache no propago el cambio en {LISTEN_LATENCY_DEADLINE_S}s — "
        f"LISTEN/NOTIFY roto o listener no arranco"
    )
    # Log de diagnostico (aparece con -s). No falla si latency_s > 1.5
    # porque el entorno de CI puede ser mas lento que el objetivo humano
    # de <1s; el deadline funcional ya es 3s arriba.
    print(f"\nLISTEN/NOTIFY propagation latency: {latency_s:.3f}s")
