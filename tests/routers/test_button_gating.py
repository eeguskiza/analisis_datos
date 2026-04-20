"""Per-role button visibility integration tests (Plan 05-05 / UIROL-04).

Tras Task 1 (wraps) + Task 2 (HTML GET hardening), verifica que los
botones sensibles aparecen o desaparecen del HTML según el permiso del
usuario. La combinación Jinja ``{% if can(...) %}`` + render server-side
implementa D-02 (zero-trust DOM).

Cobertura:

- Propietario ve "Ejecutar" en /pipeline (pipeline:run bypass).
- Usuario de produccion ve "Ejecutar" en /pipeline (tiene pipeline:run).
- Usuario de gerencia tiene pipeline:read pero NO pipeline:run →
  puede entrar pero NO ve el botón (verificación key: mismo HTTP 200
  pero distinto HTML).
- Propietario ve "Generar PDFs" + "Borrar" en /historial.
- Usuario de produccion tiene historial:read + informes:read pero NO
  informes:delete → puede entrar pero NO ve los botones destructivos.
- Usuario de ingenieria ve botones edit en /recursos (recursos:edit).
- Usuario de produccion tiene recursos:read pero NO :edit → NO ve
  botones edit.
- Usuario de rrhh GET /ciclos-calc → 302 (gate backend Task 2 antes de
  template — verifica interacción wave 5 → wave 3).

Integration — requiere Postgres up + schema ``nexo`` inicializado.
Patrón reusado de ``tests/routers/test_sidebar_filtering.py``.
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


TEST_DOMAIN = "@button-gating-test.local"
TEST_PASSWORD = "buttongatingtest12345"


# ── Helpers (patrón de test_sidebar_filtering.py) ─────────────────────────


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
    """Resetea slowapi para evitar 429 en /login entre tests."""
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


# ── Pipeline tests ───────────────────────────────────────────────────────


def test_pipeline_propietario_sees_ejecutar(client: TestClient):
    """Propietario GET /pipeline → 200 + botón Ejecutar visible (bypass)."""
    email = f"owner{TEST_DOMAIN}"
    _create_user(email, role="propietario", dept_codes=[])
    cookie = _login(client, email)

    r = client.get("/pipeline", cookies={"nexo_session": cookie})
    assert r.status_code == 200
    assert '@click="ejecutar()"' in r.text, (
        "Propietario should see the Ejecutar button (pipeline:run bypass)"
    )


def test_pipeline_produccion_user_sees_ejecutar(client: TestClient):
    """Usuario de produccion GET /pipeline → 200 + botón Ejecutar visible."""
    email = f"prod{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["produccion"])
    cookie = _login(client, email)

    r = client.get("/pipeline", cookies={"nexo_session": cookie})
    assert r.status_code == 200
    assert '@click="ejecutar()"' in r.text, (
        "produccion-usuario tiene pipeline:run y debe ver el boton"
    )


def test_pipeline_gerencia_user_does_not_see_ejecutar(client: TestClient):
    """Gerencia tiene pipeline:read pero no pipeline:run → HTML sin botón."""
    email = f"gerencia{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["gerencia"])
    cookie = _login(client, email)

    r = client.get("/pipeline", cookies={"nexo_session": cookie})
    assert r.status_code == 200, (
        f"gerencia debe poder entrar a /pipeline (tiene pipeline:read), "
        f"got {r.status_code}"
    )
    assert '@click="ejecutar()"' not in r.text, (
        "gerencia-usuario no tiene pipeline:run — el boton NO debe aparecer"
    )


def test_pipeline_gerencia_directivo_sees_page_but_not_ejecutar(client: TestClient):
    """I-04 mitigation: multi-dept gerencia-directivo entra a /pipeline (200)."""
    email = f"gerencia-dir{TEST_DOMAIN}"
    _create_user(
        email, role="directivo", dept_codes=["gerencia", "comercial"]
    )
    cookie = _login(client, email)

    r = client.get("/pipeline", cookies={"nexo_session": cookie})
    assert r.status_code == 200, (
        "gerencia-directivo con pipeline:read debe entrar (200), no bloqueado"
    )
    # No tiene pipeline:run (ni gerencia ni comercial lo tienen).
    assert '@click="ejecutar()"' not in r.text


# ── Historial tests ──────────────────────────────────────────────────────


def test_historial_propietario_sees_delete_and_regen(client: TestClient):
    """Propietario GET /historial → ve botones 'Generar PDFs' y 'Borrar'."""
    email = f"owner-hist{TEST_DOMAIN}"
    _create_user(email, role="propietario", dept_codes=[])
    cookie = _login(client, email)

    r = client.get("/historial", cookies={"nexo_session": cookie})
    assert r.status_code == 200
    assert '@click="regenerar()"' in r.text, (
        "Propietario should see Generar PDFs button"
    )
    assert '@click="borrar()"' in r.text, (
        "Propietario should see Borrar button"
    )


def test_historial_produccion_sees_no_destructive_buttons(client: TestClient):
    """produccion tiene historial:read + informes:read pero NO informes:delete."""
    email = f"prod-hist{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["produccion"])
    cookie = _login(client, email)

    r = client.get("/historial", cookies={"nexo_session": cookie})
    assert r.status_code == 200, (
        "produccion tiene historial:read → entra OK"
    )
    assert '@click="regenerar()"' not in r.text, (
        "produccion NO tiene informes:delete — Generar PDFs debe estar oculto"
    )
    assert '@click="borrar()"' not in r.text, (
        "produccion NO tiene informes:delete — Borrar debe estar oculto"
    )


# ── Recursos tests ───────────────────────────────────────────────────────


def test_recursos_ingenieria_sees_edit_buttons(client: TestClient):
    """Ingenieria-usuario tiene recursos:edit → ve Detectar + Guardar todo."""
    email = f"ing{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    r = client.get("/recursos", cookies={"nexo_session": cookie})
    assert r.status_code == 200
    assert '@click="detectar()"' in r.text, (
        "ingenieria should see Detectar button"
    )
    assert '@click="guardarTodo()"' in r.text, (
        "ingenieria should see Guardar todo button"
    )


def test_recursos_produccion_sees_no_edit_buttons(client: TestClient):
    """produccion tiene recursos:read pero NO :edit → sin botones edit."""
    email = f"prod-rec{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["produccion"])
    cookie = _login(client, email)

    r = client.get("/recursos", cookies={"nexo_session": cookie})
    assert r.status_code == 200, (
        "produccion tiene recursos:read → entra OK"
    )
    assert '@click="detectar()"' not in r.text, (
        "produccion NO tiene recursos:edit — Detectar debe estar oculto"
    )
    assert '@click="guardarTodo()"' not in r.text, (
        "produccion NO tiene recursos:edit — Guardar todo debe estar oculto"
    )


# ── Ciclos calc backend gate interaction ─────────────────────────────────


def test_ciclos_calc_non_ingenieria_blocked_at_get(client: TestClient):
    """rrhh GET /ciclos-calc → 302 (Task 2 gate, no llega al template).

    Interacción wave 5 → wave 3: el gate de ciclos:read en pages.py
    convierte el request en 403; el handler HTML de 05-03 lo transforma
    en 302 a / con flash cookie.
    """
    email = f"rrhh-ciclos{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["rrhh"])
    cookie = _login(client, email)

    r = client.get(
        "/ciclos-calc",
        cookies={"nexo_session": cookie},
        headers={"Accept": "text/html"},
    )
    assert r.status_code == 302, (
        f"rrhh sin ciclos:read debe recibir 302, got {r.status_code}"
    )
    assert r.headers["location"] == "/"


def test_ciclos_calc_ingenieria_sees_save_button(client: TestClient):
    """ingenieria tiene ciclos:edit → ve el botón openSaveDialog."""
    email = f"ing-ciclos{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    r = client.get("/ciclos-calc", cookies={"nexo_session": cookie})
    assert r.status_code == 200
    assert '@click="openSaveDialog(' in r.text, (
        "ingenieria should see the save button in ciclos_calc"
    )
