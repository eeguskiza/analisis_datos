"""Integration tests for /ajustes hub split (Plan 05-04 / UIROL-03).

Covers:

- propietario GET /ajustes → 200 + HTML contiene 6 hrefs distintos y ninguna
  referencia a SMTP.
- propietario GET /ajustes/conexion → 200 + HTML contiene el componente
  Alpine ``ajustesConexionPage`` (evita colisión de scope con el hub).
- non-propietario GET /ajustes/conexion con ``Accept: text/html`` → 302 a
  ``/`` (flujo D-07 de Plan 05-03 FlashMiddleware).
- non-propietario GET /ajustes/conexion con ``Accept: application/json`` →
  403 JSON (contract estable).
- non-propietario GET /ajustes (hub) → 302 a ``/`` (ajustes:manage →
  propietario-only, no regresión tras el refactor).
- El HTML del hub no contiene la cadena ``SMTP`` ni ``href=/ajustes/smtp``
  (D-04).

Integration — requiere Postgres up + schema ``nexo`` inicializado. Sigue
el patrón de ``tests/routers/test_sidebar_filtering.py`` y
``tests/routers/test_forbidden_redirect.py``.

I-02 mitigation: las asserciones de ausencia de SMTP usan patrón anclado
(``href="/ajustes/smtp"``) en lugar de substring ambiguo para evitar
falsos positivos en el futuro.
"""
from __future__ import annotations

import re
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


TEST_DOMAIN = "@ajustes-split-test.local"
TEST_PASSWORD = "ajustessplittest12345"


# ── Helpers (patrón de test_sidebar_filtering.py / test_forbidden_redirect.py) ─


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


def _assert_href_present(body: str, href: str) -> None:
    """Asserta que ``href="/x"`` aparece en el body (exact match, W-03)."""
    pattern = re.compile(r'href="' + re.escape(href) + r'"(?![/\w])')
    assert pattern.search(body), (
        f'Expected href="{href}" (exact, no sub-path) in response body, '
        f"not found."
    )


def _assert_href_absent(body: str, href: str) -> None:
    """Asserta que ``href="/x"`` NO aparece en el body (exact match)."""
    pattern = re.compile(r'href="' + re.escape(href) + r'"(?![/\w])')
    assert not pattern.search(body), (
        f'Unexpected href="{href}" present in response body.'
    )


# ── Tests ────────────────────────────────────────────────────────────────


def test_ajustes_hub_propietario_sees_six_cards(client: TestClient):
    """Propietario ve las 6 cards esperadas (sin SMTP per D-04)."""
    email = f"owner-6cards{TEST_DOMAIN}"
    _create_user(email, role="propietario", dept_codes=[])
    cookie = _login(client, email)

    r = client.get("/ajustes", cookies={"nexo_session": cookie})
    assert r.status_code == 200, f"status={r.status_code} body={r.text[:200]}"
    body = r.text

    # Las 6 cards esperadas:
    _assert_href_present(body, "/ajustes/conexion")
    _assert_href_present(body, "/ajustes/usuarios")
    _assert_href_present(body, "/ajustes/auditoria")
    _assert_href_present(body, "/ajustes/solicitudes")
    _assert_href_present(body, "/ajustes/limites")
    _assert_href_present(body, "/ajustes/rendimiento")

    # NO debe haber card SMTP (D-04):
    _assert_href_absent(body, "/ajustes/smtp")


def test_ajustes_hub_no_smtp_card(client: TestClient):
    """Confirma explícitamente que el HTML no contiene ``href=/ajustes/smtp``.

    I-02 mitigation: usamos el patrón anclado para evitar falsos positivos
    (por ejemplo, un día alguien podría escribir 'smtp' en una descripción
    legítima y romper un assert tipo ``'SMTP' not in body`` sin querer).
    """
    email = f"owner-nosmtp{TEST_DOMAIN}"
    _create_user(email, role="propietario", dept_codes=[])
    cookie = _login(client, email)

    r = client.get("/ajustes", cookies={"nexo_session": cookie})
    assert r.status_code == 200
    body = r.text

    # Patrón anclado: el link a una hipotética página SMTP no existe.
    assert 'href="/ajustes/smtp"' not in body, (
        'El hub no debe contener href="/ajustes/smtp" (D-04 — SMTP diferido).'
    )


def test_ajustes_conexion_propietario_renders(client: TestClient):
    """Propietario GET /ajustes/conexion → 200 + componente Alpine renombrado."""
    email = f"owner-conexion{TEST_DOMAIN}"
    _create_user(email, role="propietario", dept_codes=[])
    cookie = _login(client, email)

    r = client.get("/ajustes/conexion", cookies={"nexo_session": cookie})
    assert r.status_code == 200, f"status={r.status_code} body={r.text[:200]}"
    body = r.text

    # El componente Alpine renombrado debe estar presente (no colisión con hub).
    assert "ajustesConexionPage" in body, (
        "El componente renombrado ajustesConexionPage() debe aparecer "
        "(x-data + function declaration). Si no aparece, el template "
        "regresó al nombre viejo ajustesPage() o no se extrajo."
    )
    # Endpoints backend se referencian sin modificar (Open Q4):
    assert "/api/conexion/config" in body
    assert "/api/conexion/status" in body


def test_ajustes_conexion_non_propietario_blocked_html(client: TestClient):
    """Usuario non-propietario con Accept HTML → 302 a / (flujo D-07)."""
    email = f"ing-html{TEST_DOMAIN}"
    _create_user(email, role="directivo", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    r = client.get(
        "/ajustes/conexion",
        cookies={"nexo_session": cookie},
        headers={"Accept": "text/html"},
    )
    assert r.status_code == 302, (
        f"Esperado 302 para HTML-GET forbidden, got {r.status_code}"
    )
    assert r.headers["location"] == "/", (
        f"Expected redirect to /, got {r.headers.get('location')}"
    )


def test_ajustes_conexion_non_propietario_blocked_api(client: TestClient):
    """Usuario non-propietario con Accept JSON → 403 JSON (contract)."""
    email = f"ing-json{TEST_DOMAIN}"
    _create_user(email, role="directivo", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    r = client.get(
        "/ajustes/conexion",
        cookies={"nexo_session": cookie},
        headers={"Accept": "application/json"},
    )
    assert r.status_code == 403, (
        f"Esperado 403 para JSON-GET forbidden, got {r.status_code}"
    )
    body = r.json()
    assert body.get("detail", "").startswith("Permiso requerido:"), (
        f"Unexpected detail: {body}"
    )


def test_ajustes_hub_non_propietario_blocked(client: TestClient):
    """Usuario non-propietario GET /ajustes → 302 (hub sigue propietario-only).

    Sanity check: el refactor del hub (gates per-card) NO debe haber
    cambiado el gate de la ruta (ajustes:manage = [] → propietario-only).
    """
    email = f"ing-hub{TEST_DOMAIN}"
    _create_user(email, role="directivo", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    r = client.get(
        "/ajustes",
        cookies={"nexo_session": cookie},
        headers={"Accept": "text/html"},
    )
    assert r.status_code == 302, (
        f"Esperado 302 (ajustes:manage propietario-only), got {r.status_code}"
    )
    assert r.headers["location"] == "/"
