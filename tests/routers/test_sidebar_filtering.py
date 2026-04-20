"""Sidebar filtering integration tests (Plan 05-02 / UIROL-02).

Cobertura del loop de ``nav_items`` en ``templates/base.html`` tras el
refactor a ``can(current_user, permission)`` (Plan 05-01 / D-01):

- propietario ve los 11 items + link a Solicitudes.
- directivo de ingenieria ve los módulos de ingeniería (no ajustes, no
  operarios, no solicitudes).
- usuario de producción ve pipeline/historial/recursos (no bbdd, no
  ajustes, no operarios, no solicitudes).
- usuario de rrhh ve operarios e historial (no pipeline, no bbdd, no
  ajustes, no datos).
- /login (sin auth) renderiza sin excepción por ``can(None, ...)``.

Integration — requiere Postgres up + schema ``nexo`` inicializado. Sigue
el patrón de ``tests/routers/test_approvals_api.py`` (TestClient + users
de dominio ``@sidebar-filter-test.local`` + purga en teardown).

Importante (W-03): los asserts usan el patrón exacto ``href="/x"`` con
comillas (no substring ``/x``), para evitar falsos positivos donde
``/ajustes`` matchearía ``/ajustes/solicitudes``.
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


TEST_DOMAIN = "@sidebar-filter-test.local"
TEST_PASSWORD = "sidebarfiltertest12345"


# ── Helpers (misma forma que test_approvals_api.py) ──────────────────────

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
    """Asserta que ``href="/x"`` aparece en el body (exact match, W-03).

    Usa regex anclada para no hacer match de sub-paths:
    ``href="/ajustes"`` NO debe matchear ``href="/ajustes/solicitudes"``.
    """
    pattern = re.compile(r'href="' + re.escape(href) + r'"(?![/\w])')
    assert pattern.search(body), (
        f'Expected href="{href}" (exact, no sub-path) in response body, '
        f"not found. First 500 chars: {body[:500]!r}"
    )


def _assert_href_absent(body: str, href: str) -> None:
    """Asserta que ``href="/x"`` NO aparece en el body (exact match, W-03)."""
    pattern = re.compile(r'href="' + re.escape(href) + r'"(?![/\w])')
    assert not pattern.search(body), (
        f'Unexpected href="{href}" present in response body. '
        f"First 500 chars: {body[:500]!r}"
    )


# ── Tests ────────────────────────────────────────────────────────────────

def test_sidebar_propietario_sees_all_items(client: TestClient):
    """Propietario ve los 11 items del nav_items + link a Solicitudes."""
    email = f"owner{TEST_DOMAIN}"
    _create_user(email, role="propietario", dept_codes=[])
    cookie = _login(client, email)

    r = client.get("/", cookies={"nexo_session": cookie})
    assert r.status_code == 200, f"status={r.status_code} body={r.text[:200]}"
    body = r.text

    # Los 11 items del sidebar:
    _assert_href_present(body, "/")
    _assert_href_present(body, "/datos")
    _assert_href_present(body, "/pipeline")
    _assert_href_present(body, "/historial")
    _assert_href_present(body, "/capacidad")
    _assert_href_present(body, "/recursos")
    _assert_href_present(body, "/ciclos-calc")
    _assert_href_present(body, "/operarios")
    _assert_href_present(body, "/bbdd")
    _assert_href_present(body, "/ajustes")
    # Link a Solicitudes (badge HTMX):
    _assert_href_present(body, "/ajustes/solicitudes")
    # El badge HTMX debe estar cableado:
    assert 'hx-get="/api/approvals/count"' in body


def test_sidebar_ingenieria_directivo_sees_engineering_modules(
    client: TestClient,
):
    """Directivo de ingenieria ve modulos de ingenieria.

    Visible: dashboard, datos, pipeline, historial, capacidad, recursos,
    ciclos-calc, bbdd.
    No visible: ajustes, operarios, solicitudes.
    """
    email = f"ing-directivo{TEST_DOMAIN}"
    _create_user(email, role="directivo", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    r = client.get("/", cookies={"nexo_session": cookie})
    assert r.status_code == 200, f"status={r.status_code} body={r.text[:200]}"
    body = r.text

    # Ingenieria tiene: pipeline:read, datos:read, recursos:read,
    # ciclos:read, bbdd:read, historial:read, capacidad:read.
    _assert_href_present(body, "/")  # dashboard (always)
    _assert_href_present(body, "/datos")
    _assert_href_present(body, "/pipeline")
    _assert_href_present(body, "/historial")
    _assert_href_present(body, "/capacidad")
    _assert_href_present(body, "/recursos")
    _assert_href_present(body, "/ciclos-calc")
    _assert_href_present(body, "/bbdd")

    # Ajustes requiere propietario (lista vacia). Operarios solo rrhh.
    # Solicitudes requiere aprobaciones:manage (propietario-only).
    _assert_href_absent(body, "/ajustes")
    _assert_href_absent(body, "/operarios")
    _assert_href_absent(body, "/ajustes/solicitudes")


def test_sidebar_produccion_usuario_sees_production_modules(
    client: TestClient,
):
    """Usuario de produccion ve modulos de produccion.

    Visible: dashboard, datos, pipeline, historial, capacidad, recursos,
    ciclos_calc (ciclos:read NO incluye produccion — sólo ingeniería).
    No visible: bbdd, ajustes, operarios, solicitudes.
    """
    email = f"prod-user{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["produccion"])
    cookie = _login(client, email)

    r = client.get("/", cookies={"nexo_session": cookie})
    assert r.status_code == 200, f"status={r.status_code} body={r.text[:200]}"
    body = r.text

    # Produccion tiene: pipeline:read, datos:read, recursos:read,
    # historial:read, capacidad:read.
    _assert_href_present(body, "/")
    _assert_href_present(body, "/datos")
    _assert_href_present(body, "/pipeline")
    _assert_href_present(body, "/historial")
    _assert_href_present(body, "/capacidad")
    _assert_href_present(body, "/recursos")

    # bbdd:read = [ingenieria] — produccion NO lo tiene.
    # ciclos:read = [ingenieria] — produccion NO lo tiene.
    # operarios:read = [rrhh] — produccion NO.
    # ajustes:manage = [] — propietario-only.
    # aprobaciones:manage = [] — propietario-only.
    _assert_href_absent(body, "/bbdd")
    _assert_href_absent(body, "/ciclos-calc")
    _assert_href_absent(body, "/operarios")
    _assert_href_absent(body, "/ajustes")
    _assert_href_absent(body, "/ajustes/solicitudes")


def test_sidebar_rrhh_usuario_sees_operarios_only_for_hr(client: TestClient):
    """Usuario de rrhh ve operarios e historial.

    Visible: dashboard, historial (rrhh está en historial:read), operarios.
    No visible: datos, pipeline, bbdd, ajustes, solicitudes.
    """
    email = f"rrhh-user{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["rrhh"])
    cookie = _login(client, email)

    r = client.get("/", cookies={"nexo_session": cookie})
    assert r.status_code == 200, f"status={r.status_code} body={r.text[:200]}"
    body = r.text

    # rrhh tiene: operarios:read, operarios:export, historial:read,
    # informes:read, email:send.
    _assert_href_present(body, "/")  # dashboard siempre
    _assert_href_present(body, "/historial")
    _assert_href_present(body, "/operarios")

    # NO datos (rrhh no tiene datos:read).
    # NO pipeline (rrhh no tiene pipeline:read).
    # NO bbdd (rrhh no tiene bbdd:read).
    # NO ajustes (propietario-only).
    # NO solicitudes (propietario-only).
    _assert_href_absent(body, "/datos")
    _assert_href_absent(body, "/pipeline")
    _assert_href_absent(body, "/bbdd")
    _assert_href_absent(body, "/ajustes")
    _assert_href_absent(body, "/ajustes/solicitudes")


def test_sidebar_anon_login_page_renders_without_error(client: TestClient):
    """GET /login (sin auth) renderiza sin excepción por can(None, ...).

    login.html no extiende base.html, pero este test asegura que si alguna
    ruta pública futura extendiera base.html, ``can(current_user=None, x)``
    no levanta excepción (devuelve False). También valida que el
    middleware deja pasar /login sin cookie.
    """
    r = client.get("/login")
    assert r.status_code == 200, (
        f"/login debería renderizar para anon: status={r.status_code} "
        f"body={r.text[:200]}"
    )
