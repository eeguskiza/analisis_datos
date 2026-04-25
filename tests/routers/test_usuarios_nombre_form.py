"""Regression: /ajustes/usuarios manages local Nexo users.

Locks:
1. The HTML form renders username/name/surname/email/password fields.
2. POST /ajustes/usuarios/crear persists username, name, surname, correo
   and derived legacy nombre.
3. POST /{id}/editar round-trips the identity fields.
4. Phase 5 `usuarios:manage` permission still guards the endpoint — a
   non-propietario POST gets 403 (no silent write).

Integration — requiere Postgres arriba. Mismo patron que
``tests/routers/test_thresholds_crud.py``: TestClient + users
``@usuarios-nombre-test.local`` + purge en teardown + reset rate limiter.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select, text

from nexo.data.engines import SessionLocalNexo
from nexo.data.models_nexo import (
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


TEST_DOMAIN = "@usuarios-nombre-test.local"
TEST_PASSWORD = "usuariosnombretest12"


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    from api.main import app

    with TestClient(app, follow_redirects=False) as c:
        yield c


@pytest.fixture(autouse=True)
def _cleanup() -> Iterator[None]:
    _purge()
    _reset_rate_limit()
    yield
    _purge()


def _reset_rate_limit() -> None:
    try:
        from api.rate_limit import limiter

        limiter.reset()
    except Exception:
        pass


def _purge() -> None:
    db = SessionLocalNexo()
    try:
        users = (
            db.execute(select(NexoUser).where(NexoUser.email.like(f"%{TEST_DOMAIN}")))
            .scalars()
            .all()
        )
        for u in users:
            db.execute(delete(NexoSession).where(NexoSession.user_id == u.id))
            db.delete(u)
        db.execute(delete(NexoLoginAttempt).where(NexoLoginAttempt.email.like(f"%{TEST_DOMAIN}")))
        db.commit()
    finally:
        db.close()


def _create_user(
    email: str,
    role: str,
    dept_codes: list[str] | None = None,
    nombre: str | None = None,
) -> NexoUser:
    db = SessionLocalNexo()
    try:
        u = NexoUser(
            username=email.split("@", 1)[0],
            name=nombre.split(" ", 1)[0] if nombre else "Test",
            surname=nombre.split(" ", 1)[1] if nombre and " " in nombre else "User",
            email=email,
            nombre=nombre,
            password_hash=hash_password(TEST_PASSWORD),
            role=role,
            active=True,
            must_change_password=False,
        )
        db.add(u)
        db.flush()
        if dept_codes:
            from nexo.data.models_nexo import NexoDepartment

            depts = (
                db.execute(select(NexoDepartment).where(NexoDepartment.code.in_(dept_codes)))
                .scalars()
                .all()
            )
            u.departments = list(depts)
        db.commit()
        db.refresh(u)
        return u
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


def _fetch_user(email: str) -> NexoUser | None:
    db = SessionLocalNexo()
    try:
        return db.execute(select(NexoUser).where(NexoUser.email == email)).scalar_one_or_none()
    finally:
        db.close()


# ── Tests ─────────────────────────────────────────────────────────────


def test_form_renders_nombre_field(client: TestClient) -> None:
    """La lista /ajustes/usuarios incluye los campos de identidad."""
    email = f"owner-form{TEST_DOMAIN}"
    _create_user(email, role="propietario")
    cookie = _login(client, email)

    r = client.get("/ajustes/usuarios", cookies={"nexo_session": cookie})
    assert r.status_code == 200
    body = r.text
    assert 'name="username"' in body
    assert 'name="name"' in body
    assert 'name="surname"' in body
    assert 'name="email"' in body
    assert 'name="password"' in body


def test_post_crear_with_identity_persists(client: TestClient) -> None:
    """POST /ajustes/usuarios/crear persiste la identidad completa."""
    email = f"owner-create{TEST_DOMAIN}"
    _create_user(email, role="propietario")
    cookie = _login(client, email)

    new_email = f"ada{TEST_DOMAIN}"
    r = client.post(
        "/ajustes/usuarios/crear",
        cookies={"nexo_session": cookie},
        data={
            "username": "ada",
            "name": "Ada",
            "surname": "Lovelace",
            "email": new_email,
            "password": "AdaPasswordValid123",
            "password_repetir": "AdaPasswordValid123",
            "role": "usuario",
            "departments": ["ingenieria"],
        },
    )
    assert r.status_code in (200, 302, 303), f"body={r.text[:300]}"
    row = _fetch_user(new_email)
    assert row is not None
    assert row.username == "ada"
    assert row.name == "Ada"
    assert row.surname == "Lovelace"
    assert row.nombre == "Ada Lovelace"


def test_post_crear_requires_name_and_surname(client: TestClient) -> None:
    """Nombre y apellidos son obligatorios en el alta del propietario."""
    email = f"owner-empty{TEST_DOMAIN}"
    _create_user(email, role="propietario")
    cookie = _login(client, email)

    new_email = f"no-name{TEST_DOMAIN}"
    r = client.post(
        "/ajustes/usuarios/crear",
        cookies={"nexo_session": cookie},
        data={
            "username": "no-name",
            "name": "   ",
            "surname": "",
            "email": new_email,
            "password": "NoNamePasswordValid12",
            "password_repetir": "NoNamePasswordValid12",
            "role": "usuario",
        },
    )
    assert r.status_code == 200
    assert "Nombre obligatorio" in r.text
    row = _fetch_user(new_email)
    assert row is None


def test_post_editar_round_trip_identity(client: TestClient) -> None:
    """POST /{id}/editar actualiza username, nombre, apellidos y correo."""
    owner_email = f"owner-edit{TEST_DOMAIN}"
    _create_user(owner_email, role="propietario")
    cookie = _login(client, owner_email)

    target_email = f"target{TEST_DOMAIN}"
    target = _create_user(target_email, role="usuario", dept_codes=["rrhh"], nombre=None)
    new_email = f"target-new{TEST_DOMAIN}"

    r = client.post(
        f"/ajustes/usuarios/{target.id}/editar",
        cookies={"nexo_session": cookie},
        data={
            "username": "bob",
            "name": "Bob",
            "surname": "Builder",
            "email": new_email,
            "role": "usuario",
            "departments": ["rrhh"],
            "active": "on",
        },
    )
    assert r.status_code in (200, 302, 303), f"body={r.text[:300]}"
    row = _fetch_user(new_email)
    assert row is not None
    assert row.username == "bob"
    assert row.name == "Bob"
    assert row.surname == "Builder"
    assert row.nombre == "Bob Builder"


def test_non_propietario_cannot_post_crear(client: TestClient) -> None:
    """Phase 5 RBAC: un rol distinto de propietario recibe 403/redirect.

    ``usuarios:manage`` mapea a ``[]`` en PERMISSION_MAP -> lista vacia =
    propietario-only. Cualquier otro rol (incluso ``directivo``) debe
    ser rechazado por ``require_permission("usuarios:manage")``.
    """
    ing_email = f"ingeniero{TEST_DOMAIN}"
    _create_user(ing_email, role="directivo", dept_codes=["ingenieria"])
    cookie = _login(client, ing_email)

    r = client.post(
        "/ajustes/usuarios/crear",
        cookies={"nexo_session": cookie},
        data={
            "username": "evil",
            "name": "Evil",
            "surname": "Hacker",
            "email": f"evil{TEST_DOMAIN}",
            "password": "EvilPasswordValid12",
            "password_repetir": "EvilPasswordValid12",
            "role": "usuario",
        },
    )
    # Phase 5 FlashMiddleware: HTML 403 se convierte en 303+cookie al /;
    # JSON/HTMX se queda en 403 literal.
    assert r.status_code in (302, 303, 403), (
        f"usuarios:manage debe bloquear a non-propietario; recibido {r.status_code}"
    )

    # Y sobre todo: la escritura NO debe haber ocurrido.
    row = _fetch_user(f"evil{TEST_DOMAIN}")
    assert row is None, "RBAC fail -> se creo el usuario silenciosamente"
