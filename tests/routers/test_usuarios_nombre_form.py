"""Regression for Phase 8 / Plan 08-03: /ajustes/usuarios accepts nombre.

Locks:
1. The HTML form renders the nombre input with the UI-SPEC literal label.
2. POST /ajustes/usuarios/crear with nombre=Ada persists correctly.
3. POST sin nombre (empty form value) stores NULL.
4. POST /{id}/editar round-trips nombre (update from None -> "Bob", then
   back to None via whitespace).
5. Phase 5 `usuarios:manage` permission still guards the endpoint — a
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
    """La lista /ajustes/usuarios incluye el input `name="nombre"` con la
    etiqueta literal de UI-SPEC §Label pattern (D-24/D-26)."""
    email = f"owner-form{TEST_DOMAIN}"
    _create_user(email, role="propietario")
    cookie = _login(client, email)

    r = client.get("/ajustes/usuarios", cookies={"nexo_session": cookie})
    assert r.status_code == 200
    body = r.text
    assert 'name="nombre"' in body, "El form debe tener el input con name=nombre"
    # UI-SPEC §Label pattern (D-24 + D-26) — etiqueta literal:
    assert 'Nombre <span class="text-muted font-normal">(opcional)</span>' in body, (
        "La etiqueta del campo nombre debe seguir la plantilla D-24/D-26 literal"
    )


def test_post_crear_with_nombre_persists(client: TestClient) -> None:
    """POST /ajustes/usuarios/crear con nombre=Ada persiste la columna."""
    email = f"owner-create{TEST_DOMAIN}"
    _create_user(email, role="propietario")
    cookie = _login(client, email)

    new_email = f"ada{TEST_DOMAIN}"
    r = client.post(
        "/ajustes/usuarios/crear",
        cookies={"nexo_session": cookie},
        data={
            "email": new_email,
            "password": "AdaPasswordValid123",
            "password_repetir": "AdaPasswordValid123",
            "role": "usuario",
            "nombre": "Ada Lovelace",
            "departments": ["ingenieria"],
        },
    )
    assert r.status_code in (200, 302, 303), f"body={r.text[:300]}"
    row = _fetch_user(new_email)
    assert row is not None
    assert row.nombre == "Ada Lovelace"


def test_post_crear_empty_nombre_stores_null(client: TestClient) -> None:
    """POST con nombre='' o whitespace → persiste NULL (fallback al email)."""
    email = f"owner-empty{TEST_DOMAIN}"
    _create_user(email, role="propietario")
    cookie = _login(client, email)

    new_email = f"no-name{TEST_DOMAIN}"
    r = client.post(
        "/ajustes/usuarios/crear",
        cookies={"nexo_session": cookie},
        data={
            "email": new_email,
            "password": "NoNamePasswordValid12",
            "password_repetir": "NoNamePasswordValid12",
            "role": "usuario",
            "nombre": "   ",  # whitespace-only
        },
    )
    assert r.status_code in (200, 302, 303)
    row = _fetch_user(new_email)
    assert row is not None
    assert row.nombre is None, "whitespace-only nombre debe persistirse como NULL"


def test_post_editar_round_trip_nombre(client: TestClient) -> None:
    """POST /{id}/editar actualiza nombre (None -> "Bob"; "Bob" -> None)."""
    owner_email = f"owner-edit{TEST_DOMAIN}"
    _create_user(owner_email, role="propietario")
    cookie = _login(client, owner_email)

    # Target: usuario con nombre=None inicialmente.
    target_email = f"target{TEST_DOMAIN}"
    target = _create_user(target_email, role="usuario", dept_codes=["rrhh"], nombre=None)

    # --- Set nombre = "Bob" ---
    r = client.post(
        f"/ajustes/usuarios/{target.id}/editar",
        cookies={"nexo_session": cookie},
        data={
            "role": "usuario",
            "nombre": "Bob",
            "departments": ["rrhh"],
            "active": "on",
        },
    )
    assert r.status_code in (200, 302, 303), f"body={r.text[:300]}"
    row = _fetch_user(target_email)
    assert row is not None
    assert row.nombre == "Bob"

    # --- Clear nombre via whitespace -> None ---
    r = client.post(
        f"/ajustes/usuarios/{target.id}/editar",
        cookies={"nexo_session": cookie},
        data={
            "role": "usuario",
            "nombre": "   ",
            "departments": ["rrhh"],
            "active": "on",
        },
    )
    assert r.status_code in (200, 302, 303)
    row = _fetch_user(target_email)
    assert row is not None
    assert row.nombre is None


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
            "email": f"evil{TEST_DOMAIN}",
            "password": "EvilPasswordValid12",
            "password_repetir": "EvilPasswordValid12",
            "role": "usuario",
            "nombre": "Evil Hacker",
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
