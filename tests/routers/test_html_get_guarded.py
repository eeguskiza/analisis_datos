"""HTML GET hardening integration tests (Plan 05-05 Task 2 / Pitfall 4).

Verifica que las 8 rutas HTML GET guardadas en ``api/routers/pages.py``
rechazan correctamente a usuarios sin permiso y dejan pasar a los que
lo tienen. Sin este hardening, tipear la URL pasaría del sidebar (que
sí filtra, D-01) pero el D-07 (403 → redirect + flash) no se disparaba.

Cobertura:

- Tabla parametrizada rol × ruta × expected status:
  * rrhh-usuario GET /pipeline → 302 (no pipeline:read)
  * produccion-usuario GET /pipeline → 200
  * rrhh-usuario GET /bbdd → 302 (no bbdd:read)
  * ingenieria-usuario GET /bbdd → 200
  * comercial-usuario GET /operarios → 302 (no operarios:read)
  * rrhh-usuario GET /operarios → 200
  * gerencia-usuario GET /datos → 302 (datos:read = [ingenieria, produccion])
  * produccion-usuario GET /datos → 200
  * comercial-usuario GET /capacidad → 200 (capacidad:read incluye comercial)
  * rrhh-usuario GET /capacidad → 302

- API path: rrhh-usuario GET /bbdd con Accept: application/json
  → 403 JSON con detail estable (Plan 05-03 contract intacto).

Integration — requiere Postgres up + schema ``nexo`` inicializado.
Patrón reusado de ``tests/routers/test_forbidden_redirect.py``.
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


TEST_DOMAIN = "@html-get-guarded-test.local"
TEST_PASSWORD = "htmlgetguardedtest12345"


# ── Helpers (patrón de test_forbidden_redirect.py) ────────────────────────


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


# Mapa rol_tag → (role, dept_codes) para la tabla parametrizada. El
# email se construye a partir del tag + TEST_DOMAIN.
_ROLE_MAP: dict[str, tuple[str, list[str]]] = {
    "rrhh_user":        ("usuario", ["rrhh"]),
    "produccion_user":  ("usuario", ["produccion"]),
    "ingenieria_user":  ("usuario", ["ingenieria"]),
    "comercial_user":   ("usuario", ["comercial"]),
    "gerencia_user":    ("usuario", ["gerencia"]),
}


def _cookie_for(client: TestClient, role_tag: str) -> str:
    role, depts = _ROLE_MAP[role_tag]
    email = f"{role_tag}{TEST_DOMAIN}"
    _create_user(email, role=role, dept_codes=depts)
    return _login(client, email)


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "role_tag,route,expected",
    [
        # Pipeline: produccion sí (pipeline:read), rrhh no.
        ("rrhh_user",       "/pipeline",  302),
        ("produccion_user", "/pipeline",  200),
        # BBDD: ingenieria sí (bbdd:read = [ingenieria]), rrhh no.
        ("rrhh_user",       "/bbdd",      302),
        ("ingenieria_user", "/bbdd",      200),
        # Operarios: rrhh sí (operarios:read = [rrhh]), comercial no.
        ("comercial_user",  "/operarios", 302),
        ("rrhh_user",       "/operarios", 200),
        # Datos: produccion sí (datos:read = [ingenieria, produccion]),
        # gerencia NO (no incluida en datos:read).
        ("gerencia_user",   "/datos",     302),
        ("produccion_user", "/datos",     200),
        # Capacidad: comercial sí (capacidad:read incluye comercial), rrhh no.
        ("comercial_user",  "/capacidad", 200),
        ("rrhh_user",       "/capacidad", 302),
    ],
)
def test_html_get_guarded_per_role(
    client: TestClient, role_tag: str, route: str, expected: int
):
    """Tabla parametrizada: cada rol ↔ ruta ↔ status esperado."""
    cookie = _cookie_for(client, role_tag)
    r = client.get(
        route,
        cookies={"nexo_session": cookie},
        headers={"Accept": "text/html"},
    )
    assert r.status_code == expected, (
        f"{role_tag} GET {route} → expected {expected}, got {r.status_code}"
    )
    if expected == 302:
        assert r.headers["location"] == "/", (
            f"302 debe redirigir a / (D-07 flow), got {r.headers.get('location')}"
        )
        # Debe incluir Set-Cookie: nexo_flash con mensaje user-friendly.
        set_cookie = r.headers.get("set-cookie", "")
        assert "nexo_flash=" in set_cookie, (
            f"302 de {role_tag}→{route} debe setear cookie flash, got: {set_cookie!r}"
        )


def test_html_get_api_path_returns_403_json(client: TestClient):
    """Accept: application/json sobre ruta HTML → 403 JSON, contract estable.

    Verifica que el fix de hardening no rompe el contract JSON: clientes
    API (Accept: application/json) siguen recibiendo 403 con detail
    ``Permiso requerido: <perm>:read`` — no redirect.
    """
    cookie = _cookie_for(client, "rrhh_user")
    r = client.get(
        "/bbdd",
        cookies={"nexo_session": cookie},
        headers={"Accept": "application/json"},
    )
    assert r.status_code == 403, (
        f"Expected 403 for JSON Accept, got {r.status_code}"
    )
    body = r.json()
    assert body.get("detail") == "Permiso requerido: bbdd:read", (
        f"Unexpected detail: {body}"
    )
    # No flash cookie para JSON 403.
    set_cookie = r.headers.get("set-cookie", "")
    assert "nexo_flash" not in set_cookie, (
        f"JSON 403 no debe setear flash cookie, got: {set_cookie!r}"
    )
