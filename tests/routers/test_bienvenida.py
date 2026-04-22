"""Regression for Phase 8 / Plan 08-04: /bienvenida landing.

Locks:
1. Post-login redirect targets /bienvenida (not /).
2. must_change_password branch still redirects to /cambiar-password.
3. GET /bienvenida renders for any authenticated user (propietario + usuario).
4. GET /bienvenida without cookie redirects to /login (AuthMiddleware global).
5. hora_saludo filter returns the correct Spanish band per hour (boundary
   cases 06/11/12/20/21/05).
6. The template renders the reloj Alpine component + greeting + primary CTA.

Patrón reutilizado de ``tests/routers/test_html_get_guarded.py`` y
``tests/auth/test_rbac_smoke.py`` — integration tests que requieren
Postgres arriba y limpian usuarios del dominio @bienvenida-test.local.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterator
from zoneinfo import ZoneInfo
from types import SimpleNamespace

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


_MADRID = ZoneInfo("Europe/Madrid")


# ── Filter bands — unit tests (no DB) ────────────────────────────────────────


@pytest.mark.parametrize(
    "hour, expected",
    [
        (0, "Buenas noches"),
        (5, "Buenas noches"),
        (6, "Buenos días"),
        (11, "Buenos días"),
        (12, "Buenas tardes"),
        (20, "Buenas tardes"),
        (21, "Buenas noches"),
        (23, "Buenas noches"),
    ],
)
def test_hora_saludo_filter_bands(hour: int, expected: str):
    """Boundary cases of the 3-band greeting filter (D-23)."""
    from api.deps import hora_saludo

    dt = datetime(2026, 4, 22, hour, 0, 0, tzinfo=_MADRID)
    assert hora_saludo(dt) == expected


def test_hora_saludo_naive_datetime_assumed_madrid():
    """Naive datetime is treated as Europe/Madrid local time."""
    from api.deps import hora_saludo

    naive = datetime(2026, 4, 22, 10, 0, 0)
    assert hora_saludo(naive) == "Buenos días"


def test_hora_saludo_none_uses_now():
    """`hora_saludo()` without arg falls back to current time."""
    from api.deps import hora_saludo

    result = hora_saludo(None)
    assert result in {"Buenos días", "Buenas tardes", "Buenas noches"}


def test_hora_saludo_converts_aware_datetime_to_madrid():
    """Aware non-Madrid datetime is converted to Madrid before banding.

    A UTC datetime at 10:00 UTC falls in the "Buenas tardes" band when
    converted to Europe/Madrid (UTC+2 in April) → 12:00 Madrid.
    """
    from api.deps import hora_saludo

    utc = ZoneInfo("UTC")
    dt = datetime(2026, 4, 22, 10, 0, 0, tzinfo=utc)
    # En abril Madrid es UTC+2 (DST); 10:00 UTC = 12:00 Madrid → tardes.
    assert hora_saludo(dt) == "Buenas tardes"


def test_user_display_name_prefers_nombre_and_shortens_to_first_name():
    from api.deps import user_display_name

    user = SimpleNamespace(
        nombre="Erik Eguskiza",
        email="e.eguskiza@ecsmobility.com",
    )

    assert user_display_name(user) == "Erik Eguskiza"
    assert user_display_name(user, first_name_only=True) == "Erik"


def test_user_display_name_fallback_cleans_email_local_part():
    from api.deps import user_display_name

    user = SimpleNamespace(
        nombre=None,
        email="e.eguskiza@ecsmobility.com",
    )

    assert user_display_name(user) == "E Eguskiza"
    assert user_display_name(user, first_name_only=True) == "E"


# ── Integration tests (require Postgres) ──────────────────────────────────


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


_integration = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="Postgres no arriba: docker compose up -d db",
)


TEST_DOMAIN = "@bienvenida-test.local"
TEST_PASSWORD = "bienvenidatest12345"  # min 12 chars


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


def _create_user(
    email: str,
    role: str,
    dept_codes: list[str],
    *,
    must_change_password: bool = False,
) -> NexoUser:
    db = SessionLocalNexo()
    try:
        user = NexoUser(
            email=email,
            password_hash=hash_password(TEST_PASSWORD),
            role=role,
            active=True,
            must_change_password=must_change_password,
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


def _login(c: TestClient, email: str) -> tuple[int, str, str]:
    """Return (status_code, location, cookie). Does NOT assert status."""
    r = c.post(
        "/login",
        data={"email": email, "password": TEST_PASSWORD},
        headers={"Accept": "text/html"},
    )
    return (
        r.status_code,
        r.headers.get("location", ""),
        r.cookies.get("nexo_session", ""),
    )


# ── Post-login redirect ─────────────────────────────────────────────────────


@_integration
def test_post_login_redirects_to_bienvenida(client: TestClient):
    """Login OK de un propietario normal redirige a /bienvenida (Plan 08-04)."""
    email = f"propietario{TEST_DOMAIN}"
    _create_user(email, role="propietario", dept_codes=[])

    status, location, cookie = _login(client, email)
    assert status in (302, 303), f"login: {status}"
    assert location == "/bienvenida", (
        f"Esperado /bienvenida, recibido {location!r}"
    )
    assert cookie, "debe setear cookie nexo_session"


@_integration
def test_post_login_must_change_password_still_redirects_to_change(
    client: TestClient,
):
    """must_change_password=True preserva redirect a /cambiar-password."""
    email = f"mustchange{TEST_DOMAIN}"
    _create_user(
        email,
        role="usuario",
        dept_codes=["rrhh"],
        must_change_password=True,
    )

    status, location, _cookie = _login(client, email)
    assert status in (302, 303), f"login: {status}"
    assert location == "/cambiar-password", (
        f"must_change_password debe forzar /cambiar-password, recibido {location!r}"
    )


# ── /bienvenida route render ────────────────────────────────────────────────


@_integration
def test_bienvenida_route_renders_for_propietario(client: TestClient):
    """Propietario autenticado ve la landing con todos los componentes."""
    email = f"owner{TEST_DOMAIN}"
    _create_user(email, role="propietario", dept_codes=[])
    _, _, cookie = _login(client, email)
    assert cookie

    r = client.get(
        "/bienvenida",
        cookies={"nexo_session": cookie},
        headers={"Accept": "text/html"},
    )
    assert r.status_code == 200, r.text[:500]
    body = r.text
    assert "Ir a Centro de Mando" in body
    assert 'x-data="bienvenidaPage()"' in body
    # El saludo server-rendered (cualquiera de las 3 bandas).
    assert any(
        g in body for g in ("Buenos días", "Buenas tardes", "Buenas noches")
    ), "debe incluir un saludo por franja horaria"
    # Reloj Alpine (x-text="clock") presente.
    assert 'x-text="clock"' in body


@_integration
def test_bienvenida_route_renders_for_usuario(client: TestClient):
    """Un usuario `rrhh` normal también accede a /bienvenida (no permiso)."""
    email = f"rrhh{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["rrhh"])
    _, _, cookie = _login(client, email)
    assert cookie

    r = client.get(
        "/bienvenida",
        cookies={"nexo_session": cookie},
        headers={"Accept": "text/html"},
    )
    assert r.status_code == 200, r.text[:500]
    assert "Ir a Centro de Mando" in r.text


@_integration
def test_bienvenida_route_requires_auth(client: TestClient):
    """Sin cookie → AuthMiddleware redirige HTML a /login."""
    r = client.get(
        "/bienvenida",
        headers={"Accept": "text/html"},
    )
    assert r.status_code in (302, 303)
    assert r.headers["location"].startswith("/login")


@_integration
def test_bienvenida_template_shows_day_and_date_in_spanish(client: TestClient):
    """El día de la semana y el mes se renderizan en castellano."""
    email = f"date{TEST_DOMAIN}"
    _create_user(email, role="propietario", dept_codes=[])
    _, _, cookie = _login(client, email)
    assert cookie

    r = client.get(
        "/bienvenida",
        cookies={"nexo_session": cookie},
        headers={"Accept": "text/html"},
    )
    assert r.status_code == 200
    body = r.text
    # Al menos uno de los 7 días + uno de los 12 meses debe aparecer.
    dias = [
        "Lunes", "Martes", "Miércoles", "Jueves",
        "Viernes", "Sábado", "Domingo",
    ]
    meses = [
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
    ]
    assert any(d in body for d in dias), "debe incluir un día de la semana"
    assert any(m in body for m in meses), "debe incluir un mes"
    # Formato literal: "Es {dia}, {n} de {mes} de {YYYY}"
    assert "Es " in body and " de " in body
