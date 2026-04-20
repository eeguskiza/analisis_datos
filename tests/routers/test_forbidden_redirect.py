"""403 redirect + flash toast integration tests (Plan 05-03, D-07/D-08).

Cobertura end-to-end del handler ``http_exception_handler_403`` +
``FlashMiddleware`` + render del toast en ``base.html``:

- Usuario NO propietario (rrhh) hace GET /ajustes con ``Accept: text/html``
  → 302 a ``/`` + cookie ``nexo_flash`` con texto user-friendly.
- Siguiente GET / con cookie flash → 200, body contiene el bloque
  ``showToast('info', 'Aviso', ...)``.
- Tercer GET / sin cookie → 200, body NO contiene el bloque showToast
  (read-and-clear funcionó).
- API path (Accept: application/json) → 403 JSON con
  ``{"detail": "Permiso requerido: ajustes:manage"}``, sin Set-Cookie
  ``nexo_flash`` (contract estable).
- HX-Request path → 403 JSON (no redirect — HTMX espera contract).
- Read-and-clear directo: GET / con cookie preexistente → response
  incluye ``Set-Cookie: nexo_flash=`` con ``Max-Age=0`` o ``expires=``.
- No regression: 404 y 500 siguen respondiendo como el handler default
  (404 JSON/HTML default de FastAPI; 500 pasa por NAMING-07).

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


TEST_DOMAIN = "@forbidden-redirect-test.local"
TEST_PASSWORD = "forbiddenredirecttest12345"


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


# ── Tests ─────────────────────────────────────────────────────────────────


def test_html_403_redirects_with_flash_cookie(client: TestClient):
    """GET /ajustes con HTML Accept → 302 a /, cookie flash con user-friendly."""
    email = f"rrhh-html{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["rrhh"])
    cookie = _login(client, email)

    r = client.get(
        "/ajustes",
        cookies={"nexo_session": cookie},
        headers={"Accept": "text/html"},
    )

    assert r.status_code == 302, f"Expected 302, got {r.status_code}"
    assert r.headers["location"] == "/", (
        f"Expected redirect to /, got {r.headers.get('location')}"
    )

    set_cookie = r.headers.get("set-cookie", "")
    assert "nexo_flash=" in set_cookie, (
        f"Expected nexo_flash in Set-Cookie, got: {set_cookie!r}"
    )
    # El mensaje debe ser el user-friendly con URL-encoding (espacios → %20).
    # El friendly label de ``ajustes:manage`` es "la configuracion".
    # Starlette URL-encodea el valor, así que buscamos ambos formatos:
    assert (
        "No tienes permiso para acceder a" in set_cookie
        or "No%20tienes%20permiso%20para%20acceder%20a" in set_cookie
    ), f"Expected friendly message in cookie, got: {set_cookie!r}"


def _extract_flash_cookie(response) -> str | None:
    """Extrae el valor de ``nexo_flash`` del header Set-Cookie, si presente.

    Starlette testserver corre sobre http:// y no retiene cookies ``Secure``
    en el jar automáticamente. Los tests usan este helper para simular
    manualmente el browser: leer Set-Cookie y reenviar en el próximo request.
    """
    set_cookie = response.headers.get("set-cookie", "")
    if "nexo_flash=" not in set_cookie:
        return None
    # Parse: ``nexo_flash="valor"; HttpOnly; Max-Age=60; Path=/; ...``
    first = set_cookie.split(",")[0] if "nexo_flash=" in set_cookie.split(",")[0] else set_cookie
    parts = first.split(";")
    for p in parts:
        p = p.strip()
        if p.startswith("nexo_flash="):
            val = p[len("nexo_flash="):]
            # Strip surrounding quotes si los hay.
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            return val
    return None


def test_next_html_request_renders_flash_toast(client: TestClient):
    """GET / con cookie flash → 200, body contiene showToast.

    Simula el flujo browser reenviando manualmente la cookie flash en el
    segundo request (testserver http:// dropea cookies Secure en el jar,
    así que no podemos confiar en persistencia automática).
    """
    email = f"rrhh-toast{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["rrhh"])
    cookie = _login(client, email)

    # Request 1: dispara 403 → response trae Set-Cookie nexo_flash.
    r1 = client.get(
        "/ajustes",
        cookies={"nexo_session": cookie},
        headers={"Accept": "text/html"},
    )
    assert r1.status_code == 302
    flash_val = _extract_flash_cookie(r1)
    assert flash_val is not None, (
        f"Expected nexo_flash in Set-Cookie, got: {r1.headers.get('set-cookie', '')!r}"
    )

    # Request 2: GET / reenviando AMBAS cookies (browser-like).
    r2 = client.get(
        "/",
        cookies={"nexo_session": cookie, "nexo_flash": flash_val},
        headers={"Accept": "text/html"},
    )
    assert r2.status_code == 200, f"status={r2.status_code}"
    assert "showToast('info', 'Aviso'" in r2.text, (
        "Expected showToast('info', 'Aviso', ...) in rendered body"
    )


def test_flash_is_cleared_after_read(client: TestClient):
    """Tercer GET / (sin cookie) → body NO contiene showToast."""
    email = f"rrhh-cleared{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["rrhh"])
    cookie = _login(client, email)

    # Trigger 403 → set flash cookie.
    r1 = client.get(
        "/ajustes",
        cookies={"nexo_session": cookie},
        headers={"Accept": "text/html"},
    )
    assert r1.status_code == 302
    flash_val = _extract_flash_cookie(r1)
    assert flash_val is not None

    # Next request: consume + clear. El server responde 200 con showToast y
    # Set-Cookie: nexo_flash=; Max-Age=0 (FlashMiddleware la borra).
    r2 = client.get(
        "/",
        cookies={"nexo_session": cookie, "nexo_flash": flash_val},
        headers={"Accept": "text/html"},
    )
    assert r2.status_code == 200
    assert "showToast('info', 'Aviso'" in r2.text
    # Verify deletion signal (W-04 relaxed).
    set_cookie_r2 = r2.headers.get("set-cookie", "")
    assert "nexo_flash=" in set_cookie_r2
    assert (
        "max-age=0" in set_cookie_r2.lower()
        or "expires=" in set_cookie_r2.lower()
    )

    # Third request: simulamos que el browser ya borró la cookie — sólo
    # session. Body NO debe contener showToast.
    r3 = client.get(
        "/",
        cookies={"nexo_session": cookie},
        headers={"Accept": "text/html"},
    )
    assert r3.status_code == 200
    assert "showToast('info', 'Aviso'" not in r3.text, (
        "Flash toast should NOT render on third request (read-and-clear)"
    )


def test_api_json_403_preserves_contract(client: TestClient):
    """GET /ajustes con Accept: application/json → 403 JSON, no redirect."""
    email = f"rrhh-json{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["rrhh"])
    cookie = _login(client, email)

    r = client.get(
        "/ajustes",
        cookies={"nexo_session": cookie},
        headers={"Accept": "application/json"},
    )

    assert r.status_code == 403, f"Expected 403, got {r.status_code}"
    body = r.json()
    assert body.get("detail", "").startswith("Permiso requerido:"), (
        f"Unexpected detail: {body}"
    )
    # No debe haber set-cookie nexo_flash.
    set_cookie = r.headers.get("set-cookie", "")
    assert "nexo_flash" not in set_cookie, (
        f"Unexpected nexo_flash in Set-Cookie for JSON 403: {set_cookie!r}"
    )


def test_hx_request_403_preserves_contract(client: TestClient):
    """GET /ajustes con HX-Request → 403 (HTMX espera contract, no redirect)."""
    email = f"rrhh-hx{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["rrhh"])
    cookie = _login(client, email)

    r = client.get(
        "/ajustes",
        cookies={"nexo_session": cookie},
        headers={"HX-Request": "true"},
    )

    assert r.status_code == 403, (
        f"Expected 403 for HX-Request, got {r.status_code}"
    )
    set_cookie = r.headers.get("set-cookie", "")
    assert "nexo_flash" not in set_cookie, (
        f"Unexpected nexo_flash in Set-Cookie for HX 403: {set_cookie!r}"
    )


def test_flash_cookie_is_cleared_after_read_with_relaxed_assertion(
    client: TestClient,
):
    """GET / con cookie flash preset → response borra la cookie (W-04).

    Test directo del read-and-clear: pasamos una cookie flash
    arbitraria via kwarg y validamos que el middleware la borra en la
    response. Assertion relaxed: acepta ``Max-Age=0`` OR ``expires=``
    (cualquier format que Starlette emita).
    """
    email = f"rrhh-w04{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["rrhh"])
    cookie = _login(client, email)

    r = client.get(
        "/",
        cookies={"nexo_session": cookie, "nexo_flash": "hola"},
        headers={"Accept": "text/html"},
    )
    assert r.status_code == 200

    set_cookie = r.headers.get("set-cookie", "")
    assert "nexo_flash=" in set_cookie, (
        f"Expected Set-Cookie: nexo_flash=... (delete), got: {set_cookie!r}"
    )
    # W-04 mitigation: relaxed assertion.
    assert (
        "max-age=0" in set_cookie.lower()
        or "expires=" in set_cookie.lower()
    ), f"Expected cookie deletion marker in Set-Cookie: {set_cookie!r}"


def test_404_not_regressed_by_new_handler(client: TestClient):
    """Ruta inexistente sigue respondiendo como el handler default.

    Handler nuevo delega a ``_default_http_handler`` cuando status != 403,
    así que 404 no cambia. Usamos sesion válida para evitar el redirect a
    /login del AuthMiddleware (ruta no pública). Con session + ruta
    inexistente, FastAPI responde 404 directo (sin tocar nuestro handler
    de 403, que solo actúa sobre status==403).
    """
    email = f"rrhh-404{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["rrhh"])
    cookie = _login(client, email)

    r = client.get(
        "/ruta-que-no-existe-404",
        cookies={"nexo_session": cookie},
        headers={"Accept": "application/json"},
    )
    assert r.status_code == 404, f"Expected 404, got {r.status_code}"
    # No flash cookie (404 no es 403 → handler delega a default).
    set_cookie = r.headers.get("set-cookie", "")
    assert "nexo_flash" not in set_cookie, (
        f"404 should not set nexo_flash cookie, got: {set_cookie!r}"
    )


def test_propietario_ajustes_not_redirected(client: TestClient):
    """Sanity: propietario accede a /ajustes 200 (handler solo actua en 403)."""
    email = f"owner{TEST_DOMAIN}"
    _create_user(email, role="propietario", dept_codes=[])
    cookie = _login(client, email)

    r = client.get(
        "/ajustes",
        cookies={"nexo_session": cookie},
        headers={"Accept": "text/html"},
    )
    assert r.status_code == 200, (
        f"Propietario debería ver /ajustes 200, got {r.status_code}"
    )


def test_flash_label_coverage():
    """W-06 mitigation: todos los permisos HTML-guarded tienen friendly label.

    Cubre los permisos que gaten rutas HTML en Plan 05-05 + ajustes:manage
    + conexion:config. Si alguno falta en ``_PERMISSION_LABELS``, el
    handler devolveria el permiso raw como texto — degraded UX pero no
    roto. Este test fuerza a añadir el label cuando se añade un permiso
    nuevo al plan.
    """
    from api.main import _PERMISSION_LABELS

    HTML_GUARDED_PERMS = [
        "pipeline:read",
        "recursos:read",
        "historial:read",
        "ciclos:read",
        "operarios:read",
        "datos:read",
        "bbdd:read",
        "capacidad:read",
        "ajustes:manage",
        "conexion:config",
    ]
    for p in HTML_GUARDED_PERMS:
        assert p in _PERMISSION_LABELS, (
            f"{p} missing friendly label in api.main._PERMISSION_LABELS"
        )
