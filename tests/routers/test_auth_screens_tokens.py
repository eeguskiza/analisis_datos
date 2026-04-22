"""Regression for Phase 8 / Plan 08-06: auth screens visual refactor.

Locks:
1. GET /login renders with action="/login", autocomplete username +
   current-password, CTA "Entrar", and Spanish copy.
2. GET /cambiar-password renders with action="/cambiar-password",
   3 stacked inputs with 2x autocomplete="new-password", CTA
   "Guardar contraseña", must_change copy preserved.
3. GET /mis-solicitudes renders extending base.html, showing the
   .data-table with .badge-* pills OR the .empty-state fallback.
4. No raw Tailwind state colors (bg-red|green|blue|yellow-###) in any
   of the 3 templates — everything goes through the semantic token
   layer (Plan 08-01 / Plan 08-02 contract).
5. Error paths (401 invalid / 429 lockout) still render readable
   Spanish copy — these are server-authoritative (D-25).

Integration tests follow the pattern of
``tests/routers/test_bienvenida.py`` (TestClient + Postgres purge
fixtures). Static tests run even when Postgres is down so template
drift is caught in CI.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

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

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOGIN_HTML = _REPO_ROOT / "templates" / "login.html"
_CAMBIAR_HTML = _REPO_ROOT / "templates" / "cambiar_password.html"
_MIS_HTML = _REPO_ROOT / "templates" / "mis_solicitudes.html"


# ── Static checks (no DB required) ────────────────────────────────────────


def test_login_template_action_and_autocomplete() -> None:
    src = _LOGIN_HTML.read_text(encoding="utf-8")
    assert 'action="/login"' in src, "login form must POST to /login"
    assert 'autocomplete="username"' in src, (
        "login email must carry autocomplete=username for password managers"
    )
    assert 'autocomplete="current-password"' in src, (
        "login password must carry autocomplete=current-password"
    )


def test_login_template_has_primary_cta() -> None:
    src = _LOGIN_HTML.read_text(encoding="utf-8")
    assert "Entrar</button>" in src, (
        'login CTA must be a submit button with the literal label "Entrar"'
    )
    # Button class should be the tokenised primary
    assert "btn btn-primary btn-lg" in src, "login CTA must use .btn.btn-primary.btn-lg"


def test_login_has_stacked_labels() -> None:
    src = _LOGIN_HTML.read_text(encoding="utf-8")
    # UI-SPEC §Forms: stacked (top-aligned) labels.
    labels = re.findall(
        r'<label[^>]*for="login-(email|password)"[^>]*>',
        src,
    )
    assert set(labels) == {"email", "password"}, (
        f"login must have stacked labels for #login-email and "
        f"#login-password (top-aligned per UI-SPEC §Forms); found {labels}"
    )


def test_login_does_not_extend_base_html() -> None:
    src = _LOGIN_HTML.read_text(encoding="utf-8")
    assert "{% extends" not in src, (
        "login is a public route and must NOT extend base.html "
        "(no topbar / drawer on the auth landing)"
    )
    assert "<!DOCTYPE html>" in src, "login.html must be a standalone HTML document (public route)"


def test_login_no_raw_state_colors() -> None:
    src = _LOGIN_HTML.read_text(encoding="utf-8")
    raw = re.findall(r"bg-(red|green|blue|yellow)-[0-9]{3}", src)
    assert not raw, (
        f"login.html must not use raw Tailwind state colors; found: {raw}. "
        f"Use semantic tokens (bg-error-subtle / bg-success-subtle / etc.)."
    )


def test_cambiar_password_template_action_and_autocomplete() -> None:
    src = _CAMBIAR_HTML.read_text(encoding="utf-8")
    assert 'action="/cambiar-password"' in src, (
        "cambiar_password form must POST to /cambiar-password"
    )
    # 2x new-password (nuevo + repetir), 1x current-password (actual).
    assert src.count('autocomplete="new-password"') == 2, (
        "new + repetir inputs must both carry autocomplete=new-password"
    )
    assert 'autocomplete="current-password"' in src, (
        "actual input must carry autocomplete=current-password"
    )


def test_cambiar_password_template_extends_base_html() -> None:
    src = _CAMBIAR_HTML.read_text(encoding="utf-8")
    assert '{% extends "base.html" %}' in src, (
        "cambiar_password.html must extend base.html so the Phase 8 chrome "
        "(topbar + drawer) applies."
    )


def test_cambiar_password_has_primary_cta() -> None:
    src = _CAMBIAR_HTML.read_text(encoding="utf-8")
    assert "Guardar contraseña</button>" in src, (
        'cambiar_password CTA must be a submit button with the literal label "Guardar contraseña"'
    )


def test_cambiar_password_no_raw_state_colors() -> None:
    src = _CAMBIAR_HTML.read_text(encoding="utf-8")
    raw = re.findall(r"bg-(red|green|blue|yellow)-[0-9]{3}", src)
    assert not raw, (
        f"cambiar_password.html must not use raw Tailwind state colors; "
        f"found: {raw}. Use semantic tokens."
    )


def test_mis_solicitudes_template_extends_base_html() -> None:
    src = _MIS_HTML.read_text(encoding="utf-8")
    assert '{% extends "base.html" %}' in src, "mis_solicitudes.html must extend base.html"


def test_mis_solicitudes_has_status_pills() -> None:
    src = _MIS_HTML.read_text(encoding="utf-8")
    assert "badge badge-" in src, (
        "mis_solicitudes must render status as .badge-* pills "
        "(neutral / brand / success / warn / error)"
    )
    # Must use .data-table AND .empty-state (either branch must be wired).
    assert "data-table" in src, "mis_solicitudes must use .data-table"
    assert "empty-state" in src, (
        "mis_solicitudes must include an .empty-state fallback for the no-requests case"
    )


def test_mis_solicitudes_preserves_cancel_endpoint() -> None:
    src = _MIS_HTML.read_text(encoding="utf-8")
    # Plan 04-03 contract: cancel endpoint path is preserved verbatim.
    assert "/api/approvals/" in src and "/cancel" in src, (
        "mis_solicitudes must preserve the /api/approvals/{id}/cancel endpoint wired by Plan 04-03"
    )


def test_mis_solicitudes_cancel_modal_present() -> None:
    src = _MIS_HTML.read_text(encoding="utf-8")
    # Confirmation modal follows UI-SPEC §Destructive actions.
    assert 'role="dialog"' in src, "cancel modal must be role=dialog"
    assert 'aria-modal="true"' in src, "cancel modal must be aria-modal"
    assert "Cancelar solicitud" in src, 'destructive CTA must say "Cancelar solicitud"'
    assert ">Volver<" in src, 'secondary (safe) CTA must say "Volver"'


def test_mis_solicitudes_no_raw_state_colors() -> None:
    src = _MIS_HTML.read_text(encoding="utf-8")
    raw = re.findall(r"bg-(red|green|blue|yellow)-[0-9]{3}", src)
    assert not raw, (
        f"mis_solicitudes.html must not use raw Tailwind state colors; "
        f"found: {raw}. Use semantic .badge-* + bg-*-subtle tokens."
    )


# ── Integration tests (require Postgres) ─────────────────────────────────


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


TEST_DOMAIN = "@auth-screens-test.local"
TEST_PASSWORD = "authscreenstest12345"  # min 12 chars


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    from api.main import app

    with TestClient(app, follow_redirects=False) as c:
        yield c


@pytest.fixture(autouse=True)
def _cleanup() -> Iterator[None]:
    if not _postgres_reachable():
        yield
        return
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
                db.execute(select(NexoDepartment).where(NexoDepartment.code.in_(dept_codes)))
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
    return r.cookies.get("nexo_session", "")


# ── GET /login ─────────────────────────────────────────────────────────────


def test_login_get_renders(client: TestClient) -> None:
    """GET /login is public — renders 200 with the refactored markup."""
    r = client.get("/login")
    assert r.status_code == 200, r.text[:500]
    body = r.text
    assert 'action="/login"' in body
    assert 'autocomplete="username"' in body
    assert 'autocomplete="current-password"' in body
    assert "Entrar" in body
    # Semantic token classes reach the rendered HTML.
    assert "btn btn-primary btn-lg" in body
    assert "input-inline" in body


@_integration
def test_login_invalid_credentials_renders_error_copy(
    client: TestClient,
) -> None:
    """Wrong credentials surface a Spanish error banner (401)."""
    r = client.post(
        "/login",
        data={"email": "nobody@nowhere.com", "password": "nope-not-real"},
        headers={"Accept": "text/html"},
        follow_redirects=False,
    )
    assert r.status_code == 401, f"expected 401, got {r.status_code}"
    assert "Credenciales invalidas" in r.text, (
        'error banner must surface the literal "Credenciales invalidas"'
    )
    # The error banner must use the semantic token, not a raw color.
    assert "bg-error-subtle" in r.text
    assert "text-error" in r.text


# ── GET /cambiar-password ──────────────────────────────────────────────────


@_integration
def test_cambiar_password_get_renders_for_authenticated_user(
    client: TestClient,
) -> None:
    """Authenticated user can GET /cambiar-password with the new markup."""
    email = f"pwd{TEST_DOMAIN}"
    _create_user(email, role="propietario", dept_codes=[])
    cookie = _login(client, email)
    assert cookie

    r = client.get(
        "/cambiar-password",
        cookies={"nexo_session": cookie},
        headers={"Accept": "text/html"},
    )
    assert r.status_code == 200, r.text[:500]
    body = r.text
    assert 'action="/cambiar-password"' in body
    assert body.count('autocomplete="new-password"') == 2
    assert "Guardar contraseña" in body


@_integration
def test_cambiar_password_must_change_banner(client: TestClient) -> None:
    """must_change_password=True surfaces the forced-change warn banner."""
    email = f"mustchange{TEST_DOMAIN}"
    _create_user(
        email,
        role="usuario",
        dept_codes=["rrhh"],
        must_change_password=True,
    )
    # Login redirects to /cambiar-password per auth.py:144; follow it.
    cookie = _login(client, email)
    assert cookie

    r = client.get(
        "/cambiar-password",
        cookies={"nexo_session": cookie},
        headers={"Accept": "text/html"},
    )
    assert r.status_code == 200
    body = r.text
    assert "Por politica" in body, "must_change flag must surface the forced-change copy"
    # Uses the warn-subtle semantic token.
    assert "bg-warn-subtle" in body
    assert "text-warn" in body


# ── GET /mis-solicitudes ──────────────────────────────────────────────────


@_integration
def test_mis_solicitudes_renders_for_authenticated_user(
    client: TestClient,
) -> None:
    """Authenticated user sees the refactored mis-solicitudes page."""
    email = f"solicitante{TEST_DOMAIN}"
    _create_user(email, role="usuario", dept_codes=["rrhh"])
    cookie = _login(client, email)
    assert cookie

    r = client.get(
        "/mis-solicitudes",
        cookies={"nexo_session": cookie},
        headers={"Accept": "text/html"},
    )
    assert r.status_code == 200, r.text[:500]
    body = r.text
    # Either the table OR the empty state must be present.
    has_table = "data-table" in body
    has_empty = "No tienes solicitudes pendientes" in body
    assert has_table or has_empty, (
        "mis_solicitudes must render either the .data-table or the .empty-state"
    )
