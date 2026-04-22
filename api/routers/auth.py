"""Router de auth — /login, /logout, /cambiar-password.

Cierra el cableado iniciado en 02-01 (servicios en ``nexo.services.auth``):
valida credenciales, crea sesiones firmadas, aplica lockout progresivo y
fuerza cambio de password en primer login. Rate limit por IP via slowapi.

Todas las rutas publicas estan whitelisteadas en ``api.middleware.auth`` —
el middleware no bloquea la entrada.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy.orm import Session

from api.config import settings
from api.deps import render, templates
from api.rate_limit import limiter
from nexo.db.engine import SessionLocalNexo
from nexo.services.auth import (
    check_lockout,
    clear_attempts,
    create_session,
    get_user_by_email,
    hash_password,
    record_failed_attempt,
    revoke_all_sessions,
    revoke_session,
    unsign_session_token,
    verify_password,
)

logger = logging.getLogger("nexo.auth")

router = APIRouter(tags=["auth"])


# ── Dependencia de sesion de BD (scope por request) ──────────────────────────


def get_nexo_db():
    db = SessionLocalNexo()
    try:
        yield db
    finally:
        db.close()


MIN_PASSWORD_LEN = 12


def _client_ip(request: Request) -> str:
    """IP efectiva del cliente. Hoy lee ``request.client.host``; detras de
    Caddy hay que poblar ``X-Forwarded-For`` (pendiente Mark-IV). No bloquea
    Sprint 1."""
    return request.client.host if request.client else "unknown"


def _render_login(
    request: Request, *, error: Optional[str] = None, status_code: int = 200
) -> HTMLResponse:
    ctx = {"request": request, "error": error, "page": "login"}
    return templates.TemplateResponse(
        name="login.html", context=ctx, request=request, status_code=status_code
    )


def _render_cambiar_password(
    request: Request,
    *,
    error: Optional[str] = None,
    must_change: bool = False,
    status_code: int = 200,
) -> HTMLResponse:
    # Plan 08-06: cambiar_password.html ahora extiende base.html (chrome de
    # 08-02), que necesita `current_user` en contexto para las llamadas
    # `can(current_user, ...)` del drawer. Usamos la helper central
    # ``render()`` que inyecta current_user + flash_message desde
    # request.state (05-RESEARCH §Pitfall 3).
    return render(
        "cambiar_password.html",
        request,
        {
            "error": error,
            "page": "cambiar_password",
            "must_change": must_change,
        },
        status_code=status_code,
    )


# ── /login ────────────────────────────────────────────────────────────────────


@router.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return _render_login(request)


@router.post("/login")
@limiter.limit("20/minute")
async def login_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_nexo_db),
):
    ip = _client_ip(request)
    email_norm = email.strip().lower()

    # 1) Lockout
    if check_lockout(db, email_norm, ip):
        logger.warning("login bloqueado por lockout: %s desde %s", email_norm, ip)
        return _render_login(
            request,
            error="Cuenta temporalmente bloqueada. Reintenta en 15 minutos.",
            status_code=429,
        )

    # 2) Lookup + verify. Error generico para no filtrar si el usuario existe.
    user = get_user_by_email(db, email_norm)
    if user is None:
        record_failed_attempt(db, email_norm, ip)
        return _render_login(request, error="Credenciales invalidas", status_code=401)

    if not verify_password(user.password_hash, password):
        record_failed_attempt(db, email_norm, ip)
        return _render_login(request, error="Credenciales invalidas", status_code=401)

    # 3) Login valido
    clear_attempts(db, email_norm, ip)
    user.last_login = datetime.now(timezone.utc)
    db.commit()

    _, signed_cookie = create_session(db, user.id)

    # must_change_password → cortocircuita a /cambiar-password. El middleware
    # lo volveria a hacer en la proxima request, pero asi ahorramos un hop.
    # Plan 08-04: login OK redirige a /bienvenida (landing con saludo + reloj).
    # El must_change_password branch queda intacto — el cambio forzado sigue
    # siendo prioritario sobre la landing.
    target = "/cambiar-password" if user.must_change_password else "/bienvenida"
    response: Response = RedirectResponse(target, status_code=303)
    response.set_cookie(
        key=settings.session_cookie_name,
        value=signed_cookie,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=settings.session_ttl_hours * 3600,
        path="/",
    )
    logger.info("login OK: %s desde %s → %s", email_norm, ip, target)
    return response


# ── /logout ───────────────────────────────────────────────────────────────────


@router.post("/logout")
async def logout_post(request: Request, db: Session = Depends(get_nexo_db)):
    signed = request.cookies.get(settings.session_cookie_name)
    if signed:
        max_age = settings.session_ttl_hours * 3600
        raw = unsign_session_token(signed, max_age)
        if raw is not None:
            revoke_session(db, raw)

    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(key=settings.session_cookie_name, path="/")
    return response


# ── /cambiar-password ─────────────────────────────────────────────────────────


@router.get("/cambiar-password", response_class=HTMLResponse)
async def cambiar_password_get(request: Request):
    user = getattr(request.state, "user", None)
    must_change = bool(user and user.must_change_password)
    return _render_cambiar_password(request, must_change=must_change)


@router.post("/cambiar-password")
async def cambiar_password_post(
    request: Request,
    password_actual: str = Form(...),
    password_nuevo: str = Form(...),
    password_repetir: str = Form(...),
    db: Session = Depends(get_nexo_db),
):
    user = getattr(request.state, "user", None)
    if user is None:
        # Sanity: el middleware deberia haberlo bloqueado si no hay sesion.
        return RedirectResponse("/login", status_code=303)

    must_change = bool(user.must_change_password)

    # El user viene de una sesion distinta (la del middleware). Lo traemos
    # a esta sesion para poder commitear cambios.
    db_user = get_user_by_email(db, user.email)
    if db_user is None:
        return RedirectResponse("/login", status_code=303)

    if not verify_password(db_user.password_hash, password_actual):
        return _render_cambiar_password(
            request,
            error="La contrasena actual no es correcta.",
            must_change=must_change,
            status_code=400,
        )

    if password_nuevo != password_repetir:
        return _render_cambiar_password(
            request,
            error="Las contrasenas nuevas no coinciden.",
            must_change=must_change,
            status_code=400,
        )

    if len(password_nuevo) < MIN_PASSWORD_LEN:
        return _render_cambiar_password(
            request,
            error=f"La contrasena debe tener al menos {MIN_PASSWORD_LEN} caracteres.",
            must_change=must_change,
            status_code=400,
        )

    # OK: rehash + clear must_change + invalidar sesiones (forzar re-login)
    db_user.password_hash = hash_password(password_nuevo)
    db_user.must_change_password = False
    db.commit()
    revoke_all_sessions(db, db_user.id)

    response = RedirectResponse("/login?ok=password-cambiado", status_code=303)
    response.delete_cookie(key=settings.session_cookie_name, path="/")
    logger.info("password cambiado: %s", db_user.email)
    return response
