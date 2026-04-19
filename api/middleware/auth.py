"""AuthMiddleware — puerta de entrada de sesion.

Corre antes de cualquier handler: si no hay sesion valida redirige a
``/login`` (HTML) o devuelve ``401`` JSON (API). Si hay sesion, puebla
``request.state.user`` y fuerza ``/cambiar-password`` mientras el
usuario tenga ``must_change_password=True``.

Implementacion segun research §Pattern 3 (AuthMiddleware + Pitfall 1 sobre
orden LIFO de Starlette) y §Pattern 2 (cookie firmada con itsdangerous,
BD como fuente de verdad del mapeo token → user).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from api.config import settings
from nexo.db.engine import SessionLocalNexo
from nexo.services.auth import (
    extend_session,
    get_session,
    unsign_session_token,
)

logger = logging.getLogger("nexo.auth")


# ── Whitelists ─────────────────────────────────────────────────────────────
# Exacta: rutas que no exigen sesion. ``/api/health`` queda publico para que
# los healthchecks de Caddy/compose sigan funcionando (research §Open Q1).
_PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/login",
        "/logout",
        "/api/health",
        "/favicon.ico",
    }
)

# Prefijos: todo lo que empiece por uno de estos queda publico.
_PUBLIC_PREFIXES: tuple[str, ...] = (
    "/static/",
    "/api/docs",
    "/openapi.json",
)

# Paths donde la redireccion obligatoria a ``/cambiar-password`` NO aplica
# (porque son ellas mismas las que dejan cambiar el password o son rutas
# auxiliares necesarias para el flujo). Sin esto entras en bucle redirect.
_CHANGE_PASSWORD_ALLOWED: frozenset[str] = frozenset(
    {
        "/cambiar-password",
        "/logout",
    }
)


def _is_public(path: str) -> bool:
    if path in _PUBLIC_PATHS:
        return True
    return any(path.startswith(p) for p in _PUBLIC_PREFIXES)


def _wants_json(request: Request) -> bool:
    """Replica la heuristica del global_exception_handler: rutas ``/api/*``
    reciben JSON; el resto HTML (redirect)."""
    if request.url.path.startswith("/api/"):
        return True
    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return True
    if request.headers.get("hx-request") == "true":
        return True
    return False


def _unauthenticated_response(request: Request) -> Response:
    if _wants_json(request):
        return JSONResponse(
            status_code=401,
            content={"detail": "Not authenticated"},
        )
    # GET cualquier cosa → redirect a /login. POST/PUT sin sesion tambien
    # caen aqui; el navegador los convertira en GET /login (303 seria mas
    # correcto pero Starlette usa 302 por defecto y no cambia method).
    return RedirectResponse("/login", status_code=302)


class AuthMiddleware(BaseHTTPMiddleware):
    """Valida la cookie de sesion y adjunta ``request.state.user``.

    Flujo:
    1. Whitelist: rutas publicas pasan directas sin tocar BD.
    2. Sin cookie → 401 JSON (API) o redirect a /login (HTML).
    3. Con cookie firmada: unsign → lookup en BD → validar usuario activo.
    4. ``must_change_password`` → redirect a /cambiar-password (excepto en
       las rutas del flujo de cambio).
    5. OK → ``request.state.user = user`` + sliding expiration.
    """

    def __init__(self, app: ASGIApp, *, public_paths: Iterable[str] | None = None) -> None:
        super().__init__(app)
        self._extra_public = frozenset(public_paths or ())

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # (1) rutas publicas
        if _is_public(path) or path in self._extra_public:
            return await call_next(request)

        # (2) cookie presente?
        signed = request.cookies.get(settings.session_cookie_name)
        if not signed:
            return _unauthenticated_response(request)

        # (3) verificar firma + vigencia. Max age = TTL de sesion en segundos.
        max_age = settings.session_ttl_hours * 3600
        raw_token = unsign_session_token(signed, max_age)
        if raw_token is None:
            logger.info("AuthMiddleware: cookie con firma invalida o expirada en %s", path)
            return _unauthenticated_response(request)

        # (4) lookup en BD. Sesion propia: abrimos/cerramos sesion del pool
        # explicitamente porque el middleware corre fuera del ciclo Depends().
        db = SessionLocalNexo()
        try:
            session = get_session(db, raw_token)
            if session is None:
                logger.info("AuthMiddleware: token sin sesion activa en %s", path)
                return _unauthenticated_response(request)

            user = session.user
            if user is None or not user.active:
                logger.info(
                    "AuthMiddleware: sesion %s apunta a usuario inactivo o borrado", session.id
                )
                return _unauthenticated_response(request)

            # (5) must_change_password → cortocircuito
            if user.must_change_password and path not in _CHANGE_PASSWORD_ALLOWED:
                return RedirectResponse("/cambiar-password", status_code=302)

            # Sliding expiration: renovamos expires_at solo si ya estamos en
            # la segunda mitad del TTL. Asi evitamos un UPDATE por request.
            now = datetime.now(timezone.utc)
            halfway = session.expires_at - ((session.expires_at - session.created_at) / 2)
            if now > halfway:
                extend_session(db, session)

            # Eager-load user.departments antes de cerrar la sesion. Sin esto,
            # el acceso lazy a user.departments desde require_permission()
            # (Plan 02-03) levantaria DetachedInstanceError porque la sesion
            # ORM que lo cargo ya esta cerrada. La evaluacion de la list
            # comprehension fuerza el SELECT y cachea el resultado en
            # user.__dict__['departments'], accesible tras db.close().
            _ = list(user.departments)

            request.state.user = user
            request.state.session = session
        finally:
            db.close()

        return await call_next(request)
