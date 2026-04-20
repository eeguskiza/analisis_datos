"""FlashMiddleware — lee cookie ``nexo_flash``, la expone y la expira.

Fuente del mensaje: ``http_exception_handler_403`` (api/main.py) la
escribe tras 403 HTML. Posibles futuras fuentes: acciones con redirect
(p.ej. "Usuario creado OK") — ver 05-CONTEXT.md §Deferred ideas.

Cookie policy (D-08): HttpOnly=True, Secure=not settings.debug,
SameSite=Lax, max_age=60s. Read-and-clear: se lee en el request actual
y se borra en la response (Set-Cookie con Max-Age=0).

NO depende de sesión (anon requests también pueden cargar flash).
NO escribe en DB ni log — failure mode silencioso y seguro.
"""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


_FLASH_COOKIE = "nexo_flash"


class FlashMiddleware(BaseHTTPMiddleware):
    """Lee ``nexo_flash`` en el request y la borra en la response."""

    async def dispatch(self, request: Request, call_next):
        flash = request.cookies.get(_FLASH_COOKIE)
        request.state.flash = flash

        response = await call_next(request)

        if flash is not None:
            response.delete_cookie(_FLASH_COOKIE, path="/")
        return response
