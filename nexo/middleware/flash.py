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
            # HI-01: no pisar un Set-Cookie nexo_flash emitido por el handler
            # (p.ej. ``http_exception_handler_403`` en 403 encadenado). Sin
            # este guard, ``delete_cookie`` añade un segundo Set-Cookie con
            # ``Max-Age=0`` y el UA (RFC 6265 §4.1.2: "last one wins")
            # borra el mensaje recién puesto. Comparación case-insensitive
            # porque HTTP header names no distinguen mayúsculas, pero el
            # cookie-name sí — ``_FLASH_COOKIE`` es literal, así que el
            # prefix match es exacto.
            prefix = f"{_FLASH_COOKIE}="
            already_set = any(
                header.lstrip().startswith(prefix)
                for header in response.headers.getlist("set-cookie")
            )
            if not already_set:
                response.delete_cookie(_FLASH_COOKIE, path="/")
        return response
