"""AuditMiddleware — append-only log de cada request autenticada.

Corre DESPUES del AuthMiddleware (orden LIFO: Auth outer, Audit inner):
cuando Audit entra, ``request.state.user`` ya esta poblado. Si no hay
usuario en state (request publica /login, /api/health, /static/*), el
middleware no escribe nada — solo requests autenticadas generan fila.

Diseno segun research §Pattern 5 + §Pitfall 2:

- Body lectura: ``await request.body()`` en FastAPI 0.135.3+ es cacheado,
  seguro hacerlo sin consumir el stream del handler.
- Sanitizacion: whitelist de ``_REDACTED_ENDPOINTS`` descarta body entero;
  blacklist de ``_SENSITIVE_FIELDS`` reemplaza valor por ``[REDACTED]``
  cuando aparece un campo sensible dentro de un payload no-redacted.
- Truncado: ``_MAX_DETAILS_CHARS = 4096`` evita filas gigantes si algun
  endpoint recibe payload inesperado.
- Errores de escritura a BD no bloquean la request — el log fallido queda
  en ``logger.exception`` para investigacion offline.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from nexo.db.engine import SessionLocalNexo
from nexo.db.models import NexoAuditLog

logger = logging.getLogger("nexo.audit")


# Endpoints donde el body completo se descarta (credenciales o secretos
# estan dispersos en el payload y no queremos guardar nada aunque este
# 'sanitizado'). Los campos individuales quedan siempre fuera del log.
_REDACTED_ENDPOINTS: frozenset[str] = frozenset(
    {
        "/api/conexion/config",
        "/api/conexion/test",
        "/login",
        "/cambiar-password",
    }
)

# Claves que, si aparecen en un body no-redacted, reciben ``[REDACTED]``
# como valor antes de serializar. Case-insensitive.
_SENSITIVE_FIELDS: frozenset[str] = frozenset(
    {
        "password",
        "pwd",
        "secret",
        "token",
        "clave",
        "contrasena",
        "contrasenia",
        "password_actual",
        "password_nuevo",
        "password_repetir",
    }
)

_MAX_DETAILS_CHARS = 4096


def _sanitize_body(body_bytes: bytes) -> str | None:
    """Parsea el body como JSON y redacta campos sensibles. Devuelve
    ``None`` si el body no es JSON parseable (no bloquea la request)."""
    if not body_bytes:
        return None
    try:
        payload = json.loads(body_bytes)
    except (json.JSONDecodeError, ValueError):
        return None

    if isinstance(payload, dict):
        sanitized = {
            k: ("[REDACTED]" if k.lower() in _SENSITIVE_FIELDS else v)
            for k, v in payload.items()
        }
        return json.dumps(sanitized, ensure_ascii=False)[:_MAX_DETAILS_CHARS]

    # Lista o escalar: lo guardamos tal cual, truncado.
    return json.dumps(payload, ensure_ascii=False)[:_MAX_DETAILS_CHARS]


class AuditMiddleware(BaseHTTPMiddleware):
    """Registra cada request autenticada en ``nexo.audit_log``.

    Campos capturados:
    - ``ts``: ``datetime.now(timezone.utc)``
    - ``user_id``: ``request.state.user.id``
    - ``ip``: ``request.client.host``
    - ``method``: GET/POST/PUT/PATCH/DELETE
    - ``path``: URL path (query params no)
    - ``status``: codigo HTTP de la respuesta (incluye 403 si
      ``require_permission`` rechazo al usuario)
    - ``details_json``: body sanitizado para POST/PUT/PATCH; None para GET,
      DELETE, HEAD y endpoints de ``_REDACTED_ENDPOINTS``
    """

    async def dispatch(self, request: Request, call_next):
        user = getattr(request.state, "user", None)

        # Request publica (sin sesion): no auditamos.
        if user is None:
            return await call_next(request)

        # Capturar body ANTES de llamar al handler; FastAPI 0.135+ lo cachea.
        details: str | None = None
        if request.method in ("POST", "PUT", "PATCH"):
            path = request.url.path
            if path in _REDACTED_ENDPOINTS:
                details = None
            else:
                try:
                    body = await request.body()
                    details = _sanitize_body(body)
                except Exception:
                    # Leer el body puede fallar con streams raros; no bloquea.
                    details = None

        response = await call_next(request)

        try:
            db = SessionLocalNexo()
            try:
                db.add(
                    NexoAuditLog(
                        ts=datetime.now(timezone.utc),
                        user_id=user.id,
                        ip=request.client.host if request.client else "unknown",
                        method=request.method,
                        path=request.url.path,
                        status=response.status_code,
                        details_json=details,
                    )
                )
                db.commit()
            finally:
                db.close()
        except Exception:
            # Un fallo del log no debe tumbar la response.
            logger.exception(
                "Error escribiendo nexo.audit_log (user_id=%s, path=%s)",
                user.id,
                request.url.path,
            )

        return response
