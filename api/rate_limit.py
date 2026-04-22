"""Rate limiter global basado en slowapi.

Modulo propio (en vez de vivir en ``api.main``) para que los routers
puedan decorar endpoints con ``@limiter.limit(...)`` sin crear imports
circulares con la factory de la app.

Research §Pattern 9: ``key_func=get_remote_address`` usa
``request.client.host`` como clave; detras de Caddy/Traefik hace falta
``X-Forwarded-For`` para evitar que todos los clientes compartan la IP
del proxy — no bloqueante para Sprint 1 (LAN interna), pendiente Mark-IV.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address


limiter = Limiter(key_func=get_remote_address)
