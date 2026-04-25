"""Dependencias compartidas (templates, etc.) — evita imports circulares."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Mapping, Optional
from zoneinfo import ZoneInfo

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from api.config import settings
from nexo.services.auth import can as _can

templates = Jinja2Templates(directory=str(settings.project_root / "templates"))

# Variables disponibles en todos los templates sin necesidad de pasarlas
# explicitamente por contexto en cada endpoint. El cableado (Sprint 0
# commit 7) sustituye a los strings hardcoded en base.html y otros.
#
# ``can`` (Plan 05-01, D-03 / D-09): registrado a import-time para que
# cualquier template pueda hacer ``{% if can(current_user, "perm") %}``
# sin depender de que el endpoint pase la función por contexto. El
# registro aquí (no en lifespan ni en factory) evita que los primeros
# renders encuentren ``can`` undefined (05-RESEARCH §Pitfall 5).
# ``current_user`` NO se registra como global — es per-request y se
# inyecta via ``render()`` (05-RESEARCH §Pitfall 3).
templates.env.globals.update(
    app_name=settings.app_name,
    company_name=settings.company_name,
    app_version="v1.0.0",
    logo_path=settings.nexo_logo_path,
    ecs_logo_path=settings.nexo_ecs_logo_path,
    can=_can,
    # Plan 08-02: `getattr` expuesto como global para soportar el patrón
    # defensivo `getattr(current_user, 'nombre', None)` en base.html
    # ANTES de que Plan 08-03 añada la columna `nombre` al modelo ORM.
    # Entre Wave 2 (esta) y Wave 3 (08-03) `current_user.nombre` no existe;
    # `getattr` devuelve None y el fallback (email local-part) se renderiza.
    # Tras 08-03 el happy path (user.nombre) funciona automáticamente.
    getattr=getattr,
)


def user_display_name(user: Any | None, *, first_name_only: bool = False) -> str:
    """Devuelve un nombre legible para UI a partir del usuario autenticado.

    Prioridad:
    1. ``user.name`` + ``user.surname`` si existen.
    2. ``user.nombre`` legacy si existe y no está vacío.
    3. ``user.username`` si existe.
    4. Fallback al local-part del email, limpiando separadores comunes
       (``.``, ``_``, ``-``) para que ``e.eguskiza@...`` no se vea literal.

    ``first_name_only=True`` devuelve el primer token visible. Se usa en la
    landing para un saludo más natural ("Buenas tardes, Erik").
    """
    if user is None:
        return ""

    name = (getattr(user, "name", None) or "").strip()
    surname = (getattr(user, "surname", None) or "").strip()
    full_name = " ".join(p for p in [name, surname] if p).strip()
    if full_name:
        return full_name.split()[0] if first_name_only else full_name

    nombre = (getattr(user, "nombre", None) or "").strip()
    if nombre:
        return nombre.split()[0] if first_name_only else nombre

    username = (getattr(user, "username", None) or "").strip()
    if username:
        return username.split()[0] if first_name_only else username

    email = (getattr(user, "email", None) or "").strip()
    if not email:
        return ""

    local_part = email.split("@", 1)[0]
    parts = [p for p in re.split(r"[._-]+", local_part) if p]
    if not parts:
        fallback = local_part.strip()
        if not fallback:
            return ""
        return fallback[:1].upper() + fallback[1:]

    words = [p[:1].upper() + p[1:] for p in parts]
    return words[0] if first_name_only else " ".join(words)


templates.env.globals.update(user_display_name=user_display_name)


# ── Plan 08-04: hora_saludo filter (D-23 — saludo por franja) ───────────────
# Franjas horarias (Europe/Madrid, servidor es autoritativo):
#   06:00..11:59 → "Buenos días"
#   12:00..20:59 → "Buenas tardes"
#   21:00..05:59 → "Buenas noches"
#
# Se registra como filtro Jinja (`{{ None|hora_saludo }}`) para que la landing
# renderice el saludo server-side sin depender del reloj cliente.

_MADRID = ZoneInfo("Europe/Madrid")


def hora_saludo(now: datetime | None = None) -> str:
    """Return the time-banded Spanish greeting for the given instant.

    Bands (D-23):
      06:00..11:59 → 'Buenos días'
      12:00..20:59 → 'Buenas tardes'
      21:00..05:59 → 'Buenas noches'

    Uses Europe/Madrid (server tz is authoritative — UI-SPEC §Landing).
    Naive datetimes are assumed to already be in Madrid local time.
    """
    if now is None:
        now = datetime.now(_MADRID)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=_MADRID)
    else:
        now = now.astimezone(_MADRID)
    h = now.hour
    if 6 <= h < 12:
        return "Buenos días"
    if 12 <= h < 21:
        return "Buenas tardes"
    return "Buenas noches"


templates.env.filters["hora_saludo"] = hora_saludo


def render(
    template_name: str,
    request: Request,
    extra: Optional[Mapping[str, Any]] = None,
    *,
    status_code: int = 200,
) -> HTMLResponse:
    """Render Jinja2 inyectando ``current_user`` desde ``request.state``.

    Uso preferente en endpoints HTML: centraliza el paso de ``current_user``
    para que cada template pueda pintar el topbar con el email del usuario
    autenticado sin tener que replicarlo en cada handler.

    ``current_user`` sera ``None`` en rutas publicas (p.ej. /login) porque
    el AuthMiddleware no las toca — los templates deben chequear la
    existencia antes de usarlo.
    """
    ctx: dict[str, Any] = {
        "request": request,
        "current_user": getattr(request.state, "user", None),
        # Plan 05-03: flash_message lo puebla ``FlashMiddleware`` leyendo
        # la cookie ``nexo_flash`` (D-07/D-08). Puede ser ``None`` en
        # request limpios; ``base.html`` hace ``{% if flash_message %}``.
        "flash_message": getattr(request.state, "flash", None),
    }
    if extra:
        ctx.update(extra)
    return templates.TemplateResponse(
        name=template_name, context=ctx, request=request, status_code=status_code
    )


# ── Dependencias de base de datos (Plan 03-01) ───────────────────────────
# Tres engines distintos: APP (ecs_mobility), NEXO (Postgres), MES
# (dbizaro read-only). Cada uno con su propio generator yield-pattern para
# que FastAPI cierre la sesion al terminar la request. MES es read-only
# por convención → entregamos el Engine directamente, no una Session
# (no hay transacciones que gestionar).

from typing import Annotated, Iterator  # noqa: E402

from fastapi import Depends  # noqa: E402
from sqlalchemy.engine import Engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from nexo.data.engines import (  # noqa: E402
    SessionLocalApp,
    SessionLocalNexo,
    engine_mes as _engine_mes,
)


def get_db_app() -> Iterator[Session]:
    """Session SQL Server ``ecs_mobility``. El caller comitea."""
    db = SessionLocalApp()
    try:
        yield db
    finally:
        db.close()


def get_db_nexo() -> Iterator[Session]:
    """Session Postgres ``nexo.*``. El caller comitea."""
    db = SessionLocalNexo()
    try:
        yield db
    finally:
        db.close()


def get_engine_mes() -> Engine:
    """MES es read-only; entregamos el Engine directamente."""
    return _engine_mes


# Aliases ``Annotated`` (PEP 593) — firmas de router más limpias:
#   def endpoint(db: DbApp, engine_mes: EngineMes): ...
DbApp = Annotated[Session, Depends(get_db_app)]
DbNexo = Annotated[Session, Depends(get_db_nexo)]
EngineMes = Annotated[Engine, Depends(get_engine_mes)]
