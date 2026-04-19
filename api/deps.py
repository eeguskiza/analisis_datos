"""Dependencias compartidas (templates, etc.) — evita imports circulares."""
from __future__ import annotations

from typing import Any, Mapping, Optional

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from api.config import settings

templates = Jinja2Templates(directory=str(settings.project_root / "templates"))

# Variables disponibles en todos los templates sin necesidad de pasarlas
# explicitamente por contexto en cada endpoint. El cableado (Sprint 0
# commit 7) sustituye a los strings hardcoded en base.html y otros.
templates.env.globals.update(
    app_name=settings.app_name,
    company_name=settings.company_name,
    logo_path=settings.nexo_logo_path,
    ecs_logo_path=settings.nexo_ecs_logo_path,
)


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
