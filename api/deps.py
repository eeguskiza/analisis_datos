"""Dependencias compartidas (templates, etc.) — evita imports circulares."""
from __future__ import annotations

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
