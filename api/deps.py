"""Dependencias compartidas (templates, etc.) — evita imports circulares."""
from __future__ import annotations

from fastapi.templating import Jinja2Templates

from api.config import settings

templates = Jinja2Templates(directory=str(settings.project_root / "templates"))
