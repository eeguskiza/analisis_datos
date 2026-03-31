"""Rutas HTML — sirven las paginas Jinja2 de la interfaz."""
from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from api.config import settings
from api.deps import templates
from api.database import Recurso, get_db
from api.services import db as db_service
from api.services import informes as informes_service

router = APIRouter(tags=["pages"])


def _common_ctx(request: Request, page: str) -> dict:
    return {
        "request": request,
        "page": page,
        "today": date.today().isoformat(),
    }


@router.get("/")
def index(request: Request, db: Session = Depends(get_db)):
    ctx = _common_ctx(request, "dashboard")
    dates = informes_service.list_dates()
    ctx["dates"] = dates
    ctx["last_date"] = dates[0] if dates else None
    ctx["total_dates"] = len(dates)
    ctx["total_pdfs"] = informes_service.count_pdfs()
    ctx["n_recursos"] = db.query(Recurso).filter_by(activo=True).count()
    return templates.TemplateResponse("dashboard.html", ctx)


@router.get("/pipeline")
def pipeline_page(request: Request, db: Session = Depends(get_db)):
    ctx = _common_ctx(request, "pipeline")
    recursos = db.query(Recurso).filter_by(activo=True).order_by(Recurso.seccion, Recurso.nombre).all()
    ctx["recursos"] = [
        {"centro_trabajo": r.centro_trabajo, "nombre": r.nombre, "seccion": r.seccion}
        for r in recursos
    ]
    return templates.TemplateResponse("pipeline.html", ctx)


@router.get("/informes")
def informes_page(request: Request):
    ctx = _common_ctx(request, "informes")
    ctx["tree"] = informes_service.list_all()
    return templates.TemplateResponse("informes.html", ctx)


@router.get("/recursos")
def recursos_page(request: Request, db: Session = Depends(get_db)):
    ctx = _common_ctx(request, "recursos")
    rows = db.query(Recurso).order_by(Recurso.seccion, Recurso.nombre).all()
    ctx["recursos"] = [
        {"id": r.id, "centro_trabajo": r.centro_trabajo, "nombre": r.nombre,
         "seccion": r.seccion, "activo": r.activo}
        for r in rows
    ]
    return templates.TemplateResponse("recursos.html", ctx)


@router.get("/ciclos")
def ciclos_page(request: Request):
    ctx = _common_ctx(request, "ciclos")
    return templates.TemplateResponse("ciclos.html", ctx)


@router.get("/ajustes")
def ajustes_page(request: Request):
    ctx = _common_ctx(request, "ajustes")
    return templates.TemplateResponse("ajustes.html", ctx)
