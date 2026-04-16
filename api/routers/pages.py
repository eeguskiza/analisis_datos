"""Rutas HTML — sirven las paginas Jinja2 de la interfaz."""
from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from api.deps import templates
from api.database import Recurso, get_db

router = APIRouter(tags=["pages"])


def _common_ctx(request: Request, page: str) -> dict:
    return {
        "request": request,
        "page": page,
        "today": date.today().isoformat(),
    }


def _render(name: str, ctx: dict):
    """Render template compatible con Starlette viejo y nuevo."""
    return templates.TemplateResponse(name=name, context=ctx, request=ctx["request"])


@router.get("/")
def index(request: Request):
    ctx = _common_ctx(request, "dashboard")
    return _render("luk4.html", ctx)


@router.get("/pipeline")
def pipeline_page(request: Request, db: Session = Depends(get_db)):
    ctx = _common_ctx(request, "pipeline")
    recursos = db.query(Recurso).filter_by(activo=True).order_by(Recurso.seccion, Recurso.nombre).all()
    ctx["recursos"] = [
        {"centro_trabajo": r.centro_trabajo, "nombre": r.nombre, "seccion": r.seccion}
        for r in recursos
    ]
    return _render("pipeline.html", ctx)


@router.get("/informes")
def informes_page():
    """Redirige a historial (fusionado)."""
    return RedirectResponse("/historial", status_code=301)


@router.get("/recursos")
def recursos_page(request: Request, db: Session = Depends(get_db)):
    ctx = _common_ctx(request, "recursos")
    rows = db.query(Recurso).order_by(Recurso.seccion, Recurso.nombre).all()
    ctx["recursos"] = [
        {"id": r.id, "centro_trabajo": r.centro_trabajo, "nombre": r.nombre,
         "seccion": r.seccion, "activo": r.activo}
        for r in rows
    ]
    return _render("recursos.html", ctx)


@router.get("/historial")
def historial_page(request: Request, db: Session = Depends(get_db)):
    ctx = _common_ctx(request, "historial")
    recursos = db.query(Recurso).filter_by(activo=True).order_by(Recurso.seccion, Recurso.nombre).all()
    ctx["recursos"] = [
        {"centro_trabajo": r.centro_trabajo, "nombre": r.nombre, "seccion": r.seccion}
        for r in recursos
    ]
    return _render("historial.html", ctx)


@router.get("/ciclos-calc")
def ciclos_calc_page(request: Request, db: Session = Depends(get_db)):
    ctx = _common_ctx(request, "recursos")
    recursos = db.query(Recurso).filter_by(activo=True).order_by(Recurso.seccion, Recurso.nombre).all()
    ctx["recursos"] = [
        {"centro_trabajo": r.centro_trabajo, "nombre": r.nombre, "seccion": r.seccion}
        for r in recursos
    ]
    return _render("ciclos_calc.html", ctx)


@router.get("/operarios")
def operarios_page(request: Request):
    ctx = _common_ctx(request, "operarios")
    return _render("operarios.html", ctx)


@router.get("/datos")
def datos_page(request: Request, db: Session = Depends(get_db)):
    ctx = _common_ctx(request, "datos")
    recursos = db.query(Recurso).filter_by(activo=True).order_by(Recurso.seccion, Recurso.nombre).all()
    ctx["recursos"] = [
        {"centro_trabajo": r.centro_trabajo, "nombre": r.nombre, "seccion": r.seccion}
        for r in recursos
    ]
    return _render("datos.html", ctx)


@router.get("/bbdd")
def bbdd_page(request: Request):
    ctx = _common_ctx(request, "bbdd")
    return _render("bbdd.html", ctx)


@router.get("/capacidad")
def capacidad_page(request: Request):
    ctx = _common_ctx(request, "capacidad")
    return _render("capacidad.html", ctx)


@router.get("/ajustes")
def ajustes_page(request: Request):
    ctx = _common_ctx(request, "ajustes")
    return _render("ajustes.html", ctx)
