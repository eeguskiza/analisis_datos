"""Rutas HTML — sirven las paginas Jinja2 de la interfaz."""
from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from api.deps import render
from api.database import Recurso, get_db
from nexo.services.auth import require_permission

router = APIRouter(tags=["pages"])


def _common_extra(page: str) -> dict:
    return {
        "page": page,
        "today": date.today().isoformat(),
    }


@router.get("/")
def index(request: Request):
    return render("luk4.html", request, _common_extra("dashboard"))


@router.get("/pipeline")
def pipeline_page(request: Request, db: Session = Depends(get_db)):
    extra = _common_extra("pipeline")
    recursos = db.query(Recurso).filter_by(activo=True).order_by(Recurso.seccion, Recurso.nombre).all()
    extra["recursos"] = [
        {"centro_trabajo": r.centro_trabajo, "nombre": r.nombre, "seccion": r.seccion}
        for r in recursos
    ]
    return render("pipeline.html", request, extra)


@router.get("/informes")
def informes_page():
    """Redirige a historial (fusionado)."""
    return RedirectResponse("/historial", status_code=301)


@router.get("/recursos")
def recursos_page(request: Request, db: Session = Depends(get_db)):
    extra = _common_extra("recursos")
    rows = db.query(Recurso).order_by(Recurso.seccion, Recurso.nombre).all()
    extra["recursos"] = [
        {"id": r.id, "centro_trabajo": r.centro_trabajo, "nombre": r.nombre,
         "seccion": r.seccion, "activo": r.activo}
        for r in rows
    ]
    return render("recursos.html", request, extra)


@router.get("/historial")
def historial_page(request: Request, db: Session = Depends(get_db)):
    extra = _common_extra("historial")
    recursos = db.query(Recurso).filter_by(activo=True).order_by(Recurso.seccion, Recurso.nombre).all()
    extra["recursos"] = [
        {"centro_trabajo": r.centro_trabajo, "nombre": r.nombre, "seccion": r.seccion}
        for r in recursos
    ]
    return render("historial.html", request, extra)


@router.get("/ciclos-calc")
def ciclos_calc_page(request: Request, db: Session = Depends(get_db)):
    extra = _common_extra("ciclos_calc")
    recursos = db.query(Recurso).filter_by(activo=True).order_by(Recurso.seccion, Recurso.nombre).all()
    extra["recursos"] = [
        {"centro_trabajo": r.centro_trabajo, "nombre": r.nombre, "seccion": r.seccion}
        for r in recursos
    ]
    return render("ciclos_calc.html", request, extra)


@router.get("/operarios")
def operarios_page(request: Request):
    return render("operarios.html", request, _common_extra("operarios"))


@router.get("/datos")
def datos_page(request: Request, db: Session = Depends(get_db)):
    extra = _common_extra("datos")
    recursos = db.query(Recurso).filter_by(activo=True).order_by(Recurso.seccion, Recurso.nombre).all()
    extra["recursos"] = [
        {"centro_trabajo": r.centro_trabajo, "nombre": r.nombre, "seccion": r.seccion}
        for r in recursos
    ]
    return render("datos.html", request, extra)


@router.get("/bbdd")
def bbdd_page(request: Request):
    return render("bbdd.html", request, _common_extra("bbdd"))


@router.get("/capacidad")
def capacidad_page(request: Request):
    return render("capacidad.html", request, _common_extra("capacidad"))


@router.get(
    "/ajustes",
    dependencies=[Depends(require_permission("ajustes:manage"))],
)
def ajustes_page(request: Request):
    return render("ajustes.html", request, _common_extra("ajustes"))


@router.get(
    "/ajustes/conexion",
    dependencies=[Depends(require_permission("conexion:config"))],
)
def ajustes_conexion_page(request: Request):
    # Plan 05-04 / D-06: sub-pagina dedicada para Conexion SQL Server.
    # Gateada por conexion:config ([] en PERMISSION_MAP = propietario-only,
    # Pitfall 6). Los endpoints backend (/api/conexion/*) NO se modifican.
    return render(
        "ajustes_conexion.html", request, _common_extra("ajustes_conexion")
    )
