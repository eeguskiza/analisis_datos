"""Endpoints para listar, servir y borrar informes PDF generados."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.services import informes as informes_service

router = APIRouter(prefix="/informes", tags=["informes"])


@router.get("")
def listar():
    """Devuelve el arbol completo de informes."""
    return {"tree": informes_service.list_all(), "dates": informes_service.list_dates()}


@router.get("/pdf/{filepath:path}")
def servir_pdf(filepath: str):
    """Sirve un PDF concreto."""
    path = informes_service.get_pdf_path(filepath)
    if path is None:
        raise HTTPException(404, "PDF no encontrado")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@router.delete("/{date_str}")
def borrar_fecha(date_str: str):
    """Borra todos los informes de una fecha."""
    ok = informes_service.delete_date(date_str)
    if not ok:
        raise HTTPException(404, "Fecha no encontrada")
    return {"ok": True}
