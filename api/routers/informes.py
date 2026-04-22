"""Endpoints para listar, servir y borrar informes PDF generados."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from api.services import informes as informes_service
from nexo.services.auth import require_permission

router = APIRouter(
    prefix="/informes",
    tags=["informes"],
    dependencies=[Depends(require_permission("informes:read"))],
)


@router.get("")
def listar():
    """Devuelve el arbol completo de informes."""
    return {"tree": informes_service.list_all(), "dates": informes_service.list_dates()}


@router.get("/pdf/{filepath:path}")
def servir_pdf(filepath: str):
    """Sirve un PDF para visualizar inline — nunca fuerza descarga."""
    path = informes_service.get_pdf_path(filepath)
    if path is None:
        raise HTTPException(404, "PDF no encontrado")
    data = path.read_bytes()
    return Response(
        content=data,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "inline",
            "Content-Length": str(len(data)),
        },
    )


@router.get("/download/{filepath:path}")
def descargar_pdf(filepath: str):
    """Fuerza descarga de un PDF."""
    path = informes_service.get_pdf_path(filepath)
    if path is None:
        raise HTTPException(404, "PDF no encontrado")
    data = path.read_bytes()
    return Response(
        content=data,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{path.name}"',
            "Content-Length": str(len(data)),
        },
    )


@router.delete(
    "/{date_str}",
    dependencies=[Depends(require_permission("informes:delete"))],
)
def borrar_fecha(date_str: str):
    """Borra todos los informes de una fecha."""
    ok = informes_service.delete_date(date_str)
    if not ok:
        raise HTTPException(404, "Fecha no encontrada")
    return {"ok": True}
