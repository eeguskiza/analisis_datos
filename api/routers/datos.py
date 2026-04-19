"""Extraccion de datos de IZARO → ecs_mobility (para Power BI)."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import date
from typing import Optional

from nexo.services.auth import require_permission

router = APIRouter(
    prefix="/datos",
    tags=["datos"],
    dependencies=[Depends(require_permission("datos:read"))],
)


class ExtractRequest(BaseModel):
    fecha_inicio: date
    fecha_fin: date
    recursos: Optional[list[str]] = None


@router.post("/extraer")
def extraer(req: ExtractRequest):
    """
    Extrae datos de IZARO y calcula metricas OEE.
    No genera PDFs — solo llena las tablas para Power BI.
    Stream SSE con progreso.
    """
    from api.services.pipeline import run_pipeline

    def event_stream():
        for msg in run_pipeline(
            fecha_inicio=req.fecha_inicio,
            fecha_fin=req.fecha_fin,
            modulos=[],       # sin modulos OEE → no genera PDFs
            source="db",
            recursos=req.recursos,
        ):
            yield f"data: {msg}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
