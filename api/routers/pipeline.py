"""Endpoint de ejecucion del pipeline: extraccion + generacion de informes."""
from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.models import PipelineRequest
from api.services.pipeline import run_pipeline

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.post("/run")
def run(req: PipelineRequest):
    """
    Ejecuta el pipeline y devuelve un stream SSE con el progreso.

    Cada linea SSE tiene formato:  data: <mensaje>\n\n
    La ultima linea contiene:      data: DONE:<n_pdfs>:<pdf1|pdf2|...>\n\n
    """

    def event_stream():
        for msg in run_pipeline(
            fecha_inicio=req.fecha_inicio,
            fecha_fin=req.fecha_fin,
            modulos=req.modulos,
            source=req.source,
            recursos=req.recursos,
        ):
            yield f"data: {msg}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
