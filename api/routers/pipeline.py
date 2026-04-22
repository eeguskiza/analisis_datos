"""Endpoint de ejecucion del pipeline: extraccion + generacion de informes.

Phase 4 (Plan 04-02):
- Añade ``POST /api/pipeline/preflight`` — devuelve ``Estimation`` (coste
  previo) sin ejecutar nada. La UI la usa para decidir modal amber/red.
- Convierte ``POST /api/pipeline/run`` en ``async def`` + gate logic por
  level (green ejecuta directo; amber requiere ``force=true``; red
  requiere ``force=true + approval_id`` válido por D-15).
- Envuelve la ejecución síncrona en ``asyncio.Semaphore(3)`` +
  ``asyncio.to_thread`` + ``asyncio.wait_for(timeout=900s)`` (D-18).
  Trade-off: el SSE en tiempo real se pierde durante la ejecución (Opción
  A del research); los mensajes se re-emiten todos al final. Aceptable en
  Mark-III — ver 04-RESEARCH §Opción A.

Compatibilidad con Plan 04-03 (approvals): ``consume_approval`` se
importa bajo un guard ``try/except ImportError`` porque 04-03 aterriza
en paralelo. Cuando el servicio está disponible, ``_APPROVALS_AVAILABLE``
es True y la lógica red completa funciona. Mientras tanto, un request
``level=red + force=true`` responde 503 explicando que el flujo de
aprobación aún no está disponible (vs silently-pass que sería un bypass
de seguridad).
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.deps import DbNexo
from api.models import PipelineRequest
from api.services.pipeline import run_pipeline
from nexo.data.dto.query import Estimation
from nexo.services.auth import require_permission
from nexo.services.pipeline_lock import PIPELINE_TIMEOUT_SEC, pipeline_semaphore
from nexo.services.preflight import estimate_cost


# ── Import-guard para Plan 04-03 (approvals) ─────────────────────────────
# Plan 04-03 entrega ``nexo/services/approvals.py`` con:
#   def consume_approval(db, *, approval_id, user_id, current_params) -> NexoQueryApproval
# que hace CAS single-use (D-15). Mientras no aterriza, el wrapper devuelve
# 503 en vez de pasar silenciosamente a ejecutar un ``red`` sin aprobación.
try:
    from nexo.services.approvals import consume_approval  # type: ignore[import-not-found]

    _APPROVALS_AVAILABLE = True
except ImportError:
    _APPROVALS_AVAILABLE = False


router = APIRouter(
    prefix="/pipeline",
    tags=["pipeline"],
    dependencies=[Depends(require_permission("pipeline:read"))],
)


def _build_pipeline_params(req: PipelineRequest) -> dict:
    """Construye el dict ``params`` que consume ``estimate_cost``.

    Normaliza el shape a algo que:
      - Es JSON-serializable (fechas a ISO strings).
      - Tiene ``n_recursos`` / ``n_dias`` precalculados (evita que
        ``_estimate_pipeline`` tenga que derivarlos).
      - Se puede guardar en ``nexo.query_log.params_json`` tal cual.
    """
    recursos = req.recursos or []
    return {
        "fecha_desde": req.fecha_inicio.isoformat(),
        "fecha_hasta": req.fecha_fin.isoformat(),
        "modulos": list(req.modulos),
        "source": req.source,
        "recursos": list(recursos),
        "n_recursos": len(recursos),
        "n_dias": (req.fecha_fin - req.fecha_inicio).days + 1,
    }


@router.post(
    "/preflight",
    response_model=Estimation,
    dependencies=[Depends(require_permission("pipeline:run"))],
)
def preflight(req: PipelineRequest) -> Estimation:
    """Estima coste del pipeline SIN ejecutar nada.

    La UI lo llama antes de ``/run`` para decidir si abrir modal amber/red.
    Requiere el mismo permiso que ``/run`` (``pipeline:run``): estimar es
    una acción autorizada y no queremos que usuarios sin permiso
    obtengan información sobre el coste de queries que no pueden lanzar.
    """
    params = _build_pipeline_params(req)
    return estimate_cost("pipeline/run", params)


@router.post(
    "/run",
    dependencies=[Depends(require_permission("pipeline:run"))],
)
async def run(
    req: PipelineRequest,
    request: Request,
    db: DbNexo,
    user=Depends(require_permission("pipeline:run")),
):
    """Ejecuta el pipeline bajo semáforo + timeout + gate preflight.

    Flujo:
      1. Estima coste (preflight).
      2. Guarda ``estimated_ms`` / ``params_json`` en ``request.state``
         para el middleware ``QueryTimingMiddleware``.
      3. Gate por level:
         - green: ejecuta directo.
         - amber sin ``force``: 428 (el modal amber UX — D-05).
         - amber con ``force``: ejecuta (usuario ya confirmó).
         - red sin ``force+approval_id``: 403 con payload estimation.
         - red con ``force+approval_id``: consume el approval
           (CAS single-use por 04-03) y ejecuta.
      4. Ejecuta ``run_pipeline`` síncrono en un thread bajo semáforo
         con timeout 900s. Timeout → HTTP 504 + ``status='timeout'``
         en ``query_log``.
      5. Devuelve SSE streaming con los mensajes (Opción A — research).

    Cada línea SSE tiene formato: ``data: <mensaje>\\n\\n``.
    La última línea contiene: ``data: DONE:<n_pdfs>:<pdf1|pdf2|...>\\n\\n``.
    """
    params = _build_pipeline_params(req)
    params_json_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
    est = estimate_cost("pipeline/run", params)

    # Poblar request.state para que QueryTimingMiddleware escriba la fila
    # en nexo.query_log con estimated_ms + params_json correctos.
    request.state.estimated_ms = est.estimated_ms
    request.state.params_json = params_json_str

    # ── Gate logic ────────────────────────────────────────────────────
    if est.level == "red":
        if not (req.force and req.approval_id is not None):
            raise HTTPException(
                status_code=403,
                detail={
                    "estimation": est.model_dump(),
                    "action": "request_approval",
                },
            )
        if not _APPROVALS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Flujo de aprobación no disponible aún — esperar Plan 04-03 "
                    "(nexo/services/approvals.py)"
                ),
            )
        approval = consume_approval(
            db,
            approval_id=req.approval_id,
            user_id=user.id,
            current_params=params,
        )
        request.state.approval_id = approval.id
    elif est.level == "amber":
        if not req.force:
            raise HTTPException(
                status_code=428,
                detail={
                    "estimation": est.model_dump(),
                    "action": "confirm_amber",
                },
            )
        # amber + force=true: usuario ya confirmó en el modal; ejecutamos
        # sin exigir approval_id (amber NO requiere aprobación — D-05).

    # CR-01 fix: marca "gate pasado" para el middleware — a partir de
    # aquí SÍ vamos a ejecutar la query (o al menos lo intentaremos).
    # El middleware usa este flag para decidir si persistir la fila.
    request.state.query_executed = True

    # ── Ejecución bajo semáforo + to_thread + timeout ─────────────────
    # Opción A (research §Opción A): colectamos todos los mensajes en el
    # thread y los re-emitimos al final via SSE. Pierde progreso en
    # tiempo real pero no congela el event loop ni requiere cola thread-
    # safe. Suficiente para Mark-III.

    def _worker() -> list[str]:
        return list(
            run_pipeline(
                fecha_inicio=req.fecha_inicio,
                fecha_fin=req.fecha_fin,
                modulos=req.modulos,
                source=req.source,
                recursos=req.recursos,
            )
        )

    async with pipeline_semaphore:
        try:
            messages = await asyncio.wait_for(
                asyncio.to_thread(_worker),
                timeout=PIPELINE_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Pipeline timeout ({PIPELINE_TIMEOUT_SEC}s)",
            )

    def event_stream():
        for msg in messages:
            yield f"data: {msg}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
