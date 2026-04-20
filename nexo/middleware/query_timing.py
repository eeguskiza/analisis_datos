"""QueryTimingMiddleware — postflight log de los 4 endpoints caros.

Mide ``actual_ms`` alrededor del handler y escribe una fila en
``nexo.query_log`` con los campos que el router dejó previamente en
``request.state`` (``estimated_ms``, ``approval_id``, ``params_json``).

Orden en la cadena de middlewares (Starlette LIFO):

    Request → AuthMiddleware → AuditMiddleware → QueryTimingMiddleware → handler
                                                                      ↓
    Response ← Auth ← Audit ← QueryTiming ← (handler)

``QueryTimingMiddleware`` va registrado PRIMERO en ``api/main.py`` para
quedar innermost (más cercano al handler). Así ``request.state.user`` ya
está poblado por Auth cuando Timing entra, y ``request.state.estimated_ms``
(poblado por el router antes de ejecutar la query) es visible al leer
``request.state`` tras ``call_next``.

Clasificación del ``status`` en ``query_log``:

  - HTTP 504                       → ``status='timeout'``
  - HTTP >= 400                    → ``status='error'``
  - ``actual_ms > warn_ms × 1.5``  → ``status='slow'`` + ``log.warning``
  - otherwise                       → ``status='ok'``

Short-circuits (no escribir fila):

  - Path no está en ``_TIMED_PATHS`` o sí está en ``_EXCLUDED``.
  - ``request.state.user`` es ``None`` (request sin autenticar — la fila
    no correlacionaría con nadie y el endpoint de todas formas devolverá
    401/redirect antes de llegar aquí).
  - Endpoints de rango (``capacidad``/``operarios``) donde el router
    no populó ``request.state.estimated_ms`` → rango ≤ 90d per D-03.

Disciplina de errores (copiada de ``api.middleware.audit``):
  - Un fallo al escribir en ``nexo.query_log`` NUNCA tumba la response.
  - El fallo se loguea via ``logger.exception`` y la request sigue.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from nexo.data.engines import SessionLocalNexo
from nexo.data.models_nexo import NexoQueryLog
from nexo.services import thresholds_cache


logger = logging.getLogger("nexo.query_timing")


# Mapa path → endpoint_key (coincide con ``nexo.query_thresholds.endpoint``
# — los valores son los mismos strings que se usan en el seed D-01..D-03).
# Paths fuera de este dict atraviesan el middleware sin side effects.
_TIMED_PATHS: dict[str, str] = {
    "/api/pipeline/run": "pipeline/run",
    "/api/bbdd/query": "bbdd/query",
    "/api/capacidad": "capacidad",
    "/api/operarios": "operarios",
}

# Paths excluidos explícitamente para evitar ruido en ``query_log`` si
# algún día apareciesen en ``_TIMED_PATHS`` por error (p.ej. refactor
# accidental del router de health).
_EXCLUDED: frozenset[str] = frozenset({
    "/api/health",
    "/api/approvals/count",
})


class QueryTimingMiddleware(BaseHTTPMiddleware):
    """Mide tiempo y graba fila en ``nexo.query_log`` para 4 endpoints."""

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        endpoint_key = _TIMED_PATHS.get(path)

        # Short-circuit: paths no auditados pasan sin overhead.
        if endpoint_key is None or path in _EXCLUDED:
            return await call_next(request)

        user = getattr(request.state, "user", None)
        # Request sin auth (redirect/401 más arriba) — no correlacionamos.
        if user is None:
            return await call_next(request)

        t0 = time.monotonic()
        try:
            response = await call_next(request)
            actual_ms = int((time.monotonic() - t0) * 1000)

            # IMPORTANTE: leemos request.state DESPUÉS de call_next.
            # El router pobla estimated_ms/approval_id/params_json durante
            # la ejecución del handler (dentro de call_next). Leer antes
            # siempre da None para esos campos.
            estimated_ms: Optional[int] = getattr(
                request.state, "estimated_ms", None
            )
            approval_id: Optional[int] = getattr(
                request.state, "approval_id", None
            )
            params_snapshot: Optional[str] = getattr(
                request.state, "params_json", None
            )

            # Rango <=90d en capacidad/operarios: router NO pobló
            # estimated_ms (short-circuit per D-03). Saltamos el log:
            # ese tráfico es barato y no queremos inflar ``query_log``
            # con filas triviales.
            if estimated_ms is None and endpoint_key in ("capacidad", "operarios"):
                return response

            query_status = _classify_status(
                actual_ms=actual_ms,
                endpoint_key=endpoint_key,
                http_status=response.status_code,
            )
            _persist(
                user_id=user.id,
                endpoint=endpoint_key,
                params_json=params_snapshot,
                estimated_ms=estimated_ms,
                actual_ms=actual_ms,
                rows=None,
                status=query_status,
                approval_id=approval_id,
                ip=request.client.host if request.client else "unknown",
            )
            if query_status == "slow":
                t = thresholds_cache.get(endpoint_key)
                warn_ms = t.warn_ms if t else 0
                ratio = (actual_ms / warn_ms) if warn_ms else 0.0
                logger.warning(
                    "slow_query endpoint=%s user=%s estimated=%s actual=%s ratio=%.2f",
                    endpoint_key,
                    user.id,
                    estimated_ms,
                    actual_ms,
                    ratio,
                )
            return response
        except Exception:
            actual_ms = int((time.monotonic() - t0) * 1000)
            # Re-read state en el except por si call_next levantó ANTES de
            # que asignáramos las locals. Si el router no llegó a setear
            # state, los getattr devuelven None — aceptable en error path.
            _persist(
                user_id=user.id,
                endpoint=endpoint_key,
                params_json=getattr(request.state, "params_json", None),
                estimated_ms=getattr(request.state, "estimated_ms", None),
                actual_ms=actual_ms,
                rows=None,
                status="error",
                approval_id=getattr(request.state, "approval_id", None),
                ip=request.client.host if request.client else "unknown",
            )
            raise


def _classify_status(
    *,
    actual_ms: int,
    endpoint_key: str,
    http_status: int,
) -> str:
    """Clasifica el status que se grabará en ``query_log.status``.

    Prioridad: timeout (504) > error (>=400) > slow (actual_ms > warn_ms*1.5)
    > ok. Thresholds ausentes → ``ok`` defensivo (no clasificamos slow
    si no sabemos el umbral).
    """
    if http_status == 504:
        return "timeout"
    if http_status >= 400:
        return "error"
    t = thresholds_cache.get(endpoint_key)
    if t is None:
        return "ok"
    if actual_ms > t.warn_ms * 1.5:
        return "slow"
    return "ok"


def _persist(
    *,
    user_id: Optional[int],
    endpoint: str,
    params_json: Optional[str],
    estimated_ms: Optional[int],
    actual_ms: int,
    rows: Optional[int],
    status: str,
    approval_id: Optional[int],
    ip: str,
) -> None:
    """Inserta una fila en ``nexo.query_log`` (best-effort, sin re-raise).

    Abre su propia Session (igual que AuditMiddleware) porque corre
    fuera del ciclo ``Depends`` de FastAPI.
    """
    try:
        db = SessionLocalNexo()
        try:
            db.add(
                NexoQueryLog(
                    ts=datetime.now(timezone.utc),
                    user_id=user_id,
                    endpoint=endpoint,
                    params_json=params_json,
                    estimated_ms=estimated_ms,
                    actual_ms=actual_ms,
                    rows=rows,
                    status=status,
                    approval_id=approval_id,
                    ip=ip,
                )
            )
            db.commit()
        finally:
            db.close()
    except Exception:
        logger.exception(
            "Error escribiendo nexo.query_log (endpoint=%s user_id=%s)",
            endpoint,
            user_id,
        )


__all__ = ["QueryTimingMiddleware"]
