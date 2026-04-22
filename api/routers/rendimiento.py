"""Router de rendimiento — /ajustes/rendimiento (Plan 04-04 / D-11 / D-12).

Endpoints (paths absolutos, registrado sin prefix global):

| Método | Path                          | Auth         |
| ------ | ----------------------------- | ------------ |
| GET    | /ajustes/rendimiento          | propietario  |
| GET    | /api/rendimiento/summary      | propietario  |
| GET    | /api/rendimiento/timeseries   | propietario  |

Decisiones implementadas:

- **D-11**: página dedicada con filtros (user/endpoint/status/rango) +
  tabla summary (1 fila por endpoint) + Chart.js timeseries.
- **D-12**: Chart.js via CDN (``base.html``); fallback a tabla si CDN
  cae (``typeof Chart !== 'undefined'`` check inline en el template).

Permiso: ``rendimiento:read`` (lista vacía → propietario-only).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse

from api.deps import DbNexo, render
from nexo.data.repositories.nexo import QueryLogRepo, ThresholdRepo, UserRepo
from nexo.services.auth import require_permission

logger = logging.getLogger("nexo.rendimiento")


router = APIRouter(
    tags=["ajustes"],
    dependencies=[Depends(require_permission("rendimiento:read"))],
)


# Los 4 endpoints con preflight (mismas filas de nexo.query_thresholds).
_ENDPOINTS = ["pipeline/run", "bbdd/query", "capacidad", "operarios"]

# Status que puede aparecer en query_log (Plan 04-02 + 04-03).
_STATUSES = ["ok", "slow", "amber", "red", "approved_run", "timeout", "error"]


# ── Helpers de parseo de filtros ──────────────────────────────────────────


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """Acepta 'YYYY-MM-DD' o ISO completo. Devuelve tz-aware UTC o None."""
    if not value:
        return None
    try:
        if len(value) == 10:
            return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _resolve_range(
    rango: str,
    date_from_s: Optional[str],
    date_to_s: Optional[str],
) -> tuple[datetime, datetime]:
    """Traduce ``rango`` (preset o 'custom') a ``(date_from, date_to)``.

    Rangos preset: ``7d``, ``30d``, ``90d``. ``custom`` usa ``date_from``/
    ``date_to`` del querystring. Default: ``30d``.
    """
    now = datetime.now(timezone.utc)
    if rango == "custom":
        df = _parse_iso_datetime(date_from_s) or (now - timedelta(days=30))
        dt = _parse_iso_datetime(date_to_s) or now
        # Si viene solo fecha, extender date_to al final del día para
        # que el filtro sea inclusivo.
        if date_to_s and len(date_to_s) == 10:
            dt = dt.replace(hour=23, minute=59, second=59)
        return df, dt
    days = {"7d": 7, "30d": 30, "90d": 90}.get(rango, 30)
    return now - timedelta(days=days), now


# ── GET /ajustes/rendimiento (HTML) ───────────────────────────────────────


@router.get("/ajustes/rendimiento", response_class=HTMLResponse)
def page(
    request: Request,
    db: DbNexo,
    endpoint: Optional[str] = Query(None),
    user_id: Optional[int] = Query(None),
    status: Optional[str] = Query(None),
    rango: str = Query("30d"),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
) -> HTMLResponse:
    """Página de rendimiento — tabla summary + grafica Chart.js.

    Si ``endpoint`` es None, renderiza tabla agregada para los 4
    endpoints con preflight. La gráfica usa el endpoint seleccionado
    en el form (default: pipeline/run) — fetch a
    ``/api/rendimiento/timeseries`` desde el cliente.
    """
    df, dt = _resolve_range(rango, date_from, date_to)
    qrepo = QueryLogRepo(db)
    trepo = ThresholdRepo(db)

    endpoints_to_show = [endpoint] if endpoint else _ENDPOINTS
    summaries = []
    for ep in endpoints_to_show:
        summary = qrepo.summary(ep, df, dt)
        threshold = trepo.get(ep)
        summaries.append(
            {
                "endpoint": ep,
                "n_runs": summary.get("n_runs", 0) or 0,
                "avg_est": summary.get("avg_est"),
                "avg_actual": summary.get("avg_actual"),
                "p95": summary.get("p95"),
                "n_slow": summary.get("n_slow", 0) or 0,
                "divergence_pct": summary.get("divergence_pct", 0.0),
                "warn_ms": threshold.warn_ms if threshold else None,
                "block_ms": threshold.block_ms if threshold else None,
            }
        )

    users = UserRepo(db).list_all()

    chart_endpoint = endpoint or "pipeline/run"
    return render(
        "ajustes_rendimiento.html",
        request,
        {
            "page": "ajustes",
            "summaries": summaries,
            "endpoints": _ENDPOINTS,
            "statuses": _STATUSES,
            "users": users,
            "filters": {
                "endpoint": endpoint or "",
                "user_id": user_id or "",
                "status": status or "",
                "rango": rango,
                "date_from": date_from or df.date().isoformat(),
                "date_to": date_to or dt.date().isoformat(),
            },
            "chart_endpoint": chart_endpoint,
            "chart_date_from_iso": df.isoformat(),
            "chart_date_to_iso": dt.isoformat(),
        },
    )


# ── GET /api/rendimiento/summary (JSON) ───────────────────────────────────


@router.get("/api/rendimiento/summary")
def api_summary(
    db: DbNexo,
    endpoint: str = Query(...),
    date_from: str = Query(...),
    date_to: str = Query(...),
) -> dict:
    """JSON summary por endpoint + ventana temporal."""
    df = _parse_iso_datetime(date_from) or datetime.now(timezone.utc) - timedelta(
        days=30
    )
    dt = _parse_iso_datetime(date_to) or datetime.now(timezone.utc)
    result = QueryLogRepo(db).summary(endpoint, df, dt)
    # Asegurar tipos JSON-friendly (Decimal / datetime).
    return {
        "endpoint": endpoint,
        "date_from": df.isoformat(),
        "date_to": dt.isoformat(),
        "n_runs": int(result.get("n_runs") or 0),
        "avg_estimated_ms": (
            float(result["avg_est"]) if result.get("avg_est") is not None else None
        ),
        "avg_actual_ms": (
            float(result["avg_actual"])
            if result.get("avg_actual") is not None
            else None
        ),
        "p95_actual_ms": (
            float(result["p95"]) if result.get("p95") is not None else None
        ),
        "n_slow": int(result.get("n_slow") or 0),
        "divergence_pct": float(result.get("divergence_pct") or 0.0),
    }


# ── GET /api/rendimiento/timeseries (JSON — Chart.js) ─────────────────────


@router.get("/api/rendimiento/timeseries")
def api_timeseries(
    db: DbNexo,
    endpoint: str = Query(...),
    date_from: str = Query(...),
    date_to: str = Query(...),
) -> dict:
    """Bucketizado de estimated vs actual_ms para Chart.js (D-12).

    Granularity auto: hora si rango <= 7d, día si rango > 7d (ver
    ``QueryLogRepo.timeseries``).
    """
    df = _parse_iso_datetime(date_from) or datetime.now(timezone.utc) - timedelta(
        days=30
    )
    dt = _parse_iso_datetime(date_to) or datetime.now(timezone.utc)
    points, granularity = QueryLogRepo(db).timeseries(
        endpoint=endpoint,
        date_from=df,
        date_to=dt,
    )
    return {
        "endpoint": endpoint,
        "date_from": df.isoformat(),
        "date_to": dt.isoformat(),
        "granularity": granularity,
        "points": points,
    }


__all__ = ["router"]
