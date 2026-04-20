"""Router de umbrales — /ajustes/limites (Plan 04-04 / QUERY-02 / D-04 / D-19).

Endpoints (paths absolutos, registrado sin prefix global):

| Método | Path                                                 | Auth             |
| ------ | ---------------------------------------------------- | ---------------- |
| GET    | /ajustes/limites                                     | propietario      |
| PUT    | /api/thresholds/{endpoint:path}                      | propietario      |
| POST   | /api/thresholds/{endpoint:path}/recalibrate          | propietario      |

Decisiones implementadas:

- **D-04**: botón "Recalcular factor (30 runs)" con preview + confirm.
  Algoritmo vive en ``nexo/services/factor_learning.compute_factor``
  (compartido con el cron mensual ``factor_auto_refresh``).
- **D-19**: cada UPDATE/recalibrate emite NOTIFY
  ``nexo_thresholds_changed, <endpoint>`` via
  ``thresholds_cache.notify_changed``. El listener background de cada
  worker uvicorn reacciona en <1s (thresholds_cache.reload_one).

Permiso: ``limites:manage`` (lista vacía en PERMISSION_MAP → bypass
propietario, resto 403).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from api.deps import DbNexo, render
from nexo.data.repositories.nexo import QueryLogRepo, ThresholdRepo
from nexo.services import thresholds_cache
from nexo.services.auth import require_permission
from nexo.services.factor_learning import compute_factor

logger = logging.getLogger("nexo.limites")


router = APIRouter(
    tags=["ajustes"],
    dependencies=[Depends(require_permission("limites:manage"))],
)


# ── Pydantic bodies ───────────────────────────────────────────────────────


class UpdateThresholdBody(BaseModel):
    """Body del PUT /api/thresholds/{endpoint}.

    El factor NO se edita aquí — solo via recalibrate (que reusa
    compute_factor + emite NOTIFY + marca factor_updated_at).
    """

    warn_ms: int = Field(..., ge=0, description="Umbral amber en ms")
    block_ms: int = Field(..., ge=0, description="Umbral red en ms")


# ── GET /ajustes/limites (HTML) ───────────────────────────────────────────


@router.get("/ajustes/limites", response_class=HTMLResponse)
def page(request: Request, db: DbNexo) -> HTMLResponse:
    """Página con 4 filas editables (una por endpoint con preflight)."""
    thresholds = ThresholdRepo(db).list_all()
    qrepo = QueryLogRepo(db)
    n_runs_by_endpoint = {
        t.endpoint: qrepo.count_filtered(endpoint=t.endpoint, status="ok")
        + qrepo.count_filtered(endpoint=t.endpoint, status="slow")
        for t in thresholds
    }
    return render(
        "ajustes_limites.html",
        request,
        {
            "page": "ajustes",
            "thresholds": thresholds,
            "n_runs_by_endpoint": n_runs_by_endpoint,
        },
    )


# ── PUT /api/thresholds/{endpoint} ────────────────────────────────────────


@router.put("/api/thresholds/{endpoint:path}")
def update(
    endpoint: str,
    body: UpdateThresholdBody,
    request: Request,
    db: DbNexo,
) -> dict:
    """Actualiza warn_ms/block_ms. Emite NOTIFY para propagación cross-worker.

    ``factor_ms`` NO se modifica aquí — se recalcula via
    ``/recalibrate``. ``factor_touched=False`` preserva
    ``factor_updated_at``.
    """
    user = request.state.user
    repo = ThresholdRepo(db)
    current = repo.get(endpoint)
    if current is None:
        raise HTTPException(
            status_code=404,
            detail=f"Threshold {endpoint!r} no existe",
        )
    if body.warn_ms >= body.block_ms:
        raise HTTPException(
            status_code=400,
            detail=(
                f"warn_ms ({body.warn_ms}) debe ser < block_ms "
                f"({body.block_ms})"
            ),
        )
    repo.update(
        endpoint=endpoint,
        warn_ms=body.warn_ms,
        block_ms=body.block_ms,
        factor_ms=None,
        updated_by=user.id,
        factor_touched=False,
    )
    thresholds_cache.notify_changed(endpoint)
    logger.info(
        "threshold updated endpoint=%s warn_ms=%d block_ms=%d by=%s",
        endpoint, body.warn_ms, body.block_ms, user.email,
    )
    return {
        "ok": True,
        "endpoint": endpoint,
        "warn_ms": body.warn_ms,
        "block_ms": body.block_ms,
    }


# ── POST /api/thresholds/{endpoint}/recalibrate (D-04) ─────────────────────


@router.post("/api/thresholds/{endpoint:path}/recalibrate")
def recalibrate(
    endpoint: str,
    request: Request,
    db: DbNexo,
    confirm: bool = False,
) -> dict:
    """Recalcula factor_ms desde los últimos 30 runs (D-04).

    Flujo:
      1. ``?confirm=false`` (default) → preview: devuelve
         ``{factor_old, factor_new, sample_size}`` sin persistir.
      2. ``?confirm=true`` → persiste + emite NOTIFY + marca
         ``factor_updated_at = now()``.

    Levanta 400 si hay < 10 runs válidos tras filtrar outliers
    (Pitfall 6: actual_ms > 500ms).
    """
    user = request.state.user
    repo = ThresholdRepo(db)
    current = repo.get(endpoint)
    if current is None:
        raise HTTPException(
            status_code=404,
            detail=f"Threshold {endpoint!r} no existe",
        )

    factor_new, sample_size = compute_factor(db, endpoint, max_samples=30)
    if factor_new is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Solo hay {sample_size} runs válidos; se requieren al "
                f"menos 10 para recalibrar (actual_ms > 500ms + "
                f"params parseables si pipeline)."
            ),
        )

    if not confirm:
        # Preview: el frontend muestra factor_old vs factor_new y pide
        # confirm al usuario antes de persistir.
        return {
            "endpoint": endpoint,
            "factor_old": current.factor_ms,
            "factor_new": factor_new,
            "sample_size": sample_size,
            "preview": True,
            "committed": False,
        }

    # Persistir: update + NOTIFY. factor_touched=True marca
    # factor_updated_at=now() para el cron mensual (D-20).
    repo.update(
        endpoint=endpoint,
        warn_ms=current.warn_ms,
        block_ms=current.block_ms,
        factor_ms=factor_new,
        updated_by=user.id,
        factor_touched=True,
    )
    thresholds_cache.notify_changed(endpoint)
    logger.info(
        "threshold recalibrated endpoint=%s factor_old=%s factor_new=%.1f "
        "sample_size=%d by=%s",
        endpoint, current.factor_ms, factor_new, sample_size, user.email,
    )
    return {
        "endpoint": endpoint,
        "factor_old": current.factor_ms,
        "factor_new": factor_new,
        "sample_size": sample_size,
        "preview": False,
        "committed": True,
    }


__all__ = ["router"]
