"""Router de approvals (Plan 04-03 / QUERY-06).

Endpoints (paths absolutos — el router se registra sin prefix global en
``api/main.py``):

| Método | Path                              | Auth                      |
| ------ | --------------------------------- | ------------------------- |
| POST   | /api/approvals                    | cualquier auth user       |
| GET    | /api/approvals/count              | propietario (badge HTMX)  |
| POST   | /api/approvals/{id}/approve       | propietario               |
| POST   | /api/approvals/{id}/reject        | propietario               |
| POST   | /api/approvals/{id}/cancel        | owner (ownership check)   |
| GET    | /ajustes/solicitudes              | propietario (HTML page)   |
| GET    | /mis-solicitudes                  | cualquier auth user       |

Decisiones implementadas:

- **D-06**: modal RED → POST /api/approvals (frontend ya operativo en
  04-02).
- **D-13**: badge sidebar HTMX (/api/approvals/count) — sin email, sin
  banner.
- **D-14**: histórico 30d vía ``list_recent_non_pending(cutoff)``.
- **D-15**: ``consume_approval`` lo invocan pipeline/bbdd/capacidad/
  operarios; este router NO lo consume directamente (este flujo crea +
  aprueba + cancela; el consumo lo hace el router que ejecuta la query).
- **D-16**: POST /cancel con ownership check server-side.

Permisos:
- ``aprobaciones:manage`` (PERMISSION_MAP vacío → propietario-only) para
  /count, /approve, /reject y /ajustes/solicitudes.
- Self-service (cualquier user auth) para POST /api/approvals,
  /cancel y /mis-solicitudes. El ownership check va en la lógica (no en
  el permiso) porque cada user gestiona sus propias solicitudes.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from api.deps import DbNexo, render
from nexo.data.repositories.nexo import ApprovalRepo
from nexo.services import approvals as svc
from nexo.services.auth import require_permission

logger = logging.getLogger("nexo.approvals")


router = APIRouter(tags=["approvals"])


# ── Pydantic request body ────────────────────────────────────────────────

class CreateApprovalBody(BaseModel):
    """Body del POST /api/approvals — disparado por el modal RED del
    frontend (static/js/app.js::preflightModal::requestApproval).
    """

    endpoint: str
    params: dict
    estimated_ms: int


# ── POST /api/approvals ───────────────────────────────────────────────────

@router.post("/api/approvals")
async def create(
    body: CreateApprovalBody,
    request: Request,
    db: DbNexo,
) -> dict:
    """Crea una solicitud ``pending`` para el user autenticado.

    Self-service (cualquier user auth). Sin ``require_permission`` porque
    un user cualquiera pide aprobación para su propia query — el
    ownership check va en /cancel (no tiene sentido aquí).
    """
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    approval_id = svc.create_approval(
        db,
        user_id=user.id,
        endpoint=body.endpoint,
        params=body.params,
        estimated_ms=body.estimated_ms,
    )
    logger.info(
        "approval created id=%d user=%s endpoint=%s estimated_ms=%d",
        approval_id, user.email, body.endpoint, body.estimated_ms,
    )
    return {"approval_id": approval_id, "status": "pending"}


# ── GET /api/approvals/count (badge sidebar HTMX) ────────────────────────

@router.get(
    "/api/approvals/count",
    response_class=HTMLResponse,
    dependencies=[Depends(require_permission("aprobaciones:manage"))],
)
def count(db: DbNexo) -> HTMLResponse:
    """HTML fragment consumido por el badge sidebar del propietario
    (D-13). Devuelve string vacío si 0 pendientes o
    ``<span>(N)</span>`` si N>0.

    HTMX hace ``hx-get="/api/approvals/count" hx-trigger="every 30s"
    hx-swap="innerHTML"`` desde base.html — cada 30s el sidebar refresca.
    """
    n = ApprovalRepo(db).count_pending()
    if n == 0:
        return HTMLResponse("")
    return HTMLResponse(
        f'<span class="ml-2 text-xs bg-yellow-400 text-yellow-900 '
        f'px-1.5 py-0.5 rounded-full font-bold">({n})</span>'
    )


# ── POST /api/approvals/{id}/approve ──────────────────────────────────────

@router.post(
    "/api/approvals/{approval_id}/approve",
    dependencies=[Depends(require_permission("aprobaciones:manage"))],
)
def approve(
    approval_id: int,
    request: Request,
    db: DbNexo,
) -> RedirectResponse:
    """Propietario aprueba una solicitud. pending → approved."""
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    svc.approve(db, approval_id, user.id)
    logger.info(
        "approval approved id=%d by=%s", approval_id, user.email,
    )
    return RedirectResponse(
        "/ajustes/solicitudes?ok=approved", status_code=303,
    )


# ── POST /api/approvals/{id}/reject ───────────────────────────────────────

@router.post(
    "/api/approvals/{approval_id}/reject",
    dependencies=[Depends(require_permission("aprobaciones:manage"))],
)
def reject(
    approval_id: int,
    request: Request,
    db: DbNexo,
) -> RedirectResponse:
    """Propietario rechaza una solicitud. pending → rejected."""
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    svc.reject(db, approval_id, user.id)
    logger.info(
        "approval rejected id=%d by=%s", approval_id, user.email,
    )
    return RedirectResponse(
        "/ajustes/solicitudes?ok=rejected", status_code=303,
    )


# ── POST /api/approvals/{id}/cancel ───────────────────────────────────────

@router.post("/api/approvals/{approval_id}/cancel")
def cancel(
    approval_id: int,
    request: Request,
    db: DbNexo,
) -> RedirectResponse:
    """Owner cancela su propia solicitud pending (D-16).

    Ownership check server-side: ``svc.cancel`` devuelve False si el
    user no es el dueño o si status != pending. Traducimos False →
    HTTPException(403) — el frontend muestra el error al user.
    """
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    ok = svc.cancel(db, approval_id, user.id)
    if not ok:
        raise HTTPException(
            status_code=403,
            detail=(
                "No puedes cancelar esta solicitud "
                "(no eres el dueño o no está pending)"
            ),
        )
    logger.info(
        "approval cancelled id=%d by=%s", approval_id, user.email,
    )
    return RedirectResponse(
        "/mis-solicitudes?ok=cancelled", status_code=303,
    )


# ── GET /ajustes/solicitudes (HTML — propietario) ────────────────────────

@router.get(
    "/ajustes/solicitudes",
    response_class=HTMLResponse,
    dependencies=[Depends(require_permission("aprobaciones:manage"))],
)
def page_solicitudes(
    request: Request,
    db: DbNexo,
    ok: str | None = None,
    error: str | None = None,
) -> HTMLResponse:
    """Página de gestión de solicitudes — propietario-only.

    Dos secciones:
      - **Pendientes**: tabla con [Aprobar] y [Rechazar] inline.
      - **Histórico (30d)**: approved/rejected/cancelled/expired/
        consumed dentro de los últimos 30 días (D-14).

    ``list_recent_non_pending(cutoff)`` ya existe en ``ApprovalRepo``
    (entregado por Plan 04-01 — PC-04-06). Este router lo CONSUME.
    """
    repo = ApprovalRepo(db)
    pending = repo.list_pending()
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    historico = repo.list_recent_non_pending(cutoff)
    return render(
        "ajustes_solicitudes.html",
        request,
        {
            "page": "ajustes",
            "pending": pending,
            "historico": historico,
            "ok": ok,
            "error": error,
        },
    )


# ── GET /mis-solicitudes (HTML — cualquier user auth) ────────────────────

@router.get("/mis-solicitudes", response_class=HTMLResponse)
def page_mis_solicitudes(
    request: Request,
    db: DbNexo,
    ok: str | None = None,
    error: str | None = None,
) -> HTMLResponse:
    """Página donde el user ve sus solicitudes y cancela pendientes
    (D-16).

    Sin ``require_permission``: cualquier user autenticado accede y
    ve **solo las suyas** (WHERE user_id = current_user).
    """
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    repo = ApprovalRepo(db)
    mias = repo.list_by_user(user.id)
    return render(
        "mis_solicitudes.html",
        request,
        {
            "page": "mis_solicitudes",
            "solicitudes": mias,
            "ok": ok,
            "error": error,
        },
    )


__all__ = ["router", "CreateApprovalBody"]
