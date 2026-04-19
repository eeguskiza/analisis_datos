"""UI y export CSV de nexo.audit_log — solo propietario.

Permission: ``auditoria:read`` (lista vacia en PERMISSION_MAP → bypass
del propietario, resto 403).

Endpoints:
- ``GET /ajustes/auditoria``         → listado paginado + filtros
- ``GET /ajustes/auditoria/export``  → CSV streaming con los filtros aplicados

Filtros admitidos (query params):
- ``user_email`` (match exacto)
- ``date_from``, ``date_to`` (ISO 8601)
- ``path`` (ILIKE sustring match)
- ``status`` (match exacto, int)

Paginacion server-side: ``?page=N&limit=100`` (max 500).
"""
from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timezone
from typing import Iterator, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from api.deps import render
from nexo.db.engine import SessionLocalNexo
from nexo.db.models import NexoAuditLog, NexoUser
from nexo.services.auth import require_permission

logger = logging.getLogger("nexo.auditoria")


router = APIRouter(
    prefix="/ajustes/auditoria",
    tags=["ajustes"],
    dependencies=[Depends(require_permission("auditoria:read"))],
)


def get_nexo_db():
    db = SessionLocalNexo()
    try:
        yield db
    finally:
        db.close()


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """Acepta 'YYYY-MM-DD' o ISO completo. Devuelve tz-aware UTC o None."""
    if not value:
        return None
    try:
        # date-only → assume 00:00:00 UTC
        if len(value) == 10:
            return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _build_filtered_stmt(
    user_email: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    path: Optional[str],
    status: Optional[int],
):
    """Construye un SELECT sobre NexoAuditLog + NexoUser con los filtros
    aplicados. Devuelve el stmt sin ORDER BY/LIMIT (los aplican los
    callers segun lo necesiten)."""
    stmt = select(NexoAuditLog, NexoUser).outerjoin(
        NexoUser, NexoUser.id == NexoAuditLog.user_id
    )
    conditions = []
    if user_email:
        conditions.append(NexoUser.email == user_email.strip().lower())
    dt_from = _parse_iso_datetime(date_from)
    if dt_from:
        conditions.append(NexoAuditLog.ts >= dt_from)
    dt_to = _parse_iso_datetime(date_to)
    if dt_to:
        # Si viene solo fecha, extender a fin del dia.
        if len(date_to or "") == 10:
            dt_to = dt_to.replace(hour=23, minute=59, second=59)
        conditions.append(NexoAuditLog.ts <= dt_to)
    if path:
        conditions.append(NexoAuditLog.path.ilike(f"%{path}%"))
    if status is not None:
        conditions.append(NexoAuditLog.status == status)
    if conditions:
        stmt = stmt.where(and_(*conditions))
    return stmt


@router.get("", response_class=HTMLResponse)
async def listar(
    request: Request,
    user_email: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    path: Optional[str] = Query(None),
    status: Optional[int] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_nexo_db),
):
    from sqlalchemy import func

    stmt = _build_filtered_stmt(user_email, date_from, date_to, path, status)

    # Count total con subquery — evita materializar las filas.
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = db.execute(count_stmt).scalar_one() or 0

    offset = (page - 1) * limit
    rows = db.execute(
        stmt.order_by(NexoAuditLog.ts.desc()).limit(limit).offset(offset)
    ).all()

    serialized = [
        {
            "id": r.NexoAuditLog.id,
            "ts": r.NexoAuditLog.ts,
            "user_email": r.NexoUser.email if r.NexoUser else "(desconocido)",
            "user_id": r.NexoAuditLog.user_id,
            "ip": r.NexoAuditLog.ip,
            "method": r.NexoAuditLog.method,
            "path": r.NexoAuditLog.path,
            "status": r.NexoAuditLog.status,
            "details_json": r.NexoAuditLog.details_json,
        }
        for r in rows
    ]

    total_pages = (total + limit - 1) // limit if total else 1

    return render(
        "ajustes_auditoria.html",
        request,
        {
            "page": "ajustes",
            "rows": serialized,
            "total": total,
            "page_n": page,
            "limit": limit,
            "total_pages": total_pages,
            "filters": {
                "user_email": user_email or "",
                "date_from": date_from or "",
                "date_to": date_to or "",
                "path": path or "",
                "status": status if status is not None else "",
            },
        },
    )


@router.get("/export")
async def export_csv(
    user_email: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    path: Optional[str] = Query(None),
    status: Optional[int] = Query(None),
):
    """CSV streaming con los filtros aplicados. No hay paginacion — el
    CSV cubre todas las filas que matchean. Memory-safe via ``yield_per``."""

    def row_iterator() -> Iterator[str]:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            ["ts", "user_email", "user_id", "ip", "method", "path", "status", "details_json"]
        )
        yield buf.getvalue()
        buf.seek(0); buf.truncate(0)

        db = SessionLocalNexo()
        try:
            stmt = (
                _build_filtered_stmt(user_email, date_from, date_to, path, status)
                .order_by(NexoAuditLog.ts.desc())
                .execution_options(yield_per=500)
            )
            for row in db.execute(stmt):
                log = row.NexoAuditLog
                user = row.NexoUser
                writer.writerow(
                    [
                        log.ts.isoformat(),
                        user.email if user else "",
                        log.user_id or "",
                        log.ip or "",
                        log.method,
                        log.path,
                        log.status,
                        log.details_json or "",
                    ]
                )
                yield buf.getvalue()
                buf.seek(0); buf.truncate(0)
        finally:
            db.close()

    filename = f"nexo_audit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        row_iterator(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
