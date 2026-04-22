"""UI y export CSV de nexo.audit_log - solo propietario.

Plan 03-03 Task 4.5: refactor para consumir ``AuditRepo``
(``nexo.data.repositories.nexo``) y usar ``DbNexo`` de ``api.deps``
en vez del ``get_nexo_db`` local duplicado.

Permission: ``auditoria:read`` (lista vacia en PERMISSION_MAP -> bypass
del propietario, resto 403).

Endpoints:
- ``GET /ajustes/auditoria``         -> listado paginado + filtros
- ``GET /ajustes/auditoria/export``  -> CSV streaming con los filtros aplicados

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

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from api.deps import DbNexo, render
from nexo.data.engines import SessionLocalNexo
from nexo.data.repositories.nexo import AuditRepo
from nexo.services.auth import require_permission

logger = logging.getLogger("nexo.auditoria")


router = APIRouter(
    prefix="/ajustes/auditoria",
    tags=["ajustes"],
    dependencies=[Depends(require_permission("auditoria:read"))],
)


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """Acepta 'YYYY-MM-DD' o ISO completo. Devuelve tz-aware UTC o None."""
    if not value:
        return None
    try:
        # date-only -> assume 00:00:00 UTC
        if len(value) == 10:
            return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _parse_date_to_end_of_day(value: Optional[str]) -> Optional[datetime]:
    """Como ``_parse_iso_datetime`` pero si viene solo fecha, extiende
    al final del dia (23:59:59 UTC) para que el filtro date_to sea
    inclusivo."""
    dt = _parse_iso_datetime(value)
    if dt is None:
        return None
    if value is not None and len(value) == 10:
        dt = dt.replace(hour=23, minute=59, second=59)
    return dt


@router.get("", response_class=HTMLResponse)
async def listar(
    request: Request,
    db: DbNexo,
    user_email: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    path: Optional[str] = Query(None),
    status: Optional[int] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1, le=500),
):
    repo = AuditRepo(db)

    email_norm = user_email.strip().lower() if user_email else None
    dt_from = _parse_iso_datetime(date_from)
    dt_to = _parse_date_to_end_of_day(date_to)

    total = repo.count_filtered(
        user_email=email_norm,
        date_from=dt_from,
        date_to=dt_to,
        path=path or None,
        status=status,
    )
    dtos = repo.list_filtered(
        user_email=email_norm,
        date_from=dt_from,
        date_to=dt_to,
        path=path or None,
        status=status,
        page=page,
        limit=limit,
    )

    serialized = [
        {
            "id": r.id,
            "ts": r.ts,
            "user_email": r.user_email or "(desconocido)",
            "user_id": r.user_id,
            "ip": r.ip,
            "method": r.method,
            "path": r.path,
            "status": r.status,
            "details_json": r.details_json,
        }
        for r in dtos
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
    """CSV streaming con los filtros aplicados. No hay paginacion - el
    CSV cubre todas las filas que matchean. Memory-safe via yield_per
    en ``AuditRepo.iter_filtered``."""

    email_norm = user_email.strip().lower() if user_email else None
    dt_from = _parse_iso_datetime(date_from)
    dt_to = _parse_date_to_end_of_day(date_to)

    def row_iterator() -> Iterator[str]:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "ts",
                "user_email",
                "user_id",
                "ip",
                "method",
                "path",
                "status",
                "details_json",
            ]
        )
        yield buf.getvalue()
        buf.seek(0)
        buf.truncate(0)

        db = SessionLocalNexo()
        try:
            repo = AuditRepo(db)
            for row in repo.iter_filtered(
                user_email=email_norm,
                date_from=dt_from,
                date_to=dt_to,
                path=path or None,
                status=status,
            ):
                writer.writerow(
                    [
                        row.ts.isoformat(),
                        row.user_email or "",
                        row.user_id or "",
                        row.ip or "",
                        row.method,
                        row.path,
                        row.status,
                        row.details_json or "",
                    ]
                )
                yield buf.getvalue()
                buf.seek(0)
                buf.truncate(0)
        finally:
            db.close()

    filename = f"nexo_audit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        row_iterator(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
