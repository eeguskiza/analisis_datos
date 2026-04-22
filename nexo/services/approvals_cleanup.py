"""Job ``approvals_cleanup`` (Plan 04-03 / D-14).

Marca ``expired`` las solicitudes ``pending`` con
``created_at < now - NEXO_APPROVAL_TTL_DAYS`` (default 7).

Ejecuta:
- 1 vez por semana (Monday 03:05 UTC) desde el scheduler asyncio de
  ``cleanup_scheduler.cleanup_loop``.
- On-demand via ``run_once()`` (tests + operator debug).

Cada ejecución graba una fila en ``nexo.audit_log`` con
``path='__cleanup_approvals__'`` y ``details_json={rows_expired,
cutoff_ts, ttl_days}`` para trazabilidad (misma convención que Phase 2
audit_log + Plan 04-04 query_log_cleanup).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from nexo.data.engines import SessionLocalNexo
from nexo.data.repositories.nexo import AuditRepo
from nexo.services import approvals

log = logging.getLogger("nexo.approvals_cleanup")


def run_once() -> int:
    """Ejecuta un ciclo de expiración. Devuelve filas marcadas expired.

    Abre su propia Session (no depende del lifespan de FastAPI — se
    puede llamar desde el scheduler async o desde tests sync).
    """
    ttl_days = int(os.environ.get("NEXO_APPROVAL_TTL_DAYS", "7"))
    now_utc = datetime.now(timezone.utc)
    db = SessionLocalNexo()
    try:
        rows_expired = approvals.expire_stale(db, ttl_days)
        # Audit log entry (best-effort — si falla el audit, el expire ya
        # está commiteado por approvals.expire_stale).
        try:
            AuditRepo(db).append(
                user_id=None,
                ip="127.0.0.1",
                method="DELETE",
                path="__cleanup_approvals__",
                status=200,
                details_json=json.dumps(
                    {
                        "rows_expired": rows_expired,
                        "cutoff_ts": now_utc.isoformat(),
                        "ttl_days": ttl_days,
                    }
                ),
            )
            db.commit()
        except Exception:
            log.exception("approvals_cleanup audit append failed")
            db.rollback()
        log.info(
            "approvals_cleanup: %d solicitudes expiradas (ttl_days=%d)",
            rows_expired,
            ttl_days,
        )
        return rows_expired
    finally:
        db.close()


__all__ = ["run_once"]
