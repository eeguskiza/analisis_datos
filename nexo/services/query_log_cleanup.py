"""Job ``query_log_cleanup`` (Plan 04-04 / D-10 / QUERY-01).

Purga filas antiguas de ``nexo.query_log`` segun
``NEXO_QUERY_LOG_RETENTION_DAYS`` (default 90). ``0`` = forever (skip).

Ejecuta:
- 1 vez por semana (Monday 03:00 UTC) desde el scheduler asyncio de
  ``cleanup_scheduler.cleanup_loop``.
- On-demand via ``run_once()`` (tests + operator debug).

Cada ejecucion graba una fila en ``nexo.audit_log`` con
``path='__cleanup_query_log__'`` + ``details_json={rows_deleted,
cutoff_ts, retention_days}`` para trazabilidad.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from nexo.data.engines import SessionLocalNexo
from nexo.data.repositories.nexo import AuditRepo

log = logging.getLogger("nexo.query_log_cleanup")


def run_once() -> int:
    """Ejecuta un ciclo de purga. Devuelve filas borradas (0 si skip).

    Abre su propia Session (no depende del lifespan FastAPI — se puede
    llamar desde el scheduler async o desde tests sync).
    """
    retention_days = int(
        os.environ.get("NEXO_QUERY_LOG_RETENTION_DAYS", "90")
    )
    if retention_days <= 0:
        log.info(
            "query_log_cleanup: retention_days=%d (forever) — skip",
            retention_days,
        )
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    db = SessionLocalNexo()
    try:
        result = db.execute(
            text("DELETE FROM nexo.query_log WHERE ts < :cutoff"),
            {"cutoff": cutoff},
        )
        rows_deleted = result.rowcount or 0
        db.commit()

        # Audit log entry — best-effort. Si falla el append, el DELETE
        # ya esta commiteado (mismo patron que approvals_cleanup).
        try:
            AuditRepo(db).append(
                user_id=None,
                ip="127.0.0.1",
                method="DELETE",
                path="__cleanup_query_log__",
                status=200,
                details_json=json.dumps({
                    "rows_deleted": rows_deleted,
                    "cutoff_ts": cutoff.isoformat(),
                    "retention_days": retention_days,
                }),
            )
            db.commit()
        except Exception:
            log.exception("query_log_cleanup audit append failed")
            db.rollback()

        log.info(
            "query_log_cleanup: %d filas borradas (retention_days=%d, cutoff=%s)",
            rows_deleted, retention_days, cutoff.isoformat(),
        )
        return rows_deleted
    finally:
        db.close()


__all__ = ["run_once"]
