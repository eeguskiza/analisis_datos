"""Health check de todos los servicios."""

from __future__ import annotations

from fastapi import APIRouter

from api.database import check_db_health
from api.services import db as mes_service
from nexo.data.engines import engine_nexo

router = APIRouter(prefix="/health", tags=["health"])


def _check_postgres() -> tuple[bool, str]:
    try:
        from sqlalchemy import text
        with engine_nexo.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "OK"
    except Exception as exc:
        return False, str(exc)


@router.get("")
def health_check():
    """Estado de todos los servicios de la plataforma."""
    services = {}

    # 1. Web
    services["web"] = {"ok": True, "msg": "Operativo"}

    # 2. Postgres local (nexo.*)
    pg_ok, pg_msg = _check_postgres()
    services["db_local"] = {"ok": pg_ok, "msg": pg_msg, "database": "nexo"}

    # 3. SQL Server ecs_mobility (APP)
    ecs_ok, ecs_msg = check_db_health()
    services["db_ecs"] = {"ok": ecs_ok, "msg": ecs_msg, "database": "ecs_mobility"}

    # 4. SQL Server dbizaro (MES, read-only)
    try:
        ok, msg, server, database = mes_service.check_connection()
        services["db_izaro"] = {"ok": ok, "msg": msg, "server": server, "database": database}
    except Exception as exc:
        services["db_izaro"] = {"ok": False, "msg": str(exc), "database": "dbizaro"}

    all_ok = all(s["ok"] for s in services.values())
    return {"ok": all_ok, "services": services}
