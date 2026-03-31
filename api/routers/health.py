"""Health check de todos los servicios."""
from __future__ import annotations

from fastapi import APIRouter

from api.database import check_db_health
from api.services import db as mes_service

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health_check():
    """Estado de todos los servicios de la plataforma."""
    services = {}

    # 1. Web (siempre OK si llega aqui)
    services["web"] = {"ok": True, "msg": "Operativo"}

    # 2. BBDD local (PostgreSQL / SQLite)
    db_ok, db_msg = check_db_health()
    services["db"] = {"ok": db_ok, "msg": db_msg}

    # 3. BD MES (SQL Server remota)
    try:
        ok, msg, server, database = mes_service.check_connection()
        services["mes"] = {"ok": ok, "msg": msg, "server": server, "database": database}
    except Exception as exc:
        services["mes"] = {"ok": False, "msg": str(exc)}

    all_ok = all(s["ok"] for s in services.values())
    return {"ok": all_ok, "services": services}
