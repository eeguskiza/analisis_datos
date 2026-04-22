"""Endpoints de conexion y configuracion del SQL Server MES."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.models import ConnectionStatus
from api.services import db as db_service
from nexo.services.auth import require_permission

router = APIRouter(
    prefix="/conexion",
    tags=["conexion"],
    dependencies=[Depends(require_permission("conexion:read"))],
)


# ── Modelos ───────────────────────────────────────────────────────────────────


class MesConfig(BaseModel):
    server: str = ""
    port: str = "1433"
    database: str = "dbizaro"
    driver: str = ""
    encrypt: str = ""
    trust_server_certificate: str = ""
    user: str = ""
    password: str = ""
    uf_code: str = ""


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/status", response_model=ConnectionStatus)
def status():
    ok, msg, server, database = db_service.check_connection()
    return ConnectionStatus(ok=ok, mensaje=msg, server=server, database=database)


@router.get("/config")
def get_config():
    """Devuelve la config actual del SQL Server (con password enmascarada para display)."""
    cfg = db_service.get_config()
    return {
        "server": cfg.get("server", ""),
        "port": cfg.get("port", "1433"),
        "database": cfg.get("database", "dbizaro"),
        "driver": cfg.get("driver", ""),
        "encrypt": cfg.get("encrypt", ""),
        "trust_server_certificate": cfg.get("trust_server_certificate", ""),
        "user": cfg.get("user", ""),
        "password": cfg.get("password", ""),
        "uf_code": cfg.get("uf_code", ""),
    }


@router.put(
    "/config",
    dependencies=[Depends(require_permission("conexion:config"))],
)
def save_config(payload: MesConfig):
    """Guarda la config del SQL Server."""
    cfg = db_service.get_config()

    cfg["server"] = payload.server
    cfg["port"] = payload.port
    cfg["database"] = payload.database
    cfg["driver"] = payload.driver
    cfg["encrypt"] = payload.encrypt
    cfg["trust_server_certificate"] = payload.trust_server_certificate
    cfg["user"] = payload.user
    cfg["password"] = payload.password
    cfg["uf_code"] = payload.uf_code

    db_service.update_config(cfg)
    return {"ok": True}


@router.post("/explorar")
def explorar_columnas():
    cols = db_service.explore_columns()
    return {"columnas": cols}
