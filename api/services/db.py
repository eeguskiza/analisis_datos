"""Wrapper sobre OEE.db.connector — aísla el acceso a BD del resto de la API."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, List

from api.config import settings

from OEE.db.connector import (
    calcular_ciclos_reales,
    datos_a_csvs,
    detectar_recursos,
    explorar_columnas_fmesdtc,
    extraer_datos,
    load_config,
    save_config,
    test_conexion,
)


def get_config() -> dict:
    from api.config import settings
    cfg = load_config()
    # Inyectar credenciales desde settings (.env) si no vienen del JSON
    if not cfg.get("server") or cfg["server"] == "":
        cfg["server"] = settings.db_server
    if not cfg.get("port") or cfg["port"] == "":
        cfg["port"] = str(settings.db_port)
    if not cfg.get("user") or cfg["user"] == "":
        cfg["user"] = settings.db_user
    if not cfg.get("password") or cfg["password"] == "":
        cfg["password"] = settings.db_password
    if not cfg.get("database") or cfg["database"] == "":
        cfg["database"] = settings.izaro_db
    return cfg


def update_config(cfg: dict) -> None:
    save_config(cfg)


def check_connection() -> tuple[bool, str, str, str]:
    """Devuelve (ok, mensaje, server, database)."""
    cfg = get_config()
    msg = test_conexion(cfg)
    ok = msg.startswith("OK")
    return ok, msg, cfg.get("server", ""), cfg.get("database", "")


def explore_columns() -> List[str]:
    cfg = get_config()
    return explorar_columnas_fmesdtc(cfg)


def discover_resources() -> List[dict]:
    """Detecta centros de trabajo disponibles en IZARO."""
    cfg = get_config()
    return detectar_recursos(cfg)


def extract_data(
    fecha_inicio: date,
    fecha_fin: date,
    recursos: list[str] | None = None,
) -> List[dict]:
    """Extrae datos de IZARO y devuelve lista de dicts."""
    cfg = get_config()
    if recursos:
        cfg = {**cfg, "recursos": [
            r for r in cfg.get("recursos", [])
            if r.get("nombre") in recursos
        ]}
    return extraer_datos(cfg, fecha_inicio, fecha_fin)


def compute_real_cycles(centro_trabajo: int, dias_atras: int = 30) -> List[dict]:
    """Calcula ciclos reales desde contadores de IZARO."""
    cfg = get_config()
    return calcular_ciclos_reales(cfg, centro_trabajo, dias_atras)


def write_csvs(rows: List[dict], recursos_dir: Path | None = None) -> Dict[str, Path]:
    """Escribe datos como CSVs para los módulos OEE."""
    target = recursos_dir or settings.recursos_dir
    return datos_a_csvs(rows, target)
