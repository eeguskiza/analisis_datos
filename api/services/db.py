"""Wrapper sobre OEE.db.connector — aísla el acceso a BD del resto de la API."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, List

from api.config import settings

from OEE.db.connector import (
    explorar_columnas_fmesdtc,
    extraer_y_guardar_csv,
    load_config,
    save_config,
    test_conexion,
)


def get_config() -> dict:
    return load_config()


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


def extract_csvs(
    fecha_inicio: date,
    fecha_fin: date,
) -> Dict[str, Path]:
    """Extrae datos de la BD y genera CSVs en data/recursos/<SECCION>/."""
    cfg = get_config()
    return extraer_y_guardar_csv(cfg, fecha_inicio, fecha_fin, settings.recursos_dir)
