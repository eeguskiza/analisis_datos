"""Facade MES — aisla el acceso a la BD del resto de la API.

Plan 03-02 Task 3.7 (handoff a 03-03): este modulo es el unico
consumidor legacy de ``OEE.db.connector``. Lo usa ``api/services/pipeline.py``
indirectamente (via ``extract_data``, ``compute_real_cycles``, etc.).

Repointing: las 4 funciones wrappable delegan ahora a
``MesRepository`` (construido sobre ``engine_mes`` de
``nexo/data/engines.py``). ``load_config`` / ``save_config`` /
``test_conexion`` / ``explorar_columnas_fmesdtc`` / ``datos_a_csvs``
siguen importados directamente del connector legacy porque el
repositorio no los envuelve (no son MES-queries sino helpers de
config/diagnostico/IO).

Beneficio: ``pipeline.py`` sigue llamando ``mes_service.extract_data(...)``
y automaticamente usa el backend nuevo — NO se toca ``pipeline.py``
en este plan (propiedad de 03-03 per ``<context_handoff>`` del PLAN.md).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, List

from api.config import settings

from OEE.db.connector import (
    datos_a_csvs,
    explorar_columnas_fmesdtc,
    load_config,
    save_config,
    test_conexion,
)


def _repo():
    """Factory helper — construye un MesRepository con engine_mes.

    Imports lazy para romper el ciclo:
    ``api.services.db`` <-> ``nexo.data.repositories.mes`` (el repo
    importa ``get_config`` de este modulo — Landmine #1 del RESEARCH).

    El engine es singleton con pool compartido, asi que el repo es
    barato de instanciar por-llamada. Permite mockear ``_repo`` en
    tests unitarios del facade sin tocar el repo real.
    """
    from nexo.data.engines import engine_mes
    from nexo.data.repositories.mes import MesRepository
    return MesRepository(engine=engine_mes)


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
    return _repo().detectar_recursos()


def extract_data(
    fecha_inicio: date,
    fecha_fin: date,
    recursos: list[str] | None = None,
) -> List[dict]:
    """Extrae datos de IZARO y devuelve lista de dicts."""
    return _repo().extraer_datos_produccion(
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        recursos=recursos,
    )


def compute_real_cycles(centro_trabajo: int, dias_atras: int = 30) -> tuple[List[dict], str]:
    """Calcula ciclos reales desde contadores de IZARO. Devuelve (resultados, fuente)."""
    return _repo().calcular_ciclos_reales(centro_trabajo, dias_atras)


def live_status(centro_trabajo: int, umbral_activo_seg: int = 600) -> dict:
    """Estado en vivo: ultima lectura de contador y si la maquina esta activa."""
    return _repo().estado_maquina_live(centro_trabajo, umbral_activo_seg)


def write_csvs(rows: List[dict], recursos_dir: Path | None = None) -> Dict[str, Path]:
    """Escribe datos como CSVs para los módulos OEE."""
    target = recursos_dir or settings.recursos_dir
    return datos_a_csvs(rows, target)
