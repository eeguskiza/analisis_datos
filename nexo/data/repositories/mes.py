"""MesRepository — acceso a SQL Server ``dbizaro`` (DATA-02, Plan 03-02).

Cuatro metodos son wrappers delgados sobre ``OEE/db/connector.py``
(decision D-04: NO reescribir la logica legacy, solo envolver). El
quinto metodo ``consulta_readonly`` es el entry point para ``/bbdd``
— recibe SQL **ya validada por el whitelist del router** (D-05). El
sexto ``centro_mando_fmesmic`` es la query nueva con 2-part names +
``bindparam`` expanding (DATA-09, reemplaza el string-interpolation
del router viejo).

Naming: el constructor acepta un ``Engine`` porque MES es read-only
por convencion (no hay transacciones a gestionar). Los wrappers ignoran
el engine y delegan en el connector legacy via su propia conexion
pyodbc; el engine se usa unicamente por ``consulta_readonly`` y
``centro_mando_fmesmic``.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from nexo.data.sql.loader import load_sql

# Reusamos ``get_config`` existente — Landmine #1: NO reimplementar.
from api.services.db import get_config as _get_mes_config

from OEE.db.connector import (
    calcular_ciclos_reales as _legacy_calcular_ciclos,
    detectar_recursos as _legacy_detectar_recursos,
    estado_maquina_live as _legacy_estado_maquina,
    extraer_datos as _legacy_extraer_datos,
)


class MesRepository:
    """Repositorio MES (``dbizaro`` read-only por convencion).

    Engine inyectado (no ``Session`` — MES no tiene transaccion de
    write). Los cuatro metodos legacy-wrapping usan ``cfg`` via
    ``api.services.db.get_config``; ``consulta_readonly`` y
    ``centro_mando_fmesmic`` usan ``self._engine`` directamente.
    """

    def __init__(self, engine: Engine):
        self._engine = engine

    # ── Wrappers delgados (D-04) ────────────────────────────────────────
    def extraer_datos_produccion(
        self,
        *,
        fecha_inicio: date,
        fecha_fin: date,
        recursos: list[str] | None = None,
    ) -> list[dict]:
        """Extrae datos de produccion del MES legacy.

        Preserva la logica del turno-3 (NOTA T3 en el connector). Si
        ``recursos`` viene informado, filtra el ``cfg`` antes de llamar
        al legacy para que el connector solo consulte los CTs pedidos.
        """
        cfg = _get_mes_config()
        if recursos:
            cfg = {
                **cfg,
                "recursos": [
                    r for r in cfg.get("recursos", []) if r.get("nombre") in recursos
                ],
            }
        return _legacy_extraer_datos(cfg, fecha_inicio, fecha_fin)

    def detectar_recursos(self) -> list[dict]:
        """Devuelve los centros de trabajo detectados por el MES."""
        cfg = _get_mes_config()
        return _legacy_detectar_recursos(cfg)

    def calcular_ciclos_reales(
        self,
        centro_trabajo: int,
        dias_atras: int = 30,
    ) -> tuple[list[dict], str]:
        """Ciclos reales por referencia para un CT (legacy wrapper)."""
        cfg = _get_mes_config()
        return _legacy_calcular_ciclos(cfg, centro_trabajo, dias_atras)

    def estado_maquina_live(
        self,
        centro_trabajo: int,
        umbral_activo_seg: int = 600,
    ) -> dict:
        """Estado en vivo (ultimo evento + activa/no) de un CT."""
        cfg = _get_mes_config()
        return _legacy_estado_maquina(cfg, centro_trabajo, umbral_activo_seg)

    # ── Consulta libre validada por el router (D-05) ────────────────────
    def consulta_readonly(
        self,
        sql: str,
        database: str = "dbizaro",
    ) -> dict[str, Any]:
        """Ejecuta SQL ya validada por ``api/routers/bbdd.py`` whitelist.

        ``database`` es informativo; ``self._engine`` apunta a
        ``dbizaro`` via DSN (plan 03-01). Devuelve
        ``{"columns": [...], "rows": [[...], ...]}`` shape que consume
        la UI del explorer.
        """
        with self._engine.connect() as conn:
            result = conn.execute(text(sql))
            cols = list(result.keys())
            rows = [list(r) for r in result.fetchall()]
        return {"columns": cols, "rows": rows}

    # ── Query nueva con load_sql + bindparam expanding (DATA-09) ────────
    def centro_mando_fmesmic(self, ct_codes: list[int]) -> list[dict]:
        """Estado actual ``fmesmic`` por CT. Reemplaza ``_query_fmesmic``.

        Cambios respecto al original:

        - ``FROM admuser.fmesmic`` (2-part name) — ``engine_mes`` ya
          apunta a ``dbizaro`` via DSN.
        - ``IN :codes`` via ``bindparam(expanding=True)``, no string
          interpolation.
        """
        if not ct_codes:
            return []
        stmt = text(load_sql("mes/centro_mando_fmesmic")).bindparams(
            bindparam("codes", expanding=True),
        )
        with self._engine.connect() as conn:
            result = conn.execute(stmt, {"codes": [str(c) for c in ct_codes]})
            return [dict(r._mapping) for r in result.fetchall()]


__all__ = ["MesRepository"]
