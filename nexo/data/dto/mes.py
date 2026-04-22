"""DTOs frozen (Pydantic v2) que cruzan el borde router <-> repo para MES.

DATA-08: ``model_config = ConfigDict(frozen=True, from_attributes=True)``
en todos los ``*Row``. ``from_attributes=True`` permite
``Row.model_validate(orm_entity)`` aunque aqui los repos entreguen
``dict`` (el flag esta por consistencia con el resto de DTOs Nexo).

La fuente de verdad de los campos es el output real de
``OEE.db.connector.*``; si el legacy cambia su shape, estos DTOs deben
actualizarse.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, field_validator

from nexo.data.dto.base import ROW_CONFIG


class ProduccionRow(BaseModel):
    """Fila de produccion de ``extraer_datos_produccion``.

    Shape tomado del output actual de
    ``OEE.db.connector.extraer_datos``.
    """

    model_config = ROW_CONFIG

    recurso: str
    seccion: str
    fecha: date
    h_ini: Optional[str] = None
    h_fin: Optional[str] = None
    tiempo: float = 0.0
    proceso: Optional[str] = None
    incidencia: str = ""
    cantidad: float = 0.0
    malas: float = 0.0
    recuperadas: float = 0.0
    referencia: str = ""

    @field_validator("fecha", mode="before")
    @classmethod
    def _coerce_fecha(cls, v):
        if hasattr(v, "date") and not isinstance(v, date):
            return v.date()
        return v


class EstadoMaquinaRow(BaseModel):
    """Output de ``estado_maquina_live`` (shape usado por centro_mando)."""

    model_config = ROW_CONFIG

    centro_trabajo: int
    activa: bool
    ultimo_evento_ts: Optional[datetime] = None
    piezas_hoy: int = 0


class CicloRealRow(BaseModel):
    """Output de ``calcular_ciclos_reales`` por referencia."""

    model_config = ROW_CONFIG

    referencia: str
    ciclo_seg: float
    n_muestras: int = 0


class CapacidadRow(BaseModel):
    """Shape consumida por ``/capacidad`` endpoint."""

    model_config = ROW_CONFIG

    referencia: str
    ct: str
    ct_nombre: str
    piezas_reales: float = 0.0
    tiempo_trabajado_min: float = 0.0
    ciclo_teorico_seg: Optional[float] = None
    fecha_min: Optional[str] = None
    fecha_max: Optional[str] = None


class OperarioRow(BaseModel):
    """Shape de ``/operarios`` listar."""

    model_config = ROW_CONFIG

    codigo: int
    nombre: str
    activo: bool
    n_registros_mes: int = 0
    ultimo_registro: Optional[str] = None


__all__ = [
    "ProduccionRow",
    "EstadoMaquinaRow",
    "CicloRealRow",
    "CapacidadRow",
    "OperarioRow",
]
