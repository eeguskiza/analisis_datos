"""DTOs frozen APP (schema ``ecs_mobility``) - Plan 03-03 (DATA-08).

Pydantic v2, ``frozen=True`` + ``from_attributes=True``. Los DTOs aqui
cruzan la frontera router<->repo. El ORM (``nexo.data.models_app``)
vive detras del repo; el router consume solo DTOs.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class RecursoRow(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    id: int
    centro_trabajo: int
    nombre: str
    seccion: str = "GENERAL"
    activo: bool = True


class CicloRow(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    id: int
    maquina: str
    referencia: str
    tiempo_ciclo: float


class EjecucionRow(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    id: int
    fecha_inicio: str
    fecha_fin: str
    source: str
    status: str
    modulos: Optional[str] = ""
    log: Optional[str] = ""
    n_pdfs: int = 0
    created_at: Optional[datetime] = None


class MetricaRow(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    id: int
    ejecucion_id: int
    seccion: Optional[str] = None
    recurso: Optional[str] = None
    fecha: Optional[date] = None
    turno: Optional[str] = None
    horas_brutas: float = 0.0
    horas_disponible: float = 0.0
    horas_operativo: float = 0.0
    disponibilidad_pct: float = 0.0
    rendimiento_pct: float = 0.0
    calidad_pct: float = 0.0
    oee_pct: float = 0.0
    piezas_totales: int = 0
    piezas_malas: int = 0
    buenas_finales: int = 0


class ReferenciaRow(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    id: int
    ejecucion_id: int
    recurso: Optional[str] = None
    referencia: Optional[str] = None
    ciclo_ideal: Optional[float] = None
    ciclo_real: Optional[float] = None
    cantidad: int = 0
    horas: float = 0.0


class IncidenciaRow(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    id: int
    ejecucion_id: int
    recurso: Optional[str] = None
    nombre: Optional[str] = None
    tipo: Optional[str] = None
    horas: float = 0.0


class InformeRow(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    id: int
    ejecucion_id: Optional[int] = None
    fecha: str
    seccion: str
    maquina: str = ""
    modulo: str = ""
    pdf_path: str
    created_at: Optional[datetime] = None


class ContactoRow(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    id: int
    nombre: str
    email: str


class LukRow(BaseModel):
    """Snapshot de luk4.estado - shape consumida por api/routers/luk4.py."""
    model_config = ConfigDict(frozen=True, from_attributes=True)
    estado_global: int = 0
    codigo_error: int = 0
    pct_auto: Optional[float] = None
    pct_error: Optional[float] = None
    ultimo_ts: Optional[datetime] = None
    alarma_componente: Optional[str] = None
    alarma_mensaje: Optional[str] = None


__all__ = [
    "RecursoRow",
    "CicloRow",
    "EjecucionRow",
    "MetricaRow",
    "ReferenciaRow",
    "IncidenciaRow",
    "InformeRow",
    "ContactoRow",
    "LukRow",
]
