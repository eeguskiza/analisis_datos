"""Pydantic models para requests y responses de la API."""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel


# ── Conexión ──────────────────────────────────────────────────────────────────


class ConnectionStatus(BaseModel):
    ok: bool
    mensaje: str
    server: str = ""
    database: str = ""


# ── Recursos ──────────────────────────────────────────────────────────────────


class Recurso(BaseModel):
    centro_trabajo: int
    nombre: str
    seccion: str = "GENERAL"
    activo: bool = True


class RecursosPayload(BaseModel):
    recursos: list[Recurso]


# ── Pipeline ──────────────────────────────────────────────────────────────────


class PipelineRequest(BaseModel):
    fecha_inicio: date
    fecha_fin: date
    modulos: list[str] = ["disponibilidad", "rendimiento", "calidad", "oee_secciones"]
    source: str = "db"  # "db", "excel", "csv_only"
    recursos: Optional[list[str]] = (
        None  # nombres de recursos a procesar (None = todos)
    )
    # Phase 4 / Plan 04-01 (QUERY-06, D-15): preflight bypass con approval
    # previamente concedido. ``force=True`` requiere ``approval_id`` valido
    # (user_id match, status='approved', consumed_at IS NULL, params_json
    # match). Backwards-compatible: defaults a False / None.
    force: bool = False
    approval_id: Optional[int] = None


class PipelineResult(BaseModel):
    ok: bool
    log: list[str] = []
    pdfs: list[str] = []


# ── Informes ──────────────────────────────────────────────────────────────────


class InformeEntry(BaseModel):
    name: str
    path: str
    type: str  # "dir" | "pdf"
    children: list[InformeEntry] = []


# ── Ciclos ────────────────────────────────────────────────────────────────────


class CicloRow(BaseModel):
    maquina: str
    referencia: str
    tiempo_ciclo: float


class CiclosPayload(BaseModel):
    rows: list[CicloRow]
