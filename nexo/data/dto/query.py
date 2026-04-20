"""DTOs frozen para Phase 4 — query_log, query_thresholds, query_approvals
+ Estimation response DTO (Plan 04-01).

Los ``*Row`` cruzan la frontera router<->repo para las 3 tablas nuevas
del schema ``nexo``. ``Estimation`` es un response DTO puro (no ORM
backing row) usado por el preflight service que aterriza en Plan 04-02.

Decisiones de CONTEXT.md reflejadas:

- D-09 ``params_json`` contiene el SQL completo + params o fecha_desde
  / fecha_hasta / recursos (dict arbitrario serializado).
- D-14 ``ttl_days`` default 7 — TTL de approvals pendientes.
- D-15 ``consumed_at`` + ``consumed_run_id`` para CAS single-use.
- D-16 ``cancelled_at`` para cancelacion por usuario.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from nexo.data.dto.base import ROW_CONFIG


class QueryLogRow(BaseModel):
    """Fila de ``nexo.query_log`` — postflight log append-only."""

    model_config = ROW_CONFIG

    id: int
    ts: datetime
    user_id: Optional[int] = None
    endpoint: str
    params_json: Optional[str] = None
    estimated_ms: Optional[int] = None
    actual_ms: int
    rows: Optional[int] = None
    status: str  # ok | slow | error | timeout | approved_run
    approval_id: Optional[int] = None
    ip: Optional[str] = None


class QueryThresholdRow(BaseModel):
    """Fila de ``nexo.query_thresholds`` — 1 por endpoint con preflight."""

    model_config = ROW_CONFIG

    endpoint: str
    warn_ms: int
    block_ms: int
    factor_ms: Optional[float] = None
    updated_at: datetime
    updated_by: Optional[int] = None
    factor_updated_at: Optional[datetime] = None


class QueryApprovalRow(BaseModel):
    """Fila de ``nexo.query_approvals`` — state machine single-use."""

    model_config = ROW_CONFIG

    id: int
    user_id: int
    endpoint: str
    params_json: str
    estimated_ms: int
    status: str  # pending | approved | rejected | cancelled | expired | consumed
    created_at: datetime
    ttl_days: int
    approved_at: Optional[datetime] = None
    approved_by: Optional[int] = None
    rejected_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None
    consumed_at: Optional[datetime] = None
    consumed_run_id: Optional[int] = None


class Estimation(BaseModel):
    """Response DTO del preflight service.

    No es un Row (no hay ORM backing). ``frozen=True`` mantiene la
    invariante de inmutabilidad del resto de DTOs. No ``from_attributes``
    porque nunca se construye desde una entidad ORM.
    """

    model_config = ConfigDict(frozen=True)

    endpoint: str
    estimated_ms: int
    level: Literal["green", "amber", "red"]
    reason: str
    breakdown: str
    factor_used_ms: Optional[float] = None
    warn_ms: int
    block_ms: int


__all__ = [
    "QueryLogRow",
    "QueryThresholdRow",
    "QueryApprovalRow",
    "Estimation",
]
