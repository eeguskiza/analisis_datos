"""DTOs frozen NEXO schema (Pydantic v2) - Plan 03-03 (DATA-08).

Los DTOs aqui cruzan la frontera router<->repo para el schema ``nexo``
en Postgres. El ORM (``nexo.data.models_nexo``) es uso interno de los
repos + ``nexo.services.auth``.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class UserRow(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    id: int
    email: str
    role: str
    active: bool = True
    must_change_password: bool = True
    last_login: Optional[datetime] = None
    created_at: Optional[datetime] = None
    departments: tuple[str, ...] = ()  # codes; tuple para frozen


class RoleRow(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    id: int
    code: str
    name: str


class DepartmentRow(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    id: int
    code: str
    name: str


class AuditLogRow(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    id: int
    ts: datetime
    user_id: Optional[int] = None
    user_email: Optional[str] = None  # join con NexoUser si se resolvio
    ip: str
    method: str
    path: str
    status: int
    details_json: Optional[str] = None


__all__ = [
    "UserRow",
    "RoleRow",
    "DepartmentRow",
    "AuditLogRow",
]
