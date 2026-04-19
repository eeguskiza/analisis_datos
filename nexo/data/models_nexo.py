"""Modelos ORM del schema ``nexo`` en Postgres (Plan 03-03 — DATA-04).

Migrado desde ``nexo/db/models.py`` en Plan 03-03. El modulo
``nexo/db/models.py`` se mantiene como shim transicional que
re-exporta estos simbolos para no romper consumidores Phase 2
(``nexo/services/auth``, ``api/routers/auditoria``,
``api/routers/usuarios``, ``tests/auth/*``). Eliminacion del shim
diferida a Mark-IV cuando todos los consumers migren sus imports.

Las 8 tablas del modelo de auth/audit acordado en ``docs/AUTH_MODEL.md``:

- ``nexo.users``               usuarios
- ``nexo.roles``               catalogo de roles (propietario/directivo/usuario)
- ``nexo.departments``         catalogo de departamentos (rrhh/comercial/...)
- ``nexo.user_departments``    N:M users-departments
- ``nexo.permissions``         catalogo seed del PERMISSION_MAP (fuente de verdad: dict en codigo)
- ``nexo.sessions``            sesiones activas firmadas con ``itsdangerous``
- ``nexo.login_attempts``      contador para bloqueo progresivo 5-15 min
- ``nexo.audit_log``           append-only; cada request autenticada genera fila

Todas las columnas temporales usan ``DateTime(timezone=True)`` y
``datetime.now(timezone.utc)`` como default (research #Pitfall 3 —
``datetime.utcnow()`` esta deprecated en Python 3.12).
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


NEXO_SCHEMA = "nexo"


class NexoBase(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# -- N:M users <-> departments ------------------------------------------------

user_departments = Table(
    "user_departments",
    NexoBase.metadata,
    Column("user_id", ForeignKey("nexo.users.id", ondelete="CASCADE"), primary_key=True),
    Column("department_id", ForeignKey("nexo.departments.id", ondelete="CASCADE"), primary_key=True),
    schema=NEXO_SCHEMA,
)


# -- nexo.users ---------------------------------------------------------------

class NexoUser(NexoBase):
    __tablename__ = "users"
    __table_args__ = {"schema": NEXO_SCHEMA}

    id = Column(Integer, primary_key=True)
    email = Column(String(200), nullable=False, unique=True, index=True)
    password_hash = Column(String(200), nullable=False)
    role = Column(String(20), nullable=False)  # propietario | directivo | usuario
    active = Column(Boolean, nullable=False, default=True)
    must_change_password = Column(Boolean, nullable=False, default=True)
    last_login = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    departments = relationship(
        "NexoDepartment",
        secondary=user_departments,
        back_populates="users",
    )
    sessions = relationship(
        "NexoSession",
        back_populates="user",
        cascade="all, delete-orphan",
    )


# -- nexo.roles (catalogo) ----------------------------------------------------

class NexoRole(NexoBase):
    __tablename__ = "roles"
    __table_args__ = {"schema": NEXO_SCHEMA}

    id = Column(Integer, primary_key=True)
    code = Column(String(20), nullable=False, unique=True)  # propietario/directivo/usuario
    name = Column(String(100), nullable=False)


# -- nexo.departments (catalogo) ----------------------------------------------

class NexoDepartment(NexoBase):
    __tablename__ = "departments"
    __table_args__ = {"schema": NEXO_SCHEMA}

    id = Column(Integer, primary_key=True)
    code = Column(String(20), nullable=False, unique=True)  # rrhh/comercial/ingenieria/produccion/gerencia
    name = Column(String(100), nullable=False)

    users = relationship(
        "NexoUser",
        secondary=user_departments,
        back_populates="departments",
    )


# -- nexo.permissions (catalogo seed; fuente de verdad = PERMISSION_MAP en codigo) --

class NexoPermission(NexoBase):
    __tablename__ = "permissions"
    __table_args__ = (
        UniqueConstraint("module", "action", "role_code", "department_code", name="uq_permission_tuple"),
        {"schema": NEXO_SCHEMA},
    )

    id = Column(Integer, primary_key=True)
    module = Column(String(50), nullable=False)     # pipeline, bbdd, operarios, ...
    action = Column(String(50), nullable=False)     # read, execute, delete, ...
    role_code = Column(String(20), nullable=False)  # propietario/directivo/usuario
    department_code = Column(String(20), nullable=True)  # null = aplica a todos (propietario)


# -- nexo.sessions ------------------------------------------------------------

class NexoSession(NexoBase):
    __tablename__ = "sessions"
    __table_args__ = {"schema": NEXO_SCHEMA}

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("nexo.users.id", ondelete="CASCADE"), nullable=False)
    token = Column(String(100), nullable=False, unique=True, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    user = relationship("NexoUser", back_populates="sessions")


# -- nexo.login_attempts ------------------------------------------------------

class NexoLoginAttempt(NexoBase):
    __tablename__ = "login_attempts"
    __table_args__ = (
        Index("ix_login_attempts_email_ip", "email", "ip"),
        {"schema": NEXO_SCHEMA},
    )

    id = Column(Integer, primary_key=True)
    email = Column(String(200), nullable=False, index=True)
    ip = Column(String(64), nullable=False)
    failed_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, index=True)


# -- nexo.audit_log (append-only a nivel BBDD) --------------------------------

class NexoAuditLog(NexoBase):
    __tablename__ = "audit_log"
    __table_args__ = {"schema": NEXO_SCHEMA}

    id = Column(Integer, primary_key=True)
    ts = Column(DateTime(timezone=True), nullable=False, default=_utcnow, index=True)
    user_id = Column(Integer, ForeignKey("nexo.users.id", ondelete="SET NULL"), nullable=True)
    ip = Column(String(64), nullable=False)
    method = Column(String(10), nullable=False)
    path = Column(String(500), nullable=False)
    status = Column(Integer, nullable=False)
    details_json = Column(Text, nullable=True)  # whitelist-sanitized, ver 02-04


__all__ = [
    "NEXO_SCHEMA",
    "NexoBase",
    "NexoUser",
    "NexoRole",
    "NexoDepartment",
    "NexoPermission",
    "NexoSession",
    "NexoLoginAttempt",
    "NexoAuditLog",
    "user_departments",
]
