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
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
    text,
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


# -- nexo.query_approvals (Phase 4 — Plan 04-01 / QUERY-06) -----------------
# State machine: pending -> approved | rejected | cancelled | expired
#                approved -> consumed (single-use CAS — D-15)
# Declared BEFORE NexoQueryLog because NexoQueryLog.approval_id is a FK to
# query_approvals.id. SQLAlchemy accepts string refs so order is technically
# irrelevant, but declaration order matches dependency direction.

class NexoQueryApproval(NexoBase):
    __tablename__ = "query_approvals"
    __table_args__ = (
        Index("ix_approvals_status", "status"),
        Index("ix_approvals_user_status", "user_id", "status"),
        Index("ix_approvals_created_at", "created_at"),
        {"schema": NEXO_SCHEMA},
    )

    id = Column(Integer, primary_key=True)
    user_id = Column(
        Integer,
        ForeignKey("nexo.users.id", ondelete="CASCADE"),
        nullable=False,
    )
    endpoint = Column(String(100), nullable=False)
    params_json = Column(Text, nullable=False)
    estimated_ms = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    ttl_days = Column(Integer, nullable=False, default=7)
    approved_at = Column(DateTime(timezone=True), nullable=True)
    approved_by = Column(
        Integer,
        ForeignKey("nexo.users.id", ondelete="SET NULL"),
        nullable=True,
    )
    rejected_at = Column(DateTime(timezone=True), nullable=True)
    cancelled_at = Column(DateTime(timezone=True), nullable=True)
    expired_at = Column(DateTime(timezone=True), nullable=True)
    consumed_at = Column(DateTime(timezone=True), nullable=True)
    # Soft FK a query_log.id — NO declarado como ForeignKey para evitar
    # dependencia circular con NexoQueryLog (que tiene FK a aqui).
    consumed_run_id = Column(Integer, nullable=True)


# -- nexo.query_thresholds (Phase 4 — Plan 04-01 / QUERY-02) ----------------
# Lookup table; 1 fila por endpoint con preflight. PK en endpoint — sin
# indices adicionales (4 filas esperadas por seed D-01/D-02/D-03/D-04).

class NexoQueryThreshold(NexoBase):
    __tablename__ = "query_thresholds"
    __table_args__ = {"schema": NEXO_SCHEMA}

    endpoint = Column(String(100), primary_key=True)
    warn_ms = Column(Integer, nullable=False)
    block_ms = Column(Integer, nullable=False)
    factor_ms = Column(Float, nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_by = Column(
        Integer,
        ForeignKey("nexo.users.id", ondelete="SET NULL"),
        nullable=True,
    )
    factor_updated_at = Column(DateTime(timezone=True), nullable=True)


# -- nexo.query_log (Phase 4 — Plan 04-01 / QUERY-01) -----------------------
# Append-only postflight log (mismo patron que audit_log). 4 indices:
#   ix_query_log_ts            - purge por retencion 90d (D-10)
#   ix_query_log_endpoint_ts   - dashboards /ajustes/rendimiento (D-11)
#   ix_query_log_user_ts       - filtros por usuario
#   ix_query_log_status_slow   - partial index para WARN tracking (D-17)

class NexoQueryLog(NexoBase):
    __tablename__ = "query_log"
    __table_args__ = (
        Index("ix_query_log_ts", "ts"),
        Index("ix_query_log_endpoint_ts", "endpoint", "ts"),
        Index("ix_query_log_user_ts", "user_id", "ts"),
        Index(
            "ix_query_log_status_slow",
            "ts",
            postgresql_where=text("status = 'slow'"),
        ),
        {"schema": NEXO_SCHEMA},
    )

    id = Column(Integer, primary_key=True)
    ts = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    user_id = Column(
        Integer,
        ForeignKey("nexo.users.id", ondelete="SET NULL"),
        nullable=True,
    )
    endpoint = Column(String(100), nullable=False)
    params_json = Column(Text, nullable=True)
    estimated_ms = Column(Integer, nullable=True)
    actual_ms = Column(Integer, nullable=False)
    rows = Column(Integer, nullable=True)
    status = Column(String(20), nullable=False)
    approval_id = Column(
        Integer,
        ForeignKey("nexo.query_approvals.id", ondelete="SET NULL"),
        nullable=True,
    )
    ip = Column(String(64), nullable=True)


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
    "NexoQueryApproval",
    "NexoQueryThreshold",
    "NexoQueryLog",
    "user_departments",
]
