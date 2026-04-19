"""Repositorios NEXO (schema ``nexo`` en Postgres) - Plan 03-03 (DATA-04).

Session inyectada; caller orquesta transacciones (los repos NO
comitean). Consumen el ORM de ``nexo.data.models_nexo`` y devuelven
DTOs frozen de ``nexo.data.dto.nexo`` (con la excepcion de
``UserRepo.get_by_email_orm`` que devuelve ORM para ``nexo.services.auth``
porque necesita el ``NexoUser`` completo para session management +
password hashing - decision D-02).

DATA-05 scope (per CONTEXT.md D-01 clarificacion 2026-04-19): ORM puro,
no archivos ``.sql``. El modelo SQLAlchemy declarativo es la
representacion canonica.

IDENT-06 compat (AUTH_MODEL.md §audit_log): ``AuditRepo.append`` SOLO
hace INSERT. El commit es responsabilidad del caller (el middleware de
audit). A nivel DB, el rol ``nexo_app`` solo tiene
INSERT/SELECT sobre ``nexo.audit_log`` - defensa en profundidad sobre
el contrato del repo.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from nexo.data.dto.nexo import AuditLogRow, DepartmentRow, RoleRow, UserRow
from nexo.data.models_nexo import (
    NexoAuditLog,
    NexoDepartment,
    NexoRole,
    NexoUser,
)


class UserRepo:
    """Queries de ``nexo.users``.

    Expone dos variantes de ``get_by_email``:
    - ``get_by_email_orm``: devuelve ``NexoUser`` ORM. Uso interno de
      ``nexo.services.auth`` (necesita el modelo completo para session
      management + password hashing).
    - ``get_by_email``: devuelve ``UserRow`` DTO. Uso desde routers /
      panel de audit.
    """

    def __init__(self, db: Session):
        self._db = db

    def get_by_email_orm(self, email: str) -> Optional[NexoUser]:
        """Devuelve ORM ``NexoUser`` (solo activos). Uso interno auth."""
        return self._db.execute(
            select(NexoUser).where(
                NexoUser.email == email,
                NexoUser.active.is_(True),
            )
        ).scalar_one_or_none()

    def get_by_email(self, email: str) -> UserRow | None:
        """Devuelve DTO ``UserRow``. Uso desde routers / panel de audit."""
        orm = self.get_by_email_orm(email)
        if not orm:
            return None
        return UserRow.model_validate({
            "id": orm.id,
            "email": orm.email,
            "role": orm.role,
            "active": orm.active,
            "must_change_password": orm.must_change_password,
            "last_login": orm.last_login,
            "created_at": orm.created_at,
            "departments": tuple(d.code for d in orm.departments),
        })

    def list_all(self) -> list[UserRow]:
        rows = self._db.execute(
            select(NexoUser).order_by(NexoUser.email)
        ).scalars().all()
        return [
            UserRow.model_validate({
                "id": u.id,
                "email": u.email,
                "role": u.role,
                "active": u.active,
                "must_change_password": u.must_change_password,
                "last_login": u.last_login,
                "created_at": u.created_at,
                "departments": tuple(d.code for d in u.departments),
            })
            for u in rows
        ]


class RoleRepo:
    """Queries de ``nexo.roles`` + ``nexo.departments`` (catalogos)."""

    def __init__(self, db: Session):
        self._db = db

    def list_all(self) -> list[RoleRow]:
        rows = self._db.execute(
            select(NexoRole).order_by(NexoRole.code)
        ).scalars().all()
        return [RoleRow.model_validate(r) for r in rows]

    def list_departments(self) -> list[DepartmentRow]:
        rows = self._db.execute(
            select(NexoDepartment).order_by(NexoDepartment.code)
        ).scalars().all()
        return [DepartmentRow.model_validate(d) for d in rows]


class AuditRepo:
    """Audit log repository. IDENT-06 compat: SOLO INSERT/SELECT desde
    este repo; DB tambien bloquea UPDATE/DELETE al rol ``nexo_app``
    (defense-in-depth).
    """

    def __init__(self, db: Session):
        self._db = db

    def append(
        self,
        *,
        user_id: int | None,
        ip: str,
        method: str,
        path: str,
        status: int,
        details_json: str | None,
    ) -> None:
        """INSERT fila en nexo.audit_log. Caller orquesta commit.

        NO comitea aqui. El middleware de audit (o quien llame a este
        metodo) es responsable del commit. Contract test en
        ``tests/data/test_nexo_repository.py`` verifica ausencia de
        ``commit()`` / ``flush()`` en el source.
        """
        self._db.add(NexoAuditLog(
            user_id=user_id,
            ip=ip,
            method=method,
            path=path,
            status=status,
            details_json=details_json,
        ))

    def count_filtered(
        self,
        *,
        user_email: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        path: str | None = None,
        status: int | None = None,
    ) -> int:
        """Cuenta total de filas que matchean los filtros."""
        from sqlalchemy import func
        stmt = select(func.count()).select_from(NexoAuditLog).outerjoin(
            NexoUser, NexoUser.id == NexoAuditLog.user_id
        )
        if user_email:
            stmt = stmt.where(NexoUser.email == user_email)
        if date_from:
            stmt = stmt.where(NexoAuditLog.ts >= date_from)
        if date_to:
            stmt = stmt.where(NexoAuditLog.ts <= date_to)
        if path:
            stmt = stmt.where(NexoAuditLog.path.ilike(f"%{path}%"))
        if status is not None:
            stmt = stmt.where(NexoAuditLog.status == status)
        return self._db.execute(stmt).scalar_one() or 0

    def list_filtered(
        self,
        *,
        user_email: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        path: str | None = None,
        status: int | None = None,
        page: int = 1,
        limit: int = 100,
    ) -> list[AuditLogRow]:
        """Query con filtros para panel /ajustes/auditoria."""
        stmt = select(NexoAuditLog, NexoUser).outerjoin(
            NexoUser, NexoUser.id == NexoAuditLog.user_id
        )
        if user_email:
            stmt = stmt.where(NexoUser.email == user_email)
        if date_from:
            stmt = stmt.where(NexoAuditLog.ts >= date_from)
        if date_to:
            stmt = stmt.where(NexoAuditLog.ts <= date_to)
        if path:
            stmt = stmt.where(NexoAuditLog.path.ilike(f"%{path}%"))
        if status is not None:
            stmt = stmt.where(NexoAuditLog.status == status)
        stmt = (
            stmt.order_by(NexoAuditLog.ts.desc())
            .limit(limit)
            .offset((page - 1) * limit)
        )

        rows = self._db.execute(stmt).all()
        return [
            AuditLogRow.model_validate({
                "id": log.id,
                "ts": log.ts,
                "user_id": log.user_id,
                "user_email": user.email if user else None,
                "ip": log.ip,
                "method": log.method,
                "path": log.path,
                "status": log.status,
                "details_json": log.details_json,
            })
            for log, user in rows
        ]

    def iter_filtered(
        self,
        *,
        user_email: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        path: str | None = None,
        status: int | None = None,
    ):
        """Generator que itera sobre TODAS las filas filtradas sin paginar
        (CSV export). Memory-safe via yield_per=500.
        """
        stmt = select(NexoAuditLog, NexoUser).outerjoin(
            NexoUser, NexoUser.id == NexoAuditLog.user_id
        )
        if user_email:
            stmt = stmt.where(NexoUser.email == user_email)
        if date_from:
            stmt = stmt.where(NexoAuditLog.ts >= date_from)
        if date_to:
            stmt = stmt.where(NexoAuditLog.ts <= date_to)
        if path:
            stmt = stmt.where(NexoAuditLog.path.ilike(f"%{path}%"))
        if status is not None:
            stmt = stmt.where(NexoAuditLog.status == status)
        stmt = stmt.order_by(NexoAuditLog.ts.desc()).execution_options(
            yield_per=500
        )
        for log, user in self._db.execute(stmt):
            yield AuditLogRow.model_validate({
                "id": log.id,
                "ts": log.ts,
                "user_id": log.user_id,
                "user_email": user.email if user else None,
                "ip": log.ip,
                "method": log.method,
                "path": log.path,
                "status": log.status,
                "details_json": log.details_json,
            })


__all__ = [
    "UserRepo",
    "RoleRepo",
    "AuditRepo",
]
