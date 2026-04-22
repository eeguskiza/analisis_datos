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

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from nexo.data.dto.nexo import AuditLogRow, DepartmentRow, RoleRow, UserRow
from nexo.data.dto.query import (
    QueryApprovalRow,
    QueryLogRow,
    QueryThresholdRow,
)
from nexo.data.models_nexo import (
    NexoAuditLog,
    NexoDepartment,
    NexoQueryApproval,
    NexoQueryLog,
    NexoQueryThreshold,
    NexoRole,
    NexoUser,
    _utcnow,
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
        return UserRow.model_validate(
            {
                "id": orm.id,
                "email": orm.email,
                "nombre": orm.nombre,
                "role": orm.role,
                "active": orm.active,
                "must_change_password": orm.must_change_password,
                "last_login": orm.last_login,
                "created_at": orm.created_at,
                "departments": tuple(d.code for d in orm.departments),
            }
        )

    def list_all(self) -> list[UserRow]:
        rows = (
            self._db.execute(select(NexoUser).order_by(NexoUser.email)).scalars().all()
        )
        return [
            UserRow.model_validate(
                {
                    "id": u.id,
                    "email": u.email,
                    "nombre": u.nombre,
                    "role": u.role,
                    "active": u.active,
                    "must_change_password": u.must_change_password,
                    "last_login": u.last_login,
                    "created_at": u.created_at,
                    "departments": tuple(d.code for d in u.departments),
                }
            )
            for u in rows
        ]


class RoleRepo:
    """Queries de ``nexo.roles`` + ``nexo.departments`` (catalogos)."""

    def __init__(self, db: Session):
        self._db = db

    def list_all(self) -> list[RoleRow]:
        rows = (
            self._db.execute(select(NexoRole).order_by(NexoRole.code)).scalars().all()
        )
        return [RoleRow.model_validate(r) for r in rows]

    def list_departments(self) -> list[DepartmentRow]:
        rows = (
            self._db.execute(select(NexoDepartment).order_by(NexoDepartment.code))
            .scalars()
            .all()
        )
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
        self._db.add(
            NexoAuditLog(
                user_id=user_id,
                ip=ip,
                method=method,
                path=path,
                status=status,
                details_json=details_json,
            )
        )

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

        stmt = (
            select(func.count())
            .select_from(NexoAuditLog)
            .outerjoin(NexoUser, NexoUser.id == NexoAuditLog.user_id)
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
            AuditLogRow.model_validate(
                {
                    "id": log.id,
                    "ts": log.ts,
                    "user_id": log.user_id,
                    "user_email": user.email if user else None,
                    "ip": log.ip,
                    "method": log.method,
                    "path": log.path,
                    "status": log.status,
                    "details_json": log.details_json,
                }
            )
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
        stmt = stmt.order_by(NexoAuditLog.ts.desc()).execution_options(yield_per=500)
        for log, user in self._db.execute(stmt):
            yield AuditLogRow.model_validate(
                {
                    "id": log.id,
                    "ts": log.ts,
                    "user_id": log.user_id,
                    "user_email": user.email if user else None,
                    "ip": log.ip,
                    "method": log.method,
                    "path": log.path,
                    "status": log.status,
                    "details_json": log.details_json,
                }
            )


class QueryLogRepo:
    """Postflight log repository (Phase 4 — Plan 04-01 / QUERY-01).

    Mismo contrato que ``AuditRepo``: ``append`` NO comitea; el caller
    (middleware ``query_timing`` — Plan 04-02) orquesta la transaccion.

    Contract test (T-03-03-01 equivalente): ``QueryLogRepo.append``
    source debe NO contener ``.commit()`` ni ``.flush()``. Verificado en
    ``tests/data/test_schema_query_log.py::test_query_log_repo_append_source_has_no_commit``.
    """

    def __init__(self, db: Session):
        self._db = db

    def append(
        self,
        *,
        user_id: int | None,
        ip: str | None,
        endpoint: str,
        params_json: str | None,
        estimated_ms: int | None,
        actual_ms: int,
        rows: int | None,
        status: str,
        approval_id: int | None,
    ) -> None:
        """INSERT fila en nexo.query_log. Caller orquesta commit.

        NO comitea aqui. El middleware de query_timing (Plan 04-02) es
        responsable del commit. Contract test verifica la ausencia de
        ``commit()`` / ``flush()`` en el source.
        """
        self._db.add(
            NexoQueryLog(
                user_id=user_id,
                ip=ip,
                endpoint=endpoint,
                params_json=params_json,
                estimated_ms=estimated_ms,
                actual_ms=actual_ms,
                rows=rows,
                status=status,
                approval_id=approval_id,
            )
        )

    def list_filtered(
        self,
        *,
        endpoint: str | None = None,
        user_id: int | None = None,
        status: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        page: int = 1,
        limit: int = 100,
    ) -> list[QueryLogRow]:
        """Query con filtros para ``/ajustes/rendimiento`` (D-11)."""
        stmt = select(NexoQueryLog, NexoUser).outerjoin(
            NexoUser, NexoUser.id == NexoQueryLog.user_id
        )
        if endpoint:
            stmt = stmt.where(NexoQueryLog.endpoint == endpoint)
        if user_id is not None:
            stmt = stmt.where(NexoQueryLog.user_id == user_id)
        if status:
            stmt = stmt.where(NexoQueryLog.status == status)
        if date_from:
            stmt = stmt.where(NexoQueryLog.ts >= date_from)
        if date_to:
            stmt = stmt.where(NexoQueryLog.ts <= date_to)
        stmt = (
            stmt.order_by(NexoQueryLog.ts.desc())
            .limit(limit)
            .offset((page - 1) * limit)
        )
        rows = self._db.execute(stmt).all()
        return [
            QueryLogRow.model_validate(
                {
                    "id": log.id,
                    "ts": log.ts,
                    "user_id": log.user_id,
                    "endpoint": log.endpoint,
                    "params_json": log.params_json,
                    "estimated_ms": log.estimated_ms,
                    "actual_ms": log.actual_ms,
                    "rows": log.rows,
                    "status": log.status,
                    "approval_id": log.approval_id,
                    "ip": log.ip,
                }
            )
            for log, _user in rows
        ]

    def count_filtered(
        self,
        *,
        endpoint: str | None = None,
        user_id: int | None = None,
        status: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> int:
        """Cuenta total de filas que matchean los filtros (paginacion)."""
        stmt = select(func.count()).select_from(NexoQueryLog)
        if endpoint:
            stmt = stmt.where(NexoQueryLog.endpoint == endpoint)
        if user_id is not None:
            stmt = stmt.where(NexoQueryLog.user_id == user_id)
        if status:
            stmt = stmt.where(NexoQueryLog.status == status)
        if date_from:
            stmt = stmt.where(NexoQueryLog.ts >= date_from)
        if date_to:
            stmt = stmt.where(NexoQueryLog.ts <= date_to)
        return self._db.execute(stmt).scalar_one() or 0

    def iter_filtered(
        self,
        *,
        endpoint: str | None = None,
        user_id: int | None = None,
        status: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ):
        """Generator memory-safe para CSV export. ``yield_per=500``."""
        stmt = select(NexoQueryLog, NexoUser).outerjoin(
            NexoUser, NexoUser.id == NexoQueryLog.user_id
        )
        if endpoint:
            stmt = stmt.where(NexoQueryLog.endpoint == endpoint)
        if user_id is not None:
            stmt = stmt.where(NexoQueryLog.user_id == user_id)
        if status:
            stmt = stmt.where(NexoQueryLog.status == status)
        if date_from:
            stmt = stmt.where(NexoQueryLog.ts >= date_from)
        if date_to:
            stmt = stmt.where(NexoQueryLog.ts <= date_to)
        stmt = stmt.order_by(NexoQueryLog.ts.desc()).execution_options(yield_per=500)
        for log, _user in self._db.execute(stmt):
            yield QueryLogRow.model_validate(
                {
                    "id": log.id,
                    "ts": log.ts,
                    "user_id": log.user_id,
                    "endpoint": log.endpoint,
                    "params_json": log.params_json,
                    "estimated_ms": log.estimated_ms,
                    "actual_ms": log.actual_ms,
                    "rows": log.rows,
                    "status": log.status,
                    "approval_id": log.approval_id,
                    "ip": log.ip,
                }
            )

    def timeseries(
        self,
        *,
        endpoint: str,
        date_from: datetime,
        date_to: datetime,
    ) -> tuple[list[dict], str]:
        """Bucketiza ``nexo.query_log`` para la grafica de ``/ajustes/rendimiento`` (D-11).

        Granularity automatica (Plan 04-04 / Pitfall 8):
          - Rango <= 7 dias -> ``date_trunc('hour', ts)``.
          - Rango > 7 dias  -> ``date_trunc('day', ts)``.

        Devuelve ``(points, granularity)`` donde ``points`` es
        ``[{ts, estimated_ms, actual_ms}, ...]`` ordenado ascendente
        por bucket. ``granularity`` es ``'hour'`` o ``'day'``.

        Los buckets sin filas se omiten (no zero-fill) — Chart.js
        tolera gaps en el eje X y evita explotar la serie si el user
        escogio un rango con poca actividad.
        """
        rango = date_to - date_from
        granularity = "hour" if rango.total_seconds() <= 7 * 24 * 3600 else "day"
        sql = text(
            f"SELECT date_trunc('{granularity}', ts) AS bucket, "
            "       AVG(estimated_ms)::integer AS avg_est, "
            "       AVG(actual_ms)::integer AS avg_actual "
            "FROM nexo.query_log "
            "WHERE endpoint = :ep AND ts >= :df AND ts < :dt "
            "GROUP BY bucket "
            "ORDER BY bucket ASC"
        )
        rows = (
            self._db.execute(
                sql,
                {"ep": endpoint, "df": date_from, "dt": date_to},
            )
            .mappings()
            .all()
        )
        points = [
            {
                "ts": r["bucket"].isoformat() if r["bucket"] is not None else None,
                "estimated_ms": r["avg_est"] or 0,
                "actual_ms": r["avg_actual"] or 0,
            }
            for r in rows
        ]
        return points, granularity

    def summary(
        self,
        endpoint: str,
        date_from: datetime,
        date_to: datetime,
    ) -> dict:
        """Agregados para ``/ajustes/rendimiento`` (D-11).

        Devuelve dict con: n_runs, avg_est, avg_actual, p95, n_slow,
        divergence_pct ((avg_actual - avg_est) / avg_est * 100; 0 si
        avg_est is None/0).
        """
        sql = text(
            "SELECT "
            "  COUNT(*) AS n_runs, "
            "  AVG(estimated_ms) AS avg_est, "
            "  AVG(actual_ms) AS avg_actual, "
            "  percentile_cont(0.95) WITHIN GROUP (ORDER BY actual_ms) AS p95, "
            "  SUM(CASE WHEN status = 'slow' THEN 1 ELSE 0 END) AS n_slow "
            "FROM nexo.query_log "
            "WHERE endpoint = :ep AND ts BETWEEN :df AND :dt"
        )
        row = (
            self._db.execute(sql, {"ep": endpoint, "df": date_from, "dt": date_to})
            .mappings()
            .first()
        )
        result = (
            dict(row)
            if row
            else {
                "n_runs": 0,
                "avg_est": None,
                "avg_actual": None,
                "p95": None,
                "n_slow": 0,
            }
        )
        avg_est = result.get("avg_est")
        avg_actual = result.get("avg_actual")
        if avg_est and avg_actual:
            result["divergence_pct"] = (
                (float(avg_actual) - float(avg_est)) / float(avg_est) * 100.0
            )
        else:
            result["divergence_pct"] = 0.0
        return result


class ThresholdRepo:
    """CRUD de ``nexo.query_thresholds`` (Phase 4 — Plan 04-01 / QUERY-02).

    Este repo SI comitea en ``update`` (pattern de
    ``api.routers.usuarios.editar`` — consumer commits inline).
    ``get`` / ``list_all`` son solo lectura, sin commit.
    """

    def __init__(self, db: Session):
        self._db = db

    def get(self, endpoint: str) -> QueryThresholdRow | None:
        """Fetch una fila; None si no existe."""
        row = self._db.execute(
            select(NexoQueryThreshold).where(NexoQueryThreshold.endpoint == endpoint)
        ).scalar_one_or_none()
        if row is None:
            return None
        return QueryThresholdRow.model_validate(row)

    def list_all(self) -> list[QueryThresholdRow]:
        """Full scan (4 filas esperadas por seed inicial)."""
        rows = (
            self._db.execute(
                select(NexoQueryThreshold).order_by(NexoQueryThreshold.endpoint)
            )
            .scalars()
            .all()
        )
        return [QueryThresholdRow.model_validate(r) for r in rows]

    def update(
        self,
        *,
        endpoint: str,
        warn_ms: int,
        block_ms: int,
        factor_ms: float | None = None,
        updated_by: int | None,
        factor_touched: bool = False,
    ) -> None:
        """Mutate existing row; set updated_at = now(). Commit inline.

        Si ``factor_touched=True`` (usuario pulso "Recalcular factor" —
        D-04), tambien setea ``factor_updated_at = now()`` para el cron
        mensual fallback (D-20).
        """
        row = self._db.execute(
            select(NexoQueryThreshold).where(NexoQueryThreshold.endpoint == endpoint)
        ).scalar_one()
        now = _utcnow()
        row.warn_ms = warn_ms
        row.block_ms = block_ms
        if factor_ms is not None:
            row.factor_ms = factor_ms
        row.updated_by = updated_by
        row.updated_at = now
        if factor_touched:
            row.factor_updated_at = now
        self._db.commit()


class ApprovalRepo:
    """CRUD + state machine de ``nexo.query_approvals`` (Phase 4 — Plan 04-01).

    State machine (D-15, D-16):
        pending -> approved | rejected | cancelled | expired
        approved -> consumed  (UNIQUE atomic CAS)

    ``create`` / ``approve`` / ``reject`` / ``cancel`` / ``expire_stale``
    comitean inline. ``consume`` usa UPDATE...RETURNING atomico (T-04-01-03
    mitigation) — no hay analogo ORM puro, text() es el unico path seguro.
    """

    def __init__(self, db: Session):
        self._db = db

    def create(
        self,
        *,
        user_id: int,
        endpoint: str,
        params_json: str,
        estimated_ms: int,
        ttl_days: int = 7,
    ) -> NexoQueryApproval:
        """INSERT fila pending. Devuelve ORM row con id poblado (caller
        puede necesitar .id de vuelta). Commit inline."""
        row = NexoQueryApproval(
            user_id=user_id,
            endpoint=endpoint,
            params_json=params_json,
            estimated_ms=estimated_ms,
            ttl_days=ttl_days,
            status="pending",
        )
        self._db.add(row)
        self._db.commit()
        self._db.refresh(row)
        return row

    def get(self, approval_id: int) -> QueryApprovalRow | None:
        """Fetch single approval as DTO."""
        row = self._db.execute(
            select(NexoQueryApproval).where(NexoQueryApproval.id == approval_id)
        ).scalar_one_or_none()
        if row is None:
            return None
        return QueryApprovalRow.model_validate(row)

    def list_by_user(
        self,
        user_id: int,
        *,
        statuses: list[str] | None = None,
    ) -> list[QueryApprovalRow]:
        """Para ``/mis-solicitudes`` (D-16) — filas del usuario, mas recientes primero."""
        stmt = select(NexoQueryApproval).where(NexoQueryApproval.user_id == user_id)
        if statuses:
            stmt = stmt.where(NexoQueryApproval.status.in_(statuses))
        stmt = stmt.order_by(NexoQueryApproval.created_at.desc())
        rows = self._db.execute(stmt).scalars().all()
        return [QueryApprovalRow.model_validate(r) for r in rows]

    def list_pending(self) -> list[QueryApprovalRow]:
        """Para ``/ajustes/solicitudes`` — solo pending, mas recientes primero."""
        rows = (
            self._db.execute(
                select(NexoQueryApproval)
                .where(NexoQueryApproval.status == "pending")
                .order_by(NexoQueryApproval.created_at.desc())
            )
            .scalars()
            .all()
        )
        return [QueryApprovalRow.model_validate(r) for r in rows]

    def list_recent_non_pending(
        self,
        cutoff: datetime,
        limit: int = 100,
    ) -> list[QueryApprovalRow]:
        """Historico 30d para ``/ajustes/solicitudes`` (D-14).

        Consumido por Plan 04-03 en la tabla "Histórico" que muestra
        approved/rejected/cancelled/expired/consumed.
        """
        rows = (
            self._db.execute(
                select(NexoQueryApproval)
                .where(
                    NexoQueryApproval.status != "pending",
                    NexoQueryApproval.created_at >= cutoff,
                )
                .order_by(NexoQueryApproval.created_at.desc())
                .limit(limit)
            )
            .scalars()
            .all()
        )
        return [QueryApprovalRow.model_validate(r) for r in rows]

    def count_pending(self) -> int:
        """Badge sidebar del propietario (D-13)."""
        return (
            self._db.execute(
                select(func.count())
                .select_from(NexoQueryApproval)
                .where(NexoQueryApproval.status == "pending")
            ).scalar_one()
            or 0
        )

    def approve(self, approval_id: int, approved_by: int) -> None:
        """pending -> approved. Commit inline."""
        row = self._db.execute(
            select(NexoQueryApproval).where(NexoQueryApproval.id == approval_id)
        ).scalar_one()
        row.status = "approved"
        row.approved_at = _utcnow()
        row.approved_by = approved_by
        self._db.commit()

    def reject(self, approval_id: int, decided_by: int) -> None:
        """pending -> rejected. Commit inline.

        ``decided_by`` se almacena en ``approved_by`` column (semantica
        "quien tomo la decision"; la columna es simetrica para ambos
        outcomes — evita duplicar schema).
        """
        row = self._db.execute(
            select(NexoQueryApproval).where(NexoQueryApproval.id == approval_id)
        ).scalar_one()
        row.status = "rejected"
        row.rejected_at = _utcnow()
        row.approved_by = decided_by
        self._db.commit()

    def cancel(self, approval_id: int, user_id: int) -> bool:
        """pending -> cancelled (D-16).

        Ownership check: solo el user que creo la solicitud puede
        cancelar. Devuelve False si mismatch o status != pending (no
        lanza — el router devuelve 403/409 segun corresponda).
        """
        row = self._db.execute(
            select(NexoQueryApproval).where(NexoQueryApproval.id == approval_id)
        ).scalar_one_or_none()
        if row is None:
            return False
        if row.user_id != user_id:
            return False
        if row.status != "pending":
            return False
        row.status = "cancelled"
        row.cancelled_at = _utcnow()
        self._db.commit()
        return True

    def consume(
        self,
        *,
        approval_id: int,
        user_id: int,
        current_params_json: str,
    ) -> QueryApprovalRow | None:
        """Single-use CAS (D-15 / T-04-01-03 mitigation).

        Unica escritura de toda Phase 4 que usa ``text()`` porque
        SQLAlchemy ORM no expone UPDATE...RETURNING atomico. La
        atomicidad esta garantizada por Postgres (row lock durante el
        UPDATE).

        Verifica en una sola operacion:
          1. id = :id
          2. user_id = :uid (ownership)
          3. status = 'approved'
          4. consumed_at IS NULL (no consumed todavia)
          5. params_json = :pj (mismos parametros que al aprobar)

        Si cualquier check falla → 0 rows affected → devolvemos None.
        Si matchea → UPDATE atomico marca consumed + devuelve row.
        """
        sql = text(
            "UPDATE nexo.query_approvals "
            "SET consumed_at = (now() AT TIME ZONE 'UTC'), "
            "    status = 'consumed' "
            "WHERE id = :id "
            "  AND user_id = :uid "
            "  AND status = 'approved' "
            "  AND consumed_at IS NULL "
            "  AND params_json = :pj "
            "RETURNING id, endpoint, params_json, estimated_ms, user_id, "
            "          created_at, ttl_days, approved_at, approved_by, "
            "          rejected_at, cancelled_at, expired_at, consumed_at, "
            "          consumed_run_id, status"
        )
        result = (
            self._db.execute(
                sql,
                {
                    "id": approval_id,
                    "uid": user_id,
                    "pj": current_params_json,
                },
            )
            .mappings()
            .first()
        )
        if result is None:
            return None
        self._db.commit()
        return QueryApprovalRow.model_validate(dict(result))

    def expire_stale(self, cutoff: datetime) -> int:
        """pending -> expired para filas con created_at < cutoff (D-14).

        Devuelve numero de filas marcadas expired. Commit inline.
        Consumido por el job semanal de Plan 04-04.
        """
        sql = text(
            "UPDATE nexo.query_approvals "
            "SET status = 'expired', "
            "    expired_at = (now() AT TIME ZONE 'UTC') "
            "WHERE status = 'pending' "
            "  AND created_at < :cutoff"
        )
        result = self._db.execute(sql, {"cutoff": cutoff})
        self._db.commit()
        return result.rowcount or 0


__all__ = [
    "UserRepo",
    "RoleRepo",
    "AuditRepo",
    "QueryLogRepo",
    "ThresholdRepo",
    "ApprovalRepo",
]
