"""Repositorios APP (schema ``ecs_mobility``) - Plan 03-03 (DATA-03).

Session inyectada via FastAPI ``Depends``; caller orquesta transacciones
(los repos NO comitean). Consumen el ORM de ``nexo.data.models_app`` y
devuelven DTOs frozen de ``nexo.data.dto.app``.

DATA-05 scope (per CONTEXT.md D-01 clarificacion 2026-04-19): los
metodos aqui usan ORM puro (``select()``, ``session.query(...)``); NO
requieren archivos ``.sql`` versionados. El modelo declarativo
SQLAlchemy ES la representacion canonica de la query. Solo metodos que
usen ``text()`` con SQL hardcoded requeririan
``nexo/data/sql/app/<method>.sql`` - no es el caso de este plan.
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from nexo.data.dto.app import (
    CicloRow,
    ContactoRow,
    EjecucionRow,
    MetricaRow,
    RecursoRow,
)
from nexo.data.models_app import (
    Ciclo,
    Contacto,
    Ejecucion,
    MetricaOEE,
    Recurso,
)


class RecursoRepo:
    """CRUD + queries de ``cfg.recursos``."""

    def __init__(self, db: Session):
        self._db = db

    def list_all(self) -> list[RecursoRow]:
        rows = (
            self._db.execute(select(Recurso).order_by(Recurso.seccion, Recurso.nombre))
            .scalars()
            .all()
        )
        return [RecursoRow.model_validate(r) for r in rows]

    def list_activos(self) -> list[RecursoRow]:
        # SQL Server no acepta ``IS TRUE`` (produce ``IS 1`` -> syntax
        # error). Usar comparacion directa ``activo == True`` que
        # SQLAlchemy renderiza como ``activo = 1``.
        rows = (
            self._db.execute(
                select(Recurso)
                .where(Recurso.activo == True)  # noqa: E712
                .order_by(Recurso.seccion, Recurso.nombre)
            )
            .scalars()
            .all()
        )
        return [RecursoRow.model_validate(r) for r in rows]

    def get_by_nombre(self, nombre: str) -> RecursoRow | None:
        row = self._db.execute(
            select(Recurso).where(Recurso.nombre == nombre)
        ).scalar_one_or_none()
        return RecursoRow.model_validate(row) if row else None

    def get_orm_by_nombre(self, nombre: str) -> Recurso | None:
        """Devuelve ORM (uso interno para routers que necesitan mutar)."""
        return self._db.execute(
            select(Recurso).where(Recurso.nombre == nombre)
        ).scalar_one_or_none()

    def get_orm_by_codigo(self, centro_trabajo: int) -> Recurso | None:
        """Devuelve ORM (uso interno para routers que escriben)."""
        return self._db.execute(
            select(Recurso).where(Recurso.centro_trabajo == centro_trabajo)
        ).scalar_one_or_none()

    def list_all_orm(self) -> list[Recurso]:
        """Devuelve ORM completos. Usar solo cuando el router necesita
        mutar o iterar con relaciones (sync a config)."""
        return list(
            self._db.execute(select(Recurso).order_by(Recurso.seccion, Recurso.nombre))
            .scalars()
            .all()
        )

    def replace_all(self, recursos: list[dict]) -> None:
        """Borra todo + añade los nuevos. Caller debe comitear."""
        self._db.query(Recurso).delete()
        for r in recursos:
            self._db.add(
                Recurso(
                    centro_trabajo=r["centro_trabajo"],
                    nombre=r["nombre"],
                    seccion=r.get("seccion", "GENERAL"),
                    activo=r.get("activo", True),
                )
            )

    def add(
        self,
        *,
        centro_trabajo: int,
        nombre: str,
        seccion: str = "GENERAL",
        activo: bool = True,
    ) -> None:
        self._db.add(
            Recurso(
                centro_trabajo=centro_trabajo,
                nombre=nombre,
                seccion=seccion,
                activo=activo,
            )
        )


class CicloRepo:
    """CRUD de ``cfg.ciclos`` (tiempos de ciclo ideales)."""

    def __init__(self, db: Session):
        self._db = db

    def list_all(self) -> list[CicloRow]:
        rows = (
            self._db.execute(select(Ciclo).order_by(Ciclo.maquina, Ciclo.referencia))
            .scalars()
            .all()
        )
        return [CicloRow.model_validate(r) for r in rows]

    def list_all_orm(self) -> list[Ciclo]:
        return list(
            self._db.execute(select(Ciclo).order_by(Ciclo.maquina, Ciclo.referencia))
            .scalars()
            .all()
        )

    def exists(self, maquina: str, referencia: str) -> bool:
        return (
            self._db.execute(
                select(Ciclo).where(
                    Ciclo.maquina == maquina,
                    Ciclo.referencia == referencia,
                )
            ).scalar_one_or_none()
            is not None
        )

    def get_by_id(self, ciclo_id: int) -> Ciclo | None:
        return self._db.get(Ciclo, ciclo_id)

    def replace_all(self, rows: list[dict]) -> None:
        self._db.query(Ciclo).delete()
        for r in rows:
            self._db.add(
                Ciclo(
                    maquina=r["maquina"],
                    referencia=r["referencia"],
                    tiempo_ciclo=float(r.get("tiempo_ciclo", 0.0)),
                )
            )

    def add(self, *, maquina: str, referencia: str, tiempo_ciclo: float) -> None:
        self._db.add(
            Ciclo(
                maquina=maquina,
                referencia=referencia,
                tiempo_ciclo=tiempo_ciclo,
            )
        )

    def delete(self, ciclo: Ciclo) -> None:
        self._db.delete(ciclo)


class EjecucionRepo:
    """Queries de ``oee.ejecuciones``."""

    def __init__(self, db: Session):
        self._db = db

    def list_recent(self, limit: int = 50) -> list[EjecucionRow]:
        rows = (
            self._db.execute(
                select(Ejecucion).order_by(Ejecucion.created_at.desc()).limit(limit)
            )
            .scalars()
            .all()
        )
        return [EjecucionRow.model_validate(r) for r in rows]

    def list_recent_orm(self, limit: int = 50) -> list[Ejecucion]:
        return list(
            self._db.execute(
                select(Ejecucion).order_by(Ejecucion.created_at.desc()).limit(limit)
            )
            .scalars()
            .all()
        )

    def get_by_id(self, id: int) -> EjecucionRow | None:
        row = self._db.get(Ejecucion, id)
        return EjecucionRow.model_validate(row) if row else None

    def get_orm_by_id(self, id: int) -> Ejecucion | None:
        return self._db.get(Ejecucion, id)


class MetricaRepo:
    """Queries de ``oee.metricas`` (tendencias por recurso)."""

    def __init__(self, db: Session):
        self._db = db

    def tendencias_recurso(
        self,
        recurso: str,
        fecha_inicio: str | None = None,
        fecha_fin: str | None = None,
    ) -> list[MetricaRow]:
        stmt = select(MetricaOEE).where(
            MetricaOEE.recurso.ilike(recurso),
            MetricaOEE.fecha.isnot(None),
            MetricaOEE.turno.is_(None),
        )
        if fecha_inicio:
            stmt = stmt.where(MetricaOEE.fecha >= date.fromisoformat(fecha_inicio))
        if fecha_fin:
            stmt = stmt.where(MetricaOEE.fecha <= date.fromisoformat(fecha_fin))
        stmt = stmt.order_by(MetricaOEE.fecha)
        rows = self._db.execute(stmt).scalars().all()
        return [MetricaRow.model_validate(r) for r in rows]


class LukRepo:
    """Wrapper opcional de queries ``luk4.*`` (en APP).

    El router ``api/routers/luk4.py`` sigue con queries ``text()``
    inline por simplicidad (per CONTEXT.md D-01: texto inline en router
    no requiere ``.sql`` versionado si la query es local al router).
    Placeholder para DATA-03; metodos concretos en Mark-IV si el refactor
    de luk4 gana nuevas queries reusables.
    """

    def __init__(self, db: Session):
        self._db = db


class ContactoRepo:
    """Queries de ``cfg.contactos``."""

    def __init__(self, db: Session):
        self._db = db

    def list_all(self) -> list[ContactoRow]:
        rows = (
            self._db.execute(select(Contacto).order_by(Contacto.nombre)).scalars().all()
        )
        return [ContactoRow.model_validate(r) for r in rows]


__all__ = [
    "RecursoRepo",
    "CicloRepo",
    "EjecucionRepo",
    "MetricaRepo",
    "LukRepo",
    "ContactoRepo",
]
