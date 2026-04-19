"""``schema_guard.verify`` — corre en ``lifespan``, aborta o auto-migra.

Comportamiento canónico (D-06 / D-07 del ``03-CONTEXT.md``):

- Inspecciona el schema ``nexo`` del engine Postgres al arrancar.
- Si todas las tablas críticas existen → loguea ``INFO`` y continúa.
- Si falta alguna y ``NEXO_AUTO_MIGRATE`` no está activo → lanza
  ``RuntimeError`` con mensaje explícito y la app NO arranca.
- Si falta alguna y ``NEXO_AUTO_MIGRATE=true`` → ejecuta
  ``NexoBase.metadata.create_all()`` y loguea ``WARNING`` ("sólo dev").

``verify`` acepta ``critical_tables`` como kwarg (default
``CRITICAL_TABLES``) para que los tests inyecten nombres ficticios sin
monkeypatchear la constante del módulo (patrón más robusto si
``CRITICAL_TABLES`` migra a ``frozenset`` / computed).
"""
from __future__ import annotations

import logging
import os

from sqlalchemy import inspect
from sqlalchemy.engine import Engine

# Durante 03-01 los modelos siguen en ``nexo/db/models.py``; en 03-03 se
# mueven a ``nexo/data/models_nexo.py`` pero el shim expondrá ambos
# nombres y este import no tendrá que cambiar.
from nexo.db.models import NEXO_SCHEMA, NexoBase

log = logging.getLogger("nexo.schema_guard")


CRITICAL_TABLES: tuple[str, ...] = (
    "users",
    "roles",
    "departments",
    "user_departments",
    "permissions",
    "sessions",
    "login_attempts",
    "audit_log",
)


def _auto_migrate_enabled() -> bool:
    """True si ``NEXO_AUTO_MIGRATE`` está explícitamente en dev."""
    return os.environ.get("NEXO_AUTO_MIGRATE", "").lower() in {"1", "true", "yes"}


def verify(
    engine: Engine,
    critical_tables: tuple[str, ...] = CRITICAL_TABLES,
) -> None:
    """Verifica tablas críticas del schema ``nexo``.

    Args:
        engine: engine Postgres (normalmente ``engine_nexo``) a
            inspeccionar.
        critical_tables: nombres de tabla esperados en el schema
            ``nexo``. Por defecto, las 8 tablas críticas de Phase 2
            (``CRITICAL_TABLES``). Los tests pueden pasar una tupla
            distinta (p. ej. incluyendo ``"__nonexistent__"``) sin
            monkeypatchear el módulo.

    Raises:
        RuntimeError: si faltan tablas y ``NEXO_AUTO_MIGRATE`` no está
            activo. La ``lifespan`` dejará que la excepción suba hasta
            ``uvicorn`` y la app NO arranca — comportamiento deseado
            para detectar drift de schema antes del primer request.
    """
    insp = inspect(engine)
    existing = set(insp.get_table_names(schema=NEXO_SCHEMA))
    missing = [t for t in critical_tables if t not in existing]

    if not missing:
        log.info(
            "schema_guard OK — %d tablas nexo.* presentes", len(critical_tables)
        )
        return

    if not _auto_migrate_enabled():
        raise RuntimeError(
            f"Schema guard: faltan tablas en nexo.* -> {missing}. "
            "Ejecuta `make nexo-init` o define NEXO_AUTO_MIGRATE=true "
            "(solo dev) para crearlas automaticamente."
        )

    log.warning(
        "NEXO_AUTO_MIGRATE activo — creando %d tablas faltantes: %s",
        len(missing),
        missing,
    )
    NexoBase.metadata.create_all(bind=engine)
    log.warning("auto-migracion completada. NO usar en produccion.")


__all__ = ["verify", "CRITICAL_TABLES"]
