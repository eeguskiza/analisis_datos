"""``schema_guard.verify`` — corre en ``lifespan``, aborta o auto-migra.

Comportamiento canónico (D-06 / D-07 del ``03-CONTEXT.md``):

- Inspecciona el schema ``nexo`` del engine Postgres al arrancar.
- Si todas las tablas críticas existen → loguea ``INFO`` y continúa.
- Si falta alguna y ``NEXO_AUTO_MIGRATE`` no está activo → lanza
  ``RuntimeError`` con mensaje explícito y la app NO arranca.
- Si falta alguna y ``NEXO_AUTO_MIGRATE=true`` → ejecuta
  ``NexoBase.metadata.create_all()`` y loguea ``WARNING`` ("sólo dev").

Plan 08-03 (UIREDO-02) — Columnas crí­ticas: además de tablas, la
lifespan verifica que columnas concretas existan. ``users.nombre`` es la
primera entrada (necesaria para ``/ajustes/usuarios`` + ``base.html``
topbar + landing ``/bienvenida``). Si falta y ``NEXO_AUTO_MIGRATE=true``,
se aplica ``migration_add_users_nombre.sql`` (idempotente). Si falta y
``NEXO_AUTO_MIGRATE`` no esta activo, ``RuntimeError`` con mensaje en
espanol indicando el SQL concreto a ejecutar.

``verify`` acepta ``critical_tables`` y ``required_columns`` como kwargs
(defaults ``CRITICAL_TABLES`` / ``REQUIRED_COLUMNS``) para que los tests
inyecten nombres ficticios sin monkeypatchear las constantes del módulo.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

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
    # Phase 4 / Plan 04-01 — QUERY-01 + QUERY-02 foundation.
    "query_log",
    "query_thresholds",
    "query_approvals",
)


# Plan 08-03 — columnas críticas por tabla. Cada tupla ``(table, column)``
# se verifica tras comprobar las tablas. Si la columna falta y
# ``NEXO_AUTO_MIGRATE=true``, se ejecuta el SQL asociado en
# ``_COLUMN_MIGRATIONS``. Si no, se lanza ``RuntimeError`` con el path
# concreto del SQL a aplicar manualmente.
REQUIRED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("users", "nombre"),  # Plan 08-03 / UIREDO-02
)


# Mapa de (tabla, columna) -> Path al SQL de migración idempotente.
# Los SQL viven en ``nexo/data/sql/nexo/``. Son idempotentes
# (``ADD COLUMN IF NOT EXISTS`` + backfill con ``WHERE ... IS NULL``) y
# seguros de re-ejecutar.
_SQL_DIR = Path(__file__).resolve().parent / "sql" / "nexo"
_COLUMN_MIGRATIONS: dict[tuple[str, str], Path] = {
    ("users", "nombre"): _SQL_DIR / "migration_add_users_nombre.sql",
}


def _auto_migrate_enabled() -> bool:
    """True si ``NEXO_AUTO_MIGRATE`` está explícitamente en dev."""
    return os.environ.get("NEXO_AUTO_MIGRATE", "").lower() in {"1", "true", "yes"}


def _missing_columns(
    engine: Engine,
    required_columns: tuple[tuple[str, str], ...],
) -> list[tuple[str, str]]:
    """Devuelve lista de ``(tabla, columna)`` ausentes del schema ``nexo``."""
    insp = inspect(engine)
    missing: list[tuple[str, str]] = []
    for table, column in required_columns:
        try:
            cols = {c["name"] for c in insp.get_columns(table, schema=NEXO_SCHEMA)}
        except Exception:  # tabla inexistente — ya reportada por _missing_tables
            missing.append((table, column))
            continue
        if column not in cols:
            missing.append((table, column))
    return missing


def _apply_column_migration(
    engine: Engine,
    table_column: tuple[str, str],
) -> None:
    """Aplica el SQL de migración idempotente para ``(tabla, columna)``.

    Postgres acepta múltiples statements en una sola ejecución sólo si
    se envían via ``connection.exec_driver_sql`` (raw) — ``text()`` sólo
    acepta 1 statement. El SQL vive en archivos, por lo que lo pasamos
    por ``exec_driver_sql`` dentro de una transacción.
    """
    sql_path = _COLUMN_MIGRATIONS.get(table_column)
    if sql_path is None or not sql_path.exists():
        raise RuntimeError(
            f"schema_guard: sin migracion registrada para columna faltante "
            f"{table_column} (path esperado: {sql_path})"
        )
    sql = sql_path.read_text(encoding="utf-8")
    with engine.begin() as conn:
        conn.exec_driver_sql(sql)
    log.warning(
        "auto-migrate columna %s.%s aplicado desde %s",
        table_column[0],
        table_column[1],
        sql_path.name,
    )


def verify(
    engine: Engine,
    critical_tables: tuple[str, ...] = CRITICAL_TABLES,
    required_columns: tuple[tuple[str, str], ...] = REQUIRED_COLUMNS,
) -> None:
    """Verifica tablas y columnas críticas del schema ``nexo``.

    Args:
        engine: engine Postgres (normalmente ``engine_nexo``) a
            inspeccionar.
        critical_tables: nombres de tabla esperados en el schema
            ``nexo``. Por defecto, las 11 tablas críticas
            (``CRITICAL_TABLES``). Los tests pueden pasar una tupla
            distinta (p. ej. incluyendo ``"__nonexistent__"``) sin
            monkeypatchear el módulo.
        required_columns: lista de ``(tabla, columna)`` requeridas. Si
            alguna falta y ``NEXO_AUTO_MIGRATE=true``, se aplica el SQL
            idempotente registrado en ``_COLUMN_MIGRATIONS``. Si no,
            ``RuntimeError``.

    Raises:
        RuntimeError: si faltan tablas o columnas y
            ``NEXO_AUTO_MIGRATE`` no está activo. La ``lifespan``
            dejará que la excepción suba hasta ``uvicorn`` y la app NO
            arranca — comportamiento deseado para detectar drift de
            schema antes del primer request.
    """
    insp = inspect(engine)
    existing = set(insp.get_table_names(schema=NEXO_SCHEMA))
    missing_tables = [t for t in critical_tables if t not in existing]

    if missing_tables:
        if not _auto_migrate_enabled():
            raise RuntimeError(
                f"Schema guard: faltan tablas en nexo.* -> {missing_tables}. "
                "Ejecuta `make nexo-init` o define NEXO_AUTO_MIGRATE=true "
                "(solo dev) para crearlas automaticamente."
            )
        log.warning(
            "NEXO_AUTO_MIGRATE activo — creando %d tablas faltantes: %s",
            len(missing_tables),
            missing_tables,
        )
        NexoBase.metadata.create_all(bind=engine)
        log.warning("auto-migracion completada. NO usar en produccion.")

    # Tras asegurar que las tablas existen (o se crearon), verificamos
    # columnas. Plan 08-03: users.nombre es la primera.
    missing_cols = _missing_columns(engine, required_columns)
    if missing_cols:
        if not _auto_migrate_enabled():
            detail = ", ".join(
                f"{t}.{c} (aplicar: {_COLUMN_MIGRATIONS.get((t, c), '<sin migracion>')})"
                for t, c in missing_cols
            )
            raise RuntimeError(
                f"Schema guard: faltan columnas en nexo.* -> {detail}. "
                "Ejecuta la(s) migracion(es) manualmente o define "
                "NEXO_AUTO_MIGRATE=true (solo dev) para aplicarlas."
            )
        for table_column in missing_cols:
            _apply_column_migration(engine, table_column)

    log.info(
        "schema_guard OK — %d tablas nexo.* + %d columnas criticas",
        len(critical_tables),
        len(required_columns),
    )


__all__ = ["verify", "CRITICAL_TABLES", "REQUIRED_COLUMNS"]
