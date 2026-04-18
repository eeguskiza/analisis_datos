#!/usr/bin/env python3
"""Inicializa el schema ``nexo`` en Postgres: crea schema + 8 tablas +
seed de roles/departments/permissions + privilegios de append-only en
``nexo.audit_log``.

Idempotente: correrlo dos veces seguidas termina con exit 0 y sin
duplicar filas.

Uso::

    docker compose exec web python scripts/init_nexo_schema.py

No crea usuarios. Para bootstrap del primer ``propietario`` usa
``scripts/create_propietario.py``.

Notas de diseño
---------------

- ``CREATE SCHEMA IF NOT EXISTS`` ejecutado antes de ``create_all`` para
  que SQLAlchemy no falle al instanciar tablas con ``schema='nexo'``.
- Seed de catálogos con ``INSERT ... ON CONFLICT (code) DO NOTHING``
  — idempotencia a nivel fila.
- El catálogo ``nexo.permissions`` es un snapshot del ``PERMISSION_MAP``
  que plan 02-03 definirá como fuente de verdad en código. El snapshot
  existe sólo para que la UI de Phase 5 (sidebar filtrado) pueda
  consultarlo sin importar código Python.
- GRANT SELECT, INSERT en ``nexo.audit_log`` (research §Pattern 8 —
  alternativa robusta al REVOKE UPDATE/DELETE del AUTH_MODEL.md).

**ADVERTENCIA**: si el rol Postgres que corre este script es owner del
schema/tabla (caso de Mark-III en dev con usuario ``oee``), el
GRANT-only **no** impide UPDATE/DELETE — owner siempre los tiene. El
test IDENT-06 del plan 02-04 lo detectará. En Mark-IV se introduce un
rol ``nexo_app`` dedicado. En Mark-III la propiedad append-only queda
degradada pero documentada.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Permite ejecutar el script directo sin pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from api.config import settings  # noqa: E402
from nexo.db.engine import engine_nexo  # noqa: E402
from nexo.db.models import NexoBase  # noqa: E402


NEXO_SCHEMA = "nexo"


ROLES_SEED: list[tuple[str, str]] = [
    ("propietario", "Propietario"),
    ("directivo", "Directivo"),
    ("usuario", "Usuario"),
]

DEPARTMENTS_SEED: list[tuple[str, str]] = [
    ("rrhh", "Recursos Humanos"),
    ("comercial", "Comercial"),
    ("ingenieria", "Ingenieria"),
    ("produccion", "Produccion"),
    ("gerencia", "Gerencia"),
]

# Catálogo seed del PERMISSION_MAP. La fuente de verdad autoritativa vive
# en ``nexo/services/auth.py`` a partir del plan 02-03; este snapshot
# queda como referencia para la UI de Phase 5.
# Tupla: (module, action, role_code, department_code or None)
# department_code=None → propietario global (ignora departamento).
PERMISSIONS_SEED: list[tuple[str, str, str, str | None]] = [
    # Propietario — acceso global (department_code NULL)
    ("pipeline", "read", "propietario", None),
    ("pipeline", "execute", "propietario", None),
    ("pipeline", "delete", "propietario", None),
    ("historial", "read", "propietario", None),
    ("historial", "delete", "propietario", None),
    ("capacidad", "read", "propietario", None),
    ("recursos", "read", "propietario", None),
    ("recursos", "write", "propietario", None),
    ("ciclos", "read", "propietario", None),
    ("ciclos", "write", "propietario", None),
    ("operarios", "read", "propietario", None),
    ("bbdd", "read", "propietario", None),
    ("bbdd", "query", "propietario", None),
    ("luk4", "read", "propietario", None),
    ("centro_mando", "read", "propietario", None),
    ("datos", "read", "propietario", None),
    ("plantillas", "read", "propietario", None),
    ("ajustes", "read", "propietario", None),
    ("ajustes", "write", "propietario", None),
    ("auditoria", "read", "propietario", None),
    ("usuarios", "read", "propietario", None),
    ("usuarios", "write", "propietario", None),
    # Directivo — todos los departamentos por defecto leen; write se limita
    # por departamento en el dict de código (aquí catálogo completo).
    ("pipeline", "read", "directivo", "ingenieria"),
    ("pipeline", "execute", "directivo", "ingenieria"),
    ("pipeline", "read", "directivo", "produccion"),
    ("pipeline", "execute", "directivo", "produccion"),
    ("historial", "read", "directivo", "ingenieria"),
    ("historial", "read", "directivo", "produccion"),
    ("historial", "read", "directivo", "comercial"),
    ("historial", "read", "directivo", "gerencia"),
    ("capacidad", "read", "directivo", "comercial"),
    ("capacidad", "read", "directivo", "produccion"),
    ("capacidad", "read", "directivo", "gerencia"),
    ("recursos", "read", "directivo", "ingenieria"),
    ("ciclos", "read", "directivo", "ingenieria"),
    ("operarios", "read", "directivo", "rrhh"),
    ("luk4", "read", "directivo", "produccion"),
    ("centro_mando", "read", "directivo", "produccion"),
    ("centro_mando", "read", "directivo", "gerencia"),
    # Usuario — lectura y ejecución básica en sus departamentos
    ("pipeline", "read", "usuario", "ingenieria"),
    ("pipeline", "read", "usuario", "produccion"),
    ("historial", "read", "usuario", "ingenieria"),
    ("historial", "read", "usuario", "produccion"),
    ("capacidad", "read", "usuario", "comercial"),
    ("capacidad", "read", "usuario", "produccion"),
    ("recursos", "read", "usuario", "ingenieria"),
    ("ciclos", "read", "usuario", "ingenieria"),
    ("operarios", "read", "usuario", "rrhh"),
    ("luk4", "read", "usuario", "produccion"),
    ("centro_mando", "read", "usuario", "produccion"),
]


def _log(msg: str) -> None:
    print(f"[init_nexo_schema] {msg}")


def create_schema() -> None:
    _log(f"CREATE SCHEMA IF NOT EXISTS {NEXO_SCHEMA}")
    with engine_nexo.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{NEXO_SCHEMA}"'))


def create_tables() -> None:
    _log("create_all() sobre NexoBase.metadata (8 tablas)")
    NexoBase.metadata.create_all(engine_nexo)


def seed_roles() -> None:
    _log(f"Seed de {len(ROLES_SEED)} roles")
    with engine_nexo.begin() as conn:
        for code, name in ROLES_SEED:
            conn.execute(
                text(
                    "INSERT INTO nexo.roles (code, name) VALUES (:c, :n) "
                    "ON CONFLICT (code) DO NOTHING"
                ),
                {"c": code, "n": name},
            )


def seed_departments() -> None:
    _log(f"Seed de {len(DEPARTMENTS_SEED)} departamentos")
    with engine_nexo.begin() as conn:
        for code, name in DEPARTMENTS_SEED:
            conn.execute(
                text(
                    "INSERT INTO nexo.departments (code, name) VALUES (:c, :n) "
                    "ON CONFLICT (code) DO NOTHING"
                ),
                {"c": code, "n": name},
            )


def seed_permissions() -> None:
    _log(f"Seed de {len(PERMISSIONS_SEED)} permisos (catálogo)")
    with engine_nexo.begin() as conn:
        for module, action, role_code, dept_code in PERMISSIONS_SEED:
            conn.execute(
                text(
                    "INSERT INTO nexo.permissions (module, action, role_code, department_code) "
                    "VALUES (:m, :a, :r, :d) ON CONFLICT ON CONSTRAINT uq_permission_tuple DO NOTHING"
                ),
                {"m": module, "a": action, "r": role_code, "d": dept_code},
            )


def grant_audit_log_privileges() -> None:
    """Restringe privilegios al rol app sobre ``nexo.audit_log``.

    Efecto pretendido: el rol app sólo puede SELECT + INSERT. UPDATE y
    DELETE fallan con ``permission denied``.

    Limitación conocida: si ``settings.pg_user`` es el owner del schema
    (caso dev Mark-III), el GRANT-only no tiene efecto — owner siempre
    tiene todos los privilegios. El test de 02-04 lo detectará.
    """
    role = settings.pg_user
    _log(f"REVOKE ALL + GRANT SELECT, INSERT sobre nexo.audit_log al rol '{role}'")
    with engine_nexo.begin() as conn:
        conn.execute(text("REVOKE ALL ON nexo.audit_log FROM PUBLIC"))
        conn.execute(text(f'REVOKE ALL ON nexo.audit_log FROM "{role}"'))
        conn.execute(text(f'GRANT SELECT, INSERT ON nexo.audit_log TO "{role}"'))
        # La secuencia del PK también necesita permiso para que INSERT
        # pueda obtener nextval.
        conn.execute(
            text(f'GRANT USAGE, SELECT ON SEQUENCE nexo.audit_log_id_seq TO "{role}"')
        )


def main() -> int:
    _log(f"Usando DSN: postgresql+psycopg2://{settings.pg_user}:***@"
         f"{settings.pg_host}:{settings.pg_port}/{settings.pg_db}")
    create_schema()
    create_tables()
    seed_roles()
    seed_departments()
    seed_permissions()
    grant_audit_log_privileges()
    _log("OK — schema nexo listo.")
    _log(
        "ADVERTENCIA: si el rol app es owner del schema, el GRANT-only no "
        "impide UPDATE/DELETE en nexo.audit_log. El test IDENT-06 (plan "
        "02-04) lo verificará. Aceptable en Mark-III; Mark-IV introduce "
        "un rol 'nexo_app' dedicado."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
