"""Gate IDENT-06: la app (rol nexo_app) NO puede modificar nexo.audit_log.

Plan 02-04, Tarea 4.4. Verifica a nivel de BD que el audit_log es
append-only desde el rol que usa la app en runtime:

- INSERT permitido (lo necesita el AuditMiddleware para escribir filas).
- UPDATE y DELETE bloqueados con ``permission denied``.
- El owner del schema (``oee``) conserva los privilegios completos
  implícitamente — tareas de mantenimiento offline siguen siendo
  posibles conectando explícitamente como owner.

Requisitos de entorno:
- ``docker compose up -d db web`` corriendo.
- Script ``scripts/create_nexo_app_role.sql`` aplicado (via
  ``make nexo-app-role``).
- ``.env`` con ``NEXO_PG_APP_USER=nexo_app`` y
  ``NEXO_PG_APP_PASSWORD=<secreto>``.

Si falta cualquiera de estos tres, ``pytestmark skipif`` salta la suite.
"""
from __future__ import annotations

from typing import Iterator

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError, DBAPIError

from api.config import settings
from nexo.db.engine import engine_nexo


def _connected_as_nexo_app() -> bool:
    """True solo si el engine runtime conecta como ``nexo_app``. Si la app
    aún usa el owner (``oee``), no podemos validar el gate — saltamos."""
    try:
        with engine_nexo.connect() as c:
            user = c.execute(text("SELECT current_user")).scalar()
        return user == "nexo_app"
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _connected_as_nexo_app(),
        reason=(
            "Engine runtime no conecta como nexo_app. Requiere "
            "`make nexo-app-role` + NEXO_PG_APP_USER/PASSWORD en .env "
            "+ `docker compose up -d --force-recreate web`."
        ),
    ),
]


TEST_PATH = "/_test_ident06_"


def _owner_engine():
    """Engine temporal conectando como el OWNER (``oee``) — unico rol que
    puede DELETE en nexo.audit_log. Se usa solo para cleanup de filas de
    test; la validacion del gate IDENT-06 se hace contra engine_nexo
    (nexo_app)."""
    from sqlalchemy import create_engine

    dsn = (
        f"postgresql+psycopg2://{settings.pg_user}:{settings.pg_password}"
        f"@{settings.pg_host}:{settings.pg_port}/{settings.pg_db}"
    )
    return create_engine(dsn, pool_pre_ping=False)


@pytest.fixture(autouse=True)
def _clean_test_rows():
    """Limpia filas de test antes y despues usando el rol owner.
    nexo_app no puede DELETE — ese es exactamente el gate — asi que
    necesitamos conectar con las credenciales del owner para limpiar."""
    owner = _owner_engine()

    def purge():
        try:
            with owner.begin() as c:
                c.execute(
                    text("DELETE FROM nexo.audit_log WHERE path = :p"),
                    {"p": TEST_PATH},
                )
        except Exception:
            # Silencioso: si falla el cleanup, el test sobrevive porque
            # las filas se identifican por path unico y no chocan entre
            # tests.
            pass

    purge()
    yield
    purge()
    owner.dispose()


def _get_test_user_id() -> int:
    """user_id existente para satisfacer la FK de nexo.audit_log.user_id."""
    with engine_nexo.connect() as c:
        uid = c.execute(text("SELECT id FROM nexo.users LIMIT 1")).scalar()
    assert uid is not None, "No hay users en nexo.users — ¿se ejecutó el bootstrap?"
    return uid


def test_insert_en_audit_log_esta_permitido():
    """AuditMiddleware necesita INSERT; nexo_app debe poder hacerlo."""
    uid = _get_test_user_id()
    with engine_nexo.begin() as c:
        c.execute(
            text(
                "INSERT INTO nexo.audit_log "
                "(ts, user_id, ip, method, path, status) "
                "VALUES (NOW(), :uid, :ip, :m, :p, :s)"
            ),
            {"uid": uid, "ip": "127.0.0.1", "m": "GET", "p": TEST_PATH, "s": 200},
        )

    # Confirmar que se escribió
    with engine_nexo.connect() as c:
        count = c.execute(
            text("SELECT COUNT(*) FROM nexo.audit_log WHERE path = :p"),
            {"p": TEST_PATH},
        ).scalar()
    assert count >= 1


def test_delete_en_audit_log_falla_con_permission_denied():
    """IDENT-06 gate: nexo_app NO puede borrar filas del audit log."""
    with pytest.raises((ProgrammingError, DBAPIError)) as exc_info:
        with engine_nexo.begin() as c:
            c.execute(text(f"DELETE FROM nexo.audit_log WHERE path = '{TEST_PATH}'"))
    error_msg = str(exc_info.value).lower()
    assert (
        "permission denied" in error_msg
        or "insufficient privilege" in error_msg
    ), f"Error no parece ser de permisos: {exc_info.value}"


def test_update_en_audit_log_falla_con_permission_denied():
    """IDENT-06 gate: nexo_app NO puede modificar filas del audit log.
    Tampoco se pueden tamperar timestamps, user_id, paths, etc. — el log
    es efectivamente inmutable desde la app."""
    with pytest.raises((ProgrammingError, DBAPIError)) as exc_info:
        with engine_nexo.begin() as c:
            c.execute(
                text(f"UPDATE nexo.audit_log SET status = 999 WHERE path = '{TEST_PATH}'")
            )
    error_msg = str(exc_info.value).lower()
    assert (
        "permission denied" in error_msg
        or "insufficient privilege" in error_msg
    ), f"Error no parece ser de permisos: {exc_info.value}"


def test_truncate_en_audit_log_falla_con_permission_denied():
    """Defensa adicional: TRUNCATE también debe fallar. Algunos atacantes
    intentan TRUNCATE si DELETE da error."""
    with pytest.raises((ProgrammingError, DBAPIError)) as exc_info:
        with engine_nexo.begin() as c:
            c.execute(text("TRUNCATE nexo.audit_log"))
    error_msg = str(exc_info.value).lower()
    # TRUNCATE requiere el privilegio TRUNCATE o ser owner. nexo_app no
    # tiene ninguno, así que falla con permission denied o must be owner.
    assert (
        "permission denied" in error_msg
        or "must be owner" in error_msg
    ), f"Error no parece ser de permisos/ownership: {exc_info.value}"


def test_select_en_audit_log_esta_permitido():
    """nexo_app sí necesita SELECT — la UI /ajustes/auditoria lo usa para
    listar filas y exportar CSV."""
    with engine_nexo.connect() as c:
        # No importa cuántas filas haya, solo que la query no falle.
        c.execute(text("SELECT COUNT(*) FROM nexo.audit_log")).scalar()
