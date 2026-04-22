"""Regression for Phase 8 / Plan 08-03: nexo.users.nombre column.

Locks:
1. NexoUser model has ``nombre`` as optional String(120).
2. Schema guard lists the column as required.
3. Migration is idempotent (safe to run twice).
4. New users can be created with and without nombre.
5. Empty / whitespace nombre is stored as NULL (router-level helper).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import delete, inspect, select, text

from nexo.data.engines import SessionLocalNexo
from nexo.data.models_nexo import NexoLoginAttempt, NexoSession, NexoUser
from nexo.data.schema_guard import REQUIRED_COLUMNS
from nexo.services.auth import hash_password

_MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "nexo"
    / "data"
    / "sql"
    / "nexo"
    / "migration_add_users_nombre.sql"
)


TEST_DOMAIN = "@nombre-col-test.local"
TEST_PASSWORD = "nombrecoltest12345"


def _postgres_reachable() -> bool:
    try:
        db = SessionLocalNexo()
        try:
            db.execute(text("SELECT 1"))
            return True
        finally:
            db.close()
    except Exception:
        return False


# Los tests que tocan DB necesitan Postgres arriba; los que solo leen
# archivos / inspeccionan la clase ORM corren siempre.

_pg_required = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="Postgres no arriba: docker compose up -d db",
)
_integration = pytest.mark.integration


# ── Offline (no Postgres) ────────────────────────────────────────────


def test_migration_file_exists() -> None:
    assert _MIGRATION.exists(), f"Missing migration file: {_MIGRATION}"


def test_migration_is_idempotent_text() -> None:
    sql = _MIGRATION.read_text(encoding="utf-8")
    assert "ADD COLUMN IF NOT EXISTS nombre" in sql
    assert "UPDATE nexo.users" in sql
    assert "WHERE nombre IS NULL" in sql


def test_nexo_user_has_nombre_attribute() -> None:
    assert hasattr(NexoUser, "nombre"), "ORM NexoUser debe declarar `nombre`"
    col = NexoUser.__table__.columns["nombre"]
    assert col.nullable is True
    assert "VARCHAR" in str(col.type).upper()


def test_schema_guard_requires_nombre_column() -> None:
    assert ("users", "nombre") in REQUIRED_COLUMNS


# ── Integration (Postgres) ───────────────────────────────────────────


@pytest.fixture
def nexo_db_session() -> Iterator:
    if not _postgres_reachable():
        pytest.skip("Postgres no arriba: docker compose up -d db")
    db = SessionLocalNexo()
    try:
        yield db
    finally:
        db.rollback()
        db.close()


@pytest.fixture(autouse=True)
def _cleanup_test_users() -> Iterator[None]:
    """Purga usuarios con dominio ``@nombre-col-test.local`` antes y despues."""
    _purge()
    yield
    _purge()


def _purge() -> None:
    if not _postgres_reachable():
        return
    db = SessionLocalNexo()
    try:
        users = (
            db.execute(select(NexoUser).where(NexoUser.email.like(f"%{TEST_DOMAIN}")))
            .scalars()
            .all()
        )
        for u in users:
            db.execute(delete(NexoSession).where(NexoSession.user_id == u.id))
            db.delete(u)
        db.execute(delete(NexoLoginAttempt).where(NexoLoginAttempt.email.like(f"%{TEST_DOMAIN}")))
        db.commit()
    finally:
        db.close()


def _make_user(db, email: str, nombre: str | None) -> NexoUser:
    u = NexoUser(
        email=email,
        nombre=nombre,
        password_hash=hash_password(TEST_PASSWORD),
        role="usuario",
        active=True,
        must_change_password=False,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


@_integration
@_pg_required
def test_nombre_column_is_nullable_and_short() -> None:
    """Runtime check against the migrated schema."""
    from nexo.data.engines import engine_nexo

    insp = inspect(engine_nexo)
    cols = {c["name"]: c for c in insp.get_columns("users", schema="nexo")}
    assert "nombre" in cols
    assert cols["nombre"]["nullable"] is True
    assert "VARCHAR" in str(cols["nombre"]["type"]).upper()


@_integration
@_pg_required
def test_backfill_populates_existing_rows() -> None:
    """Backfill must have run: cada usuario pre-existente tiene nombre != NULL."""
    db = SessionLocalNexo()
    try:
        rows = list(
            db.execute(
                text(
                    "SELECT email, nombre FROM nexo.users "
                    f"WHERE email NOT LIKE '%{TEST_DOMAIN}' AND email IS NOT NULL"
                )
            )
        )
        for email, nombre in rows:
            assert nombre is not None, f"{email} still has NULL nombre after migration"
            # Primera letra uppercase (UPPER(LEFT(..., 1))).
            assert nombre[0] == nombre[0].upper()
    finally:
        db.close()


@_integration
@_pg_required
def test_create_user_with_nombre(nexo_db_session) -> None:
    u = _make_user(nexo_db_session, f"ada{TEST_DOMAIN}", nombre="Ada Lovelace")
    assert u.nombre == "Ada Lovelace"


@_integration
@_pg_required
def test_create_user_without_nombre(nexo_db_session) -> None:
    u = _make_user(nexo_db_session, f"nobody{TEST_DOMAIN}", nombre=None)
    assert u.nombre is None


@_integration
@_pg_required
def test_nombre_120_char_hard_limit(nexo_db_session) -> None:
    """VARCHAR(120) hard limit (T-08-03-04 DoS mitigation)."""
    valid = "A" * 120
    u = _make_user(nexo_db_session, f"max-len{TEST_DOMAIN}", nombre=valid)
    assert u.nombre == valid
    assert len(u.nombre) == 120
