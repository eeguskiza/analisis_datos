"""DATA-04: integration tests contra Postgres ``nexo.*``.

Skipif automatico si Postgres no esta arriba (conftest.db_nexo fixture).

Incluye contract test T-03-03-01 (threat register): verifica que
``AuditRepo.append`` NO comitea internamente; el caller orquesta la
transaccion. IDENT-06 gate (DB-level permission denied) sigue como
defense-in-depth en tests/auth/test_audit_append_only.py.
"""
from __future__ import annotations

import inspect

import pytest

from nexo.data.dto.nexo import AuditLogRow, RoleRow, UserRow
from nexo.data.repositories.nexo import AuditRepo, RoleRepo, UserRepo

pytestmark = pytest.mark.integration


def test_user_repo_get_by_email_returns_dto_or_none(db_nexo):
    """UserRepo.get_by_email devuelve UserRow DTO o None sin lanzar."""
    repo = UserRepo(db_nexo)
    # Propietario suele existir tras Phase 2; si no, None es valido.
    result = repo.get_by_email("propietario@ecsmobility.com")
    assert result is None or isinstance(result, UserRow)
    if result is not None:
        assert result.email == "propietario@ecsmobility.com"


def test_user_repo_get_by_email_orm_returns_orm_or_none(db_nexo):
    """UserRepo.get_by_email_orm devuelve NexoUser (ORM) o None."""
    repo = UserRepo(db_nexo)
    result = repo.get_by_email_orm("no_existe_nunca@example.com")
    assert result is None


def test_user_repo_list_all_returns_dtos(db_nexo):
    """UserRepo.list_all devuelve list[UserRow]."""
    repo = UserRepo(db_nexo)
    users = repo.list_all()
    assert isinstance(users, list)
    assert all(isinstance(u, UserRow) for u in users)
    # departments es tuple (frozen-compatible)
    if users:
        assert isinstance(users[0].departments, tuple)


def test_role_repo_lists_roles(db_nexo):
    """RoleRepo.list_all incluye los 3 roles seed de Phase 2."""
    repo = RoleRepo(db_nexo)
    roles = repo.list_all()
    assert all(isinstance(r, RoleRow) for r in roles)
    codes = {r.code for r in roles}
    assert codes.issuperset({"propietario", "directivo", "usuario"})


def test_role_repo_list_departments(db_nexo):
    """RoleRepo.list_departments incluye los 5 departments seed."""
    repo = RoleRepo(db_nexo)
    depts = repo.list_departments()
    codes = {d.code for d in depts}
    assert codes.issuperset({"rrhh", "comercial", "ingenieria", "produccion", "gerencia"})


def test_audit_repo_append_source_has_no_commit():
    """Contract test T-03-03-01 (threat register):
    AuditRepo.append NO debe contener .commit() ni .flush() en su source.
    El caller orquesta la transaccion. Defense-in-depth: el rol nexo_app
    tampoco puede UPDATE/DELETE a nivel DB (tests/auth/test_audit_append_only.py).
    """
    src = inspect.getsource(AuditRepo.append)
    # Comprobar calls, no palabras en docstrings
    assert ".commit()" not in src, f"AuditRepo.append contiene .commit(): {src}"
    assert ".flush()" not in src, f"AuditRepo.append contiene .flush(): {src}"
    assert "db.commit" not in src, f"AuditRepo.append contiene db.commit: {src}"


def test_audit_repo_append_does_not_commit(db_nexo):
    """AuditRepo.append mete fila en session.new sin commit implicito.

    Dentro de la sesion abierta por la fixture, `session.new` contiene
    el objeto pendiente; al cerrar la fixture con rollback (conftest),
    la fila desaparece.
    """
    repo = AuditRepo(db_nexo)

    before_new = len(list(db_nexo.new))
    repo.append(
        user_id=None,
        ip="127.0.0.1",
        method="GET",
        path="/_test_03_03_no_commit_",
        status=200,
        details_json=None,
    )
    after_new = len(list(db_nexo.new))

    # Fila anadida al session.new pero NO comiteada
    assert after_new == before_new + 1, (
        f"Esperado +1 en session.new, got before={before_new} after={after_new}"
    )
    # in_transaction sigue activo (no hubo commit implicito)
    assert db_nexo.in_transaction(), (
        "La sesion salio de la transaccion - AuditRepo.append comiteo implicitamente"
    )


def test_audit_repo_list_filtered_returns_dtos(db_nexo):
    """list_filtered devuelve AuditLogRow sin petar, incluso con filtros vacios."""
    repo = AuditRepo(db_nexo)
    rows = repo.list_filtered(limit=5)
    assert isinstance(rows, list)
    assert all(isinstance(r, AuditLogRow) for r in rows)


def test_audit_repo_count_filtered_returns_int(db_nexo):
    """count_filtered devuelve un int >= 0."""
    repo = AuditRepo(db_nexo)
    total = repo.count_filtered()
    assert isinstance(total, int)
    assert total >= 0


def test_audit_repo_iter_filtered_is_generator(db_nexo):
    """iter_filtered devuelve un iterable de AuditLogRow (para CSV streaming)."""
    repo = AuditRepo(db_nexo)
    gen = repo.iter_filtered()
    # Debe ser iterable; consumir primeros items sin petar
    count = 0
    for row in gen:
        assert isinstance(row, AuditLogRow)
        count += 1
        if count >= 3:
            break
