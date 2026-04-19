"""D-05: contract test del whitelist anti-DDL en ``api/routers/bbdd.py``.

Mitiga T-03-02-01 (SQL injection via ``consulta_readonly`` si el
whitelist falla). Plan 03-02 Task 4.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from api.routers.bbdd import _validate_sql


FORBIDDEN_STATEMENTS = [
    "DROP TABLE foo",
    "DELETE FROM foo",
    "UPDATE foo SET a=1",
    "INSERT INTO foo VALUES (1)",
    "TRUNCATE TABLE foo",
    "ALTER TABLE foo ADD b INT",
    "GRANT SELECT ON foo TO bar",
    "REVOKE SELECT ON foo FROM bar",
    "CREATE TABLE foo (a INT)",
    "  delete  from foo  ",  # case insensitive + whitespace
    "select 1; DROP TABLE foo;",  # multi-statement injection
]

ALLOWED_STATEMENTS = [
    "SELECT 1",
    "SELECT * FROM admuser.fmesmic WHERE mi020 = '47'",
    "select count(*) from admuser.fmesdtc",
]


@pytest.mark.parametrize("stmt", FORBIDDEN_STATEMENTS)
def test_whitelist_rejects_ddl_dml(stmt):
    with pytest.raises(HTTPException) as exc_info:
        _validate_sql(stmt)
    # Todos los DDL/DML deben devolver 400 (Bad Request), no 500.
    assert exc_info.value.status_code == 400


@pytest.mark.parametrize("stmt", ALLOWED_STATEMENTS)
def test_whitelist_allows_select(stmt):
    # No raise = OK
    _validate_sql(stmt)


def test_whitelist_rejects_empty():
    """SQL vacia o solo whitespace devuelve 400."""
    with pytest.raises(HTTPException) as exc_info:
        _validate_sql("")
    assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException) as exc_info:
        _validate_sql("   \n\t")
    assert exc_info.value.status_code == 400


def test_whitelist_rejects_sp_prefix():
    """Los prefijos sp_ / xp_ estan rechazados por la logica de prefix del whitelist."""
    with pytest.raises(HTTPException):
        _validate_sql("SELECT * FROM sp_who")
    with pytest.raises(HTTPException):
        _validate_sql("EXEC xp_cmdshell 'dir'")
