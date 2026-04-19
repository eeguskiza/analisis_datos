"""DATA-09: ``engine_mes`` apunta a ``dbizaro`` (smoke contra SQL Server real).

Plan 03-02 Task 4. Integration test: skipea si no hay conectividad LAN
al servidor MES (caso tipico en CI). En preprod/produccion debe correr
verde — garantiza que el DSN ``DATABASE=dbizaro`` esta aplicado y que
queries con 2-part names (``admuser.fmesmic``) funcionan sin 3-part.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

from api.config import settings
from nexo.data.engines import engine_mes


def _mssql_reachable() -> bool:
    try:
        with engine_mes.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _mssql_reachable(),
        reason="SQL Server MES no alcanzable — skipeable en CI",
    ),
]


def test_engine_mes_default_db_is_dbizaro():
    """DB_NAME() == settings.mes_db == 'dbizaro' (DATA-09 baseline)."""
    with engine_mes.connect() as conn:
        db = conn.execute(text("SELECT DB_NAME()")).scalar()
    assert db == settings.mes_db
    assert db == "dbizaro", (
        f"engine_mes default catalog esperado 'dbizaro', got {db!r}"
    )


def test_two_part_name_query_works():
    """Valida que ``admuser.fmesmic`` resuelve sin prefix de catalog."""
    with engine_mes.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM admuser.fmesmic")).scalar()
    assert isinstance(n, int)
