"""Backwards-compat shim para imports existentes.

Historia: Plan 03-01 introdujo las fixtures ``db_nexo`` / ``db_app`` /
``engine_mes_mock`` y los helpers ``_postgres_reachable`` /
``_mssql_reachable`` aqui, pero pytest solo auto-descubre conftest.py
en el mismo directorio o padres, no en hermanos. En Plan 04-01
movimos los fixtures a ``tests/conftest.py`` (raiz) para que
``tests/services/`` y ``tests/routers/`` puedan consumirlos igual.

Este modulo **reexporta** los simbolos para no romper los imports
existentes que hacen ``from tests.data.conftest import _postgres_reachable``
(ver ``tests/data/test_schema_guard.py``, ``test_schema_query_log.py``,
``test_schema_guard_extended.py`` y ``tests/services/test_approvals.py``,
``test_thresholds_cache.py``).

Mantener este shim hasta que todos los consumers migren sus imports a
``tests.conftest``. Eliminar en Sprint 2 cuando se reorganice la
jerarquia de tests.
"""
from tests.conftest import (  # noqa: F401
    _mssql_reachable,
    _postgres_reachable,
    db_app,
    db_nexo,
    engine_mes_mock,
)
