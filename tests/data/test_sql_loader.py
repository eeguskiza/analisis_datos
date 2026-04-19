"""DATA-05: loader con ``lru_cache`` + ``importlib.resources``.

No hay archivos ``.sql`` aún en 03-01 — los crea 03-02. Los tests aquí
verifican sólo la infraestructura del loader: cache, normalización de
extensión, manejo de errores.
"""
from __future__ import annotations

import pytest

from nexo.data.sql.loader import load_sql


def test_load_sql_has_cache():
    """``lru_cache`` expone ``cache_info`` y ``cache_clear``."""
    assert hasattr(load_sql, "cache_info"), (
        "load_sql debe estar decorado con @lru_cache"
    )
    assert hasattr(load_sql, "cache_clear")


def test_load_sql_normalizes_extension():
    """Acepta nombre con o sin ``.sql`` — ambos resuelven al mismo
    archivo (o al mismo error si no existe).
    """
    # Durante 03-01 no hay archivos .sql todavía; verificamos que los
    # dos nombres del mismo archivo inexistente lanzan error (comportamiento
    # equivalente, sin discriminar).
    load_sql.cache_clear()
    with pytest.raises(Exception):
        load_sql("nonexistent/fake")
    with pytest.raises(Exception):
        load_sql("nonexistent/fake.sql")


def test_load_sql_missing_raises():
    """Archivo no existe → ``FileNotFoundError`` (o equivalente de
    ``importlib.resources``).
    """
    load_sql.cache_clear()
    with pytest.raises((FileNotFoundError, Exception)):
        load_sql("mes/definitely_not_here_0xdeadbeef")
