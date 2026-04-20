"""Fixtures compartidos para tests OEE."""
import sys
from pathlib import Path

import pytest

# Asegurar que el proyecto esta en el path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_configure(config):
    """Registra markers custom para evitar PytestUnknownMarkWarning."""
    config.addinivalue_line(
        "markers",
        "integration: tests que requieren infraestructura live (BD Postgres, "
        "servicios externos). Filtrables con `pytest -m 'not integration'`.",
    )


# ── Phase 4 / Plan 04-01 — thresholds_cache fixtures ──────────────────────

@pytest.fixture
def thresholds_cache_empty():
    """Resetea el cache global de ``nexo.services.thresholds_cache``.

    Util para tests que quieren asegurar que el cache arranca vacio o
    que una llamada a ``get`` fuerza ``full_reload`` por safety-net.

    Yield: el modulo thresholds_cache con cache vaciado.
    Teardown: deja el cache vaciado; el proximo ``get`` en otro test
    volvera a recargar via safety-net.
    """
    from nexo.services import thresholds_cache

    with thresholds_cache._cache_lock:
        thresholds_cache._cache.clear()
    # Reset global loaded_at_global — la unica forma sin referencia
    # es asignar al atributo modulo.
    thresholds_cache._loaded_at_global = None
    yield thresholds_cache
    with thresholds_cache._cache_lock:
        thresholds_cache._cache.clear()
    thresholds_cache._loaded_at_global = None
