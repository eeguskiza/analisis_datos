"""Fixtures compartidos para tests OEE."""
import sys
from pathlib import Path

# Asegurar que el proyecto esta en el path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_configure(config):
    """Registra markers custom para evitar PytestUnknownMarkWarning."""
    config.addinivalue_line(
        "markers",
        "integration: tests que requieren infraestructura live (BD Postgres, "
        "servicios externos). Filtrables con `pytest -m 'not integration'`.",
    )
