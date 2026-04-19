"""DATA-07: smoke HTTP sobre los 10 routers.

Requiere cookie valida (propietario real en BD). Skipea si no hay
``NEXO_TEST_EMAIL`` / ``NEXO_TEST_PASSWORD`` en env. En ese caso, los
meta-tests (``test_no_raw_pyodbc_in_routers``, identity checks) cubren
DATA-07 a nivel codigo.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client_with_cookie():
    from api.main import app

    c = TestClient(app, follow_redirects=False)
    email = os.environ.get("NEXO_TEST_EMAIL")
    password = os.environ.get("NEXO_TEST_PASSWORD")
    if not email or not password:
        pytest.skip(
            "NEXO_TEST_EMAIL / NEXO_TEST_PASSWORD no definidos - "
            "smoke HTTP requiere un propietario real."
        )
    r = c.post("/login", data={"email": email, "password": password})
    assert r.status_code in (302, 303), f"login fallo: {r.status_code}"
    return c


# path, expected_primary_status (algunos routers pueden devolver 502 si
# el MES no es alcanzable - eso no es fallo del refactor, es transporte).
ROUTERS_OK = [
    ("/api/recursos", 200),
    ("/api/ciclos", 200),
    ("/api/historial?limit=5", 200),
    ("/api/operarios", 200),
    ("/api/bbdd/databases", 200),
    ("/ajustes", 200),
    ("/ajustes/usuarios", 200),
    ("/ajustes/auditoria", 200),
    ("/api/health", 200),
    ("/api/luk4", 200),
]


@pytest.mark.parametrize("path,expected", ROUTERS_OK)
def test_router_returns_expected_status(client_with_cookie, path, expected):
    """Smoke HTTP: cada router responde 200 (o 502 si MES no alcanzable)."""
    r = client_with_cookie.get(path)
    assert r.status_code in (expected, 502, 503), (
        f"{path} -> {r.status_code} (esperado {expected} o 502/503)"
    )
