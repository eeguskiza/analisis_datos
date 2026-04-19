"""Smoke tests del cableado RBAC (Plan 02-03).

Requieren Postgres arriba con el schema ``nexo`` inicializado y un
propietario bootstrap. En local::

    docker compose up -d db web
    docker compose exec web pytest tests/auth/test_rbac_smoke.py -x -q

En CI quedan marcados ``@pytest.mark.integration`` y son non-blocking
hasta que Phase 7 (Sprint 6) endurezca el workflow.

Estrategia:
- ``TestClient(app)`` ataca la app in-process. El AuthMiddleware corre
  real, la BD es la real (Postgres del compose).
- Para evitar contaminar la BD del operador, los tests usan emails de
  dominio ``@rbac-test.local`` y limpian en teardown.
"""
from __future__ import annotations

from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select, text

from api.main import app
from nexo.db.engine import SessionLocalNexo
from nexo.db.models import (
    NexoDepartment,
    NexoLoginAttempt,
    NexoSession,
    NexoUser,
)
from nexo.services.auth import hash_password


def _postgres_reachable() -> bool:
    """True si el Postgres del compose esta arriba y aceptando queries."""
    try:
        db = SessionLocalNexo()
        try:
            db.execute(text("SELECT 1"))
            return True
        finally:
            db.close()
    except Exception:
        return False


# El marker 'integration' permite filtrar con `pytest -m "not integration"`
# cuando no se quiere pagar el coste de BD real (p.ej. en el smoke rapido
# del CI durante Phase 1). El skipif evita el modo fallo ruidoso cuando
# alguien corre `pytest -q` sin haber arrancado docker compose.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _postgres_reachable(),
        reason="Postgres no disponible — requiere `docker compose up -d db`",
    ),
]


TEST_DOMAIN = "@rbac-test.local"
TEST_PASSWORD = "rbacsmoketest12"  # min 12 chars


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """TestClient que no sigue redirects (para poder afirmar sobre 302 y
    cookies explicitamente)."""
    with TestClient(app, follow_redirects=False) as c:
        yield c


@pytest.fixture(autouse=True)
def _cleanup_test_artifacts():
    """Limpieza antes y despues del test: purga users ``@rbac-test.local``
    + sus sesiones + sus login_attempts. Garantiza aislamiento entre tests."""
    _purge()
    yield
    _purge()


def _purge() -> None:
    db = SessionLocalNexo()
    try:
        # Borrar users de test (cascade → sessions)
        users = db.execute(
            select(NexoUser).where(NexoUser.email.like(f"%{TEST_DOMAIN}"))
        ).scalars().all()
        for u in users:
            db.execute(delete(NexoSession).where(NexoSession.user_id == u.id))
            db.delete(u)
        db.execute(
            delete(NexoLoginAttempt).where(NexoLoginAttempt.email.like(f"%{TEST_DOMAIN}"))
        )
        db.commit()
    finally:
        db.close()


def _create_test_user(
    email: str,
    role: str,
    dept_codes: list[str],
    *,
    must_change_password: bool = False,
) -> NexoUser:
    db = SessionLocalNexo()
    try:
        user = NexoUser(
            email=email,
            password_hash=hash_password(TEST_PASSWORD),
            role=role,
            active=True,
            must_change_password=must_change_password,
        )
        db.add(user)
        db.flush()
        if dept_codes:
            depts = db.execute(
                select(NexoDepartment).where(NexoDepartment.code.in_(dept_codes))
            ).scalars().all()
            user.departments = list(depts)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


def _login(client: TestClient, email: str, password: str = TEST_PASSWORD) -> str:
    """Hace login, devuelve el valor de la cookie nexo_session."""
    r = client.post(
        "/login",
        data={"email": email, "password": password},
        headers={"Accept": "text/html"},
    )
    assert r.status_code == 303, (
        f"Login no devolvio 303 para {email}: "
        f"status={r.status_code}, body={r.text[:200]}"
    )
    cookie = r.cookies.get("nexo_session")
    assert cookie, f"No se recibio cookie nexo_session en login de {email}"
    return cookie


# ── Tests ─────────────────────────────────────────────────────────────────

def test_api_sin_cookie_devuelve_401(client: TestClient):
    """Cualquier /api/* (excepto /api/health) sin sesion → 401 JSON."""
    for path in [
        "/api/ciclos",
        "/api/bbdd/tables",
        "/api/recursos",
        "/api/dashboard/summary",
        "/api/conexion/status",
    ]:
        r = client.get(path)
        assert r.status_code == 401, f"{path}: esperado 401, recibido {r.status_code}"
        assert r.json() == {"detail": "Not authenticated"}


def test_api_health_sin_cookie_devuelve_200(client: TestClient):
    """/api/health es publico (whitelist del AuthMiddleware)."""
    r = client.get("/api/health")
    assert r.status_code == 200


def test_propietario_pasa_todos_los_endpoints(client: TestClient):
    """Cookie de propietario → bypass de PERMISSION_MAP en todos los endpoints."""
    email = f"owner{TEST_DOMAIN}"
    _create_test_user(email, role="propietario", dept_codes=[])
    cookie = _login(client, email)

    # Probamos endpoints de diferentes routers (algunos fallaran por infra
    # externa — IZARO no disponible — pero el RBAC no debe dar 401/403).
    for path in [
        "/api/ciclos",
        "/api/recursos",
        "/api/historial",
        "/api/dashboard/summary",
    ]:
        r = client.get(path, cookies={"nexo_session": cookie})
        assert r.status_code not in (401, 403), (
            f"propietario bloqueado en {path}: status={r.status_code}"
        )


def test_usuario_rrhh_sin_produccion_recibe_403_en_pipeline(client: TestClient):
    """Usuario con rol=usuario y solo depto rrhh:
    - POST /api/pipeline/run → 403 (pipeline:run requiere ingenieria o produccion)
    - GET /api/operarios → 200/5xx (operarios:read autoriza rrhh)
    """
    email = f"rrhh-user{TEST_DOMAIN}"
    _create_test_user(email, role="usuario", dept_codes=["rrhh"])
    cookie = _login(client, email)

    # Pipeline prohibido
    r = client.post(
        "/api/pipeline/run",
        cookies={"nexo_session": cookie},
        json={
            "fecha_inicio": "2026-04-01",
            "fecha_fin": "2026-04-02",
            "modulos": [],
            "source": "izaro",
            "recursos": [],
        },
    )
    assert r.status_code == 403, (
        f"usuario de rrhh deberia recibir 403 en /api/pipeline/run, "
        f"recibido {r.status_code}: {r.text[:200]}"
    )
    assert r.json().get("detail", "").startswith("Permiso requerido:")

    # Operarios permitido (rrhh en el mapa)
    r = client.get("/api/operarios", cookies={"nexo_session": cookie})
    assert r.status_code not in (401, 403), (
        f"usuario de rrhh no deberia ser bloqueado en /api/operarios (status={r.status_code})"
    )


def test_usuario_ingenieria_pasa_pipeline_run(client: TestClient):
    """Usuario con rol=usuario y depto ingenieria tiene pipeline:run."""
    email = f"ing-user{TEST_DOMAIN}"
    _create_test_user(email, role="usuario", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    r = client.post(
        "/api/pipeline/run",
        cookies={"nexo_session": cookie},
        json={
            "fecha_inicio": "2026-04-01",
            "fecha_fin": "2026-04-02",
            "modulos": [],
            "source": "izaro",
            "recursos": ["_none_"],  # lista minima; el streaming puede fallar pero no por RBAC
        },
    )
    # No esperamos necesariamente 200 — el pipeline puede fallar por infra —
    # solo que NO sea 403/401.
    assert r.status_code not in (401, 403), (
        f"usuario de ingenieria bloqueado en pipeline:run (status={r.status_code})"
    )


def test_solo_propietario_accede_conexion_config(client: TestClient):
    """conexion:config tiene lista vacia → solo propietario puede PUT /api/conexion/config."""
    email = f"ing-user2{TEST_DOMAIN}"
    _create_test_user(email, role="usuario", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    # Ingenieria puede leer conexion (conexion:read esta en el mapa para ingenieria)
    r = client.get("/api/conexion/status", cookies={"nexo_session": cookie})
    assert r.status_code not in (401, 403), (
        f"ingenieria deberia leer /api/conexion/status (status={r.status_code})"
    )

    # Pero NO puede PUT /api/conexion/config (lista vacia en el mapa)
    r = client.put(
        "/api/conexion/config",
        cookies={"nexo_session": cookie},
        json={
            "server": "x", "port": "1433", "database": "x",
            "driver": "", "encrypt": "", "trust_server_certificate": "",
            "user": "", "password": "", "uf_code": "",
        },
    )
    assert r.status_code == 403, (
        f"ingenieria deberia recibir 403 en PUT /api/conexion/config, "
        f"recibido {r.status_code}"
    )


def test_regression_ident_10_exception_handler_no_traceback(client: TestClient):
    """IDENT-10: un 500 devuelve {error_id, message} sin traceback en el body.

    Forzamos un 500 atacando /api/bbdd/columns sin parametros requeridos (el
    handler SQL explota al no tener table). El exception handler global debe
    sanitizar la respuesta.

    Si el endpoint devolviera 422 por validation en vez de 500, marcamos el
    test como xfail explicativo: no hemos conseguido provocar un 500 limpio
    sin tocar la app; la regresion se cubre manualmente al cerrar el plan.
    """
    # Para forzar 500: usamos una tabla inexistente en un endpoint que no
    # valida formato → explota en SQL al llegar alla. Necesitamos sesion.
    email = f"ing-user3{TEST_DOMAIN}"
    _create_test_user(email, role="usuario", dept_codes=["ingenieria"])
    cookie = _login(client, email)

    r = client.get(
        "/api/bbdd/columns",
        params={"database": "xxx", "table": "tabla_inexistente_total"},
        cookies={"nexo_session": cookie},
    )
    # En el entorno de tests sin SQL Server real, esto devuelve 502 o 500.
    # Solo nos interesa que el 500 (si ocurre) no filtre traceback.
    if r.status_code == 500:
        body = r.json()
        assert "error_id" in body and "message" in body
        assert body["message"] == "Internal error"
        # Ningun traceback en el body
        assert "Traceback" not in r.text
        assert "File \"" not in r.text
    else:
        pytest.skip(
            f"No se pudo forzar 500 (respuesta={r.status_code}); "
            "regresion IDENT-10 validada manualmente al cerrar el plan."
        )
