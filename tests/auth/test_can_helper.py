"""Unit tests for ``nexo.services.auth.can`` (Plan 05-01).

``can(user, permission) -> bool`` es la fuente de verdad RBAC (D-03).
Esta suite cubre todas las ramas del helper sin TestClient ni DB:

- Entrada ``None`` → False.
- Rol ``propietario`` → True siempre (incluso con mapa vacío o permiso
  desconocido).
- Intersección ``user.departments ∩ PERMISSION_MAP[permission]``.
- Permiso desconocido (``PERMISSION_MAP.get`` devuelve ``[]``) → False.
- Lista vacía (propietario-only en el mapa) → False para cualquier no-propietario.
- Usuario multi-departamento: basta una intersección.

Los "usuarios" son stubs ligeros (``SimpleNamespace`` con ``role`` y
``departments``, una lista de ``_FakeDept(code=...)``). ``can`` nunca
toca la BD ni serializa; no hace falta fixtures pesadas.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from nexo.services.auth import PERMISSION_MAP, can

# Nota: NO se marca ``pytestmark = pytest.mark.unit`` porque el repo sólo
# registra el marker ``integration`` en ``tests/conftest.py`` (todo lo que
# no sea integration es unit por defecto). Añadir ``unit`` generaría
# ``PytestUnknownMarkWarning`` sin aportar valor.


# ── Builders ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _FakeDept:
    """Stub inmutable de ``NexoDepartment``. Solo necesitamos ``code``."""

    code: str


def _make_user(role: str, dept_codes: list[str]) -> SimpleNamespace:
    """Devuelve un stub duck-typed de ``NexoUser`` válido para ``can``.

    ``can`` solo lee ``user.role`` y ``user.departments[*].code``; no
    necesita un ORM real. Usamos ``SimpleNamespace`` porque ``can`` no
    depende del tipo concreto.
    """
    return SimpleNamespace(
        role=role,
        departments=[_FakeDept(code=c) for c in dept_codes],
    )


# ── Rama 1: user is None ──────────────────────────────────────────────────

def test_can_none_user_returns_false():
    """Usuario anónimo nunca tiene permisos, sea cual sea el permiso."""
    assert can(None, "pipeline:run") is False
    assert can(None, "operarios:read") is False
    assert can(None, "permiso:inexistente") is False


# ── Rama 2: propietario bypass ────────────────────────────────────────────

def test_can_propietario_bypass():
    """Propietario pasa incluso sin departamentos asignados."""
    user = _make_user(role="propietario", dept_codes=[])
    assert can(user, "ajustes:manage") is True
    assert can(user, "pipeline:run") is True
    assert can(user, "operarios:read") is True


def test_can_propietario_ignores_unknown_perm():
    """Propietario no consulta el mapa — permiso desconocido → True."""
    user = _make_user(role="propietario", dept_codes=[])
    assert can(user, "permiso:no_registrado") is True
    assert can(user, "") is True


# ── Rama 3a: intersección positiva ────────────────────────────────────────

@pytest.mark.parametrize(
    "dept_codes,permission",
    [
        (["ingenieria"], "pipeline:run"),
        (["produccion"], "pipeline:run"),
        (["ingenieria"], "pipeline:read"),
        (["rrhh"], "operarios:read"),
        (["rrhh"], "operarios:export"),
        (["comercial"], "capacidad:read"),
        (["gerencia"], "historial:read"),
        (["ingenieria"], "bbdd:read"),
    ],
)
def test_can_dept_intersection_true(dept_codes: list[str], permission: str):
    """Usuario con rol distinto a propietario: pasa si su depto ∈ mapa."""
    user = _make_user(role="usuario", dept_codes=dept_codes)
    assert can(user, permission) is True


def test_can_directivo_role_also_uses_dept_intersection():
    """Rol ``directivo`` NO tiene bypass — solo depende del depto."""
    user = _make_user(role="directivo", dept_codes=["ingenieria"])
    assert can(user, "pipeline:run") is True


# ── Rama 3b: intersección vacía ───────────────────────────────────────────

@pytest.mark.parametrize(
    "dept_codes,permission",
    [
        (["rrhh"], "pipeline:run"),
        (["comercial"], "recursos:edit"),
        (["produccion"], "operarios:read"),
        (["comercial"], "pipeline:run"),
        (["gerencia"], "recursos:edit"),
    ],
)
def test_can_dept_intersection_false(dept_codes: list[str], permission: str):
    """Usuario con depto fuera del mapa → False."""
    user = _make_user(role="usuario", dept_codes=dept_codes)
    assert can(user, permission) is False


# ── Rama 4: lista vacía = propietario-only ────────────────────────────────

@pytest.mark.parametrize(
    "permission",
    [
        "ajustes:manage",
        "auditoria:read",
        "usuarios:manage",
        "conexion:config",
        "aprobaciones:manage",
        "limites:manage",
        "rendimiento:read",
    ],
)
def test_can_empty_list_is_propietario_only(permission: str):
    """Permisos con lista vacía rechazan a cualquier no-propietario,
    incluso a un directivo de ingeniería o usuario de múltiples deptos."""
    all_depts = ["rrhh", "comercial", "ingenieria", "produccion", "gerencia"]
    user_directivo = _make_user(role="directivo", dept_codes=["ingenieria"])
    user_multi = _make_user(role="usuario", dept_codes=all_depts)
    assert can(user_directivo, permission) is False
    assert can(user_multi, permission) is False
    # Verificación cruzada: el permiso SÍ existe en el mapa con lista []
    assert PERMISSION_MAP.get(permission) == []


# ── Rama 5: permiso desconocido ───────────────────────────────────────────

def test_can_unknown_permission_returns_false():
    """``PERMISSION_MAP.get(permission, [])`` → []  → False."""
    user = _make_user(role="usuario", dept_codes=["ingenieria"])
    assert can(user, "permiso:no_registrado") is False
    assert can(user, "") is False
    assert can(user, "foo:bar") is False


# ── Rama 6: usuario multi-departamento ────────────────────────────────────

def test_can_multi_dept_user_needs_single_match():
    """Basta una intersección para autorizar; los demás deptos no importan."""
    user = _make_user(
        role="usuario",
        dept_codes=["rrhh", "ingenieria"],  # rrhh NO tiene pipeline:run; ingenieria sí
    )
    assert can(user, "pipeline:run") is True
    # Y sigue rechazando permisos que ninguno de sus deptos tiene
    assert can(user, "ajustes:manage") is False


def test_can_user_without_departments_fails_everything_but_owner_perms():
    """Usuario sin departamentos solo pasa si es propietario (bypass).
    Para cualquier otro rol sin deptos, todo debe ser False."""
    user = _make_user(role="usuario", dept_codes=[])
    for permission in PERMISSION_MAP:
        assert can(user, permission) is False, f"falló con {permission}"
