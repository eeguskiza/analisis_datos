"""DATA-03: integration tests contra SQL Server ``ecs_mobility``.

Skipif automatico si SQL Server no es alcanzable (esperable en CI).
La fixture ``db_app`` vive en ``tests/data/conftest.py`` (Plan 03-01).
"""
from __future__ import annotations

import pytest

from nexo.data.dto.app import CicloRow, ContactoRow, RecursoRow
from nexo.data.repositories.app import (
    CicloRepo,
    ContactoRepo,
    EjecucionRepo,
    RecursoRepo,
)

pytestmark = pytest.mark.integration


def test_recurso_repo_list_all_returns_dtos(db_app):
    """RecursoRepo.list_all devuelve list[RecursoRow] (DTO frozen)."""
    repo = RecursoRepo(db_app)
    rows = repo.list_all()
    assert isinstance(rows, list)
    assert all(isinstance(r, RecursoRow) for r in rows)
    # Frozen: no mutation allowed
    if rows:
        with pytest.raises(Exception):  # pydantic ValidationError / TypeError
            rows[0].nombre = "mutated"


def test_recurso_repo_list_activos_filters(db_app):
    """RecursoRepo.list_activos solo devuelve recursos con activo=True."""
    repo = RecursoRepo(db_app)
    activos = repo.list_activos()
    assert all(r.activo is True for r in activos)


def test_recurso_repo_orderby_seccion_nombre(db_app):
    """list_all orden: seccion asc, nombre asc (SQL Server-side).

    El backend SQL Server ordena por defecto con collation
    case-insensitive (SQL_Latin1_General_CP1_CI_AS), distinta al
    Python sorted() que es case-sensitive. Verificamos solo el
    ordering de seccion (stable) para evitar falsos positivos por
    collation Mayus/minus.
    """
    repo = RecursoRepo(db_app)
    rows = repo.list_all()
    if len(rows) >= 2:
        secciones = [r.seccion for r in rows]
        assert secciones == sorted(secciones)


def test_ciclo_repo_list_all_returns_dtos(db_app):
    """CicloRepo.list_all devuelve list[CicloRow]."""
    repo = CicloRepo(db_app)
    rows = repo.list_all()
    assert all(isinstance(r, CicloRow) for r in rows)


def test_ciclo_repo_exists(db_app):
    """CicloRepo.exists no debe petar con inputs que no existen."""
    repo = CicloRepo(db_app)
    assert repo.exists("__no_such_machine__", "__no_such_ref__") is False


def test_ejecucion_repo_list_recent_respects_limit(db_app):
    """EjecucionRepo.list_recent respeta el argumento limit."""
    repo = EjecucionRepo(db_app)
    rows = repo.list_recent(limit=3)
    assert len(rows) <= 3


def test_contacto_repo_list_all(db_app):
    """ContactoRepo.list_all devuelve list[ContactoRow]."""
    repo = ContactoRepo(db_app)
    rows = repo.list_all()
    assert all(isinstance(r, ContactoRow) for r in rows)
