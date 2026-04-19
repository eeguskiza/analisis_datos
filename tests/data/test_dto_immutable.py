"""DATA-08: DTOs ``frozen=True`` (Pydantic v2).

Los DTOs concretos (``*Row``) se definen en 03-02 y 03-03. Aquí
validamos que la configuración compartida ``ROW_CONFIG`` produce
inmutabilidad real (Pydantic v2 lanza ``ValidationError`` al intentar
mutar un campo).
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from nexo.data.dto.base import ROW_CONFIG


class _RowDummy(BaseModel):
    """DTO ad-hoc para validar la config compartida."""

    model_config = ROW_CONFIG
    id: int
    nombre: str


def test_frozen_dto_rejects_mutation():
    """Intentar mutar un campo en un DTO frozen → ``ValidationError``."""
    row = _RowDummy(id=1, nombre="test")
    with pytest.raises(ValidationError):
        row.id = 2


def test_frozen_dto_roundtrips_from_attributes():
    """``from_attributes=True`` permite hidratar desde objetos con atributos."""

    class _Src:
        id = 7
        nombre = "src"

    src = _Src()
    row = _RowDummy.model_validate(src)
    assert row.id == 7
    assert row.nombre == "src"
