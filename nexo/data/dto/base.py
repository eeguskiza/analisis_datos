"""Base compartida para DTOs ``*Row`` (Pydantic v2, frozen + from_attributes).

Los DTOs concretos se definen en 03-02 / 03-03. Aquí exponemos la
``ConfigDict`` canónica para que cada archivo (``app.py``, ``mes.py``,
``nexo.py``) la reuse en lugar de duplicar la configuración.

Uso::

    from pydantic import BaseModel
    from nexo.data.dto.base import ROW_CONFIG

    class RecursoRow(BaseModel):
        model_config = ROW_CONFIG
        id: int
        nombre: str
"""

from __future__ import annotations

from pydantic import ConfigDict


# ``frozen=True``  → inmutabilidad (mutation lanza ValidationError).
# ``from_attributes=True`` → permite ``Row.model_validate(orm_entity)``.
ROW_CONFIG: ConfigDict = ConfigDict(frozen=True, from_attributes=True)


__all__ = ["ROW_CONFIG"]
