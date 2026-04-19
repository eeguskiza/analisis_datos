"""Paquete ``nexo.data`` — capa de datos de Nexo (Phase 3).

Contiene:

- ``engines``: factory de los 3 engines SQLAlchemy (nexo/app/mes).
- ``sql``: loader ``load_sql(name)`` y archivos ``.sql`` versionados.
- ``dto``: DTOs Pydantic ``frozen=True`` que cruzan la frontera HTTP.
- ``schema_guard``: validación al arrancar del schema ``nexo``.

Los repos concretos (``repositories/{mes,app,nexo}.py``) aterrizan en los
plans 03-02 y 03-03.
"""
