"""Paquete de repositorios Nexo (acceso a datos encapsulado).

Cada sub-modulo expone una clase ``*Repository`` que recibe un
``Engine`` o ``Session`` en el constructor y expone metodos de negocio
(no de fila) — ver ``mes.py`` como referencia canonica.

Contrato minimo:

- El repo NO gestiona transacciones (caller ``commit()``/``rollback()``).
- El repo devuelve DTOs frozen (``nexo/data/dto/``) o ``dict`` / tuplas
  cuando la semantica legacy lo requiere (p. ej. wrappers sobre
  ``OEE/db/connector.py``).
"""
