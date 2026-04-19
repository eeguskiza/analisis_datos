"""SHIM — movido a ``nexo/data/engines.py`` en Plan 03-01.

Durante Phase 3 este módulo re-exporta los símbolos originales para no
romper consumidores Phase 2 (``nexo/services/auth.py``,
``api/routers/auditoria.py``, ``api/routers/usuarios.py``,
``tests/auth/*``). Se elimina al final de Plan 03-03, una vez los
consumers migren sus imports a ``nexo.data.engines``.
"""
from nexo.data.engines import SessionLocalNexo, engine_nexo  # noqa: F401
