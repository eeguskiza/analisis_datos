"""SHIM - modelos ORM del schema ``nexo`` movidos a ``nexo.data.models_nexo``
en Plan 03-03 (DATA-04). Este modulo re-exporta los simbolos originales
para no romper consumidores (``nexo.services.auth``,
``api.routers.auditoria``, ``api.routers.usuarios``, ``tests/auth/*``).

Se elimina cuando todos los consumers migren sus imports a
``nexo.data.models_nexo`` (Mark-IV).
"""
from nexo.data.models_nexo import (  # noqa: F401
    NEXO_SCHEMA,
    NexoAuditLog,
    NexoBase,
    NexoDepartment,
    NexoLoginAttempt,
    NexoPermission,
    NexoRole,
    NexoSession,
    NexoUser,
    user_departments,
)
