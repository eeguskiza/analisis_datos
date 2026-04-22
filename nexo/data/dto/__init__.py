"""Paquete DTO — Pydantic ``frozen=True`` rows por dominio.

Los DTOs concretos (``*Row``) aterrizan en los plans 03-02, 03-03 y
04-01. En 03-01 se sento el paquete + ``base.py`` con la ConfigDict
compartida; 03-02/03-03 añadieron ``mes.py``, ``app.py`` y ``nexo.py``;
04-01 añade ``query.py`` con los DTOs de Phase 4 (query_log,
query_thresholds, query_approvals + Estimation).

Para imports ergonomicos los nombres del dominio "query" (Phase 4) se
re-exportan aqui. Los DTOs de phases anteriores siguen importandose
explicitamente desde su submodulo (patron vigente en repos/*.py y
tests/*.py).
"""

from nexo.data.dto.query import (  # noqa: F401
    Estimation,
    QueryApprovalRow,
    QueryLogRow,
    QueryThresholdRow,
)


__all__ = [
    "Estimation",
    "QueryApprovalRow",
    "QueryLogRow",
    "QueryThresholdRow",
]
