"""Wave 0 stub — query_timing middleware tests (QUERY-05, D-17).

Implementacion del middleware ``nexo/middleware/query_timing.py``
aterriza en Plan 04-02. Este archivo documenta los contratos.

Planned tests:
  - test_writes_query_log_on_timed_path: request a /pipeline/run
    escribe fila en query_log con estimated_ms + actual_ms.
  - test_slow_status_written_when_actual_exceeds_warn_1_5x: si
    actual_ms > warn_ms * 1.5 → status='slow' + log.warning (D-17).
  - test_excluded_paths_bypass: /login, /static, /favicon NO escriben
    query_log (no hay preflight configurado; middleware skips).
  - test_error_status_on_exception: cuando el endpoint lanza,
    query_log graba status='error' + actual_ms hasta la excepcion.
  - test_middleware_sees_user_id_from_auth: el middleware se monta
    despues de auth → tiene request.state.user para poblar user_id.
"""
from __future__ import annotations

import pytest


pytest.skip(
    "Implemented in Plan 04-02 (query_timing middleware).",
    allow_module_level=True,
)
