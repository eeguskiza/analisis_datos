"""Wave 0 stub — pipeline lock tests (QUERY-08, D-18).

Implementacion del servicio ``nexo/services/pipeline_lock.py``
(asyncio.Semaphore + asyncio.to_thread + timeout) aterriza en Plan
04-02. Este archivo documenta los contratos.

Planned tests:
  - test_semaphore_limits_to_max: 4 invocaciones concurrentes con
    max_concurrent=3 → la 4a espera (D-18 semaforo global).
  - test_timeout_raises_504: pipeline que excede NEXO_PIPELINE_TIMEOUT_SEC
    cancela el thread + escribe query_log con status='timeout'.
  - test_to_thread_does_not_block_event_loop: confirmar que la UI sigue
    respondiendo durante run_pipeline_sync (Success Criterion #5).
  - test_timeout_records_actual_ms_as_timeout_ms: fila de timeout en
    query_log tiene actual_ms = timeout_ms exacto.
"""
from __future__ import annotations

import pytest


pytest.skip(
    "Implemented in Plan 04-02 (pipeline_lock + asyncio.to_thread).",
    allow_module_level=True,
)
