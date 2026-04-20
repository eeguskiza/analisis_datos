"""Wave 0 stub — preflight service tests (QUERY-03, QUERY-04).

Implementacion del servicio ``nexo/services/preflight.py`` aterriza en
Plan 04-02. Este archivo documenta los contratos que los tests
implementaran entonces.

Planned tests:
  - test_pipeline_classify_green: estimated_ms < warn_ms → level='green'.
  - test_pipeline_classify_amber: warn_ms <= estimated_ms < block_ms → 'amber'.
  - test_pipeline_classify_red:   estimated_ms >= block_ms → 'red'.
  - test_bbdd_classify_green/amber/red: mismo patron con umbrales D-02.
  - test_capacidad_skipped_when_rango_le_90d: rango <= 90d no dispara preflight.
  - test_factor_recalc_median: recalculo factor desde ultimos 30 runs (D-04).
  - test_estimation_contains_breakdown: breakdown string para UI modal (D-07).
"""
from __future__ import annotations

import pytest


pytest.skip(
    "Implemented in Plan 04-02 (preflight service + heuristicas).",
    allow_module_level=True,
)
