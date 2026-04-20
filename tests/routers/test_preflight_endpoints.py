"""Wave 0 stub — preflight router contracts (QUERY-04, QUERY-07).

Implementacion de los endpoints ``/preflight`` + extension de
``pipeline/run``, ``bbdd/query``, ``capacidad``, ``operarios`` con
force/approval_id aterriza en Plan 04-02. Este archivo documenta los
contratos.

Planned tests:
  - test_preflight_returns_estimation_json: POST /pipeline/preflight
    devuelve Estimation JSON (endpoint/estimated_ms/level/reason/
    breakdown/factor_used_ms/warn_ms/block_ms).
  - test_preflight_level_green_amber_red: classifier en endpoint
    devuelve el level correcto segun umbrales cacheados.
  - test_run_without_force_on_amber_returns_428: HTTP 428 Precondition
    Required si level=amber y force=False (D-05 modal).
  - test_run_red_without_approval_returns_403: level=red sin
    approval_id valido → 403 (D-06).
  - test_run_red_with_valid_approval_executes: level=red + approval_id
    approved + user match + params match + consumed_at IS NULL → ejecuta
    + marca consumed (D-15).
  - test_preflight_scope: capacidad/operarios con rango <=90d NO
    disparan preflight (D-03).
"""
from __future__ import annotations

import pytest


pytest.skip(
    "Implemented in Plan 04-02 (preflight endpoints + force/approval_id).",
    allow_module_level=True,
)
