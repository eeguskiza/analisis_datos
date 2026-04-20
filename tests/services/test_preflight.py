"""Preflight service unit tests (Plan 04-02, QUERY-03, QUERY-04).

Suite puramente unitaria — NO toca Postgres. ``thresholds_cache.get`` se
monkeypatch-ea para devolver ``ThresholdEntry`` sintéticos con los
umbrales D-01 (pipeline), D-02 (bbdd), D-03 (capacidad/operarios).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from nexo.services import preflight, thresholds_cache
from nexo.services.thresholds_cache import ThresholdEntry


# ── Fixtures / helpers ─────────────────────────────────────────────────

def _entry(
    endpoint: str,
    *,
    warn_ms: int,
    block_ms: int,
    factor_ms: float | None,
) -> ThresholdEntry:
    return ThresholdEntry(
        endpoint=endpoint,
        warn_ms=warn_ms,
        block_ms=block_ms,
        factor_ms=factor_ms,
        loaded_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_thresholds(monkeypatch):
    """Monkeypatcha ``thresholds_cache.get`` para devolver umbrales seed.

    Mapea endpoint → ``ThresholdEntry`` con los valores D-01/D-02/D-03.
    Tests individuales pueden sobreescribir entradas via ``entries[key] = ...``
    antes de llamar a ``estimate_cost``.
    """
    entries: dict[str, ThresholdEntry] = {
        "pipeline/run": _entry(
            "pipeline/run", warn_ms=120_000, block_ms=600_000, factor_ms=2000.0
        ),
        "bbdd/query": _entry(
            "bbdd/query", warn_ms=3_000, block_ms=30_000, factor_ms=1000.0
        ),
        "capacidad": _entry(
            "capacidad", warn_ms=3_000, block_ms=30_000, factor_ms=50.0
        ),
        "operarios": _entry(
            "operarios", warn_ms=3_000, block_ms=30_000, factor_ms=50.0
        ),
    }

    def _fake_get(endpoint: str):
        return entries.get(endpoint)

    monkeypatch.setattr(thresholds_cache, "get", _fake_get)
    return entries


# ── pipeline/run ───────────────────────────────────────────────────────

def test_estimate_cost_pipeline_green(mock_thresholds):
    """2 recursos × 5 días × 2000ms = 20_000ms < warn 120_000 → green."""
    est = preflight.estimate_cost(
        "pipeline/run", {"n_recursos": 2, "n_dias": 5}
    )
    assert est.estimated_ms == 20_000
    assert est.level == "green"
    assert est.factor_used_ms == 2000.0
    assert "2 recursos" in est.breakdown
    assert est.warn_ms == 120_000
    assert est.block_ms == 600_000


def test_estimate_cost_pipeline_amber(mock_thresholds):
    """10 recursos × 10 días × 2000ms = 200_000ms → amber (entre 120k y 600k)."""
    est = preflight.estimate_cost(
        "pipeline/run", {"n_recursos": 10, "n_dias": 10}
    )
    assert est.estimated_ms == 200_000
    assert est.level == "amber"


def test_estimate_cost_pipeline_red(mock_thresholds):
    """10 recursos × 30 días × 2000ms = 600_000ms == block → red (strict)."""
    est = preflight.estimate_cost(
        "pipeline/run", {"n_recursos": 10, "n_dias": 30}
    )
    assert est.estimated_ms == 600_000
    assert est.level == "red"
    assert "10 min" in est.reason or "aprob" in est.reason.lower() or "Excede" in est.reason


def test_estimate_cost_pipeline_from_recursos_list(mock_thresholds):
    """n_recursos se deriva de len(recursos) cuando no viene explícito."""
    est = preflight.estimate_cost(
        "pipeline/run", {"recursos": ["r1", "r2", "r3"], "n_dias": 10}
    )
    assert est.estimated_ms == 60_000  # 3 × 10 × 2000


# ── bbdd/query ─────────────────────────────────────────────────────────

def test_estimate_cost_bbdd_green_default(mock_thresholds):
    """Baseline 1000ms < warn 3000 → green."""
    est = preflight.estimate_cost(
        "bbdd/query", {"sql": "SELECT TOP 10 * FROM t", "database": "dbizaro"}
    )
    assert est.estimated_ms == 1000
    assert est.level == "green"
    assert "baseline" in est.breakdown


def test_estimate_cost_bbdd_amber_with_high_factor(mock_thresholds):
    """Si factor se calibra a 5000ms > warn 3000 → amber."""
    mock_thresholds["bbdd/query"] = _entry(
        "bbdd/query", warn_ms=3_000, block_ms=30_000, factor_ms=5000.0
    )
    est = preflight.estimate_cost("bbdd/query", {"sql": "x", "database": "y"})
    assert est.estimated_ms == 5000
    assert est.level == "amber"


# ── capacidad / operarios (rango) ──────────────────────────────────────

def test_estimate_cost_rango_short_skips_preflight(mock_thresholds):
    """rango_dias <= 90 → green con razón 'rango <=90d'."""
    est = preflight.estimate_cost("capacidad", {"rango_dias": 30})
    assert est.level == "green"
    assert "<=90d" in est.reason or "90d" in est.reason


def test_estimate_cost_rango_long_amber(mock_thresholds):
    """180 días × 50ms = 9000ms. warn 3000 < 9000 < block 30000 → amber."""
    est = preflight.estimate_cost("operarios", {"rango_dias": 180})
    assert est.estimated_ms == 9_000
    assert est.level == "amber"
    assert "días" in est.breakdown


def test_estimate_cost_rango_long_red(mock_thresholds):
    """1000 días × 50ms = 50000ms > block 30000 → red."""
    est = preflight.estimate_cost("capacidad", {"rango_dias": 1000})
    assert est.estimated_ms == 50_000
    assert est.level == "red"


def test_estimate_cost_rango_derives_from_dates(mock_thresholds):
    """Si rango_dias no viene, se deriva de fecha_desde/fecha_hasta."""
    est = preflight.estimate_cost(
        "capacidad",
        {"fecha_desde": "2025-01-01", "fecha_hasta": "2025-06-30"},
    )
    # 2025-01-01 → 2025-06-30 = 181 días (+1 inclusive)
    # 181 × 50 = 9050 → amber
    assert est.estimated_ms == 9050
    assert est.level == "amber"


# ── Defensive paths ────────────────────────────────────────────────────

def test_estimate_cost_unknown_endpoint_defensive_green(mock_thresholds):
    """Endpoint no soportado → green defensivo (no bloqueamos por typo)."""
    est = preflight.estimate_cost("foobar", {})
    assert est.level == "green"
    assert est.estimated_ms == 0


def test_estimate_cost_threshold_missing_defensive_green(monkeypatch):
    """thresholds_cache.get devuelve None → green defensivo."""
    monkeypatch.setattr(thresholds_cache, "get", lambda e: None)
    est = preflight.estimate_cost("pipeline/run", {"n_recursos": 100})
    assert est.level == "green"
    assert "no configurado" in est.reason


def test_estimate_cost_pipeline_factor_fallback(mock_thresholds):
    """Si factor_ms es None, usa fallback 2000ms (D-04 seed)."""
    mock_thresholds["pipeline/run"] = _entry(
        "pipeline/run", warn_ms=120_000, block_ms=600_000, factor_ms=None
    )
    est = preflight.estimate_cost(
        "pipeline/run", {"n_recursos": 2, "n_dias": 5}
    )
    assert est.estimated_ms == 20_000  # 2×5×2000
    assert est.factor_used_ms == 2000.0


# ── Future: factor_recalc (skipped, tracked for Plan 04-04) ──────────

@pytest.mark.skip(reason="Implemented in Plan 04-04 (recalibrate button)")
def test_factor_recalc_skipped():
    """Recálculo manual del factor_ms (D-04) aterriza en Plan 04-04."""
    pass
