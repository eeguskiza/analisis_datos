"""Wave 0 — thresholds_cache tests (QUERY-02, D-19).

Unit group (runs en 04-01): ThresholdEntry frozen + get para endpoint
unknown + full_reload contra DB seed.

Integration group (skip en 04-01; implementado en Plan 04-04):
LISTEN/NOTIFY roundtrip debe propagar cambios en <1s.
"""
from __future__ import annotations

import pytest

from tests.data.conftest import _postgres_reachable


# ── Unit group (siempre corre) ─────────────────────────────────────────────

def test_entry_is_frozen():
    """ThresholdEntry es frozen — mutation lanza FrozenInstanceError."""
    from dataclasses import FrozenInstanceError
    from datetime import datetime, timezone

    from nexo.services.thresholds_cache import ThresholdEntry

    entry = ThresholdEntry(
        endpoint="pipeline/run",
        warn_ms=120_000,
        block_ms=600_000,
        factor_ms=2000.0,
        loaded_at=datetime.now(timezone.utc),
    )
    with pytest.raises(FrozenInstanceError):
        entry.warn_ms = 99  # type: ignore[misc]


def test_fallback_refresh_seconds_is_300():
    """Safety-net timeout de 5 min (D-19)."""
    from nexo.services.thresholds_cache import FALLBACK_REFRESH_SECONDS
    assert FALLBACK_REFRESH_SECONDS == 300


def test_module_exports_public_api():
    """Public API estable para consumers de 04-02."""
    from nexo.services import thresholds_cache
    for name in ("get", "full_reload", "reload_one", "notify_changed",
                 "ThresholdEntry", "FALLBACK_REFRESH_SECONDS"):
        assert hasattr(thresholds_cache, name), (
            f"thresholds_cache debe exportar {name!r}"
        )


# ── Integration group (requiere Postgres + seeds) ──────────────────────────

@pytest.mark.integration
@pytest.mark.skipif(
    not _postgres_reachable(),
    reason="Postgres no arriba: docker compose up -d db",
)
def test_full_reload_populates_cache_from_seeds():
    """full_reload lee los 4 seeds D-01..D-04 y los deja en el cache."""
    from nexo.services import thresholds_cache
    # Reset state y recarga
    with thresholds_cache._cache_lock:
        thresholds_cache._cache.clear()
    thresholds_cache.full_reload()

    entry = thresholds_cache.get("pipeline/run")
    assert entry is not None, "El seed pipeline/run debe estar presente"
    assert entry.warn_ms == 120_000
    assert entry.block_ms == 600_000
    assert entry.factor_ms == 2000.0


@pytest.mark.integration
@pytest.mark.skipif(
    not _postgres_reachable(),
    reason="Postgres no arriba: docker compose up -d db",
)
def test_get_returns_none_for_unknown_endpoint():
    """Endpoints sin fila en query_thresholds → None."""
    from nexo.services import thresholds_cache
    thresholds_cache.full_reload()  # asegura cache cargado
    result = thresholds_cache.get("__no_existe_nunca__")
    assert result is None


# ── Deferred to Plan 04-04 ─────────────────────────────────────────────────

@pytest.mark.skip(reason="LISTEN/NOTIFY roundtrip implementado en Plan 04-04")
def test_notify_triggers_reload_under_1s():
    """Edit threshold → NOTIFY → LISTEN → cache refreshed en <1s (D-19)."""
    raise NotImplementedError("Plan 04-04")
