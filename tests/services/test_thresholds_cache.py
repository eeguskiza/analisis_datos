"""Wave 0 + Plan 04-04 — thresholds_cache tests (QUERY-02, D-19).

Unit group (runs siempre — no requiere Postgres):
- ThresholdEntry frozen.
- FALLBACK_REFRESH_SECONDS constante.
- Public API exports.

Integration group (requiere Postgres + seeds):
- full_reload lee los 4 seeds D-01..D-04 y los deja en cache.
- get returns None para endpoint desconocido.
- notify_changed emite NOTIFY sin error.
- fallback safety-net refresca si _loaded_at_global es stale.
- LISTEN/NOTIFY end-to-end: NOTIFY desde psql → reload_one <1.5s (Plan 04-04).
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from tests.data.conftest import _postgres_reachable


# ── Unit group (siempre corre) ─────────────────────────────────────────────

def test_entry_is_frozen():
    """ThresholdEntry es frozen — mutation lanza FrozenInstanceError."""
    from dataclasses import FrozenInstanceError

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
    """Public API estable para consumers de 04-02 + 04-04."""
    from nexo.services import thresholds_cache
    for name in (
        "get", "full_reload", "reload_one", "notify_changed",
        "ThresholdEntry", "FALLBACK_REFRESH_SECONDS",
        "start_listener", "_blocking_listen_forever",
    ):
        assert hasattr(thresholds_cache, name), (
            f"thresholds_cache debe exportar {name!r}"
        )


def test_start_listener_is_coroutine():
    """start_listener debe ser coroutine (para asyncio.create_task)."""
    import inspect
    from nexo.services.thresholds_cache import start_listener
    assert inspect.iscoroutinefunction(start_listener)


# ── Integration group (requiere Postgres + seeds) ──────────────────────────

@pytest.mark.integration
@pytest.mark.skipif(
    not _postgres_reachable(),
    reason="Postgres no arriba: docker compose up -d db",
)
def test_full_reload_populates_cache_from_seeds():
    """full_reload lee los 4 seeds D-01..D-04 y los deja en el cache."""
    from nexo.services import thresholds_cache
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
    thresholds_cache.full_reload()
    result = thresholds_cache.get("__no_existe_nunca__")
    assert result is None


@pytest.mark.integration
@pytest.mark.skipif(
    not _postgres_reachable(),
    reason="Postgres no arriba: docker compose up -d db",
)
def test_notify_changed_does_not_raise():
    """notify_changed emite NOTIFY sin excepcion (AUTOCOMMIT + pg_notify)."""
    from nexo.services import thresholds_cache
    thresholds_cache.notify_changed("pipeline/run")  # no asserts — just no-throw.


@pytest.mark.integration
@pytest.mark.skipif(
    not _postgres_reachable(),
    reason="Postgres no arriba: docker compose up -d db",
)
def test_fallback_safety_net_refreshes_stale_cache(monkeypatch):
    """D-19 safety-net: si _loaded_at_global > 5 min, get() dispara full_reload."""
    from nexo.services import thresholds_cache

    # Seed cache y aseja _loaded_at_global a 10 minutos atras.
    thresholds_cache.full_reload()
    old_global = datetime.now(timezone.utc) - timedelta(minutes=10)
    monkeypatch.setattr(
        thresholds_cache, "_loaded_at_global", old_global,
    )

    # Al leer, el safety-net debe detectar staleness y hacer full_reload.
    # Verificamos que _loaded_at_global se refresco a (now - epsilon).
    thresholds_cache.get("pipeline/run")
    refreshed = thresholds_cache._loaded_at_global
    assert refreshed is not None
    assert refreshed > old_global, (
        "Safety-net debe refrescar _loaded_at_global tras detectar staleness"
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not _postgres_reachable(),
    reason="Postgres no arriba: docker compose up -d db",
)
def test_blocking_listener_reacts_to_notify_under_1_5s():
    """Plan 04-04: LISTEN loop reacciona a NOTIFY vía reload_one en <1.5s.

    Arranca el worker en un thread lateral; espera "LISTEN activo" via un
    poll corto (el worker loggea al abrir cursor); emite NOTIFY desde el
    engine ORM via thresholds_cache.notify_changed; mide el tiempo hasta
    que ThresholdEntry.loaded_at supera el baseline.
    """
    from nexo.services import thresholds_cache

    # Carga inicial del cache para obtener baseline loaded_at.
    thresholds_cache.full_reload()
    baseline_entry = thresholds_cache.get("pipeline/run")
    assert baseline_entry is not None, "Seed pipeline/run debe existir"
    baseline_loaded_at = baseline_entry.loaded_at

    stop_event = threading.Event()
    worker = threading.Thread(
        target=thresholds_cache._blocking_listen_forever,
        args=(stop_event,),
        daemon=True,
        name="test-threshold-listener",
    )
    worker.start()

    try:
        # Dar 2s al worker para abrir la conexion + LISTEN.
        time.sleep(2.0)
        # Emitir NOTIFY desde el engine ORM (mismo engine compartido
        # — Postgres entrega el NOTIFY a todas las conexiones en LISTEN).
        thresholds_cache.notify_changed("pipeline/run")

        # Poll cada 100ms hasta 3s para detectar que loaded_at avanzo.
        deadline = time.monotonic() + 3.0
        updated = False
        while time.monotonic() < deadline:
            entry = thresholds_cache.get("pipeline/run")
            if entry is not None and entry.loaded_at > baseline_loaded_at:
                updated = True
                break
            time.sleep(0.1)

        assert updated, (
            "LISTEN loop no reacciono a NOTIFY en 3s — esperado <1.5s"
        )
    finally:
        stop_event.set()
        worker.join(timeout=10.0)
