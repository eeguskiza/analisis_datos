"""In-memory cache de ``nexo.query_thresholds`` (Phase 4 — Plan 04-01).

D-19 ("cache + invalidate on edit") se implementa en dos planes:

- **04-01 (este)**: skeleton con ``_cache`` dict + ``full_reload`` +
  ``reload_one`` + ``get`` con safety-net (re-lee BD si
  ``loaded_at_global`` esta stale > 5min). Tambien expone
  ``notify_changed(endpoint)`` para que el CRUD endpoint de
  thresholds (Plan 04-04) pueda emitir NOTIFY — la emision de NOTIFY
  vive aqui porque la interfaz quedara estable across plans.
- **04-04 (siguiente)**: ``listen_loop`` + ``start_listener`` con un
  thread que hace ``LISTEN nexo_thresholds_changed`` y reacciona en <1s
  a los NOTIFY enviados desde el CRUD.

Contrato estable de 04-01 hacia 04-02:
  ``thresholds_cache.get(endpoint) -> Optional[ThresholdEntry]``

Los preflight services (Plan 04-02) consumen solo ``get`` — no dependen
del mecanismo de invalidacion.

Concurrencia:
- Reader path (`get`): sync, llamado desde middleware + routers. Usa
  ``threading.Lock`` (no ``asyncio.Lock``) para proteger el dict.
- Refresh path (``full_reload`` / ``reload_one``): sync, abre Session
  propia. Thread-safe.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Optional

from sqlalchemy import text

from nexo.data.engines import SessionLocalNexo, engine_nexo
from nexo.data.repositories.nexo import ThresholdRepo


log = logging.getLogger("nexo.thresholds_cache")


# D-19 safety net: si ``loaded_at_global`` se queda atras > 5 min
# (ej. porque LISTEN fallo silenciosamente), el proximo ``get`` fuerza
# un ``full_reload`` sincrono. Trade-off: latencia +100ms cada 5 min en
# el peor caso, a cambio de coherencia garantizada sin listener.
FALLBACK_REFRESH_SECONDS = 300


@dataclass(frozen=True)
class ThresholdEntry:
    """Snapshot inmutable de una fila de ``nexo.query_thresholds``.

    ``loaded_at`` es el timestamp del ultimo refresh de ESTA fila;
    permite a los consumers detectar staleness individual (no usado
    todavia en 04-01; util en 04-04 para debugging del listener).
    """

    endpoint: str
    warn_ms: int
    block_ms: int
    factor_ms: Optional[float]
    loaded_at: datetime


# Module-level state — protegido por ``_cache_lock``.
_cache: dict[str, ThresholdEntry] = {}
_cache_lock: Lock = Lock()
_loaded_at_global: Optional[datetime] = None


def full_reload() -> None:
    """Recarga TODAS las filas de ``nexo.query_thresholds`` a memoria.

    Llamado:
      - Al arrancar la app (``lifespan`` — Plan 04-02 lo cablea).
      - Por ``get`` como safety-net si ``_loaded_at_global`` es stale.
      - Por el job de recovery si el listener se cae (Plan 04-04).

    Thread-safe. Abre una Session propia y la cierra explicitamente.
    """
    global _loaded_at_global
    now = datetime.now(timezone.utc)
    session = SessionLocalNexo()
    try:
        rows = ThresholdRepo(session).list_all()
    finally:
        session.close()

    with _cache_lock:
        _cache.clear()
        for row in rows:
            _cache[row.endpoint] = ThresholdEntry(
                endpoint=row.endpoint,
                warn_ms=row.warn_ms,
                block_ms=row.block_ms,
                factor_ms=row.factor_ms,
                loaded_at=now,
            )
        _loaded_at_global = now
    log.info("thresholds_cache full_reload — %d entries loaded", len(rows))


def reload_one(endpoint: str) -> None:
    """Recarga una fila especifica (tras edit o NOTIFY en Plan 04-04).

    Si la fila fue borrada (repo.get devuelve None), la purga del cache.
    Thread-safe.
    """
    session = SessionLocalNexo()
    try:
        row = ThresholdRepo(session).get(endpoint)
    finally:
        session.close()

    now = datetime.now(timezone.utc)
    with _cache_lock:
        if row is None:
            _cache.pop(endpoint, None)
            log.info("thresholds_cache reload_one %s → REMOVED", endpoint)
            return
        _cache[endpoint] = ThresholdEntry(
            endpoint=row.endpoint,
            warn_ms=row.warn_ms,
            block_ms=row.block_ms,
            factor_ms=row.factor_ms,
            loaded_at=now,
        )
    log.info("thresholds_cache reload_one %s → warn=%d block=%d factor=%s",
             endpoint, row.warn_ms, row.block_ms, row.factor_ms)


def get(endpoint: str) -> Optional[ThresholdEntry]:
    """Lee una entrada del cache. Dispara safety-net full_reload si stale.

    D-19: si ``_loaded_at_global`` es None o ``now - loaded_at_global >
    FALLBACK_REFRESH_SECONDS``, fuerza un ``full_reload`` sincrono
    antes de leer. Protege contra LISTEN caido silenciosamente.
    """
    global _loaded_at_global
    now = datetime.now(timezone.utc)
    needs_reload = (
        _loaded_at_global is None
        or (now - _loaded_at_global).total_seconds() > FALLBACK_REFRESH_SECONDS
    )
    if needs_reload:
        log.warning(
            "thresholds_cache stale (loaded_at_global=%s) — forcing full_reload",
            _loaded_at_global,
        )
        full_reload()

    with _cache_lock:
        return _cache.get(endpoint)


def notify_changed(endpoint: str) -> None:
    """Emite ``NOTIFY nexo_thresholds_changed, <endpoint>`` en Postgres.

    Llamado por el CRUD endpoint de thresholds (Plan 04-04) tras un
    UPDATE. Requiere AUTOCOMMIT porque NOTIFY fuera de una transaccion
    explicita necesita commit inmediato (psycopg2 quirk).

    La interfaz vive aqui para que:
      - 04-02 (preflight) pueda llamarla si alguna vez actualiza
        factor tras recomputacion sin pasar por el CRUD.
      - 04-04 (CRUD endpoint) tenga un punto central.

    NOTE: el listener que REACCIONA a este NOTIFY (LISTEN loop) se
    implementa en Plan 04-04. En 04-01 solo emitimos; el safety-net
    de ``get`` cubre el gap de 5 min.
    """
    raw = engine_nexo.raw_connection()
    try:
        raw.set_isolation_level(0)  # AUTOCOMMIT para NOTIFY
        cur = raw.cursor()
        cur.execute("SELECT pg_notify('nexo_thresholds_changed', %s)", (endpoint,))
        cur.close()
    finally:
        raw.close()
    log.info("thresholds_cache NOTIFY emitted for endpoint=%s", endpoint)


# NOTE: LISTEN/NOTIFY listener (``listen_loop`` / ``start_listener``)
# es implementado en Plan 04-04. En 04-01 solo emitimos NOTIFY y
# confiamos en el safety-net de 5min del ``get``.


__all__ = [
    "FALLBACK_REFRESH_SECONDS",
    "ThresholdEntry",
    "full_reload",
    "get",
    "notify_changed",
    "reload_one",
]
