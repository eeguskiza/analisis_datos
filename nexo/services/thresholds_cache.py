"""In-memory cache de ``nexo.query_thresholds`` (Phase 4 — Plans 04-01 + 04-04).

D-19 ("cache + invalidate on edit") se implementa en dos planes:

- **04-01**: skeleton con ``_cache`` dict + ``full_reload`` +
  ``reload_one`` + ``get`` con safety-net (re-lee BD si
  ``loaded_at_global`` esta stale > 5min). Tambien ``notify_changed(endpoint)``
  para que el CRUD endpoint emita NOTIFY.
- **04-04 (este plan)**: ``_blocking_listen_forever`` + ``start_listener``.
  Un worker thread abre una conexion psycopg2 dedicada (AUTOCOMMIT) con
  ``LISTEN nexo_thresholds_changed`` y reacciona en <1s a cada NOTIFY.
  El thread es ejecutado via ``asyncio.to_thread`` dentro del lifespan de
  FastAPI para no bloquear el event loop.

Contrato estable hacia 04-02:
  ``thresholds_cache.get(endpoint) -> Optional[ThresholdEntry]``

Los preflight services (Plan 04-02) consumen solo ``get`` — no dependen
del mecanismo de invalidacion.

Concurrencia:

- Reader path (``get``): sync, llamado desde middleware + routers. Usa
  ``threading.Lock`` para proteger el dict.
- Refresh path (``full_reload`` / ``reload_one``): sync, abre Session
  propia. Thread-safe.
- Listener path (``_blocking_listen_forever``): corre en thread pool
  (``asyncio.to_thread``). Reacciona a NOTIFY llamando ``reload_one``.
  Shutdown graceful via ``threading.Event``.

LISTEN/NOTIFY es best-effort (Postgres no encola mensajes perdidos). El
safety-net de 5 min en ``get()`` cubre caidas silenciosas del listener.
Reconexion automatica: si ``psycopg2.connect`` falla, retry tras 5s.
"""

from __future__ import annotations

import asyncio
import logging
import select
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Optional

import psycopg2
import psycopg2.extensions

from api.config import settings
from nexo.data.engines import SessionLocalNexo
from nexo.data.repositories.nexo import ThresholdRepo


log = logging.getLogger("nexo.thresholds_cache")


# D-19 safety net: si ``loaded_at_global`` se queda atras > 5 min
# (ej. porque LISTEN fallo silenciosamente), el proximo ``get`` fuerza
# un ``full_reload`` sincrono. Trade-off: latencia +100ms cada 5 min en
# el peor caso, a cambio de coherencia garantizada sin listener.
FALLBACK_REFRESH_SECONDS = 300


# H-02 / H-04 fix — canonical source of truth para los endpoints con
# preflight/postflight. Comparten allowlist:
#   - POST /api/approvals (H-02): evita pollution de la tabla con
#     endpoints arbitrarios (p.ej. "../admin/wipe_db").
#   - PUT/POST /api/thresholds/{endpoint:path} (H-04): evita que un
#     propietario mistype/construct endpoints inexistentes.
#   - QueryTimingMiddleware._TIMED_PATHS.values() debe coincidir
#     (duplicated literal; si divergen, se detecta en tests).
ALLOWED_ENDPOINTS: frozenset[str] = frozenset(
    {
        "pipeline/run",
        "bbdd/query",
        "capacidad",
        "operarios",
    }
)


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
    log.info(
        "thresholds_cache reload_one %s → warn=%d block=%d factor=%s",
        endpoint,
        row.warn_ms,
        row.block_ms,
        row.factor_ms,
    )


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

    **Pool hygiene (Plan 04-04 fix)**: NO reutilizar la conexion del
    pool de SQLAlchemy. Si pedimos un ``raw_connection`` y cambiamos su
    ``isolation_level`` a AUTOCOMMIT, al cerrarla vuelve al pool con el
    nivel alterado -> el siguiente uso (ej. ``db_nexo`` fixture con
    ``yield_per`` + named cursor) falla con "can't use a named cursor
    outside of transactions". En su lugar abrimos una conexion psycopg2
    dedicada que se descarta (no va al pool).

    La interfaz vive aqui para que:
      - 04-02 (preflight) pueda llamarla si alguna vez actualiza
        factor tras recomputacion sin pasar por el CRUD.
      - 04-04 (CRUD endpoint + factor_auto_refresh) tenga un punto central.
    """
    # Conexion dedicada fuera del pool SQLAlchemy. psycopg2 lee los
    # mismos parametros que el engine_nexo original (settings).
    # Import local para evitar coste al importar el modulo.
    conn = psycopg2.connect(
        host=settings.pg_host,
        port=settings.pg_port,
        user=settings.effective_pg_user,
        password=settings.effective_pg_password,
        dbname=settings.pg_db,
    )
    try:
        conn.set_isolation_level(
            psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT,
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_notify('nexo_thresholds_changed', %s)",
                (endpoint,),
            )
    finally:
        conn.close()
    log.info("thresholds_cache NOTIFY emitted for endpoint=%s", endpoint)


# ── LISTEN/NOTIFY listener (Plan 04-04 — D-19 completo) ───────────────────


# Timeout del select(). Permite revisar stop_event sin busy-spin.
_LISTEN_SELECT_TIMEOUT_S = 5.0
# Backoff tras fallo de conexion / loop caido.
_LISTEN_RETRY_BACKOFF_S = 5.0


def _blocking_listen_forever(stop_event: threading.Event) -> None:
    """Worker sincrono — LISTEN nexo_thresholds_changed hasta shutdown.

    Abre una conexion psycopg2 dedicada en AUTOCOMMIT (requisito de
    LISTEN/NOTIFY) y entra en un loop ``select()`` que despierta cada 5s
    para revisar ``stop_event``. Cada NOTIFY recibido dispara
    ``reload_one(endpoint)`` (o ``full_reload`` si payload vacio).

    Reconexion automatica: si ``psycopg2.connect`` falla, retry tras
    ``_LISTEN_RETRY_BACKOFF_S`` segundos. El retry es interruptible via
    ``stop_event.wait(timeout=...)``.

    Thread-safe: no comparte la conexion; cada reintento abre una nueva.
    Se ejecuta dentro de ``asyncio.to_thread`` para no bloquear el event
    loop del lifespan.
    """
    while not stop_event.is_set():
        conn = None
        try:
            conn = psycopg2.connect(
                host=settings.pg_host,
                port=settings.pg_port,
                user=settings.effective_pg_user,
                password=settings.effective_pg_password,
                dbname=settings.pg_db,
            )
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            cur = conn.cursor()
            cur.execute("LISTEN nexo_thresholds_changed")
            cur.close()
            log.info("thresholds_cache LISTEN activo (nexo_thresholds_changed)")

            while not stop_event.is_set():
                # select() con timeout 5s: permite revisar stop_event sin
                # busy-spin. El FD de la conn se despierta cuando llega
                # un NOTIFY async.
                if select.select([conn], [], [], _LISTEN_SELECT_TIMEOUT_S) == (
                    [],
                    [],
                    [],
                ):
                    # Timeout sin actividad; revisa stop y continua.
                    continue
                conn.poll()
                while conn.notifies:
                    notify = conn.notifies.pop(0)
                    endpoint = notify.payload or ""
                    log.info(
                        "thresholds_cache NOTIFY recibido endpoint=%r",
                        endpoint,
                    )
                    try:
                        if endpoint:
                            reload_one(endpoint)
                        else:
                            full_reload()
                    except Exception:
                        log.exception(
                            "thresholds_cache error en reload tras NOTIFY",
                        )
        except Exception:
            log.exception(
                "thresholds_cache LISTEN loop caido — reintentando en %.0fs",
                _LISTEN_RETRY_BACKOFF_S,
            )
            # stop_event.wait es interruptible, no bloquea shutdown.
            stop_event.wait(timeout=_LISTEN_RETRY_BACKOFF_S)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    log.info("thresholds_cache LISTEN worker detenido (stop_event)")


async def start_listener(stop_event: threading.Event) -> None:
    """Arranca el worker de LISTEN en el thread pool de asyncio.

    Llamado desde ``api/main.py::lifespan`` via
    ``asyncio.create_task(start_listener(stop_event))``. Al cerrar la
    app, el lifespan debe:

      1. ``stop_event.set()``  -> termina el while interno del worker.
      2. ``task.cancel()``    -> termina el to_thread wrapper.

    psycopg2 LISTEN es sincrono (select() nativo sobre el FD de la conn)
    por lo que vive en thread pool para no bloquear el event loop.
    """
    await asyncio.to_thread(_blocking_listen_forever, stop_event)


__all__ = [
    "FALLBACK_REFRESH_SECONDS",
    "ThresholdEntry",
    "_blocking_listen_forever",
    "full_reload",
    "get",
    "notify_changed",
    "reload_one",
    "start_listener",
]
