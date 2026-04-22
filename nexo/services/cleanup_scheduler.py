"""Scheduler asyncio para cleanup jobs (Plans 04-03 + 04-04).

Loop async registrado en ``api/main.py::lifespan`` vía
``asyncio.create_task(cleanup_loop())``. Shutdown limpio via
``cleanup_task.cancel()`` en el finally del lifespan.

**Jobs registrados:**

- **query_log_cleanup** (Plan 04-04 / D-10) — Monday 03:00 UTC. Borra
  filas de ``nexo.query_log`` con ``ts < now - NEXO_QUERY_LOG_RETENTION_DAYS``
  (default 90d). 0 = forever (skip).
- **approvals_cleanup** (Plan 04-03 / D-14) — Monday 03:05 UTC. Marca
  ``expired`` las solicitudes ``pending`` con
  ``created_at < now - NEXO_APPROVAL_TTL_DAYS`` (default 7d).
- **factor_auto_refresh** (Plan 04-04 / D-20) — 1er Monday de cada mes
  03:10 UTC. Recalcula ``factor_ms`` por endpoint si ``factor_updated_at``
  > ``NEXO_AUTO_REFRESH_STALE_DAYS`` (default 60d).

Diseño:

- Loop forever calcula el ``seconds_until`` de TODOS los jobs; duerme
  hasta el minimo (asyncio.sleep cancellable desde lifespan).
- Al despertar, dispara CADA job cuyo ``seconds_to_X - min_sleep < 60``
  (tolerancia ±60s para permitir triggers colapsados).
- Para ``factor_auto_refresh`` aplica filtro adicional: solo dispara si
  ``datetime.utcnow().day <= 7`` (1er Monday del mes implica dia <= 7).
- Cada job corre en ``asyncio.to_thread`` — no bloquea el event loop.
- Excepciones de jobs se loggean como ``log.exception`` y el loop
  continúa (un job roto no tumba el scheduler).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timedelta, timezone

from nexo.services import approvals_cleanup, factor_auto_refresh, query_log_cleanup

log = logging.getLogger("nexo.cleanup_scheduler")


# ── Helpers de horario ────────────────────────────────────────────────────


def _seconds_until(target_time: time, *, dow: int | None = None) -> float:
    """Segundos hasta ``target_time`` (UTC) del próximo día coincidente.

    Args:
        target_time: hora objetivo ``datetime.time(h, m)``.
        dow: weekday (Mon=0..Sun=6). Si ``None``, objetivo es hoy-o-mañana.

    Returns:
        Delta segundos > 0 hasta el próximo disparo.
    """
    now = datetime.now(timezone.utc)
    target = now.replace(
        hour=target_time.hour,
        minute=target_time.minute,
        second=target_time.second,
        microsecond=0,
    )
    if dow is not None:
        days_ahead = (dow - now.weekday()) % 7
        if days_ahead == 0 and target <= now:
            days_ahead = 7
        target += timedelta(days=days_ahead)
    else:
        if target <= now:
            target += timedelta(days=1)
    return (target - now).total_seconds()


# ── Loop principal ────────────────────────────────────────────────────────

# Monday = 0 (datetime.weekday())
_DOW_MONDAY = 0
_QUERY_LOG_TIME = time(3, 0, 0)  # 03:00 UTC (Plan 04-04 / D-10)
_APPROVALS_TIME = time(3, 5, 0)  # 03:05 UTC (Plan 04-03 / D-14)
_FACTOR_TIME = time(3, 10, 0)  # 03:10 UTC (Plan 04-04 / D-20)

# Tolerancia para disparos colapsados en el mismo wake-up.
_TRIGGER_TOLERANCE_S = 60


async def cleanup_loop() -> None:
    """Loop forever — dispara jobs programados.

    Cancellable via ``task.cancel()`` en lifespan shutdown.
    """
    log.info(
        "cleanup_scheduler started "
        "(query_log_cleanup Mon 03:00, approvals_cleanup Mon 03:05, "
        "factor_auto_refresh 1er Mon del mes 03:10, todo UTC)"
    )
    while True:
        try:
            # Calcular cuántos segundos hasta cada target.
            sec_qlog = _seconds_until(_QUERY_LOG_TIME, dow=_DOW_MONDAY)
            sec_appr = _seconds_until(_APPROVALS_TIME, dow=_DOW_MONDAY)
            sec_fact = _seconds_until(_FACTOR_TIME, dow=_DOW_MONDAY)
            next_sec = min(sec_qlog, sec_appr, sec_fact)
            log.info(
                "cleanup_scheduler next run in %.0fs "
                "(qlog=%.0fs, appr=%.0fs, fact=%.0fs)",
                next_sec,
                sec_qlog,
                sec_appr,
                sec_fact,
            )
            await asyncio.sleep(next_sec)

            # Al despertar, disparar los jobs cuya ventana de tolerancia
            # haya entrado. Varios pueden dispararse en el mismo wake
            # (Monday 03:00, 03:05 y si es 1er Mon tambien 03:10).
            now = datetime.now(timezone.utc)

            if abs(sec_qlog - next_sec) < _TRIGGER_TOLERANCE_S:
                try:
                    n = await asyncio.to_thread(query_log_cleanup.run_once)
                    log.info("query_log_cleanup completed: %d rows deleted", n)
                except Exception:
                    log.exception("query_log_cleanup job failed")

            if abs(sec_appr - next_sec) < _TRIGGER_TOLERANCE_S:
                try:
                    n = await asyncio.to_thread(approvals_cleanup.run_once)
                    log.info("approvals_cleanup completed: %d rows expired", n)
                except Exception:
                    log.exception("approvals_cleanup job failed")

            if (
                abs(sec_fact - next_sec) < _TRIGGER_TOLERANCE_S
                and now.day <= 7  # 1er Monday del mes (D-20)
            ):
                try:
                    updated = await asyncio.to_thread(
                        factor_auto_refresh.run_once,
                    )
                    log.info(
                        "factor_auto_refresh completed: %d endpoints updated",
                        len(updated),
                    )
                except Exception:
                    log.exception("factor_auto_refresh job failed")

        except asyncio.CancelledError:
            log.info("cleanup_scheduler cancelled (shutdown)")
            raise
        except Exception:
            # Safety-net: cualquier error en el bucle se loggea y el loop
            # continúa. Previene que un bug one-off tumbe el scheduler.
            log.exception("cleanup_scheduler loop error — continuing")
            await asyncio.sleep(60)


__all__ = ["cleanup_loop"]
