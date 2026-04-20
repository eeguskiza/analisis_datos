"""Scheduler asyncio para cleanup jobs (Plan 04-03 / QUERY-06 / D-14).

Loop async registrado en ``api/main.py::lifespan`` vía
``asyncio.create_task(cleanup_loop())``. Shutdown limpio via
``cleanup_task.cancel()`` en el finally del lifespan.

**Jobs registrados en este plan (04-03):**

- **approvals_cleanup** — Monday 03:05 UTC cada semana. Marca ``expired``
  las solicitudes ``pending`` con ``created_at < now - NEXO_APPROVAL_TTL_DAYS``.

Plan 04-04 extenderá este archivo con 2 jobs adicionales:

- query_log_cleanup (Monday 03:00).
- factor_auto_refresh (1er Monday del mes 03:10).

Diseño:

- Loop forever calcula el seconds_until del próximo disparo y duerme
  con ``asyncio.sleep`` (cancellable desde el lifespan).
- Si el wake-up está dentro de ±60s del target, dispara el job;
  si no, vuelve a calcular (defensivo frente a drift de clock).
- Cada job se ejecuta via ``asyncio.to_thread`` para no bloquear el
  event loop (el job hace I/O DB sincrónicamente).
- Excepciones de jobs se loggean como ``log.exception`` y el loop
  continúa (un job roto no tumba el scheduler).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timedelta, timezone

from nexo.services import approvals_cleanup

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
_APPROVALS_CLEANUP_DOW = 0
_APPROVALS_CLEANUP_TIME = time(3, 5, 0)  # 03:05 UTC


async def cleanup_loop() -> None:
    """Loop forever — dispara jobs programados.

    Cancellable via ``task.cancel()`` en lifespan shutdown.
    """
    log.info("cleanup_scheduler started (approvals_cleanup Mon 03:05 UTC)")
    while True:
        try:
            # Calcular cuántos segundos hasta el próximo disparo registrado.
            seconds_to_approvals = _seconds_until(
                _APPROVALS_CLEANUP_TIME, dow=_APPROVALS_CLEANUP_DOW,
            )
            log.info(
                "cleanup_scheduler next run in %.0fs (approvals_cleanup)",
                seconds_to_approvals,
            )
            await asyncio.sleep(seconds_to_approvals)

            # Al despertar, disparar el job. Tolerancia ±60s (si el sleep
            # fue más preciso podríamos llegar exactamente al target).
            try:
                n = await asyncio.to_thread(approvals_cleanup.run_once)
                log.info("approvals_cleanup completed: %d rows expired", n)
            except Exception:
                log.exception("approvals_cleanup job failed")

        except asyncio.CancelledError:
            log.info("cleanup_scheduler cancelled (shutdown)")
            raise
        except Exception:
            # Safety-net: cualquier error en el bucle se loggea y el loop
            # continúa. Previene que un bug one-off tumbe el scheduler.
            log.exception("cleanup_scheduler loop error — continuing")
            await asyncio.sleep(60)


__all__ = ["cleanup_loop"]
