"""Pipeline concurrency primitive — semáforo global + timeout soft.

D-18: máximo ``NEXO_PIPELINE_MAX_CONCURRENT=3`` ejecuciones simultáneas de
``api.services.pipeline.run_pipeline``. Timeout ``NEXO_PIPELINE_TIMEOUT_SEC
=900`` (15 min) tras el cual la corrutina cancela y el handler devuelve
HTTP 504.

.. note::
    # NOTE: asyncio.wait_for does NOT kill the underlying thread.
    # Timeout is UX-soft — matplotlib keeps running until it completes.
    # (ES) NO mata el thread subyacente. Ver 04-RESEARCH.md Pitfall 1.

    Semántica de ``asyncio.wait_for(asyncio.to_thread(...), timeout=T)``:
      - Cuando ``T`` expira, la corrutina recibe ``CancelledError`` y se
        levanta ``TimeoutError`` en el handler.
      - El thread que ejecuta el código síncrono **sigue vivo** hasta que
        termine por su cuenta. Python no puede interrumpir threads
        bloqueados en C extensions (matplotlib, numpy, pyodbc).
      - El slot del ``asyncio.Semaphore`` se libera al salir del
        ``async with`` en la corrutina, pero el thread sigue consumiendo
        CPU/RAM hasta que matplotlib cierre sus figs.
      - El ``finally: shutil.rmtree(tmp_root, ignore_errors=True)`` del
        pipeline corre dentro del thread, por lo que la limpieza ocurre
        aunque el handler haya devuelto 504.

    Implicación operativa: tras un timeout, un slot se libera a la
    corrutina pero el thread puede seguir hasta 30-60s más. Esto es
    aceptable en Mark-III (LAN + operador único) — ver
    ``04-RESEARCH.md`` §Pitfall 1 para el análisis completo.

Env vars:
  - ``NEXO_PIPELINE_MAX_CONCURRENT``: número de slots (default 3, rango
    recomendado 1-5 — cada slot ~200MB pico de matplotlib).
  - ``NEXO_PIPELINE_TIMEOUT_SEC``: timeout duro a nivel corrutina (default
    900, rango recomendado 300-1800).

Uso desde router:

    from nexo.services.pipeline_lock import pipeline_semaphore, run_with_lock

    async def run(...):
        messages = await run_with_lock(run_pipeline_sync, fecha_inicio=..., ...)
        return StreamingResponse(replay(messages))

O explícitamente:

    async with pipeline_semaphore:
        messages = await asyncio.wait_for(
            asyncio.to_thread(run_pipeline_sync, ...),
            timeout=PIPELINE_TIMEOUT_SEC,
        )
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Callable, TypeVar


# ── Env-driven constants (leídas al import; no se releen en runtime) ──────

MAX_CONCURRENT: int = int(os.environ.get("NEXO_PIPELINE_MAX_CONCURRENT", "3"))
PIPELINE_TIMEOUT_SEC: float = float(os.environ.get("NEXO_PIPELINE_TIMEOUT_SEC", "900"))


# ── Semáforo global (1 instancia por proceso worker uvicorn) ──────────────
# Si uvicorn se escala a --workers N, cada worker tiene su propio semáforo
# independiente → la concurrencia real es N × MAX_CONCURRENT. En Mark-III
# el default es workers=1, así que no nos preocupa. Documentado por si
# alguien cambia el setting.

pipeline_semaphore: asyncio.Semaphore = asyncio.Semaphore(MAX_CONCURRENT)


T = TypeVar("T")


async def run_with_lock(
    fn: Callable[..., T],
    /,
    *args: Any,
    **kwargs: Any,
) -> T:
    """Ejecuta ``fn(*args, **kwargs)`` en un thread bajo semáforo + timeout.

    Patrón canónico para envolver código síncrono bloqueante (matplotlib,
    pyodbc, pandas) desde un handler FastAPI async sin congelar el event
    loop.

    Raises:
        asyncio.TimeoutError: si ``fn`` excede ``PIPELINE_TIMEOUT_SEC``.
            El thread subyacente sigue corriendo; documentado en el
            docstring del módulo.
        Cualquier excepción de ``fn``: se propaga al caller.

    Returns:
        Lo que devuelva ``fn``.
    """
    async with pipeline_semaphore:
        return await asyncio.wait_for(
            asyncio.to_thread(fn, *args, **kwargs),
            timeout=PIPELINE_TIMEOUT_SEC,
        )


__all__ = [
    "MAX_CONCURRENT",
    "PIPELINE_TIMEOUT_SEC",
    "pipeline_semaphore",
    "run_with_lock",
]
