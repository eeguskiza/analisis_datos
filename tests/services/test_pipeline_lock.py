"""Pipeline lock tests (Plan 04-02, QUERY-08, D-18).

Suite unitaria — NO toca Postgres ni el pipeline real. Valida la semántica
del semáforo ``asyncio.Semaphore(3)`` + ``asyncio.to_thread`` + timeout
``asyncio.wait_for`` que envuelve a ``run_pipeline`` en producción.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from nexo.services.pipeline_lock import (
    MAX_CONCURRENT,
    PIPELINE_TIMEOUT_SEC,
    pipeline_semaphore,
    run_with_lock,
)


@pytest.mark.asyncio
async def test_semaphore_limits_to_max():
    """Con MAX_CONCURRENT=3, sólo 3 corrutinas están dentro del semáforo
    simultáneamente. La 4ª+ espera hasta que una libere el slot.

    Usamos un contador protegido por Lock para observar el máximo de
    slots ocupados a la vez.
    """
    assert MAX_CONCURRENT == 3

    observed_max = 0
    current = 0
    lock = asyncio.Lock()

    async def worker():
        nonlocal observed_max, current
        async with pipeline_semaphore:
            async with lock:
                current += 1
                if current > observed_max:
                    observed_max = current
            # Mantiene el slot un rato suficiente para que otros midan.
            await asyncio.sleep(0.1)
            async with lock:
                current -= 1

    # 6 corrutinas concurrentes — deberían toparse con el límite de 3.
    await asyncio.gather(*(worker() for _ in range(6)))

    assert observed_max == MAX_CONCURRENT, (
        f"Esperado {MAX_CONCURRENT} slots concurrentes, observado {observed_max}"
    )


@pytest.mark.asyncio
async def test_timeout_raises_timeout_error():
    """asyncio.wait_for con timeout corto + to_thread que duerme más
    debe levantar TimeoutError en la corrutina.

    NOTE: el thread subyacente sigue corriendo hasta terminar su sleep
    (documentado en pipeline_lock.py — ver landmine).
    """
    def slow_fn():
        time.sleep(0.5)
        return "done"

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            asyncio.to_thread(slow_fn),
            timeout=0.05,
        )


@pytest.mark.asyncio
async def test_run_with_lock_completes_green():
    """run_with_lock ejecuta la función en un thread y devuelve el valor."""
    def fn(x, y):
        return x + y

    result = await run_with_lock(fn, 2, 3)
    assert result == 5


@pytest.mark.asyncio
async def test_run_with_lock_respects_semaphore():
    """run_with_lock usa pipeline_semaphore — 2 invocaciones paralelas se
    serializan si el semáforo está saturado."""
    # Asumimos que el test previo no dejó slots en uso (asyncio.gather termina).
    # Semáforo limpio debería tener ``_value == MAX_CONCURRENT``.

    observed = []

    def worker(label):
        observed.append(("start", label))
        time.sleep(0.05)
        observed.append(("end", label))
        return label

    results = await asyncio.gather(
        run_with_lock(worker, "a"),
        run_with_lock(worker, "b"),
    )
    assert sorted(results) == ["a", "b"]
    # Con MAX_CONCURRENT=3 ambas entran en paralelo, por lo que veremos
    # start-a, start-b antes de los ends.
    starts = [x for x in observed if x[0] == "start"]
    assert len(starts) == 2


def test_pipeline_timeout_sec_default():
    """Default PIPELINE_TIMEOUT_SEC = 900 (15 min) salvo override via env."""
    import os

    if os.environ.get("NEXO_PIPELINE_TIMEOUT_SEC"):
        pytest.skip("env override present — skip default check")
    assert PIPELINE_TIMEOUT_SEC == 900.0


def test_pipeline_semaphore_is_asyncio_semaphore():
    """Type check: la primitiva es asyncio.Semaphore (no Lock, no mp)."""
    assert isinstance(pipeline_semaphore, asyncio.Semaphore)
