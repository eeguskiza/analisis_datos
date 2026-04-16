"""Utilidades centralizadas de turnos — unica fuente de verdad."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import NamedTuple


TURNOS = (
    {"name": "T1", "start": 6, "end": 14},
    {"name": "T2", "start": 14, "end": 22},
    {"name": "T3", "start": 22, "end": 6},
)


class TurnoBoundary(NamedTuple):
    name: str
    start: datetime
    end: datetime


def get_jornada_start(now: datetime | None = None) -> datetime:
    """Devuelve las 06:00 del dia de jornada (si ahora <06, es ayer)."""
    now = now or datetime.now()
    if now.hour < 6:
        base = now - timedelta(days=1)
    else:
        base = now
    return base.replace(hour=6, minute=0, second=0, microsecond=0)


def get_turno_actual(now: datetime | None = None) -> str:
    """Devuelve el nombre del turno actual: T1, T2 o T3."""
    now = now or datetime.now()
    if 6 <= now.hour < 14:
        return "T1"
    if 14 <= now.hour < 22:
        return "T2"
    return "T3"


def get_turno_boundaries(now: datetime | None = None) -> list[TurnoBoundary]:
    """Devuelve los limites de cada turno de la jornada actual.

    Para el turno en curso, end = now (parcial).
    Para turnos futuros no se incluyen.
    T3 cruza medianoche: 22:00 -> 06:00 del dia siguiente.
    """
    now = now or datetime.now()
    jornada = get_jornada_start(now)
    turno_actual = get_turno_actual(now)
    result: list[TurnoBoundary] = []

    for t in TURNOS:
        name = t["name"]
        if name == "T3":
            start_dt = jornada.replace(hour=22)
            end_dt = jornada + timedelta(days=1)  # 06:00 dia siguiente
        else:
            start_dt = jornada.replace(hour=t["start"])
            end_dt = jornada.replace(hour=t["end"])

        # Para el turno en curso, truncar end a now
        if name == turno_actual:
            end_dt = now

        result.append(TurnoBoundary(name, start_dt, end_dt))

    return result


def turno_from_hour(hour: int) -> str:
    """Mapea una hora (0-23) al turno correspondiente."""
    if 6 <= hour < 14:
        return "T1"
    if 14 <= hour < 22:
        return "T2"
    return "T3"
