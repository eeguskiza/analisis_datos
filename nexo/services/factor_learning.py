"""Factor learning helper (Plan 04-04 / D-04 + D-20).

Compartido entre:

- ``api/routers/limites.py::recalibrate`` — botón manual "Recalcular
  desde últimos 30 runs" en ``/ajustes/limites``.
- ``nexo/services/factor_auto_refresh.py::run_once`` — cron mensual que
  recalcula factor si ``factor_updated_at`` es stale.

Algoritmo (D-04):

- Para ``pipeline/run``: parse ``params_json`` de cada fila, extrae
  ``n_recursos`` y ``n_dias``; factor = median(actual_ms / (n_recursos
  × n_dias)). Filtra filas con ``actual_ms <= 500`` (Pitfall 6 —
  outliers triviales que no reflejan coste real de render).
- Para el resto de endpoints (``bbdd/query``, ``capacidad``,
  ``operarios``): factor = median(actual_ms). Mismo filtro de
  ``actual_ms > 500``.

Ventana: últimos ``max_samples`` runs con ``status IN ('ok','slow')``.
Si tras filtrar quedan < 10 muestras, devuelve ``(None, sample_size)``:
el caller decide si levantar 400 (API) o skip (cron).
"""

from __future__ import annotations

import json
import logging
import statistics
from typing import Optional

from sqlalchemy.orm import Session

from nexo.data.models_nexo import NexoQueryLog
from sqlalchemy import select

log = logging.getLogger("nexo.factor_learning")


# Pitfall 6: filas triviales (cache hits / errores rápidos) distorsionan
# el median. 500ms es un piso razonable — cualquier endpoint real de
# preflight tarda >500ms aunque sea trivial.
_MIN_ACTUAL_MS = 500

# Umbral de muestras mínimas para que el factor calculado sea
# estadísticamente defendible. Por debajo devolvemos None — el caller
# recibe el sample_size y da feedback al usuario.
_MIN_SAMPLE_SIZE = 10


def compute_factor(
    db: Session,
    endpoint: str,
    max_samples: int = 30,
) -> tuple[Optional[float], int]:
    """Calcula factor_ms nuevo desde los ``max_samples`` runs mas recientes.

    Args:
        db: Session Postgres nexo.
        endpoint: clave de ``nexo.query_thresholds`` (e.g.
            ``pipeline/run``, ``bbdd/query``, ``capacidad``,
            ``operarios``).
        max_samples: tamaño de ventana. Default 30 (D-04).

    Returns:
        ``(factor_nuevo, sample_size)``. ``factor_nuevo`` es ``None`` si
        no hay suficientes muestras (< _MIN_SAMPLE_SIZE) tras filtrar
        outliers. ``sample_size`` es el número de filas que entraron al
        median final (despues de parse + outlier filter).
    """
    stmt = (
        select(NexoQueryLog)
        .where(
            NexoQueryLog.endpoint == endpoint,
            NexoQueryLog.status.in_(["ok", "slow"]),
        )
        .order_by(NexoQueryLog.ts.desc())
        .limit(max_samples)
    )
    rows = db.execute(stmt).scalars().all()

    if endpoint == "pipeline/run":
        per_unit: list[float] = []
        for r in rows:
            if not r.actual_ms or r.actual_ms <= _MIN_ACTUAL_MS:
                continue
            try:
                p = json.loads(r.params_json or "{}")
            except Exception:
                continue
            n_r = int(p.get("n_recursos", 0) or 0)
            n_d = int(p.get("n_dias", 0) or 0)
            if n_r <= 0 or n_d <= 0:
                continue
            per_unit.append(r.actual_ms / (n_r * n_d))

        sample_size = len(per_unit)
        if sample_size < _MIN_SAMPLE_SIZE:
            log.info(
                "compute_factor %s: insuficientes muestras (%d < %d)",
                endpoint,
                sample_size,
                _MIN_SAMPLE_SIZE,
            )
            return None, sample_size
        return float(statistics.median(per_unit)), sample_size

    # Endpoints no-pipeline: median(actual_ms) con outlier filter.
    durations = [
        r.actual_ms for r in rows if r.actual_ms and r.actual_ms > _MIN_ACTUAL_MS
    ]
    sample_size = len(durations)
    if sample_size < _MIN_SAMPLE_SIZE:
        log.info(
            "compute_factor %s: insuficientes muestras (%d < %d)",
            endpoint,
            sample_size,
            _MIN_SAMPLE_SIZE,
        )
        return None, sample_size
    return float(statistics.median(durations)), sample_size


__all__ = ["compute_factor"]
