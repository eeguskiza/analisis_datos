"""Preflight service — estima coste de una request ANTES de ejecutarla.

Pure function layer que mapea ``(endpoint, params)`` a un ``Estimation``
(level green/amber/red + estimated_ms + breakdown para UI). Sin side
effects; lee umbrales desde ``nexo.services.thresholds_cache`` (in-memory
dict con safety-net de 5 min por D-19).

Contrato:

    from nexo.services.preflight import estimate_cost
    est = estimate_cost("pipeline/run", {"n_recursos": 4, "n_dias": 15})
    # Estimation(level="amber", estimated_ms=120000, breakdown="...", ...)

Heurísticas (D-04):
- ``pipeline/run``: ``n_recursos × n_dias × factor_ms`` (factor por defecto
  2000 ms según D-04 seed; aprende vía recálculo manual en /ajustes/limites).
- ``bbdd/query``: baseline plano ``factor_ms`` (sin EXPLAIN — D-03 del
  research: EXPLAIN SQL Server es frágil para un preflight genérico).
- ``capacidad`` / ``operarios``: ``n_dias × factor_ms`` **sólo si
  ``rango_dias > 90``** (D-03); en rangos cortos devolvemos ``green``
  defensivo con razón explícita.

Clasificación (strict less-than):
  - ``estimated_ms < warn_ms``  → ``green``
  - ``warn_ms <= estimated_ms < block_ms`` → ``amber``
  - ``block_ms <= estimated_ms`` → ``red``  (600000 exacto → red)

Endpoints no soportados o sin threshold configurado devuelven un ``green``
defensivo con ``reason`` explícita — falla segura (no bloqueamos queries
válidas por config ausente).
"""
from __future__ import annotations

from datetime import date
from typing import Literal

from nexo.data.dto.query import Estimation
from nexo.services import thresholds_cache
from nexo.services.thresholds_cache import ThresholdEntry


Level = Literal["green", "amber", "red"]


# Factores de fallback cuando ``threshold.factor_ms`` es ``None`` (primera
# calibración aún no ejecutada). Mantienen estimate_cost puro — no dependen
# del estado del cache para poder estimar algo razonable.
_FALLBACK_PIPELINE_FACTOR_MS: float = 2000.0  # D-04 seed inicial
_FALLBACK_BBDD_FACTOR_MS: float = 1000.0      # baseline 1s para SQL libre
_FALLBACK_RANGO_FACTOR_MS: float = 50.0       # ms por día para capacidad/operarios


def estimate_cost(endpoint: str, params: dict) -> Estimation:
    """Estima coste de ``endpoint`` con ``params`` ANTES de ejecutar.

    Contrato:
      - No ejecuta la query real (ni toca SQL Server ni Postgres).
      - No muta ningún estado.
      - Es seguro llamarlo múltiples veces con los mismos inputs.
      - Cuando el threshold no existe o el endpoint no es conocido,
        devuelve un ``Estimation(level='green', estimated_ms=0, ...)``
        con ``reason`` explícita — el router decide si eso es aceptable.

    Args:
        endpoint: Una de ``pipeline/run | bbdd/query | capacidad | operarios``.
        params: Dict específico por endpoint (ver D-09 del CONTEXT).

    Returns:
        ``Estimation`` frozen (Pydantic model) con level/estimated_ms/
        breakdown/reason/factor_used_ms/warn_ms/block_ms/endpoint.
    """
    t = thresholds_cache.get(endpoint)
    if t is None:
        return _defensive_green(
            endpoint=endpoint,
            reason="threshold no configurado — default green",
        )

    if endpoint == "pipeline/run":
        return _estimate_pipeline(endpoint, params, t)
    if endpoint == "bbdd/query":
        return _estimate_bbdd(endpoint, params, t)
    if endpoint in ("capacidad", "operarios"):
        return _estimate_rango(endpoint, params, t)

    return _defensive_green(
        endpoint=endpoint,
        reason=f"endpoint {endpoint!r} no soportado — default green",
        warn_ms=t.warn_ms,
        block_ms=t.block_ms,
    )


# ── Endpoint-specific estimators ───────────────────────────────────────────

def _estimate_pipeline(
    endpoint: str, params: dict, t: ThresholdEntry
) -> Estimation:
    """Pipeline OEE: coste ~ n_recursos × n_dias × factor (D-04)."""
    n_recursos = int(
        params.get("n_recursos", 0)
        or len(params.get("recursos", []) or [])
    )
    n_dias = int(params.get("n_dias", 0) or _calc_days(params))
    factor = t.factor_ms if t.factor_ms is not None else _FALLBACK_PIPELINE_FACTOR_MS

    estimated = int(n_recursos * n_dias * factor)
    level = _classify(estimated, t)
    return Estimation(
        endpoint=endpoint,
        estimated_ms=estimated,
        level=level,
        reason=_reason_for(level, t),
        breakdown=f"{n_recursos} recursos × {n_dias} días × ~{factor/1000:.1f}s/run",
        factor_used_ms=factor,
        warn_ms=t.warn_ms,
        block_ms=t.block_ms,
    )


def _estimate_bbdd(
    endpoint: str, params: dict, t: ThresholdEntry
) -> Estimation:
    """SQL libre sobre MES: baseline plano (sin EXPLAIN — D-03 research)."""
    factor = t.factor_ms if t.factor_ms is not None else _FALLBACK_BBDD_FACTOR_MS
    estimated = int(factor)
    level = _classify(estimated, t)
    return Estimation(
        endpoint=endpoint,
        estimated_ms=estimated,
        level=level,
        reason=_reason_for(level, t),
        breakdown=f"baseline ~{factor/1000:.1f}s (sin EXPLAIN)",
        factor_used_ms=factor,
        warn_ms=t.warn_ms,
        block_ms=t.block_ms,
    )


def _estimate_rango(
    endpoint: str, params: dict, t: ThresholdEntry
) -> Estimation:
    """Capacidad/operarios: sólo preflight si rango_dias > 90 (D-03).

    Para rangos cortos devolvemos green con ``reason`` clara — la UI no
    abre modal y el router ejecuta directo. El router también filtra
    antes (short-circuit) pero este helper es defensivo: no queremos
    clasificar una petición de 30 días como amber por casualidad.
    """
    n_dias = int(params.get("rango_dias", 0) or _calc_days(params))
    if n_dias <= 90:
        return Estimation(
            endpoint=endpoint,
            estimated_ms=0,
            level="green",
            reason="rango <=90d — preflight desactivado",
            breakdown="",
            factor_used_ms=None,
            warn_ms=t.warn_ms,
            block_ms=t.block_ms,
        )

    factor = t.factor_ms if t.factor_ms is not None else _FALLBACK_RANGO_FACTOR_MS
    estimated = int(n_dias * factor)
    level = _classify(estimated, t)
    return Estimation(
        endpoint=endpoint,
        estimated_ms=estimated,
        level=level,
        reason=_reason_for(level, t),
        breakdown=f"{n_dias} días × ~{factor}ms/día",
        factor_used_ms=factor,
        warn_ms=t.warn_ms,
        block_ms=t.block_ms,
    )


# ── Helpers ────────────────────────────────────────────────────────────────

def _classify(estimated_ms: int, t: ThresholdEntry) -> Level:
    """Clasifica en ``green|amber|red`` según umbrales (strict less-than).

    Edge: ``estimated_ms == block_ms`` → ``red`` (600000 exacto cae ahí).
    """
    if estimated_ms < t.warn_ms:
        return "green"
    if estimated_ms < t.block_ms:
        return "amber"
    return "red"


def _reason_for(level: Level, t: ThresholdEntry) -> str:
    """Texto humano para la UI (modal amber/red) y logs."""
    if level == "green":
        return "Ejecución estándar"
    if level == "amber":
        return f"Supera umbral de aviso ({t.warn_ms}ms)"
    return f"Excede límite configurado de {t.block_ms/1000/60:.0f} min"


def _calc_days(params: dict) -> int:
    """Calcula ``(fecha_hasta - fecha_desde).days + 1`` desde params.

    Acepta múltiples nombres de campo porque cada endpoint usa los suyos
    (``fecha_desde``/``fecha_hasta`` vs ``fecha_inicio``/``fecha_fin``).
    Convierte strings ISO a date automáticamente. Devuelve 0 si faltan
    ambos — el caller decide qué hacer (pipeline calcula 0 recursos × 0
    días = 0 ms → green).
    """
    fi = params.get("fecha_desde") or params.get("fecha_inicio")
    ff = params.get("fecha_hasta") or params.get("fecha_fin")
    if not fi or not ff:
        return 0
    if isinstance(fi, str):
        fi = date.fromisoformat(fi)
    if isinstance(ff, str):
        ff = date.fromisoformat(ff)
    return (ff - fi).days + 1


def _defensive_green(
    *,
    endpoint: str,
    reason: str,
    warn_ms: int = 0,
    block_ms: int = 0,
) -> Estimation:
    """Construye un ``Estimation`` green defensivo (threshold ausente o
    endpoint desconocido). No bloqueamos queries por config incompleta."""
    return Estimation(
        endpoint=endpoint,
        estimated_ms=0,
        level="green",
        reason=reason,
        breakdown="",
        factor_used_ms=None,
        warn_ms=warn_ms,
        block_ms=block_ms,
    )


__all__ = ["estimate_cost"]
