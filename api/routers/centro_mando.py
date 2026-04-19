"""API del Centro de Mando — estado de maquinas en tiempo real desde IZARO fmesmic."""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.database import Recurso, engine, get_db
from nexo.services.auth import require_permission

router = APIRouter(
    prefix="/dashboard",
    tags=["dashboard"],
    dependencies=[Depends(require_permission("centro_mando:read"))],
)
log = logging.getLogger(__name__)

# ── Cache ───────────────────────────────────────────────────────────────────
CACHE_TTL = 15 * 60  # 15 minutos

_cache: dict = {
    "fecha": None,
    "data": None,
    "timestamp": 0,
    "error": None,
}


def _fecha_produccion() -> date:
    now = datetime.now()
    return (now - timedelta(days=1)).date() if now.hour < 6 else now.date()


def _query_fmesmic(ct_codes: list[int]) -> dict[int, dict]:
    """Consulta fmesmic en dbizaro para el estado actual de las maquinas.

    Devuelve {ct_code: {piezas_hoy, ultimo_evento, referencia}} para las
    maquinas que tienen actividad hoy.
    """
    if not ct_codes:
        return {}

    placeholders = ",".join(f":ct{i}" for i in range(len(ct_codes)))
    params = {f"ct{i}": str(ct) for i, ct in enumerate(ct_codes)}

    sql = text(f"""
        SELECT
            CAST(RTRIM(mi020) AS INT)          AS ct,
            COUNT(*)                            AS piezas_hoy,
            MAX(CAST(mi100 AS TIME))            AS ultimo_evento,
            (SELECT TOP 1 RTRIM(m2.mi060)
             FROM dbizaro.admuser.fmesmic m2
             WHERE RTRIM(m2.mi020) = RTRIM(m.mi020)
               AND CONVERT(DATE, m2.mi090) = CONVERT(DATE, GETDATE())
             ORDER BY m2.mi050 DESC)            AS referencia
        FROM dbizaro.admuser.fmesmic m
        WHERE CONVERT(DATE, mi090) = CONVERT(DATE, GETDATE())
          AND RTRIM(mi020) IN ({placeholders})
        GROUP BY RTRIM(mi020)
    """)

    result = {}
    with engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()
        for r in rows:
            ct = int(r[0])
            ultimo = r[2]
            # Determinar si esta activa: ultimo evento hace menos de 30 min
            ahora = datetime.now().time()
            if ultimo:
                ultimo_dt = datetime.combine(date.today(), ultimo)
                ahora_dt = datetime.combine(date.today(), ahora)
                diff_min = (ahora_dt - ultimo_dt).total_seconds() / 60
                activa = diff_min < 30
            else:
                activa = False

            result[ct] = {
                "piezas_hoy": int(r[1]),
                "ultimo_evento": str(ultimo)[:5] if ultimo else None,
                "referencia": r[3] or "",
                "activa_reciente": activa,
            }
    return result


def _get_cached_data(ct_codes: list[int]) -> tuple[dict, str | None, float]:
    now = time.time()
    hoy = _fecha_produccion()

    if (_cache["fecha"] == hoy
            and (now - _cache["timestamp"]) < CACHE_TTL
            and _cache["data"] is not None):
        return _cache["data"], _cache["error"], _cache["timestamp"]

    try:
        data = _query_fmesmic(ct_codes)
        _cache["data"] = data
        _cache["fecha"] = hoy
        _cache["timestamp"] = now
        _cache["error"] = None
        log.info("Centro de Mando: %d CTs con actividad en fmesmic", len(data))
    except Exception as exc:
        log.warning("Centro de Mando: error consultando fmesmic: %s", exc)
        _cache["error"] = str(exc)
        if _cache["data"] is None:
            _cache["data"] = {}

    return _cache["data"], _cache["error"], _cache["timestamp"]


@router.get("/summary")
def summary(db: Session = Depends(get_db)):
    hoy = _fecha_produccion()

    all_recursos = (
        db.query(Recurso)
        .filter_by(activo=True)
        .order_by(Recurso.seccion, Recurso.nombre)
        .all()
    )
    ct_codes = [r.centro_trabajo for r in all_recursos]
    mic_data, izaro_error, cache_ts = _get_cached_data(ct_codes)

    maquinas = []
    for rec in all_recursos:
        info = mic_data.get(rec.centro_trabajo, {})
        if info.get("activa_reciente"):
            estado = "produciendo"
        elif info.get("piezas_hoy", 0) > 0:
            estado = "incidencia"  # tuvo actividad hoy pero no reciente
        else:
            estado = "sin_datos"

        maquinas.append({
            "nombre": rec.nombre,
            "seccion": rec.seccion or "GENERAL",
            "estado": estado,
            "piezas_hoy": info.get("piezas_hoy", 0),
            "ultimo_evento": info.get("ultimo_evento"),
            "referencia": info.get("referencia", ""),
        })

    estado_order = {"produciendo": 0, "incidencia": 1, "sin_datos": 2}
    sec_order = {"LINEAS": 0, "TALLADORAS": 1, "GENERAL": 9}
    maquinas.sort(key=lambda m: (
        estado_order.get(m["estado"], 9),
        sec_order.get(m["seccion"], 5),
        m["nombre"],
    ))

    n_activas = sum(1 for m in maquinas if m["estado"] == "produciendo")
    last_izaro = datetime.fromtimestamp(cache_ts).strftime("%H:%M:%S") if cache_ts > 0 else None

    return {
        "fecha": hoy.isoformat(),
        "maquinas": maquinas,
        "n_activas": n_activas,
        "n_total": len(maquinas),
        "last_izaro": last_izaro,
        "izaro_error": izaro_error,
    }


@router.post("/refresh")
def force_refresh():
    _cache["timestamp"] = 0
    return {"ok": True}
