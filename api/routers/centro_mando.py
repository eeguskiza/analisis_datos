"""API del Centro de Mando — estado de maquinas en tiempo real desde IZARO fmesmic.

Plan 03-02 Task 3.2: elimina la query inline con 3-part names hacia
``admuser.fmesmic`` y el patron string-interpolation ``ct0,ct1,...``
— todo eso pasa a ``MesRepository.centro_mando_fmesmic`` (2-part
names + bindparam expanding, DATA-09). El router solo orquesta
cache y shape de respuesta.
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends

from api.database import Recurso
from api.deps import DbApp, EngineMes
from nexo.data.repositories.mes import MesRepository
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


def _rows_to_cache_shape(rows: list[dict]) -> dict[int, dict]:
    """Convierte las filas de ``MesRepository.centro_mando_fmesmic`` al
    shape dict-by-CT que consume el handler ``summary`` (equivalente al
    retorno de ``_query_fmesmic`` pre-refactor).
    """
    data: dict[int, dict] = {}
    ahora = datetime.now().time()
    for r in rows:
        ct = int(r["ct"])
        ultimo = r.get("ultimo_evento")
        if ultimo:
            ultimo_dt = datetime.combine(date.today(), ultimo)
            ahora_dt = datetime.combine(date.today(), ahora)
            diff_min = (ahora_dt - ultimo_dt).total_seconds() / 60
            activa = diff_min < 30
        else:
            activa = False

        data[ct] = {
            "piezas_hoy": int(r.get("piezas_hoy") or 0),
            "ultimo_evento": str(ultimo)[:5] if ultimo else None,
            "referencia": r.get("referencia") or "",
            "activa_reciente": activa,
        }
    return data


def _get_cached_data(
    ct_codes: list[int],
    mes_repo: MesRepository,
) -> tuple[dict, str | None, float]:
    now = time.time()
    hoy = _fecha_produccion()

    if (_cache["fecha"] == hoy
            and (now - _cache["timestamp"]) < CACHE_TTL
            and _cache["data"] is not None):
        return _cache["data"], _cache["error"], _cache["timestamp"]

    try:
        rows = mes_repo.centro_mando_fmesmic(ct_codes)
        data = _rows_to_cache_shape(rows)
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
def summary(db: DbApp, engine_mes: EngineMes):
    hoy = _fecha_produccion()
    mes_repo = MesRepository(engine=engine_mes)

    all_recursos = (
        db.query(Recurso)
        .filter_by(activo=True)
        .order_by(Recurso.seccion, Recurso.nombre)
        .all()
    )
    ct_codes = [r.centro_trabajo for r in all_recursos]
    mic_data, izaro_error, cache_ts = _get_cached_data(ct_codes, mes_repo)

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
