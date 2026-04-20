"""
Capacidad: fabricado real vs capacidad teorica por referencia.

Las piezas "salidas" se cuentan UNA sola vez, en el centro de trabajo final
(linea de montaje). Se consideran CTs finales aquellos cuyo nombre en
admuser.fmesrec (columna re020) empieza por 'Linea'.

Capacidad teorica por referencia = tiempo_trabajado_min * 60 / ciclo_teorico,
donde ciclo_teorico = percentil 10 de los ciclos observados en los ultimos
180 dias (mejor ritmo sostenido realista).

Plan 03-02 Task 3.3: elimina el bloque ``pyodbc.connect(...)`` + queries
inline; ahora usa ``engine_mes`` (2-part names, DATA-09) + ``load_sql``
sobre ``nexo/data/sql/mes/capacidad_*.sql``.

Plan 04-02: preflight condicional por D-03 — si ``rango_dias > 90`` se
estima coste y se aplica gate (green ejecuta, amber sin force -> 428,
red sin approval -> 403). Rangos cortos saltan el preflight completamente
(no modal, no query_log).
"""
from __future__ import annotations

import json
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import bindparam, text

from api.deps import DbNexo, EngineMes
from nexo.data.sql.loader import load_sql
from nexo.services.auth import require_permission
from nexo.services.preflight import estimate_cost


# ── Import-guard para Plan 04-03 (approvals) ─────────────────────────────
try:
    from nexo.services.approvals import consume_approval  # type: ignore[import-not-found]

    _APPROVALS_AVAILABLE = True
except ImportError:
    _APPROVALS_AVAILABLE = False


router = APIRouter(
    prefix="/capacidad",
    tags=["capacidad"],
    dependencies=[Depends(require_permission("capacidad:read"))],
)


def _p10(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = max(0, int(round(0.1 * (len(s) - 1))))
    return s[idx]


@router.get("")
def capacidad(
    engine_mes: EngineMes,
    request: Request,
    db: DbNexo,
    fecha_inicio: date = Query(...),
    fecha_fin: date = Query(...),
    force: bool = Query(False),
    approval_id: Optional[int] = Query(None),
    user=Depends(require_permission("capacidad:read")),
):
    """Devuelve capacidad vs fabricado por referencia en el periodo."""
    if fecha_fin < fecha_inicio:
        raise HTTPException(400, "fecha_fin < fecha_inicio")

    fi = fecha_inicio.strftime("%Y-%m-%d")
    ff = fecha_fin.strftime("%Y-%m-%d")

    # ── Preflight condicional (D-03: solo rango > 90d) ──────────────────
    # En rangos cortos, saltamos preflight completamente — no poblamos
    # request.state.estimated_ms y el middleware no escribe query_log.
    rango_dias = (fecha_fin - fecha_inicio).days
    if rango_dias > 90:
        params = {
            "fecha_desde": fi,
            "fecha_hasta": ff,
            "rango_dias": rango_dias,
        }
        params_json_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
        est = estimate_cost("capacidad", params)
        request.state.estimated_ms = est.estimated_ms
        request.state.params_json = params_json_str

        if est.level == "red":
            if not (force and approval_id is not None):
                raise HTTPException(
                    status_code=403,
                    detail={
                        "estimation": est.model_dump(),
                        "action": "request_approval",
                    },
                )
            if not _APPROVALS_AVAILABLE:
                raise HTTPException(
                    status_code=503,
                    detail="Flujo de aprobación no disponible aún — esperar Plan 04-03",
                )
            approval = consume_approval(
                db,
                approval_id=approval_id,
                user_id=user.id,
                current_params=params,
            )
            request.state.approval_id = approval.id
        elif est.level == "amber":
            if not force:
                raise HTTPException(
                    status_code=428,
                    detail={
                        "estimation": est.model_dump(),
                        "action": "confirm_amber",
                    },
                )

    try:
        with engine_mes.connect() as conn:
            # 1) Produccion real SOLO en CTs cuyo nombre empieza por 'Linea'
            piezas_sql = text(load_sql("mes/capacidad_piezas_linea"))
            real_rows = conn.execute(
                piezas_sql,
                {"fecha_inicio": fi, "fecha_fin": ff},
            ).fetchall()

            if not real_rows:
                return _empty(fi, ff)

            base: dict[tuple[str, str], dict] = {}
            for r in real_rows:
                key = (r[0], r[1])
                base[key] = {
                    "referencia": r[0] or "(sin orden)",
                    "ct": r[1],
                    "ct_nombre": r[2],
                    "piezas_reales": float(r[3] or 0),
                    "tiempo_trabajado_min": float(r[4] or 0),
                    "fecha_min": r[5].isoformat() if r[5] else None,
                    "fecha_max": r[6].isoformat() if r[6] else None,
                }

            # 2) Ciclo teorico (P10 de los ultimos 180 dias) por (ref, CT).
            #    La query acota :refs e :cts via bindparams expanding
            #    (DATA-09 — antes: OR dinamico construido como string).
            #    El filtrado final por par exacto (ref, ct) se hace en
            #    Python porque SQL Server no soporta IN con tuplas
            #    compuestas de forma nativa.
            pairs = list(base.keys())
            if pairs:
                refs = sorted({ref for ref, _ in pairs})
                cts = sorted({ct for _, ct in pairs})
                allowed_pairs = set(pairs)

                ciclos_sql = text(load_sql("mes/capacidad_ciclos_p10_180d")).bindparams(
                    bindparam("refs", expanding=True),
                    bindparam("cts", expanding=True),
                )

                ciclos: dict[tuple[str, str], list[float]] = {}
                for row in conn.execute(
                    ciclos_sql, {"refs": refs, "cts": cts}
                ).fetchall():
                    ref = row[0]
                    ct = row[1]
                    if (ref, ct) not in allowed_pairs:
                        # dos bindparams IN generan producto cartesiano;
                        # descartamos los pares no pedidos.
                        continue
                    tmin = float(row[2] or 0)
                    cant = float(row[3] or 0)
                    if tmin > 0 and cant > 0:
                        ciclo = tmin * 60.0 / cant
                        if 1 <= ciclo <= 600:
                            ciclos.setdefault((ref, ct), []).append(ciclo)

                for key, entry in base.items():
                    muestras = ciclos.get(key, [])
                    p10 = _p10(muestras)
                    entry["n_muestras_ciclo"] = len(muestras)

                    if entry["piezas_reales"] > 0 and entry["tiempo_trabajado_min"] > 0:
                        ciclo_real = entry["tiempo_trabajado_min"] * 60.0 / entry["piezas_reales"]
                        entry["ciclo_real_seg"] = round(ciclo_real, 2)
                    else:
                        ciclo_real = None
                        entry["ciclo_real_seg"] = None

                    # Ciclo teorico = el mejor (mas bajo) entre P10 historico y
                    # ciclo real del periodo. Evita eficiencias > 100%.
                    candidatos = [c for c in (p10, ciclo_real) if c and c > 0]
                    ct_teorico = min(candidatos) if candidatos else None
                    entry["ciclo_teorico_seg"] = round(ct_teorico, 2) if ct_teorico else None

                    if ct_teorico and entry["tiempo_trabajado_min"] > 0:
                        entry["piezas_teoricas"] = round(
                            entry["tiempo_trabajado_min"] * 60.0 / ct_teorico
                        )
                    else:
                        entry["piezas_teoricas"] = None

                    pt = entry["piezas_teoricas"]
                    entry["eficiencia_pct"] = (
                        round(100.0 * entry["piezas_reales"] / pt, 1)
                        if pt and pt > 0 else None
                    )

        por_referencia = sorted(base.values(), key=lambda x: -x["piezas_reales"])
        total_real = sum(x["piezas_reales"] for x in por_referencia)
        total_teorico = sum(x["piezas_teoricas"] or 0 for x in por_referencia)
        ef_global = (
            round(100.0 * total_real / total_teorico, 1)
            if total_teorico > 0 else None
        )

        # Totales por linea (CT)
        por_linea: dict[str, dict] = {}
        for r in por_referencia:
            k = r["ct"]
            d = por_linea.setdefault(k, {
                "ct": k, "ct_nombre": r["ct_nombre"],
                "piezas_reales": 0, "piezas_teoricas": 0, "tiempo_min": 0,
            })
            d["piezas_reales"] += r["piezas_reales"]
            d["piezas_teoricas"] += r["piezas_teoricas"] or 0
            d["tiempo_min"] += r["tiempo_trabajado_min"]

        for d in por_linea.values():
            d["piezas_reales"] = int(round(d["piezas_reales"]))
            d["piezas_teoricas"] = int(round(d["piezas_teoricas"]))
            d["tiempo_min"] = int(round(d["tiempo_min"]))
            d["eficiencia_pct"] = (
                round(100.0 * d["piezas_reales"] / d["piezas_teoricas"], 1)
                if d["piezas_teoricas"] > 0 else None
            )

        return {
            "fecha_inicio": fi,
            "fecha_fin": ff,
            "criterio": "CTs cuyo nombre empieza por 'Linea' (fmesrec.re020)",
            "totales": {
                "piezas_reales": int(round(total_real)),
                "piezas_teoricas": int(round(total_teorico)),
                "eficiencia_pct": ef_global,
                "n_referencias": len({r["referencia"] for r in por_referencia}),
                "n_lineas": len(por_linea),
            },
            "por_linea": sorted(
                por_linea.values(), key=lambda x: -x["piezas_reales"]
            ),
            "por_referencia": por_referencia,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Error calculando capacidad: {exc}")


def _empty(fi: str, ff: str) -> dict:
    return {
        "fecha_inicio": fi,
        "fecha_fin": ff,
        "criterio": "CTs cuyo nombre empieza por 'Linea' (fmesrec.re020)",
        "totales": {
            "piezas_reales": 0,
            "piezas_teoricas": 0,
            "eficiencia_pct": None,
            "n_referencias": 0,
            "n_lineas": 0,
        },
        "por_linea": [],
        "por_referencia": [],
    }
