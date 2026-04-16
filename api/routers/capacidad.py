"""
Capacidad: fabricado real vs capacidad teorica por referencia.

Las piezas "salidas" se cuentan UNA sola vez, en el centro de trabajo final
(linea de montaje). Se consideran CTs finales aquellos cuyo nombre en
admuser.fmesrec (columna re020) empieza por 'Linea'.

Capacidad teorica por referencia = tiempo_trabajado_min * 60 / ciclo_teorico,
donde ciclo_teorico = percentil 10 de los ciclos observados en los ultimos
180 dias (mejor ritmo sostenido realista).
"""
from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/capacidad", tags=["capacidad"])


def _p10(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = max(0, int(round(0.1 * (len(s) - 1))))
    return s[idx]


@router.get("")
def capacidad(
    fecha_inicio: date = Query(...),
    fecha_fin: date = Query(...),
):
    """Devuelve capacidad vs fabricado por referencia en el periodo."""
    if fecha_fin < fecha_inicio:
        raise HTTPException(400, "fecha_fin < fecha_inicio")

    import pyodbc
    from api.services.db import get_config
    from OEE.db.connector import _build_connection_string

    cfg = {**get_config(), "database": "dbizaro"}

    try:
        conn = pyodbc.connect(_build_connection_string(cfg), timeout=30)
        cursor = conn.cursor()

        fi = fecha_inicio.strftime("%Y-%m-%d")
        ff = fecha_fin.strftime("%Y-%m-%d")

        # 1) Produccion real SOLO en CTs cuyo nombre empieza por 'Linea'
        #    (Lineas de montaje final). Evita doble contabilidad.
        cursor.execute("""
            SELECT
                RTRIM(COALESCE(lof.lo030, ''))          AS referencia,
                RTRIM(CAST(dtc.dt150 AS NVARCHAR(50)))  AS ct,
                RTRIM(rec.re020)                         AS ct_nombre,
                SUM(CASE WHEN dtc.dt110 = '0'
                         THEN COALESCE(CAST(dtc.dt130 AS FLOAT), 0)
                         ELSE 0 END)                    AS piezas,
                SUM(CASE WHEN dtc.dt110 IN ('0','1')
                         THEN COALESCE(CAST(dtc.dt090 AS FLOAT), 0)
                         ELSE 0 END)                    AS tiempo_min,
                MIN(dtc.dt060)                           AS fecha_min,
                MAX(dtc.dt060)                           AS fecha_max
            FROM admuser.fmesdtc dtc
            INNER JOIN admuser.fmesrec rec
                ON CAST(rec.re010 AS INT) = CAST(dtc.dt150 AS INT)
            LEFT JOIN admuser.fprolof lof
                ON  RTRIM(lof.lo010) = RTRIM(dtc.dt020)
                AND lof.lo020 = dtc.dt030
            WHERE CONVERT(DATE, dtc.dt060) BETWEEN ? AND ?
              AND RTRIM(rec.re020) LIKE 'Linea%'
            GROUP BY RTRIM(COALESCE(lof.lo030, '')),
                     RTRIM(CAST(dtc.dt150 AS NVARCHAR(50))),
                     RTRIM(rec.re020)
            HAVING SUM(CASE WHEN dtc.dt110 = '0'
                            THEN COALESCE(CAST(dtc.dt130 AS FLOAT), 0)
                            ELSE 0 END) > 0
        """, (fi, ff))

        real_rows = cursor.fetchall()
        if not real_rows:
            conn.close()
            return _empty(fi, ff)

        base = {}
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

        # 2) Ciclo teorico (P10 de los ultimos 180 dias) por (ref, CT)
        pairs = list(base.keys())
        if pairs:
            or_clauses = " OR ".join([
                "(RTRIM(COALESCE(lof.lo030, '')) = ? AND "
                "RTRIM(CAST(dtc.dt150 AS NVARCHAR(50))) = ?)"
                for _ in pairs
            ])
            cparams: list[Any] = []
            for ref, ct in pairs:
                cparams.extend([ref, ct])

            cursor.execute(f"""
                SELECT
                    RTRIM(COALESCE(lof.lo030, ''))          AS referencia,
                    RTRIM(CAST(dtc.dt150 AS NVARCHAR(50)))  AS ct,
                    CAST(dtc.dt090 AS FLOAT)                AS tiempo_min,
                    CAST(dtc.dt130 AS FLOAT)                AS cantidad
                FROM admuser.fmesdtc dtc
                LEFT JOIN admuser.fprolof lof
                    ON  RTRIM(lof.lo010) = RTRIM(dtc.dt020)
                    AND lof.lo020 = dtc.dt030
                WHERE dtc.dt060 >= DATEADD(DAY, -180, GETDATE())
                  AND dtc.dt110 = '0'
                  AND COALESCE(CAST(dtc.dt130 AS FLOAT), 0) >= 5
                  AND COALESCE(CAST(dtc.dt090 AS FLOAT), 0) >= 2
                  AND ({or_clauses})
            """, cparams)

            ciclos: dict[tuple[str, str], list[float]] = {}
            for row in cursor.fetchall():
                ref, ct, tmin, cant = row[0], row[1], float(row[2] or 0), float(row[3] or 0)
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

        conn.close()

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
