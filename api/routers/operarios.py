"""Endpoints de rendimiento de operarios — consulta directa a IZARO."""
from __future__ import annotations

from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query

from api.services import db as mes_service
from nexo.services.auth import require_permission

router = APIRouter(
    prefix="/operarios",
    tags=["operarios"],
    dependencies=[Depends(require_permission("operarios:read"))],
)


def _connect():
    from OEE.db.connector import _build_connection_string
    import pyodbc
    cfg = mes_service.get_config()
    return pyodbc.connect(_build_connection_string(cfg), timeout=15)


@router.get("")
def listar_operarios(activos: bool = True):
    """
    Lista operarios de IZARO.
    Devuelve codigo, nombre, activo, y nº registros último mes.
    """
    conn = _connect()
    try:
        c = conn.cursor()
        c.execute("""
            SELECT
                CAST(ope.op010 AS INT) AS codigo,
                RTRIM(ope.op020) AS nombre,
                CAST(ope.op060 AS INT) AS activo,
                COALESCE(act.n_mes, 0) AS n_registros_mes,
                act.ultimo
            FROM admuser.fmesope ope
            LEFT JOIN (
                SELECT CAST(dt140 AS INT) AS ope_cod,
                    COUNT(*) AS n_mes,
                    MAX(CONVERT(DATE, dt060)) AS ultimo
                FROM admuser.fmesdtc
                WHERE dt060 >= DATEADD(MONTH, -1, GETDATE())
                    AND dt140 IS NOT NULL AND RTRIM(dt140) != ''
                GROUP BY CAST(dt140 AS INT)
            ) act ON act.ope_cod = CAST(ope.op010 AS INT)
            WHERE ope.op000 = 'ALGA'
            ORDER BY COALESCE(act.n_mes, 0) DESC, ope.op010
        """)
        rows = c.fetchall()
    finally:
        conn.close()

    result = []
    for r in rows:
        if activos and r[3] == 0:
            continue
        result.append({
            "codigo": r[0],
            "nombre": (r[1] or "").strip(),
            "activo": r[2] == 1,
            "n_registros_mes": r[3],
            "ultimo_registro": r[4].isoformat() if r[4] else None,
        })
    return {"operarios": result}


@router.get("/{codigo}")
def ficha_operario(
    codigo: int,
    desde: date = Query(None),
    hasta: date = Query(None),
):
    """
    Ficha completa de un operario con estadísticas de rendimiento.

    Parámetros:
    - codigo: código del operario en IZARO
    - desde/hasta: rango de fechas (por defecto último mes)
    """
    if hasta is None:
        hasta = date.today()
    if desde is None:
        desde = hasta - timedelta(days=30)

    conn = _connect()
    try:
        c = conn.cursor()

        # 1. Datos del operario
        c.execute("""
            SELECT RTRIM(op020), CAST(op060 AS INT), op055
            FROM admuser.fmesope
            WHERE op010 = ? AND op000 = 'ALGA'
        """, [codigo])
        row = c.fetchone()
        if not row:
            raise HTTPException(404, "Operario no encontrado")

        operario = {
            "codigo": codigo,
            "nombre": (row[0] or "").strip(),
            "activo": row[1] == 1,
            "dni": (row[2] or "").strip() if row[2] else "",
        }

        fi = desde.isoformat()
        fh = hasta.isoformat()

        # 2. Resumen por centro de trabajo
        c.execute("""
            SELECT
                CAST(dtc.dt150 AS INT) AS ct,
                RTRIM(rec.re020) AS nombre_ct,
                COUNT(*) AS registros,
                SUM(CASE WHEN dtc.dt110 = '0' THEN 1 ELSE 0 END) AS regs_produccion,
                SUM(CASE WHEN dtc.dt110 = '0' THEN CAST(dtc.dt130 AS FLOAT) ELSE 0 END) AS piezas,
                SUM(CASE WHEN dtc.dt110 = '0' THEN CAST(dtc.dt090 AS FLOAT) ELSE 0 END) AS horas_prod,
                SUM(CASE WHEN dtc.dt110 = '1' THEN CAST(dtc.dt090 AS FLOAT) ELSE 0 END) AS horas_prep,
                SUM(CASE WHEN dtc.dt110 = '2' THEN CAST(dtc.dt090 AS FLOAT) ELSE 0 END) AS horas_inci,
                COUNT(DISTINCT CONVERT(DATE, dtc.dt060)) AS dias_trabajados
            FROM admuser.fmesdtc dtc
            LEFT JOIN admuser.fmesrec rec
                ON rec.re010 = dtc.dt150 AND RTRIM(rec.re000) = RTRIM(dtc.dt000)
            WHERE RTRIM(dtc.dt140) = ?
                AND CONVERT(DATE, dtc.dt060) BETWEEN ? AND ?
            GROUP BY CAST(dtc.dt150 AS INT), RTRIM(rec.re020)
            ORDER BY piezas DESC
        """, [str(codigo), fi, fh])

        centros = []
        total_piezas = 0
        total_horas_prod = 0
        total_horas_prep = 0
        total_horas_inci = 0
        total_dias = set()

        for r in c.fetchall():
            piezas = float(r[4] or 0)
            horas_prod = float(r[5] or 0)
            centros.append({
                "codigo_ct": r[0],
                "nombre_ct": (r[1] or "").strip(),
                "registros": r[2],
                "regs_produccion": r[3],
                "piezas": piezas,
                "horas_produccion": round(horas_prod, 2),
                "horas_preparacion": round(float(r[6] or 0), 2),
                "horas_incidencia": round(float(r[7] or 0), 2),
                "dias_trabajados": r[8],
                "piezas_hora": round(piezas / horas_prod, 1) if horas_prod > 0 else 0,
            })
            total_piezas += piezas
            total_horas_prod += horas_prod
            total_horas_prep += float(r[6] or 0)
            total_horas_inci += float(r[7] or 0)

        # 3. Resumen por referencia (piezas/hora vs teórico)
        c.execute("""
            SELECT
                RTRIM(lof.lo030) AS referencia,
                CAST(dtc.dt150 AS INT) AS ct,
                RTRIM(rec.re020) AS nombre_ct,
                SUM(CAST(dtc.dt130 AS FLOAT)) AS piezas,
                SUM(CAST(dtc.dt090 AS FLOAT)) AS horas,
                COUNT(*) AS registros
            FROM admuser.fmesdtc dtc
            LEFT JOIN admuser.fprolof lof
                ON RTRIM(lof.lo010) = RTRIM(dtc.dt020) AND lof.lo020 = dtc.dt030
            LEFT JOIN admuser.fmesrec rec
                ON rec.re010 = dtc.dt150 AND RTRIM(rec.re000) = RTRIM(dtc.dt000)
            WHERE RTRIM(dtc.dt140) = ?
                AND CONVERT(DATE, dtc.dt060) BETWEEN ? AND ?
                AND dtc.dt110 = '0'
                AND RTRIM(lof.lo030) != ''
            GROUP BY RTRIM(lof.lo030), CAST(dtc.dt150 AS INT), RTRIM(rec.re020)
            ORDER BY piezas DESC
        """, [str(codigo), fi, fh])

        referencias = []
        for r in c.fetchall():
            pzas = float(r[3] or 0)
            hrs = float(r[4] or 0)
            referencias.append({
                "referencia": (r[0] or "").strip(),
                "codigo_ct": r[1],
                "nombre_ct": (r[2] or "").strip(),
                "piezas": pzas,
                "horas": round(hrs, 2),
                "piezas_hora": round(pzas / hrs, 1) if hrs > 0 else 0,
                "registros": r[5],
            })

        # 4. Evolución diaria
        c.execute("""
            SELECT
                CONVERT(DATE, dtc.dt060) AS dia,
                SUM(CASE WHEN dtc.dt110 = '0' THEN CAST(dtc.dt130 AS FLOAT) ELSE 0 END) AS piezas,
                SUM(CASE WHEN dtc.dt110 = '0' THEN CAST(dtc.dt090 AS FLOAT) ELSE 0 END) AS horas_prod,
                SUM(CASE WHEN dtc.dt110 = '1' THEN CAST(dtc.dt090 AS FLOAT) ELSE 0 END) AS horas_prep,
                SUM(CASE WHEN dtc.dt110 = '2' THEN CAST(dtc.dt090 AS FLOAT) ELSE 0 END) AS horas_inci,
                COUNT(*) AS registros
            FROM admuser.fmesdtc dtc
            WHERE RTRIM(dtc.dt140) = ?
                AND CONVERT(DATE, dtc.dt060) BETWEEN ? AND ?
            GROUP BY CONVERT(DATE, dtc.dt060)
            ORDER BY dia
        """, [str(codigo), fi, fh])

        evolucion = []
        for r in c.fetchall():
            pzas = float(r[1] or 0)
            hrs = float(r[2] or 0)
            evolucion.append({
                "fecha": r[0].isoformat(),
                "piezas": pzas,
                "horas_produccion": round(hrs, 2),
                "horas_preparacion": round(float(r[3] or 0), 2),
                "horas_incidencia": round(float(r[4] or 0), 2),
                "piezas_hora": round(pzas / hrs, 1) if hrs > 0 else 0,
                "registros": r[5],
            })

    finally:
        conn.close()

    # Cargar ciclos teóricos para comparar
    from api.database import Ciclo, SessionLocal
    with SessionLocal() as db:
        ciclos_db = db.query(Ciclo).all()
        ciclos_teoricos = {
            (c.maquina, c.referencia): c.tiempo_ciclo
            for c in ciclos_db
        }

    # Añadir teórico a referencias
    from api.database import SECTION_MAP
    # Mapeo inverso CT -> nombre recurso
    cfg = mes_service.get_config()
    ct_to_nombre = {r["centro_trabajo"]: r["nombre"] for r in cfg.get("recursos", [])}

    for ref in referencias:
        nombre_recurso = ct_to_nombre.get(ref["codigo_ct"], "")
        teorico = ciclos_teoricos.get((nombre_recurso, ref["referencia"]), 0)
        ref["piezas_hora_teorico"] = teorico
        ref["eficiencia_pct"] = round(
            (ref["piezas_hora"] / teorico) * 100, 1
        ) if teorico > 0 else None

    return {
        "operario": operario,
        "periodo": {"desde": desde.isoformat(), "hasta": hasta.isoformat()},
        "resumen": {
            "total_piezas": total_piezas,
            "total_horas_produccion": round(total_horas_prod, 2),
            "total_horas_preparacion": round(total_horas_prep, 2),
            "total_horas_incidencia": round(total_horas_inci, 2),
            "piezas_hora_media": round(total_piezas / total_horas_prod, 1) if total_horas_prod > 0 else 0,
            "n_centros": len(centros),
            "n_dias": len(evolucion),
        },
        "centros": centros,
        "referencias": referencias,
        "evolucion": evolucion,
    }
