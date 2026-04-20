"""Endpoints de rendimiento de operarios — consulta directa a IZARO.

Plan 03-02 Task 3.4: elimina ``_connect()`` (pyodbc crudo) y usa
``engine_mes`` (2-part names via DSN + SQLAlchemy) con queries
extraidas a ``nexo/data/sql/mes/operarios_*.sql``. La ficha detalle
mantiene tres queries inline parametrizadas con ``:codigo`` /
``:fecha_inicio`` / ``:fecha_fin`` (sus agregaciones eran monolithic
y no requieren `.sql` versionado por separado — D-01 scope).

Plan 04-02: preflight condicional por D-03 — si ``rango_dias > 90`` se
estima coste y se aplica gate en ``/operarios/{codigo}``. El endpoint
de listado (``GET /operarios``) no tiene rango asociado y pasa directo.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import text

from api.deps import DbNexo, EngineMes
from api.services import db as mes_service
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
    prefix="/operarios",
    tags=["operarios"],
    dependencies=[Depends(require_permission("operarios:read"))],
)


@router.get("")
def listar_operarios(engine_mes: EngineMes, activos: bool = True):
    """Lista operarios de IZARO.

    Devuelve ``codigo``, ``nombre``, ``activo``, y nº registros del
    ultimo mes. Con ``activos=True`` (default) se filtran los que no
    tienen actividad registrada en el periodo.
    """
    sql = text(load_sql("mes/operarios_listar"))
    with engine_mes.connect() as conn:
        rows = conn.execute(sql).fetchall()

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
    engine_mes: EngineMes,
    request: Request,
    db: DbNexo,
    desde: date = Query(None),
    hasta: date = Query(None),
    force: bool = Query(False),
    approval_id: Optional[int] = Query(None),
    user=Depends(require_permission("operarios:read")),
):
    """
    Ficha completa de un operario con estadísticas de rendimiento.

    Parámetros:
    - codigo: código del operario en IZARO
    - desde/hasta: rango de fechas (por defecto último mes)
    - force/approval_id (Plan 04-02): confirmación amber / aprobación red

    Plan 04-02: preflight condicional — sólo si ``rango_dias > 90``.
    """
    if hasta is None:
        hasta = date.today()
    if desde is None:
        desde = hasta - timedelta(days=30)

    fi = desde.isoformat()
    fh = hasta.isoformat()

    # ── Preflight condicional (D-03: solo rango > 90d) ──────────────────
    rango_dias = (hasta - desde).days
    if rango_dias > 90:
        params = {
            "codigo": codigo,
            "fecha_desde": fi,
            "fecha_hasta": fh,
            "rango_dias": rango_dias,
        }
        params_json_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
        est = estimate_cost("operarios", params)
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

    # CR-01 fix: gate pasó (o rango corto sin gate) — marcamos que sí
    # ejecutaremos para el middleware postflight.
    request.state.query_executed = True

    with engine_mes.connect() as conn:
        # 1. Datos del operario
        row = conn.execute(
            text(load_sql("mes/operarios_ficha")),
            {"codigo": codigo},
        ).fetchone()
        if not row:
            raise HTTPException(404, "Operario no encontrado")

        operario = {
            "codigo": codigo,
            "nombre": (row[0] or "").strip(),
            "activo": row[1] == 1,
            "dni": (row[2] or "").strip() if row[2] else "",
        }

        # 2. Resumen por centro de trabajo
        centros_rows = conn.execute(
            text("""
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
                WHERE RTRIM(dtc.dt140) = :codigo_txt
                    AND CONVERT(DATE, dtc.dt060) BETWEEN :fi AND :fh
                GROUP BY CAST(dtc.dt150 AS INT), RTRIM(rec.re020)
                ORDER BY piezas DESC
            """),
            {"codigo_txt": str(codigo), "fi": fi, "fh": fh},
        ).fetchall()

        centros = []
        total_piezas = 0.0
        total_horas_prod = 0.0
        total_horas_prep = 0.0
        total_horas_inci = 0.0

        for r in centros_rows:
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
        refs_rows = conn.execute(
            text("""
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
                WHERE RTRIM(dtc.dt140) = :codigo_txt
                    AND CONVERT(DATE, dtc.dt060) BETWEEN :fi AND :fh
                    AND dtc.dt110 = '0'
                    AND RTRIM(lof.lo030) != ''
                GROUP BY RTRIM(lof.lo030), CAST(dtc.dt150 AS INT), RTRIM(rec.re020)
                ORDER BY piezas DESC
            """),
            {"codigo_txt": str(codigo), "fi": fi, "fh": fh},
        ).fetchall()

        referencias = []
        for r in refs_rows:
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
        evo_rows = conn.execute(
            text("""
                SELECT
                    CONVERT(DATE, dtc.dt060) AS dia,
                    SUM(CASE WHEN dtc.dt110 = '0' THEN CAST(dtc.dt130 AS FLOAT) ELSE 0 END) AS piezas,
                    SUM(CASE WHEN dtc.dt110 = '0' THEN CAST(dtc.dt090 AS FLOAT) ELSE 0 END) AS horas_prod,
                    SUM(CASE WHEN dtc.dt110 = '1' THEN CAST(dtc.dt090 AS FLOAT) ELSE 0 END) AS horas_prep,
                    SUM(CASE WHEN dtc.dt110 = '2' THEN CAST(dtc.dt090 AS FLOAT) ELSE 0 END) AS horas_inci,
                    COUNT(*) AS registros
                FROM admuser.fmesdtc dtc
                WHERE RTRIM(dtc.dt140) = :codigo_txt
                    AND CONVERT(DATE, dtc.dt060) BETWEEN :fi AND :fh
                GROUP BY CONVERT(DATE, dtc.dt060)
                ORDER BY dia
            """),
            {"codigo_txt": str(codigo), "fi": fi, "fh": fh},
        ).fetchall()

        evolucion = []
        for r in evo_rows:
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

    # Cargar ciclos teóricos para comparar
    from api.database import Ciclo, SessionLocal
    with SessionLocal() as db:
        ciclos_db = db.query(Ciclo).all()
        ciclos_teoricos = {
            (c.maquina, c.referencia): c.tiempo_ciclo
            for c in ciclos_db
        }

    # Añadir teórico a referencias
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
