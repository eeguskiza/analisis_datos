"""API del panel Pabellon 5 — datos LUK4 + fmesmic."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

from api.database import engine

router = APIRouter(prefix="/luk4", tags=["luk4"])
log = logging.getLogger(__name__)


@router.get("/status")
def luk4_status():
    """Estado en tiempo real de LUK4 y zonas del pabellon."""
    telemetria = {}
    luk4_status_str = "stopped"
    alarma = None

    # Jornada: de 6:00 a 6:00 del dia siguiente
    now = datetime.now()
    if now.hour < 6:
        shift_start = (now - timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)
    else:
        shift_start = now.replace(hour=6, minute=0, second=0, microsecond=0)

    if 6 <= now.hour < 14:
        turno_actual = "T1"
    elif 14 <= now.hour < 22:
        turno_actual = "T2"
    else:
        turno_actual = "T3"

    try:
        with engine.connect() as conn:
            # ── LUK4 estado + alarma activa ─────────────────────────────
            est = conn.execute(text("""
                SELECT TOP 1 e.timestamp, e.estado_global, e.codigo_error,
                       e.porcentaje_manual, e.porcentaje_auto, e.porcentaje_error,
                       a.componente, a.mensaje
                FROM luk4.estado e
                LEFT JOIN luk4.alarmas a ON a.codigo = e.codigo_error
                ORDER BY e.idcelula_estado DESC
            """)).fetchone()

            if est:
                estado_global = int(est[1]) if est[1] is not None else 0
                codigo_error = int(est[2]) if est[2] is not None else 0
                telemetria["estado_global"] = estado_global
                telemetria["codigo_error"] = codigo_error
                telemetria["pct_auto"] = round(float(est[4]) / 100, 1) if est[4] else None
                telemetria["pct_error"] = round(float(est[5]) / 100, 1) if est[5] else None
                telemetria["estado_ts"] = est[0].strftime("%H:%M:%S") if est[0] else None

                # Color de LUK4 basado en estado_global (codigo_error es sticky, no indica error activo)
                if estado_global == 3:
                    luk4_status_str = "incidence"
                    comp = est[6] or "SISTEMA"
                    msg = est[7] or f"Error activo (codigo {codigo_error})"
                    alarma = {"componente": comp, "mensaje": msg, "codigo": codigo_error}
                    telemetria["alarma_componente"] = comp
                    telemetria["alarma_mensaje"] = msg
                elif estado_global == 1:
                    luk4_status_str = "producing"
                else:
                    luk4_status_str = "stopped"

            # ── LUK4 tiempos de ciclo ───────────────────────────────────
            tc = conn.execute(text("""
                SELECT TOP 1 timestamp, tiempo_ciclo_total, tiempo_ciclo_temple,
                       tiempo_ciclo_revenido, tiempo_ciclo_torno
                FROM luk4.tiempos_ciclo ORDER BY idtiempos_ciclo DESC
            """)).fetchone()
            if tc:
                ct = tc[1] / 1000 if tc[1] else 0
                telemetria["ciclo_total"] = round(ct, 1) if ct else None
                telemetria["ciclo_temple"] = round(tc[2] / 1000, 1) if tc[2] else None
                telemetria["ciclo_revenido"] = round(tc[3] / 1000, 1) if tc[3] else None
                telemetria["ciclo_torno"] = round(tc[4] / 1000, 1) if tc[4] else None
                telemetria["pph_total"] = round(3600 / ct) if ct > 0 else None
                telemetria["pph_temple"] = round(3600 / (tc[2] / 1000)) if tc[2] and tc[2] > 0 else None
                telemetria["pph_revenido"] = round(3600 / (tc[3] / 1000)) if tc[3] and tc[3] > 0 else None
                telemetria["pph_torno"] = round(3600 / (tc[4] / 1000)) if tc[4] and tc[4] > 0 else None

            # ── LUK4 piezas del dia (delta contadores) ──────────────────
            piezas = conn.execute(text("""
                SELECT
                    (SELECT TOP 1 contador_piezas_buenas FROM luk4.tiempos_ciclo
                     WHERE timestamp >= :shift_start AND contador_piezas_buenas IS NOT NULL
                     ORDER BY idtiempos_ciclo ASC) AS first_buenas,
                    (SELECT TOP 1 contador_piezas_buenas FROM luk4.tiempos_ciclo
                     WHERE timestamp >= :shift_start AND contador_piezas_buenas IS NOT NULL
                     ORDER BY idtiempos_ciclo DESC) AS last_buenas,
                    (SELECT TOP 1 contador_piezas_malas FROM luk4.tiempos_ciclo
                     WHERE timestamp >= :shift_start AND contador_piezas_malas IS NOT NULL
                     ORDER BY idtiempos_ciclo ASC) AS first_malas,
                    (SELECT TOP 1 contador_piezas_malas FROM luk4.tiempos_ciclo
                     WHERE timestamp >= :shift_start AND contador_piezas_malas IS NOT NULL
                     ORDER BY idtiempos_ciclo DESC) AS last_malas,
                    (SELECT TOP 1 contador_piezas_totales FROM luk4.tiempos_ciclo
                     WHERE timestamp >= :shift_start AND contador_piezas_totales IS NOT NULL
                     ORDER BY idtiempos_ciclo DESC) AS last_totales
            """), {"shift_start": shift_start}).fetchone()
            if piezas and piezas[0] is not None:
                telemetria["piezas_buenas_hoy"] = int(piezas[1] - piezas[0])
                telemetria["piezas_malas_hoy"] = int(piezas[3] - piezas[2])
            else:
                telemetria["piezas_buenas_hoy"] = 0
                telemetria["piezas_malas_hoy"] = 0

    except Exception as exc:
        log.warning("LUK4: error: %s", exc)

    # ── Timeline pz/h por proceso (cada 5 min) ────────────────────
    timeline = []
    top_alarmas = []
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT DATEPART(HOUR, timestamp) as h,
                       (DATEPART(MINUTE, timestamp) / 5) * 5 as m5,
                       AVG(tiempo_ciclo_total) as tc_total,
                       AVG(tiempo_ciclo_temple) as tc_temple,
                       AVG(tiempo_ciclo_revenido) as tc_rev,
                       AVG(tiempo_ciclo_torno) as tc_torno
                FROM luk4.tiempos_ciclo
                WHERE timestamp >= :shift_start
                  AND tiempo_ciclo_total IS NOT NULL
                GROUP BY DATEPART(HOUR, timestamp), (DATEPART(MINUTE, timestamp) / 5) * 5
                ORDER BY (DATEPART(HOUR, timestamp) + 18) % 24, m5
            """), {"shift_start": shift_start}).fetchall()
            for r in rows:
                def pph(ms): return round(3600 / (ms / 1000), 0) if ms and ms > 0 else 0
                h = int(r[0])
                turno = "T1" if 6 <= h < 14 else "T2" if 14 <= h < 22 else "T3"
                timeline.append({
                    "t": f"{h:02d}:{int(r[1]):02d}",
                    "total": pph(r[2]),
                    "temple": pph(r[3]),
                    "revenido": pph(r[4]),
                    "torno": pph(r[5]),
                    "turno": turno,
                })

            # Top alarmas del dia
            rows = conn.execute(text("""
                SELECT TOP 5 e.codigo_error, a.componente, a.mensaje, COUNT(*) as n,
                       MAX(e.timestamp) as ultimo
                FROM luk4.estado e
                LEFT JOIN luk4.alarmas a ON a.codigo = e.codigo_error
                WHERE e.timestamp >= :shift_start
                  AND e.codigo_error > 0
                GROUP BY e.codigo_error, a.componente, a.mensaje
                ORDER BY n DESC
            """), {"shift_start": shift_start}).fetchall()
            for r in rows:
                top_alarmas.append({
                    "codigo": int(r[0]),
                    "componente": r[1] or "SISTEMA",
                    "mensaje": r[2] or f"Error {r[0]}",
                    "count": int(r[3]),
                    "ultimo": r[4].strftime("%H:%M") if r[4] else None,
                })
    except Exception as exc:
        log.warning("LUK4 timeline: %s", exc)

    return {
        "luk4_status": luk4_status_str,
        "alarma": alarma,
        "telemetria": telemetria,
        "timeline": timeline,
        "top_alarmas": top_alarmas,
        "turno_actual": turno_actual,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }


# ── Zonas del plano (persistidas en BBDD) ───────────────────────────────

class ZonaIn(BaseModel):
    id: str
    label: str
    left: float
    top: float
    width: float
    height: float
    source: str = "none"


@router.get("/zonas")
def get_zonas():
    """Devuelve las zonas guardadas del plano."""
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT id, label, left_pct, top_pct, width_pct, height_pct, source FROM luk4.plano_zonas ORDER BY label"
            )).fetchall()
            return [
                {"id": r[0], "label": r[1], "left": r[2], "top": r[3],
                 "width": r[4], "height": r[5], "source": r[6]}
                for r in rows
            ]
    except Exception as exc:
        log.warning("Error cargando zonas: %s", exc)
        return []


@router.put("/zonas")
def save_zonas(zonas: List[ZonaIn]):
    """Guarda las zonas del plano (reemplaza todas)."""
    try:
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM luk4.plano_zonas"))
            for z in zonas:
                conn.execute(text("""
                    INSERT INTO luk4.plano_zonas (id, label, left_pct, top_pct, width_pct, height_pct, source)
                    VALUES (:id, :label, :left, :top, :width, :height, :source)
                """), {"id": z.id, "label": z.label, "left": z.left, "top": z.top,
                       "width": z.width, "height": z.height, "source": z.source})
            conn.commit()
        return {"ok": True, "count": len(zonas)}
    except Exception as exc:
        log.warning("Error guardando zonas: %s", exc)
        return {"ok": False, "error": str(exc)}
