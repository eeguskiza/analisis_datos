"""API del panel Pabellon 5 — datos LUK4 + fmesmic."""

from __future__ import annotations

import logging
from calendar import monthrange
from datetime import datetime

from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text

from nexo.data.engines import engine_app as engine
from api.services.turnos import (
    get_jornada_start,
    get_turno_actual,
    get_turno_boundaries,
    turno_from_hour,
)
from nexo.services.auth import require_permission

router = APIRouter(
    prefix="/luk4",
    tags=["luk4"],
    dependencies=[Depends(require_permission("luk4:read"))],
)
log = logging.getLogger(__name__)

# Registros/hora estimados en luk4.estado (~443 segun analisis)
_EST_RECORDS_PER_HOUR = 443


@router.get("/status")
def luk4_status():
    """Estado en tiempo real de LUK4 y zonas del pabellon."""
    telemetria = {}
    luk4_status_str = "stopped"
    alarma = None

    now = datetime.now()
    shift_start = get_jornada_start(now)
    turno_actual = get_turno_actual(now)

    try:
        with engine.connect() as conn:
            # ── LUK4 estado + alarma activa ─────────────────────────────
            est = conn.execute(
                text("""
                SELECT TOP 1 e.timestamp, e.estado_global, e.codigo_error,
                       e.porcentaje_manual, e.porcentaje_auto, e.porcentaje_error,
                       a.componente, a.mensaje
                FROM luk4.estado e
                LEFT JOIN luk4.alarmas a ON a.codigo = e.codigo_error
                ORDER BY e.idcelula_estado DESC
            """)
            ).fetchone()

            if est:
                estado_global = int(est[1]) if est[1] is not None else 0
                codigo_error = int(est[2]) if est[2] is not None else 0
                telemetria["estado_global"] = estado_global
                telemetria["codigo_error"] = codigo_error
                telemetria["pct_auto"] = (
                    round(float(est[4]) / 100, 1) if est[4] else None
                )
                telemetria["pct_error"] = (
                    round(float(est[5]) / 100, 1) if est[5] else None
                )
                telemetria["estado_ts"] = (
                    est[0].strftime("%H:%M:%S") if est[0] else None
                )

                # Color de LUK4 basado en estado_global (codigo_error es sticky, no indica error activo)
                if estado_global == 3:
                    luk4_status_str = "incidence"
                    comp = est[6] or "SISTEMA"
                    msg = est[7] or f"Error activo (codigo {codigo_error})"
                    alarma = {
                        "componente": comp,
                        "mensaje": msg,
                        "codigo": codigo_error,
                    }
                    telemetria["alarma_componente"] = comp
                    telemetria["alarma_mensaje"] = msg
                elif estado_global == 1:
                    luk4_status_str = "producing"
                else:
                    luk4_status_str = "stopped"

            # ── LUK4 tiempos de ciclo ───────────────────────────────────
            tc = conn.execute(
                text("""
                SELECT TOP 1 timestamp, tiempo_ciclo_total, tiempo_ciclo_temple,
                       tiempo_ciclo_revenido, tiempo_ciclo_torno
                FROM luk4.tiempos_ciclo ORDER BY idtiempos_ciclo DESC
            """)
            ).fetchone()
            if tc:
                ct = tc[1] / 1000 if tc[1] else 0
                telemetria["ciclo_total"] = round(ct, 1) if ct else None
                telemetria["ciclo_temple"] = round(tc[2] / 1000, 1) if tc[2] else None
                telemetria["ciclo_revenido"] = round(tc[3] / 1000, 1) if tc[3] else None
                telemetria["ciclo_torno"] = round(tc[4] / 1000, 1) if tc[4] else None
                telemetria["pph_total"] = round(3600 / ct) if ct > 0 else None
                telemetria["pph_temple"] = (
                    round(3600 / (tc[2] / 1000)) if tc[2] and tc[2] > 0 else None
                )
                telemetria["pph_revenido"] = (
                    round(3600 / (tc[3] / 1000)) if tc[3] and tc[3] > 0 else None
                )
                telemetria["pph_torno"] = (
                    round(3600 / (tc[4] / 1000)) if tc[4] and tc[4] > 0 else None
                )

            # ── LUK4 piezas del dia (delta contadores) ──────────────────
            piezas = conn.execute(
                text("""
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
            """),
                {"shift_start": shift_start},
            ).fetchone()
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
            rows = conn.execute(
                text("""
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
            """),
                {"shift_start": shift_start},
            ).fetchall()
            for r in rows:

                def pph(ms):
                    return round(3600 / (ms / 1000), 0) if ms and ms > 0 else 0

                h = int(r[0])
                turno = turno_from_hour(h)
                timeline.append(
                    {
                        "t": f"{h:02d}:{int(r[1]):02d}",
                        "total": pph(r[2]),
                        "temple": pph(r[3]),
                        "revenido": pph(r[4]),
                        "torno": pph(r[5]),
                        "turno": turno,
                    }
                )

            # Top alarmas del dia
            rows = conn.execute(
                text("""
                SELECT TOP 5 e.codigo_error, a.componente, a.mensaje, COUNT(*) as n,
                       MAX(e.timestamp) as ultimo
                FROM luk4.estado e
                LEFT JOIN luk4.alarmas a ON a.codigo = e.codigo_error
                WHERE e.timestamp >= :shift_start
                  AND e.codigo_error > 0
                GROUP BY e.codigo_error, a.componente, a.mensaje
                ORDER BY n DESC
            """),
                {"shift_start": shift_start},
            ).fetchall()
            for r in rows:
                top_alarmas.append(
                    {
                        "codigo": int(r[0]),
                        "componente": r[1] or "SISTEMA",
                        "mensaje": r[2] or f"Error {r[0]}",
                        "count": int(r[3]),
                        "ultimo": r[4].strftime("%H:%M") if r[4] else None,
                    }
                )
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


# ── Detalle por turno (barras + alarmas) ─────────────────────────────────


@router.get("/turno-detail")
def turno_detail():
    """Datos para el panel de detalle: barras de disponibilidad y alarmas por turno."""
    now = datetime.now()
    jornada = get_jornada_start(now)
    turno_actual = get_turno_actual(now)
    boundaries = get_turno_boundaries(now)

    turnos_breakdown = []
    alarmas_turno: list[dict] = []
    alarmas_summary = {
        "total_count": 0,
        "distinct_codes": 0,
        "estimated_stop_minutes": 0,
    }

    def _elapsed_pct(t_start: datetime, t_end: datetime, is_future: bool) -> float:
        """% del turno transcurrido (0-100). Turnos de 8h."""
        if is_future:
            return 0.0
        elapsed = (t_end - t_start).total_seconds()
        return min(round(elapsed / (8 * 3600) * 100, 1), 100.0)

    try:
        with engine.connect() as conn:
            for name, t_start, t_end in boundaries:
                is_future = t_start > now
                ep = _elapsed_pct(t_start, t_end, is_future)

                if is_future:
                    turnos_breakdown.append(
                        {
                            "turno": name,
                            "start": t_start.strftime("%H:%M"),
                            "end": t_end.strftime("%H:%M"),
                            "is_current": False,
                            "is_future": True,
                            "elapsed_pct": ep,
                            "total_records": 0,
                            "producing": 0,
                            "incidence": 0,
                            "off": 0,
                            "other": 0,
                            "availability_pct": 0,
                            "pz_buenas": 0,
                            "pz_malas": 0,
                            "pz_totales": 0,
                        }
                    )
                    continue

                # Estado + piezas en 2 queries (directo por timestamp, sin PK intermedio)
                params = {"t_start": t_start, "t_end": t_end}

                rows = conn.execute(
                    text("""
                    SELECT estado_global, COUNT(*) as n
                    FROM luk4.estado
                    WHERE timestamp >= :t_start AND timestamp < :t_end
                    GROUP BY estado_global
                """),
                    params,
                ).fetchall()

                if not rows:
                    turnos_breakdown.append(
                        {
                            "turno": name,
                            "start": t_start.strftime("%H:%M"),
                            "end": t_end.strftime("%H:%M"),
                            "is_current": name == turno_actual,
                            "is_future": False,
                            "elapsed_pct": ep,
                            "total_records": 0,
                            "producing": 0,
                            "incidence": 0,
                            "off": 0,
                            "other": 0,
                            "availability_pct": 0,
                            "pz_buenas": 0,
                            "pz_malas": 0,
                            "pz_totales": 0,
                        }
                    )
                    continue

                counts = {int(r[0]): int(r[1]) for r in rows}
                producing = counts.get(1, 0)
                incidence = counts.get(3, 0)
                off = counts.get(0, 0)
                other = sum(v for k, v in counts.items() if k not in (0, 1, 3))
                total = producing + incidence + off + other
                avail = round(100 * producing / total, 1) if total > 0 else 0

                # Baseline = ultima lectura ANTES del turno para cerrar la cadena
                # sin perder piezas en el hueco de muestreo de la frontera.
                # Fallback: primera lectura dentro del turno (primer turno historico).
                pz = conn.execute(
                    text("""
                    SELECT
                        COALESCE(
                            (SELECT TOP 1 contador_piezas_buenas FROM luk4.tiempos_ciclo
                             WHERE timestamp < :t_start AND contador_piezas_buenas IS NOT NULL
                             ORDER BY timestamp DESC),
                            (SELECT TOP 1 contador_piezas_buenas FROM luk4.tiempos_ciclo
                             WHERE timestamp >= :t_start AND contador_piezas_buenas IS NOT NULL
                             ORDER BY timestamp ASC)
                        ),
                        (SELECT TOP 1 contador_piezas_buenas FROM luk4.tiempos_ciclo
                         WHERE timestamp < :t_end AND contador_piezas_buenas IS NOT NULL
                         ORDER BY timestamp DESC),
                        COALESCE(
                            (SELECT TOP 1 contador_piezas_malas FROM luk4.tiempos_ciclo
                             WHERE timestamp < :t_start AND contador_piezas_malas IS NOT NULL
                             ORDER BY timestamp DESC),
                            (SELECT TOP 1 contador_piezas_malas FROM luk4.tiempos_ciclo
                             WHERE timestamp >= :t_start AND contador_piezas_malas IS NOT NULL
                             ORDER BY timestamp ASC)
                        ),
                        (SELECT TOP 1 contador_piezas_malas FROM luk4.tiempos_ciclo
                         WHERE timestamp < :t_end AND contador_piezas_malas IS NOT NULL
                         ORDER BY timestamp DESC)
                """),
                    params,
                ).fetchone()
                pz_buenas = (
                    int(pz[1] - pz[0])
                    if pz and pz[0] is not None and pz[1] is not None
                    else 0
                )
                pz_malas = (
                    int(pz[3] - pz[2])
                    if pz and pz[2] is not None and pz[3] is not None
                    else 0
                )

                turnos_breakdown.append(
                    {
                        "turno": name,
                        "start": t_start.strftime("%H:%M"),
                        "end": t_end.strftime("%H:%M"),
                        "is_current": name == turno_actual,
                        "is_future": False,
                        "elapsed_pct": ep,
                        "total_records": total,
                        "producing": producing,
                        "incidence": incidence,
                        "off": off,
                        "other": other,
                        "availability_pct": avail,
                        "pz_buenas": max(pz_buenas, 0),
                        "pz_malas": max(pz_malas, 0),
                        "pz_totales": max(pz_buenas + pz_malas, 0),
                    }
                )

                # Alarmas solo del turno actual
                if name == turno_actual:
                    alarm_rows = conn.execute(
                        text("""
                        SELECT e.codigo_error, a.componente, a.mensaje,
                               COUNT(*) as n,
                               MIN(e.timestamp) as first_ts,
                               MAX(e.timestamp) as last_ts
                        FROM luk4.estado e
                        LEFT JOIN luk4.alarmas a ON a.codigo = e.codigo_error
                        WHERE e.timestamp >= :t_start AND e.timestamp < :t_end
                          AND e.codigo_error > 0
                        GROUP BY e.codigo_error, a.componente, a.mensaje
                        ORDER BY n DESC
                    """),
                        params,
                    ).fetchall()

                    total_alarm_records = 0
                    for ar in alarm_rows:
                        cnt = int(ar[3])
                        total_alarm_records += cnt
                        est_min = round(cnt * (3600 / _EST_RECORDS_PER_HOUR) / 60, 1)
                        alarmas_turno.append(
                            {
                                "codigo": int(ar[0]),
                                "componente": ar[1] or "SISTEMA",
                                "mensaje": ar[2] or f"Error {ar[0]}",
                                "count": cnt,
                                "first_seen": ar[4].strftime("%H:%M")
                                if ar[4]
                                else None,
                                "estimated_minutes": est_min,
                            }
                        )

                    alarmas_summary = {
                        "total_count": total_alarm_records,
                        "distinct_codes": len(alarm_rows),
                        "estimated_stop_minutes": round(
                            total_alarm_records * (3600 / _EST_RECORDS_PER_HOUR) / 60, 0
                        ),
                    }

    except Exception as exc:
        log.warning("LUK4 turno-detail: %s", exc)

    return {
        "turno_actual": turno_actual,
        "jornada": jornada.strftime("%Y-%m-%d"),
        "turnos_breakdown": turnos_breakdown,
        "alarmas_turno": alarmas_turno,
        "alarmas_summary": alarmas_summary,
        "timestamp": now.strftime("%H:%M:%S"),
    }


@router.get("/month-daily")
def month_daily():
    """Serie diaria de piezas LUK4 del mes actual hasta hoy."""
    now = datetime.now()
    month_start = datetime(now.year, now.month, 1)
    if now.month == 12:
        month_end = datetime(now.year + 1, 1, 1)
    else:
        month_end = datetime(now.year, now.month + 1, 1)

    pieces_by_day: dict[datetime.date, int] = {}
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                WITH daily_counter AS (
                    SELECT
                        CAST([timestamp] AS date) AS d,
                        contador_piezas_totales,
                        ROW_NUMBER() OVER (
                            PARTITION BY CAST([timestamp] AS date)
                            ORDER BY [timestamp] ASC, idtiempos_ciclo ASC
                        ) AS rn_first,
                        ROW_NUMBER() OVER (
                            PARTITION BY CAST([timestamp] AS date)
                            ORDER BY [timestamp] DESC, idtiempos_ciclo DESC
                        ) AS rn_last
                    FROM luk4.tiempos_ciclo
                    WHERE [timestamp] >= :month_start
                      AND [timestamp] < :month_end
                      AND contador_piezas_totales IS NOT NULL
                )
                SELECT
                    d,
                    MAX(CASE WHEN rn_first = 1 THEN contador_piezas_totales END) AS first_total,
                    MAX(CASE WHEN rn_last = 1 THEN contador_piezas_totales END) AS last_total
                FROM daily_counter
                GROUP BY d
                ORDER BY d ASC
            """),
                {"month_start": month_start, "month_end": month_end},
            ).fetchall()

            for row in rows:
                day = row[0]
                first_total = int(row[1] or 0)
                last_total = int(row[2] or 0)
                pieces_by_day[day] = max(last_total - first_total, 0)
    except Exception as exc:
        log.warning("LUK4 month-daily: %s", exc)

    labels: list[str] = []
    values: list[int] = []
    total_month = 0
    for day_num in range(1, now.day + 1):
        day_date = month_start.date().replace(day=day_num)
        day_value = int(pieces_by_day.get(day_date, 0))
        labels.append(f"{day_num:02d}")
        values.append(day_value)
        total_month += day_value

    return {
        "month": now.month,
        "year": now.year,
        "days_in_month": monthrange(now.year, now.month)[1],
        "labels": labels,
        "values": values,
        "total_month": total_month,
        "avg_per_day": round(total_month / max(len(values), 1), 1),
    }


# ── Zonas del plano (persistidas en BBDD) ───────────────────────────────

_VALID_PABELLONES = {"p2", "p3", "p4", "p5"}
_schema_ready = False


def _ensure_pabellon_schema() -> None:
    """Migracion idempotente: anade columna pabellon a luk4.plano_zonas si falta."""
    global _schema_ready
    if _schema_ready:
        return
    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                IF COL_LENGTH('luk4.plano_zonas', 'pabellon') IS NULL
                BEGIN
                    ALTER TABLE luk4.plano_zonas
                    ADD pabellon NVARCHAR(10) NOT NULL
                    CONSTRAINT DF_plano_zonas_pabellon DEFAULT 'p5' WITH VALUES;
                END
            """)
            )
            conn.commit()
        _schema_ready = True
    except Exception as exc:
        log.warning("No pude asegurar columna pabellon: %s", exc)


def _validar_pabellon(pab: str) -> str:
    p = (pab or "p5").lower()
    if p not in _VALID_PABELLONES:
        p = "p5"
    return p


class ZonaIn(BaseModel):
    id: str
    label: str
    left: float
    top: float
    width: float
    height: float
    source: str = "none"


@router.get("/zonas")
def get_zonas(pabellon: str = "p5"):
    """Devuelve las zonas guardadas del plano para el pabellon indicado."""
    _ensure_pabellon_schema()
    pab = _validar_pabellon(pabellon)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                SELECT id, label, left_pct, top_pct, width_pct, height_pct, source
                FROM luk4.plano_zonas
                WHERE pabellon = :pab
                ORDER BY label
            """),
                {"pab": pab},
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "label": r[1],
                    "left": r[2],
                    "top": r[3],
                    "width": r[4],
                    "height": r[5],
                    "source": r[6],
                }
                for r in rows
            ]
    except Exception as exc:
        log.warning("Error cargando zonas (%s): %s", pab, exc)
        return []


@router.put("/zonas")
def save_zonas(zonas: List[ZonaIn], pabellon: str = "p5"):
    """Reemplaza las zonas del pabellon indicado sin tocar las de otros."""
    _ensure_pabellon_schema()
    pab = _validar_pabellon(pabellon)
    try:
        with engine.connect() as conn:
            conn.execute(
                text("DELETE FROM luk4.plano_zonas WHERE pabellon = :pab"), {"pab": pab}
            )
            for z in zonas:
                conn.execute(
                    text("""
                    INSERT INTO luk4.plano_zonas
                        (id, pabellon, label, left_pct, top_pct, width_pct, height_pct, source)
                    VALUES (:id, :pab, :label, :left, :top, :width, :height, :source)
                """),
                    {
                        "id": z.id,
                        "pab": pab,
                        "label": z.label,
                        "left": z.left,
                        "top": z.top,
                        "width": z.width,
                        "height": z.height,
                        "source": z.source,
                    },
                )
            conn.commit()
        return {"ok": True, "count": len(zonas), "pabellon": pab}
    except Exception as exc:
        log.warning("Error guardando zonas (%s): %s", pab, exc)
        return {"ok": False, "error": str(exc)}
