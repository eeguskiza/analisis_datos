from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle

from OEE.utils.data_files import RESOURCE_DIR_NAME

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[2] / "informes" / "oee_secciones"
SHIFT_LABELS = ("T1", "T2", "T3")
REF_ROWS_PER_PAGE = 28
ORDER_PRIORITY = {"luk1": 0, "luk2": 1, "luk3": 2, "luk6": 3, "coroa": 4, "vw1": 5}


def machine_sort_key(machine: "MachineSectionMetrics") -> tuple[int, str]:
    """Orden preferente para LINEAS y, si no aplica, orden alfabético."""
    name_lower = machine.name.lower()
    return (ORDER_PRIORITY.get(name_lower, 999), name_lower)


@dataclass
class MachineSectionMetrics:
    name: str
    full_name: str
    week_code: Optional[str]
    horas_brutas: float
    horas_produccion: float
    horas_incidencias: float
    horas_preparacion: float
    tiempo_ideal: float
    perdidas_rend: float
    piezas_totales: float
    piezas_malas: float
    piezas_recuperadas: float
    buenas_finales: float
    disponibilidad_pct: float
    rendimiento_pct: float
    calidad_pct: float
    oee_pct: float
    start: Optional[datetime]
    end: Optional[datetime]
    daily_stats: Dict[date, Dict[str, float]] = field(default_factory=dict)
    shift_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    daily_shift_stats: Dict[date, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    ref_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)


def parse_float(value: Optional[str]) -> float:
    if value is None or value == "":
        return 0.0
    value = value.replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return 0.0


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return None


def normalizar_proceso(value: Optional[str]) -> str:
    if not value:
        return "produccion"
    text = value.strip().lower()
    if text.startswith("produ"):
        return "produccion"
    if text.startswith("prepa"):
        return "preparacion"
    if text.startswith("inci"):
        return "incidencias"
    return "produccion"


def normalize_ref(value: Optional[str]) -> str:
    if not value:
        return ""
    text = str(value).strip().lower()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def crear_raw_metricas() -> Dict[str, float]:
    return {
        "horas_produccion": 0.0,
        "horas_incidencias": 0.0,
        "horas_preparacion": 0.0,
        "tiempo_ideal": 0.0,
        "piezas_totales": 0.0,
        "piezas_malas": 0.0,
        "piezas_recuperadas": 0.0,
    }


def convertir_raw_a_metricas(raw: Dict[str, float]) -> Dict[str, float]:
    horas_produccion = raw.get("horas_produccion", 0.0)
    horas_incidencias = raw.get("horas_incidencias", 0.0)
    horas_preparacion = raw.get("horas_preparacion", 0.0)
    horas_brutas = horas_produccion + horas_incidencias + horas_preparacion
    tiempo_ideal = raw.get("tiempo_ideal", 0.0)
    perdidas_rend = max(horas_produccion - tiempo_ideal, 0.0)
    piezas_totales = raw.get("piezas_totales", 0.0)
    piezas_malas = raw.get("piezas_malas", 0.0)
    piezas_recuperadas = raw.get("piezas_recuperadas", 0.0)
    scrap_final = max(piezas_malas - piezas_recuperadas, 0.0)
    buenas_finales = max(piezas_totales - scrap_final, 0.0)
    disponibilidad_pct = clamp_pct((horas_produccion / horas_brutas * 100) if horas_brutas > 0 else 0.0)
    rendimiento_pct = clamp_pct(
        (tiempo_ideal / horas_produccion * 100) if horas_produccion > 0 and tiempo_ideal > 0 else 0.0
    )
    calidad_pct = clamp_pct((buenas_finales / piezas_totales * 100) if piezas_totales > 0 else 0.0)
    oee_pct = (disponibilidad_pct * rendimiento_pct * calidad_pct) / 10000.0
    return {
        "horas_brutas": horas_brutas,
        "horas_produccion": horas_produccion,
        "horas_incidencias": horas_incidencias,
        "horas_preparacion": horas_preparacion,
        "tiempo_ideal": tiempo_ideal,
        "perdidas_rend": perdidas_rend,
        "buenas_finales": buenas_finales,
        "piezas_totales": piezas_totales,
        "piezas_malas": piezas_malas,
        "piezas_recuperadas": piezas_recuperadas,
        "disponibilidad_pct": disponibilidad_pct,
        "rendimiento_pct": rendimiento_pct,
        "calidad_pct": calidad_pct,
        "oee_pct": oee_pct,
    }


def obtener_turno_y_limite(dt: datetime) -> tuple[str, datetime, date]:
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    base_day = dt.date()
    if 6 <= hour < 14:
        return "T1", datetime.combine(base_day, time(14, 0)), base_day
    if 14 <= hour < 22:
        return "T2", datetime.combine(base_day, time(22, 0)), base_day
    # T3 de 22:00 a 06:00 (madrugada siguiente). Asignamos el día al inicio del turno (día anterior si < 06:00).
    limit_day = base_day if hour < 22 else base_day + timedelta(days=1)
    return "T3", datetime.combine(limit_day, time(6, 0)), base_day


def dividir_en_turnos(start_dt: datetime, duration_hours: float) -> List[tuple[str, date, float]]:
    if not start_dt or duration_hours <= 0:
        return []
    segments: List[tuple[str, date, float]] = []
    end_dt = start_dt + timedelta(hours=duration_hours)
    current = start_dt
    while current < end_dt:
        shift, limit, day_key = obtener_turno_y_limite(current)
        seg_end = min(limit, end_dt)
        seg_hours = (seg_end - current).total_seconds() / 3600.0
        if seg_hours > 0:
            segments.append((shift, day_key, seg_hours))
        current = seg_end
    return segments


def parse_time_value(value: Optional[str]) -> Optional[time]:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).time()
        except ValueError:
            continue
    return None


def determinar_turno(hora: Optional[time]) -> str:
    if not hora:
        return "T1"
    hour = hora.hour + hora.minute / 60.0
    if 6 <= hour < 14:
        return "T1"
    if 14 <= hour < 22:
        return "T2"
    return "T3"


def cargar_logo(logo_path: Optional[Path]):
    if not logo_path:
        return None
    try:
        return mpimg.imread(logo_path)
    except (FileNotFoundError, OSError):
        return None


def cargar_ciclos(base_path: Path) -> Dict[str, Dict[str, float]]:
    ciclos: Dict[str, Dict[str, float]] = {}
    archivo = base_path / "ciclos.csv"
    if not archivo.exists():
        return ciclos
    with archivo.open(encoding="utf-8-sig") as handler:
        reader = csv.DictReader(handler)
        for row in reader:
            maquina = (row.get("maquina") or "").strip().lower()
            referencia = normalize_ref(row.get("referencia"))
            tiempo = parse_float(row.get("tiempo_ciclo"))
            if not maquina or not referencia or tiempo <= 0:
                continue
            ciclos.setdefault(maquina, {})[referencia] = tiempo
    return ciclos


def hours_to_hhmmss(hours: float) -> str:
    total_seconds = max(int(round(hours * 3600)), 0)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_pct(value: float) -> str:
    return f"{value:0.2f}".replace(".", ",") + "%"


def format_units(value: float) -> str:
    entero = int(round(value))
    return f"{entero:,}".replace(",", ".")


def format_pph(value: Optional[float]) -> str:
    if value is None or value <= 0:
        return "-"
    return f"{value:0.1f} p/h"


def format_cycle(seconds: Optional[float]) -> str:
    if seconds is None or seconds <= 0:
        return "-"
    return f"{seconds:0.1f}s"


def calcular_paros(stats: Dict[str, float]) -> float:
    """Devuelve el tiempo parado (incidencias + preparación) usando las horas brutas como base."""
    horas_brutas = stats.get("horas_brutas", 0.0)
    horas_produccion = stats.get("horas_produccion", 0.0)
    paros = horas_brutas - horas_produccion
    return paros if paros > 0 else 0.0


def calcular_preparacion(stats: Dict[str, float]) -> float:
    if "horas_preparacion" in stats:
        return stats.get("horas_preparacion", 0.0) or 0.0
    horas_brutas = stats.get("horas_brutas", 0.0)
    horas_produccion = stats.get("horas_produccion", 0.0)
    horas_incidencias = stats.get("horas_incidencias", 0.0)
    prep = horas_brutas - horas_produccion - horas_incidencias
    return prep if prep > 0 else 0.0


def clamp_pct(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 100:
        return 100.0
    return value


def leer_maquina(csv_path: Path, ciclos: Dict[str, Dict[str, float]]) -> MachineSectionMetrics:
    horas_produccion = 0.0
    horas_incidencias = 0.0
    horas_preparacion = 0.0
    tiempo_ideal = 0.0
    piezas_totales = 0.0
    piezas_malas = 0.0
    piezas_recuperadas = 0.0
    start: Optional[datetime] = None
    end: Optional[datetime] = None

    stem = csv_path.stem
    parts = stem.split("-")
    display_name = parts[0] if parts else stem
    week_code = parts[2] if len(parts) >= 3 else None
    recurso = display_name.lower()
    ciclos_maquina = ciclos.get(recurso, {})
    daily_stats: Dict[date, Dict[str, float]] = defaultdict(crear_raw_metricas)
    shift_raw: Dict[str, Dict[str, float]] = {shift: crear_raw_metricas() for shift in SHIFT_LABELS}
    daily_shift_raw: Dict[date, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: {shift: crear_raw_metricas() for shift in SHIFT_LABELS}
    )
    ref_stats: Dict[str, Dict[str, float]] = {}

    with csv_path.open(encoding="utf-8-sig") as handler:
        reader = csv.DictReader(handler)
        if reader.fieldnames:
            reader.fieldnames = [(name or "").strip() for name in reader.fieldnames]
        for row in reader:
            start_time = parse_time_value(row.get("H Ini"))
            end_time = parse_time_value(row.get("F Fin"))
            fecha = parse_datetime(row.get("Fecha"))
            tiempo_csv = parse_float(row.get("Tiempo"))

            # Calculamos la duración con H Ini/F Fin para repartir turnos, pero
            # normalizamos siempre al valor de la columna Tiempo para que los totales
            # coincidan con el CSV original.
            start_dt = datetime.combine(fecha.date(), start_time) if fecha and start_time else None
            end_dt_calc = None
            if start_dt and end_time:
                end_dt_calc = datetime.combine(start_dt.date(), end_time)
                if end_dt_calc < start_dt:
                    end_dt_calc += timedelta(days=1)
                elif end_dt_calc == start_dt:
                    # mismo inicio y fin: usamos el tiempo del CSV (parada puntual)
                    end_dt_calc = None

            horas_calc = (end_dt_calc - start_dt).total_seconds() / 3600.0 if start_dt and end_dt_calc else 0.0
            horas = tiempo_csv if tiempo_csv > 0 else horas_calc
            if horas <= 0 and horas_calc > 0:
                horas = horas_calc
            proceso = normalizar_proceso(row.get("Proceso"))
            cantidad = parse_float(row.get("Cantidad"))
            malas = parse_float(row.get("Malas"))
            recuperadas = parse_float(row.get("Recu."))
            if not start_dt and fecha:
                start_dt = fecha

            if fecha:
                start = fecha if not start or fecha < start else start
                end = fecha if not end or fecha > end else end

            detalle = (row.get("Incidencia", "") or "").strip() or "Sin detalle"

            if proceso == "produccion":
                horas_produccion += horas
            elif proceso == "incidencias":
                horas_incidencias += horas
            else:
                horas_preparacion += horas

            piezas_totales += cantidad
            piezas_malas += malas
            piezas_recuperadas += recuperadas

            ref = normalize_ref(row.get("Refer."))
            ciclo_seg = ciclos_maquina.get(ref)
            duration_for_segments = horas_calc if horas_calc > 0 else horas
            segmentos = dividir_en_turnos(start_dt, duration_for_segments)
            if not segmentos:
                turno_def = determinar_turno(start_time)
                day_def = fecha.date() if fecha else datetime.today().date()
                segmentos = [(turno_def, day_def, horas)]
            else:
                total_seg_hours_calc = sum(h for _, _, h in segmentos)
                # Reescalamos los segmentos para que el total respete la duración del CSV.
                if horas > 0 and total_seg_hours_calc > 0 and abs(total_seg_hours_calc - horas) > 1e-6:
                    factor = horas / total_seg_hours_calc
                    segmentos = [(t, d, h * factor) for t, d, h in segmentos]

            total_seg_hours = sum(h for _, _, h in segmentos) or 1.0
            hour_base = horas if horas > 0 else total_seg_hours

            for turno_seg, day_seg, horas_seg in segmentos:
                peso = horas_seg / hour_base if hour_base > 0 else 0.0
                piezas_seg = cantidad * peso
                malas_seg = malas * peso
                rec_seg = recuperadas * peso

                shift_stats = shift_raw[turno_seg]
                day_stats = daily_stats[day_seg]
                day_shift_stats = daily_shift_raw[day_seg][turno_seg]

                shift_stats["piezas_totales"] += piezas_seg
                shift_stats["piezas_malas"] += malas_seg
                shift_stats["piezas_recuperadas"] += rec_seg
                day_stats["piezas_totales"] += piezas_seg
                day_stats["piezas_malas"] += malas_seg
                day_stats["piezas_recuperadas"] += rec_seg
                day_shift_stats["piezas_totales"] += piezas_seg
                day_shift_stats["piezas_malas"] += malas_seg
                day_shift_stats["piezas_recuperadas"] += rec_seg

                if proceso == "produccion":
                    shift_stats["horas_produccion"] += horas_seg
                    day_stats["horas_produccion"] += horas_seg
                    day_shift_stats["horas_produccion"] += horas_seg
                    if ciclo_seg and ciclo_seg > 0 and piezas_seg > 0:
                        seg_tiempo_ideal = piezas_seg / ciclo_seg
                        tiempo_ideal += seg_tiempo_ideal
                        shift_stats["tiempo_ideal"] += seg_tiempo_ideal
                        day_stats["tiempo_ideal"] += seg_tiempo_ideal
                        day_shift_stats["tiempo_ideal"] += seg_tiempo_ideal
                    # Estadística por referencia
                    if piezas_seg > 0 or horas_seg > 0:
                        ref_entry = ref_stats.setdefault(
                            ref,
                            {
                                "ciclo_ideal": ciclo_seg if ciclo_seg and ciclo_seg > 0 else None,
                                "dias": {},
                            },
                        )
                        if ciclo_seg and ciclo_seg > 0:
                            ref_entry["ciclo_ideal"] = ciclo_seg
                        day_entry = ref_entry["dias"].setdefault(day_seg, {"piezas": 0.0, "horas": 0.0})
                        day_entry["piezas"] += piezas_seg
                        day_entry["horas"] += horas_seg
                elif proceso == "incidencias":
                    shift_stats["horas_incidencias"] += horas_seg
                    day_stats["horas_incidencias"] += horas_seg
                    day_shift_stats["horas_incidencias"] += horas_seg
                else:
                    shift_stats["horas_preparacion"] += horas_seg
                    day_stats["horas_preparacion"] += horas_seg
                    day_shift_stats["horas_preparacion"] += horas_seg

    horas_brutas = horas_produccion + horas_incidencias + horas_preparacion
    perdidas_rend = max(horas_produccion - tiempo_ideal, 0.0)
    scrap_final = max(piezas_malas - piezas_recuperadas, 0.0)
    buenas_finales = max(piezas_totales - scrap_final, 0.0)

    disponibilidad_pct = clamp_pct((horas_produccion / horas_brutas * 100) if horas_brutas > 0 else 0.0)
    rendimiento_pct = clamp_pct((tiempo_ideal / horas_produccion * 100) if horas_produccion > 0 and tiempo_ideal > 0 else 0.0)
    calidad_pct = clamp_pct((buenas_finales / piezas_totales * 100) if piezas_totales > 0 else 0.0)
    oee_pct = (disponibilidad_pct * rendimiento_pct * calidad_pct) / 10000.0

    shift_stats_summary: Dict[str, Dict[str, float]] = {}
    for shift, raw_data in shift_raw.items():
        metrics_shift = convertir_raw_a_metricas(raw_data)
        metrics_shift["raw"] = raw_data.copy()
        shift_stats_summary[shift] = metrics_shift

    daily_shift_stats: Dict[date, Dict[str, Dict[str, float]]] = {}
    for day, shift_data in daily_shift_raw.items():
        stats_day: Dict[str, Dict[str, float]] = {}
        for shift, raw_data in shift_data.items():
            stats_day[shift] = convertir_raw_a_metricas(raw_data)
        daily_shift_stats[day] = stats_day

    return MachineSectionMetrics(
        name=display_name,
        full_name=stem,
        week_code=week_code,
        horas_brutas=horas_brutas,
        horas_produccion=horas_produccion,
        horas_incidencias=horas_incidencias,
        horas_preparacion=horas_preparacion,
        tiempo_ideal=tiempo_ideal,
        perdidas_rend=perdidas_rend,
        piezas_totales=piezas_totales,
        piezas_malas=piezas_malas,
        piezas_recuperadas=piezas_recuperadas,
        buenas_finales=buenas_finales,
        disponibilidad_pct=disponibilidad_pct,
        rendimiento_pct=rendimiento_pct,
        calidad_pct=calidad_pct,
        oee_pct=oee_pct,
        start=start,
        end=end,
        daily_stats={day: dict(stats) for day, stats in daily_stats.items()},
        shift_stats=shift_stats_summary,
        daily_shift_stats=daily_shift_stats,
        ref_stats={
            ref: {
                "ciclo_ideal": data.get("ciclo_ideal"),
                "dias": {
                    day: {"piezas": val.get("piezas", 0.0), "horas": val.get("horas", 0.0)}
                    for day, val in sorted(data.get("dias", {}).items())
                },
            }
            for ref, data in ref_stats.items()
        },
    )


def calcular_totales(maquinas: List[MachineSectionMetrics]) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    horas_brutas = sum(m.horas_brutas for m in maquinas)
    horas_produccion = sum(m.horas_produccion for m in maquinas)
    horas_incidencias = sum(m.horas_incidencias for m in maquinas)
    horas_preparacion = sum((m.horas_brutas - m.horas_produccion - m.horas_incidencias) for m in maquinas)
    tiempo_ideal = sum(m.tiempo_ideal for m in maquinas)
    piezas_totales = sum(m.piezas_totales for m in maquinas)
    piezas_malas = sum(m.piezas_malas for m in maquinas)
    piezas_recuperadas = sum(m.piezas_recuperadas for m in maquinas)

    perdidas_rend = max(horas_produccion - tiempo_ideal, 0.0)
    scrap_final = max(piezas_malas - piezas_recuperadas, 0.0)
    buenas_finales = max(piezas_totales - scrap_final, 0.0)

    disponibilidad_pct = clamp_pct((horas_produccion / horas_brutas * 100) if horas_brutas > 0 else 0.0)
    rendimiento_pct = clamp_pct((tiempo_ideal / horas_produccion * 100) if horas_produccion > 0 and tiempo_ideal > 0 else 0.0)
    calidad_pct = clamp_pct((buenas_finales / piezas_totales * 100) if piezas_totales > 0 else 0.0)
    oee_pct = (disponibilidad_pct * rendimiento_pct * calidad_pct) / 10000.0

    totals = {
        "horas_brutas": horas_brutas,
        "horas_produccion": horas_produccion,
        "horas_incidencias": horas_incidencias,
        "tiempo_ideal": tiempo_ideal,
        "horas_preparacion": horas_preparacion,
        "perdidas_rend": perdidas_rend,
        "buenas_finales": buenas_finales,
        "piezas_totales": piezas_totales,
        "disponibilidad_pct": disponibilidad_pct,
        "rendimiento_pct": rendimiento_pct,
        "calidad_pct": calidad_pct,
        "oee_pct": oee_pct,
    }
    shift_totals: Dict[str, Dict[str, float]] = {}
    for shift in SHIFT_LABELS:
        horas_brutas_turno = sum(m.shift_stats.get(shift, {}).get("horas_brutas", 0.0) for m in maquinas)
        horas_p = sum(m.shift_stats.get(shift, {}).get("horas_produccion", 0.0) for m in maquinas)
        horas_i = sum(m.shift_stats.get(shift, {}).get("horas_incidencias", 0.0) for m in maquinas)
        horas_pre = sum(m.shift_stats.get(shift, {}).get("raw", {}).get("horas_preparacion", 0.0) for m in maquinas)
        horas_total = horas_brutas_turno if horas_brutas_turno > 0 else horas_p + horas_i + horas_pre
        tiempo_ideal_turno = sum(m.shift_stats.get(shift, {}).get("tiempo_ideal", 0.0) for m in maquinas)
        perdidas_turno = max(horas_p - tiempo_ideal_turno, 0.0)
        piezas_totales_turno = sum(m.shift_stats.get(shift, {}).get("piezas_totales", 0.0) for m in maquinas)
        piezas_malas_turno = sum(m.shift_stats.get(shift, {}).get("piezas_malas", 0.0) for m in maquinas)
        piezas_rec_turno = sum(m.shift_stats.get(shift, {}).get("piezas_recuperadas", 0.0) for m in maquinas)
        scrap_turno = max(piezas_malas_turno - piezas_rec_turno, 0.0)
        buenas_turno = max(piezas_totales_turno - scrap_turno, 0.0)
        disp_turno = clamp_pct((horas_p / horas_total * 100) if horas_total > 0 else 0.0)
        rdo_turno = clamp_pct(
            (tiempo_ideal_turno / horas_p * 100) if horas_p > 0 and tiempo_ideal_turno > 0 else 0.0
        )
        cal_turno = clamp_pct(
            (buenas_turno / piezas_totales_turno * 100) if piezas_totales_turno > 0 else 0.0
        )
        oee_turno = (disp_turno * rdo_turno * cal_turno) / 10000.0
        shift_totals[shift] = {
            "horas_brutas": horas_total,
            "horas_produccion": horas_p,
            "horas_incidencias": horas_i,
            "tiempo_ideal": tiempo_ideal_turno,
            "perdidas_rend": perdidas_turno,
            "buenas_finales": buenas_turno,
            "piezas_totales": piezas_totales_turno,
            "disponibilidad_pct": disp_turno,
            "rendimiento_pct": rdo_turno,
            "calidad_pct": cal_turno,
            "oee_pct": oee_turno,
        }

    return totals, shift_totals

def calcular_resumen_diario(maquinas: List[MachineSectionMetrics]) -> List[Tuple[date, Dict[str, float]]]:
    diarios: Dict[date, Dict[str, float]] = defaultdict(
        lambda: {
            "horas_produccion": 0.0,
            "horas_incidencias": 0.0,
            "horas_preparacion": 0.0,
            "tiempo_ideal": 0.0,
            "piezas_totales": 0.0,
            "piezas_malas": 0.0,
            "piezas_recuperadas": 0.0,
        }
    )

    for maquina in maquinas:
        for day, data in maquina.daily_stats.items():
            stats = diarios[day]
            stats["horas_produccion"] += data.get("horas_produccion", 0.0)
            stats["horas_incidencias"] += data.get("horas_incidencias", 0.0)
            stats["horas_preparacion"] += data.get("horas_preparacion", 0.0)
            stats["tiempo_ideal"] += data.get("tiempo_ideal", 0.0)
            stats["piezas_totales"] += data.get("piezas_totales", 0.0)
            stats["piezas_malas"] += data.get("piezas_malas", 0.0)
            stats["piezas_recuperadas"] += data.get("piezas_recuperadas", 0.0)

    resumen: List[Tuple[date, Dict[str, float]]] = []
    for day in sorted(diarios.keys()):
        data = diarios[day]
        horas_brutas = data["horas_produccion"] + data["horas_incidencias"] + data["horas_preparacion"]
        perdidas_rend = max(data["horas_produccion"] - data["tiempo_ideal"], 0.0)
        scrap_final = max(data["piezas_malas"] - data["piezas_recuperadas"], 0.0)
        buenas_finales = max(data["piezas_totales"] - scrap_final, 0.0)

        disponibilidad_pct = clamp_pct(
            (data["horas_produccion"] / horas_brutas * 100) if horas_brutas > 0 else 0.0
        )
        rendimiento_pct = clamp_pct(
            (data["tiempo_ideal"] / data["horas_produccion"] * 100)
            if data["horas_produccion"] > 0 and data["tiempo_ideal"] > 0
            else 0.0
        )
        calidad_pct = clamp_pct(
            (buenas_finales / data["piezas_totales"] * 100) if data["piezas_totales"] > 0 else 0.0
        )
        oee_pct = (disponibilidad_pct * rendimiento_pct * calidad_pct) / 10000.0

        resumen.append(
            (
                day,
                {
                    "horas_brutas": horas_brutas,
                    "horas_produccion": data["horas_produccion"],
                    "horas_preparacion": data["horas_preparacion"],
                    "tiempo_ideal": data["tiempo_ideal"],
                    "perdidas_rend": perdidas_rend,
                    "buenas_finales": buenas_finales,
                    "disponibilidad_pct": disponibilidad_pct,
                    "rendimiento_pct": rendimiento_pct,
                    "calidad_pct": calidad_pct,
                    "oee_pct": oee_pct,
                },
            )
        )

    return resumen


def obtener_secciones(
    data_path: Path, nombres: Optional[Sequence[str]] = None
) -> List[Tuple[str, Path]]:
    recursos_dir = data_path / RESOURCE_DIR_NAME
    if recursos_dir.exists():
        disponibles = {p.name: p for p in recursos_dir.iterdir() if p.is_dir()}
        disponibles_ci = {nombre.lower(): (nombre, path) for nombre, path in disponibles.items()}
        if nombres:
            secciones = []
            for nombre in nombres:
                entry = disponibles.get(nombre)
                if entry is None:
                    entry = disponibles_ci.get(nombre.lower())
                if entry is None:
                    raise FileNotFoundError(f"No existe la sección '{nombre}' en {recursos_dir}")
                if isinstance(entry, tuple):
                    real_name, path = entry
                else:
                    real_name, path = nombre, entry
                secciones.append((real_name, path))
            return secciones
        return sorted(disponibles.items())

    if nombres:
        raise FileNotFoundError(f"No existe la carpeta de secciones en {data_path}")

    csv_files = list(data_path.glob("*.csv"))
    if csv_files:
        return [("general", data_path)]
    return []


def crear_tabla(
    maquinas: List[MachineSectionMetrics], totales: Dict[str, float]
) -> Tuple[List[str], List[List[str]]]:
    col_labels = [
        "Sección",
        "T. Disp.",
        "T. Oper.",
        "T. Prep.",
        "T. Paros",
        "T. Ideal",
        "Perd. Rdo.",
        "Totales",
        "Buenas",
        "Disp.%",
        "Rdto.%",
        "Q.%",
        "OEE%",
    ]

    rows: List[List[str]] = []
    for metric in sorted(maquinas, key=machine_sort_key):
        rows.append(
            [
                metric.name.upper(),
                hours_to_hhmmss(metric.horas_brutas),
                hours_to_hhmmss(metric.horas_produccion),
                hours_to_hhmmss(calcular_preparacion({"horas_brutas": metric.horas_brutas, "horas_produccion": metric.horas_produccion, "horas_incidencias": metric.horas_incidencias})),
                hours_to_hhmmss(max(metric.horas_brutas - metric.horas_produccion, 0.0)),
                hours_to_hhmmss(metric.tiempo_ideal),
                hours_to_hhmmss(metric.perdidas_rend),
                format_units(metric.piezas_totales),
                format_units(metric.buenas_finales),
                format_pct(metric.disponibilidad_pct),
                format_pct(metric.rendimiento_pct),
                format_pct(metric.calidad_pct),
                format_pct(metric.oee_pct),
            ]
        )

    rows.append(
        [
            "Total sección",
            hours_to_hhmmss(totales["horas_brutas"]),
            hours_to_hhmmss(totales["horas_produccion"]),
            hours_to_hhmmss(calcular_preparacion(totales)),
            hours_to_hhmmss(max(totales["horas_brutas"] - totales["horas_produccion"], 0.0)),
            hours_to_hhmmss(totales["tiempo_ideal"]),
            hours_to_hhmmss(totales["perdidas_rend"]),
            format_units(totales["piezas_totales"]),
            format_units(totales["buenas_finales"]),
            format_pct(totales["disponibilidad_pct"]),
            format_pct(totales["rendimiento_pct"]),
            format_pct(totales["calidad_pct"]),
            format_pct(totales["oee_pct"]),
        ]
    )

    return col_labels, rows


def build_machine_pages(maquinas: List[MachineSectionMetrics], semana_label: Optional[str], logo_image=None) -> List[plt.Figure]:
    figures: List[plt.Figure] = []
    col_labels = [
        "Periodo",
        "T. Disp.",
        "T. Oper.",
        "T. Prep.",
        "T. Paros",
        "T. Ideal",
        "Perd. Rdo.",
        "Totales",
        "Buenas",
        "Disp.%",
        "Rdto.%",
        "Q.%",
        "OEE%",
    ]

    def format_row(label: str, stats: Dict[str, float]) -> List[str]:
        return [
            label,
            hours_to_hhmmss(stats.get("horas_brutas", 0.0)),
            hours_to_hhmmss(stats.get("horas_produccion", 0.0)),
            hours_to_hhmmss(calcular_preparacion(stats)),
            hours_to_hhmmss(calcular_paros(stats)),
            hours_to_hhmmss(stats.get("tiempo_ideal", 0.0)),
            hours_to_hhmmss(stats.get("perdidas_rend", 0.0)),
            format_units(stats.get("piezas_totales", 0.0)),
            format_units(stats.get("buenas_finales", 0.0)),
            format_pct(stats.get("disponibilidad_pct", 0.0)),
            format_pct(stats.get("rendimiento_pct", 0.0)),
            format_pct(stats.get("calidad_pct", 0.0)),
            format_pct(stats.get("oee_pct", 0.0)),
        ]

    def general_stats(metric: MachineSectionMetrics) -> Dict[str, float]:
        return {
            "horas_brutas": metric.horas_brutas,
            "horas_produccion": metric.horas_produccion,
            "horas_incidencias": metric.horas_incidencias,
            "horas_preparacion": calcular_preparacion(
                {
                    "horas_brutas": metric.horas_brutas,
                    "horas_produccion": metric.horas_produccion,
                    "horas_incidencias": metric.horas_incidencias,
                }
            ),
            "tiempo_ideal": metric.tiempo_ideal,
            "perdidas_rend": metric.perdidas_rend,
            "buenas_finales": metric.buenas_finales,
            "piezas_totales": metric.piezas_totales,
            "disponibilidad_pct": metric.disponibilidad_pct,
            "rendimiento_pct": metric.rendimiento_pct,
            "calidad_pct": metric.calidad_pct,
            "oee_pct": metric.oee_pct,
        }

    for metric in sorted(maquinas, key=machine_sort_key):
        fig = plt.figure(figsize=(11.69, 8.27))
        titulo = metric.name.upper()
        if semana_label:
            titulo = f"{titulo} · W{semana_label}"
        fig.suptitle(titulo, fontsize=16, fontweight="bold", color="#263238")
        ax = fig.add_subplot(111)
        ax.axis("off")
        rows: List[List[str]] = []
        bold_rows: set[int] = set()
        highlight_rows: set[int] = set()

        # Detalle diario
        for day in sorted(metric.daily_stats.keys()):
            day_idx = len(rows)
            rows.append([day.strftime("%d-%b")] + [""] * (len(col_labels) - 1))
            bold_rows.add(day_idx)
            shift_day = metric.daily_shift_stats.get(day, {})
            for shift in SHIFT_LABELS:
                stats = shift_day.get(shift)
                if stats:
                    rows.append(format_row(shift, stats))
            general_day = convertir_raw_a_metricas(metric.daily_stats[day])
            total_idx = len(rows)
            rows.append(format_row("Total", general_day))
            bold_rows.add(total_idx)
            rows.append([""] * len(col_labels))  # separador

        # Totales turno global
        for shift in SHIFT_LABELS:
            shift_stats = metric.shift_stats.get(shift)
            if shift_stats:
                idx = len(rows)
                rows.append(format_row(f"{shift} Total", shift_stats))
                highlight_rows.add(idx)
        final_idx = len(rows)
        rows.append(format_row(f"TOTAL {metric.name.upper()}", general_stats(metric)))
        bold_rows.add(final_idx)

        table = ax.table(
            cellText=rows,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
            colWidths=[0.18] + [0.085] * (len(col_labels) - 1),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.1)
        for (row_idx, col_idx), cell in table.get_celld().items():
            cell.set_edgecolor("#CFD8DC")
            if row_idx == 0:
                cell.set_facecolor("#ECEFF1")
                cell.set_text_props(fontweight="bold")
            elif (row_idx - 1) in bold_rows:
                cell.set_text_props(fontweight="bold")
            elif (row_idx - 1) in highlight_rows:
                cell.set_facecolor("#F7F9FA")
            else:
                cell.set_facecolor("white")
        figures.append(fig)
    return figures


def build_cover_page(
    section_name: str,
    totales: Dict[str, float],
    fecha_inicio: Optional[datetime],
    fecha_fin: Optional[datetime],
    semana_label: Optional[str],
    logo,
) -> plt.Figure:
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_subplot(111)
    ax.axis("off")
    padding = 0.03
    ax.add_patch(
        Rectangle(
            (padding, padding),
            1 - 2 * padding,
            1 - 2 * padding,
            transform=ax.transAxes,
            facecolor="#F4F6F8",
            edgecolor="#CFD8DC",
            linewidth=1.5,
        )
    )

    if logo is not None:
        image = OffsetImage(logo, zoom=0.28)
        ab = AnnotationBbox(image, (0.2, 0.65), frameon=False, xycoords="axes fraction")
        ax.add_artist(ab)

    ax.text(0.55, 0.78, "Informe OEE", fontsize=30, fontweight="bold", color="#263238")
    ax.text(0.55, 0.64, f"Sección: {section_name.upper()}", fontsize=18, color="#546E7A")

    fecha_ini_txt = fecha_inicio.strftime("%d/%m/%Y") if fecha_inicio else "-"
    fecha_fin_txt = fecha_fin.strftime("%d/%m/%Y") if fecha_fin else "-"
    paros_totales = calcular_paros(totales)
    info_lines = [
        f"Periodo: {fecha_ini_txt} → {fecha_fin_txt}",
        f"Semana: {semana_label or '-'}",
        f"OEE total: {format_pct(totales['oee_pct'])}",
        f"Disponibilidad: {format_pct(totales['disponibilidad_pct'])}",
        f"Rendimiento: {format_pct(totales['rendimiento_pct'])}",
        f"Calidad: {format_pct(totales['calidad_pct'])}",
        f"T. disponible: {hours_to_hhmmss(totales['horas_brutas'])}",
        f"T. operación: {hours_to_hhmmss(totales['horas_produccion'])}",
        f"T. paros: {hours_to_hhmmss(paros_totales)}",
        f"T. ideal: {hours_to_hhmmss(totales['tiempo_ideal'])}",
        f"Pérdidas rendimiento: {hours_to_hhmmss(totales['perdidas_rend'])}",
    ]
    y = 0.5
    for line in info_lines:
        ax.text(0.55, y, line, fontsize=14, color="#37474F")
        y -= 0.045
    return fig


def build_master_table_page(
    section_name: str,
    chunk: List[List[str]],
    col_labels: List[str],
    page_index: int,
    bold_rows: Optional[set[int]] = None,
    highlight_rows: Optional[set[int]] = None,
    semana_label: Optional[str] = None,
    periodo_text: Optional[str] = None,
    resumen: Optional[Dict[str, float]] = None,
    logo_image=None,
) -> plt.Figure:
    bold_rows = bold_rows or set()
    highlight_rows = highlight_rows or set()
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_subplot(111)
    ax.axis("off")
    title_x = 0.06 if logo_image is not None else 0.02
    if logo_image is not None:
        image = OffsetImage(logo_image, zoom=0.08)
        ab = AnnotationBbox(
            image,
            (-0.015, 1.04),
            frameon=False,
            xycoords="axes fraction",
            box_alignment=(0, 1),
        )
        ax.add_artist(ab)
    title_color = "#1B2631"
    accent_color = "#2C3E50"
    ax.text(
        title_x,
        0.97,
        f"{section_name.upper()}/ General",
        fontsize=13,
        fontweight="bold",
        color=title_color,
    )
    if periodo_text and resumen:
        banner_x, banner_y, w, h = 0.54, 0.825, 0.44, 0.18
        ax.add_patch(
            Rectangle(
                (banner_x, banner_y),
                w,
                h,
                transform=ax.transAxes,
                facecolor="#EEF2F5",
                edgecolor=accent_color,
                linewidth=1.0,
                zorder=0,
            )
        )
        line_y = banner_y + h - 0.03
        if semana_label:
            ax.text(
                banner_x + 0.015,
                line_y,
                f"Semana: W{semana_label}",
                fontsize=11,
                fontweight="bold",
                color=accent_color,
            )
            line_y -= 0.04
        ax.text(
            banner_x + 0.015,
            line_y,
            f"Periodo: {periodo_text}",
            fontsize=11,
            fontweight="bold",
            color=accent_color,
        )
        line_y -= 0.05
        metrics = [
            f"Disp.: {format_pct(resumen.get('disponibilidad_pct', 0.0))}",
            f"Rdto.: {format_pct(resumen.get('rendimiento_pct', 0.0))}",
            f"Q.: {format_pct(resumen.get('calidad_pct', 0.0))}",
        ]
        ax.text(
            banner_x + 0.015,
            line_y,
            " · ".join(metrics),
            fontsize=11,
            fontweight="bold",
            color=title_color,
        )
        oee_line = f"OEE: {format_pct(resumen.get('oee_pct', 0.0))}"
        ax.text(
            banner_x + 0.015,
            banner_y + 0.025,
            oee_line,
            fontsize=13,
            fontweight="bold",
            color=accent_color,
        )
    table = ax.table(
        cellText=chunk,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.18] + [0.075] * (len(col_labels) - 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.05)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#CFD8DC")
        if row_idx == 0:
            cell.set_facecolor("#ECEFF1")
            cell.set_text_props(fontweight="bold", color="#263238")
        elif (row_idx - 1) in bold_rows:
            cell.set_text_props(fontweight="bold")
        elif (row_idx - 1) in highlight_rows:
            cell.set_facecolor("#F7F9FA")
        else:
            cell.set_facecolor("white")
    return fig


def build_reference_summary_pages(
    section_name: str,
    maquinas: List[MachineSectionMetrics],
    semana_label: Optional[str],
    logo_image=None,
) -> List[plt.Figure]:
    col_labels = ["Sección", "Referencia", "Día", "Ciclo ideal (p/h)", "Ciclo real (p/h)", "Cantidad"]

    def build_rows(selected: List[MachineSectionMetrics]) -> List[tuple[List[str], bool]]:
        ref_rows: List[tuple[List[str], bool]] = []
        for metric in sorted(selected, key=machine_sort_key):
            refs = sorted(metric.ref_stats.items())
            if not refs:
                continue
            for _, (ref, stats) in enumerate(refs):
                dias = stats.get("dias", {}) or {}
                ciclo_ideal = stats.get("ciclo_ideal")
                total_piezas = 0.0
                total_horas = 0.0
                for day in sorted(dias.keys()):
                    piezas = dias[day].get("piezas", 0.0)
                    horas = dias[day].get("horas", 0.0)
                    total_piezas += piezas
                    total_horas += horas
                    ciclo_real = (piezas / horas) if horas > 0 else None  # piezas por hora real
                    ref_rows.append(
                        (
                            [
                                metric.name.upper(),
                                ref or "-",
                                day.strftime("%d-%b"),
                                format_pph(ciclo_ideal),
                                format_pph(ciclo_real),
                                format_units(piezas),
                            ],
                            False,
                        )
                    )
                ciclo_real_avg = (total_piezas / total_horas) if total_horas > 0 else None
                ref_rows.append(
                    (
                        [
                            metric.name.upper(),
                            ref or "-",
                            "Media ref",
                            format_pph(ciclo_ideal),
                            format_pph(ciclo_real_avg),
                            format_units(total_piezas),
                        ],
                        True,
                    )
                )
            ref_rows.append(([""] * len(col_labels), False))
        if ref_rows and all(not cell for cell in ref_rows[-1][0]):
            ref_rows.pop()
        if not ref_rows:
            ref_rows.append((["-"] * len(col_labels), False))
        return ref_rows

    pages: List[plt.Figure] = []
    # Si es LINEAS, agrupamos en dos páginas fijas; en otras secciones, un único grupo.
    if section_name.upper() == "LINEAS":
        names_group1 = {"coroa", "luk1", "luk2"}
        groups = [
            (0, build_rows([m for m in maquinas if m.name.lower() in names_group1])),
            (1, build_rows([m for m in maquinas if m.name.lower() not in names_group1])),
        ]
    else:
        groups = [(0, build_rows(maquinas))]

    for group_index, ref_rows in groups:
        if not ref_rows:
            continue
        chunk_size = REF_ROWS_PER_PAGE
        for page_idx in range(0, len(ref_rows), chunk_size):
            chunk = ref_rows[page_idx : page_idx + chunk_size]
            rows = [row for row, _ in chunk]
            summary_rows = {idx for idx, (_, is_summary) in enumerate(chunk) if is_summary}

            fig = plt.figure(figsize=(11.69, 8.27))
            ax = fig.add_subplot(111)
            ax.axis("off")
            title_color = "#1B2631"
            if logo_image is not None:
                image = OffsetImage(logo_image, zoom=0.08)
                ab = AnnotationBbox(
                    image, (-0.015, 1.04), frameon=False, xycoords="axes fraction", box_alignment=(0, 1)
                )
                ax.add_artist(ab)
            title = f"{section_name.upper()}. Referencias"
            if semana_label:
                title = f"{title} · W{semana_label}"
            if section_name.upper() == "LINEAS":
                subtitle = "COROA / LUK1 / LUK2" if group_index == 0 else "LUK3 / LUK6 / VW1"
                ax.text(0.06, 0.955, subtitle, fontsize=10, fontweight="bold", color="#455A64", va="top")
            ax.text(0.06, 0.99, title, fontsize=13, fontweight="bold", color=title_color, va="top")

            table = ax.table(
                cellText=rows,
                colLabels=col_labels,
                cellLoc="center",
                loc="center",
                colWidths=[0.14, 0.16, 0.16, 0.16, 0.16, 0.1],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.1)
            for (row_idx, col_idx), cell in table.get_celld().items():
                cell.set_edgecolor("#CFD8DC")
                if row_idx == 0:
                    cell.set_facecolor("#ECEFF1")
                    cell.set_text_props(fontweight="bold", color="#263238")
                elif (row_idx - 1) in summary_rows:
                    cell.set_facecolor("#E5EBF3")
                    cell.set_text_props(fontweight="bold", color="#1B2631")
                    cell.set_edgecolor("#90A4AE")
                    cell.set_linewidth(1.2)
                else:
                    cell.set_facecolor("white")
            pages.append(fig)

    return pages


def render_section_report(
    section_name: str,
    maquinas: List[MachineSectionMetrics],
    totales: Dict[str, float],
    totales_turnos: Dict[str, Dict[str, float]],
    daily_summary: List[Tuple[date, Dict[str, float]]],
    output_dir: Path,
    logo,
    fecha_inicio: Optional[datetime],
    fecha_fin: Optional[datetime],
    semana_label: Optional[str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fecha_ini_txt = fecha_inicio.strftime("%d/%m/%Y") if fecha_inicio else "-"
    fecha_fin_txt = fecha_fin.strftime("%d/%m/%Y") if fecha_fin else "-"
    periodo_text = f"{fecha_ini_txt} → {fecha_fin_txt}"

    col_labels = [
        "Sección",
        "T. Disp.",
        "T. Oper.",
        "T. Prep.",
        "T. Paros",
        "T. Ideal",
        "Perd. Rdo.",
        "Totales",
        "Buenas",
        "Disp.%",
        "Rdto.%",
        "Q.%",
        "OEE%",
    ]

    def format_stats_row(label: str, stats: Dict[str, float]) -> List[str]:
        return [
            label,
            hours_to_hhmmss(stats.get("horas_brutas", 0.0)),
            hours_to_hhmmss(stats.get("horas_produccion", 0.0)),
            hours_to_hhmmss(calcular_preparacion(stats)),
            hours_to_hhmmss(calcular_paros(stats)),
            hours_to_hhmmss(stats.get("tiempo_ideal", 0.0)),
            hours_to_hhmmss(stats.get("perdidas_rend", 0.0)),
            format_units(stats.get("piezas_totales", 0.0)),
            format_units(stats.get("buenas_finales", 0.0)),
            format_pct(stats.get("disponibilidad_pct", 0.0)),
            format_pct(stats.get("rendimiento_pct", 0.0)),
            format_pct(stats.get("calidad_pct", 0.0)),
            format_pct(stats.get("oee_pct", 0.0)),
        ]

    def general_stats(metric: MachineSectionMetrics) -> Dict[str, float]:
        return {
            "horas_brutas": metric.horas_brutas,
            "horas_produccion": metric.horas_produccion,
            "horas_incidencias": metric.horas_incidencias,
            "horas_preparacion": calcular_preparacion(
                {
                    "horas_brutas": metric.horas_brutas,
                    "horas_produccion": metric.horas_produccion,
                    "horas_incidencias": metric.horas_incidencias,
                }
            ),
            "tiempo_ideal": metric.tiempo_ideal,
            "perdidas_rend": metric.perdidas_rend,
            "buenas_finales": metric.buenas_finales,
            "piezas_totales": metric.piezas_totales,
            "disponibilidad_pct": metric.disponibilidad_pct,
            "rendimiento_pct": metric.rendimiento_pct,
            "calidad_pct": metric.calidad_pct,
            "oee_pct": metric.oee_pct,
        }

    def build_machine_block(metric: MachineSectionMetrics) -> tuple[List[List[str]], set[int]]:
        rows: List[List[str]] = []
        bold_rows: set[int] = set()
        header = [metric.name.upper()] + [""] * (len(col_labels) - 1)
        rows.append(header)
        bold_rows.add(0)
        shift_count = 0
        for shift in SHIFT_LABELS:
            shift_stats = metric.shift_stats.get(shift)
            if shift_stats:
                rows.append(format_stats_row(shift, shift_stats))
                shift_count += 1
        total_row_idx = len(rows)
        rows.append(format_stats_row("Total", general_stats(metric)))
        bold_rows.add(total_row_idx)
        rows.append([""] * len(col_labels))  # separador
        return rows, bold_rows

    machine_blocks: Dict[str, tuple[List[List[str]], set[int]]] = {}
    ordered_names: List[str] = []
    for metric in sorted(maquinas, key=machine_sort_key):
        name = metric.name.upper()
        ordered_names.append(name)
        machine_blocks[name] = build_machine_block(metric)

    total_rows: List[List[str]] = []
    total_bold: set[int] = set()
    total_highlight: set[int] = set()
    for shift in SHIFT_LABELS:
        shift_stats = totales_turnos.get(shift)
        if shift_stats:
            total_rows.append(format_stats_row(f"{shift} Total", shift_stats))
            total_highlight.add(len(total_rows) - 1)
    total_rows.append([""] * len(col_labels))
    final_idx = len(total_rows)
    total_rows.append(format_stats_row("TOTAL SECCION", totales))
    total_bold.add(final_idx)

    page_rows: List[List[List[str]]] = []
    page_bold: List[set[int]] = []
    page_highlight: List[set[int]] = []

    def append_block(block_rows: List[List[str]], bold_rows: set[int], highlight_rows: set[int], limit: int = 18):
        nonlocal page_rows, page_bold, page_highlight
        if not page_rows:
            page_rows.append([])
            page_bold.append(set())
            page_highlight.append(set())
        current_rows = page_rows[-1]
        current_bold = page_bold[-1]
        current_highlight = page_highlight[-1]
        if current_rows and len(current_rows) + len(block_rows) > limit:
            page_rows.append([])
            page_bold.append(set())
            page_highlight.append(set())
            current_rows = page_rows[-1]
            current_bold = page_bold[-1]
            current_highlight = page_highlight[-1]
        offset = len(current_rows)
        current_rows.extend(block_rows)
        current_bold.update({offset + idx for idx in bold_rows})
        current_highlight.update({offset + idx for idx in highlight_rows})

    groups = [
        ["LUK1", "LUK2", "LUK3"],
        ["COROA", "LUK6", "VW1"],
    ]
    used = set()
    for group in groups:
        for name in ordered_names:
            if name in group and name in machine_blocks:
                block_rows, bold_rows = machine_blocks[name]
                append_block(block_rows, bold_rows, set())
                used.add(name)

    for name in ordered_names:
        if name in used:
            continue
        block = machine_blocks.get(name)
        if not block:
            continue
        block_rows, bold_rows = block
        append_block(block_rows, bold_rows, set())

    if not page_rows:
        page_rows.append([])
        page_bold.append(set())
        page_highlight.append(set())
    append_block(total_rows, total_bold, total_highlight, limit=18)

    table_pages = [
        build_master_table_page(
            section_name,
            rows,
            col_labels,
            idx + 1,
            page_bold[idx],
            page_highlight[idx],
            semana_label,
            periodo_text,
            totales,
            logo,
        )
        for idx, rows in enumerate(page_rows)
    ]
    ref_pages = build_reference_summary_pages(section_name, maquinas, semana_label, logo)

    output_path = output_dir / f"{section_name}_oee_seccion.pdf"
    machine_pages = build_machine_pages(maquinas, semana_label, None)
    with PdfPages(output_path) as pdf:
        for fig in table_pages:
            pdf.savefig(fig)
            plt.close(fig)
        for ref_page in ref_pages:
            pdf.savefig(ref_page)
            plt.close(ref_page)
        for page in machine_pages:
            pdf.savefig(page)
            plt.close(page)
    return output_path


def generar_informes_oee_secciones(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    logo_path: Optional[Path] = None,
    secciones: Optional[Iterable[str]] = None,
) -> Iterable[Path]:
    data_path = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    report_path = Path(output_dir) if output_dir else DEFAULT_REPORT_DIR
    report_path.mkdir(parents=True, exist_ok=True)

    secciones_lista = obtener_secciones(data_path, list(secciones) if secciones else None)
    if not secciones_lista:
        raise FileNotFoundError(f"No se encontraron secciones válidas en {data_path}")

    ciclos = cargar_ciclos(data_path)
    logo = cargar_logo(logo_path)

    resultados = []
    for nombre_seccion, ruta_seccion in secciones_lista:
        csv_files = sorted(ruta_seccion.glob("*.csv"))
        if not csv_files:
            print(f"[OEE Secciones] Se omite {nombre_seccion}: no hay CSV.")
            continue

        maquinas = []
        fecha_inicio = None
        fecha_fin = None
        for csv_file in csv_files:
            metric = leer_maquina(csv_file, ciclos)
            maquinas.append(metric)
            if metric.start:
                fecha_inicio = metric.start if not fecha_inicio or metric.start < fecha_inicio else fecha_inicio
            if metric.end:
                fecha_fin = metric.end if not fecha_fin or metric.end > fecha_fin else fecha_fin

        maquinas.sort(key=machine_sort_key)
        totales, totales_turnos = calcular_totales(maquinas)
        daily_summary = calcular_resumen_diario(maquinas)
        week_codes = {m.week_code for m in maquinas if m.week_code}
        semana_label = ", ".join(sorted(week_codes)) if week_codes else None
        section_dir = report_path / nombre_seccion
        pdf_path = render_section_report(
            nombre_seccion,
            maquinas,
            totales,
            totales_turnos,
            daily_summary,
            section_dir,
            logo,
            fecha_inicio,
            fecha_fin,
            semana_label,
        )
        print(f"[OEE Secciones] {nombre_seccion}: {totales['oee_pct']:0.2f}% (PDF: {pdf_path})")
        resultados.append(pdf_path)

    return resultados


if __name__ == "__main__":
    generar_informes_oee_secciones()
