from __future__ import annotations

import csv
import re
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
MIN_PIEZAS_OEE = 50  # Mínimo de piezas para calcular OEE en un turno/día

# Regex para identificar incidencias que afectan a la DISPONIBILIDAD
# Estas se restan del T. Bruto para obtener T. Disponible
INCIDENCIA_DISPONIBILIDAD_PATTERNS = [
    re.compile(r"averia", re.IGNORECASE),
    re.compile(r"mant.*prevent", re.IGNORECASE),  # mantenimiento preventivo
    re.compile(r"limpieza", re.IGNORECASE),
]


def clasificar_incidencia(texto_incidencia: str) -> str:
    """
    Clasifica una incidencia en:
    - 'disponibilidad': Avería, Mant. Preventivo, Limpieza (restan de T. Bruto → T. Disponible)
    - 'paros': Resto de incidencias (restan de T. Disponible → T. Operativo)
    """
    if not texto_incidencia:
        return "paros"
    for pattern in INCIDENCIA_DISPONIBILIDAD_PATTERNS:
        if pattern.search(texto_incidencia):
            return "disponibilidad"
    return "paros"


def machine_sort_key(machine: "MachineSectionMetrics") -> tuple[int, str]:
    """Orden preferente para LINEAS y, si no aplica, orden alfabético."""
    name_lower = machine.name.lower()
    return (ORDER_PRIORITY.get(name_lower, 999), name_lower)


@dataclass
class MachineSectionMetrics:
    name: str
    full_name: str
    week_code: Optional[str]
    # Tiempos según estructura OEE
    horas_brutas: float           # T. Bruto = Producción + Preparación (todo lo registrado)
    horas_indisponibilidad: float # Avería + Mant. Preventivo + Limpieza
    horas_disponible: float       # T. Bruto - Indisponibilidad
    horas_preparacion: float      # Registros con Proceso = Preparación
    horas_paros: float            # Otras incidencias (no disponibilidad)
    horas_operativo: float        # T. Disponible - Preparación - Paros (tiempo produciendo)
    tiempo_ideal: float           # Tiempo ideal según ciclo de piezas
    perdidas_rend: float          # T. Operativo - T. Ideal
    piezas_totales: float
    piezas_malas: float
    piezas_recuperadas: float
    buenas_finales: float
    disponibilidad_pct: float     # T. Disponible / T. Bruto
    rendimiento_pct: float        # T. Ideal / T. Operativo
    calidad_pct: float            # Piezas Buenas / Piezas Totales
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
        "horas_produccion": 0.0,        # Tiempo bruto de producción (H Ini - F Fin)
        "horas_preparacion": 0.0,       # Tiempo de preparación
        "horas_indisponibilidad": 0.0,  # Incidencias: avería, mant. prev., limpieza
        "horas_paros": 0.0,             # Otras incidencias
        "tiempo_ideal": 0.0,
        "piezas_totales": 0.0,
        "piezas_malas": 0.0,
        "piezas_recuperadas": 0.0,
    }


def convertir_raw_a_metricas(raw: Dict[str, float]) -> Dict[str, float]:
    horas_produccion = raw.get("horas_produccion", 0.0)
    horas_preparacion = raw.get("horas_preparacion", 0.0)
    horas_indisponibilidad = raw.get("horas_indisponibilidad", 0.0)
    horas_paros = raw.get("horas_paros", 0.0)

    # T. Bruto = Producción + Preparación (tiempo total registrado sin contar incidencias aparte)
    horas_brutas = horas_produccion + horas_preparacion

    # T. Disponible = T. Bruto - Indisponibilidad (avería, mant. prev., limpieza)
    horas_disponible = max(horas_brutas - horas_indisponibilidad, 0.0)

    # T. Operativo = T. Disponible - Preparación - Paros
    horas_operativo = max(horas_disponible - horas_preparacion - horas_paros, 0.0)

    tiempo_ideal = raw.get("tiempo_ideal", 0.0)
    perdidas_rend = max(horas_operativo - tiempo_ideal, 0.0)
    piezas_totales = raw.get("piezas_totales", 0.0)
    piezas_malas = raw.get("piezas_malas", 0.0)
    piezas_recuperadas = raw.get("piezas_recuperadas", 0.0)
    scrap_final = max(piezas_malas - piezas_recuperadas, 0.0)
    buenas_finales = max(piezas_totales - scrap_final, 0.0)

    # Disponibilidad = (T. Disponible - T. Paros) / T. Bruto
    # Los paros reducen disponibilidad además de las incidencias de indisponibilidad
    disponibilidad_pct = clamp_pct(((horas_disponible - horas_paros) / horas_brutas * 100) if horas_brutas > 0 else 0.0)
    # Rendimiento = T. Ideal / T. Operativo
    rendimiento_pct = clamp_pct(
        (tiempo_ideal / horas_operativo * 100) if horas_operativo > 0 and tiempo_ideal > 0 else 0.0
    )
    # Calidad = Piezas Buenas / Piezas Totales
    calidad_pct = clamp_pct((buenas_finales / piezas_totales * 100) if piezas_totales > 0 else 0.0)
    oee_pct = (disponibilidad_pct * rendimiento_pct * calidad_pct) / 10000.0

    # Si hay muy pocas piezas el turno/día no es representativo: anular OEE
    if piezas_totales < MIN_PIEZAS_OEE:
        disponibilidad_pct = 0.0
        rendimiento_pct = 0.0
        calidad_pct = 0.0
        oee_pct = 0.0

    return {
        "horas_brutas": horas_brutas,
        "horas_disponible": horas_disponible,
        "horas_operativo": horas_operativo,
        "horas_produccion": horas_produccion,
        "horas_preparacion": horas_preparacion,
        "horas_indisponibilidad": horas_indisponibilidad,
        "horas_paros": horas_paros,
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
    # T3 de 22:00 a 06:00
    # El día del turno es cuando EMPIEZA el T3 (a las 22:00)
    if hour >= 22:
        # Empieza T3 esta noche, termina mañana a las 6
        limit = datetime.combine(base_day + timedelta(days=1), time(6, 0))
        day_key = base_day  # El turno pertenece al día de hoy
    else:
        # Estamos en la madrugada (hour < 6), el T3 empezó ayer
        limit = datetime.combine(base_day, time(6, 0))
        day_key = base_day - timedelta(days=1)  # El turno pertenece al día de ayer
    return "T3", limit, day_key


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


def calcular_paros_totales(stats: Dict[str, float]) -> float:
    """Devuelve el tiempo total de paros (indisponibilidad + paros)."""
    horas_indisponibilidad = stats.get("horas_indisponibilidad", 0.0)
    horas_paros = stats.get("horas_paros", 0.0)
    return horas_indisponibilidad + horas_paros


def calcular_preparacion(stats: Dict[str, float]) -> float:
    """Devuelve el tiempo de preparación."""
    return stats.get("horas_preparacion", 0.0) or 0.0


def clamp_pct(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 100:
        return 100.0
    return value


@dataclass
class RegistroCSV:
    """Representa un registro del CSV con sus tiempos calculados."""
    start_dt: Optional[datetime]
    end_dt: Optional[datetime]
    horas: float
    horas_netas: float
    proceso: str
    incidencia: str
    cantidad: float
    malas: float
    recuperadas: float
    referencia: str
    fecha: Optional[datetime]


def calcular_solapamiento(start1: datetime, end1: datetime, start2: datetime, end2: datetime) -> float:
    """Calcula las horas de solapamiento entre dos intervalos de tiempo."""
    if not all([start1, end1, start2, end2]):
        return 0.0
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    if overlap_start < overlap_end:
        return (overlap_end - overlap_start).total_seconds() / 3600.0
    return 0.0


def leer_maquina(csv_path: Path, ciclos: Dict[str, Dict[str, float]]) -> MachineSectionMetrics:
    # Totales
    total_horas_produccion = 0.0  # Tiempo bruto de registros Producción
    total_horas_preparacion = 0.0  # Tiempo bruto de registros Preparación
    total_horas_indisponibilidad = 0.0  # Avería + Mant. Prev. + Limpieza
    total_horas_paros = 0.0  # Otras incidencias
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

    # PASO 1: Leer todos los registros y calcular tiempos con H Ini/F Fin
    registros: List[RegistroCSV] = []
    with csv_path.open(encoding="utf-8-sig") as handler:
        reader = csv.DictReader(handler)
        if reader.fieldnames:
            reader.fieldnames = [(name or "").strip() for name in reader.fieldnames]
        for row in reader:
            start_time = parse_time_value(row.get("H Ini"))
            end_time = parse_time_value(row.get("F Fin"))
            fecha = parse_datetime(row.get("Fecha"))

            # Calcular datetime de inicio y fin
            start_dt = datetime.combine(fecha.date(), start_time) if fecha and start_time else None
            end_dt = None
            if start_dt and end_time:
                end_dt = datetime.combine(start_dt.date(), end_time)
                if end_dt < start_dt:
                    end_dt += timedelta(days=1)
                elif end_dt == start_dt:
                    end_dt = None

            # Calcular horas SIEMPRE con H Ini/F Fin
            horas = (end_dt - start_dt).total_seconds() / 3600.0 if start_dt and end_dt else 0.0

            proceso = normalizar_proceso(row.get("Proceso"))
            incidencia = (row.get("Incidencia", "") or "").strip()
            cantidad = parse_float(row.get("Cantidad"))
            malas = parse_float(row.get("Malas"))
            recuperadas = parse_float(row.get("Recu."))
            referencia = normalize_ref(row.get("Refer."))

            registros.append(RegistroCSV(
                start_dt=start_dt,
                end_dt=end_dt,
                horas=horas,
                horas_netas=parse_float(row.get("Tiempo")),
                proceso=proceso,
                incidencia=incidencia,
                cantidad=cantidad,
                malas=malas,
                recuperadas=recuperadas,
                referencia=referencia,
                fecha=fecha,
            ))

            # Actualizar rango de fechas
            if fecha:
                start = fecha if not start or fecha < start else start
                end = fecha if not end or fecha > end else end

    # PASO 2: Separar registros por tipo
    registros_prod_prep = [r for r in registros if r.proceso in ("produccion", "preparacion")]

    # Clasificar incidencias en dos categorías
    incidencias_disponibilidad = [r for r in registros if r.proceso == "incidencias"
                                   and clasificar_incidencia(r.incidencia) == "disponibilidad"]
    incidencias_paros = [r for r in registros if r.proceso == "incidencias"
                         and clasificar_incidencia(r.incidencia) == "paros"]

    # PASO 3: Procesar cada registro de Producción/Preparación
    for reg in registros_prod_prep:
        if not reg.start_dt or not reg.end_dt:
            continue

        # Calcular incidencias de DISPONIBILIDAD solapadas (avería, mant. prev., limpieza)
        tiempo_indisp_solapado = 0.0
        for inc in incidencias_disponibilidad:
            if inc.start_dt and inc.end_dt:
                tiempo_indisp_solapado += calcular_solapamiento(
                    reg.start_dt, reg.end_dt, inc.start_dt, inc.end_dt
                )

        # Calcular incidencias de PAROS solapadas (otras incidencias)
        tiempo_paros_solapado = 0.0
        for inc in incidencias_paros:
            if inc.start_dt and inc.end_dt:
                tiempo_paros_solapado += calcular_solapamiento(
                    reg.start_dt, reg.end_dt, inc.start_dt, inc.end_dt
                )

        ciclo_seg = ciclos_maquina.get(reg.referencia)

        # Dividir en turnos usando el tiempo BRUTO para distribución proporcional
        segmentos = dividir_en_turnos(reg.start_dt, reg.horas)
        if not segmentos:
            start_time = reg.start_dt.time() if reg.start_dt else None
            turno_def = determinar_turno(start_time)
            day_def = reg.fecha.date() if reg.fecha else datetime.today().date()
            segmentos = [(turno_def, day_def, reg.horas)]

        total_seg_hours = sum(h for _, _, h in segmentos) or 1.0

        # Evitar doble descuento cuando incidencias solapan entre si
        total_inc = tiempo_indisp_solapado + tiempo_paros_solapado
        if total_inc > reg.horas and reg.horas > 0:
            scale = reg.horas / total_inc
            tiempo_indisp_solapado *= scale
            tiempo_paros_solapado *= scale

        # Factores de reducción
        factor_indisp = tiempo_indisp_solapado / reg.horas if reg.horas > 0 else 0.0
        factor_paros = tiempo_paros_solapado / reg.horas if reg.horas > 0 else 0.0

        # Acumular totales
        if reg.proceso == "produccion":
            total_horas_produccion += reg.horas
        else:  # preparacion
            total_horas_preparacion += reg.horas

        total_horas_indisponibilidad += tiempo_indisp_solapado
        total_horas_paros += tiempo_paros_solapado

        piezas_totales += reg.cantidad
        piezas_malas += reg.malas
        piezas_recuperadas += reg.recuperadas

        # Distribuir por turnos
        for turno_seg, day_seg, horas_seg_bruto in segmentos:
            # Distribuir proporcionalmente las incidencias en cada segmento
            horas_seg_indisp = horas_seg_bruto * factor_indisp
            horas_seg_paros = horas_seg_bruto * factor_paros

            peso = horas_seg_bruto / total_seg_hours if total_seg_hours > 0 else 0.0
            piezas_seg = reg.cantidad * peso
            malas_seg = reg.malas * peso
            rec_seg = reg.recuperadas * peso

            shift_stats = shift_raw[turno_seg]
            day_stats = daily_stats[day_seg]
            day_shift_stats = daily_shift_raw[day_seg][turno_seg]

            # Acumular piezas
            shift_stats["piezas_totales"] += piezas_seg
            shift_stats["piezas_malas"] += malas_seg
            shift_stats["piezas_recuperadas"] += rec_seg
            day_stats["piezas_totales"] += piezas_seg
            day_stats["piezas_malas"] += malas_seg
            day_stats["piezas_recuperadas"] += rec_seg
            day_shift_stats["piezas_totales"] += piezas_seg
            day_shift_stats["piezas_malas"] += malas_seg
            day_shift_stats["piezas_recuperadas"] += rec_seg

            # Acumular incidencias (igual para prod y prep)
            shift_stats["horas_indisponibilidad"] += horas_seg_indisp
            day_stats["horas_indisponibilidad"] += horas_seg_indisp
            day_shift_stats["horas_indisponibilidad"] += horas_seg_indisp
            shift_stats["horas_paros"] += horas_seg_paros
            day_stats["horas_paros"] += horas_seg_paros
            day_shift_stats["horas_paros"] += horas_seg_paros

            if reg.proceso == "produccion":
                shift_stats["horas_produccion"] += horas_seg_bruto
                day_stats["horas_produccion"] += horas_seg_bruto
                day_shift_stats["horas_produccion"] += horas_seg_bruto
                if ciclo_seg and ciclo_seg > 0 and piezas_seg > 0:
                    seg_tiempo_ideal = piezas_seg / ciclo_seg
                    tiempo_ideal += seg_tiempo_ideal
                    shift_stats["tiempo_ideal"] += seg_tiempo_ideal
                    day_stats["tiempo_ideal"] += seg_tiempo_ideal
                    day_shift_stats["tiempo_ideal"] += seg_tiempo_ideal
                # Estadística por referencia
                if piezas_seg > 0 or horas_seg_bruto > 0:
                    horas_seg_neto = reg.horas_netas * peso
                    ref_entry = ref_stats.setdefault(
                        reg.referencia,
                        {
                            "ciclo_ideal": ciclo_seg if ciclo_seg and ciclo_seg > 0 else None,
                            "dias": {},
                        },
                    )
                    if ciclo_seg and ciclo_seg > 0:
                        ref_entry["ciclo_ideal"] = ciclo_seg
                    day_entry = ref_entry["dias"].setdefault(
                        day_seg,
                        {"piezas": 0.0, "horas_brutas": 0.0, "horas_netas": 0.0},
                    )
                    day_entry["piezas"] += piezas_seg
                    day_entry["horas_brutas"] += horas_seg_bruto
                    day_entry["horas_netas"] += horas_seg_neto
            else:  # preparacion
                shift_stats["horas_preparacion"] += horas_seg_bruto
                day_stats["horas_preparacion"] += horas_seg_bruto
                day_shift_stats["horas_preparacion"] += horas_seg_bruto

    # CÁLCULOS FINALES según estructura OEE
    # T. Bruto = Producción + Preparación (todo lo registrado)
    horas_brutas = total_horas_produccion + total_horas_preparacion

    # T. Disponible = T. Bruto - Indisponibilidad (avería, mant. prev., limpieza)
    horas_disponible = max(horas_brutas - total_horas_indisponibilidad, 0.0)

    # T. Operativo = T. Disponible - Preparación - Paros
    horas_operativo = max(horas_disponible - total_horas_preparacion - total_horas_paros, 0.0)

    perdidas_rend = max(horas_operativo - tiempo_ideal, 0.0)
    scrap_final = max(piezas_malas - piezas_recuperadas, 0.0)
    buenas_finales = max(piezas_totales - scrap_final, 0.0)

    # Disponibilidad = (T. Disponible - T. Paros) / T. Bruto
    disponibilidad_pct = clamp_pct(((horas_disponible - total_horas_paros) / horas_brutas * 100) if horas_brutas > 0 else 0.0)
    # Rendimiento = T. Ideal / T. Operativo
    rendimiento_pct = clamp_pct((tiempo_ideal / horas_operativo * 100) if horas_operativo > 0 and tiempo_ideal > 0 else 0.0)
    # Calidad = Piezas Buenas / Piezas Totales
    calidad_pct = clamp_pct((buenas_finales / piezas_totales * 100) if piezas_totales > 0 else 0.0)
    oee_pct = (disponibilidad_pct * rendimiento_pct * calidad_pct) / 10000.0

    shift_stats_summary: Dict[str, Dict[str, float]] = {}
    for shift, raw_data in shift_raw.items():
        metrics_shift = convertir_raw_a_metricas(raw_data)
        metrics_shift["raw"] = raw_data.copy()
        shift_stats_summary[shift] = metrics_shift

    daily_shift_stats_final: Dict[date, Dict[str, Dict[str, float]]] = {}
    for day, shift_data in daily_shift_raw.items():
        stats_day: Dict[str, Dict[str, float]] = {}
        for shift, raw_data in shift_data.items():
            stats_day[shift] = convertir_raw_a_metricas(raw_data)
        daily_shift_stats_final[day] = stats_day

    return MachineSectionMetrics(
        name=display_name,
        full_name=stem,
        week_code=week_code,
        horas_brutas=horas_brutas,
        horas_indisponibilidad=total_horas_indisponibilidad,
        horas_disponible=horas_disponible,
        horas_preparacion=total_horas_preparacion,
        horas_paros=total_horas_paros,
        horas_operativo=horas_operativo,
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
        daily_shift_stats=daily_shift_stats_final,
        ref_stats={
            ref: {
                "ciclo_ideal": data.get("ciclo_ideal"),
                "dias": {
                    day: {
                        "piezas": val.get("piezas", 0.0),
                        "horas_brutas": val.get("horas_brutas", 0.0),
                        "horas_netas": val.get("horas_netas", 0.0),
                    }
                    for day, val in sorted(data.get("dias", {}).items())
                },
            }
            for ref, data in ref_stats.items()
        },
    )


def calcular_totales(maquinas: List[MachineSectionMetrics]) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    horas_brutas = sum(m.horas_brutas for m in maquinas)
    horas_indisponibilidad = sum(m.horas_indisponibilidad for m in maquinas)
    horas_disponible = sum(m.horas_disponible for m in maquinas)
    horas_preparacion = sum(m.horas_preparacion for m in maquinas)
    horas_paros = sum(m.horas_paros for m in maquinas)
    horas_operativo = sum(m.horas_operativo for m in maquinas)
    tiempo_ideal = sum(m.tiempo_ideal for m in maquinas)
    piezas_totales = sum(m.piezas_totales for m in maquinas)
    piezas_malas = sum(m.piezas_malas for m in maquinas)
    piezas_recuperadas = sum(m.piezas_recuperadas for m in maquinas)

    perdidas_rend = max(horas_operativo - tiempo_ideal, 0.0)
    scrap_final = max(piezas_malas - piezas_recuperadas, 0.0)
    buenas_finales = max(piezas_totales - scrap_final, 0.0)

    # Disponibilidad = (T. Disponible - T. Paros) / T. Bruto
    disponibilidad_pct = clamp_pct(((horas_disponible - horas_paros) / horas_brutas * 100) if horas_brutas > 0 else 0.0)
    # Rendimiento = T. Ideal / T. Operativo
    rendimiento_pct = clamp_pct((tiempo_ideal / horas_operativo * 100) if horas_operativo > 0 and tiempo_ideal > 0 else 0.0)
    calidad_pct = clamp_pct((buenas_finales / piezas_totales * 100) if piezas_totales > 0 else 0.0)
    oee_pct = (disponibilidad_pct * rendimiento_pct * calidad_pct) / 10000.0

    totals = {
        "horas_brutas": horas_brutas,
        "horas_indisponibilidad": horas_indisponibilidad,
        "horas_disponible": horas_disponible,
        "horas_preparacion": horas_preparacion,
        "horas_paros": horas_paros,
        "horas_operativo": horas_operativo,
        "tiempo_ideal": tiempo_ideal,
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
        shift_data = [m.shift_stats.get(shift, {}) for m in maquinas]
        horas_brutas_t = sum(s.get("horas_brutas", 0.0) for s in shift_data)
        horas_disponible_t = sum(s.get("horas_disponible", 0.0) for s in shift_data)
        horas_operativo_t = sum(s.get("horas_operativo", 0.0) for s in shift_data)
        horas_preparacion_t = sum(s.get("horas_preparacion", 0.0) for s in shift_data)
        horas_indisponibilidad_t = sum(s.get("horas_indisponibilidad", 0.0) for s in shift_data)
        horas_paros_t = sum(s.get("horas_paros", 0.0) for s in shift_data)
        tiempo_ideal_t = sum(s.get("tiempo_ideal", 0.0) for s in shift_data)
        piezas_totales_t = sum(s.get("piezas_totales", 0.0) for s in shift_data)
        piezas_malas_t = sum(s.get("piezas_malas", 0.0) for s in shift_data)
        piezas_rec_t = sum(s.get("piezas_recuperadas", 0.0) for s in shift_data)

        perdidas_t = max(horas_operativo_t - tiempo_ideal_t, 0.0)
        scrap_t = max(piezas_malas_t - piezas_rec_t, 0.0)
        buenas_t = max(piezas_totales_t - scrap_t, 0.0)

        disp_t = clamp_pct(((horas_disponible_t - horas_paros_t) / horas_brutas_t * 100) if horas_brutas_t > 0 else 0.0)
        rdo_t = clamp_pct((tiempo_ideal_t / horas_operativo_t * 100) if horas_operativo_t > 0 and tiempo_ideal_t > 0 else 0.0)
        cal_t = clamp_pct((buenas_t / piezas_totales_t * 100) if piezas_totales_t > 0 else 0.0)
        oee_t = (disp_t * rdo_t * cal_t) / 10000.0

        shift_totals[shift] = {
            "horas_brutas": horas_brutas_t,
            "horas_disponible": horas_disponible_t,
            "horas_operativo": horas_operativo_t,
            "horas_preparacion": horas_preparacion_t,
            "horas_indisponibilidad": horas_indisponibilidad_t,
            "horas_paros": horas_paros_t,
            "tiempo_ideal": tiempo_ideal_t,
            "perdidas_rend": perdidas_t,
            "buenas_finales": buenas_t,
            "piezas_totales": piezas_totales_t,
            "disponibilidad_pct": disp_t,
            "rendimiento_pct": rdo_t,
            "calidad_pct": cal_t,
            "oee_pct": oee_t,
        }

    return totals, shift_totals


def calcular_resumen_diario(maquinas: List[MachineSectionMetrics]) -> List[Tuple[date, Dict[str, float]]]:
    diarios: Dict[date, Dict[str, float]] = defaultdict(
        lambda: {
            "horas_produccion": 0.0,
            "horas_preparacion": 0.0,
            "horas_indisponibilidad": 0.0,
            "horas_paros": 0.0,
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
            stats["horas_preparacion"] += data.get("horas_preparacion", 0.0)
            stats["horas_indisponibilidad"] += data.get("horas_indisponibilidad", 0.0)
            stats["horas_paros"] += data.get("horas_paros", 0.0)
            stats["tiempo_ideal"] += data.get("tiempo_ideal", 0.0)
            stats["piezas_totales"] += data.get("piezas_totales", 0.0)
            stats["piezas_malas"] += data.get("piezas_malas", 0.0)
            stats["piezas_recuperadas"] += data.get("piezas_recuperadas", 0.0)

    resumen: List[Tuple[date, Dict[str, float]]] = []
    for day in sorted(diarios.keys()):
        data = diarios[day]
        # T. Bruto = Producción + Preparación
        horas_brutas = data["horas_produccion"] + data["horas_preparacion"]
        # T. Disponible = T. Bruto - Indisponibilidad
        horas_disponible = max(horas_brutas - data["horas_indisponibilidad"], 0.0)
        # T. Operativo = T. Disponible - Preparación - Paros
        horas_operativo = max(horas_disponible - data["horas_preparacion"] - data["horas_paros"], 0.0)

        perdidas_rend = max(horas_operativo - data["tiempo_ideal"], 0.0)
        scrap_final = max(data["piezas_malas"] - data["piezas_recuperadas"], 0.0)
        buenas_finales = max(data["piezas_totales"] - scrap_final, 0.0)

        disponibilidad_pct = clamp_pct(
            ((horas_disponible - data["horas_paros"]) / horas_brutas * 100) if horas_brutas > 0 else 0.0
        )
        rendimiento_pct = clamp_pct(
            (data["tiempo_ideal"] / horas_operativo * 100)
            if horas_operativo > 0 and data["tiempo_ideal"] > 0
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
                    "horas_disponible": horas_disponible,
                    "horas_operativo": horas_operativo,
                    "horas_preparacion": data["horas_preparacion"],
                    "horas_indisponibilidad": data["horas_indisponibilidad"],
                    "horas_paros": data["horas_paros"],
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
        paros_total = metric.horas_indisponibilidad + metric.horas_paros
        rows.append(
            [
                metric.name.upper(),
                hours_to_hhmmss(metric.horas_disponible),
                hours_to_hhmmss(metric.horas_operativo),
                hours_to_hhmmss(metric.horas_preparacion),
                hours_to_hhmmss(paros_total),
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

    paros_total_sec = totales.get("horas_indisponibilidad", 0.0) + totales.get("horas_paros", 0.0)
    rows.append(
        [
            "Total sección",
            hours_to_hhmmss(totales.get("horas_disponible", 0.0)),
            hours_to_hhmmss(totales.get("horas_operativo", 0.0)),
            hours_to_hhmmss(totales.get("horas_preparacion", 0.0)),
            hours_to_hhmmss(paros_total_sec),
            hours_to_hhmmss(totales.get("tiempo_ideal", 0.0)),
            hours_to_hhmmss(totales.get("perdidas_rend", 0.0)),
            format_units(totales.get("piezas_totales", 0.0)),
            format_units(totales.get("buenas_finales", 0.0)),
            format_pct(totales.get("disponibilidad_pct", 0.0)),
            format_pct(totales.get("rendimiento_pct", 0.0)),
            format_pct(totales.get("calidad_pct", 0.0)),
            format_pct(totales.get("oee_pct", 0.0)),
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
            hours_to_hhmmss(stats.get("horas_disponible", 0.0)),
            hours_to_hhmmss(stats.get("horas_operativo", 0.0)),
            hours_to_hhmmss(stats.get("horas_preparacion", 0.0)),
            hours_to_hhmmss(calcular_paros_totales(stats)),
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
            "horas_disponible": metric.horas_disponible,
            "horas_operativo": metric.horas_operativo,
            "horas_preparacion": metric.horas_preparacion,
            "horas_indisponibilidad": metric.horas_indisponibilidad,
            "horas_paros": metric.horas_paros,
            "tiempo_ideal": metric.tiempo_ideal,
            "perdidas_rend": metric.perdidas_rend,
            "buenas_finales": metric.buenas_finales,
            "piezas_totales": metric.piezas_totales,
            "disponibilidad_pct": metric.disponibilidad_pct,
            "rendimiento_pct": metric.rendimiento_pct,
            "calidad_pct": metric.calidad_pct,
            "oee_pct": metric.oee_pct,
        }

    MAX_ROWS_PER_PAGE = 20

    def _render_table_page(titulo: str, rows, col_labels, bold_rows, highlight_rows):
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle(titulo, fontsize=16, fontweight="bold", color="#263238")
        ax = fig.add_subplot(111)
        ax.axis("off")
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
        return fig

    for metric in sorted(maquinas, key=machine_sort_key):
        titulo = metric.name.upper()
        if semana_label:
            titulo = f"{titulo} · W{semana_label}"

        # Construir bloques diarios (un bloque = cabecera día + turnos + total + separador)
        day_blocks: List[List[List[str]]] = []
        day_bold_offsets: List[List[int]] = []
        for day in sorted(metric.daily_stats.keys()):
            block: List[List[str]] = []
            bold_offsets: List[int] = []
            bold_offsets.append(len(block))
            block.append([day.strftime("%d-%b")] + [""] * (len(col_labels) - 1))
            shift_day = metric.daily_shift_stats.get(day, {})
            for shift in SHIFT_LABELS:
                stats = shift_day.get(shift)
                if stats:
                    block.append(format_row(shift, stats))
            general_day = convertir_raw_a_metricas(metric.daily_stats[day])
            bold_offsets.append(len(block))
            block.append(format_row("Total", general_day))
            block.append([""] * len(col_labels))  # separador
            day_blocks.append(block)
            day_bold_offsets.append(bold_offsets)

        # Totales globales (siempre en la última página)
        footer_rows: List[List[str]] = []
        footer_bold: List[int] = []
        footer_highlight: List[int] = []
        for shift in SHIFT_LABELS:
            shift_stats = metric.shift_stats.get(shift)
            if shift_stats:
                footer_highlight.append(len(footer_rows))
                footer_rows.append(format_row(f"{shift} Total", shift_stats))
        footer_bold.append(len(footer_rows))
        footer_rows.append(format_row(f"TOTAL {metric.name.upper()}", general_stats(metric)))

        # Paginar bloques diarios
        page_rows: List[List[str]] = []
        page_bold: set[int] = set()
        page_highlight: set[int] = set()
        page_num = 0

        for i, (block, bold_offs) in enumerate(zip(day_blocks, day_bold_offsets)):
            # Si añadir este bloque supera el límite, flush página actual
            if page_rows and len(page_rows) + len(block) > MAX_ROWS_PER_PAGE:
                page_title = f"{titulo} ({page_num + 1})" if len(day_blocks) > 5 else titulo
                figures.append(_render_table_page(page_title, page_rows, col_labels, page_bold, page_highlight))
                page_rows = []
                page_bold = set()
                page_highlight = set()
                page_num += 1

            base = len(page_rows)
            for off in bold_offs:
                page_bold.add(base + off)
            page_rows.extend(block)

        # Añadir footer de totales a la última página
        # Si no cabe, crear página nueva
        if page_rows and len(page_rows) + len(footer_rows) > MAX_ROWS_PER_PAGE + 5:
            page_title = f"{titulo} ({page_num + 1})" if page_num > 0 else titulo
            figures.append(_render_table_page(page_title, page_rows, col_labels, page_bold, page_highlight))
            page_rows = []
            page_bold = set()
            page_highlight = set()
            page_num += 1

        base = len(page_rows)
        for off in footer_highlight:
            page_highlight.add(base + off)
        for off in footer_bold:
            page_bold.add(base + off)
        page_rows.extend(footer_rows)

        page_title = f"{titulo} ({page_num + 1})" if page_num > 0 else titulo
        figures.append(_render_table_page(page_title, page_rows, col_labels, page_bold, page_highlight))

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
        image = OffsetImage(logo, zoom=0.15)
        ab = AnnotationBbox(
            image,
            (0.06, 0.86),
            frameon=False,
            xycoords="axes fraction",
            box_alignment=(0, 0.5),
        )
        ax.add_artist(ab)

    ax.text(0.55, 0.78, "Informe OEE", fontsize=30, fontweight="bold", color="#263238")
    ax.text(0.55, 0.64, f"Sección: {section_name.upper()}", fontsize=18, color="#546E7A")

    fecha_ini_txt = fecha_inicio.strftime("%d/%m/%Y") if fecha_inicio else "-"
    fecha_fin_txt = fecha_fin.strftime("%d/%m/%Y") if fecha_fin else "-"
    paros_totales = calcular_paros_totales(totales)
    info_lines = [
        f"Periodo: {fecha_ini_txt} → {fecha_fin_txt}",
        f"Semana: {semana_label or '-'}",
        f"OEE total: {format_pct(totales.get('oee_pct', 0.0))}",
        f"Disponibilidad: {format_pct(totales.get('disponibilidad_pct', 0.0))}",
        f"Rendimiento: {format_pct(totales.get('rendimiento_pct', 0.0))}",
        f"Calidad: {format_pct(totales.get('calidad_pct', 0.0))}",
        f"T. bruto: {hours_to_hhmmss(totales.get('horas_brutas', 0.0))}",
        f"T. disponible: {hours_to_hhmmss(totales.get('horas_disponible', 0.0))}",
        f"T. operativo: {hours_to_hhmmss(totales.get('horas_operativo', 0.0))}",
        f"T. paros: {hours_to_hhmmss(paros_totales)}",
        f"T. ideal: {hours_to_hhmmss(totales.get('tiempo_ideal', 0.0))}",
        f"Pérdidas rendimiento: {hours_to_hhmmss(totales.get('perdidas_rend', 0.0))}",
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
        image = OffsetImage(logo_image, zoom=0.05)
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
                total_horas_netas = 0.0
                for day in sorted(dias.keys()):
                    piezas = dias[day].get("piezas", 0.0)
                    horas = dias[day].get("horas_netas", dias[day].get("horas_brutas", 0.0))
                    total_piezas += piezas
                    total_horas_netas += horas
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
                ciclo_real_avg = (total_piezas / total_horas_netas) if total_horas_netas > 0 else None
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
                image = OffsetImage(logo_image, zoom=0.05)
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
            hours_to_hhmmss(stats.get("horas_disponible", 0.0)),
            hours_to_hhmmss(stats.get("horas_operativo", 0.0)),
            hours_to_hhmmss(stats.get("horas_preparacion", 0.0)),
            hours_to_hhmmss(calcular_paros_totales(stats)),
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
            "horas_disponible": metric.horas_disponible,
            "horas_operativo": metric.horas_operativo,
            "horas_preparacion": metric.horas_preparacion,
            "horas_indisponibilidad": metric.horas_indisponibilidad,
            "horas_paros": metric.horas_paros,
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
        resultados.append(pdf_path)

    return resultados


if __name__ == "__main__":
    generar_informes_oee_secciones()
