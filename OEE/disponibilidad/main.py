from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import unicodedata

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle, Patch
from collections import defaultdict

from OEE.utils.data_files import listar_csv_por_seccion


CATEGORY_CONFIG: Dict[str, Dict[str, str]] = {
    "produccion": {"label": "Producción", "color": "#2E7D32"},
    "preparacion": {"label": "Preparación", "color": "#F9A825"},
    "incidencias": {"label": "Incidencias", "color": "#C62828"},
}

SECTION_TITLE_STYLE = {"fontsize": 11, "fontweight": "bold", "color": "#263238"}
BODY_FONT_SIZE = 10

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[1] / "informes"


@dataclass
class DisponibilidadMetrics:
    resource_name: str
    start: Optional[datetime]
    end: Optional[datetime]
    produccion: float
    preparacion: float
    incidencias: float
    incidencias_detalle: List[Tuple[str, float]] = field(default_factory=list)
    total_incidents: int = 0
    averia_count: int = 0
    averia_hours: float = 0.0
    time_unit: str = "mes"
    time_unit_plural: str = "meses"
    time_unit_title: str = "mensual"
    period_stats: Dict[date, Dict[str, float]] = field(default_factory=dict)
    period_incidents: Dict[date, Dict[str, float]] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return self.produccion + self.preparacion + self.incidencias

    @property
    def disponibilidad(self) -> float:
        if self.total == 0:
            return 0.0
        return self.produccion / self.total

    def top_incidencias(self, n: int = 3) -> List[Tuple[str, float]]:
        return self.incidencias_detalle[:n]

    @property
    def dias_registrados(self) -> Optional[int]:
        if self.start and self.end:
            return (self.end.date() - self.start.date()).days + 1
        return None

    @property
    def mtbf_mecanico(self) -> Optional[float]:
        if self.averia_count == 0:
            return None
        if self.produccion == 0:
            return None
        return self.produccion / self.averia_count

    @property
    def mttr_mecanico(self) -> Optional[float]:
        if self.averia_count == 0:
            return None
        return self.averia_hours / self.averia_count


def normalizar_proceso(value: str) -> str:
    if not value:
        return "produccion"
    value = (
        unicodedata.normalize("NFKD", value)
        .encode("ASCII", "ignore")
        .decode("utf-8")
        .lower()
        .strip()
    )
    if value.startswith("produ"):
        return "produccion"
    if value.startswith("prepa"):
        return "preparacion"
    if value.startswith("inci"):
        return "incidencias"
    return "produccion"


def parse_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return None


def parse_float(value: str) -> float:
    if not value:
        return 0.0
    value = value.replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return 0.0


def extraer_nombre_recurso(csv_path: Path) -> str:
    return csv_path.stem.split("-")[0]


def determine_time_unit(
    start: Optional[datetime], end: Optional[datetime]
) -> Tuple[str, str, str]:
    if not start or not end:
        return "mes", "meses", "mensual"
    days = (end.date() - start.date()).days + 1
    if days <= 7:
        return "día", "días", "diaria"
    if days <= 60:
        return "semana", "semanas", "semanal"
    return "mes", "meses", "mensual"


def normalize_period_start(fecha: datetime, unit: str) -> date:
    if unit == "día":
        return fecha.date()
    if unit == "semana":
        iso = fecha.isocalendar()
        return datetime.fromisocalendar(iso.year, iso.week, 1).date()
    # mes
    return fecha.replace(day=1).date()


def format_period_label(period_start: date, unit: str) -> str:
    if unit == "día":
        return period_start.strftime("%d-%b")
    if unit == "semana":
        iso = period_start.isocalendar()
        return f"Sem {iso.week:02d}-{str(iso.year)[-2:]}"
    return period_start.strftime("%b-%y")


def leer_metricas(csv_path: Path) -> DisponibilidadMetrics:
    produccion = 0.0
    preparacion = 0.0
    incidencias = 0.0
    incidencias_por_tipo: Dict[str, float] = defaultdict(float)
    total_incidents = 0
    averia_count = 0
    averia_hours = 0.0
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    records: List[Tuple[Optional[datetime], str, float, str]] = []

    with csv_path.open(encoding="utf-8-sig", newline="") as handler:
        reader = csv.DictReader(handler)
        if reader.fieldnames:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            hours = parse_float(row.get("Tiempo", ""))
            proceso = normalizar_proceso(row.get("Proceso", ""))
            fecha = parse_datetime(row.get("Fecha", ""))

            if fecha:
                start = fecha if not start or fecha < start else start
                end = fecha if not end or fecha > end else end

            detalle = (row.get("Incidencia", "") or "").strip() or "Sin detalle"

            if proceso == "produccion":
                produccion += hours
            elif proceso == "preparacion":
                preparacion += hours
            else:
                incidencias += hours
                incidencias_por_tipo[detalle] += hours
                total_incidents += 1
                if detalle.upper() == "AVERIA":
                    averia_count += 1
                    averia_hours += hours
            records.append((fecha, proceso, hours, detalle))

    incidencias_detalle = sorted(
        incidencias_por_tipo.items(), key=lambda item: item[1], reverse=True
    )

    unit, unit_plural, unit_title = determine_time_unit(start, end)
    period_stats_map: Dict[date, Dict[str, float]] = defaultdict(
        lambda: {"produccion": 0.0, "preparacion": 0.0, "incidencias": 0.0}
    )
    period_incidents_map: Dict[date, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for fecha, proceso, hours, detalle in records:
        if not fecha:
            continue
        period_key = normalize_period_start(fecha, unit)
        if proceso == "produccion":
            period_stats_map[period_key]["produccion"] += hours
        elif proceso == "preparacion":
            period_stats_map[period_key]["preparacion"] += hours
        else:
            period_stats_map[period_key]["incidencias"] += hours
            period_incidents_map[period_key][detalle] += hours

    ordered_period_stats = dict(sorted(period_stats_map.items()))
    ordered_period_incidents = {
        period: dict(sorted(inc.items()))
        for period, inc in sorted(period_incidents_map.items())
    }

    return DisponibilidadMetrics(
        resource_name=extraer_nombre_recurso(csv_path),
        start=start,
        end=end,
        produccion=produccion,
        preparacion=preparacion,
        incidencias=incidencias,
        incidencias_detalle=incidencias_detalle,
        total_incidents=total_incidents,
        averia_count=averia_count,
        averia_hours=averia_hours,
        time_unit=unit,
        time_unit_plural=unit_plural,
        time_unit_title=unit_title,
        period_stats=ordered_period_stats,
        period_incidents=ordered_period_incidents,
    )


def cargar_logo(logo_path: Optional[Path]):
    if not logo_path:
        return None
    try:
        return mpimg.imread(logo_path)
    except (FileNotFoundError, OSError):
        return None


def enmarcar_seccion(ax) -> None:
    rect = Rectangle(
        (0.0, 0.0),
        1.0,
        1.0,
        transform=ax.transAxes,
        facecolor="#FAFAFA",
        edgecolor="#CFD8DC",
        linewidth=0.8,
        zorder=-1,
        clip_on=False,
    )
    ax.add_patch(rect)


def format_optional_hours(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:0.2f} h"


def build_page_one(
    metrics: DisponibilidadMetrics, logo_image: Optional[any]
) -> plt.Figure:
    fig = plt.figure(figsize=(8.27, 11.69))
    gs = fig.add_gridspec(
        nrows=5, ncols=1, height_ratios=[0.55, 0.95, 0.75, 0.25, 1.0]
    )

    header_ax = fig.add_subplot(gs[0, 0])
    header_ax.axis("off")
    header_ax.set_xlim(0, 1)
    header_ax.set_ylim(0, 1)
    if logo_image is not None:
        imagebox = OffsetImage(logo_image, zoom=0.18)
        ab = AnnotationBbox(
            imagebox, (0.08, 0.5), frameon=False, xycoords="axes fraction"
        )
        header_ax.add_artist(ab)
    else:
        header_ax.text(
            0.08,
            0.5,
            "Logo corporativo",
            fontsize=12,
            fontweight="bold",
            va="center",
        )

    title_x = 0.4
    header_ax.text(
        title_x,
        0.55,
        "Informe Disponibilidad",
        fontsize=22,
        fontweight="bold",
        color="#000000",
        ha="left",
        transform=header_ax.transAxes,
        zorder=2,
    )
    periodo = "Sin datos"
    if metrics.start and metrics.end:
        periodo = f"{metrics.start.strftime('%d/%m/%Y')} → {metrics.end.strftime('%d/%m/%Y')}"

    header_ax.text(
        title_x,
        0.30,
        f"Recurso: {metrics.resource_name.upper()}",
        fontsize=11,
        color="#424242",
        ha="left",
        transform=header_ax.transAxes,
        zorder=2,
    )
    header_ax.text(
        title_x,
        0.08,
        f"Periodo: {periodo}",
        fontsize=11,
        fontweight="bold",
        color="#263238",
        ha="left",
        transform=header_ax.transAxes,
        zorder=2,
    )

    resumen_ax = fig.add_subplot(gs[1, 0])
    resumen_ax.axis("off")
    enmarcar_seccion(resumen_ax)
    resumen_ax.text(
        0.0, 0.92, "Resumen ejecutivo de disponibilidad", **SECTION_TITLE_STYLE
    )
    resumen_ax.text(
        0.00,
        1.06,
        f"Disponibilidad del periodo: {metrics.disponibilidad * 100:0.2f} %",
        transform=resumen_ax.transAxes,
        fontsize=18,
        fontweight="bold",
    )
    left_lines = [
        f"Horas brutas: {metrics.total:0.2f} h",
        f"Producción efectiva: {metrics.produccion:0.2f} h",
    ]
    right_lines = [
        f"Preparación: {metrics.preparacion:0.2f} h",
        f"Incidencias: {metrics.incidencias:0.2f} h",
        (
            f"MTBF – Tiempo medio entre averías: "
            f"{format_optional_hours(metrics.mtbf_mecanico)}"
        ),
        (
            f"MTTR – Tiempo medio de reparación: "
            f"{format_optional_hours(metrics.mttr_mecanico)}"
        ),
    ]
    start_y_left = 0.78
    step_left = 0.12
    for idx, text in enumerate(left_lines):
        resumen_ax.text(
            0.02,
            start_y_left - idx * step_left,
            text,
            fontsize=BODY_FONT_SIZE,
        )

    start_y_right = 0.78
    step_right = 0.11
    for idx, text in enumerate(right_lines):
        resumen_ax.text(
            0.55,
            start_y_right - idx * step_right,
            text,
            fontsize=BODY_FONT_SIZE,
        )
    resumen_ax.text(
        0.02,
        0.18,
        "MTBF – Mean Time Between Failures: tiempo medio entre averías (solo inc. 'AVERIA').",
        fontsize=9,
        color="#424242",
    )
    resumen_ax.text(
        0.02,
        0.05,
        "MTTR – Mean Time To Repair: tiempo medio de reparación (solo inc. 'AVERIA').",
        fontsize=9,
        color="#424242",
    )

    bar_ax = fig.add_subplot(gs[2, 0])
    enmarcar_seccion(bar_ax)
    bar_ax.set_title(
        "Distribución global de tiempos en el periodo",
        loc="left",
        pad=8,
        fontdict=SECTION_TITLE_STYLE,
    )

    total = max(metrics.total, 0.01)
    left = 0.0
    legend_entries = []
    for key in ("produccion", "preparacion", "incidencias"):
        value = getattr(metrics, key)
        config = CATEGORY_CONFIG[key]
        bar_ax.barh(
            y=[0],
            width=[value],
            left=left,
            color=config["color"],
            height=0.05,
        )
        porcentaje = value / metrics.total * 100 if metrics.total else 0.0
        legend_entries.append(
            (
                config["color"],
                f"{config['label']} ({value:0.2f} h · {porcentaje:0.1f}%)",
            )
        )
        left += value

    bar_ax.set_xlim(0, total)
    tick_count = 5
    xticks = [total * i / tick_count for i in range(tick_count + 1)]
    bar_ax.set_xticks(xticks)
    bar_ax.set_xticklabels([f"{tick:0.0f}" for tick in xticks], fontsize=9)
    bar_ax.set_yticks([])
    bar_ax.set_ylim(-0.2, 0.2)
    for spine in bar_ax.spines.values():
        spine.set_visible(False)
    bar_ax.grid(axis="x", linestyle="--", alpha=0.2)
    bar_ax.text(
        0.0,
        -0.25,
        f"Objetivo de disponibilidad: 85%  →  Situación actual: {metrics.disponibilidad * 100:0.2f}%",
        transform=bar_ax.transAxes,
        fontsize=9,
        color="#424242",
    )
    handles = [Patch(facecolor=color, edgecolor=color) for color, _ in legend_entries]
    legend_ax = fig.add_subplot(gs[3, 0])
    enmarcar_seccion(legend_ax)
    legend_ax.axis("off")
    for idx, (handle, (_, label)) in enumerate(zip(handles, legend_entries)):
        x = 0.02 + idx * 0.32
        legend_ax.add_patch(
            Rectangle((x, 0.35), 0.03, 0.3, facecolor=handle.get_facecolor())
        )
        legend_ax.text(x + 0.04, 0.5, label, va="center", fontsize=9)

    tabla_ax = fig.add_subplot(gs[4, 0])
    tabla_ax.axis("off")
    enmarcar_seccion(tabla_ax)
    tabla_ax.text(
        0.0,
        0.88,
        "Top incidencias acumuladas (horas)",
        **SECTION_TITLE_STYLE,
    )

    top_incidencias = metrics.top_incidencias(5)
    if top_incidencias and metrics.incidencias > 0:
        col_labels = ["#", "Incidencia", "Horas", "% incidencias"]
        cell_text = []
        for idx, (nombre, horas) in enumerate(top_incidencias):
            porcentaje = horas / metrics.incidencias * 100 if metrics.incidencias else 0.0
            cell_text.append(
                [str(idx + 1), nombre, f"{horas:0.2f}", f"{porcentaje:0.1f}%"]
            )
        table = tabla_ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc="left",
            colWidths=[0.07, 0.55, 0.18, 0.2],
            bbox=[0.0, 0.0, 1.0, 0.78],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.15)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#D7DADB")
            if row == 0:
                cell.set_facecolor("#F4F4F4")
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor("white")
    else:
        tabla_ax.text(0.0, 0.5, "Sin incidencias registradas.", fontsize=11, color="#555555")

    fig.tight_layout()
    return fig


def render_report(
    metrics: DisponibilidadMetrics, output_dir: Path, logo_path: Optional[Path]
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / f"{metrics.resource_name}_disponibilidad"
    pdf_path = output_stem.with_suffix(".pdf")
    logo_image = cargar_logo(logo_path)

    fig1 = build_page_one(metrics, logo_image)
    fig1.savefig(pdf_path)
    plt.close(fig1)
    return pdf_path


def generar_informes_disponibilidad(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    logo_path: Optional[Path] = None,
) -> Iterable[Path]:
    data_path = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    base_output = Path(output_dir) if output_dir else DEFAULT_REPORT_DIR / "disponibilidad"
    recursos_dir = data_path / "recursos"
    busqueda = recursos_dir if recursos_dir.exists() else data_path
    csv_entries = listar_csv_por_seccion(data_path)
    if not csv_entries:
        raise FileNotFoundError(f"No se encontraron CSV en {busqueda}")

    resultados = []
    for seccion, csv_path in csv_entries:
        metrics = leer_metricas(csv_path)
        section_dir = base_output / seccion
        section_dir.mkdir(parents=True, exist_ok=True)
        recurso_dir = section_dir / metrics.resource_name.lower()
        pdf_file = render_report(metrics, recurso_dir, logo_path=logo_path)
        resultados.append(pdf_file)

    return resultados


if __name__ == "__main__":
    generar_informes_disponibilidad()
