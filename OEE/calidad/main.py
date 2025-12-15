from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages

from OEE.utils.data_files import listar_csv_por_seccion

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
REPORT_DIR = Path(__file__).resolve().parents[1] / "informes" / "calidad"


def parse_float(value: str) -> float:
    if value is None or value == "":
        return 0.0
    value = value.replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return 0.0


def parse_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return None


def normalizar_proceso(value: str) -> str:
    """Normaliza el tipo de proceso para comparación consistente."""
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


def cargar_logo(logo_path: Optional[Path]):
    if not logo_path:
        return None
    try:
        return mpimg.imread(logo_path)
    except (FileNotFoundError, OSError):
        return None


def enmarcar(ax) -> None:
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


@dataclass
class ReferenciaCalidad:
    referencia: str
    piezas: float = 0.0
    malas: float = 0.0
    recuperadas: float = 0.0

    @property
    def scrap_final(self) -> float:
        return max(self.malas - self.recuperadas, 0.0)

    @property
    def scrap_pct(self) -> Optional[float]:
        if self.piezas <= 0:
            return None
        return (self.scrap_final / self.piezas) * 100


@dataclass
class CalidadMetrics:
    resource: str
    start: Optional[datetime]
    end: Optional[datetime]
    piezas_totales: float
    piezas_malas: float
    piezas_recuperadas: float
    scrap_final: float
    buenas_finales: float
    calidad_oee: Optional[float]
    scrap_bruto_pct: Optional[float]
    scrap_final_pct: Optional[float]
    tasa_recuperacion: Optional[float]
    fpy_pct: Optional[float]
    top_referencias: List[ReferenciaCalidad] = field(default_factory=list)
    daily_stats: Dict[datetime, Dict[str, float]] = field(default_factory=dict)
    daily_references: Dict[datetime, Dict[str, ReferenciaCalidad]] = field(default_factory=dict)


def leer_calidad(csv_path: Path) -> CalidadMetrics:
    piezas_totales = 0.0
    piezas_malas = 0.0
    piezas_recuperadas = 0.0
    start = None
    end = None
    referencias: Dict[str, ReferenciaCalidad] = {}

    # Para estadísticas por día
    from collections import defaultdict
    daily_stats: Dict[datetime, Dict[str, float]] = defaultdict(
        lambda: {"piezas_totales": 0.0, "piezas_malas": 0.0, "piezas_recuperadas": 0.0}
    )
    daily_references: Dict[datetime, Dict[str, ReferenciaCalidad]] = defaultdict(dict)

    with csv_path.open(encoding="utf-8-sig", newline="") as handler:
        reader = csv.DictReader(handler)
        if reader.fieldnames:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            proceso = normalizar_proceso(row.get("Proceso", ""))
            if proceso != "produccion":
                continue

            piezas = parse_float(row.get("Cantidad", "0"))
            malas = parse_float(row.get("Malas", "0"))
            recuperadas = parse_float(row.get("Recu.", row.get("Recu", "0")))
            fecha = parse_datetime(row.get("Fecha", ""))

            if fecha:
                start = fecha if not start or fecha < start else start
                end = fecha if not end or fecha > end else end

            piezas_totales += piezas
            piezas_malas += malas
            piezas_recuperadas += recuperadas

            # Acumular por día
            if fecha:
                day_key = fecha.date()
                day_data = daily_stats[day_key]
                day_data["piezas_totales"] += piezas
                day_data["piezas_malas"] += malas
                day_data["piezas_recuperadas"] += recuperadas

            ref_key = (row.get("Refer.") or "").strip() or "Sin referencia"
            ref = referencias.setdefault(ref_key, ReferenciaCalidad(ref_key))
            ref.piezas += piezas
            ref.malas += malas
            ref.recuperadas += recuperadas

            # Referencias por día
            if fecha:
                day_ref = daily_references[day_key].setdefault(ref_key, ReferenciaCalidad(ref_key))
                day_ref.piezas += piezas
                day_ref.malas += malas
                day_ref.recuperadas += recuperadas

    scrap_final = max(piezas_malas - piezas_recuperadas, 0.0)
    buenas_finales = piezas_totales - scrap_final

    def pct(value: float, base: float) -> Optional[float]:
        if base <= 0:
            return None
        return value / base * 100

    calidad_oee = pct(buenas_finales, piezas_totales)
    scrap_bruto = pct(piezas_malas, piezas_totales)
    scrap_final_pct = pct(scrap_final, piezas_totales)
    tasa_recuperacion = pct(piezas_recuperadas, piezas_malas)
    fpy_pct = pct(piezas_totales - piezas_malas, piezas_totales)

    top_refs = sorted(
        referencias.values(),
        key=lambda ref: ref.scrap_final,
        reverse=True,
    )[:10]

    return CalidadMetrics(
        resource=csv_path.stem.split("-")[0],
        start=start,
        end=end,
        piezas_totales=piezas_totales,
        piezas_malas=piezas_malas,
        piezas_recuperadas=piezas_recuperadas,
        scrap_final=scrap_final,
        buenas_finales=buenas_finales,
        calidad_oee=calidad_oee,
        scrap_bruto_pct=scrap_bruto,
        scrap_final_pct=scrap_final_pct,
        tasa_recuperacion=tasa_recuperacion,
        fpy_pct=fpy_pct,
        top_referencias=top_refs,
        daily_stats=dict(daily_stats),
        daily_references=dict(daily_references),
    )


def format_pct(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f} %"


def render_calidad_day(
    resource_name: str,
    day: datetime,
    day_stats: Dict[str, float],
    day_references: Dict[str, ReferenciaCalidad],
    logo_image
) -> plt.Figure:
    """Genera una página de calidad para un día específico."""
    piezas_totales = day_stats.get("piezas_totales", 0.0)
    piezas_malas = day_stats.get("piezas_malas", 0.0)
    piezas_recuperadas = day_stats.get("piezas_recuperadas", 0.0)
    scrap_final = max(piezas_malas - piezas_recuperadas, 0.0)
    buenas_finales = max(piezas_totales - scrap_final, 0.0)

    def pct(value: float, base: float) -> Optional[float]:
        if base <= 0:
            return None
        return value / base * 100

    calidad_oee = pct(buenas_finales, piezas_totales)
    scrap_bruto = pct(piezas_malas, piezas_totales)
    scrap_final_pct = pct(scrap_final, piezas_totales)
    tasa_recuperacion = pct(piezas_recuperadas, piezas_malas)
    fpy_pct = pct(piezas_totales - piezas_malas, piezas_totales)

    fig = plt.figure(figsize=(8.27, 11.69))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.55, 0.9, 0.5, 1.2])

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

    header_ax.text(
        0.4,
        0.55,
        "Informe Calidad",
        fontsize=22,
        fontweight="bold",
        color="#000000",
    )

    header_ax.text(
        0.4,
        0.30,
        f"Recurso: {resource_name.upper()}",
        fontsize=11,
        color="#424242",
    )
    header_ax.text(
        0.4,
        0.08,
        f"Día: {day.strftime('%d/%m/%Y')}",
        fontsize=11,
        fontweight="bold",
        color="#263238",
    )

    resumen_ax = fig.add_subplot(gs[1, 0])
    resumen_ax.axis("off")
    enmarcar(resumen_ax)
    resumen_ax.text(
        0.0,
        1.05,
        f"Calidad del día: {format_pct(calidad_oee, 2)}",
        fontsize=18,
        fontweight="bold",
        color="#000000",
    )
    resumen_ax.text(
        0.0, 0.82, "Resumen ejecutivo de calidad", fontsize=11, fontweight="bold", color="#263238"
    )
    left_lines = [
        f"Piezas producidas: {piezas_totales:0.0f}",
        f"Piezas malas: {piezas_malas:0.0f}",
        f"Piezas recuperadas: {piezas_recuperadas:0.0f}",
        f"Scrap final (pzas): {scrap_final:0.0f}",
    ]
    right_lines = [
        f"Scrap bruto: {format_pct(scrap_bruto, 2)}",
        f"Scrap final: {format_pct(scrap_final_pct, 2)}",
        f"Tasa de recuperación: {format_pct(tasa_recuperacion, 2)}",
        f"FPY: {format_pct(fpy_pct, 2)}",
    ]
    start_left = 0.62
    step = 0.14
    for idx, text in enumerate(left_lines):
        resumen_ax.text(0.02, start_left - idx * step, text, fontsize=10)
    start_right = 0.62
    for idx, text in enumerate(right_lines):
        resumen_ax.text(0.55, start_right - idx * step, text, fontsize=10)

    graf_ax = fig.add_subplot(gs[2, 0])
    enmarcar(graf_ax)
    graf_ax.set_title("Distribución de piezas", loc="left", fontsize=11, color="#263238")

    totales = max(piezas_totales, 0.01)
    valores = [
        ("Buenas finales", buenas_finales, "#2E7D32"),
        ("Recuperadas", piezas_recuperadas, "#F9A825"),
        ("Scrap final", scrap_final, "#C62828"),
    ]
    left = 0.0
    for label, value, color in valores:
        graf_ax.barh(["Piezas"], [value], left=left, color=color, label=f"{label} ({value:0.0f})")
        left += value
    graf_ax.set_xlim(0, totales)
    graf_ax.set_xlabel("Piezas")
    graf_ax.set_yticks([])
    graf_ax.grid(axis="x", linestyle="--", alpha=0.2)
    graf_ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.3), frameon=False, fontsize=9)

    tabla_ax = fig.add_subplot(gs[3, 0])
    enmarcar(tabla_ax)
    tabla_ax.axis("off")
    tabla_ax.text(
        0.0,
        0.95,
        "Referencias del día por scrap final",
        fontsize=11,
        fontweight="bold",
        color="#263238",
    )
    inset = tabla_ax.inset_axes([0.0, 0.1, 1.0, 0.8])
    inset.axis("off")

    refs_list = sorted(day_references.values(), key=lambda ref: ref.scrap_final, reverse=True)[:10]
    if refs_list:
        col_labels = ["#", "Referencia", "Piezas", "Malas", "Recuperadas", "Scrap final", "Scrap final (%)"]
        rows = []
        for idx, ref in enumerate(refs_list, start=1):
            scrap_pct = f"{ref.scrap_pct:0.2f}%" if ref.scrap_pct is not None else "N/A"
            rows.append(
                [
                    str(idx),
                    ref.referencia,
                    f"{ref.piezas:0.0f}",
                    f"{ref.malas:0.0f}",
                    f"{ref.recuperadas:0.0f}",
                    f"{ref.scrap_final:0.0f}",
                    scrap_pct,
                ]
            )
        table = inset.table(
            cellText=rows,
            colLabels=col_labels,
            loc="upper left",
            colWidths=[0.05, 0.25, 0.12, 0.12, 0.12, 0.12, 0.15],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#D7DADB")
            if row == 0:
                cell.set_facecolor("#F4F4F4")
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor("white")
    else:
        inset.text(0.0, 0.4, "No hay referencias con scrap registrado.", fontsize=9)

    fig.tight_layout()
    return fig


def render_calidad(metrics: CalidadMetrics, logo_image) -> plt.Figure:
    fig = plt.figure(figsize=(8.27, 11.69))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.55, 0.9, 0.5, 1.2])

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

    header_ax.text(
        0.4,
        0.55,
        "Informe Calidad",
        fontsize=22,
        fontweight="bold",
        color="#000000",
    )
    periodo = "Sin datos"
    if metrics.start and metrics.end:
        periodo = f"{metrics.start.strftime('%d/%m/%Y')} → {metrics.end.strftime('%d/%m/%Y')}"

    header_ax.text(
        0.4,
        0.30,
        f"Recurso: {metrics.resource.upper()}",
        fontsize=11,
        color="#424242",
    )
    header_ax.text(
        0.4,
        0.08,
        f"Periodo: {periodo}",
        fontsize=11,
        fontweight="bold",
        color="#263238",
    )

    resumen_ax = fig.add_subplot(gs[1, 0])
    resumen_ax.axis("off")
    enmarcar(resumen_ax)
    resumen_ax.text(
        0.0,
        1.05,
        f"Calidad del periodo: {format_pct(metrics.calidad_oee, 2)}",
        fontsize=18,
        fontweight="bold",
        color="#000000",
    )
    resumen_ax.text(
        0.0, 0.82, "Resumen ejecutivo de calidad", fontsize=11, fontweight="bold", color="#263238"
    )
    left_lines = [
        f"Piezas producidas: {metrics.piezas_totales:0.0f}",
        f"Piezas malas: {metrics.piezas_malas:0.0f}",
        f"Piezas recuperadas: {metrics.piezas_recuperadas:0.0f}",
        f"Scrap final (pzas): {metrics.scrap_final:0.0f}",
    ]
    right_lines = [
        f"Scrap bruto: {format_pct(metrics.scrap_bruto_pct, 2)}",
        f"Scrap final: {format_pct(metrics.scrap_final_pct, 2)}",
        f"Tasa de recuperación: {format_pct(metrics.tasa_recuperacion, 2)}",
        f"FPY: {format_pct(metrics.fpy_pct, 2)}",
    ]
    start_left = 0.62
    step = 0.14
    for idx, text in enumerate(left_lines):
        resumen_ax.text(0.02, start_left - idx * step, text, fontsize=10)
    start_right = 0.62
    for idx, text in enumerate(right_lines):
        resumen_ax.text(0.55, start_right - idx * step, text, fontsize=10)

    graf_ax = fig.add_subplot(gs[2, 0])
    enmarcar(graf_ax)
    graf_ax.set_title("Distribución de piezas", loc="left", fontsize=11, color="#263238")

    totales = max(metrics.piezas_totales, 0.01)
    valores = [
        ("Buenas finales", metrics.buenas_finales, "#2E7D32"),
        ("Recuperadas", metrics.piezas_recuperadas, "#F9A825"),
        ("Scrap final", metrics.scrap_final, "#C62828"),
    ]
    left = 0.0
    for label, value, color in valores:
        graf_ax.barh(["Piezas"], [value], left=left, color=color, label=f"{label} ({value:0.0f})")
        left += value
    graf_ax.set_xlim(0, totales)
    graf_ax.set_xlabel("Piezas")
    graf_ax.set_yticks([])
    graf_ax.grid(axis="x", linestyle="--", alpha=0.2)
    graf_ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.3), frameon=False, fontsize=9)

    tabla_ax = fig.add_subplot(gs[3, 0])
    enmarcar(tabla_ax)
    tabla_ax.axis("off")
    tabla_ax.text(
        0.0,
        0.95,
        "Top referencias por scrap final",
        fontsize=11,
        fontweight="bold",
        color="#263238",
    )
    inset = tabla_ax.inset_axes([0.0, 0.1, 1.0, 0.8])
    inset.axis("off")
    if metrics.top_referencias:
        col_labels = ["#", "Referencia", "Piezas", "Malas", "Recuperadas", "Scrap final", "Scrap final (%)"]
        rows = []
        for idx, ref in enumerate(metrics.top_referencias, start=1):
            scrap_pct = f"{ref.scrap_pct:0.2f}%" if ref.scrap_pct is not None else "N/A"
            rows.append(
                [
                    str(idx),
                    ref.referencia,
                    f"{ref.piezas:0.0f}",
                    f"{ref.malas:0.0f}",
                    f"{ref.recuperadas:0.0f}",
                    f"{ref.scrap_final:0.0f}",
                    scrap_pct,
                ]
            )
        table = inset.table(
            cellText=rows,
            colLabels=col_labels,
            loc="upper left",
            colWidths=[0.05, 0.25, 0.12, 0.12, 0.12, 0.12, 0.15],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#D7DADB")
            if row == 0:
                cell.set_facecolor("#F4F4F4")
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor("white")
    else:
        inset.text(0.0, 0.4, "No hay referencias con scrap registrado.", fontsize=9)

    fig.tight_layout()
    return fig


def generar_informes_calidad(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    logo_path: Optional[Path] = None,
) -> Iterable[Path]:
    data_path = Path(data_dir) if data_dir else DATA_DIR
    base_output = Path(output_dir) if output_dir else REPORT_DIR

    resultados = []
    for seccion, csv_file in listar_csv_por_seccion(data_path):
        metrics = leer_calidad(csv_file)
        logo = cargar_logo(logo_path)
        section_dir = base_output / seccion
        section_dir.mkdir(parents=True, exist_ok=True)
        recurso_dir = section_dir / metrics.resource.lower()
        recurso_dir.mkdir(parents=True, exist_ok=True)
        output = recurso_dir / f"{metrics.resource}_calidad.pdf"

        # Si no hay datos por día, generar una sola página
        if not metrics.daily_stats:
            fig = render_calidad(metrics, logo)
            fig.savefig(output)
            plt.close(fig)
        else:
            # Generar una página por cada día
            with PdfPages(output) as pdf:
                for day in sorted(metrics.daily_stats.keys()):
                    day_stats = metrics.daily_stats[day]
                    day_references = metrics.daily_references.get(day, {})
                    day_dt = datetime.combine(day, datetime.min.time())
                    fig = render_calidad_day(
                        metrics.resource,
                        day_dt,
                        day_stats,
                        day_references,
                        logo
                    )
                    pdf.savefig(fig)
                    plt.close(fig)

        resultados.append(output)
    return resultados


if __name__ == "__main__":
    generar_informes_calidad()
