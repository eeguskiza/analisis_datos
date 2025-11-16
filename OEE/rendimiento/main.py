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


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CYCLES_FILE = DATA_DIR / "ciclos.csv"
REPORT_DIR = Path(__file__).resolve().parents[1] / "informes" / "rendimiento"


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


def normalize_ref(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.lower()


def cargar_logo(logo_path: Optional[Path]):
    if not logo_path:
        return None
    try:
        return mpimg.imread(logo_path)
    except (FileNotFoundError, OSError):
        return None


def enmarcar(ax):
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
class ReferenciaRendimiento:
    referencia: str
    piezas: float
    horas: float
    ideal: float

    @property
    def perdidas(self) -> float:
        return max(self.horas - self.ideal, 0.0)

    @property
    def rendimiento_pct(self) -> Optional[float]:
        if self.horas <= 0 or self.ideal <= 0:
            return None
        return (self.ideal / self.horas) * 100


@dataclass
class RendimientoMetrics:
    resource: str
    start: Optional[datetime]
    end: Optional[datetime]
    horas_produccion: float
    piezas_totales: float
    tiempo_ideal_horas: float
    ciclo_ideal_medio: Optional[float]
    perdidas_velocidad: float
    rendimiento_pct: Optional[float]
    top_referencias: List[ReferenciaRendimiento] = field(default_factory=list)
    registros_sin_ciclo: int = 0


def cargar_tiempos_ciclo() -> Dict[str, Dict[str, float]]:
    ciclos: Dict[str, Dict[str, float]] = {}
    if not CYCLES_FILE.exists():
        print(f"[Rendimiento] Aviso: no se encontró {CYCLES_FILE}, se omitirá el cálculo ideal.")
        return ciclos

    with CYCLES_FILE.open(encoding="utf-8-sig") as handler:
        reader = csv.DictReader(handler)
        for row in reader:
            maquina = (row.get("maquina") or "").strip().lower()
            referencia = normalize_ref(row.get("referencia") or "")
            tiempo = parse_float(row.get("tiempo_ciclo", "0"))
            if not maquina or not referencia or tiempo <= 0:
                continue
            ciclos.setdefault(maquina, {})[referencia] = tiempo
    return ciclos


def leer_rendimiento(csv_path: Path, ciclos_maquina: Dict[str, float]) -> RendimientoMetrics:
    horas_produccion = 0.0
    piezas_totales = 0.0
    tiempo_ideal_horas = 0.0
    start = None
    end = None
    total_rate_weighted = 0.0
    total_piezas_ciclo = 0.0
    registros_sin_ciclo = 0
    referencias: Dict[str, ReferenciaRendimiento] = {}

    with csv_path.open(encoding="utf-8-sig", newline="") as handler:
        reader = csv.DictReader(handler)
        for row in reader:
            proceso = (row.get("Proceso") or "").strip().lower()
            if proceso != "producción":
                continue
            horas = parse_float(row.get("Tiempo", "0"))
            piezas = parse_float(row.get("Cantidad", "0"))
            ref_norm = normalize_ref(row.get("Refer.") or "")
            fecha = parse_datetime(row.get("Fecha", ""))

            if fecha:
                start = fecha if not start or fecha < start else start
                end = fecha if not end or fecha > end else end

            horas_produccion += horas
            piezas_totales += piezas

            rate_pph = ciclos_maquina.get(ref_norm)
            if not rate_pph:
                registros_sin_ciclo += 1
                continue

            ideal_horas = piezas / rate_pph
            tiempo_ideal_horas += ideal_horas
            total_rate_weighted += rate_pph * piezas
            total_piezas_ciclo += piezas

            ref_stat = referencias.setdefault(
                ref_norm,
                ReferenciaRendimiento(referencia=row.get("Refer.") or ref_norm, piezas=0.0, horas=0.0, ideal=0.0),
            )
            ref_stat.piezas += piezas
            ref_stat.horas += horas
            ref_stat.ideal += ideal_horas

    perdidas = max(horas_produccion - tiempo_ideal_horas, 0.0)
    rendimiento_pct = None
    if horas_produccion > 0 and tiempo_ideal_horas > 0:
        rendimiento_pct = (tiempo_ideal_horas / horas_produccion) * 100
    ciclo_ideal = None
    if total_piezas_ciclo > 0:
        ciclo_ideal = total_rate_weighted / total_piezas_ciclo

    top_refs = [
        ref
        for ref in sorted(referencias.values(), key=lambda ref: ref.perdidas, reverse=True)
        if ref.perdidas > 0
    ]

    return RendimientoMetrics(
        resource=csv_path.stem.split("-")[0],
        start=start,
        end=end,
        horas_produccion=horas_produccion,
        piezas_totales=piezas_totales,
        tiempo_ideal_horas=tiempo_ideal_horas,
        ciclo_ideal_medio=ciclo_ideal,
        perdidas_velocidad=perdidas,
        rendimiento_pct=rendimiento_pct,
        top_referencias=top_refs,
        registros_sin_ciclo=registros_sin_ciclo,
    )


def render_rendimiento(metrics: RendimientoMetrics, logo_image) -> plt.Figure:
    fig = plt.figure(figsize=(8.27, 11.69))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.55, 0.85, 0.25, 1.35])

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
        "Informe Rendimiento",
        fontsize=22,
        fontweight="bold",
        color="#000000",
    )
    header_ax.text(
        0.4,
        0.24,
        f"Recurso: {metrics.resource.upper()}",
        fontsize=11,
        color="#424242",
    )

    periodo = "Sin datos"
    if metrics.start and metrics.end:
        periodo = f"{metrics.start.date().isoformat()} → {metrics.end.date().isoformat()}"

    fig.text(
        0.03,
        0.82,
        f"Rendimiento del periodo: {metrics.rendimiento_pct or 0:0.2f} %",
        fontsize=18,
        fontweight="bold",
        color="#000000",
    )

    resumen_ax = fig.add_subplot(gs[1, 0])
    resumen_ax.axis("off")
    enmarcar(resumen_ax)
    resumen_ax.text(
        0.0, 0.9, "Resumen ejecutivo de rendimiento", fontsize=11, fontweight="bold", color="#263238"
    )
    left_lines = [
        f"Período: {periodo}",
        f"Horas de producción: {metrics.horas_produccion:0.2f} h",
        f"Piezas producidas: {metrics.piezas_totales:0.0f}",
        (
            f"Ciclo ideal medio: "
            f"{metrics.ciclo_ideal_medio:0.1f} piezas/h"
            if metrics.ciclo_ideal_medio
            else "Ciclo ideal medio: N/A"
        ),
    ]
    right_lines = [
        f"Tiempo ideal: {metrics.tiempo_ideal_horas:0.2f} h",
        f"Pérdidas de velocidad: {metrics.perdidas_velocidad:0.2f} h",
        f"Registros sin ciclo ideal: {metrics.registros_sin_ciclo}",
    ]
    start_left = 0.65
    step = 0.13
    for idx, text in enumerate(left_lines):
        resumen_ax.text(0.02, start_left - idx * step, text, fontsize=10)
    start_right = 0.65
    for idx, text in enumerate(right_lines):
        resumen_ax.text(0.55, start_right - idx * step, text, fontsize=10)

    notas_ax = fig.add_subplot(gs[2, 0])
    notas_ax.axis("off")
    notas = [
        "Tiempo ideal: horas necesarias si todas las órdenes respetan las piezas/hora objetivo.",
        "Pérdidas de velocidad: horas adicionales respecto al ideal. Registros sin ciclo: órdenes sin dato ideal (no suman al ideal).",
        "Ciclo ideal medio: media ponderada de piezas/hora procedente del fichero de ciclos.",
    ]
    y = 0.8
    for nota in notas:
        notas_ax.text(0.02, y, nota, fontsize=8.5, color="#424242")
        y -= 0.28

    referencias_ax = fig.add_subplot(gs[3, 0])
    enmarcar(referencias_ax)
    referencias_ax.axis("off")
    referencias_ax.text(
        0.0,
        0.96,
        "Referencias analizadas ordenadas por pérdidas",
        fontsize=11,
        fontweight="bold",
        color="#263238",
    )
    tabla_ax = referencias_ax.inset_axes([0.0, 0.07, 1.0, 0.85])
    tabla_ax.axis("off")
    if metrics.top_referencias:
        col_labels = ["Referencia", "Piezas", "Ciclo ideal (pzas/h)", "Ciclo real (pzas/h)", "Rendimiento", "Pérdidas (h)"]
        rows = []
        for ref in sorted(metrics.top_referencias, key=lambda r: r.perdidas, reverse=True):
            if ref.piezas > 0 and ref.ideal > 0:
                ciclo_ideal = f"{ref.piezas / ref.ideal:0.1f}"
            else:
                ciclo_ideal = "N/A"
            if ref.piezas > 0 and ref.horas > 0:
                ciclo_real = f"{ref.piezas / ref.horas:0.1f}"
            else:
                ciclo_real = "N/A"
            rendimiento_ref = (
                f"{ref.rendimiento_pct:0.1f}%" if ref.rendimiento_pct is not None else "N/A"
            )
            rows.append(
                [
                    ref.referencia,
                    f"{ref.piezas:0.0f}",
                    ciclo_ideal,
                    ciclo_real,
                    rendimiento_ref,
                    f"{ref.perdidas:0.2f}",
                ]
            )
        table = tabla_ax.table(
            cellText=rows,
            colLabels=col_labels,
            loc="upper left",
            colWidths=[0.18, 0.16, 0.2, 0.18, 0.14, 0.14],
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
        tabla_ax.text(0.0, 0.4, "No hay referencias con tiempo de ciclo definido.", fontsize=9)

    fig.tight_layout()
    return fig


def generar_informes_rendimiento(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    logo_path: Optional[Path] = None,
) -> Iterable[Path]:
    data_path = Path(data_dir) if data_dir else DATA_DIR
    report_path = Path(output_dir) if output_dir else REPORT_DIR
    report_path.mkdir(parents=True, exist_ok=True)

    ciclos = cargar_tiempos_ciclo()

    resultados = []
    for csv_file in sorted(data_path.glob("*.csv")):
        recurso = csv_file.stem.split("-")[0].lower()
        ciclos_maquina = ciclos.get(recurso, {})
        metrics = leer_rendimiento(csv_file, ciclos_maquina)
        logo = cargar_logo(logo_path)
        fig = render_rendimiento(metrics, logo)
        output = report_path / f"{metrics.resource}_rendimiento.pdf"
        fig.savefig(output)
        plt.close(fig)
        print(f"[Rendimiento] {metrics.resource}: {metrics.rendimiento_pct or 0:0.2f}% (PDF: {output})")
        resultados.append(output)
    return resultados


if __name__ == "__main__":
    generar_informes_rendimiento()
