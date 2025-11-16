from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CYCLES_FILE = DATA_DIR / "ciclos.csv"
REPORT_DIR = Path(__file__).resolve().parents[1] / "informes" / "oee"


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
class OeeMetrics:
    resource: str
    start: Optional[datetime]
    end: Optional[datetime]
    disponibilidad: Optional[float]
    rendimiento: Optional[float]
    calidad: Optional[float]

    @property
    def oee(self) -> Optional[float]:
        if (
            self.disponibilidad is None
            or self.rendimiento is None
            or self.calidad is None
        ):
            return None
        return (self.disponibilidad * self.rendimiento * self.calidad) / 10000.0


def cargar_ciclos() -> dict[str, dict[str, float]]:
    ciclos = {}
    if not CYCLES_FILE.exists():
        return ciclos
    with CYCLES_FILE.open(encoding="utf-8-sig") as handler:
        reader = csv.DictReader(handler)
        for row in reader:
            maquina = (row.get("maquina") or "").strip().lower()
            referencia = (row.get("referencia") or "").strip().lower().rstrip(".0")
            rate = parse_float(row.get("tiempo_ciclo", "0"))
            if maquina and referencia and rate > 0:
                ciclos.setdefault(maquina, {})[referencia] = rate
    return ciclos


def leer_oee(csv_path: Path, ciclos: dict[str, dict[str, float]]) -> OeeMetrics:
    horas_produccion = 0.0
    horas_preparacion = 0.0
    horas_incidencias = 0.0
    tiempo_ideal_horas = 0.0
    total_rate = 0.0
    total_piezas_rate = 0.0
    piezas_totales = 0.0
    piezas_malas = 0.0
    piezas_recuperadas = 0.0
    start = None
    end = None
    recurso = csv_path.stem.split("-")[0].lower()
    ciclos_maquina = ciclos.get(recurso, {})

    with csv_path.open(encoding="utf-8-sig", newline="") as handler:
        reader = csv.DictReader(handler)
        for row in reader:
            proceso = (row.get("Proceso") or "").strip().lower()
            horas = parse_float(row.get("Tiempo", "0"))
            piezas = parse_float(row.get("Cantidad", "0"))
            malas = parse_float(row.get("Malas", "0"))
            recu = parse_float(row.get("Recu.", row.get("Recu", "0")))
            fecha = parse_datetime(row.get("Fecha", ""))

            if fecha:
                start = fecha if not start or fecha < start else start
                end = fecha if not end or fecha > end else end

            if proceso == "producción":
                horas_produccion += horas
                piezas_totales += piezas
                piezas_malas += malas
                piezas_recuperadas += recu

                ref = (row.get("Refer.") or "").strip().lower().rstrip(".0")
                rate = ciclos_maquina.get(ref)
                if rate:
                    ideal = piezas / rate
                    tiempo_ideal_horas += ideal
                    total_rate += rate * piezas
                    total_piezas_rate += piezas
            elif proceso == "preparación":
                horas_preparacion += horas
            else:
                horas_incidencias += horas

    horas_brutas = horas_produccion + horas_preparacion + horas_incidencias
    disp = (horas_produccion / horas_brutas * 100) if horas_brutas > 0 else None
    rend = (
        tiempo_ideal_horas / horas_produccion * 100
        if horas_produccion > 0 and tiempo_ideal_horas > 0
        else None
    )
    scrap_final = max(piezas_malas - piezas_recuperadas, 0.0)
    buenas = piezas_totales - scrap_final
    cal = (buenas / piezas_totales * 100) if piezas_totales > 0 else None

    return OeeMetrics(
        resource=recurso,
        start=start,
        end=end,
        disponibilidad=disp,
        rendimiento=rend,
        calidad=cal,
    )


def render_oee(resumen: list[OeeMetrics], logo_image) -> plt.Figure:
    fig = plt.figure(figsize=(8.27, 11.69))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.4, 1.4])

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
        "Informe OEE",
        fontsize=22,
        fontweight="bold",
        color="#000000",
    )
    header_ax.text(
        0.4,
        0.24,
        "Resumen por recurso",
        fontsize=11,
        color="#424242",
    )

    tabla_ax = fig.add_subplot(gs[1, 0])
    tabla_ax.axis("off")
    tabla_ax.text(
        0.0,
        0.95,
        "OEE y factores por recurso",
        fontsize=11,
        fontweight="bold",
        color="#263238",
    )
    y = 0.8
    for metric in resumen:
        disp = f"{metric.disponibilidad:.2f}%" if metric.disponibilidad is not None else "N/A%"
        rend = f"{metric.rendimiento:.2f}%" if metric.rendimiento is not None else "N/A%"
        cal = f"{metric.calidad:.2f}%" if metric.calidad is not None else "N/A%"
        oee = f"{metric.oee:.2f}%" if metric.oee is not None else "N/A"
        tabla_ax.text(
            0.0,
            y,
            f"{metric.resource.upper()} · Disp: {disp} | Rend: {rend} | Cal: {cal} | OEE: {oee}",
            fontsize=10,
            color="#424242",
        )
        y -= 0.08

    fig.tight_layout()
    return fig


def generar_informes_oee(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    logo_path: Optional[Path] = None,
) -> Iterable[Path]:
    data_path = Path(data_dir) if data_dir else DATA_DIR
    report_path = Path(output_dir) if output_dir else REPORT_DIR
    report_path.mkdir(parents=True, exist_ok=True)

    ciclos = cargar_ciclos()
    metrics_list = []
    for csv_file in sorted(data_path.glob("*.csv")):
        metrics = leer_oee(csv_file, ciclos)
        metrics_list.append(metrics)

    logo = cargar_logo(logo_path)
    fig = render_oee(metrics_list, logo)
    output = report_path / "resumen_oee.pdf"
    fig.savefig(output)
    plt.close(fig)
    print(f"[OEE] Resumen generado en {output}")
    return [output]


if __name__ == "__main__":
    generar_informes_oee()
