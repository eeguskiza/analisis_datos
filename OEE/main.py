from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from OEE.disponibilidad.main import generar_informes_disponibilidad
from OEE.rendimiento.main import generar_informes_rendimiento
from OEE.calidad.main import generar_informes_calidad
from OEE.oee_secciones.main import (
    generar_informes_oee_secciones,
    leer_maquina,
    cargar_ciclos,
)
from OEE.utils.excel_import import procesar_excels
from OEE.utils.data_files import listar_csv_por_seccion


def imprimir_tabla_consolidada(data_dir: Path) -> None:
    """Imprime una tabla consolidada con las métricas OEE de todos los recursos."""
    ciclos = cargar_ciclos(data_dir)
    csv_entries = listar_csv_por_seccion(data_dir)

    if not csv_entries:
        return

    metricas_por_seccion: Dict[str, List[Tuple[str, float, float, float, float]]] = defaultdict(list)

    for seccion, csv_path in csv_entries:
        metrics = leer_maquina(csv_path, ciclos)
        recurso = metrics.name.upper()
        disp = metrics.disponibilidad_pct or 0.0
        rend = metrics.rendimiento_pct or 0.0
        cal = metrics.calidad_pct or 0.0
        oee = metrics.oee_pct or 0.0
        metricas_por_seccion[seccion].append((recurso, disp, rend, cal, oee))

    print("\n" + "=" * 80)
    print("RESUMEN OEE - TODAS LAS SECCIONES")
    print("=" * 80)

    for seccion in sorted(metricas_por_seccion.keys()):
        print(f"\n{seccion}:")
        print("-" * 80)
        print(f"{'Recurso':<12} {'Disponibilidad':>14} {'Rendimiento':>14} {'Calidad':>14} {'OEE':>10}")
        print("-" * 80)

        for recurso, disp, rend, cal, oee in sorted(metricas_por_seccion[seccion]):
            print(f"{recurso:<12} {disp:>13.2f}% {rend:>13.2f}% {cal:>13.2f}% {oee:>9.2f}%")

        recursos = metricas_por_seccion[seccion]
        avg_disp = sum(r[1] for r in recursos) / len(recursos)
        avg_rend = sum(r[2] for r in recursos) / len(recursos)
        avg_cal = sum(r[3] for r in recursos) / len(recursos)
        avg_oee = sum(r[4] for r in recursos) / len(recursos)

        print("-" * 80)
        print(f"{'PROMEDIO':<12} {avg_disp:>13.2f}% {avg_rend:>13.2f}% {avg_cal:>13.2f}% {avg_oee:>9.2f}%")

    print("\n" + "=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generador de informes OEE. Por defecto crea los informes de "
            "disponibilidad con los CSV presentes en ./data."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Ruta con los CSV de cada máquina (por defecto ./data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Ruta donde se guardarán los informes (por defecto ./OEE/informes).",
    )
    parser.add_argument(
        "--logo",
        type=Path,
        default=None,
        help="Ruta a un archivo de logo para incrustar en la cabecera del informe.",
    )
    parser.add_argument(
        "--seccion",
        dest="secciones",
        action="append",
        help="Nombre de una sección concreta para el informe maestro por secciones.",
    )
    parser.add_argument(
        "--modulo",
        choices=["todos", "disponibilidad", "rendimiento", "calidad", "oee_secciones"],
        default="todos",
        help="Permite ejecutar un módulo concreto.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    base_dir = Path(__file__).resolve().parents[1]
    logo_path = args.logo
    if logo_path is None:
        default_logo = base_dir / "data" / "ecs-logo.png"
        if default_logo.exists():
            logo_path = default_logo

    data_root = Path(data_dir) if data_dir else base_dir / "data"
    # Informes fuera de la carpeta OEE (paralelo a data)
    reports_base = Path(output_dir) if output_dir else base_dir / "informes"
    run_dir_name = datetime.now().strftime("%Y-%m-%d")
    run_dir = reports_base / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Paso previo: procesar Excel en data/excels -> data/csv y mover a data/recursos
    procesar_excels(data_root)

    if args.modulo in ("todos", "disponibilidad", "oee_secciones"):
        generar_informes_disponibilidad(
            data_dir=data_dir,
            output_dir=run_dir,
            logo_path=logo_path,
        )
    if args.modulo in ("todos", "rendimiento", "oee_secciones"):
        generar_informes_rendimiento(
            data_dir=data_dir,
            output_dir=run_dir,
            logo_path=logo_path,
        )
    if args.modulo in ("todos", "calidad", "oee_secciones"):
        generar_informes_calidad(
            data_dir=data_dir,
            output_dir=run_dir,
            logo_path=logo_path,
        )
    if args.modulo in ("todos", "oee_secciones"):
        generar_informes_oee_secciones(
            data_dir=data_dir,
            output_dir=run_dir,
            logo_path=logo_path,
            secciones=args.secciones,
        )

    # Imprimir tabla consolidada al final
    if args.modulo == "todos":
        imprimir_tabla_consolidada(data_root)


if __name__ == "__main__":
    main()
