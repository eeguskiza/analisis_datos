from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from OEE.disponibilidad.main import generar_informes_disponibilidad
from OEE.rendimiento.main import generar_informes_rendimiento
from OEE.calidad.main import generar_informes_calidad
from OEE.oee.main import generar_informes_oee
from OEE.oee_secciones.main import generar_informes_oee_secciones
from OEE.utils.excel_import import procesar_excels


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
        choices=["todos", "disponibilidad", "rendimiento", "calidad", "oee", "oee_secciones"],
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
    if args.modulo in ("todos", "oee"):
        generar_informes_oee(
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


if __name__ == "__main__":
    main()
