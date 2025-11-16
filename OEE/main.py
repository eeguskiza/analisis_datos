from __future__ import annotations

import argparse
from pathlib import Path

from OEE.disponibilidad.main import generar_informes_disponibilidad
from OEE.rendimiento.main import generar_informes_rendimiento
from OEE.calidad.main import generar_informes_calidad
from OEE.oee.main import generar_informes_oee


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
        "--modulo",
        choices=["todos", "disponibilidad", "rendimiento", "calidad", "oee"],
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

    if args.modulo in ("todos", "disponibilidad"):
        generar_informes_disponibilidad(
            data_dir=data_dir,
            output_dir=output_dir,
            logo_path=logo_path,
        )
    if args.modulo in ("todos", "rendimiento"):
        generar_informes_rendimiento(
            data_dir=data_dir,
            output_dir=None,
            logo_path=logo_path,
        )
    if args.modulo in ("todos", "calidad"):
        generar_informes_calidad(
            data_dir=data_dir,
            output_dir=None,
            logo_path=logo_path,
        )
    if args.modulo in ("todos", "oee"):
        generar_informes_oee(
            data_dir=data_dir,
            output_dir=None,
            logo_path=logo_path,
        )


if __name__ == "__main__":
    main()
