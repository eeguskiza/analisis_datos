from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

RESOURCE_SECTION_MAP: Dict[str, str] = {
    "luk1": "LINEAS",
    "luk2": "LINEAS",
    "luk3": "LINEAS",
    "luk6": "LINEAS",
    "coroa": "LINEAS",
    "vw1": "LINEAS",
    "omr": "LINEAS",
    "t48": "TALLADORAS",
}


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))[:60]


def convert_excel_to_csv(
    excel_dir: Path,
    csv_out_dir: Path,
    sheet: Optional[str | int] = None,
    export_all_sheets: bool = False,
    sep: str = ",",
    encoding: str = "utf-8-sig",
    decimal: str = ".",
) -> int:
    """Convierte todos los Excel de excel_dir a CSV en csv_out_dir. Devuelve nº de CSV generados."""
    excel_dir = excel_dir.expanduser()
    csv_out_dir = csv_out_dir.expanduser()
    csv_out_dir.mkdir(parents=True, exist_ok=True)
    files = [*excel_dir.glob("*.xlsx"), *excel_dir.glob("*.xls"), *excel_dir.glob("*.xlsm")]
    total = 0
    for f in files:
        try:
            if export_all_sheets:
                xls = pd.ExcelFile(f)
                for s in xls.sheet_names:
                    df = pd.read_excel(f, sheet_name=s, dtype=object, engine="openpyxl")
                    out_path = csv_out_dir / f"{f.stem}__{sanitize(s)}.csv"
                    df.to_csv(out_path, index=False, sep=sep, encoding=encoding, decimal=decimal)
                    total += 1
            else:
                sn = 0 if sheet is None else sheet
                df = pd.read_excel(f, sheet_name=sn, dtype=object, engine="openpyxl")
                out_path = csv_out_dir / f"{f.stem}.csv"
                df.to_csv(out_path, index=False, sep=sep, encoding=encoding, decimal=decimal)
                total += 1
            # elimina el Excel original si todo fue bien
            f.unlink(missing_ok=True)
        except Exception as e:
            print(f"[Excel->CSV] Error en {f.name}: {e}")
    return total


def distribuir_csv_en_recursos(
    csv_dir: Path,
    recursos_dir: Path,
    section_map: Optional[Dict[str, str]] = None,
    extra_sources: Optional[Iterable[Path]] = None,
) -> int:
    """
    Mueve los CSV a data/recursos/<SECCION>/ usando un mapa recurso->sección.

    Se admiten rutas adicionales (extra_sources) para consolidar CSV previos.
    """
    section_map = section_map or RESOURCE_SECTION_MAP
    moved = 0
    sources = [csv_dir]
    if extra_sources:
        sources.extend(extra_sources)
    for src in sources:
        src = src.expanduser()
        for csv_path in src.glob("*.csv"):
            recurso = csv_path.stem.split("-")[0].split("__")[0].lower()
            seccion = section_map.get(recurso, "GENERAL")
            destino_dir = recursos_dir / seccion
            destino_dir.mkdir(parents=True, exist_ok=True)
            destino_path = destino_dir / csv_path.name
            destino_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.replace(destino_path)
            moved += 1
    return moved


def procesar_excels(
    base_data_dir: Path,
    excel_subdir: str = "excels",
    csv_subdir: str = "csv",
    section_map: Optional[Dict[str, str]] = None,
) -> None:
    excel_dir = base_data_dir / excel_subdir
    csv_out_dir = base_data_dir / csv_subdir
    recursos_dir = base_data_dir / "recursos"
    if not excel_dir.exists():
        return
    generated = convert_excel_to_csv(excel_dir, csv_out_dir)
    moved = distribuir_csv_en_recursos(csv_out_dir, recursos_dir, section_map, extra_sources=[csv_out_dir])
    if generated > 0 or moved > 0:
        print(f"[Excel->CSV] Generados {generated} CSV y movidos {moved} a {recursos_dir}")
