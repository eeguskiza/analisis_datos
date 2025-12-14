#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
import pandas as pd

def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))[:60]

def convert_one(xlsx_path: Path, out_dir: Path, sheet, all_sheets, sep, encoding, decimal):
    try:
        if all_sheets:
            xls = pd.ExcelFile(xlsx_path)
            n = 0
            for s in xls.sheet_names:
                df = pd.read_excel(xlsx_path, sheet_name=s, dtype=object, engine="openpyxl")
                (out_dir / f"{xlsx_path.stem}__{sanitize(s)}.csv").write_text(
                    df.to_csv(index=False, sep=sep, encoding=encoding, decimal=decimal)
                )
                n += 1
            return n
        else:
            sn = 0 if sheet is None else sheet   # ← clave: primera hoja por defecto
            df = pd.read_excel(xlsx_path, sheet_name=sn, dtype=object, engine="openpyxl")
            df.to_csv(out_dir / f"{xlsx_path.stem}.csv", index=False, sep=sep, encoding=encoding, decimal=decimal)
            return 1
    except Exception as e:
        print(f"[ERROR] {xlsx_path.name}: {e}", file=sys.stderr)
        return 0

def main():
    ap = argparse.ArgumentParser(description="Convierte Excel de una carpeta a CSV con el mismo nombre.")
    ap.add_argument("in_dir", nargs="?", default="excel")
    ap.add_argument("out_dir", nargs="?", default="csv")
    ap.add_argument("--sheet", help="Nombre o índice de hoja. Por defecto, 0 (primera).", default=None)
    ap.add_argument("--all-sheets", action="store_true", help="Exporta todas las hojas con sufijo __<hoja>.")
    ap.add_argument("--sep", default=",", help="Separador CSV. Ej: ';'")
    ap.add_argument("--encoding", default="utf-8-sig", help="Codificación. Excel-friendly: utf-8-sig.")
    ap.add_argument("--decimal", default=".", help="Carácter decimal para floats al exportar. Ej: ','")
    args = ap.parse_args()

    # Parse índice de hoja si viene como número
    try:
        if args.sheet is not None and str(args.sheet).strip().isdigit():
            args.sheet = int(args.sheet)
    except:
        pass

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [*in_dir.glob("*.xlsx"), *in_dir.glob("*.xls"), *in_dir.glob("*.xlsm")]
    if not files:
        print(f"No hay ficheros Excel en {in_dir}", file=sys.stderr)
        sys.exit(1)

    total = 0
    for f in files:
        total += convert_one(f, out_dir, args.sheet, args.all_sheets, args.sep, args.encoding, args.decimal)
    print(f"OK. Generados {total} CSV en {out_dir}")

if __name__ == "__main__":
    main()
