"""Graba PDF de referencia pre-refactor para el gate de regresion (Plan 03-02 success #5).

Uso (desde el host, operador con acceso LAN a MES)::

    docker compose exec -T web python scripts/gen_pdf_reference.py --fecha=2026-03-15

El ``--fecha`` debe apuntar a una fecha estable (>=30 dias atras, para
que IZARO no siga actualizando esos registros y dos runs del pipeline
produzcan el mismo output byte a byte en el caso ideal).

Output::

    tests/data/reference/pipeline_<fecha>.pdf        (PDF principal)
    tests/data/reference/pipeline_<fecha>.pdf.sha256 (hash + nombre)

El operador commitea SOLO el ``.sha256`` (gitignore el PDF si >5MB).
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import date
from pathlib import Path


REF_DIR = Path(__file__).resolve().parents[1] / "tests" / "data" / "reference"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Graba PDF de referencia pre-refactor (Plan 03-02)",
    )
    ap.add_argument(
        "--fecha",
        required=True,
        help="YYYY-MM-DD fecha estable para el pipeline (>=30 dias atras)",
    )
    args = ap.parse_args()

    fecha = date.fromisoformat(args.fecha)
    REF_DIR.mkdir(parents=True, exist_ok=True)

    # Import diferido — evita ejecutar el pipeline al simplemente cargar
    # el script (por ejemplo, desde test_pdf_scripts_parse).
    from api.services.pipeline import run_pipeline

    out_dir = REF_DIR / f"run_{fecha.isoformat()}"
    out_dir.mkdir(exist_ok=True)

    run_pipeline(
        fecha_inicio=fecha,
        fecha_fin=fecha,
        recursos=None,
        source="db",
        output_dir=out_dir,
    )

    pdfs = sorted(out_dir.glob("*.pdf"))
    if not pdfs:
        print(f"ERROR: pipeline no genero PDFs en {out_dir}")
        return 1

    main_pdf = pdfs[0]
    target = REF_DIR / f"pipeline_{fecha.isoformat()}.pdf"
    target.write_bytes(main_pdf.read_bytes())

    digest = _sha256(target)
    sha_file = REF_DIR / f"pipeline_{fecha.isoformat()}.pdf.sha256"
    sha_file.write_text(f"{digest}  {target.name}\n", encoding="utf-8")

    print("OK — referencia grabada:")
    print(f"  path:   {target}")
    print(f"  sha256: {digest}")
    print(f"  size:   {target.stat().st_size} bytes")
    print("  pages:  inspeccionar con pdfinfo/preview")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
