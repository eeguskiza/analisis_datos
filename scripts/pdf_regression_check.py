"""Gate final del Plan 03-02: compara PDF post-refactor contra baseline.

Uso (operador, despues del refactor de routers)::

    docker compose exec -T web python scripts/pdf_regression_check.py --fecha=2026-03-15

Exit codes::

    0 -> OK (hash identico, o page_count identico + size dentro de +-5%)
    1 -> WARN (hash difiere pero size +-5% y page_count identico —
              aceptable si el operador inspecciona visualmente y
              confirma equivalencia; matplotlib puede emitir
              timestamps distintos entre runs)
    2 -> REGRESSION (page_count distinto o size fuera de +-5%)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import date
from pathlib import Path


REF_DIR = Path(__file__).resolve().parents[1] / "tests" / "data" / "reference"
SIZE_TOLERANCE = 0.05  # +-5%


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pdf_pages(path: Path) -> int:
    """Conteo rustico: ocurrencias de ``/Type /Page`` menos las de ``/Pages``."""
    data = path.read_bytes()
    return data.count(b"/Type /Page") - data.count(b"/Type /Pages")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Gate de regresion PDF post-refactor (Plan 03-02)",
    )
    ap.add_argument("--fecha", required=True, help="Misma fecha que gen_pdf_reference.py")
    args = ap.parse_args()

    fecha = date.fromisoformat(args.fecha)
    ref_pdf = REF_DIR / f"pipeline_{fecha.isoformat()}.pdf"
    ref_sha = REF_DIR / f"pipeline_{fecha.isoformat()}.pdf.sha256"

    if not ref_pdf.exists() or not ref_sha.exists():
        print(
            f"ERROR: baseline ausente en {REF_DIR}. "
            f"Corre gen_pdf_reference.py --fecha={args.fecha} primero."
        )
        return 2

    # Import diferido — el parse del script no debe ejecutar el pipeline.
    from api.services.pipeline import run_pipeline

    tmp_out = REF_DIR / f"_postref_{fecha.isoformat()}"
    tmp_out.mkdir(exist_ok=True)

    run_pipeline(
        fecha_inicio=fecha,
        fecha_fin=fecha,
        recursos=None,
        source="db",
        output_dir=tmp_out,
    )

    new_pdfs = sorted(tmp_out.glob("*.pdf"))
    if not new_pdfs:
        print("ERROR: pipeline post-refactor no genero PDFs.")
        return 2
    new_pdf = new_pdfs[0]

    new_hash = _sha256(new_pdf)
    ref_hash_line = ref_sha.read_text(encoding="utf-8").strip().split()[0]

    if new_hash == ref_hash_line:
        print(f"OK — hash identico ({new_hash})")
        return 0

    ref_size = ref_pdf.stat().st_size
    new_size = new_pdf.stat().st_size
    diff_pct = abs(new_size - ref_size) / ref_size if ref_size else 1.0

    ref_pages = _pdf_pages(ref_pdf)
    new_pages = _pdf_pages(new_pdf)

    print("hash difiere:")
    print(f"  ref:  {ref_hash_line[:16]}... size={ref_size} pages={ref_pages}")
    print(f"  new:  {new_hash[:16]}... size={new_size} pages={new_pages}")
    print(f"  diff: size {diff_pct * 100:.1f}%, pages delta={new_pages - ref_pages}")

    if ref_pages != new_pages:
        print("REGRESSION: page count mismatch")
        return 2
    if diff_pct > SIZE_TOLERANCE:
        print(f"REGRESSION: size diff > {SIZE_TOLERANCE * 100:.0f}%")
        return 2

    print(
        f"WARN: bytes difieren (matplotlib timestamps?) pero size +-{SIZE_TOLERANCE * 100:.0f}% "
        f"y pages OK. Inspeccionar visualmente para decidir."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
