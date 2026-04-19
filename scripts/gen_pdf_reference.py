"""Graba PDF de referencia pre-refactor para el gate de regresion (Plan 03-02 success #5).

Uso (desde el host, operador con acceso LAN a MES)::

    docker compose exec -T web python scripts/gen_pdf_reference.py --fecha=2026-03-15

El ``--fecha`` debe apuntar a una fecha estable (>=30 dias atras, para
que IZARO no siga actualizando esos registros y dos runs del pipeline
produzcan el mismo output byte a byte en el caso ideal).

Output::

    tests/data/reference/pipeline_<fecha>.pdf        (PDF principal — primero alfabeticamente)
    tests/data/reference/pipeline_<fecha>.pdf.sha256 (hash + nombre)

El operador commitea SOLO el ``.sha256`` (gitignore el PDF si >5MB).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import date
from pathlib import Path


# sys.path shim: ``docker compose exec web python scripts/X.py`` pone
# /app/scripts en sys.path[0], no /app. Insertamos el parent para que
# ``from api...`` funcione igual que desde el repl del container.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


REF_DIR = _REPO_ROOT / "tests" / "data" / "reference"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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

    # Import diferido para que cargar el script no arranque el pipeline
    # (p.ej. en test_pdf_scripts_parse).
    from api.config import settings
    from api.services.pipeline import run_pipeline

    informes_dir: Path = settings.informes_dir

    # Snapshot de PDFs previos para diff post-run — asi localizamos SOLO
    # los generados en este run (no los de historicos).
    before: set[Path] = set(informes_dir.rglob("*.pdf")) if informes_dir.exists() else set()

    # run_pipeline es un generador (SSE log lines). Hay que consumirlo
    # para que el pipeline avance.
    print(f"[gen_pdf_reference] Pipeline start fecha={fecha.isoformat()}")
    for line in run_pipeline(
        fecha_inicio=fecha,
        fecha_fin=fecha,
        source="db",
    ):
        print(f"  >> {line}")

    after: set[Path] = set(informes_dir.rglob("*.pdf"))
    new_pdfs = sorted(after - before)
    if not new_pdfs:
        print("ERROR: pipeline no genero PDFs nuevos en {informes_dir}")
        return 1

    # PDF canonico para el hash: primero alfabeticamente de los nuevos.
    # Cualquier regresion del pipeline post-refactor cambiara este hash
    # o el page-count (detectado por pdf_regression_check).
    main_pdf = new_pdfs[0]
    target = REF_DIR / f"pipeline_{fecha.isoformat()}.pdf"
    target.write_bytes(main_pdf.read_bytes())

    digest = _sha256(target)
    sha_file = REF_DIR / f"pipeline_{fecha.isoformat()}.pdf.sha256"
    sha_file.write_text(f"{digest}  {target.name}\n", encoding="utf-8")

    print()
    print("OK — referencia grabada:")
    print(f"  source:   {main_pdf}")
    print(f"  path:     {target}")
    print(f"  sha256:   {digest}")
    print(f"  size:     {target.stat().st_size} bytes")
    print(f"  new PDFs: {len(new_pdfs)} (referenciando el primero alfabetico)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
