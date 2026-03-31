"""Servicio para listar y gestionar informes generados."""
from __future__ import annotations

import shutil
from pathlib import Path

from api.config import settings


def _build_tree(directory: Path, base: Path) -> list[dict]:
    """Construye un arbol recursivo de directorios y PDFs."""
    entries: list[dict] = []
    if not directory.exists():
        return entries

    for child in sorted(directory.iterdir()):
        if child.name.startswith("."):
            continue
        rel = str(child.relative_to(base))
        if child.is_dir():
            children = _build_tree(child, base)
            if children:
                entries.append({
                    "name": child.name,
                    "path": rel,
                    "type": "dir",
                    "children": children,
                })
        elif child.suffix.lower() == ".pdf":
            entries.append({
                "name": child.name,
                "path": rel,
                "type": "pdf",
                "children": [],
            })
    return entries


def list_all() -> list[dict]:
    """Devuelve el arbol completo de informes/ ordenado por fecha desc."""
    tree = _build_tree(settings.informes_dir, settings.informes_dir)
    tree.sort(key=lambda e: e["name"], reverse=True)
    return tree


def list_dates() -> list[str]:
    """Devuelve las fechas (nombres de carpetas) disponibles."""
    if not settings.informes_dir.exists():
        return []
    return sorted(
        [d.name for d in settings.informes_dir.iterdir() if d.is_dir() and not d.name.startswith(".")],
        reverse=True,
    )


def get_pdf_path(relative: str) -> Path | None:
    """Devuelve la ruta absoluta de un PDF si existe."""
    full = settings.informes_dir / relative
    if full.exists() and full.suffix.lower() == ".pdf":
        return full
    return None


def delete_date(date_str: str) -> bool:
    """Elimina un directorio de informes de una fecha completa."""
    target = settings.informes_dir / date_str
    if target.exists() and target.is_dir():
        shutil.rmtree(target)
        return True
    return False


def count_pdfs(date_str: str | None = None) -> int:
    """Cuenta PDFs, opcionalmente para una fecha."""
    base = settings.informes_dir
    if date_str:
        base = base / date_str
    if not base.exists():
        return 0
    return sum(1 for _ in base.rglob("*.pdf"))
