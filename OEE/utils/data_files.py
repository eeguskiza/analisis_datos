from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple, List

RESOURCE_DIR_NAME = "recursos"
DEFAULT_EXCLUDED_FILES: Sequence[str] = ("ciclos.csv",)


def listar_csv_recursos(
    base_path: Path, exclude_names: Iterable[str] = DEFAULT_EXCLUDED_FILES
) -> list[Path]:
    """Mantiene compatibilidad: solo retorna la lista plana de CSV de recursos."""
    return [path for _, path in listar_csv_por_seccion(base_path, exclude_names)]


def listar_csv_por_seccion(
    base_path: Path, exclude_names: Iterable[str] = DEFAULT_EXCLUDED_FILES
) -> List[Tuple[str, Path]]:
    """
    Devuelve tuplas (sección, ruta_csv), detectando las subcarpetas de `data/recursos`.

    Si no existen secciones, todos los CSV (excepto los excluidos) se agrupan bajo "general".
    """
    base_path = Path(base_path)
    recursos_dir = base_path / RESOURCE_DIR_NAME
    excluded = {name.lower() for name in exclude_names}
    resultados: List[Tuple[str, Path]] = []

    if recursos_dir.exists():
        for section_dir in sorted(p for p in recursos_dir.iterdir() if p.is_dir()):
            for csv_path in sorted(section_dir.glob("*.csv")):
                if csv_path.name.lower() in excluded or not csv_path.is_file():
                    continue
                resultados.append((section_dir.name, csv_path))
        return resultados

    for csv_path in sorted(base_path.glob("*.csv")):
        if csv_path.name.lower() in excluded or not csv_path.is_file():
            continue
        resultados.append(("general", csv_path))
    return resultados
