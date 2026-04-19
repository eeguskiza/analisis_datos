"""DATA-09 + DATA-07 meta-tests (grep-based).

No hay ``dbizaro.<algo>`` en ``api/`` (``OEE/db/connector.py`` esta
exento por D-04 — sigue teniendo 3-part names legacy). Tampoco hay
``import pyodbc`` en los routers.
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
API_DIR = ROOT / "api"
NEXO_DIR = ROOT / "nexo"


def _collect_py_files(base: pathlib.Path) -> list[pathlib.Path]:
    return [p for p in base.rglob("*.py") if "__pycache__" not in str(p)]


def test_no_three_part_names_in_api_dir():
    """Ni un solo ``dbizaro.<something>`` en ``api/`` (excluye comentarios ``#``)."""
    pattern = re.compile(r"dbizaro\.\w+", re.IGNORECASE)
    offenders: list[tuple[str, int, str]] = []
    for py in _collect_py_files(API_DIR):
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            # Ignorar comentarios de linea
            code = line.split("#")[0]
            if pattern.search(code):
                offenders.append((str(py.relative_to(ROOT)), i, line.strip()))
    assert not offenders, (
        f"3-part names en api/: {offenders}"
    )


def test_no_three_part_names_in_sql_files():
    """Ni en los ``.sql`` versionados de ``nexo/data/sql/mes/`` (excluye ``--`` SQL comments)."""
    pattern = re.compile(r"dbizaro\.\w+", re.IGNORECASE)
    offenders: list[tuple[str, int, str]] = []
    for sql in (NEXO_DIR / "data" / "sql" / "mes").glob("*.sql"):
        for i, line in enumerate(sql.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("--")[0]
            if pattern.search(code):
                offenders.append((str(sql.relative_to(ROOT)), i, line.strip()))
    assert not offenders, (
        f"3-part names en .sql: {offenders}"
    )


def test_no_pyodbc_import_in_routers():
    """DATA-07: ningun router importa ``pyodbc`` directo."""
    routers = (API_DIR / "routers").glob("*.py")
    offenders = []
    for py in routers:
        code = py.read_text(encoding="utf-8")
        for i, line in enumerate(code.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "import pyodbc" in stripped:
                offenders.append((py.name, i))
    assert not offenders, f"Routers con `import pyodbc`: {offenders}"
