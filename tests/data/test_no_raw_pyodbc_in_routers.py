"""DATA-07: ningun router de api/routers/ importa ``pyodbc`` directo.

Excepcion documentada (per 03-02 SUMMARY): ``api/routers/bbdd.py``
mantiene pyodbc SOLO para metadata ops (list_databases, list_tables,
list_columns, preview) contra connection strings dinamicos (master,
dbizaro, ecs_mobility). Las queries de data siguen usando
``MesRepository.consulta_readonly`` con whitelist anti-DDL. Esta
excepcion se permite y bbdd.py se excluye del meta-test.

Implementacion: AST-based (no line-grep) para no dar falsos positivos
con strings/docstrings que mencionen pyodbc textualmente.
"""
from __future__ import annotations

import ast
import pathlib

ROUTERS_DIR = pathlib.Path(__file__).resolve().parents[2] / "api" / "routers"

# bbdd.py es excepcion documentada (03-02 D-05: metadata ops).
ALLOWED_PYODBC_FILES = {"bbdd.py"}


def _iter_imports(tree: ast.AST):
    """Yield (lineno, module_or_name) para cada Import / ImportFrom."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            yield node.lineno, mod


def _iter_attribute_calls(tree: ast.AST):
    """Yield (lineno, 'foo.bar') para calls como foo.bar(...)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func
            # Solo chequeamos Attribute(Name(...), attr); e.g. pyodbc.connect
            if isinstance(attr.value, ast.Name):
                yield node.lineno, f"{attr.value.id}.{attr.attr}"


def test_no_raw_pyodbc_imports():
    """Ningun router (excepto bbdd.py) importa pyodbc directo.

    AST-based: ignora strings/docstrings que contengan la palabra
    ``pyodbc`` pero no sean un import real.
    """
    offenders: list[tuple[str, int, str]] = []
    for py in ROUTERS_DIR.glob("*.py"):
        if py.name in ALLOWED_PYODBC_FILES:
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for lineno, name in _iter_imports(tree):
            if name == "pyodbc" or name.startswith("pyodbc."):
                offenders.append((py.name, lineno, name))
    assert not offenders, (
        f"Routers con pyodbc directo (no permitido): {offenders}"
    )


def test_no_pyodbc_connect_calls_in_routers():
    """Double-check AST: no hay ``pyodbc.connect(...)`` en codigo ejecutable.

    Ignora docstrings y comentarios que mencionen la string textualmente
    (p.ej. api/routers/capacidad.py que documenta el refactor 03-02).
    bbdd.py se excluye (excepcion D-05).
    """
    offenders: list[tuple[str, int, str]] = []
    for py in ROUTERS_DIR.glob("*.py"):
        if py.name in ALLOWED_PYODBC_FILES:
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for lineno, full_attr in _iter_attribute_calls(tree):
            if full_attr == "pyodbc.connect":
                offenders.append((py.name, lineno, full_attr))
    assert not offenders, f"Routers con pyodbc.connect: {offenders}"


def test_bbdd_is_the_only_pyodbc_consumer():
    """Sanity: bbdd.py efectivamente sigue usando pyodbc (asegura que la
    excepcion esta documentada y no se ha ido silenciosamente).

    Si bbdd.py deja de usar pyodbc (p.ej. migrado 100% a MesRepository),
    este test fallaria y recordaria eliminar la excepcion.
    """
    bbdd = ROUTERS_DIR / "bbdd.py"
    assert bbdd.exists(), "bbdd.py no existe"
    content = bbdd.read_text(encoding="utf-8")
    assert "pyodbc" in content, (
        "bbdd.py ya no usa pyodbc - actualizar ALLOWED_PYODBC_FILES"
    )
