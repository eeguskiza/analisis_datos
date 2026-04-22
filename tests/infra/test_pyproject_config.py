"""Regresion tests sobre `pyproject.toml` (Phase 7 / DEVEX-01 + DEVEX-02).

Congelan el contrato de tooling Python: ruff + mypy + pytest + coverage.
Si alguien cambia la estructura o elimina un bloque critico (p. ej.
tool.coverage.report.fail_under), pytest falla antes del merge.
"""

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — repo target is py311+
    import tomli as tomllib  # type: ignore[import-not-found]

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "pyproject.toml"


def _load() -> dict:
    with CONFIG.open("rb") as fh:
        return tomllib.load(fh)


def test_file_exists() -> None:
    assert CONFIG.exists(), f"Falta {CONFIG} (DEVEX-01)."


def test_has_tool_ruff_block() -> None:
    data = _load()
    assert "tool" in data, "pyproject.toml sin tabla [tool]."
    assert "ruff" in data["tool"], "pyproject.toml sin [tool.ruff]."


def test_has_tool_mypy_block() -> None:
    data = _load()
    assert "mypy" in data["tool"], "pyproject.toml sin [tool.mypy]."


def test_has_tool_coverage_block_with_fail_under() -> None:
    data = _load()
    assert "coverage" in data["tool"], "pyproject.toml sin [tool.coverage]."
    report = data["tool"]["coverage"].get("report", {})
    fail_under = report.get("fail_under")
    assert fail_under is not None, "[tool.coverage.report] sin fail_under."
    assert isinstance(fail_under, int | float), (
        f"fail_under debe ser numerico; recibido {type(fail_under)}."
    )
    assert fail_under >= 50, (
        f"fail_under={fail_under}; se espera >= 50 (DEVEX-02 baseline Phase 7)."
    )


def test_mypy_has_overrides_for_legacy_libs() -> None:
    data = _load()
    overrides = data["tool"]["mypy"].get("overrides", [])
    assert overrides, "[[tool.mypy.overrides]] ausente."
    # Flatten modules
    all_modules: list[str] = []
    for ov in overrides:
        mods = ov.get("module", [])
        if isinstance(mods, str):
            all_modules.append(mods)
        else:
            all_modules.extend(mods)
    joined = " ".join(all_modules)
    for required in ("OEE.", "matplotlib.", "pyodbc."):
        assert required in joined, (
            f"mypy overrides sin '{required}*'; encontrados={all_modules}. "
            "DEVEX-01: libs legacy sin types deben ignorarse explicitamente."
        )


def test_pytest_config_has_asyncio_mode_auto() -> None:
    data = _load()
    pytest_cfg = data["tool"].get("pytest", {}).get("ini_options", {})
    assert pytest_cfg, "[tool.pytest.ini_options] ausente."
    assert pytest_cfg.get("asyncio_mode") == "auto", (
        f"asyncio_mode={pytest_cfg.get('asyncio_mode')}; se espera 'auto' "
        "(patron heredado de Phase 4 para tests LISTEN/NOTIFY)."
    )


def test_coverage_source_is_api_and_nexo() -> None:
    data = _load()
    run = data["tool"]["coverage"].get("run", {})
    sources = run.get("source", [])
    assert set(sources) >= {"api", "nexo"}, (
        f"[tool.coverage.run] source={sources}; se espera ['api','nexo'] "
        "(scope DEVEX-01)."
    )


def test_ruff_line_length_is_100() -> None:
    data = _load()
    ruff_cfg = data["tool"]["ruff"]
    assert ruff_cfg.get("line-length") == 100, (
        f"ruff line-length={ruff_cfg.get('line-length')}; se espera 100."
    )


def test_ruff_lint_has_selected_rules() -> None:
    """Ruff lint debe seleccionar reglas E/W/F/I/B/C4/UP/SIM (minimo 6)."""
    data = _load()
    lint_cfg = data["tool"]["ruff"].get("lint", {})
    selected = lint_cfg.get("select", [])
    required = {"E", "F", "I", "B"}
    missing = required - set(selected)
    assert not missing, f"Reglas ruff ausentes: {missing}. Seleccionadas={selected}."
