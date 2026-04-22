"""Regresion tests sobre `Makefile` (Phase 7 / DEVEX-03).

Congelan el contrato de targets:
- `test`, `lint`, `format`, `migrate`: nuevos en Phase 7 / DEVEX-03.
- `backup`: ya existia desde Phase 6 (verificamos que NO se perdio).
- Invariante duro Phase 1: `make up` y `make dev` NO arrancan el servicio mcp.

Estrategia: usar `make -n <target>` (dry-run) y parsear el stdout. Evitamos
ejecutar pytest/ruff/docker de verdad — solo verificamos que la receta
contiene los tokens relevantes.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_dry_run(target: str) -> subprocess.CompletedProcess[str]:
    """Ejecuta `make -n <target>` en el repo root y devuelve el CompletedProcess."""
    return subprocess.run(
        ["make", "-n", target],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_test_target_exists() -> None:
    result = _make_dry_run("test")
    assert result.returncode == 0, (
        f"`make -n test` exit={result.returncode}. stderr={result.stderr!r}"
    )
    assert "pytest" in result.stdout, (
        f"`make -n test` no contiene `pytest`: stdout={result.stdout!r}"
    )


def test_lint_target_exists() -> None:
    result = _make_dry_run("lint")
    assert result.returncode == 0, (
        f"`make -n lint` exit={result.returncode}. stderr={result.stderr!r}"
    )
    stdout = result.stdout
    assert "ruff check" in stdout, f"`make -n lint` sin `ruff check`: {stdout!r}"
    assert "ruff format --check" in stdout, (
        f"`make -n lint` sin `ruff format --check`: {stdout!r}"
    )
    assert "mypy" in stdout, f"`make -n lint` sin `mypy`: {stdout!r}"


def test_format_target_exists() -> None:
    result = _make_dry_run("format")
    assert result.returncode == 0, (
        f"`make -n format` exit={result.returncode}. stderr={result.stderr!r}"
    )
    stdout = result.stdout
    assert "ruff check --fix" in stdout, (
        f"`make -n format` sin `ruff check --fix`: {stdout!r}"
    )
    # `ruff format` (sin --check) debe aparecer — filtramos el --check de lint.
    lines = [line for line in stdout.splitlines() if "ruff format" in line]
    non_check = [ln for ln in lines if "--check" not in ln]
    assert non_check, (
        f"`make -n format` sin `ruff format` auto-format: {stdout!r}"
    )


def test_migrate_target_exists() -> None:
    result = _make_dry_run("migrate")
    assert result.returncode == 0, (
        f"`make -n migrate` exit={result.returncode}. stderr={result.stderr!r}"
    )
    assert "init_nexo_schema.py" in result.stdout, (
        f"`make -n migrate` sin `init_nexo_schema.py`: stdout={result.stdout!r}"
    )


def test_backup_target_still_exists() -> None:
    """Conservacion Phase 6: `backup` ya existia antes de DEVEX-03."""
    result = _make_dry_run("backup")
    assert result.returncode == 0, (
        f"`make -n backup` exit={result.returncode}. El target de Phase 6 no debe perderse. "
        f"stderr={result.stderr!r}"
    )


def test_up_target_does_not_start_mcp() -> None:
    """Invariante Phase 1: `make up` NO arranca el servicio mcp."""
    result = _make_dry_run("up")
    combined = result.stdout + result.stderr
    assert "profile mcp" not in combined, (
        f"`make -n up` arranca mcp profile (viola invariante Phase 1). "
        f"stdout={result.stdout!r}"
    )


def test_dev_target_does_not_start_mcp() -> None:
    """Invariante Phase 1: `make dev` NO arranca el servicio mcp."""
    result = _make_dry_run("dev")
    combined = result.stdout + result.stderr
    assert "profile mcp" not in combined, (
        f"`make -n dev` arranca mcp profile (viola invariante Phase 1). "
        f"stdout={result.stdout!r}"
    )


def test_phony_declares_new_targets() -> None:
    """La primera linea `.PHONY:` del Makefile debe incluir test/lint/format/migrate."""
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    first_phony_line: str | None = None
    for raw in makefile.splitlines():
        if raw.strip().startswith(".PHONY:"):
            first_phony_line = raw
            break
    assert first_phony_line is not None, "Makefile sin ninguna linea `.PHONY:`."
    for target in ("test", "lint", "format", "migrate"):
        # Match como palabra separada (no substring de e.g. 'test-docker').
        assert re.search(rf"\b{target}\b", first_phony_line), (
            f"`.PHONY:` no declara `{target}`. Linea: {first_phony_line!r}"
        )


def test_test_target_has_cov_fail_under() -> None:
    """Dry-run de `make -n test` debe pasar `--cov-fail-under=<entero>`."""
    result = _make_dry_run("test")
    assert result.returncode == 0, result.stderr
    match = re.search(r"--cov-fail-under=(\d+)", result.stdout)
    assert match is not None, (
        f"`make -n test` sin `--cov-fail-under=N`: stdout={result.stdout!r}. "
        "DEVEX-03 exige paridad con CI (pyproject.toml fail_under=60)."
    )
    value = int(match.group(1))
    assert value >= 50, (
        f"--cov-fail-under={value}; se espera >= 50 (baseline Phase 7 = 60)."
    )
