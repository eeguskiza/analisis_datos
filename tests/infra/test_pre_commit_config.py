"""Regresion tests sobre `.pre-commit-config.yaml` (Phase 7 / DEVEX-01).

Congelan el contrato del archivo: scope a api/+nexo/, ruff-format sustituye a
black, hook de mypy con --config-file=pyproject.toml, ruff-pre-commit pinned
en v0.15.x. Si alguien cambia la estructura, pytest falla y lo detecta antes
del merge.
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / ".pre-commit-config.yaml"


def _load() -> dict:
    with CONFIG.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_file_exists() -> None:
    assert CONFIG.exists(), f"Falta {CONFIG} (DEVEX-01)."


def test_file_is_valid_yaml() -> None:
    data = _load()
    assert isinstance(data, dict), ".pre-commit-config.yaml no parsea a dict."
    assert "repos" in data, "Falta clave 'repos' en el config."
    assert isinstance(data["repos"], list) and data["repos"], "'repos' debe ser lista no vacia."


def test_hooks_scoped_to_api_nexo() -> None:
    """Al menos 3 hooks language-specific deben limitarse a ^(api|nexo)/."""
    data = _load()
    scoped = 0
    for repo in data["repos"]:
        for hook in repo.get("hooks", []):
            files = hook.get("files", "")
            if files == "^(api|nexo)/":
                scoped += 1
    assert scoped >= 3, (
        f"Se esperan >= 3 hooks con files='^(api|nexo)/'; encontrados {scoped}. "
        "DEVEX-01 literal: hooks solo tocan api/+nexo/."
    )


def test_ruff_format_present_no_black() -> None:
    """Debe existir hook ruff-format y NO existir hook psf/black ni id: black."""
    content = CONFIG.read_text(encoding="utf-8")
    assert "ruff-format" in content, "Falta hook 'ruff-format' (sustituye a black en DEVEX-01)."
    assert "psf/black" not in content, (
        "No debe existir referencia a 'psf/black' en pre-commit (decision Phase 7: "
        "ruff-format reemplaza a black)."
    )
    data = _load()
    for repo in data["repos"]:
        for hook in repo.get("hooks", []):
            assert hook.get("id") != "black", "No debe haber hook con 'id: black'."


def test_mypy_config_file_is_pyproject() -> None:
    """El hook de mypy debe apuntar a --config-file=pyproject.toml."""
    data = _load()
    mypy_args: list[str] = []
    for repo in data["repos"]:
        for hook in repo.get("hooks", []):
            if hook.get("id") == "mypy":
                mypy_args = hook.get("args", [])
    assert mypy_args, "No se encontro hook mypy con args."
    joined = " ".join(mypy_args)
    assert "--config-file=pyproject.toml" in joined, (
        f"Hook mypy no apunta a --config-file=pyproject.toml; args={mypy_args}."
    )


def test_ruff_pre_commit_rev_is_v0_15_x() -> None:
    """El repo astral-sh/ruff-pre-commit debe estar pineado a v0.15.*."""
    data = _load()
    ruff_rev: str | None = None
    for repo in data["repos"]:
        if "astral-sh/ruff-pre-commit" in repo.get("repo", ""):
            ruff_rev = repo.get("rev", "")
    assert ruff_rev is not None, "No se encontro repo astral-sh/ruff-pre-commit."
    assert ruff_rev.startswith("v0.15."), (
        f"ruff-pre-commit rev={ruff_rev}; se espera v0.15.* (version estable 2026-04)."
    )


def test_mypy_has_additional_dependencies() -> None:
    """Hook mypy debe declarar additional_dependencies (fastapi, pydantic-settings, sqlalchemy)."""
    data = _load()
    for repo in data["repos"]:
        for hook in repo.get("hooks", []):
            if hook.get("id") == "mypy":
                deps = hook.get("additional_dependencies", [])
                joined = " ".join(deps)
                assert "fastapi" in joined, "mypy additional_dependencies sin fastapi."
                assert "pydantic-settings" in joined, (
                    "mypy additional_dependencies sin pydantic-settings."
                )
                assert "sqlalchemy" in joined, "mypy additional_dependencies sin sqlalchemy."
                return
    pytest.fail("Hook mypy no encontrado.")
