"""Regresion tests sobre `.github/workflows/ci.yml` (Phase 7 / DEVEX-02).

Congelan el contrato del CI:
- Matriz Python 3.11 + 3.12 en los jobs `lint` y `test`.
- Coverage gate duro (`--cov-fail-under=N`) en el job `test`.
- Nuevo job `smoke` que arranca `docker compose up -d --build db web` y
  cura `localhost:8001/api/health`.
- Service container `postgres:16-alpine` en el job `test` (aislado del
  compose del smoke).
- `continue-on-error: true` conservado SOLO en el job `secrets`
  (historical findings en docs/SECURITY_AUDIT.md).

Si alguien relaja el CI (vuelve a Py 3.11-only, elimina el gate, o reintroduce
`continue-on-error` en `test`), estos tests fallan.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _load() -> dict[str, Any]:
    with CI_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _job_steps_as_text(job: dict[str, Any]) -> str:
    """Concatena todos los bloques `run` de los steps en un solo string."""
    chunks: list[str] = []
    for step in job.get("steps", []) or []:
        run = step.get("run")
        if isinstance(run, str):
            chunks.append(run)
    return "\n".join(chunks)


def test_ci_yml_exists_and_parseable() -> None:
    assert CI_PATH.exists(), f"Falta {CI_PATH} (DEVEX-02)."
    data = _load()
    assert isinstance(data, dict), "ci.yml no parsea a dict."
    assert "jobs" in data, "ci.yml sin sección `jobs`."


def _python_versions_of(job: dict[str, Any]) -> list[str]:
    strategy = job.get("strategy", {}) or {}
    matrix = strategy.get("matrix", {}) or {}
    versions = matrix.get("python-version", []) or []
    # YAML puede cargarlos como floats (3.11) si no se cita. Normalizamos a str.
    return [str(v) for v in versions]


def test_lint_job_has_matrix_3_11_and_3_12() -> None:
    data = _load()
    lint = data["jobs"].get("lint")
    assert lint is not None, "Falta job `lint`."
    versions = _python_versions_of(lint)
    assert "3.11" in versions, f"lint.matrix sin '3.11'; got={versions}."
    assert "3.12" in versions, f"lint.matrix sin '3.12'; got={versions}."


def test_test_job_has_matrix_3_11_and_3_12() -> None:
    data = _load()
    test_job = data["jobs"].get("test")
    assert test_job is not None, "Falta job `test`."
    versions = _python_versions_of(test_job)
    assert "3.11" in versions, f"test.matrix sin '3.11'; got={versions}."
    assert "3.12" in versions, f"test.matrix sin '3.12'; got={versions}."


def test_test_job_has_cov_fail_under() -> None:
    data = _load()
    test_job = data["jobs"].get("test")
    assert test_job is not None, "Falta job `test`."
    blob = _job_steps_as_text(test_job)
    match = re.search(r"--cov-fail-under=(\d+)", blob)
    assert match is not None, (
        "Step pytest del job `test` sin `--cov-fail-under=N`. "
        "DEVEX-02 exige coverage gate duro."
    )
    value = int(match.group(1))
    assert value >= 50, (
        f"--cov-fail-under={value}; se espera >= 50 (baseline Phase 7 = 60)."
    )


def test_test_job_has_no_continue_on_error() -> None:
    data = _load()
    test_job = data["jobs"].get("test")
    assert test_job is not None, "Falta job `test`."
    assert test_job.get("continue-on-error") is not True, (
        "Job `test` tiene `continue-on-error: true`. DEVEX-02 exige que sea "
        "bloqueante tras activar el coverage gate."
    )


def test_smoke_job_exists() -> None:
    data = _load()
    smoke = data["jobs"].get("smoke")
    assert smoke is not None, "Falta job `smoke`."
    blob = _job_steps_as_text(smoke)
    # (a) .env sembrado desde .env.example
    assert "cp .env.example .env" in blob, (
        "Job `smoke` debe sembrar `.env` desde `.env.example`."
    )
    # (b) docker compose up -d --build db web
    assert re.search(r"docker compose up -d.*--build.*db.*web", blob, re.DOTALL), (
        "Job `smoke` debe arrancar `docker compose up -d --build db web`. "
        "NOTA: NO debe incluir el servicio mcp (invariante Phase 1)."
    )
    # (c) curl contra el puerto 8001 del host (NEXO_WEB_HOST_PORT default)
    assert "localhost:8001/api/health" in blob, (
        "Job `smoke` debe curlear `http://localhost:8001/api/health` "
        "(puerto host por defecto de compose, NO 8000)."
    )


def test_secrets_job_keeps_continue_on_error() -> None:
    data = _load()
    secrets_job = data["jobs"].get("secrets")
    assert secrets_job is not None, "Falta job `secrets`."
    assert secrets_job.get("continue-on-error") is True, (
        "Job `secrets` debe mantener `continue-on-error: true` (historical "
        "findings en docs/SECURITY_AUDIT.md hasta Mark-IV)."
    )


def test_lint_job_runs_mypy() -> None:
    data = _load()
    lint = data["jobs"].get("lint")
    assert lint is not None, "Falta job `lint`."
    blob = _job_steps_as_text(lint)
    assert re.search(r"\bmypy\s+api/\s+nexo/", blob), (
        "Job `lint` debe correr `mypy api/ nexo/` (DEVEX-01 pide mypy scoped)."
    )


def test_no_ruff_format_continue_on_error() -> None:
    data = _load()
    lint = data["jobs"].get("lint")
    assert lint is not None, "Falta job `lint`."
    for step in lint.get("steps", []) or []:
        name = (step.get("name") or "").lower()
        run = step.get("run") or ""
        # Identifica el step de ruff format --check (independiente del
        # wording exacto del `name`).
        if "ruff format --check" in run or "ruff format" in name:
            assert step.get("continue-on-error") is not True, (
                f"Step ruff format --check tiene `continue-on-error: true`: "
                f"{step}. Tras el backfill de 07-01 debe ser bloqueante."
            )


def test_test_job_has_postgres_service() -> None:
    data = _load()
    test_job = data["jobs"].get("test")
    assert test_job is not None, "Falta job `test`."
    services = test_job.get("services") or {}
    postgres = services.get("postgres")
    assert postgres is not None, (
        "Job `test` sin service-container `postgres`. DEVEX-02 lo requiere "
        "para tests de BD aislados del compose del smoke."
    )
    image = str(postgres.get("image", ""))
    assert "postgres:16-alpine" in image, (
        f"service.postgres.image={image!r}; se espera `postgres:16-alpine` "
        "(paridad con compose base)."
    )
