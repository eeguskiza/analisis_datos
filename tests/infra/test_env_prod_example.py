"""Validaciones estaticas de .env.prod.example (Phase 6 / Plan 06-01).

DEPLOY-08: el archivo enumera TODAS las NEXO_* con placeholders y SIN valores
reales. Bloque SMTP comentado con # TODO Mark-IV (D-27).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
ENV_PROD: Final[Path] = REPO_ROOT / ".env.prod.example"
GITIGNORE: Final[Path] = REPO_ROOT / ".gitignore"

# Vars criticas que DEBEN aparecer (NEXO_* obligatorias del compose y api/config.py).
# Extraido de RESEARCH.md Topic 10. Si falta alguna, pydantic-settings peta al arrancar.
REQUIRED_NEXO_VARS: Final[tuple[str, ...]] = (
    "NEXO_HOST",
    "NEXO_PORT",
    "NEXO_DEBUG",
    "NEXO_SECRET_KEY",
    "NEXO_SESSION_COOKIE_NAME",
    "NEXO_SESSION_TTL_HOURS",
    "NEXO_PG_HOST",
    "NEXO_PG_PORT",
    "NEXO_PG_USER",
    "NEXO_PG_PASSWORD",
    "NEXO_PG_DB",
    "NEXO_PG_APP_USER",
    "NEXO_PG_APP_PASSWORD",
    "NEXO_APP_SERVER",
    "NEXO_APP_USER",
    "NEXO_APP_PASSWORD",
    "NEXO_MES_SERVER",
    "NEXO_MES_USER",
    "NEXO_MES_PASSWORD",
    "NEXO_APP_NAME",
    "NEXO_COMPANY_NAME",
    "NEXO_LOGO_PATH",
    "NEXO_ECS_LOGO_PATH",
    "NEXO_DATA_DIR",
    "NEXO_INFORMES_DIR",
)


def test_env_prod_example_exists() -> None:
    assert ENV_PROD.exists(), f"Falta {ENV_PROD} (DEPLOY-08)."


def test_env_prod_example_has_smtp_todo_markiv() -> None:
    content = ENV_PROD.read_text(encoding="utf-8")
    assert "# TODO Mark-IV" in content, (
        "El bloque SMTP debe estar comentado con marca `# TODO Mark-IV` (D-27)."
    )


def test_env_prod_example_smtp_lines_commented() -> None:
    content = ENV_PROD.read_text(encoding="utf-8")
    # Todas las lineas NEXO_SMTP_* deben empezar por `#`.
    for line in content.splitlines():
        if "NEXO_SMTP_" in line and "#" not in line.split("NEXO_SMTP_")[0]:
            raise AssertionError(
                f"Linea SMTP no comentada: {line!r}. SMTP debe estar "
                "comentado al 100% en Mark-III (D-27)."
            )


def test_env_prod_example_has_all_required_nexo_vars() -> None:
    content = ENV_PROD.read_text(encoding="utf-8")
    missing: list[str] = []
    for var in REQUIRED_NEXO_VARS:
        # Var debe aparecer como asignacion (no como comentario).
        pattern = re.compile(rf"^{re.escape(var)}=", re.MULTILINE)
        if not pattern.search(content):
            missing.append(var)
    assert not missing, (
        f"Variables NEXO_* faltantes en .env.prod.example: {missing}"
    )


def test_env_prod_example_has_literal_host_port() -> None:
    content = ENV_PROD.read_text(encoding="utf-8")
    assert "NEXO_HOST=0.0.0.0" in content, "NEXO_HOST debe ser literal 0.0.0.0 (D-28)."
    assert "NEXO_PORT=8000" in content, "NEXO_PORT debe ser literal 8000 (D-28)."


def test_env_prod_example_has_compose_project_name() -> None:
    content = ENV_PROD.read_text(encoding="utf-8")
    # Landmine 10: COMPOSE_PROJECT_NAME=nexo para containers nexo-web-1, etc.
    assert "COMPOSE_PROJECT_NAME=nexo" in content, (
        "COMPOSE_PROJECT_NAME=nexo requerido para naming convention nexo-* (CLAUDE.md)."
    )


def test_env_prod_example_no_real_secrets() -> None:
    """Los campos sensibles deben ser placeholders <CHANGEME-*>, nunca valores reales."""
    content = ENV_PROD.read_text(encoding="utf-8")
    forbidden_patterns = [
        # Passwords con contenido no-placeholder (simple heuristica)
        re.compile(r"^NEXO_SECRET_KEY=(?!<CHANGEME)(?!\s*$).+", re.MULTILINE),
        re.compile(r"^NEXO_PG_PASSWORD=(?!<CHANGEME)(?!\s*$).+", re.MULTILINE),
        re.compile(r"^NEXO_PG_APP_PASSWORD=(?!<CHANGEME)(?!\s*$).+", re.MULTILINE),
        re.compile(r"^NEXO_APP_PASSWORD=(?!<CHANGEME)(?!\s*$).+", re.MULTILINE),
        re.compile(r"^NEXO_MES_PASSWORD=(?!<CHANGEME)(?!\s*$).+", re.MULTILINE),
    ]
    offenders: list[str] = []
    for pat in forbidden_patterns:
        m = pat.search(content)
        if m:
            offenders.append(m.group(0))
    assert not offenders, (
        f"Valores reales detectados en .env.prod.example: {offenders}. "
        "Todos los secretos deben ser <CHANGEME-*> (DEPLOY-08)."
    )


def test_gitignore_ignores_env_prod_but_not_example() -> None:
    content = GITIGNORE.read_text(encoding="utf-8")
    # `.env.prod` debe aparecer explicito
    assert ".env.prod" in content, (
        ".gitignore debe ignorar `.env.prod` (puede ser `.env.prod` literal "
        "o un patron `.env.*` con whitelist)."
    )
    # `.env.prod.example` DEBE poder commitearse. Verificacion: si el
    # gitignore usa un patron amplio como `.env.*`, debe existir negacion
    # `!.env.prod.example`. Implementacion simple: si `.env.*` aparece
    # como linea aislada, exigir `!.env.prod.example`.
    if re.search(r"^\.env\.\*\s*$", content, re.MULTILINE):
        assert "!.env.prod.example" in content, (
            "Si .gitignore usa patron amplio `.env.*`, debe haber whitelist "
            "`!.env.prod.example` para que el template se pueda commitear."
        )
