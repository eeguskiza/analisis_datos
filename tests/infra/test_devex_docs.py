"""Validaciones estaticas del quartet de docs DevEx de Phase 7.

Previene regresiones sobre los 4 docs de cierre de Mark-III:
  - docs/ARCHITECTURE.md (DEVEX-04) — mapa tecnico con Mermaid + 3 engines.
  - docs/RUNBOOK.md (DEVEX-05) — 5 escenarios de incidencia con hallazgos criticos.
  - docs/RELEASE.md (DEVEX-06) — checklist semver + deploy + smoke.
  - CHANGELOG.md (DEVEX-06) — Keep a Changelog con historial Mark-III.

Si alguien borra accidentalmente una seccion critica (el diagrama Mermaid, un
escenario, el checklist, la seccion [1.0.0]) pytest falla y lo detecta antes
del merge.

Invariante hard (CLAUDE.md): ninguno de los 4 docs puede contener emojis.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCH = REPO_ROOT / "docs" / "ARCHITECTURE.md"
RUNBOOK = REPO_ROOT / "docs" / "RUNBOOK.md"
RELEASE = REPO_ROOT / "docs" / "RELEASE.md"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

# Regex emoji-heavy (coincidente con el check del PLAN.md verification).
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\U0001F600-\U0001F64F☀-➿]")


# ---- ARCHITECTURE.md ---------------------------------------------------------


def test_architecture_md_exists_and_has_mermaid():
    assert ARCH.exists(), f"Falta {ARCH} (DEVEX-04)."
    content = ARCH.read_text(encoding="utf-8")
    assert "mermaid" in content, (
        "ARCHITECTURE.md debe contener un bloque ```mermaid con el diagrama "
        "de los 3 engines."
    )


def test_architecture_md_has_three_engines():
    content = ARCH.read_text(encoding="utf-8")
    for engine in ("engine_mes", "engine_app", "engine_nexo"):
        assert engine in content, (
            f"ARCHITECTURE.md debe mencionar el engine {engine} "
            "(seccion 3 Los 3 engines)."
        )


def test_architecture_md_has_middleware_stack():
    content = ARCH.read_text(encoding="utf-8")
    for mw in (
        "AuthMiddleware",
        "AuditMiddleware",
        "FlashMiddleware",
        "QueryTimingMiddleware",
    ):
        assert mw in content, (
            f"ARCHITECTURE.md seccion 5 debe mencionar {mw} en el middleware stack."
        )


def test_architecture_md_cross_links():
    content = ARCH.read_text(encoding="utf-8")
    for doc in ("DEPLOY_LAN.md", "RUNBOOK.md", "RELEASE.md"):
        assert doc in content, (
            f"ARCHITECTURE.md seccion Enlaces rapidos debe cross-linkear a {doc}."
        )


def test_architecture_md_has_min_length():
    lines = ARCH.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 150, (
        f"ARCHITECTURE.md tiene {len(lines)} lineas; minimo 150 para cubrir "
        "las 9 secciones requeridas."
    )


# ---- RUNBOOK.md --------------------------------------------------------------


def test_runbook_md_has_exactly_5_scenarios():
    content = RUNBOOK.read_text(encoding="utf-8")
    scenarios = re.findall(r"^## Escenario [1-5]:", content, flags=re.MULTILINE)
    assert len(scenarios) == 5, (
        f"RUNBOOK.md debe tener EXACTAMENTE 5 escenarios; encontrados {len(scenarios)}."
    )


def test_runbook_md_scenario_structure():
    """Cada escenario debe tener las 4 sub-secciones: 5 * 4 = 20."""
    content = RUNBOOK.read_text(encoding="utf-8")
    subsections = re.findall(
        r"^### (Sintomas|Diagnostico|Remedio|Prevencion)",
        content,
        flags=re.MULTILINE,
    )
    assert len(subsections) >= 20, (
        f"RUNBOOK.md debe tener al menos 20 sub-secciones (5 escenarios x 4); "
        f"encontradas {len(subsections)}."
    )


def test_runbook_md_pipeline_scenario_has_restart_remedy():
    """Escenario 4 (Pipeline atascado): hallazgo critico list_locks inexistente."""
    content = RUNBOOK.read_text(encoding="utf-8")
    assert "pipeline_semaphore" in content, (
        "Escenario 4 debe mencionar pipeline_semaphore (atributo privado para "
        "diagnostico indirecto)."
    )
    assert "docker compose restart web" in content, (
        "Escenario 4 debe documentar el remedio nuclear: docker compose restart web."
    )
    assert "list_locks" in content or "NO existe" in content or "NO hay" in content, (
        "Escenario 4 debe documentar explicitamente que NO existe helper list_locks()."
    )


def test_runbook_md_lockout_scenario_has_delete_remedy():
    """Escenario 5 (Lockout propietario): hallazgo critico unlock_user inexistente."""
    content = RUNBOOK.read_text(encoding="utf-8")
    assert "DELETE FROM nexo.login_attempts" in content, (
        "Escenario 5 debe documentar el remedio DELETE FROM nexo.login_attempts."
    )
    # Prevencion literal >=2 propietarios (regex flexible para admitir variantes).
    assert re.search(r"(>= ?2|>=2|dos|al menos 2) propietarios", content), (
        "Escenario 5 prevencion debe exigir literal >=2 propietarios."
    )


def test_runbook_md_caddy_scenario_has_rotation_info():
    """Escenario 3 (Caddy): rotacion root CA 10 anos / intermediate 7 dias."""
    content = RUNBOOK.read_text(encoding="utf-8")
    assert "caddy_data" in content, (
        "Escenario 3 debe mencionar el volumen caddy_data (prevencion down -v)."
    )
    assert re.search(r"7 d|10 a|intermediate|root CA", content), (
        "Escenario 3 debe documentar la rotacion de certificados "
        "(7 dias intermediate, 10 anos root CA)."
    )


def test_runbook_md_postgres_scenario_mentions_pgdata():
    """Escenario 2 (Postgres): pgdata volume + Landmine 6 (no down -v)."""
    content = RUNBOOK.read_text(encoding="utf-8")
    assert re.search(r"pgdata|down -v|backup_nightly", content), (
        "Escenario 2 debe mencionar pgdata, backup_nightly.sh o el warning down -v."
    )


def test_runbook_md_has_min_length():
    lines = RUNBOOK.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 250, (
        f"RUNBOOK.md tiene {len(lines)} lineas; minimo 250 para cubrir los 5 escenarios."
    )


# ---- RELEASE.md --------------------------------------------------------------


def test_release_md_has_semver():
    content = RELEASE.read_text(encoding="utf-8")
    assert re.search(r"v[0-9]+\.[0-9]+\.[0-9]+", content), (
        "RELEASE.md debe mencionar una version semver (ej. v1.0.0)."
    )


def test_release_md_has_checklist_with_min_7_items():
    content = RELEASE.read_text(encoding="utf-8")
    checkboxes = re.findall(r"^- \[ \]", content, flags=re.MULTILINE)
    assert len(checkboxes) >= 7, (
        f"RELEASE.md debe tener al menos 7 items de checklist; "
        f"encontrados {len(checkboxes)}."
    )


def test_release_md_references_deploy_script():
    content = RELEASE.read_text(encoding="utf-8")
    assert "scripts/deploy.sh" in content, (
        "RELEASE.md debe referenciar scripts/deploy.sh (Phase 6 DEPLOY-04)."
    )
    assert "deploy_smoke.sh" in content, (
        "RELEASE.md debe referenciar tests/infra/deploy_smoke.sh (Phase 6 DEPLOY-07)."
    )


# ---- CHANGELOG.md ------------------------------------------------------------


def test_changelog_md_has_keep_a_changelog():
    content = CHANGELOG.read_text(encoding="utf-8")
    assert "Keep a Changelog" in content, (
        "CHANGELOG.md debe referenciar Keep a Changelog (formato canonico)."
    )
    assert "Semantic Versioning" in content or "semver" in content.lower(), (
        "CHANGELOG.md debe adherir a Semantic Versioning."
    )


def test_changelog_md_has_unreleased_and_1_0_0():
    content = CHANGELOG.read_text(encoding="utf-8")
    assert "## [Unreleased]" in content, (
        "CHANGELOG.md debe tener seccion [Unreleased] para el trabajo en curso."
    )
    assert "## [1.0.0]" in content, (
        "CHANGELOG.md debe tener seccion [1.0.0] con el cierre de Mark-III."
    )


def test_changelog_md_covers_all_phases():
    """Las Phases 1-7 deben estar mencionadas en el CHANGELOG."""
    content = CHANGELOG.read_text(encoding="utf-8")
    for n in range(1, 8):
        assert f"Phase {n}" in content, (
            f"CHANGELOG.md debe mencionar Phase {n} en la seccion [1.0.0] Added."
        )


# ---- Hard invariant: no emojis -----------------------------------------------


def test_all_docs_have_no_emojis():
    """Hard invariant CLAUDE.md: no emojis en ninguno de los 4 docs."""
    for doc in (ARCH, RUNBOOK, RELEASE, CHANGELOG):
        content = doc.read_text(encoding="utf-8")
        match = EMOJI_RE.search(content)
        assert match is None, (
            f"{doc.name} contiene emoji en offset {match.start() if match else 0}: "
            f"{match.group() if match else ''!r}. Prohibido por CLAUDE.md."
        )
