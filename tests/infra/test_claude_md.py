"""Tests de regresion sobre CLAUDE.md tras cierre Phase 7 (DEVEX-07).

CLAUDE.md es la guia autoritativa para IAs y devs nuevos. Estos tests
congelan las 5 invariantes criticas tras Mark-III / Sprint 6:

  1. "Ultima revision" bumpeada a fecha >= 2026-04-21 (cierre Phase 7).
  2. Seccion "Tooling DevEx" presente con pre-commit + make lint/format + quartet.
  3. Seccion "Despliegue productivo" presente con make prod-* + DEPLOY_LAN.md.
  4. Seccion "Que NO hacer" anade prohibiciones sobre --no-verify y cobertura 60%.
  5. Seccion "Fuente de verdad" lista ARCHITECTURE.md + RUNBOOK.md + RELEASE.md + CHANGELOG.md.

Cualquier PR que relaje estas reglas rompe el test y es visible en CI (T-07-16
del threat model del plan 07-04).

Ademas, congela invariantes pre-existentes (T-07-16) para detectar cualquier
regresion de decisiones previas: OEE/ sin rename, make up sin mcp, filter-repo
prohibido.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"


# ---- Test 1: existencia y legibilidad ---------------------------------------

def test_file_exists():
    assert CLAUDE_MD.exists(), f"Falta {CLAUDE_MD} — es el archivo mas critico del repo."


# ---- Test 2: header bumpeado tras Phase 7 -----------------------------------

def test_ultima_revision_bumped_post_phase7():
    """La fecha 'Ultima revision' debe ser >= 2026-04-21 (fecha cierre Phase 7)."""
    text = CLAUDE_MD.read_text(encoding="utf-8")
    m = re.search(r"[ÚUú]ltima revisi[óo]n:\s*(\d{4}-\d{2}-\d{2})", text)
    assert m is not None, "No se encontro linea 'Ultima revision' con fecha."
    fecha = m.group(1)
    assert fecha >= "2026-04-21", (
        f"Ultima revision no actualizada tras Phase 7: {fecha} (esperado >= 2026-04-21)."
    )


# ---- Test 3: seccion Tooling DevEx presente ---------------------------------

def test_has_tooling_devex_section():
    text = CLAUDE_MD.read_text(encoding="utf-8")
    assert "Tooling DevEx" in text, (
        "Falta seccion '## Tooling DevEx' (DEVEX-07 literal)."
    )


# ---- Test 4: Tooling DevEx referencia pre-commit ---------------------------

def test_tooling_devex_references_pre_commit():
    text = CLAUDE_MD.read_text(encoding="utf-8")
    assert "pre-commit install" in text, (
        "Seccion Tooling DevEx debe documentar 'pre-commit install' (setup dev)."
    )


# ---- Test 5: Tooling DevEx referencia make lint y make format --------------

def test_tooling_devex_references_make_lint_format():
    text = CLAUDE_MD.read_text(encoding="utf-8")
    assert "make lint" in text, "Falta referencia 'make lint' en CLAUDE.md."
    assert "make format" in text, "Falta referencia 'make format' en CLAUDE.md."


# ---- Test 6: Tooling DevEx cross-linkea con el quartet Phase 7 -------------

def test_tooling_devex_cross_links_quartet():
    text = CLAUDE_MD.read_text(encoding="utf-8")
    for doc in ("ARCHITECTURE.md", "RUNBOOK.md", "RELEASE.md"):
        assert doc in text, f"Cross-link ausente: {doc} debe aparecer en CLAUDE.md."


# ---- Test 7: seccion Despliegue productivo ---------------------------------

def test_has_deploy_production_section():
    text = CLAUDE_MD.read_text(encoding="utf-8")
    assert "Despliegue productivo" in text, (
        "Falta seccion '## Despliegue productivo' (Phase 6)."
    )
    assert "make prod-" in text, (
        "Seccion Despliegue productivo debe listar targets 'make prod-*'."
    )


# ---- Test 8: seccion Despliegue referencia DEPLOY_LAN.md -------------------

def test_deploy_section_references_lan_doc():
    text = CLAUDE_MD.read_text(encoding="utf-8")
    assert "DEPLOY_LAN.md" in text, (
        "Seccion Despliegue productivo debe referenciar docs/DEPLOY_LAN.md."
    )


# ---- Test 9: Que NO hacer — regla --no-verify ------------------------------

def test_que_no_hacer_has_no_verify_rule():
    """La prohibicion sobre --no-verify debe aparecer en la seccion 'Que NO hacer'.

    La entrada existente en 'Politica de commits' ("No se aplica --no-verify
    salvo decision explicita") NO es suficiente: tras Phase 7 pre-commit es
    obligatorio y queremos una regla HARD en el listado de 'Que NO hacer'.
    """
    text = CLAUDE_MD.read_text(encoding="utf-8")
    # Aislar la seccion "Qué NO hacer" (empieza con el heading hasta EOF o
    # siguiente heading ##).
    m = re.search(
        r"(?ms)^##\s+Qu[éeÉE]\s+NO\s+hacer\b(.*?)(?=^##\s|\Z)",
        text,
    )
    assert m is not None, "No se encontro la seccion '## Que NO hacer'."
    no_hacer_block = m.group(1)
    assert "no-verify" in no_hacer_block, (
        "La seccion 'Que NO hacer' debe contener prohibicion explicita sobre --no-verify."
    )


# ---- Test 10: Que NO hacer — regla cobertura 60% ---------------------------

def test_que_no_hacer_has_coverage_rule():
    text = CLAUDE_MD.read_text(encoding="utf-8")
    # Buscar referencia al gate de cobertura (60% o --cov-fail-under)
    has_gate = "cov-fail-under" in text or re.search(r"60\s*%", text) is not None
    assert has_gate, (
        "Falta prohibicion sobre bajar cobertura <60% (gate CI Phase 7)."
    )


# ---- Test 11: Fuente de verdad lista los 4 docs nuevos del quartet ---------

def test_fuente_de_verdad_lists_new_docs():
    text = CLAUDE_MD.read_text(encoding="utf-8")
    for doc in ("ARCHITECTURE.md", "RUNBOOK.md", "RELEASE.md", "CHANGELOG.md"):
        assert doc in text, (
            f"Seccion 'Fuente de verdad del plan' debe listar {doc} (Phase 7)."
        )


# ---- Test 12: sin emojis ----------------------------------------------------

def test_no_emojis():
    text = CLAUDE_MD.read_text(encoding="utf-8")
    emoji_re = re.compile(
        r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F600-\U0001F64F]"
    )
    m = emoji_re.search(text)
    assert m is None, f"Emoji prohibido en CLAUDE.md: {m.group(0)!r}"


# ---- Test 13: invariantes pre-existentes preservados -----------------------

def test_preserves_existing_invariants():
    """Decisiones Mark-III previas a Phase 7 NO deben desaparecer del doc."""
    text = CLAUDE_MD.read_text(encoding="utf-8")

    # MCP invariante: make up y make dev NO arrancan mcp (Phase 1 decision)
    assert "make up" in text and "mcp" in text.lower(), (
        "Invariante 'make up NO arranca mcp' debe seguir documentado."
    )
    assert "profiles:" in text or "profile mcp" in text, (
        "La regla del compose profile mcp debe seguir documentada."
    )

    # OEE/ no se renombra en Mark-III (diferido a Sprint 2 / Phase 3)
    assert "carpeta `OEE/`" in text or "carpeta OEE/" in text, (
        "Decision 'NO renombrar la carpeta OEE/' debe seguir documentada."
    )

    # Credenciales SQL Server: la politica original ("no rotar en Sprint 0") permanece
    assert "credenciales SQL Server" in text or "Credenciales SQL Server" in text, (
        "Decision sobre credenciales SQL Server debe seguir documentada."
    )

    # filter-repo esta prohibido sin autorizacion explicita
    assert "filter-repo" in text, (
        "Prohibicion de filter-repo debe seguir documentada."
    )
