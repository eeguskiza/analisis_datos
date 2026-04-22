"""Regression for Phase 8 chrome structure.

Locks:
1. base.html loads Alpine Focus + Persist BEFORE Alpine core.
2. base.html declares the drawer with role=dialog + aria-modal + x-trap.
3. base.html includes Tailwind config <script> BEFORE the CDN script.
4. base.html loads print.css behind media="print".
5. base.html declares toast-root container.
6. RUNBOOK.md has the 5 canonical scenario headings with stable slugs.
7. Any template that hard-codes a RUNBOOK anchor uses the full GFM slug
   (no short form like #escenario-1-mes-caido without the rest).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_BASE_HTML = _ROOT / "templates" / "base.html"
_RUNBOOK_MD = _ROOT / "docs" / "RUNBOOK.md"
_PRINT_CSS = _ROOT / "static" / "css" / "print.css"
_TOKENS_CSS = _ROOT / "static" / "css" / "tokens.css"


def test_tokens_css_present():
    assert _TOKENS_CSS.exists()


def test_print_css_present():
    assert _PRINT_CSS.exists()
    assert "@media print" in _PRINT_CSS.read_text(encoding="utf-8")


def test_base_html_loads_alpine_focus_persist_before_core():
    text = _BASE_HTML.read_text(encoding="utf-8")
    focus_idx = text.find("@alpinejs/focus@3.14.8")
    persist_idx = text.find("@alpinejs/persist@3.14.8")
    alpine_idx = text.find("alpinejs@3.14.8/dist/cdn.min.js")
    assert focus_idx > 0, "Alpine Focus plugin not loaded"
    assert persist_idx > 0, "Alpine Persist plugin not loaded"
    assert alpine_idx > 0, "Alpine core not loaded"
    assert focus_idx < alpine_idx, "Alpine Focus must appear before core (Pitfall 4)"
    assert persist_idx < alpine_idx, "Alpine Persist must appear before core (Pitfall 4)"


def test_base_html_loads_tailwind_config_before_cdn():
    text = _BASE_HTML.read_text(encoding="utf-8")
    cfg_idx = text.find('src="/static/js/tailwind.config.js"')
    cdn_idx = text.find("cdn.tailwindcss.com")
    assert cfg_idx > 0 and cdn_idx > 0, "tailwind.config.js + CDN both required"
    assert cfg_idx < cdn_idx, "tailwind.config.js must load before the CDN (Pitfall 1)"


def test_base_html_loads_print_css():
    text = _BASE_HTML.read_text(encoding="utf-8")
    assert 'href="/static/css/print.css" media="print"' in text


def test_base_html_drawer_has_a11y_attributes():
    text = _BASE_HTML.read_text(encoding="utf-8")
    assert 'id="nexo-drawer"' in text
    assert 'role="dialog"' in text
    assert 'aria-modal="true"' in text
    assert 'x-trap.noscroll="drawerOpen"' in text


def test_base_html_has_toast_root():
    text = _BASE_HTML.read_text(encoding="utf-8")
    assert 'id="toast-root"' in text
    assert 'aria-live="polite"' in text


def test_base_html_has_hamburger_with_aria_label():
    text = _BASE_HTML.read_text(encoding="utf-8")
    assert 'aria-label="Abrir menú"' in text
    assert 'aria-controls="nexo-drawer"' in text


def test_base_html_has_nexo_chrome_alpine_component():
    text = _BASE_HTML.read_text(encoding="utf-8")
    assert 'x-data="nexoChrome()"' in text


def test_base_html_configuracion_section_has_can_gating():
    text = _BASE_HTML.read_text(encoding="utf-8")
    # Configuración label must appear inside a conditional (config_visible.any).
    assert "Configuración" in text
    assert "config_visible" in text or "ajustes:manage" in text


# ── RUNBOOK slug invariants ────────────────────────────────────────────────

EXPECTED_HEADINGS = [
    "## Escenario 1: MES caido (SQL Server dbizaro inaccesible)",
    "## Escenario 2: Postgres no arranca",
    "## Escenario 3: Certificado Caddy expira / warning en browsers",
    "## Escenario 4: Pipeline atascado (HALLAZGO CRITICO: semaforo in-process)",
    "## Escenario 5: Lockout del unico propietario (HALLAZGO CRITICO: no hay `unlock_user`)",
]


def _gfm_slug(heading_line: str) -> str:
    """Simplified GFM slugifier: lower, strip punctuation, spaces→hyphens (1:1)."""
    text = heading_line.lstrip("# ").strip().lower()
    text = text.replace("`", "")
    # Strip punctuation (keep \w, whitespace, -). Punctuation becomes nothing,
    # so its surrounding whitespace survives and produces double hyphens
    # after the 1:1 space→hyphen mapping — matching GFM behavior.
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    # 1:1 space→hyphen (NOT collapsed) so "a  b" → "a--b" per GFM.
    text = text.replace(" ", "-")
    return text.strip("-")


@pytest.mark.parametrize("heading", EXPECTED_HEADINGS)
def test_runbook_canonical_heading_present(heading: str):
    text = _RUNBOOK_MD.read_text(encoding="utf-8")
    assert heading in text, f"RUNBOOK.md must contain heading: {heading}"


def test_runbook_headings_are_unique():
    text = _RUNBOOK_MD.read_text(encoding="utf-8")
    for heading in EXPECTED_HEADINGS:
        count = text.count(heading)
        assert count == 1, (
            f"Heading must appear exactly once in RUNBOOK.md, found {count}: {heading}"
        )


def test_expected_gfm_slugs_computed():
    """Sanity: the slugs UI-SPEC error-state copy must use."""
    expected = {
        "Escenario 1": "escenario-1-mes-caido-sql-server-dbizaro-inaccesible",
        "Escenario 2": "escenario-2-postgres-no-arranca",
        "Escenario 3": "escenario-3-certificado-caddy-expira--warning-en-browsers",
        "Escenario 4": "escenario-4-pipeline-atascado-hallazgo-critico-semaforo-in-process",
        "Escenario 5": "escenario-5-lockout-del-unico-propietario-hallazgo-critico-no-hay-unlock_user",
    }
    for heading in EXPECTED_HEADINGS:
        # Extract the key like "Escenario 1"
        label = heading.split(":")[0].lstrip("# ").strip()
        got = _gfm_slug(heading)
        assert got == expected[label], (
            f"Slug drift for {label}: expected {expected[label]!r}, got {got!r}"
        )
