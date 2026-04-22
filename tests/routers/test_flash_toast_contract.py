"""Regression for Phase 8 / Pitfall 3: showToast 3-arg contract.

Locks:
1. base.html flash consumer calls showToast('info', 'Aviso', msg) — 3-arg.
2. app.js defines ONE window.showToast with the 3-arg signature.
3. No remaining 2-arg callers in static/js or templates — every legacy
   caller was rewritten in Plan 08-02.
4. No duplicate showToast definition in base.html (the old inline one
   was removed in favour of the single definition in app.js).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_BASE_HTML = _ROOT / "templates" / "base.html"
_APP_JS = _ROOT / "static" / "js" / "app.js"
_TEMPLATES_DIR = _ROOT / "templates"


def test_base_html_has_no_inline_showtoast_definition():
    text = _BASE_HTML.read_text(encoding="utf-8")
    # The DEFINITION is gone — only the call site for flash_message remains.
    matches = re.findall(r"window\.showToast\s*=\s*function", text)
    assert matches == [], (
        "base.html must not define window.showToast — the single definition "
        "lives in /static/js/app.js per D-30"
    )


def test_base_html_flash_block_uses_three_arg_call():
    text = _BASE_HTML.read_text(encoding="utf-8")
    # The flash consumer MUST pass (type, title, msg)
    # Regex tolerates the Jinja {{ flash_message|tojson }} expression.
    pattern = re.compile(
        r"window\.showToast\(\s*'info'\s*,\s*'Aviso'\s*,\s*\{\{\s*flash_message\|tojson\s*\}\}\s*\)"
    )
    assert pattern.search(text) is not None, (
        "Jinja flash consumer must call window.showToast('info', 'Aviso', {{ flash_message|tojson }})"
    )


def test_app_js_defines_three_arg_show_toast():
    text = _APP_JS.read_text(encoding="utf-8")
    # Canonical definition must be present.
    assert re.search(
        r"window\.showToast\s*=\s*function\s*\(\s*type\s*,\s*title\s*,\s*msg\s*\)",
        text,
    ), "app.js must declare `window.showToast = function (type, title, msg)`"
    # Old 2-arg function declaration must be gone.
    assert (
        re.search(
            r"^\s*function\s+showToast\s*\(\s*message\s*,\s*type",
            text,
            re.MULTILINE,
        )
        is None
    ), "app.js must not contain the legacy 2-arg showToast function"


LEGACY_2_ARG_RE = re.compile(
    # showToast('foo', 'error')  /  showToast(`foo`, 'error')
    # EXCLUDES 3-arg: showToast('error', 'Title', 'msg')
    r"\bshowToast\(\s*(['\"`])([^'\"`]+)\1\s*,\s*(['\"])(error|info|success|warn)\3\s*\)"
)


@pytest.mark.parametrize(
    "template_name",
    [
        "historial.html",
        "informes.html",
        "ciclos.html",
        "ciclos_calc.html",
        "plantillas.html",
    ],
)
def test_no_legacy_two_arg_callers_in_template(template_name: str):
    text = (_TEMPLATES_DIR / template_name).read_text(encoding="utf-8")
    matches = LEGACY_2_ARG_RE.findall(text)
    assert matches == [], f"{template_name} still has 2-arg showToast callers: {matches}"


def test_no_legacy_two_arg_callers_in_app_js():
    text = _APP_JS.read_text(encoding="utf-8")
    matches = LEGACY_2_ARG_RE.findall(text)
    assert matches == [], f"app.js still has 2-arg showToast callers: {matches}"
