"""Regression tests for Phase 8 tokens.css + tailwind.config.js.

Static parse — no runtime deps. Asserts:
- tokens.css declares every required semantic CSS var.
- Color tokens use space-separated RGB format (no commas, no rgb() wrapper).
- tailwind.config.js references the declared vars via the
  rgb(var(--token) / <alpha-value>) pattern.
- docs/BRANDING.md has the Phase 8 Tokens section.

Pitfall 1/2 from 08-RESEARCH.md: token drift or format regressions
break the entire Tailwind pipeline silently. Catch at CI boundary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_TOKENS_CSS = _ROOT / "static" / "css" / "tokens.css"
_TAILWIND_JS = _ROOT / "static" / "js" / "tailwind.config.js"
_BRANDING_MD = _ROOT / "docs" / "BRANDING.md"
_APP_CSS = _ROOT / "static" / "css" / "app.css"


REQUIRED_SEMANTIC_COLOR_TOKENS = [
    "--color-surface-base",
    "--color-surface-app",
    "--color-surface-subtle",
    "--color-surface-muted",
    "--color-text-body",
    "--color-text-heading",
    "--color-text-muted",
    "--color-text-disabled",
    "--color-text-on-accent",
    "--color-border-subtle",
    "--color-border-strong",
    "--color-primary",
    "--color-primary-hover",
    "--color-primary-active",
    "--color-primary-subtle",
    "--color-focus-ring",
    "--color-success",
    "--color-success-subtle",
    "--color-warn",
    "--color-warn-subtle",
    "--color-error",
    "--color-error-subtle",
    "--color-info",
    "--color-info-subtle",
]

REQUIRED_RADIUS_TOKENS = [
    "--radius-sm",
    "--radius-md",
    "--radius-lg",
    "--radius-xl",
    "--radius-pill",
]

REQUIRED_SHADOW_TOKENS = [
    "--shadow-card",
    "--shadow-popover",
    "--shadow-modal",
    "--shadow-drawer",
]

REQUIRED_MOTION_TOKENS = [
    "--duration-fast",
    "--duration-base",
    "--duration-slow",
    "--ease-standard",
    "--ease-emphasized",
    "--ease-accelerate",
]

REQUIRED_Z_TOKENS = [
    "--z-topbar",
    "--z-backdrop",
    "--z-drawer",
    "--z-modal",
    "--z-popover",
    "--z-toast",
]


@pytest.fixture(scope="module")
def tokens_css_text() -> str:
    assert _TOKENS_CSS.exists(), f"Missing {_TOKENS_CSS}"
    return _TOKENS_CSS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tailwind_js_text() -> str:
    assert _TAILWIND_JS.exists(), f"Missing {_TAILWIND_JS}"
    return _TAILWIND_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def branding_md_text() -> str:
    assert _BRANDING_MD.exists(), f"Missing {_BRANDING_MD}"
    return _BRANDING_MD.read_text(encoding="utf-8")


@pytest.mark.parametrize("token", REQUIRED_SEMANTIC_COLOR_TOKENS)
def test_tokens_file_declares_semantic_color(token: str, tokens_css_text: str):
    assert f"{token}:" in tokens_css_text, f"tokens.css must declare {token}"


@pytest.mark.parametrize(
    "token",
    REQUIRED_RADIUS_TOKENS + REQUIRED_SHADOW_TOKENS + REQUIRED_MOTION_TOKENS + REQUIRED_Z_TOKENS,
)
def test_tokens_file_declares_layout_tokens(token: str, tokens_css_text: str):
    assert f"{token}:" in tokens_css_text, f"tokens.css must declare {token}"


def test_color_tokens_do_not_wrap_rgb(tokens_css_text: str):
    """--color-* tokens must be raw triples — Pitfall 2."""
    in_root = False
    for line in tokens_css_text.splitlines():
        if line.strip().startswith(":root"):
            in_root = True
            continue
        if in_root and line.strip() == "}":
            break
        if in_root and line.lstrip().startswith("--color-"):
            # --color-foo: value; OR --color-foo: var(--color-bar);
            _, _, raw = line.partition(":")
            value = raw.strip().rstrip(";").strip()
            assert not value.startswith("rgb("), f"Color token wraps rgb(): {line.strip()}"
            assert "," not in value or value.startswith("var("), (
                f"Color token uses commas (expect space-separated triples): {line.strip()}"
            )


def test_reduced_motion_block_present(tokens_css_text: str):
    assert "@media (prefers-reduced-motion: reduce)" in tokens_css_text


def test_app_css_imports_tokens():
    text = _APP_CSS.read_text(encoding="utf-8")
    first_line = text.splitlines()[0]
    assert first_line.strip() == "@import url('./tokens.css');", (
        f"app.css first line must import tokens.css, got: {first_line!r}"
    )


@pytest.mark.parametrize("token", REQUIRED_SEMANTIC_COLOR_TOKENS[:6])
def test_tailwind_config_references_tokens(token: str, tailwind_js_text: str):
    pattern = f"var({token})"
    assert pattern in tailwind_js_text, f"tailwind.config.js must reference {token}"


def test_tailwind_uses_alpha_value_pattern(tailwind_js_text: str):
    # At least one use of the Tailwind alpha pattern (sanity).
    assert "/ <alpha-value>" in tailwind_js_text


def test_branding_md_has_tokens_section(branding_md_text: str):
    assert "## Tokens (Phase 8)" in branding_md_text
    assert "60 / 30 / 10" in branding_md_text
    assert "`--color-primary`" in branding_md_text
