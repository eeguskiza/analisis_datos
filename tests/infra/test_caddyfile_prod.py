"""Validaciones estaticas de caddy/Caddyfile.prod (Phase 6 / Plan 06-01).

No requiere Docker ni servidor Ubuntu real. Lee el archivo y comprueba
presencia/ausencia de literales criticos.
"""
from pathlib import Path

CADDYFILE_PROD = Path(__file__).resolve().parents[2] / "caddy" / "Caddyfile.prod"


def test_caddyfile_prod_exists():
    assert CADDYFILE_PROD.exists(), f"No existe {CADDYFILE_PROD}"


def test_caddyfile_prod_contains_hostname():
    content = CADDYFILE_PROD.read_text(encoding="utf-8")
    assert "nexo.ecsmobility.local" in content, (
        "Caddyfile.prod debe declarar el bloque nexo.ecsmobility.local (D-01, D-06)."
    )


def test_caddyfile_prod_uses_tls_internal():
    content = CADDYFILE_PROD.read_text(encoding="utf-8")
    assert "tls internal" in content, (
        "Caddyfile.prod debe usar `tls internal` (D-04)."
    )


def test_caddyfile_prod_no_auto_https_disable_redirects():
    """Landmine 1: copiar literal el Caddyfile dev rompe el redirect 80->443.

    El bloque global `auto_https disable_redirects` inhibe el listener :80
    redirect, contradiciendo D-16 / DEPLOY-06.
    """
    content = CADDYFILE_PROD.read_text(encoding="utf-8")
    assert "auto_https disable_redirects" not in content, (
        "Caddyfile.prod NO debe llevar `auto_https disable_redirects` — "
        "romperia el redirect automatico 80->443 (D-16, Landmine 1)."
    )


def test_caddyfile_prod_reverse_proxy_web_8000():
    content = CADDYFILE_PROD.read_text(encoding="utf-8")
    assert "reverse_proxy web:8000" in content, (
        "Caddyfile.prod debe reverse-proxy a web:8000 (red interna compose)."
    )
