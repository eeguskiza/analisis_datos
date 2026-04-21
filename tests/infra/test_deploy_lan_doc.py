"""Validaciones estaticas de docs/DEPLOY_LAN.md y tests/infra/deploy_smoke.sh.

Previene regresiones del runbook: si alguien borra accidentalmente una seccion
critica (ufw, root CA, landmine down -v), pytest falla y lo detecta antes del
merge.

Cubre Phase 6 / Plan 06-03:
  - DEPLOY-05: runbook para admin de respaldo (contenido minimo, placeholders).
  - DEPLOY-06: reglas ufw documentadas + advertencia Docker bypass.
  - DEPLOY-07: smoke script scripted para peer LAN.
"""
import re
import shutil
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "docs" / "DEPLOY_LAN.md"
SMOKE = REPO_ROOT / "tests" / "infra" / "deploy_smoke.sh"


# ---- DEPLOY_LAN.md: estructura y contenido critico ---------------------------

def test_doc_exists():
    assert DOC.exists(), f"Falta {DOC} (DEPLOY-05)."


def test_doc_has_min_length():
    lines = DOC.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 300, (
        f"DEPLOY_LAN.md tiene {len(lines)} lineas; minimo 300 para cubrir "
        "las 16 secciones requeridas."
    )


def test_doc_has_required_sections():
    content = DOC.read_text(encoding="utf-8")
    # Secciones obligatorias (por keywords en headings — tolerante a numeracion)
    required = [
        "Docker",          # instalacion
        "hosts",           # hosts-file por SO
        "ufw",             # firewall
        "backup",          # cron y restore
        "Landmines",       # avisos criticos
    ]
    missing = [kw for kw in required if kw.lower() not in content.lower()]
    assert not missing, f"DEPLOY_LAN.md sin secciones: {missing}"


def test_doc_has_ufw_rules():
    content = DOC.read_text(encoding="utf-8")
    # DEPLOY-06: reglas ufw explicitas
    assert "ufw allow 443/tcp" in content, "Regla ufw 443 ausente."
    assert "ufw allow 80/tcp" in content, "Regla ufw 80 ausente."
    assert "ufw allow from" in content, "Regla ufw SSH subnet ausente."
    assert "ufw default deny incoming" in content, "Default deny ausente."


def test_doc_warns_docker_bypasses_ufw():
    """Landmine 3: critico que el operador entienda que Docker NO pasa por ufw."""
    content = DOC.read_text(encoding="utf-8")
    assert "bypass" in content.lower() or "DOCKER-USER" in content or (
        "Docker" in content and "ufw" in content and "!reset" in content
    ), (
        "El runbook debe advertir que Docker bypassa ufw (Landmine 3)."
    )


def test_doc_warns_down_v_landmine():
    """Landmine 6: docker compose down -v borra pgdata."""
    content = DOC.read_text(encoding="utf-8")
    assert "down -v" in content, (
        "El runbook debe mencionar el landmine `down -v` (Landmine 6)."
    )


def test_doc_mentions_root_ca_distribution():
    content = DOC.read_text(encoding="utf-8")
    assert "root.crt" in content, "Path del root CA debe estar documentado."
    # Windows: certutil o certmgr; Linux: update-ca-certificates; macOS: security add-trusted
    os_steps = (
        "certutil" in content or "certmgr" in content,
        "update-ca-certificates" in content,
        "add-trusted-cert" in content or "keychain" in content.lower(),
    )
    assert sum(os_steps) >= 2, (
        "Distribucion root CA debe cubrir al menos 2 SOs (Windows/Linux/macOS)."
    )


def test_doc_mentions_hosts_file_paths():
    content = DOC.read_text(encoding="utf-8")
    assert "/etc/hosts" in content, "Path /etc/hosts ausente."
    assert "drivers\\etc\\hosts" in content or "System32" in content, (
        "Path Windows hosts-file ausente."
    )


def test_doc_references_scripts():
    content = DOC.read_text(encoding="utf-8")
    assert "scripts/deploy.sh" in content, "Referencia a scripts/deploy.sh ausente."
    assert "scripts/backup_nightly.sh" in content, (
        "Referencia a scripts/backup_nightly.sh ausente."
    )


def test_doc_references_compose_files():
    content = DOC.read_text(encoding="utf-8")
    assert "docker-compose.prod.yml" in content, (
        "Referencia al override prod ausente."
    )
    assert ".env.prod.example" in content, "Referencia al template env ausente."


def test_doc_has_restore_section():
    content = DOC.read_text(encoding="utf-8")
    # Restore debe cubrir zcat | psql + DROP SCHEMA para evitar conflictos
    assert "zcat" in content, "Comando zcat para restore ausente."
    assert "DROP SCHEMA IF EXISTS nexo" in content, (
        "Paso DROP SCHEMA en restore ausente (evita conflictos)."
    )


def test_doc_has_cron_entry():
    content = DOC.read_text(encoding="utf-8")
    assert "/etc/cron.d/nexo-backup" in content, "Cron entry ausente."
    assert "0 3 * * *" in content, "Schedule cron 03:00 UTC ausente (D-07)."


def test_doc_has_placeholders():
    content = DOC.read_text(encoding="utf-8")
    # D-03 y D-14: placeholders explicitos
    assert "<IP_NEXO>" in content, "Placeholder <IP_NEXO> ausente (D-03)."
    assert "<SUBNET_LAN>" in content, "Placeholder <SUBNET_LAN> ausente (D-14)."
    assert "<ADMIN_BACKUP_NAME>" in content, (
        "Placeholder <ADMIN_BACKUP_NAME> ausente (D-12, bus factor 2)."
    )


def test_doc_no_emoji():
    content = DOC.read_text(encoding="utf-8")
    emoji_re = re.compile(
        r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F600-\U0001F64F]"
    )
    m = emoji_re.search(content)
    assert m is None, f"Emoji prohibido en DEPLOY_LAN.md: {m.group(0)!r}"


def test_doc_no_internet_exposure():
    """LAN-only es decision cerrada. El doc NO debe sugerir Let's Encrypt publico."""
    content = DOC.read_text(encoding="utf-8").lower()
    # Let's Encrypt DNS-01 descartado en CONTEXT. Si aparece la palabra,
    # debe estar en seccion "mejoras futuras" o "descartado".
    if "let's encrypt" in content or "letsencrypt" in content:
        assert any(marker in content for marker in [
            "descartad", "deferred", "out of scope", "out-of-scope",
            "dns-01 descartado",
        ]), (
            "Menciones de Let's Encrypt deben indicarse como descartado/deferred."
        )


def test_doc_references_future_dns_improvement():
    """D-02: migracion a DNS interno documentada como mejora futura."""
    content = DOC.read_text(encoding="utf-8").lower()
    assert "dns interno" in content or "dns corporativo" in content, (
        "Mejora futura 'DNS interno' ausente (seccion 15)."
    )


def test_doc_has_rto_target():
    """D-11: RTO 1-2h declarado explicitamente."""
    content = DOC.read_text(encoding="utf-8")
    assert "1-2" in content and ("RTO" in content or "hora" in content.lower()), (
        "RTO 1-2h debe estar documentado (D-11)."
    )


# ---- deploy_smoke.sh: sintaxis y cobertura -----------------------------------

def test_smoke_sh_exists():
    assert SMOKE.exists(), f"Falta {SMOKE}"


def test_smoke_sh_is_executable():
    assert SMOKE.stat().st_mode & stat.S_IXUSR, "deploy_smoke.sh debe ser ejecutable."


def test_smoke_sh_bash_syntax_valid():
    if shutil.which("bash") is None:
        import pytest
        pytest.skip("bash no disponible")
    r = subprocess.run(
        ["bash", "-n", str(SMOKE)],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert r.returncode == 0, r.stderr


def test_smoke_sh_covers_deploy_requirements():
    content = SMOKE.read_text(encoding="utf-8")
    for req in ("DEPLOY-01", "DEPLOY-02", "DEPLOY-03", "DEPLOY-06"):
        assert req in content, f"smoke.sh no cubre {req}"


def test_smoke_sh_has_exit_count():
    """El smoke debe exit con el contador de fallos para integrarse con CI/cron."""
    content = SMOKE.read_text(encoding="utf-8")
    assert "FAILS" in content or "FAIL_COUNT" in content, (
        "smoke.sh debe tener contador de fallos."
    )
    assert "exit" in content, "smoke.sh debe hacer exit explicito con el contador."


def test_smoke_sh_no_emoji():
    """Coherencia con CLAUDE.md + rules/common/coding-style.md."""
    content = SMOKE.read_text(encoding="utf-8")
    emoji_re = re.compile(
        r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F600-\U0001F64F]"
    )
    m = emoji_re.search(content)
    assert m is None, f"Emoji prohibido en deploy_smoke.sh: {m.group(0)!r}"


def test_smoke_sh_uses_set_safety_flags():
    """Buenas practicas bash operacional."""
    content = SMOKE.read_text(encoding="utf-8")
    # Usamos `set -uo pipefail` (no `-e` para que el script siga tras fallos
    # individuales y pueda contar FAILS). `-u` atrapa vars sin definir.
    assert "set -" in content and "pipefail" in content, (
        "deploy_smoke.sh debe usar set -u + pipefail minimo."
    )
