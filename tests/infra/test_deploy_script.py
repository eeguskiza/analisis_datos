"""Validaciones estaticas de scripts/deploy.sh (Phase 6 / Plan 06-02).

No ejecuta el script. Valida sintaxis bash, presencia de pasos D-25, y
ausencia de landmines (--force, --no-verify, emojis, rm -rf).
"""
import re
import shutil
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SH = REPO_ROOT / "scripts" / "deploy.sh"


def test_deploy_sh_exists():
    assert DEPLOY_SH.exists(), f"Falta {DEPLOY_SH} (DEPLOY-04)."


def test_deploy_sh_is_executable():
    mode = DEPLOY_SH.stat().st_mode
    assert mode & stat.S_IXUSR, "scripts/deploy.sh debe ser ejecutable (chmod +x)."


def test_deploy_sh_bash_syntax_valid():
    if shutil.which("bash") is None:
        import pytest
        pytest.skip("bash no disponible")
    r = subprocess.run(["bash", "-n", str(DEPLOY_SH)],
                       capture_output=True, text=True, timeout=10, check=False)
    assert r.returncode == 0, f"bash -n fallo: {r.stderr}"


def test_deploy_sh_has_set_euo_pipefail():
    assert "set -euo pipefail" in DEPLOY_SH.read_text(encoding="utf-8")


def test_deploy_sh_git_pull_ff_only():
    content = DEPLOY_SH.read_text(encoding="utf-8")
    assert "git pull --ff-only" in content, "D-25: git pull --ff-only obligatorio."
    assert "--force" not in content, "CLAUDE.md prohibe --force en git."


def test_deploy_sh_pg_dump_predeploy():
    content = DEPLOY_SH.read_text(encoding="utf-8")
    assert "pg_dump" in content, "D-25: pg_dump pre-deploy obligatorio."
    assert "predeploy" in content, "Backup debe taggearse con 'predeploy-'."


def test_deploy_sh_atomic_rename():
    content = DEPLOY_SH.read_text(encoding="utf-8")
    # Pitfall 4: .tmp + mv
    assert ".tmp" in content, "Pitfall 4: backup debe escribir a .tmp."
    assert "mv " in content, "Pitfall 4: mv .tmp final obligatorio."


def test_deploy_sh_build_and_up():
    content = DEPLOY_SH.read_text(encoding="utf-8")
    assert "build --pull" in content
    assert "up -d" in content


def test_deploy_sh_smoke_curl_https():
    content = DEPLOY_SH.read_text(encoding="utf-8")
    assert "curl" in content and "/api/health" in content
    assert "https://nexo.ecsmobility.local" in content


def test_deploy_sh_uses_compose_override():
    assert "docker-compose.prod.yml" in DEPLOY_SH.read_text(encoding="utf-8")


def test_deploy_sh_no_no_verify():
    assert "--no-verify" not in DEPLOY_SH.read_text(encoding="utf-8")


def test_deploy_sh_no_emoji():
    content = DEPLOY_SH.read_text(encoding="utf-8")
    emoji_re = re.compile(
        r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F600-\U0001F64F]"
    )
    m = emoji_re.search(content)
    assert m is None, f"Emoji prohibido: {m.group(0)!r}"


def test_deploy_sh_rollback_hint():
    assert "rollback" in DEPLOY_SH.read_text(encoding="utf-8").lower()
