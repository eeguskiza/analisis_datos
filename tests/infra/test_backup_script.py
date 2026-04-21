"""Validaciones estaticas de scripts/backup_nightly.sh (Phase 6 / Plan 06-02).

D-07, D-08, D-09, D-10 + Pitfall 4 + Landmine 6/7.
"""
import shutil
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKUP_SH = REPO_ROOT / "scripts" / "backup_nightly.sh"


def test_backup_sh_exists():
    assert BACKUP_SH.exists()


def test_backup_sh_is_executable():
    assert BACKUP_SH.stat().st_mode & stat.S_IXUSR


def test_backup_sh_bash_syntax_valid():
    if shutil.which("bash") is None:
        import pytest
        pytest.skip("bash no disponible")
    r = subprocess.run(["bash", "-n", str(BACKUP_SH)],
                       capture_output=True, text=True, timeout=10, check=False)
    assert r.returncode == 0, r.stderr


def test_backup_sh_has_set_euo_pipefail():
    assert "set -euo pipefail" in BACKUP_SH.read_text(encoding="utf-8")


def test_backup_sh_target_dir():
    assert "/var/backups/nexo" in BACKUP_SH.read_text(encoding="utf-8")


def test_backup_sh_pg_dump_via_compose():
    content = BACKUP_SH.read_text(encoding="utf-8")
    assert "pg_dump" in content
    assert "docker compose" in content or "compose exec" in content


def test_backup_sh_atomic_rename():
    content = BACKUP_SH.read_text(encoding="utf-8")
    assert ".tmp" in content and "mv " in content, "Pitfall 4: .tmp + mv"


def test_backup_sh_retention_7d():
    content = BACKUP_SH.read_text(encoding="utf-8")
    assert "-mtime +7" in content and "-delete" in content, "D-09"


def test_backup_sh_chmod_600():
    content = BACKUP_SH.read_text(encoding="utf-8")
    assert "chmod 600" in content or "umask" in content


def test_backup_sh_uses_prod_override():
    assert "docker-compose.prod.yml" in BACKUP_SH.read_text(encoding="utf-8")


def test_backup_sh_no_rm_rf():
    assert "rm -rf" not in BACKUP_SH.read_text(encoding="utf-8"), "Landmine 6"
