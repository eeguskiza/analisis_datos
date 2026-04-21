"""Validaciones estaticas de docker-compose.prod.yml (Phase 6 / Plan 06-01).

Dos niveles de test:
1. Parser directo del archivo override (sin Docker) — comprueba presencia de
   `!reset`, `!override`, healthchecks, resource limits.
2. `docker compose config` merged (si Docker disponible) — comprueba que la
   fusion con el base produce ports vacios, volumes correctos (./tests fuera),
   healthchecks prod. Skip si docker no esta instalado.

Nota sobre etiquetas YAML Compose v2.24+:
  - `ports: !reset []`  deja la lista vacia tras el merge (usado en db y web).
  - `volumes: !override <lista>`  reemplaza la lista heredada del base
    (usado en web y caddy). `volumes: !reset` seguido de una lista es un
    BUG — deja la lista vacia (Landmine 2b).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Final

import pytest

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
OVERRIDE: Final[Path] = REPO_ROOT / "docker-compose.prod.yml"
BASE: Final[Path] = REPO_ROOT / "docker-compose.yml"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True, timeout=10, check=False,
        )
        return r.returncode == 0
    except Exception:
        return False


def _compose_config_json() -> dict:
    """Run `docker compose config --format json` and return parsed dict."""
    r = subprocess.run(
        ["docker", "compose",
         "-f", str(BASE),
         "-f", str(OVERRIDE),
         "config", "--format", "json"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert r.returncode == 0, f"compose config fallo: {r.stderr}"
    return json.loads(r.stdout)


# Nivel 1: parse directo del archivo override (sin Docker)

def test_override_exists() -> None:
    assert OVERRIDE.exists(), f"No existe {OVERRIDE}"


def test_override_mentions_reset_for_db_ports() -> None:
    content = OVERRIDE.read_text(encoding="utf-8")
    # Landmine 2: ports: [] NO resetea. Debe aparecer `!reset` junto a ports en db.
    assert "!reset" in content, (
        "Override debe usar `!reset` para borrar ports heredados (Landmine 2)."
    )
    # Verificacion mas estricta: el texto entre 'db:' y el siguiente service
    # debe contener 'ports: !reset'.
    db_block_start = content.find("  db:")
    assert db_block_start >= 0, "Falta service db: en el override."
    db_block = content[db_block_start:]
    next_service = db_block.find("\n  web:")
    if next_service > 0:
        db_block = db_block[:next_service]
    assert "ports: !reset" in db_block, (
        "`db.ports: !reset []` es obligatorio para cerrar 5432 al host (DEPLOY-02)."
    )


def test_override_uses_override_tag_for_volumes() -> None:
    """Landmine 2b: `volumes: !reset` seguido de una lista deja la lista vacia.

    Para reemplazar la lista heredada del base hay que usar `!override`, no
    `!reset`. Este test falla si alguien vuelve a usar `!reset` en volumes.
    """
    content = OVERRIDE.read_text(encoding="utf-8")
    assert "!override" in content, (
        "Override debe usar `!override` para reemplazar listas de volumes "
        "(Landmine 2b). `!reset` sobre volumes con lista debajo deja la lista "
        "vacia silenciosamente."
    )
    # Confirmamos que web.volumes y caddy.volumes usan !override (no !reset).
    for service in ("web", "caddy"):
        marker = f"  {service}:"
        idx = content.find(marker)
        assert idx >= 0, f"Falta service {service}: en el override."
        block = content[idx:]
        # Acotar al siguiente service de top-level (2 espacios indent).
        for nxt in ("\n  db:", "\n  web:", "\n  caddy:"):
            if nxt == f"\n  {service}:":
                continue
            pos = block.find(nxt)
            if pos > 0:
                block = block[:pos]
        assert "volumes: !override" in block, (
            f"{service}.volumes debe usar `!override` (no `!reset`) para "
            "que el merge conserve los elementos de la lista (Landmine 2b)."
        )


def test_override_web_healthcheck_curl_api_health() -> None:
    content = OVERRIDE.read_text(encoding="utf-8")
    assert "curl" in content and "/api/health" in content, (
        "web healthcheck debe usar curl contra /api/health (D-22)."
    )
    # Pitfall 3: NO anadir jq ni parsing al healthcheck — solo curl -fs.
    assert "jq" not in content, (
        "healthcheck NO debe parsear JSON con jq — MES caido marcaria unhealthy "
        "falsamente. Dejar `curl -fs` puro (Pitfall 3)."
    )


def test_override_caddy_healthcheck_wget_spider() -> None:
    content = OVERRIDE.read_text(encoding="utf-8")
    assert "wget" in content and "--spider" in content, (
        "caddy healthcheck debe usar wget --spider (D-23)."
    )


def test_override_has_resource_limits() -> None:
    content = OVERRIDE.read_text(encoding="utf-8")
    # D-21: web 4g/2cpu, db 2g/1cpu, caddy 256m/0.5cpu
    assert "memory: 4g" in content, "web debe tener memory: 4g (D-21)."
    assert "memory: 2g" in content, "db debe tener memory: 2g (D-21)."
    assert "memory: 256m" in content, "caddy debe tener memory: 256m (D-21)."


def test_override_caddyfile_prod_bind_mount() -> None:
    content = OVERRIDE.read_text(encoding="utf-8")
    assert "./caddy/Caddyfile.prod:/etc/caddy/Caddyfile:ro" in content, (
        "caddy debe montar Caddyfile.prod en lugar del dev (D-19)."
    )


# Nivel 2: merge con base via `docker compose config` (opcional)

@pytest.mark.skipif(not _docker_available(), reason="docker compose no disponible")
def test_compose_config_merges_without_error() -> None:
    """Valida que la fusion del base + override es sintacticamente valida."""
    cfg = _compose_config_json()
    assert "services" in cfg


@pytest.mark.skipif(not _docker_available(), reason="docker compose no disponible")
def test_compose_config_db_ports_empty_after_merge() -> None:
    """DEPLOY-02: tras el merge, db.ports debe ser vacia (None o [])."""
    cfg = _compose_config_json()
    # Compose emite la clave `ports` como null (o la omite) cuando `!reset []`
    # resuelve a lista vacia. Ambas formas son "sin ports publicados".
    db_ports = cfg.get("services", {}).get("db", {}).get("ports")
    assert db_ports in (None, []), (
        f"db.ports tras merge debe estar vacio (None o []), pero es {db_ports!r}. "
        "Landmine 2: probable que falte `!reset`."
    )


@pytest.mark.skipif(not _docker_available(), reason="docker compose no disponible")
def test_compose_config_web_ports_empty_after_merge() -> None:
    cfg = _compose_config_json()
    web_ports = cfg.get("services", {}).get("web", {}).get("ports")
    assert web_ports in (None, []), (
        f"web.ports tras merge debe estar vacio (None o []), pero es {web_ports!r}."
    )


@pytest.mark.skipif(not _docker_available(), reason="docker compose no disponible")
def test_compose_config_web_volumes_exclude_tests() -> None:
    """Landmine 2b: ./tests del dev NO debe montarse en prod.

    Si este test falla con 3 volumes (data, informes, tests), es senal de que
    `volumes: !override` se perdio y Compose volvio a fusionar listas.
    """
    cfg = _compose_config_json()
    web_vols = cfg.get("services", {}).get("web", {}).get("volumes") or []
    targets = [v.get("target") for v in web_vols if isinstance(v, dict)]
    assert "/app/tests" not in targets, (
        f"web.volumes NO debe incluir /app/tests en prod. Merged volumes: {targets}. "
        "Posible regresion del `!override` a `!reset`."
    )
    # Los dos que si queremos.
    assert "/app/data" in targets, f"Falta /app/data en web.volumes: {targets}"
    assert "/app/informes" in targets, f"Falta /app/informes en web.volumes: {targets}"


@pytest.mark.skipif(not _docker_available(), reason="docker compose no disponible")
def test_compose_config_caddy_mounts_caddyfile_prod() -> None:
    """D-19: caddy debe montar Caddyfile.prod (NO el dev) tras el merge."""
    cfg = _compose_config_json()
    caddy_vols = cfg.get("services", {}).get("caddy", {}).get("volumes") or []
    sources = [v.get("source", "") for v in caddy_vols if isinstance(v, dict)]
    assert any(src.endswith("/caddy/Caddyfile.prod") for src in sources), (
        f"caddy debe bind-mount ./caddy/Caddyfile.prod. Sources: {sources}"
    )
    # Y el dev NO debe quedarse:
    assert not any(src.endswith("/caddy/Caddyfile") and not src.endswith(".prod")
                   for src in sources), (
        f"caddy NO debe mantener el Caddyfile dev. Sources: {sources}"
    )
