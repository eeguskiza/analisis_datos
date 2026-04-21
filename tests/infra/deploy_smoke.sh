#!/usr/bin/env bash
# tests/infra/deploy_smoke.sh — Checklist post-deploy Phase 6.
#
# Se ejecuta en el servidor Ubuntu tras `bash scripts/deploy.sh`, o desde
# un peer LAN despues de configurar hosts-file + root CA (seccion 11 del
# runbook docs/DEPLOY_LAN.md). NO corre en CI ni en pytest — requiere:
#
#   - Docker + compose plugin disponibles.
#   - Stack prod levantado (make prod-up).
#   - ufw configurado (seccion 5 del runbook).
#   - Acceso sudo (ss y ufw requieren privilegios).
#
# Cada check imprime una linea con formato:
#
#   [DEPLOY-XX] OK: <mensaje>
#   [DEPLOY-XX] FAIL: <mensaje> (cmd: <comando>)
#
# Exit 0 si todos los checks pasan. Exit N (count de fallos) en caso
# contrario, para que cron/CI pueda usar el exit code como senal.

set -uo pipefail
IFS=$'\n\t'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

COMPOSE="docker compose -f docker-compose.yml -f docker-compose.prod.yml"
FAILS=0

check() {
    local req="$1"
    local msg="$2"
    local cmd="$3"
    if eval "${cmd}" >/dev/null 2>&1; then
        echo "[${req}] OK: ${msg}"
    else
        echo "[${req}] FAIL: ${msg} (cmd: ${cmd})"
        FAILS=$((FAILS + 1))
    fi
}

echo "=== Nexo deploy smoke — $(date -u +%FT%TZ) ==="
echo "project=${PROJECT_DIR}"

# DEPLOY-01: HTTPS via Caddy responde 200 en /api/health
# -k (insecure): el servidor no necesariamente tiene la root CA instalada
# en su trust store (la CA es la que Caddy acaba de emitir). El smoke valida
# "Caddy + web operativos", no "cadena TLS valida desde el servidor".
check "DEPLOY-01" "HTTPS /api/health responde 200" \
    "curl -fs -m 10 -k https://nexo.ecsmobility.local/api/health"

# DEPLOY-01: cert emitido por Caddy local CA
check "DEPLOY-01" "cert emitido por Caddy Local Authority" \
    "openssl s_client -connect nexo.ecsmobility.local:443 -servername nexo.ecsmobility.local </dev/null 2>/dev/null | grep -q 'Caddy Local Authority'"

# DEPLOY-02: Postgres 5432 NO escucha en el host
check "DEPLOY-02" "Postgres 5432 NO escucha en el host" \
    "! sudo ss -tlnp 2>/dev/null | grep -q ':5432 '"

# DEPLOY-02: psql via exec sigue funcionando (red interna compose)
check "DEPLOY-02" "psql via docker compose exec OK" \
    "${COMPOSE} exec -T db bash -c 'psql -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" -c \"SELECT 1;\"' | grep -q '1 row'"

# DEPLOY-03: web healthcheck healthy
check "DEPLOY-03" "web container healthcheck healthy" \
    "[ \"\$(docker inspect \$(${COMPOSE} ps -q web) --format '{{.State.Health.Status}}' 2>/dev/null)\" = 'healthy' ]"

# DEPLOY-03: caddy healthcheck healthy
check "DEPLOY-03" "caddy container healthcheck healthy" \
    "[ \"\$(docker inspect \$(${COMPOSE} ps -q caddy) --format '{{.State.Health.Status}}' 2>/dev/null)\" = 'healthy' ]"

# DEPLOY-03: restart policy unless-stopped (sobrevive reboots)
check "DEPLOY-03" "web restart policy = unless-stopped" \
    "[ \"\$(docker inspect \$(${COMPOSE} ps -q web) --format '{{.HostConfig.RestartPolicy.Name}}' 2>/dev/null)\" = 'unless-stopped' ]"

# DEPLOY-06: ufw activo
check "DEPLOY-06" "ufw activo" \
    "sudo ufw status 2>/dev/null | head -1 | grep -q 'Status: active'"

# DEPLOY-06: ufw permite 443/tcp
check "DEPLOY-06" "ufw permite 443/tcp" \
    "sudo ufw status 2>/dev/null | grep -qE '^443/tcp.*ALLOW'"

# DEPLOY-06: ufw permite 80/tcp (para redirect)
check "DEPLOY-06" "ufw permite 80/tcp (redirect HTTP->HTTPS)" \
    "sudo ufw status 2>/dev/null | grep -qE '^80/tcp.*ALLOW'"

# DEPLOY-06: default incoming = deny
check "DEPLOY-06" "ufw default incoming = deny" \
    "sudo ufw status verbose 2>/dev/null | grep -qE 'Default:.*deny \\(incoming\\)'"

echo "=== Fallos: ${FAILS} ==="
exit "${FAILS}"
