#!/usr/bin/env bash
# scripts/deploy.sh — Deploy idempotente Nexo (D-25, D-26)
# Uso: bash scripts/deploy.sh
#
# Pasos: git pull --ff-only -> pre-deploy backup -> build --pull -> up -d -> smoke.
#
# Exit codes:
#   0  = deploy OK (smoke pasa)
#   1  = cualquier step falla
#
# NO hay rollback automatico — el log incluye hint; el operador decide.

set -euo pipefail
IFS=$'\n\t'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

COMPOSE="docker compose -f docker-compose.yml -f docker-compose.prod.yml"
PREDEPLOY_DIR="/var/backups/nexo/predeploy"
LOG_PREFIX="[deploy $(date -u +%FT%TZ)]"

log() { echo "${LOG_PREFIX} $*"; }
fail() { echo "${LOG_PREFIX} ERROR: $*" >&2; exit 1; }

log "start"
log "project=${PROJECT_DIR} branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD)"

# 1. git pull --ff-only (sin force)
log "step 1/5: git pull --ff-only"
git pull --ff-only || fail "git pull fallo (posible divergencia local). Resolver manualmente con 'git status'. No commitear en el servidor."

# 2. Pre-deploy backup (skip si db no corre aun)
log "step 2/5: pre-deploy backup"
if $COMPOSE ps db --format json 2>/dev/null | grep -q '"State":"running"'; then
    mkdir -p "${PREDEPLOY_DIR}"
    HASH="$(git rev-parse --short HEAD)"
    STAMP="$(date -u +%Y%m%d-%H%M)"
    OUT="${PREDEPLOY_DIR}/predeploy-${HASH}-${STAMP}.sql.gz"
    $COMPOSE exec -T db bash -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
        | gzip > "${OUT}.tmp" || fail "pg_dump pre-deploy fallo"
    mv "${OUT}.tmp" "${OUT}"
    chmod 600 "${OUT}"
    log "  backup: ${OUT} ($(du -h "${OUT}" | cut -f1))"
    find "${PREDEPLOY_DIR}" -name "predeploy-*.sql.gz" -mtime +30 -delete 2>/dev/null || true
else
    log "  db no esta corriendo — primer deploy, skip backup"
fi

# 3. Build imagenes
log "step 3/5: build --pull"
$COMPOSE build --pull || fail "build fallo"

# 4. Up en background
log "step 4/5: up -d"
$COMPOSE up -d || fail "up -d fallo"

# 5. Smoke test — /api/health via Caddy HTTPS
log "step 5/5: smoke test"
# sleep 15: dar tiempo a start_period=20s de web y 10s de caddy.
# -k: TLS internal; el host de deploy no siempre tiene la root CA instalada.
# -m 10: timeout total 10s.
sleep 15
if curl -fs -m 10 -k https://nexo.ecsmobility.local/api/health >/dev/null; then
    log "smoke OK"
else
    fail "smoke fallo — rollback manual: git reset --hard HEAD~1 && bash scripts/deploy.sh"
fi

log "DONE"
