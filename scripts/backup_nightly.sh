#!/usr/bin/env bash
# scripts/backup_nightly.sh — Backup nightly de Postgres Nexo.
# Invocado por cron (D-07: 03:00 UTC daily).
#
# Comportamiento: pg_dump plain SQL | gzip -> /var/backups/nexo/nexo-<stamp>.sql.gz
# con atomic rename (Pitfall 4) y rotacion -mtime +7 (D-09).
#
# Exit codes:
#   0 = backup OK
#   1 = pg_dump o gzip fallaron (archivo .tmp queda para debug)

set -euo pipefail
IFS=$'\n\t'

# /opt/nexo = convencion Linux ops (documentada en DEPLOY_LAN.md).
PROJECT_DIR="/opt/nexo"
BACKUP_DIR="/var/backups/nexo"
STAMP="$(date -u +%Y%m%d-%H%M)"
OUT="${BACKUP_DIR}/nexo-${STAMP}.sql.gz"

mkdir -p "${BACKUP_DIR}"
cd "${PROJECT_DIR}"

COMPOSE="docker compose -f docker-compose.yml -f docker-compose.prod.yml"

# docker compose exec -T (sin tty): necesario bajo cron.
# POSTGRES_USER / POSTGRES_DB inyectados por compose en el container.
# Password nunca por argv — pg_dump hereda env.
$COMPOSE exec -T db bash -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
    | gzip > "${OUT}.tmp"

# Atomic rename (Pitfall 4): si peto a mitad, .tmp queda pero rotacion NO lo valida.
mv "${OUT}.tmp" "${OUT}"

# chmod 600: backup contiene passwords argon2 + audit_log sensibles.
chmod 600 "${OUT}"

# Rotacion (D-09): nexo-*.sql.gz mtime > 7 dias.
find "${BACKUP_DIR}" -maxdepth 1 -name "nexo-*.sql.gz" -type f -mtime +7 -delete

echo "[$(date -u +%FT%TZ)] backup ok: ${OUT} ($(du -h "${OUT}" | cut -f1))"
