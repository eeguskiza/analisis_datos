#!/usr/bin/env bash
set -euo pipefail

COMPOSE=${COMPOSE:-docker compose}
SERVICES=${NEXO_SERVICES:-db web caddy}
STOPPED=0
PROJECT_NAME=${COMPOSE_PROJECT_NAME:-$(basename "$PWD")}
WEB_IMAGE=${NEXO_WEB_IMAGE:-${PROJECT_NAME}-web:latest}

print_links() {
  printf '\n'
  printf '\033[1;42;30m NEXO LISTO \033[0m\n'
  printf '\033[1;32m  App:\033[0m http://localhost\n'
  printf '\033[1;32m  LAN:\033[0m https://nexo.ecsmobility.local\n'
  printf '\033[1;32m  API:\033[0m http://localhost/api/docs\n'
  printf '\033[1;90m  Nota: deja esta terminal abierta para ver logs; Ctrl+C para parar.\033[0m\n\n'
}

if [ ! -f ".env" ]; then
  printf '\033[1;31m[nexo-init]\033[0m Falta .env. Crea uno desde .env.example antes de arrancar.\n' >&2
  exit 1
fi

cleanup() {
  local status=$?
  if [ "$STOPPED" -eq 1 ]; then
    exit "$status"
  fi
  STOPPED=1
  printf '\n\033[1;33m[nexo-init]\033[0m Parando contenedores: %s\n' "$SERVICES"
  # stop conserva volumenes y containers; no borra datos.
  $COMPOSE stop $SERVICES >/dev/null 2>&1 || true
  printf '\033[1;32m[nexo-init]\033[0m Stack parado. Datos conservados en volumenes Docker.\n'
  exit "$status"
}

trap cleanup INT TERM

if [ "${NEXO_BUILD:-0}" = "1" ] || ! docker image inspect "$WEB_IMAGE" >/dev/null 2>&1; then
  printf '\033[1;36m[nexo-init]\033[0m Construyendo imagen web (%s)...\n' "$WEB_IMAGE"
  $COMPOSE build web
else
  printf '\033[1;36m[nexo-init]\033[0m Usando imagen web local: %s\n' "$WEB_IMAGE"
  printf '\033[1;90m[nexo-init]\033[0m Para reconstruir: NEXO_BUILD=1 make init\n'
fi

printf '\033[1;36m[nexo-init]\033[0m Arrancando Postgres...\n'
$COMPOSE up -d db

printf '\033[1;36m[nexo-init]\033[0m Esperando a Postgres...\n'
until $COMPOSE exec -T db pg_isready \
  -U "${NEXO_PG_USER:-${OEE_PG_USER:-oee}}" \
  -d "${NEXO_PG_DB:-${OEE_PG_DB:-oee_planta}}" >/dev/null 2>&1; do
  sleep 1
done

printf '\033[1;36m[nexo-init]\033[0m Aplicando schema nexo (idempotente)...\n'
$COMPOSE run --rm --no-deps web python scripts/init_nexo_schema.py

printf '\033[1;32m[nexo-init]\033[0m Arrancando stack completo con logs en tiempo real.\n'
printf '\033[1;90m[nexo-init]\033[0m Docker Compose colorea y etiqueta cada linea por servicio.\n'
printf '\033[1;90m[nexo-init]\033[0m Ctrl+C para parar db/web/caddy sin borrar datos.\n\n'
print_links

# Recreate aplica cambios de compose como bind mounts sin reconstruir imagen.
$COMPOSE up -d --force-recreate web caddy
$COMPOSE logs -f --tail=80 $SERVICES
cleanup
