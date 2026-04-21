---
phase: 06-despliegue-lan-https
plan: 02
subsystem: infra
tags: [infra, deploy, backup, cron, scripts, makefile]

# Dependency graph
requires:
  - phase: 06-despliegue-lan-https
    plan: 01
    provides: "caddy/Caddyfile.prod (smoke target URL), docker-compose.prod.yml (override consumido por scripts), .env.prod.example (template para secretos prod)"

provides:
  - "scripts/deploy.sh idempotente: git pull --ff-only + pre-deploy pg_dump atomic + build --pull + up -d + smoke HTTPS /api/health"
  - "scripts/backup_nightly.sh ready-for-cron: pg_dump | gzip atomic a /var/backups/nexo/nexo-<stamp>.sql.gz + chmod 600 + retencion -mtime +7"
  - "Makefile targets prod-up/prod-down/prod-logs/prod-status/prod-health/deploy/backup con PROD_COMPOSE var y .PHONY ampliado"
  - "24 tests estaticos (tests/infra/test_deploy_script.py + test_backup_script.py) que congelan landmines: --force prohibido, --no-verify prohibido, emojis prohibidos, rm -rf prohibido, atomic rename obligatorio, 7d retention obligatoria"
affects: [06-03-runbook-smoke, 07-*]

# Tech tracking
tech-stack:
  added:
    - "Bash scripts con set -euo pipefail + IFS safety como patron para scripts de operaciones (primer bash script del repo con esta disciplina)"
    - "Patron atomic rename .tmp -> mv para pg_dump (Pitfall 4) aplicado en dos scripts"
  patterns:
    - "Compose override invocado como variable Makefile PROD_COMPOSE para DRY"
    - "Tests de infra Wave 0: validacion estatica sin ejecutar scripts (subprocess bash -n + grep via .read_text)"
    - "Trust-boundary: pg_dump hereda POSTGRES_USER/POSTGRES_DB del container via `docker compose exec -T db bash -c ...` — nunca por argv (no aparece en process list)"

key-files:
  created:
    - "scripts/deploy.sh (70 lineas) — 5 pasos D-25 con atomic pre-deploy backup y smoke HTTPS"
    - "scripts/backup_nightly.sh (41 lineas) — pg_dump cron-ready con atomic rename y retencion 7d"
    - "tests/infra/test_deploy_script.py (87 lineas, 13 tests) — validacion estatica deploy.sh"
    - "tests/infra/test_backup_script.py (65 lineas, 11 tests) — validacion estatica backup_nightly.sh"
  modified:
    - "Makefile (+34 lineas): 7 targets prod-* + deploy + backup, variable PROD_COMPOSE, .PHONY ampliado; comentario explicito warning sobre Landmine 6 (no `down -v` en prod)"

key-decisions:
  - "Pre-deploy backup vive en subdirectorio dedicado `/var/backups/nexo/predeploy/` (Claude's discretion opcion del CONTEXT §D-25). Rotacion mas agresiva que nightly: -mtime +30 (backup temporal de una version anterior, no backup operativo)."
  - "Smoke test tras deploy: `sleep 15 && curl -fs -m 10 -k https://nexo.ecsmobility.local/api/health`. `-k` intencional (Landmine 4): el host de deploy puede no tener la root CA de Caddy instalada; la integridad del canal no es el objetivo del smoke — lo es verificar que web+caddy arrancaron y /api/health responde 200."
  - "`make prod-down` NO lleva flag `-v` (Landmine 6: borra pgdata). Comentario explicito en el Makefile ademas del naming. `make clean` (dev) SI usa `-v` — documentado que no se ejecute nunca en prod."
  - "Pre-deploy backup skip condicional: si `db` container no esta corriendo (primer deploy) se salta el dump en lugar de fallar. Fallback razonable — el primer deploy no tiene datos que perder."
  - "Backup file password nunca por argv: usa `docker compose exec -T db bash -c 'pg_dump -U \"$POSTGRES_USER\" -d \"$POSTGRES_DB\"'` para heredar env. No aparece en `ps aux`."

patterns-established:
  - "Bash operacional: `set -euo pipefail` + `IFS=$'\\n\\t'` + funciones `log()`/`fail()` para log con timestamp UTC."
  - "Atomic write de archivo importante: escribir a `${OUT}.tmp`, `mv` solo tras exito. Aplica a backups y a cualquier artefacto que otros procesos (rotacion, restore) puedan leer mid-write."
  - "Makefile prod-* como wrapper del compose override: cualquier target prod futuro reutiliza `$(PROD_COMPOSE)` en lugar de duplicar `-f docker-compose.yml -f docker-compose.prod.yml`."

requirements-completed: [DEPLOY-04]

# Metrics
duration: ~18 min
completed: 2026-04-21
---

# Phase 06-despliegue-lan-https Plan 02: Deploy Script + Backup Script + Makefile Prod Targets Summary

**`scripts/deploy.sh` idempotente con pre-deploy pg_dump atomic + smoke HTTPS, `scripts/backup_nightly.sh` cron-ready con atomic rename y retencion 7d, Makefile ampliado con 7 targets prod + variable PROD_COMPOSE. 24 tests estaticos que congelan landmines (--force/--no-verify/rm -rf/emojis prohibidos, atomic rename obligatorio).**

## Performance

- **Duration:** ~18 min
- **Completed:** 2026-04-21
- **Tasks:** 3 de 3 (todas autonomous, sin checkpoints)
- **Files created:** 4 (2 shell scripts + 2 pytest modules)
- **Files modified:** 1 (Makefile)

## Accomplishments

- `scripts/deploy.sh` implementa los 5 pasos D-25: (1) `git pull --ff-only` — sin force, exit 1 con mensaje claro en caso de divergencia; (2) pre-deploy `pg_dump` con atomic rename `.tmp`→final y `chmod 600`, skippeable si `db` no corre (primer deploy); (3) `build --pull` con compose override; (4) `up -d`; (5) smoke `curl -fs -m 10 -k https://nexo.ecsmobility.local/api/health`. Funciones `log()` y `fail()` loguean con timestamp UTC. Rollback no automatico: el mensaje de fail incluye el comando manual `git reset --hard HEAD~1 && bash scripts/deploy.sh`.
- `scripts/backup_nightly.sh` cron-ready: `cd /opt/nexo`, `pg_dump | gzip > ${OUT}.tmp`, `mv`, `chmod 600`, rotacion `find /var/backups/nexo -maxdepth 1 -name 'nexo-*.sql.gz' -mtime +7 -delete`. Password Postgres heredada del container (no argv). Echo final imprime path y tamaño para que cron lo capture al log del sistema.
- Makefile ampliado: variable `PROD_COMPOSE = docker compose -f docker-compose.yml -f docker-compose.prod.yml` + 7 targets `prod-up`, `prod-down`, `prod-logs`, `prod-status`, `prod-health`, `deploy`, `backup`. `.PHONY` actualizado. Comentario explicito warning sobre Landmine 6 (no `down -v` en prod).
- Dev targets intactos: `make -n up` sigue emitiendo `docker compose up -d` (sin override), `make -n clean` sigue emitiendo `docker compose down -v` (dev uso -v intencional), `make -n dev` sigue arrancando uvicorn con el mismo flow.
- 24 tests estaticos (`tests/infra/test_deploy_script.py` + `tests/infra/test_backup_script.py`) validan sintaxis bash, presencia de pasos canonicos, y ausencia de landmines (--force, --no-verify, rm -rf, emojis).
- Suite completa `tests/infra/`: 49 tests (25 Wave 0 de Plan 06-01 + 24 Wave 2 de Plan 06-02) — todos pasan en `<1s`.

## Task Commits

Cada tarea committeada atomicamente en `feature/Mark-III`:

1. **Task 1: scripts/deploy.sh + test_deploy_script.py** — `38f6bb2` (feat)
2. **Task 2: scripts/backup_nightly.sh + test_backup_script.py** — `79ad364` (feat)
3. **Task 3: Makefile prod targets + deploy + backup** — `b476d05` (feat)

_(Plan metadata commit se crea al cierre del ciclo execute-phase.)_

## Files Created/Modified

- `scripts/deploy.sh` (70 lineas) — ejecutable, bash -n OK, pasa 13 tests.
- `scripts/backup_nightly.sh` (41 lineas) — ejecutable, bash -n OK, pasa 11 tests.
- `tests/infra/test_deploy_script.py` (87 lineas, 13 tests).
- `tests/infra/test_backup_script.py` (65 lineas, 11 tests).
- `Makefile` (+34 lineas) — 7 nuevos targets + PROD_COMPOSE + .PHONY ampliado. Sin regresion en targets dev (validado via `make -n`).

## Decisions Made

- **D-25 cerrado**: deploy.sh sigue los 5 pasos literales (pull FF-only, pre-deploy dump, build --pull, up -d, smoke HTTPS). Atomic rename añadido al pre-deploy (RESEARCH lo recomienda aunque el snippet Topic 7 lo omitia).
- **D-26 cerrado**: deploy.sh idempotente — `git pull --ff-only` es no-op si ya al dia; `build --pull` cachea; `up -d` es no-op si containers sanos. Ejecutarlo dos veces seguidas no rompe nada.
- **D-07 cerrado**: `backup_nightly.sh` hace pg_dump plain SQL | gzip pensado para cron 03:00 UTC; la linea cron se documenta en el runbook 06-03 (no es responsabilidad de este plan crearla).
- **D-08 cerrado**: BACKUP_DIR fijo en `/var/backups/nexo/`. Pre-deploy backups en subdir `/var/backups/nexo/predeploy/` para distinguirlos del backup nightly operativo (Claude's discretion del CONTEXT §Claude's Discretion).
- **D-09 cerrado**: retencion 7d via `find -mtime +7 -delete`. Pre-deploy subdir con retencion mas agresiva (-mtime +30) porque es backup transicional.
- **D-10 aceptado**: RPO 24h se respeta; el cron diario es la frecuencia minima.
- **Landmine 4 (-k en curl smoke)**: intencional y documentado en comentario inline. El host de deploy no siempre tiene la root CA de Caddy instalada; el smoke valida "web+caddy vivos y /api/health responde 200", no "la cadena TLS es valida".
- **Landmine 6 (no -v en prod)**: `prod-down` del Makefile NUNCA usa `-v`. Comentario explicito en el bloque de targets prod. Contraste documentado con el dev `clean` que SI usa `-v` intencionalmente.
- **Password Postgres nunca por argv**: `pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB"` con env heredado del container, jamas `-p PASSWORD`. Protege contra leak via `ps aux`.

## Deviations from Plan

**None.** Las 3 tareas se ejecutaron exactamente como prescribe el plan, sin surprises. Los scripts se escribieron verbatim del RESEARCH.md/CONTEXT.md con el atomic-rename añadido (que el plan explicitaba), y las 3 ediciones del Makefile siguieron el patron pedido (.PHONY + bloque nuevo antes del target help).

## Landmines Cubiertas (como tests)

Tests que bloquean reintroducir landmines en futuros plans:

| Landmine | Fuente | Test preventivo |
|----------|--------|-----------------|
| Pitfall 4 (atomic write) — crash mid-pg_dump deja archivo invalido que rotacion trata como bueno | RESEARCH §Pitfall 4 | `test_deploy_sh_atomic_rename` + `test_backup_sh_atomic_rename` (asserts `.tmp in content` + `mv in content`) |
| Pitfall 5 (git --force) — force pull trae codigo no revisado y lo despliega | CLAUDE.md "Qué NO hacer" | `test_deploy_sh_git_pull_ff_only` (asserts `--force not in content`) |
| Landmine 6 (`docker compose down -v`) — borra pgdata en prod | RESEARCH §Landmine 6 | Makefile `prod-down` NO tiene `-v`; comentario explicito |
| Emojis en scripts ops | CLAUDE.md + rules/common/coding-style.md | `test_deploy_sh_no_emoji` (regex de ranges unicode) |
| `--no-verify` en scripts | CLAUDE.md | `test_deploy_sh_no_no_verify` |
| `rm -rf` en scripts ops | CLAUDE.md + seguridad | `test_backup_sh_no_rm_rf` |
| Password Postgres por argv | Seguridad | Test implicito: ambos scripts usan `bash -c 'pg_dump -U "$POSTGRES_USER"...'` (env inherit, no argv) |
| Landmine 7 (pg_dump no atomico) — escribir directo al path final | RESEARCH §Landmine 7 | `test_backup_sh_atomic_rename` |

## Threat Model Compliance

Los 6 threats del `<threat_model>` del plan estan mitigados por artefactos:

| Threat | Mitigation | Evidence |
|--------|------------|----------|
| T-06-07 (InfoDisclosure backup permissive) | `chmod 600 "${OUT}"` en ambos scripts | `test_backup_sh_chmod_600` + codigo deploy.sh L39 |
| T-06-08 (Tampering backup parcial) | Atomic rename `.tmp`→final | Tests atomic_rename + codigo |
| T-06-09 (Elevation git --force) | `git pull --ff-only` + test guard | `test_deploy_sh_git_pull_ff_only` |
| T-06-10 (Repudiation backup fallido silencioso) | `set -euo pipefail` + log final solo tras `mv` | Codigo de ambos scripts |
| T-06-11 (DoS `rm -rf` accidental) | No hay `rm -rf`; rotacion usa `find -delete` | `test_backup_sh_no_rm_rf` |
| T-06-12 (.env leak en log) | deploy.sh no lee .env ni loguea vars | Accept (riesgo residual: root del host trusted) |

## Verification Commands Ejecutados

```bash
# 1. Ejecutabilidad + sintaxis
test -x scripts/deploy.sh && bash -n scripts/deploy.sh                     # exit 0
test -x scripts/backup_nightly.sh && bash -n scripts/backup_nightly.sh     # exit 0

# 2. Tests especificos 06-02
/tmp/nexo-testvenv/bin/pytest -x -q --confcutdir=tests/infra \
    tests/infra/test_deploy_script.py \
    tests/infra/test_backup_script.py
# 24 passed in 0.03s

# 3. Suite completa tests/infra (Wave 0 + Wave 2)
/tmp/nexo-testvenv/bin/pytest -x -q --confcutdir=tests/infra tests/infra/
# 49 passed in 0.96s

# 4. Makefile dry-run: dev sin regresion
make -n up      # docker compose up -d
make -n clean   # docker compose down -v
make -n dev     # flow original uvicorn

# 5. Makefile dry-run: prod targets funcionan
make -n prod-up   # docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
make -n deploy    # bash scripts/deploy.sh
make -n backup    # bash scripts/backup_nightly.sh

# 6. make help lista nuevos targets
make help | grep -E 'prod-|^  deploy|^  backup'
# Muestra los 5 prod-* + deploy + backup
```

## Requirements Cubiertos

- **DEPLOY-04**: `scripts/deploy.sh` implementa `git pull && docker compose --profile prod build --pull && docker compose --profile prod up -d` (adaptado a override en lugar de profile — decision D-18). Incluye extras: pre-deploy backup, smoke HTTPS, exit propagation via `set -euo pipefail`.

Nota: la verificacion end-to-end real (script ejecutado contra servidor Ubuntu con `.env` prod) queda para Plan 06-03 cuando haya maquina fisica. Este plan cierra los artefactos; el runbook 06-03 cerrara el smoke manual.

## Issues Encountered

- **Shell escaping en commit post-check (Task 2)**: El command compuesto de commit + deletion-check exploto por escaping de `$()` en heredoc inline. El commit sí completo (`79ad364`); solo la verificacion post-commit fallo su shell parse. Recuperado con reejecucion limpia del check en una Bash call separada. No afecta al resultado.
- **Sin shellcheck disponible**: El sistema WSL no tiene `shellcheck` instalado. No bloqueante per el quality_gate del prompt ("non-blocking if not installed"). `bash -n` si se ejecuto con exito.
- **Working tree pre-existente no-tocado**: `.gitignore`, `api/main.py` modificados y `.codex/`, `static/img/brand/nexo/logo-tab-navegador.png` untracked desde antes del plan. Respetados exactamente como pidio el critical_constraint #7 — los 3 commits de 06-02 solo staged archivos de la lista `files_modified`.

## User Setup Required

Ninguno para este plan. El plan 06-03 (runbook) referenciara estos scripts y explicara al operador como:
1. Instalar la linea cron `0 3 * * * /opt/nexo/scripts/backup_nightly.sh` (D-07).
2. Invocar `bash scripts/deploy.sh` vía `make deploy` en cada release.
3. Verificar el bit ejecutable tras clonar (git preserva el bit — deberia estar OK).

## Next Phase Readiness

- **Plan 06-03 (DEPLOY_LAN.md + smoke)** tiene todo lo que cita el runbook:
  - Script de deploy listo para ejecutarse (`bash scripts/deploy.sh` o `make deploy`).
  - Script de backup listo para cron (`/opt/nexo/scripts/backup_nightly.sh`).
  - Makefile prod-* documentables como comandos operativos frecuentes.
  - 24 tests que el runbook puede citar como "acceptance automatico" en el ciclo CI.
- **Pendientes para el handoff**:
  - El runbook debe documentar la linea cron exacta.
  - El runbook debe incluir el smoke end-to-end manual contra servidor fisico.
  - El runbook puede citar los commits `38f6bb2`/`79ad364`/`b476d05` como "artefactos de este sprint".

---
*Phase: 06-despliegue-lan-https*
*Completed: 2026-04-21*

## Self-Check: PASSED

Verificaciones post-summary:

- [x] `scripts/deploy.sh` existe en disco y tiene bit ejecutable.
- [x] `scripts/backup_nightly.sh` existe en disco y tiene bit ejecutable.
- [x] `tests/infra/test_deploy_script.py` existe (13 tests pasando).
- [x] `tests/infra/test_backup_script.py` existe (11 tests pasando).
- [x] `Makefile` contiene targets `prod-up`, `prod-down`, `prod-logs`, `prod-status`, `prod-health`, `deploy`, `backup` + variable `PROD_COMPOSE`.
- [x] Commit `38f6bb2` existe (Task 1: deploy.sh + tests).
- [x] Commit `79ad364` existe (Task 2: backup_nightly.sh + tests).
- [x] Commit `b476d05` existe (Task 3: Makefile prod targets).
- [x] Suite completa `tests/infra/`: 49 passed in 0.96s.
- [x] `make -n up`/`clean`/`dev` sin regresion (output byte-identico a pre-plan).
- [x] Ningun commit tiene deletions (`git diff --diff-filter=D HEAD~3 HEAD` vacio).
- [x] Sin emojis en scripts/Makefile/tests (coherencia con CLAUDE.md).
- [x] Pre-existing user changes (`api/main.py`, `.gitignore`, `.codex/`, `static/img/brand/nexo/logo-tab-navegador.png`) NO tocados — solo commits para archivos de `files_modified` del plan.
