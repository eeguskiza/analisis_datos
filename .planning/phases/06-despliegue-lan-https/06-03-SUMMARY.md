---
phase: 06-despliegue-lan-https
plan: 03
subsystem: infra
tags: [infra, docs, runbook, smoke, ufw, ca, backup]

# Dependency graph
requires:
  - phase: 06-despliegue-lan-https
    plan: 01
    provides: "caddy/Caddyfile.prod, docker-compose.prod.yml, .env.prod.example — referenciados desde el runbook"
  - phase: 06-despliegue-lan-https
    plan: 02
    provides: "scripts/deploy.sh, scripts/backup_nightly.sh, Makefile prod targets — instrucciones operativas en runbook"

provides:
  - "docs/DEPLOY_LAN.md — runbook operativo 16 secciones en espanol para admin de respaldo (D-11, D-12); RTO 1-2h"
  - "tests/infra/deploy_smoke.sh — checklist scriptable post-deploy con 11 checks DEPLOY-01/02/03/06 y exit code = numero de fallos"
  - "tests/infra/test_deploy_lan_doc.py — 24 tests de regresion doc + smoke; bloquean merge si se borra seccion critica (ufw, down -v, root CA, cron)"
affects: [07-devex-hardening]

# Tech tracking
tech-stack:
  added:
    - "Bash smoke post-deploy con output estructurado [DEPLOY-XX] OK|FAIL + exit code = FAILS (integrable con cron/CI)"
    - "Pytest doc-regression pattern: validar literales criticos del runbook para prevenir silent drift"
  patterns:
    - "Runbook en espanol, comandos en ingles (copy-pasteable); placeholders <IP_NEXO>/<SUBNET_LAN>/<ADMIN_BACKUP_NAME> literales"
    - "Landmines en seccion dedicada (9.1-9.7) con warnings explicitos para las 2 mas criticas (Docker bypassa ufw, down -v borra pgdata)"
    - "Test de regresion con end-to-end verification: verificar que borrando X el test falla (validacion meta-test)"

key-files:
  created:
    - "docs/DEPLOY_LAN.md — runbook 740 lineas, 16 secciones + 2 apendices"
    - "tests/infra/deploy_smoke.sh — 93 lineas, 11 checks DEPLOY-*, ejecutable, bash -n OK"
    - "tests/infra/test_deploy_lan_doc.py — 233 lineas, 24 tests"
  modified: []

key-decisions:
  - "Placeholders literales <IP_NEXO>/<SUBNET_LAN>/<ADMIN_BACKUP_NAME>/<URL_REPO_NEXO>: el runbook no asume IP asignada (D-03, D-12, D-14)."
  - "Orden de secciones: landmines (9) ANTES de cron (10) y validacion (11) para que el lector interiorice las trampas antes de tocar comandos destructivos. CONTEXT Claude's Discretion ejercido."
  - "Smoke script usa `set -uo pipefail` (no `-e`): queremos que el script siga tras fallos individuales para contar FAILS y reportar todos los checks fallidos en una ejecucion, no abortar al primero."
  - "curl con `-k` en smoke: el servidor no tiene necesariamente la root CA instalada en su trust store (la acaba de emitir). El smoke valida 'Caddy + web vivos', no 'cadena TLS valida desde el servidor'. Landmine 4 del plan 06-02 ya cubrio esta decision; se reafirma aqui."
  - "Regression tests validados end-to-end: borrar 'ufw allow 443/tcp' / 'down -v' / inyectar emoji efectivamente hacen fallar los tests correspondientes. No es teoria; se verifico con sed -i + restore."

patterns-established:
  - "Doc regresion: pytest sobre contenido critico de runbook es mas robusto que depender de revision humana; guarda el contrato frente a refactors descuidados."
  - "Smoke script separado de CI: checklist manual en servidor Ubuntu real, exit code = FAILS count (usable por cron diario post-deploy)."
  - "Tabla DEPLOY-* -> comando -> esperado en el runbook: doble fuente (scriptada en smoke + manual en doc) para que el operador pueda validar por cualquiera de los dos caminos."

requirements-completed: [DEPLOY-05, DEPLOY-06, DEPLOY-07]

# Metrics
duration: ~7 min
completed: 2026-04-21
---

# Phase 06-despliegue-lan-https Plan 03: Runbook + Smoke + Doc Regression Summary

**Runbook operativo `docs/DEPLOY_LAN.md` (740 lineas, 16 secciones, espanol, sin emojis) que convierte los artefactos de 06-01 y 06-02 en procedimiento reproducible por admin de respaldo. Smoke script `tests/infra/deploy_smoke.sh` con 11 checks DEPLOY-* y exit = FAILS. 24 tests pytest que atrapan regresiones del doc. Suite tests/infra/ total: 73 passed en <1s (49 pre-existentes + 24 nuevos).**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-04-21T17:24Z
- **Completed:** 2026-04-21T17:31Z
- **Tasks:** 3 de 3 (todas autonomous, sin checkpoints)
- **Files created:** 3
- **Files modified:** 0

## Accomplishments

- **`docs/DEPLOY_LAN.md`** (740 lineas, 18 secciones `##`): runbook end-to-end
  en espanol para admin de respaldo con conocimiento Linux basico pero sin
  contexto del proyecto Nexo. Cubre:
  - Secciones 1-2: vista general, modelo de amenazas LAN-only, prerrequisitos.
  - Seccion 3: instalar Docker CE + compose plugin en Ubuntu 24.04 limpio
    (Topic 9 literal con apt repo oficial).
  - Seccion 4: clone `/opt/nexo`, `cp .env.prod.example .env`, rellenar
    `<CHANGEME-*>`, generar `NEXO_SECRET_KEY` con `secrets.token_urlsafe(48)`.
  - Seccion 5: ufw (Topic 5 literal) con orden seguro (reglas antes de
    `enable` para no dejarse fuera).
  - Seccion 6: `make prod-up` + verificacion `make prod-status`.
  - Seccion 7: extraer y distribuir root CA Caddy por SO (Topic 1 literal):
    Windows certutil, Linux update-ca-certificates, macOS security
    add-trusted-cert, Firefox trust store separado.
  - Seccion 8: hosts-file por SO (Topic 6 literal).
  - Seccion 9: **Landmines** numeradas 9.1-9.7 con warnings explicitos:
    - 9.1 Docker bypassa ufw (HIGH) con snippet nmap/ss verificatorios.
    - 9.2 `down -v` borra pgdata (HIGH) con AVISO destacado.
    - 9.3 No `git commit` en servidor.
    - 9.4 No copiar Caddyfile dev a prod.
    - 9.5 Root CA expira 10 anos + comando de fecha.
    - 9.6 `/api/health` consulta MES pero no tira el container.
    - 9.7 No rotar SQL Server en deploy rutinario.
  - Seccion 10: cron `/etc/cron.d/nexo-backup` con schedule `0 3 * * *` UTC
    (D-07).
  - Seccion 11: tabla DEPLOY-* + referencia al smoke script; duplica
    comandos para validacion manual.
  - Seccion 12: operaciones rutinarias (Makefile targets + psql interactivo).
  - Seccion 13: restaurar (Topic 4 literal con DROP SCHEMA + zcat | psql).
  - Seccion 14: recuperacion desde cero RTO 1-2h (D-11) con 8 pasos
    numerados, resaltando que la root CA del nuevo Caddy es distinta.
  - Seccion 15: mejoras futuras (DNS interno, NAS sync, CA corporativa
    ECS, WAL archiving, smoke extendido, monitoring). Marcadas como
    deferred — no bloquean Mark-III.
  - Seccion 16: glosario minimo + referencias cruzadas + contactos.
  - Apendices A y B: comandos de un vistazo y secuencia mental del
    deploy rutinario.

- **`tests/infra/deploy_smoke.sh`** (93 lineas, 11 checks): script bash
  post-deploy para ejecutar en el servidor Ubuntu tras cada `scripts/deploy.sh`.
  Output `[DEPLOY-XX] OK|FAIL: mensaje`, exit code = contador de fallos.
  - DEPLOY-01 (HTTPS + cert Caddy): 2 checks — curl 200 + openssl verifica
    issuer `Caddy Local Authority`.
  - DEPLOY-02 (Postgres cerrado): 2 checks — `ss -tlnp :5432` sin output
    + psql exec OK (red interna compose sigue funcionando).
  - DEPLOY-03 (healthchecks): 3 checks — web healthy, caddy healthy,
    restart policy = `unless-stopped`.
  - DEPLOY-06 (ufw): 4 checks — activo, 443 allow, 80 allow, default
    incoming deny.
  - `set -uo pipefail` intencional (no `-e`) para que el script siga
    contando fallos tras el primero.
  - `curl -k` intencional (Landmine 4): smoke valida "Caddy + web vivos",
    no "cadena TLS valida desde el servidor".

- **`tests/infra/test_deploy_lan_doc.py`** (233 lineas, 24 tests): suite
  de regresion que congela el contenido critico del runbook y del smoke.
  Si alguien borra `ufw allow 443/tcp`, `down -v`, el path `/etc/hosts`,
  la entrada cron, el placeholder `<IP_NEXO>`, etc., pytest FALLA y
  bloquea el merge.
  - 17 tests sobre DEPLOY_LAN.md (estructura, placeholders, secciones,
    cron, restore, emojis, Let's Encrypt guardado como deferred).
  - 7 tests sobre deploy_smoke.sh (sintaxis bash, ejecutable, cobertura
    DEPLOY-*, exit count, emojis).

- **Suite tests/infra/ total:** 73 tests pasados en <1s. 49 pre-existentes
  (Plan 06-01: 25 + Plan 06-02: 24) + 24 nuevos de este plan.

## Task Commits

Cada tarea committeada atomicamente en `feature/Mark-III`:

1. **Task 1: docs/DEPLOY_LAN.md** — `8cd513d` (docs)
2. **Task 2: tests/infra/deploy_smoke.sh** — `4e25f49` (feat)
3. **Task 3: tests/infra/test_deploy_lan_doc.py** — `a363fcf` (test)

_(Plan metadata commit se crea al cierre del ciclo execute-phase con este
SUMMARY + actualizaciones STATE/ROADMAP/REQUIREMENTS.)_

## Files Created

- `docs/DEPLOY_LAN.md` — 740 lineas, 16 secciones + 2 apendices.
- `tests/infra/deploy_smoke.sh` — 93 lineas, ejecutable, `bash -n` OK.
- `tests/infra/test_deploy_lan_doc.py` — 233 lineas, 24 tests.

Total: 3 archivos nuevos, 1066 lineas. Ningun archivo modificado.

## Landmines Surfaceadas en el Runbook

Resumen de las landmines numeradas en §9 del runbook, listas para el
admin de respaldo:

| # | Landmine | Severidad | Ubicacion | Test preventivo |
|---|----------|-----------|-----------|-----------------|
| 9.1 | Docker bypassa ufw (`ufw deny 5432` inoperante) | HIGH | §9.1 con `nmap`/`ss` verificatorios | `test_doc_warns_docker_bypasses_ufw` |
| 9.2 | `docker compose down -v` borra pgdata | HIGH | §9.2 con AVISO destacado | `test_doc_warns_down_v_landmine` |
| 9.3 | `git commit` en servidor rompe `pull --ff-only` | MEDIUM | §9.3 + flujo correcto | (heredado de 06-02 test) |
| 9.4 | Copiar Caddyfile dev a prod mata redirect 80->443 | MEDIUM | §9.4 + diff documentado | `test_caddyfile_prod_no_auto_https_disable_redirects` (06-01) |
| 9.5 | Root CA expira ~10 anos; re-distribucion manual | LOW | §9.5 + comando fecha | — (operacional) |
| 9.6 | `/api/health` consulta MES; no cambiar healthcheck a `jq -e .ok` | LOW | §9.6 | `test_override_web_healthcheck_curl_api_health` (06-01) |
| 9.7 | No rotar credenciales SQL Server en deploy rutinario | LOW | §9.7 | — (procedimiento) |

Landmines de 06-01 y 06-02 ya cubiertas por sus tests respectivos
(L1 Caddy auto_https, L2 ports reset, L2b volumes override, L10 COMPOSE_PROJECT_NAME,
Pitfall 4 atomic rename, Pitfall 5 git force).

## Decisions Made

- **D-05 cerrado (runbook)**: Distribucion root CA documentada con snippets
  por SO literales de Topic 1 (Windows certutil + GUI, Linux
  update-ca-certificates, macOS security add-trusted-cert, Firefox GUI).
- **D-11 cerrado**: RTO 1-2h explicitamente declarado en §1 + §14 con
  desglose (1h instalacion + 30 min restore + 30 min re-distribucion CA).
- **D-12 cerrado**: Bus factor 2 reflejado en tono del doc (asume admin
  con Linux basico, no dev Nexo) + placeholder `<ADMIN_BACKUP_NAME>`
  en header y §16 Contactos.
- **D-13 cerrado**: Secuencia runbook = instalar Docker -> clone -> .env
  -> ufw -> prod-up -> extraer CA -> hosts-file -> validacion -> cron
  nightly. Orden refinado (ufw antes que prod-up para que no haya ventana
  sin firewall; landmines antes de cron/validacion para que el lector las
  interiorice antes de tocar comandos destructivos).
- **D-14/D-15/D-16/D-17 cerrados**: seccion 5 con los comandos ufw
  literales. Orden seguro: reglas antes de `enable`.
- **Placeholders literales** (D-03, D-12, D-14): `<IP_NEXO>`,
  `<SUBNET_LAN>`, `<ADMIN_BACKUP_NAME>`, `<URL_REPO_NEXO>` presentes con
  angle brackets para que el operador los reemplace al deploy real.
- **Sin emojis** (CLAUDE.md + `rules/common/coding-style.md`):
  `test_doc_no_emoji` bloquea cualquier reintroduccion.

## Deviations from Plan

**None.** Las 3 tareas se ejecutaron exactamente como prescribe el plan,
sin surprises. El runbook siguio la plantilla del plan literalmente (16
secciones + apendices). El smoke script es 1:1 con el snippet del plan.
El test pytest anadio 6 tests adicionales sobre el minimo de 18 exigido
(24 total) para cubrir:

1. `test_doc_has_placeholders` valida los 3 placeholders juntos
   (incluye `<ADMIN_BACKUP_NAME>` que el plan mencionaba pero no
   testeaba).
2. `test_doc_references_future_dns_improvement` garantiza la mejora
   futura DNS interno queda documentada.
3. `test_doc_has_rto_target` valida el RTO 1-2h (D-11) literal.
4. `test_smoke_sh_no_emoji` y `test_smoke_sh_uses_set_safety_flags`
   para el smoke script.

Todos dentro del scope del plan (quality_gate pedia >=18 tests + literales
criticos).

## Threat Model Compliance

Los 7 threats del `<threat_model>` del plan estan mitigados por artefactos:

| Threat | Mitigation | Evidence |
|--------|------------|----------|
| T-06-13 (Repudiation admin interpreta mal ufw/Docker) | §9.1 dedicada a "Docker bypassa ufw" + comandos de verificacion (ss, nmap) | `test_doc_warns_docker_bypasses_ufw` |
| T-06-14 (InfoDisclosure CA por canal inseguro) | §7.1 instruye "scp, USB, chat interno — NO email publico" | Literal en el runbook; auditable |
| T-06-15 (DoS `down -v` accidental) | §9.2 + test regresion | `test_doc_warns_down_v_landmine` |
| T-06-16 (Tampering git commit en servidor) | §9.3 + deploy.sh usa `--ff-only` (heredado) | `test_deploy_sh_git_pull_ff_only` (06-02) |
| T-06-17 (Spoofing Caddyfile dev a prod) | §9.4 documenta el landmine | `test_caddyfile_prod_no_auto_https_disable_redirects` (06-01) |
| T-06-18 (Elevation internet exposure) | `test_doc_no_internet_exposure` valida Let's Encrypt solo marcado como deferred | Test automatizado |
| T-06-19 (InfoDisclosure backup permisivo) | §10 + convencion `chmod 600` heredada de 06-02 | `test_backup_sh_chmod_600` (06-02) |

## Verification Commands Ejecutados

```bash
# 1. Existencia de los 3 artefactos
test -f docs/DEPLOY_LAN.md                                     # exit 0
test -x tests/infra/deploy_smoke.sh                            # exit 0 (executable)
test -f tests/infra/test_deploy_lan_doc.py                     # exit 0

# 2. Runbook stats
wc -l docs/DEPLOY_LAN.md                                       # 740 >= 300
grep -cE '^## ' docs/DEPLOY_LAN.md                             # 18 >= 12

# 3. Smoke script sintaxis
bash -n tests/infra/deploy_smoke.sh                            # exit 0

# 4. Tests especificos 06-03
/tmp/nexo-testvenv/bin/pytest -x -q --confcutdir=tests/infra \
    tests/infra/test_deploy_lan_doc.py
# 24 passed in 0.03s

# 5. Suite completa tests/infra/ (los 3 planes Phase 6)
/tmp/nexo-testvenv/bin/pytest -q --confcutdir=tests/infra tests/infra/
# 73 passed in 0.99s

# 6. Regresion end-to-end verificada (sed -i + restore):
#    - Borrar 'ufw allow 443/tcp' -> test_doc_has_ufw_rules FAIL
#    - Borrar 'down -v'           -> test_doc_warns_down_v_landmine FAIL
#    - Inyectar emoji             -> test_doc_no_emoji FAIL
```

## Requirements Cubiertos

- **DEPLOY-05**: `docs/DEPLOY_LAN.md` runbook para admin de respaldo con
  Docker install, clone, .env, ufw, arranque, distribucion CA, hosts-file,
  validacion DEPLOY-*, operaciones, restore, recuperacion desde cero,
  landmines, mejoras futuras.
- **DEPLOY-06**: ufw documentado en §5 con reglas literales (SSH
  `<SUBNET_LAN>` + 80 + 443 + deny incoming) + seccion §9.1 "Docker bypassa
  ufw" que explica por que la defensa real del 5432 es el
  `docker-compose.prod.yml` (cubierto por 06-01 + este doc). Smoke script
  valida el estado ufw post-deploy.
- **DEPLOY-07**: Smoke script `deploy_smoke.sh` scripted para ejecutar
  desde peer LAN + `nmap` comandos documentados en §11 + instalacion de
  root CA + hosts-file por SO en §7-§8 + verificacion openssl final en §7.6.

**Nota**: Los 3 requirements se cierran a nivel **documentacion + tests de
regresion**. La ejecucion empirica contra un servidor Ubuntu real queda
deferred hasta que IT asigne el host (ver RESEARCH §"Environment
Availability" y Deferred Verification en STATE.md abajo).

## Deferred Verification

El smoke E2E real contra un servidor Ubuntu fisico con `.env.prod`
rellenado no se puede ejecutar hoy (host no asignado; CONTEXT D-03 lo
pone como `<IP_NEXO>` placeholder). Se agrega a `STATE.md` como
Deferred Verification:

- **Plan**: 06-03
- **Verification**: Ejecutar `bash tests/infra/deploy_smoke.sh` en el
  servidor Ubuntu real con stack prod levantado. Aceptacion: 0 fallos
  (exit 0, 11 checks OK).
- **Operador**: e.eguskiza@ecsmobility.com
- **Comando**: `ssh <IP_NEXO> 'cd /opt/nexo && bash tests/infra/deploy_smoke.sh'`

Sin deadline duro porque depende de la asignacion del host por IT.

## Issues Encountered

- **Sin pytest en host WSL ni en container web**: se reutilizo el venv
  temporal `/tmp/nexo-testvenv/` creado en Plan 06-01. Ejecucion con
  `--confcutdir=tests/infra` para evitar el `tests/conftest.py` global
  que importa SQLAlchemy. Diferido para Phase 7 (DevEx) formalizar un
  target Makefile `test-infra` con setup automatico.
- **Working tree pre-existente no-tocado**: `.gitignore`, `api/main.py`
  modificados y `.codex/`, `static/img/brand/nexo/logo-tab-navegador.png`
  untracked desde antes del plan. Respetados exactamente como pidio el
  critical_constraint #11 del prompt: los 3 commits de 06-03 solo staged
  archivos de la lista `files_modified` del plan.

## User Setup Required

Ninguno para completar este plan. El operador necesita:

1. **Para validacion empirica (deferred)**: un servidor Ubuntu 24.04 con
   IP LAN asignada + acceso SSH + credenciales `.env.prod`.
2. **Para validacion DEPLOY-07 real**: un segundo equipo LAN con la root
   CA + hosts-file configurados (los pasos estan en §7-§8 del runbook).

## Next Phase Readiness

- **Phase 6 completa**: los 3 plans cierrados. 8 requisitos DEPLOY-*
  cubiertos por artefactos + tests estaticos. Pendiente la validacion
  empirica en Ubuntu fisico (deferred a `STATE.md`).
- **Phase 7 (DevEx hardening)** puede arrancar. Items de tooling que
  Phase 6 dejo flotando:
  - Formalizar `make test-infra` con setup automatico (pytest venv).
  - Pre-commit hooks (bash syntax, markdown linting).
  - CI extendido con cobertura.
  - `docs/ARCHITECTURE.md`, `docs/RUNBOOK.md`, `docs/RELEASE.md`.

---
*Phase: 06-despliegue-lan-https*
*Completed: 2026-04-21*

## Self-Check: PASSED

Verificaciones post-summary:

- [x] `docs/DEPLOY_LAN.md` existe en disco (`test -f` exit 0, 740 lineas).
- [x] `tests/infra/deploy_smoke.sh` existe en disco y tiene bit ejecutable.
- [x] `tests/infra/test_deploy_lan_doc.py` existe en disco (24 tests pasando).
- [x] Commit `8cd513d` existe (Task 1: DEPLOY_LAN.md).
- [x] Commit `4e25f49` existe (Task 2: deploy_smoke.sh).
- [x] Commit `a363fcf` existe (Task 3: test_deploy_lan_doc.py).
- [x] Suite `tests/infra/`: 73 passed in 0.99s (sin regresiones).
- [x] `bash -n tests/infra/deploy_smoke.sh` exit 0.
- [x] Regresion verificada end-to-end: 3 tests fallan al manipular el doc
      (ufw rule, down -v, emoji) y vuelven a pasar tras restaurar.
- [x] Sin emojis en ninguno de los 3 archivos creados (coherencia con
      CLAUDE.md y rules/common/coding-style.md).
- [x] Pre-existing user changes (`.gitignore`, `api/main.py`, `.codex/`,
      `static/img/brand/nexo/logo-tab-navegador.png`) NO tocados — solo
      commits para archivos de `files_modified` del plan.
- [x] No hay deleciones en ninguno de los 3 commits
      (`git diff --diff-filter=D --name-only HEAD~3 HEAD` vacio).
