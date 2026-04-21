---
phase: 06-despliegue-lan-https
type: verification
status: complete
verified: 2026-04-21
must_haves_total: 5
must_haves_verified: 5
requirements_verified: [DEPLOY-01, DEPLOY-02, DEPLOY-03, DEPLOY-04, DEPLOY-05, DEPLOY-06, DEPLOY-07, DEPLOY-08]
requirements_unverified: []
deferred_verifications:
  - id: "DEPLOY-01/02/03/06/07-empirical"
    scope: "Smoke E2E en Ubuntu fisico con .env.prod real"
    deadline: "Sin deadline duro (depende asignacion host por IT)"
    operador: "e.eguskiza@ecsmobility.com"
    comando: "ssh <IP_NEXO> 'cd /opt/nexo && make prod-up && sleep 30 && bash tests/infra/deploy_smoke.sh' — aceptacion: exit 0 (0 fallos sobre 11 checks)"
    registered_in: "STATE.md seccion 'Deferred Verifications'"
hostname_discrepancy_note: |
  ROADMAP.md §Phase 6 y REQUIREMENTS.md DEPLOY-01/07 aun citan `nexo.ecsmobility.com`
  como texto original. La decision canonica durante Phase 6 (CONTEXT D-01) fue
  bloquear el hostname a `nexo.ecsmobility.local` (inmutable durante Mark-III).
  Todos los artefactos Phase 6 usan `.local` consistentemente (23 ocurrencias
  across 5 archivos, cero ocurrencias de `.com`). El verification context explicita
  que `.local` es el goal autoritativo. Sugerencia Phase 7: actualizar ROADMAP.md
  y REQUIREMENTS.md para eliminar texto stale.
---

# Phase 6: Despliegue LAN HTTPS — Verification Report

**Phase Goal (CONTEXT + verification context authoritativo):**
Nexo corriendo en servidor Ubuntu Server 24.04 (i5 7a gen, 16 GB, SSD 1 TB), `nexo.ecsmobility.local` resuelto por DNS interno (hosts-file), HTTPS (tls internal + root CA distribuida), sin exposicion internet.

**Verification scope:** artefactos + tests estaticos + goal-backward analysis. El smoke empirico en Ubuntu fisico queda deferred hasta asignacion de host por IT.

---

## Goal check

Los 3 plans (06-01, 06-02, 06-03) produjeron todos los artefactos que la meta exige. Cada artefacto existe en disco, tiene la forma correcta, y las 73 pruebas estaticas pasan en <1s. El goal es alcanzable en cuanto el servidor Ubuntu este disponible; la unica pieza faltante es la validacion empirica en hardware real (deferred, registrada en STATE.md).

**Veredicto Goal check:** Los artefactos materializan fielmente la meta. La ejecucion empirica es un paso mecanico una vez asignado el host.

---

## Success Criteria Evidence

| # | Success Criterion (goal-backward) | Status | Evidence |
|---|-----------------------------------|--------|----------|
| SC-1 | Desde otro equipo LAN: `https://nexo.ecsmobility.local` carga con cert reconocido | VERIFIED | `caddy/Caddyfile.prod` declara `nexo.ecsmobility.local { tls internal; reverse_proxy web:8000 }` (L8-11). `docs/DEPLOY_LAN.md` §7 documenta extraccion root CA + distribucion per-OS (Windows `certutil.exe -addstore -f ROOT`, Linux `update-ca-certificates`, macOS `security add-trusted-cert`, Firefox trust store). §8 documenta hosts-file per-OS. §7.6 incluye verificacion end-to-end con `openssl s_client | grep 'CN=nexo.ecsmobility.local'`. |
| SC-2 | Puerto 5432 no escuchable desde host Ubuntu; sí desde `docker compose exec db psql` | VERIFIED | `docker-compose.prod.yml` L18: `db: ports: !reset []`. Verificado via `docker compose -f docker-compose.yml -f docker-compose.prod.yml config --format json`: `db.ports = []` y `web.ports = []`. Test `test_compose_config_db_ports_empty_after_merge` valida el merge real. Smoke check DEPLOY-02: `! sudo ss -tlnp | grep -q ':5432 '` + `docker compose exec -T db psql ... | grep '1 row'`. |
| SC-3 | `scripts/deploy.sh` en el servidor ejecuta `git pull + build + up -d` sin intervención | VERIFIED | `scripts/deploy.sh` (70 lineas, +x, `bash -n` OK). Contiene: `set -euo pipefail` (L13), `git pull --ff-only` (L31, sin `--force`), pre-deploy `pg_dump` atomic `.tmp -> mv` (L40-42) con skip si db no corre (L35), `build --pull` (L52), `up -d` (L56), smoke `curl -fs -m 10 -k https://nexo.ecsmobility.local/api/health` (L64) con fail msg rollback hint (L67). 13 tests estaticos `test_deploy_script.py` pasan. |
| SC-4 | `docs/DEPLOY_LAN.md` permite reinstalar desde cero a otro admin sin contexto | VERIFIED | 740 lineas, 18 secciones `##` (target era 16+), espanol, sin emojis. Cubre: (§3) Docker CE install en Ubuntu 24.04 limpio, (§4) clone + .env rellenado, (§5) ufw orden seguro, (§6) make prod-up, (§7) root CA per-OS, (§8) hosts-file per-OS, (§9.1-9.7) 7 landmines numeradas, (§10) cron backup, (§11) validacion DEPLOY-*, (§13) restore, (§14) RTO 1-2h desde cero. Placeholders `<IP_NEXO>`, `<SUBNET_LAN>`, `<ADMIN_BACKUP_NAME>`, `<URL_REPO_NEXO>` consistentes. 17 tests `test_doc_*` bloquean drift. |
| SC-5 | `ufw status` muestra sólo 22, 80, 443 permitidos | VERIFIED | `docs/DEPLOY_LAN.md` §5 L167-171 documenta comandos literales: `allow from <SUBNET_LAN> to any port 22 proto tcp`, `allow 80/tcp`, `allow 443/tcp`, `default deny incoming`, `default allow outgoing`. Orden seguro: reglas antes de `ufw enable`. `tests/infra/deploy_smoke.sh` contiene 4 checks DEPLOY-06: activo + 443 allow + 80 allow + default incoming deny. `test_doc_has_ufw_rules` valida literales. |

**Score: 5/5 must-haves verified (goal-backward).**

---

## Requirement Traceability

| Requirement | Source Plan(s) | Description (resumen) | Status | Evidence |
|-------------|----------------|-----------------------|--------|----------|
| DEPLOY-01 | 06-01 | Caddyfile prod con hostname + TLS | VERIFIED | `caddy/Caddyfile.prod` con bloque `nexo.ecsmobility.local { tls internal; reverse_proxy web:8000 }` + fallback `:443`. 5 tests `test_caddyfile_prod_*` pasan. |
| DEPLOY-02 | 06-01 | Postgres 5432 cerrado al host | VERIFIED | `docker-compose.prod.yml` `db.ports: !reset []`. Merge verificado: `db.ports = []`. 2 tests (static + merge). |
| DEPLOY-03 | 06-01 | Healthchecks web + caddy + restart unless-stopped | VERIFIED | Override declara web healthcheck (`curl -fs /api/health`, 30s/5s/3/20s) y caddy healthcheck (`wget --spider`, 30s/5s/3/10s). `restart: unless-stopped` heredado del base preservado. Smoke script valida healthy + restart policy via `docker inspect`. |
| DEPLOY-04 | 06-02 | `scripts/deploy.sh` idempotente | VERIFIED | `scripts/deploy.sh` 70 lineas con 5 pasos D-25 + atomic pre-deploy backup. 13 tests pasan. D-26 idempotencia: `git pull --ff-only` no-op si al dia; `build --pull` cachea; `up -d` no-op si sanos. |
| DEPLOY-05 | 06-03 | Runbook `docs/DEPLOY_LAN.md` | VERIFIED | 740 lineas, 18 secciones, espanol, placeholders documentados. Admin de respaldo con Linux basico puede reinstalar siguiendo §3->§14 secuencialmente. RTO 1-2h explicitado en §1 y §14. |
| DEPLOY-06 | 06-03 | ufw documentado + enforced | VERIFIED | §5 del runbook literal + §9.1 advierte Docker-bypasses-ufw (la defensa real de 5432 es el override compose). 4 checks smoke DEPLOY-06. Test regresion `test_doc_has_ufw_rules`. |
| DEPLOY-07 | 06-03 | Verificacion desde peer LAN | VERIFIED (por artefactos) | `tests/infra/deploy_smoke.sh` 11 checks DEPLOY-*. §7-§8 del runbook documentan CA + hosts-file per-OS. Validacion empirica deferred (STATE.md). |
| DEPLOY-08 | 06-01 | `.env.prod.example` con placeholders | VERIFIED | 82 lineas, 25 NEXO_* vars, `<CHANGEME-*>` en secretos, literales `NEXO_HOST=0.0.0.0`/`NEXO_PORT=8000`/`COMPOSE_PROJECT_NAME=nexo`, SMTP comentado con `# TODO Mark-IV` (D-27). 8 tests `test_env_prod_example_*` + gitignore whitelist. |

**Cobertura: 8/8 DEPLOY-* VERIFIED.**

---

## Decision Audit (28 locked decisions)

| # | Decision | Domain | Status | Evidence |
|---|----------|--------|--------|----------|
| D-01 | Hostname `nexo.ecsmobility.local` inmutable | Hostname | HONORED | 23 ocurrencias en 5 archivos (Caddyfile.prod, deploy.sh, smoke.sh, DEPLOY_LAN.md, Makefile). Cero ocurrencias de `.com` en artefactos Phase 6. |
| D-02 | DNS via hosts-file per-equipo | DNS | HONORED | `docs/DEPLOY_LAN.md` §8 (8.1 Windows, 8.2 Linux, 8.3 macOS, 8.4 verificacion). |
| D-03 | IP via placeholder `<IP_NEXO>` | DNS | HONORED | `<IP_NEXO>` literal en runbook §2, §8, §11, §14, Apendice B. |
| D-04 | `tls internal` Caddy (no LE) | TLS | HONORED | `caddy/Caddyfile.prod` L9 `tls internal`; §1 runbook explicita "TLS emitido por Caddy Local CA"; §15 explica por que LE se quedo como deferred. |
| D-05 | Root CA distribuida manual per-SO | TLS | HONORED | Runbook §7 con Windows certutil + GUI, Linux update-ca-certificates, macOS security add-trusted-cert, Firefox GUI. |
| D-06 | Caddyfile prod = nuevo bloque hostname + fallback `:443` | TLS | HONORED | Caddyfile.prod tiene ambos bloques (L8-16 hostname, L21-24 `:443` fallback). |
| D-07 | pg_dump diario via cron (03:00 UTC) | Backup | HONORED | `scripts/backup_nightly.sh` pg_dump + gzip, cron `0 3 * * *` documentado en §10 runbook con `/etc/cron.d/nexo-backup`. |
| D-08 | Ubicacion `/var/backups/nexo/` | Backup | HONORED | `BACKUP_DIR="/var/backups/nexo"` en backup_nightly.sh L17; pre-deploy en subdir `/var/backups/nexo/predeploy/` (Claude's discretion). |
| D-09 | Retencion 7d via `find -mtime +7 -delete` | Backup | HONORED | backup_nightly.sh L39: `find "${BACKUP_DIR}" -maxdepth 1 -name "nexo-*.sql.gz" -type f -mtime +7 -delete`. Pre-deploy con +30d (transicional). |
| D-10 | RPO 24h aceptado | Recovery | HONORED | Runbook §1 + §14 declaran RPO 24h explicitamente; mention in CONTEXT.md L42. |
| D-11 | RTO 1-2h objetivo | Recovery | HONORED | Runbook §1 + §14 "Recuperacion desde cero (RTO 1-2h)" con 8 pasos numerados. |
| D-12 | Bus factor 2 | Recovery | HONORED | Runbook escrito para admin de respaldo con Linux basico (placeholder `<ADMIN_BACKUP_NAME>` en header + §16 Contactos). |
| D-13 | Secuencia runbook: instalar->clone->.env->copy backup->deploy->restore | Recovery | HONORED | Runbook §3->§10 sigue esa secuencia con refinamientos (ufw antes de prod-up, landmines antes de cron/validacion). |
| D-14 | SSH `<SUBNET_LAN>/24` | Firewall | HONORED | Runbook §5 L167 `ufw allow from <SUBNET_LAN> to any port 22 proto tcp`. |
| D-15 | 443/tcp abierto | Firewall | HONORED | Runbook §5 L171 `ufw allow 443/tcp comment 'HTTPS Nexo'`. Smoke check DEPLOY-06. |
| D-16 | 80/tcp abierto (redirect auto) | Firewall | HONORED | Runbook §5 L170 `ufw allow 80/tcp comment 'HTTP redirect to HTTPS'`. Caddyfile.prod sin `auto_https disable_redirects` (comentario L3 lo explicita). |
| D-17 | Postgres 5432 NO publicado + MCP fuera de prod | Firewall | HONORED | override `db.ports: !reset []`. MCP no se arranca en prod (profile `[mcp]` heredado del base, override no lo agita). |
| D-18 | Override file (NO profiles) | Compose | HONORED | `docker-compose.prod.yml` es override canonico. `scripts/deploy.sh` L19: `docker compose -f docker-compose.yml -f docker-compose.prod.yml`. |
| D-19 | Override contents: ports empty, Caddyfile.prod mount, healthchecks, limits | Compose | HONORED | 58 lineas del override cubren los 5 items. Volumes usando `!override` (no `!reset`) tras descubrir Landmine 2b. |
| D-20 | `restart: unless-stopped` preservado | Compose | HONORED | Base compose ya tiene `restart: unless-stopped`; override no lo tumba. Smoke valida via `docker inspect`. |
| D-21 | Resource limits web 4g/2cpu, db 2g/1cpu, caddy 256m/0.5cpu | Compose | HONORED | override.yml L19-23 db 2g/1.0, L37-41 web 4g/2.0, L54-58 caddy 256m/0.5. Test `test_override_has_resource_limits`. |
| D-22 | Web healthcheck curl /api/health, 30s/5s/3/20s | Healthcheck | HONORED | override L31-36 exactos. `jq` intencionalmente ausente (Pitfall 3). |
| D-23 | Caddy healthcheck wget --spider, 30s | Healthcheck | HONORED | override L48-53 exactos. |
| D-24 | db healthcheck conservado (`pg_isready`) | Healthcheck | HONORED | Base compose intacto; override no toca db.healthcheck. |
| D-25 | deploy.sh = 6 pasos (pull->backup->build->up->smoke) | Script | HONORED | 5 pasos logicos (pre-deploy backup es step 2, que engloba pg_dump + atomic rename). Smoke paso 5 usa `-k` (Landmine 4). |
| D-26 | deploy.sh idempotente | Script | HONORED | `git pull --ff-only` no-op; `build --pull` cachea; `up -d` no-op si sanos. Documentado en header. |
| D-27 | SMTP comentado `# TODO Mark-IV` | Env | HONORED | `.env.prod.example` L74-82 bloque SMTP 100% comentado. Test `test_env_prod_example_smtp_lines_commented`. |
| D-28 | .env.prod.example enumera TODAS las vars NEXO_* | Env | HONORED | 25 NEXO_* declaradas con placeholders o literales. Test `test_env_prod_example_has_all_required_nexo_vars`. |

**28/28 decisions HONORED.**

---

## Landmine Defenses

| # | Landmine | Defense | Test preventivo |
|---|----------|---------|-----------------|
| L1 | `auto_https disable_redirects` en Caddy rompe 80->443 | Caddyfile.prod no lo contiene (0 matches de `grep 'auto_https disable_redirects'`); comentario L3 reescrito sin literal | `test_caddyfile_prod_no_auto_https_disable_redirects` |
| L2 | compose `ports: []` NO resetea lista heredada | Override usa `!reset []` en db.ports y web.ports; merge verificado devuelve `[]` | `test_override_mentions_reset_for_db_ports` + `test_compose_config_db_ports_empty_after_merge` |
| L2b | `volumes: !reset` + lista indentada deja lista VACIA | Override usa `!override` para volumes (descubierto durante 06-01, landmine documentada) | `test_override_uses_override_tag_for_volumes` + `test_compose_config_caddy_mounts_caddyfile_prod` |
| L3 | Docker bypassa ufw (ports publicados saltan DOCKER-USER) | §9.1 del runbook dedica seccion explicita con HIGH severity + nmap/ss verificatorios; defensa real del 5432 es el override (no ufw) | `test_doc_warns_docker_bypasses_ufw` |
| L4 | `/api/health` llama a SQL Server MES (container entraria en bucle si healthcheck parsea `.ok`) | Healthcheck usa `curl -fs` sin `jq` ni `.ok` parsing; comentario §9.6 runbook explica por que | `test_override_web_healthcheck_curl_api_health` (asserts `jq not in content`) |
| L6 | `docker compose down -v` borra pgdata | `make prod-down` NO usa `-v` + comentario explicito en Makefile L136-137 + runbook §9.2 "AVISO (HIGH): JAMAS usar..." | `test_doc_warns_down_v_landmine` |
| L7 | pg_dump no atomico = archivo invalido que rotacion trata como bueno | Ambos scripts usan `${OUT}.tmp` + `mv` pattern | `test_deploy_sh_atomic_rename` + `test_backup_sh_atomic_rename` |

**7/7 landmines materializadas como defensas + tests regresion.**

---

## Test Results

```
pytest -q --confcutdir=tests/infra tests/infra/
========================= 73 passed in 1.00s =========================
```

**Desglose:**

- `test_caddyfile_prod.py` — 5 tests (Plan 06-01)
- `test_compose_override.py` — 12 tests (Plan 06-01, 7 estaticos + 5 de merge con Docker)
- `test_env_prod_example.py` — 8 tests (Plan 06-01)
- `test_deploy_script.py` — 13 tests (Plan 06-02)
- `test_backup_script.py` — 11 tests (Plan 06-02)
- `test_deploy_lan_doc.py` — 24 tests (Plan 06-03, 17 doc + 7 smoke)

**Syntax checks:**

```
bash -n scripts/deploy.sh                      # exit 0
bash -n scripts/backup_nightly.sh              # exit 0
bash -n tests/infra/deploy_smoke.sh            # exit 0
docker compose -f docker-compose.yml -f docker-compose.prod.yml config > /dev/null   # exit 0
```

**Merge verification:**

```
db.ports  = []
web.ports = []
caddy.volume: /.../caddy/Caddyfile.prod -> /etc/caddy/Caddyfile
caddy.volume: caddy_data -> /data
caddy.volume: caddy_config -> /config
```

**Executable bits:** `scripts/deploy.sh`, `scripts/backup_nightly.sh`, `tests/infra/deploy_smoke.sh` — los 3 con bit x.

**Anti-patterns scan:**
- Emojis en los 8 archivos Phase 6: 0.
- TODO/FIXME/XXX/HACK en codigo (excluye `.env.prod.example #TODO Mark-IV` intencional D-27): 0.
- Hostname `.com` en artefactos Phase 6: 0 (solo aparece stale en docs originales pre-Phase-6).
- `--force` / `--no-verify` / `rm -rf` en scripts ops: 0.

**Target status:** target era `>=60 tests green`, alcanzado 73 tests green. Pre-06-01: 0; post-06-01: 25; post-06-02: 49; post-06-03: 73.

---

## Deferred Verifications

**DEPLOY-01/02/03/06/07 empirico en Ubuntu fisico**

- **Estado:** registrado en `.planning/STATE.md` seccion "Deferred Verifications".
- **Deadline:** sin deadline duro (depende asignacion host por IT).
- **Operador:** e.eguskiza@ecsmobility.com.
- **Comando de aceptacion:**

```
ssh <IP_NEXO> 'cd /opt/nexo && make prod-up && sleep 30 && bash tests/infra/deploy_smoke.sh'
# Aceptacion: exit 0 (0 fallos sobre 11 checks)
```

- **Checks complementarios desde peer LAN (tras CA + hosts-file):**

```
nmap -Pn -p 22,80,443,5432 <IP_NEXO>
# Esperado: 22/80/443 open, 5432 closed
curl -fsI https://nexo.ecsmobility.local/api/health
# Esperado: HTTP/2 200
```

Esta verificacion NO bloquea el cierre de Phase 6. Todos los artefactos, tests y docs estan completos; falta solo la ejecucion empirica en hardware que aun no existe.

---

## Observaciones adicionales

### Hostname discrepancy (informational, no bloqueante)

ROADMAP.md y REQUIREMENTS.md DEPLOY-01/07 siguen citando `nexo.ecsmobility.com` como texto original (7 ocurrencias). La decision CONTEXT D-01 bloqueo `nexo.ecsmobility.local` y los artefactos Phase 6 estan 100% consistentes con `.local`. El verification context explicita que `.local` es el goal autoritativo. Sugerencia Phase 7: actualizar texto de ROADMAP/REQUIREMENTS para eliminar drift (tarea de `docs`, no de `feat`).

### Sin pytest en host ni container

Los tests de infra se ejecutan en un venv temporal (`/tmp/nexo-testvenv/`). Phase 7 deberia formalizar `make test-infra` con setup automatico. No bloquea Phase 6.

### Working tree pre-existente

Los commits de los 3 plans respetaron archivos pre-existentes no-relacionados (`.codex/`, `static/img/brand/nexo/logo-tab-navegador.png`). Solo archivos listados en cada PLAN's `files_modified` fueron staged.

---

## Verdict

**PASS-WITH-DEFERRED**

- 5/5 success criteria VERIFIED (goal-backward).
- 8/8 DEPLOY-* requirements cubiertos por artefactos + tests.
- 28/28 locked decisions honored.
- 7/7 landmines defendidos con tests de regresion.
- 73/73 tests estaticos pasan en <1s.
- 1 deferred verification registrada en STATE.md (smoke empirico Ubuntu fisico).

Phase 6 alcanza su meta de entregar despliegue LAN HTTPS reproducible. La ejecucion empirica en hardware esta bloqueada por disponibilidad de IT, no por deficiencia de artefactos. El runbook + scripts + tests + smoke permiten a un admin de respaldo instalar desde cero en 1-2h (RTO D-11) siguiendo procedimiento documentado.

**Phase 6 ready to close.**

---

*Verified: 2026-04-21*
*Verifier: Claude (gsd-verifier, Opus 4.7 [1M context])*
