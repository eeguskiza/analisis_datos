---
phase: 06-despliegue-lan-https
plan: 01
subsystem: infra
tags: [docker-compose, caddy, tls, env, pytest, compose-override]

# Dependency graph
requires:
  - phase: 02-auth-multiusuario
    provides: "NEXO_SECRET_KEY, NEXO_SESSION_* vars consumidas por el middleware de auth"
  - phase: 04-consultas-pesadas
    provides: "NEXO_QUERY_LOG_RETENTION_DAYS, NEXO_AUTO_REFRESH_STALE_DAYS en .env.prod.example"
  - phase: 05-ui-roles
    provides: "Caddyfile dev estable (reverse-proxy web:8000) sobre el que se calca el prod"

provides:
  - "caddy/Caddyfile.prod con hostname nexo.ecsmobility.local + tls internal + redirect 80->443 automatico"
  - "docker-compose.prod.yml override que cierra Postgres 5432 al host, monta Caddyfile.prod, anade healthchecks web/caddy y resource limits por servicio"
  - ".env.prod.example canonico con todas las NEXO_* como <CHANGEME-*>, literales NEXO_HOST=0.0.0.0/NEXO_PORT=8000/COMPOSE_PROJECT_NAME=nexo, SMTP comentado con # TODO Mark-IV"
  - ".gitignore reforzado: .env.prod ignorado, !.env.prod.example whitelisted"
  - "tests/infra/ suite (25 tests) — 17 estaticos + 8 dinamicos via docker compose config"
affects: [06-02-deploy-script-backup-ufw, 06-03-runbook-smoke, 07-*]

# Tech tracking
tech-stack:
  added:
    - "Compose v2 YAML tags: `!reset []` para resetear listas a vacio, `!override` para reemplazar listas con contenido nuevo"
    - "tests/infra/ directorio nuevo para validaciones estaticas de config de despliegue"
  patterns:
    - "Override compose: base dev intacto + override prod via `-f docker-compose.yml -f docker-compose.prod.yml`"
    - "Test de infra en dos niveles: parse directo (sin Docker, Wave 0) + `docker compose config --format json` (skippable si docker ausente)"
    - "Whitelist explicita en .gitignore (`!.env.prod.example`) como defensa contra patrones amplios futuros"

key-files:
  created:
    - "caddy/Caddyfile.prod — config Caddy prod (hostname fijo + tls internal + fallback :443)"
    - "docker-compose.prod.yml — override prod (ports reset, volumes override, healthchecks, limits)"
    - ".env.prod.example — template env prod con placeholders"
    - "tests/infra/__init__.py — marker de paquete pytest"
    - "tests/infra/test_caddyfile_prod.py — 5 tests de validacion estatica"
    - "tests/infra/test_compose_override.py — 12 tests (7 estaticos + 5 de merge con docker)"
    - "tests/infra/test_env_prod_example.py — 8 tests de completitud y ausencia de secretos"
  modified:
    - ".gitignore — anadidas reglas .env.prod, .env.*.local y whitelist de ejemplos"

key-decisions:
  - "Compose v2 tag switch: `!reset` solo para ports (a lista vacia); `!override` para volumes donde queremos reemplazar con nueva lista. Descubierto durante verificacion que `volumes: !reset` + lista indentada deja la lista vacia silenciosamente (Landmine 2b)."
  - "Healthcheck web (curl -fs /api/health) intencionalmente NO parsea JSON: MES caido devuelve 200 con ok:false y el healthcheck sigue OK; de lo contrario el container entraria en bucle de reinicio (Pitfall 3)."
  - ".gitignore whitelist explicita `!.env.example` y `!.env.prod.example` para que los templates sobrevivan a cualquier patron amplio `.env.*` que alguien anada en el futuro."
  - "Solo se modifica el hunk de Credenciales del .gitignore; la entrada `.codex/` preexistente queda fuera de este plan (critical_constraint del prompt)."

patterns-established:
  - "Test de infra Wave 0: correr sin servidor real, solo validar estructura de archivos y output de herramientas disponibles (docker CLI si esta, skip si no)."
  - "Comentario docstring en el override.yml explicando exactamente que Landmine cubre cada tag YAML (`!reset` vs `!override`)."
  - "Acceptance criteria del plan se traducen en pytest assertions mas especificas: en vez de `grep -q 'curl -fs'` (que falla con el JSON-list form), los tests validan los tres tokens por separado."

requirements-completed: [DEPLOY-01, DEPLOY-02, DEPLOY-03, DEPLOY-08]

# Metrics
duration: 34 min
completed: 2026-04-21
---

# Phase 06-despliegue-lan-https Plan 01: Compose Override Prod + Caddyfile Prod + Env Template Summary

**Override Compose v2 con `!reset []` en ports y `!override` en volumes, Caddyfile prod con `nexo.ecsmobility.local` y `tls internal`, `.env.prod.example` con 25 NEXO_* como placeholders, y suite de 25 tests de infra (Wave 0) que congelan el contrato antes de que los plans 06-02/06-03 lo consuman.**

## Performance

- **Duration:** ~34 min
- **Started:** 2026-04-21T16:55Z (aprox)
- **Completed:** 2026-04-21T17:09Z
- **Tasks:** 3 de 3 (todas autonomous, sin checkpoints)
- **Files modified:** 8 (7 creados + 1 editado)

## Accomplishments

- Caddyfile prod con hostname inmutable `nexo.ecsmobility.local` + `tls internal` + bloque `:443` fallback IP-directa; SIN flag global que desactive redirect, por lo que Caddy mantiene 80->443 automatico (D-01, D-04, D-06, D-16).
- Override prod compose que cierra 5432 al host (DEPLOY-02), monta `caddy/Caddyfile.prod:/etc/caddy/Caddyfile:ro`, descarta el bind `./tests` del dev, anade healthchecks `curl -fs /api/health` (web) y `wget --spider` (caddy), y fija limits 4g/2cpu web, 2g/1cpu db, 256m/0.5cpu caddy (D-19 a D-24).
- `.env.prod.example` canonico: 25 NEXO_* requeridas con `<CHANGEME-*>` en secretos, literales `NEXO_HOST=0.0.0.0`, `NEXO_PORT=8000`, `COMPOSE_PROJECT_NAME=nexo`; bloque SMTP comentado con marca `# TODO Mark-IV` (D-27, D-28).
- `.gitignore` cerrado: `.env.prod` ignorado, `.env.*.local` ignorado, `!.env.example` y `!.env.prod.example` whitelisted.
- Suite tests/infra/ con 25 tests (5 + 12 + 8). Los 17 tests estaticos pasan siempre; los 8 dinamicos se ejecutan si `docker compose` esta disponible (5 pasan hoy, 3 quedarian skipped en CI sin docker).
- Merge `docker compose -f docker-compose.yml -f docker-compose.prod.yml config` validado: db.ports y web.ports vacios, web.volumes solo `./data` y `./informes`, caddy.volumes con Caddyfile.prod + caddy_data + caddy_config.

## Task Commits

Cada tarea committeada atomicamente en feature/Mark-III:

1. **Task 1: caddy/Caddyfile.prod + tests** — `d1ae920` (feat)
2. **Task 2: docker-compose.prod.yml + tests** — `d76e8f7` (feat)
3. **Task 3: .env.prod.example + .gitignore + tests** — `839dbf5` (feat)

_(Plan metadata commit se crea al cierre del ciclo execute-phase.)_

## Files Created/Modified

- `caddy/Caddyfile.prod` (24 lineas) — configuracion Caddy produccion.
- `docker-compose.prod.yml` (58 lineas) — override prod con `!reset` y `!override`.
- `.env.prod.example` (83 lineas) — template env con placeholders.
- `.gitignore` (modificado, +4 lineas en Credenciales) — `.env.prod`, `.env.*.local`, whitelist de ejemplos.
- `tests/infra/__init__.py` (vacio) — marker pytest.
- `tests/infra/test_caddyfile_prod.py` (5 tests, 49 lineas) — validacion estatica del Caddyfile prod.
- `tests/infra/test_compose_override.py` (12 tests, 199 lineas) — validacion estatica + merge con docker.
- `tests/infra/test_env_prod_example.py` (8 tests, 131 lineas) — validacion de .env.prod.example y .gitignore.

## Decisions Made

- **D-01/D-06 cerrados**: hostname `nexo.ecsmobility.local` literal en Caddyfile.prod; bloque `:443` se conserva como fallback para quien escriba `https://<IP_NEXO>`.
- **D-19 cerrado**: `./caddy/Caddyfile.prod:/etc/caddy/Caddyfile:ro` montado en el servicio `caddy` del override.
- **D-22 cerrado**: healthcheck web `curl -fs http://localhost:8000/api/health`, interval 30s, timeout 5s, retries 3, start_period 20s. Sin `jq` por diseno (Pitfall 3).
- **D-23 cerrado**: healthcheck caddy `wget --spider -q http://localhost/`, interval 30s.
- **D-21 cerrado**: limits web 4g/2cpu, db 2g/1cpu, caddy 256m/0.5cpu — todos con "cpus" como string y "memory" con sufijo `g`/`m`.
- **D-27 cerrado**: bloque SMTP comentado al 100% con marca `# TODO Mark-IV`; test `test_env_prod_example_smtp_lines_commented` falla si alguien descomenta.
- **D-28 cerrado**: `NEXO_HOST=0.0.0.0`, `NEXO_PORT=8000`, `COMPOSE_PROJECT_NAME=nexo`, `NEXO_PG_APP_USER=nexo_app`, `NEXO_APP_DB=ecs_mobility`, `NEXO_MES_DB=dbizaro` como literales; el resto como `<CHANGEME-*>`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Landmine 2b: `volumes: !reset` seguido de lista indentada deja la lista vacia**

- **Found during:** Task 2 (verificacion de `docker compose config` merged).
- **Issue:** El plan (y RESEARCH.md Topic 3) prescribe `volumes: !reset` seguido de una lista indentada para reemplazar la lista de volumes heredada del base. Al ejecutar `docker compose -f docker-compose.yml -f docker-compose.prod.yml config --format json`, el merged caddy aparecia con `volumes: null` y web con `volumes: null` — la lista se perdia silenciosamente. El tag `!reset` de Compose v2 esta disenado para resetear a vacio (equivalente a `!reset []`): cuando se coloca al inicio de un bloque con lista indentada debajo, consume el bloque entero como un reset a nulo. Si no se hubiera detectado, el container caddy habria arrancado sin Caddyfile.prod montado (bootloop) y el container web sin `./data`/`./informes` (perdida de informes al primer restart) — DEPLOY-01 y DEPLOY-03 fallarian sin que ningun test estatico Level 1 lo notara.
- **Fix:** Cambiar `volumes: !reset` a `volumes: !override` en los servicios `web` y `caddy`. `!override` es el tag canonico de Compose v2.24+ para reemplazar una lista heredada con nueva contenido (diferente semantica a `!reset`, que es "borrar"). `ports: !reset []` se mantiene sin cambios en db y web porque si queremos vaciarlos (no republicarlos). Documentado en el header del override.yml con comentario explicito sobre Landmine 2b.
- **Files modified:** `docker-compose.prod.yml` (2 lineas: web.volumes y caddy.volumes), `tests/infra/test_compose_override.py` (anadidos 3 tests que hubieran detectado el bug antes: `test_override_uses_override_tag_for_volumes` estatico, `test_compose_config_web_volumes_exclude_tests` y `test_compose_config_caddy_mounts_caddyfile_prod` dinamicos).
- **Verification:** `docker compose -f docker-compose.yml -f docker-compose.prod.yml config --format json` ahora emite `web.volumes` con 2 entradas (`/app/data`, `/app/informes`) y `caddy.volumes` con 3 (`/etc/caddy/Caddyfile`, `/data`, `/config`). Los 25 tests de `tests/infra/` pasan.
- **Committed in:** `d76e8f7` (Task 2 commit unico).

**2. [Rule 1 - Bug] Comentario en Caddyfile.prod contenia el literal `auto_https disable_redirects`**

- **Found during:** Task 1 (primera ejecucion del acceptance criterion `grep -q 'auto_https disable_redirects'` esperado exit 1).
- **Issue:** El comentario explicativo al principio del Caddyfile.prod decia "SIN `auto_https disable_redirects`" — el string literal en un comentario hace que grep -q lo encuentre y el test `test_caddyfile_prod_no_auto_https_disable_redirects` (que hace `"auto_https disable_redirects" not in content`) falla aunque el flag no este activo funcionalmente.
- **Fix:** Reescribir el comentario sin mencionar el literal: "Sin el flag global de Caddy que desactiva redirects automaticos, para que Caddy inserte redirect 80->443 por defecto (D-16)". La decision sigue documentada; la busqueda de strings no confunde comentario con configuracion activa.
- **Files modified:** `caddy/Caddyfile.prod` (1 linea de comentario).
- **Verification:** `grep -q 'auto_https disable_redirects' caddy/Caddyfile.prod` exit 1. Los 5 tests de `test_caddyfile_prod.py` pasan.
- **Committed in:** `d1ae920` (Task 1 commit unico).

**3. [Rule 1 - Bug] Acceptance criterion `grep -q 'curl -fs'` incompatible con Docker JSON-list healthcheck**

- **Found during:** Task 2 (check de acceptance criteria despues de crear el override).
- **Issue:** El plan formula la acceptance como `grep -q 'curl -fs' docker-compose.prod.yml` esperando exit 0. Pero la healthcheck declarada como `test: ["CMD", "curl", "-fs", "http://localhost:8000/api/health"]` (forma recomendada Docker "exec form") escribe `curl` y `-fs` como dos items de lista separados por coma y espacio — `grep -q 'curl -fs'` NO matchea (busca exactamente `curl -fs` con espacio simple). El espiritu del criterio ("healthcheck usa curl con -fs contra /api/health") se cumple, pero el grep literal fallaria.
- **Fix:** En los tests pytest, validar los tres tokens por separado (`"curl"`, `"-fs"`, `/api/health` presentes en el YAML) en vez de exigir la secuencia literal `curl -fs`. La decision tecnica de mantener JSON-list form es correcta (no depende del shell del container y es la forma canonica recomendada por Docker).
- **Files modified:** Ninguno adicional — los tests del plan ya usan la forma tokenizada; la acceptance criterion del plan es la que queda desactualizada respecto a la realidad del YAML. Documentado aqui para futuros verificadores.
- **Verification:** `grep -q '"curl"' docker-compose.prod.yml && grep -q '"-fs"' docker-compose.prod.yml && grep -q '/api/health' docker-compose.prod.yml` todos exit 0. Test `test_override_web_healthcheck_curl_api_health` pasa.
- **Committed in:** No requiere commit separado; cubierto por `d76e8f7`.

---

**Total deviations:** 3 auto-fixed (3 x Rule 1 — bugs/mismatches detectados durante verificacion).
**Impact on plan:** Los 3 eran necesarios para que el plan produjera artefactos correctos. Landmine 2b es la mas critica: sin el fix, `docker compose up` habria dejado caddy sin Caddyfile.prod y web sin volumes persistentes, rompiendo DEPLOY-01 y DEPLOY-03 en el primer arranque real de prod. No hay scope creep — todas las correcciones estan dentro del alcance del plan.

## Landmines Cubiertas (como tests)

Resumen de las landmines documentadas en RESEARCH.md que este plan deja atrapadas por tests automatizados, listas para que 06-02/06-03 no las reintroduzcan:

| Landmine | Ubicacion | Test preventivo |
|----------|-----------|-----------------|
| L1 (Caddy) — `auto_https disable_redirects` rompe redirect 80->443 | `caddy/Caddyfile.prod` | `test_caddyfile_prod_no_auto_https_disable_redirects` |
| L2 (Compose) — `ports: []` NO resetea lista heredada | `docker-compose.prod.yml` | `test_override_mentions_reset_for_db_ports` + `test_compose_config_db_ports_empty_after_merge` |
| L2b (Compose) — `volumes: !reset` + lista indentada deja la lista vacia | `docker-compose.prod.yml` | `test_override_uses_override_tag_for_volumes` + `test_compose_config_web_volumes_exclude_tests` + `test_compose_config_caddy_mounts_caddyfile_prod` |
| Pitfall 3 (Healthcheck) — parsear JSON del /api/health con `jq` causa bucles de restart cuando MES esta caido | `docker-compose.prod.yml` | `test_override_web_healthcheck_curl_api_health` (asserts `jq not in content`) |
| L10 (Naming) — sin `COMPOSE_PROJECT_NAME=nexo` los containers seran `analisis_datos-web-1` | `.env.prod.example` | `test_env_prod_example_has_compose_project_name` |
| D-27 (Secrets) — SMTP descomentado hace que pydantic-settings intente conectar al arrancar | `.env.prod.example` | `test_env_prod_example_smtp_lines_commented` + `test_env_prod_example_has_smtp_todo_markiv` |
| DEPLOY-08 (Secrets) — un password real pegado accidentalmente queda commiteable | `.env.prod.example` | `test_env_prod_example_no_real_secrets` (regex guard) |
| Gitignore whitelist — `.env.*` amplio mata `.env.prod.example` | `.gitignore` | `test_gitignore_ignores_env_prod_but_not_example` |

## Verification Commands Ejecutados

```bash
# 1. Existencia de artefactos
test -f caddy/Caddyfile.prod                           # exit 0
test -f docker-compose.prod.yml                        # exit 0
test -f .env.prod.example                              # exit 0
test -f tests/infra/{__init__,test_caddyfile_prod,test_compose_override,test_env_prod_example}.py  # exit 0

# 2. Tests Wave 0 (sin docker)
/tmp/nexo-testvenv/bin/pytest -x -q --confcutdir=tests/infra \
    tests/infra/test_caddyfile_prod.py \
    tests/infra/test_env_prod_example.py
# 13 passed

# 3. Tests compose (con docker disponible)
/tmp/nexo-testvenv/bin/pytest -x -q --confcutdir=tests/infra tests/infra/test_compose_override.py
# 12 passed

# 4. Merge compose prod
docker compose -f docker-compose.yml -f docker-compose.prod.yml config > /dev/null
# exit 0

# 5. Gitignore
git check-ignore .env.prod.example   # exit 1 (NO ignorado — correcto)
git check-ignore .env.prod           # exit 0 (SI ignorado — correcto)
```

Nota sobre pytest en CI: ni el host WSL ni el container web tienen `pytest` instalado por defecto. Durante la ejecucion se creo un venv temporal en `/tmp/nexo-testvenv/` con `pytest==8.3.4` y se ejecuta con `--confcutdir=tests/infra` para evitar el `tests/conftest.py` global que importa SQLAlchemy (no necesario para tests de infra). El plan 06-02 o un plan de tooling posterior deberia formalizar un target Makefile `test-infra` que haga este setup automaticamente.

## Requirements Cubiertos

- **DEPLOY-01**: Caddyfile prod con hostname fijo + tls internal + redirect 80->443 automatico.
- **DEPLOY-02**: Override prod cierra 5432 al host (`!reset []`) y 8001 de la web; verificado via `docker compose config` en el test.
- **DEPLOY-03**: Healthchecks web (curl /api/health) y caddy (wget --spider) con periodicidad 30s y start_period ajustados.
- **DEPLOY-08**: `.env.prod.example` con todas las vars requeridas, placeholders, SMTP diferido con marca Mark-IV, y gitignore que protege secretos sin bloquear el template.

## Issues Encountered

- Tests de infra requieren pytest que no esta ni en host WSL ni en el container web. Resuelto con venv temporal en `/tmp/nexo-testvenv/`. No bloquea el plan pero deja abierto un item de tooling para 06-02 o posterior: formalizar `make test-infra` con setup automatico.
- Working tree tenia modificaciones pre-existentes (`.gitignore` con `.codex/`, `api/main.py`, `static/img/brand/nexo/logo-tab-navegador.png` nuevo) no relacionadas con este plan. Se respeto el critical_constraint del prompt: se aplico solo el hunk de "Credenciales" al .gitignore via `git apply --cached`; las demas modificaciones quedan en el working tree sin stagear.

## User Setup Required

None — no external service configuration required for this plan. El operador SI necesita `.env.prod` real (con secretos) cuando Plan 06-02 produzca el `scripts/deploy.sh`. El template ya esta listo.

## Next Phase Readiness

- **Plan 06-02 (deploy.sh + backup + ufw)** tiene todos los artefactos que consume: `caddy/Caddyfile.prod` (lo instalara vivo), `docker-compose.prod.yml` (lo referenciara desde `scripts/deploy.sh` con `-f docker-compose.yml -f docker-compose.prod.yml`), `.env.prod.example` (referenciara en el runbook para el paso `cp`).
- **Plan 06-03 (DEPLOY_LAN.md + smoke)** puede citar literalmente los contratos: hostname `nexo.ecsmobility.local`, IP placeholder `<IP_NEXO>`, backup path `/var/backups/nexo/`, logs via `docker compose logs`.
- **Pendientes para el handoff**:
  - Decidir si el smoke post-deploy de 06-02 usa los mismos healthchecks o hace un curl adicional a `/` (HTML de login).
  - Formalizar `make test-infra` si se considera que 25 tests de infra justifican un target Makefile propio (se puede diferir a Phase 7).

---
*Phase: 06-despliegue-lan-https*
*Completed: 2026-04-21*

## Self-Check: PASSED

Verificaciones post-summary:

- [x] `caddy/Caddyfile.prod` existe en disco (`test -f` exit 0).
- [x] `docker-compose.prod.yml` existe en disco.
- [x] `.env.prod.example` existe en disco.
- [x] `tests/infra/__init__.py` existe.
- [x] `tests/infra/test_caddyfile_prod.py` existe y 5 tests pasan.
- [x] `tests/infra/test_compose_override.py` existe y 12 tests pasan.
- [x] `tests/infra/test_env_prod_example.py` existe y 8 tests pasan.
- [x] `.gitignore` modificado (hunk de Credenciales), `.codex/` no tocado.
- [x] Commits existentes: `d1ae920`, `d76e8f7`, `839dbf5` — verificados via `git log --oneline -5`.
- [x] Merge `docker compose -f docker-compose.yml -f docker-compose.prod.yml config` exit 0.
- [x] `git check-ignore .env.prod` exit 0 (ignorado); `git check-ignore .env.prod.example` exit 1 (no ignorado).
- [x] No hay deleciones en ninguno de los tres commits (`git diff --diff-filter=D --name-only` vacio).
- [x] Sin emojis en ningun archivo creado (coherencia con CLAUDE.md y rules/common/coding-style.md).
