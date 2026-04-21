# Phase 6: Despliegue LAN HTTPS — Context

**Gathered:** 2026-04-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Nexo corriendo en un servidor Ubuntu Server 24.04 (i5 7ª gen, 16 GB, SSD 1TB,
IP estática en LAN ECS), accesible en `https://nexo.ecsmobility.local` desde
cualquier equipo de la red interna con cert HTTPS aceptado, sin exposición a
internet, con backup automático de la BD y runbook de recuperación
documentado para que un admin de respaldo pueda levantarlo desde cero en
1–2 horas.

**Estado de partida (commit `2b8ce59`):**
- `caddy/Caddyfile` actual = `tls internal` + `:443` sin hostname (autofirmado, sin distribuir root CA).
- `docker-compose.yml` = un único archivo dev: db expone 5433→5432, web expone 8001→8000, healthcheck SOLO en `db`, sin `mem_limit`/`cpus`, sin perfil prod.
- Sin `docker-compose.prod.yml`, sin `scripts/deploy.sh`, sin `.env.prod.example`, sin `docs/DEPLOY_LAN.md`.
- Postgres 5432 publicado al host (riesgo bajo en LAN, se cierra en prod).
- MCP service en `profiles: [mcp]` — no arranca con `make up`. Se mantiene fuera de prod.

</domain>

<decisions>
## Implementation Decisions

### Hostname y resolución DNS
- **D-01:** Hostname interno fijo = `nexo.ecsmobility.local`. Inmutable durante Mark-III. Pensado para que los usuarios bookmarkeen la URL una vez y no cambie.
- **D-02:** Resolución DNS = hosts-file por usuario para Phase 6. Cada equipo LAN añade `<IP_NEXO> nexo.ecsmobility.local` en `C:\Windows\System32\drivers\etc\hosts` (Windows) o `/etc/hosts` (Linux/Mac). Pasos por SO documentados en `docs/DEPLOY_LAN.md` con capturas.
- **D-03:** IP del Ubuntu de prod aún no asignada — `docs/DEPLOY_LAN.md` y `.env.prod.example` usan placeholder literal `<IP_NEXO>` para que el operador la reemplace antes del primer deploy.

### Certificado HTTPS
- **D-04:** Estrategia cert = `tls internal` (Caddy genera CA y certs automáticamente). Let's Encrypt DNS-01 descartado: ECS no controla un dominio público utilizable para este proyecto. Cert firmado por CA corporativa ECS descartado: no hay confirmación de que IT mantenga una.
- **D-05:** Distribución de la root CA = procedimiento documentado en `docs/DEPLOY_LAN.md`. Pasos: (1) `docker compose exec caddy cat /data/caddy/pki/authorities/local/root.crt`, (2) descargar el archivo, (3) instalar en cada equipo LAN (Windows: `certmgr.msc` → Trusted Root, Linux: `/usr/local/share/ca-certificates/` + `update-ca-certificates`, browser: import manual si no usa el trust store del SO). Una sola vez por equipo. Sin esto, browser muestra warning aceptable pero feo.
- **D-06:** Caddyfile prod = nuevo bloque con `nexo.ecsmobility.local { tls internal; reverse_proxy web:8000 }`. El bloque `:443` actual se conserva como fallback para acceso por IP directa durante el periodo de transición.

### Backup de Postgres (pgdata)
- **D-07:** Tipo de backup = `pg_dump` lógico diario via cron en el host Ubuntu. Comando: `docker compose exec -T db pg_dump -U <user> <db> | gzip > /var/backups/nexo/nexo-$(date +%Y%m%d).sql.gz`. Cron 03:00 UTC.
- **D-08:** Ubicación = `/var/backups/nexo/` en el propio servidor Ubuntu. Sync a NAS / equipo externo = **deferred** (mejora futura — Phase 6 no bloquea por esto).
- **D-09:** Retención = 7 días. Rotación con `find /var/backups/nexo/ -name "*.sql.gz" -mtime +7 -delete` en el mismo cron.
- **D-10:** RPO aceptable = 24h. En el peor caso (server muere a las 17:00 con backup de las 03:00) se pierden ~14h de audit_log/query_log/approvals. Reintroducción manual aceptada.

### Plan de recuperación
- **D-11:** RTO objetivo = 1–2h con runbook claro. Cualquier admin con SSH y acceso al backup más reciente debe poder levantar Nexo en otra máquina Ubuntu siguiendo el checklist.
- **D-12:** Bus factor objetivo = 2. `docs/DEPLOY_LAN.md` se escribe asumiendo que el lector tiene conocimientos básicos de Linux (no del proyecto Nexo). e.eguskiza + 1 persona de respaldo (a designar) son los responsables nombrados.
- **D-13:** Runbook incluye: (1) instalar Docker + plugin compose en Ubuntu nuevo, (2) `git clone <repo> && git checkout main`, (3) `cp .env.prod.example .env` y editar con valores reales, (4) copiar el último `nexo-YYYYMMDD.sql.gz` al servidor, (5) `bash scripts/deploy.sh`, (6) `docker compose exec -T db psql ... < nexo-*.sql.gz` para restore.

### Firewall (ufw)
- **D-14:** SSH (22/tcp) = `ufw allow from <subred LAN del Ubuntu>/24 to any port 22`. La subred exacta se determina al asignar la IP del servidor (placeholder `<SUBNET_LAN>` en `docs/DEPLOY_LAN.md`).
- **D-15:** HTTPS (443/tcp) = `ufw allow 443/tcp` (open a toda la LAN — el control de acceso lo hace la auth de Phase 2).
- **D-16:** HTTP (80/tcp) = `ufw allow 80/tcp` para que Caddy pueda hacer redirect automático a 443 (configurado por defecto cuando `:443` y `:80` coexisten en el Caddyfile). Sin redirect, los usuarios que escriban `http://nexo...` verían `connection refused`.
- **D-17:** Resto = `ufw default deny incoming`. Postgres 5432 NO se publica al host (acceso solo via `docker compose exec db psql`). MCP profile `[mcp]` no se arranca en prod, por tanto su contenedor ni siquiera existe.

### Estructura compose prod
- **D-18:** Estrategia = `docker-compose.prod.yml` como **override file** (NO profiles). `docker-compose.yml` se mantiene como dev intacto. Deploy: `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build`.
- **D-19:** Override prod incluye:
  - `db.ports`: vacío (cierra 5432 al host).
  - `web.ports`: vacío (Caddy llega por red interna).
  - `caddy.volumes`: monta `caddy/Caddyfile.prod` en lugar del dev.
  - `web.healthcheck`: `curl -fs http://localhost:8000/api/health` cada 30s, timeout 5s, retries 3, start_period 20s.
  - `caddy.healthcheck`: `wget --spider -q http://localhost/` cada 30s.
  - Resource limits (D-21).
- **D-20:** `restart: unless-stopped` en todos los servicios prod (web, db, caddy). Ya está en dev — se preserva.

### Resource limits
- **D-21:** Límites suaves por servicio:
  - `web`: `mem_limit: 4g`, `cpus: 2.0` (deja respirar al pipeline matplotlib pesado).
  - `db`: `mem_limit: 2g`, `cpus: 1.0`.
  - `caddy`: `mem_limit: 256m`, `cpus: 0.5`.
  - Total ~6.25G techo + ~9G de margen para SO + cache. Si `web` toca el techo, OOM-killer mata el proceso del pipeline (no el server entero) y deja entrada en el log del contenedor.

### Healthchecks
- **D-22:** `web` healthcheck = `curl -fs http://localhost:8000/api/health` (endpoint que ya existe — `make health` lo usa). Frecuencia 30s, timeout 5s, retries 3, `start_period 20s` para dar tiempo al schema_guard de Phase 3 a validar el esquema.
- **D-23:** `caddy` healthcheck = `wget --spider -q http://localhost/` (responde 308 redirect a HTTPS, suficiente para confirmar que Caddy escucha).
- **D-24:** `db` healthcheck = se conserva el actual (`pg_isready`).

### Script de deploy
- **D-25:** `scripts/deploy.sh` = básico + backup pre-deploy + smoke post-deploy. Pasos:
  1. `set -euo pipefail` + log inicio con timestamp.
  2. `git pull --ff-only` (si falla por divergencia, exit 1 — no force).
  3. `pg_dump pre-deploy` con tag `predeploy-<commit-hash>` en `/var/backups/nexo/`.
  4. `docker compose -f docker-compose.yml -f docker-compose.prod.yml build --pull`.
  5. `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d`.
  6. `sleep 15` y `curl -fs https://nexo.ecsmobility.local/api/health -k` (smoke).
  7. Si smoke falla, exit 1 y log "DEPLOY FAILED — rollback manual: `git reset --hard HEAD~1 && bash scripts/deploy.sh`".
- **D-26:** Script idempotente. Ejecutarlo dos veces seguidas no rompe nada (build cachea, up -d es no-op si todo está ya levantado).

### .env.prod.example
- **D-27:** SMTP vars presentes pero comentadas con `# TODO Mark-IV`. Razón: SMTP está Out of Scope Mark-III pero la app referencia las vars; comentadas evita que `pydantic-settings` peté al arrancar y deja constancia de que es intencional.
- **D-28:** `.env.prod.example` enumera TODAS las vars de `.env.example` (NEXO_*) con placeholders descriptivos `<CHANGEME>`. Vars con valores ya conocidos se rellenan literal: `NEXO_HOST=0.0.0.0`, `NEXO_PORT=8000`.

### Claude's Discretion
- Estructura interna del runbook DEPLOY_LAN.md (orden de secciones, nivel de detalle por paso, capturas de pantalla concretas) — Claude decide siguiendo el patrón de `CLAUDE.md` y los docs existentes.
- Convención exacta de naming para los `.sql.gz` rotados.
- Si `scripts/deploy.sh` añade emojis o no (responder NO — coherencia con `.claude/rules/common/coding-style.md`).
- Decidir si el pre-deploy backup vive en `/var/backups/nexo/predeploy/` separado del backup nightly o convive en el mismo dir con tag distinto.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project specs y planning
- `docs/MARK_III_PLAN.md` §"Sprint 5 — Despliegue LAN con HTTPS" (líneas 388–451) — entregables originales, riesgos identificados, dependencias.
- `docs/OPEN_QUESTIONS.md` §"Preguntas que quedan abiertas para sprints posteriores" (líneas 406–411) — preguntas DNS-01, backups, Ubuntu 24.04 detalles. Phase 6 las resuelve.
- `.planning/REQUIREMENTS.md` líneas 87–94 — DEPLOY-01..08 (texto literal de cada requisito).
- `.planning/ROADMAP.md` §"Phase 6: Despliegue LAN HTTPS" — success criteria oficiales.
- `CLAUDE.md` §"Convenciones de naming", "Política de commits", "Qué NO hacer" — políticas que el plan debe respetar (no `--no-verify`, no force push, no exposición internet).

### Estado actual del repo (no tocar lógica, solo extender / overrider)
- `caddy/Caddyfile` — config dev actual, base sobre la que se crea `caddy/Caddyfile.prod`.
- `docker-compose.yml` — compose dev actual, base sobre la que se crea `docker-compose.prod.yml` (override).
- `Makefile` — targets actuales (`make up`, `make health`, `make db-shell`); el deploy script no debe romperlos.
- `.env.example` — fuente de vars para construir `.env.prod.example`.
- `api/routers/` — endpoint `/api/health` que el healthcheck consume (verificar que responde 200 sin auth).

### Decisiones de fases anteriores que afectan a Phase 6
- `.planning/phases/02-identidad-auth-rbac-audit/02-CONTEXT.md` — auth en producción (cookies, secret keys, lockout). El `.env.prod.example` debe incluir `NEXO_SECRET_KEY` con instrucción de generación.
- `.planning/phases/03-capa-de-datos/03-CONTEXT.md` — `schema_guard` valida en lifespan; el script de deploy NO necesita ejecutar migraciones manualmente.
- `.planning/phases/04-consultas-pesadas/04-CONTEXT.md` — `cleanup_scheduler`, `factor_auto_refresh`, `listen_loop` arrancan en lifespan; deben sobrevivir a deploys (verificar que no pierden estado entre `up -d`).

### Decisiones cerradas en CLAUDE.md
- `CLAUDE.md` §"Decisiones cerradas de Mark-III" — SMTP out of scope, exposición internet descartada, repo NO se renombra. Phase 6 las honra.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Caddyfile** (`caddy/Caddyfile`): la sintaxis `tls internal` y `reverse_proxy web:8000` ya está. Reutilizar en el bloque prod añadiendo el hostname `nexo.ecsmobility.local`.
- **docker-compose.yml**: las vars `NEXO_PG_USER/PASSWORD/DB`, `NEXO_DATABASE_URL`, healthcheck de `pg_isready`, y la red interna entre `web`/`db`/`caddy` están funcionando. El override prod hereda esto y solo modifica lo necesario.
- **Makefile** targets `up`, `health`, `db-shell` — el deploy script puede invocarlos en lugar de duplicar comandos.
- **`.env.example`** — fuente exhaustiva de vars NEXO_* con compat OEE_*. `.env.prod.example` se construye como copia anonimizada.

### Established Patterns
- **Compose con vars + defaults**: el patrón `${NEXO_X:-${OEE_X:-default}}` se mantiene durante Mark-III (decisión cerrada en Sprint 0). Phase 6 lo respeta.
- **Bind mounts persistentes**: `./data`, `./informes`, `./tests` son bind mounts en dev. En prod NO se montan `./tests` (solo dev necesita el baseline de PDFs).
- **Healthcheck pattern**: `db` ya usa `test: ["CMD-SHELL", ...]` con interval/timeout/retries. Mismo formato para `web` y `caddy` en prod override.

### Integration Points
- `web` consume `/api/health` — endpoint validado en Plan 03-02. Healthcheck prod depende de él.
- `caddy` reverse-proxy a `web:8000` por red interna `default` de compose. No requiere cambios.
- `db` accesible solo por red interna en prod (puerto 5432 cerrado al host) — uso desde host: `docker compose exec db psql ...` (ya documentado en `make db-shell`).

</code_context>

<specifics>
## Specific Ideas

- El usuario dejó claro que **el hostname tiene que ser fijo y bookmarkable** ("para que los usuarios puedan fijarse la web como un marcador en su navegador"). `nexo.ecsmobility.local` no se cambia ni se rota durante Mark-III ni Mark-IV salvo decisión explícita del operador.
- El deploy se hace en una **máquina aún por preparar** (no existe físicamente todavía). Por eso `docs/DEPLOY_LAN.md` debe asumir Ubuntu Server 24.04 limpio sin Docker instalado.
- **No se pre-asume conocimiento DNS** del operador — el runbook explica qué es el archivo hosts y por qué se toca.
- Operador es e.eguskiza (autor del proyecto) + 1 persona de respaldo a designar. El runbook se escribe pensando en que el segundo lector NO es desarrollador del proyecto.

</specifics>

<deferred>
## Deferred Ideas

- **Migrar a DNS interno** (router/AD/dnsmasq) cuando IT lo permita — quita el paso de hosts-file por usuario. Documentar la migración en una sección "Mejora futura" de `DEPLOY_LAN.md`.
- **Sync de backups a NAS o equipo externo** — Phase 6 deja los backups en `/var/backups/nexo/` del propio servidor. Si el SSD muere, se pierde todo. Cuando haya NAS disponible, añadir cron de `rsync` al destino externo.
- **Cert firmado por CA corporativa ECS** — si en el futuro IT confirma que tienen una CA interna ya distribuida en los equipos del personal, migrar de `tls internal` a cert emitido por esa CA elimina la fricción de distribuir la root CA Caddy.
- **WAL archiving / replicación Postgres** — para llegar a RPO < 1h. Sobre-engineering para Mark-III pero opción si Nexo se vuelve operacionalmente crítico.
- **Smoke test extendido** — el smoke actual solo cura `/api/health`. Una versión más robusta probaría login + 1 query + 1 PDF generado. Mark-IV cuando haya tiempo.
- **Monitoring / alertas** (Prometheus + Grafana, o uptime-kuma) — Out of Scope Mark-III. Phase 7 (DevEx) puede incluirlo si sobra alcance.

</deferred>

---

*Phase: 06-despliegue-lan-https*
*Context gathered: 2026-04-21*
