# Phase 6: Despliegue LAN HTTPS — Research

**Researched:** 2026-04-21
**Domain:** Ops/DevOps — Ubuntu 24.04 + Docker Compose + Caddy (tls internal) + pg_dump + ufw
**Confidence:** HIGH (todo verificado contra docs oficiales o inspección del repo)

---

## RESEARCH COMPLETE — Executive Summary

- **Caddyfile prod** reutiliza la sintaxis actual (`tls internal`, `reverse_proxy web:8000`) añadiendo el hostname `nexo.ecsmobility.local`. Caddy inserta el redirect 80→443 automáticamente cuando hay un sitio HTTPS declarado y el flag `auto_https disable_redirects` **NO** está presente — por tanto el Caddyfile dev actual (que sí lo tiene) **no** se copia tal cual: prod omite ese flag.
- **Root CA de Caddy** vive en `/data/caddy/pki/authorities/local/root.crt` dentro del container (validez 10 años / 3600d). El intermediate tiene 7d pero Caddy lo renueva automáticamente; los clientes sólo instalan el **root**. Distribución manual una sola vez por equipo LAN.
- **Docker bypassa ufw** por diseño (publica puertos vía iptables `DOCKER-USER` chain, no pasa por `ufw allow/deny`). Consecuencia: `ufw deny 5432` no cierra Postgres si `ports:` sigue publicándolo; la única defensa es **no publicar 5432 en el override prod** (ya es D-19). ufw sigue necesario para SSH/Caddy/80+443 del host.
- **Port list override**: `ports: []` en el override **NO** resetea la lista heredada — los Compose v2 específicos la fusionan. Patrón correcto: usar la etiqueta YAML `!reset []` (soportada desde Compose v2.24+). Docs oficiales lo confirman.
- **pg_dump custom format (`-Fc`)** es la elección operativa estándar sobre Postgres 16 (compresión nativa, `pg_restore` con paralelismo y restore selectivo). El runbook DEPLOY_LAN.md incluye ambos lados del ciclo (dump + restore).

**Primary recommendation:** Construir Phase 6 como 4 task groups incrementales — (1) override compose prod + Caddyfile prod, (2) deploy.sh + backup cron + ufw, (3) DEPLOY_LAN.md end-to-end, (4) smoke validation desde LAN peer. Todo scriptable sin intervención interactiva excepto la edición de `.env.prod`.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Hostname y DNS:**
- **D-01**: Hostname = `nexo.ecsmobility.local`. Inmutable Mark-III.
- **D-02**: Resolución DNS = hosts-file por usuario. Pasos por SO en `docs/DEPLOY_LAN.md`.
- **D-03**: IP Ubuntu prod = placeholder `<IP_NEXO>` en docs y .env.prod.example.

**Certificado HTTPS:**
- **D-04**: `tls internal` (LE DNS-01 y CA ECS descartadas).
- **D-05**: Root CA distribuida desde `/data/caddy/pki/authorities/local/root.crt`. Pasos Windows/Linux/browser en docs.
- **D-06**: `Caddyfile.prod` nuevo con bloque `nexo.ecsmobility.local { tls internal; reverse_proxy web:8000 }`. Bloque `:443` IP-directa se conserva como fallback.

**Backup:**
- **D-07**: `pg_dump` lógico diario via cron host (03:00 UTC) hacia `/var/backups/nexo/nexo-YYYYMMDD.sql.gz`.
- **D-08**: Ubicación `/var/backups/nexo/`. Sync externo **deferred**.
- **D-09**: Retención 7d con `find ... -mtime +7 -delete`.
- **D-10**: RPO aceptado 24h.

**Recuperación:**
- **D-11**: RTO 1–2h con runbook.
- **D-12**: Bus factor 2. Runbook asume admin con Linux básico, no dev Nexo.
- **D-13**: Runbook steps: Docker install → clone → cp .env.prod.example → deploy.sh → restore pg_dump.

**Firewall (ufw):**
- **D-14**: SSH = `ufw allow from <SUBNET_LAN>/24 to any port 22`.
- **D-15**: HTTPS = `ufw allow 443/tcp`.
- **D-16**: HTTP = `ufw allow 80/tcp` (para redirect Caddy 80→443).
- **D-17**: `ufw default deny incoming`. Postgres 5432 no publicado al host. MCP no arranca.

**Compose prod:**
- **D-18**: `docker-compose.prod.yml` como override (NO profiles).
- **D-19**: Override incluye: db.ports vacío, web.ports vacío, caddy monta Caddyfile.prod, web healthcheck `curl -fs /api/health`, caddy healthcheck `wget --spider`, resource limits.
- **D-20**: `restart: unless-stopped` preservado.

**Resource limits:**
- **D-21**: web 4g/2cpu, db 2g/1cpu, caddy 256m/0.5cpu.

**Healthchecks:**
- **D-22**: web `curl -fs http://localhost:8000/api/health` 30s/5s/3/20s.
- **D-23**: caddy `wget --spider -q http://localhost/`.
- **D-24**: db conserva `pg_isready` actual.

**Deploy script:**
- **D-25**: `scripts/deploy.sh` = set -euo pipefail + git pull --ff-only + pre-deploy pg_dump tag `predeploy-<hash>` + build --pull + up -d + smoke `curl -fs https://nexo.ecsmobility.local/api/health -k` + rollback hint.
- **D-26**: Script idempotente.

**.env.prod.example:**
- **D-27**: SMTP vars presentes pero comentadas con `# TODO Mark-IV`.
- **D-28**: Enumera TODAS las vars NEXO_* con placeholders `<CHANGEME>`. Literales: `NEXO_HOST=0.0.0.0`, `NEXO_PORT=8000`.

### Claude's Discretion
- Estructura interna del runbook DEPLOY_LAN.md (orden de secciones, nivel de detalle).
- Convención exacta de naming para los `.sql.gz` rotados.
- Si `scripts/deploy.sh` añade emojis (NO — coherencia con coding-style.md).
- Si pre-deploy backup vive en `/var/backups/nexo/predeploy/` separado o mismo dir con tag distinto.

### Deferred Ideas (OUT OF SCOPE)
- DNS interno (router/AD/dnsmasq) — documentar como "Mejora futura".
- Sync backups a NAS — cron rsync cuando haya NAS.
- Cert firmado por CA corporativa ECS — si IT confirma CA interna.
- WAL archiving / replicación Postgres.
- Smoke test extendido (login + query + PDF).
- Monitoring/alertas (Prometheus/Grafana/uptime-kuma).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DEPLOY-01 | `docker-compose.prod.yml` con Caddyfile usando hostname — `tls internal` como estrategia final (LE DNS-01 descartado) | Topics 1, 2, 3 — Caddyfile.prod + override compose |
| DEPLOY-02 | Postgres 5432 no publicado al host; acceso vía `docker compose exec` | Topics 3, 5, 11 — override `db.ports` + landmine ufw/Docker |
| DEPLOY-03 | Healthchecks en web y caddy; `restart: unless-stopped` consistente | Topic 3 — healthcheck syntax + start_period |
| DEPLOY-04 | `scripts/deploy.sh` idempotente con build + up -d | Topic 7 — deploy.sh completo |
| DEPLOY-05 | `docs/DEPLOY_LAN.md` runbook con Docker install, hosts-file, CA, backup, recovery | Topics 1, 6, 9 — instrucciones por SO |
| DEPLOY-06 | ufw: 22 (subred LAN), 80, 443; deny resto | Topic 5 — ufw recipe completo + landmine DOCKER-USER |
| DEPLOY-07 | Verificación desde otro equipo LAN con cert aceptado | Topic 1 (dist CA) + Topic 12 (validation) |
| DEPLOY-08 | `.env.prod.example` sin valores reales | Topic 10 — estructura completa |
</phase_requirements>

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| TLS termination | Caddy container (reverse proxy) | — | Caddy ya hace eso en dev; sólo cambia el Caddyfile |
| DNS resolution LAN | Hosts-file client-side | — | D-02: LAN sin DNS interno; futuro migración a dnsmasq |
| Root CA trust | Client-side OS trust store | Caddy (emite) | Browser/sistema operativo valida el cert sólo si confía en la CA |
| Firewall perimeter | ufw host (SSH/HTTP/HTTPS) | Docker iptables (interno) | ufw filtra tráfico al host; Docker NO pasa por ufw (DOCKER-USER chain) |
| Service lifecycle | systemd (dockerd) + compose restart policies | — | `restart: unless-stopped` + `systemctl enable docker` sobreviven reboots |
| Backup persistence | Host filesystem `/var/backups/nexo/` | — | D-08: backup local; sync externo deferred |
| Backup scheduling | Host cron (root o ubuntu) | — | D-07: cron 03:00 UTC |
| App bootstrap (schema) | web container `lifespan` → schema_guard | — | Phase 3; deploy.sh NO ejecuta migraciones manuales |
| Background jobs | web container `lifespan` → cleanup_scheduler, factor_auto_refresh, listen_loop | — | Phase 4; deploy.sh debe garantizar que lifespan corre limpio tras `up -d` |
| Health observability | Caddy healthcheck + web healthcheck + `docker compose ps` | — | D-22/23/24 |

---

## Standard Stack

### Core (ya presente en el repo — sólo se extiende)

| Componente | Versión | Propósito | Por qué es estándar |
|------------|---------|-----------|---------------------|
| Docker CE | 26+ (repo oficial) | Runtime de contenedores | [VERIFIED: docs.docker.com/engine/install/ubuntu] Repo oficial Docker para Ubuntu 24.04 (noble) |
| docker compose plugin | v2.24+ | Orquestación multi-file | [VERIFIED: docs.docker.com] Plugin oficial; legacy `docker-compose` (v1) deprecado |
| Caddy | `caddy:2-alpine` | Reverse proxy + TLS | [VERIFIED: caddy/Caddyfile actual] Ya en el stack; `tls internal` nativo desde v2.0 |
| Postgres | `postgres:16-alpine` | BD app | [VERIFIED: docker-compose.yml:4] Ya en el stack; pg_dump custom format estándar |
| ufw | Ubuntu 24.04 default | Firewall host | [CITED: ubuntu.com/server/docs] Frontend iptables oficial Ubuntu |
| cron | Ubuntu 24.04 default | Scheduler de backups | [CITED: Ubuntu 24.04 ships cron] `/etc/cron.d/` o `crontab -e` |

### Supporting (nuevos archivos, sin deps externas nuevas)

| Archivo | Propósito | Origen |
|---------|-----------|--------|
| `caddy/Caddyfile.prod` | Config Caddy con hostname + redirect | Nuevo (basado en `caddy/Caddyfile` dev) |
| `docker-compose.prod.yml` | Override compose para producción | Nuevo |
| `scripts/deploy.sh` | Deploy idempotente | Nuevo |
| `scripts/backup_nightly.sh` | pg_dump + rotación (invocado por cron) | Nuevo (opcional — alternativa: cron ejecuta comandos inline) |
| `.env.prod.example` | Template .env producción | Nuevo |
| `docs/DEPLOY_LAN.md` | Runbook operador | Nuevo |

### Alternativas Consideradas (descartadas en CONTEXT)

| En lugar de | Se podría usar | Por qué descartado |
|-------------|----------------|--------------------|
| `tls internal` | Let's Encrypt DNS-01 | ECS no controla dominio público (D-04) |
| `tls internal` | CA corporativa ECS | Sin confirmación de que IT la mantenga (D-04) |
| `-Fc` custom format | pg_dumpall SQL plano | -Fc comprime nativo + restore selectivo; plano sólo si operador quiere inspeccionar con vi |
| cron host | Scheduler dentro de contenedor (ofelia, supercronic) | Un único servidor, operador Linux básico — cron del host es estándar universal |

**Installation (host Ubuntu 24.04, una única vez):**

```bash
# Docker CE + compose plugin (fuente: docs.docker.com/engine/install/ubuntu)
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu noble stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Ubuntu usuario sin sudo para docker
sudo usermod -aG docker "$USER"
# (relogin o `newgrp docker` para que el grupo aplique)

# Systemd sanity
sudo systemctl enable --now docker

# Verificar
docker compose version    # Docker Compose version v2.x.x
```

---

## Architecture Patterns

### System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│  LAN ECS (192.168.x.0/24)                                            │
│                                                                      │
│   [Equipo LAN]                                                       │
│   browser → hosts-file → <IP_NEXO> nexo.ecsmobility.local            │
│     │                                                                │
│     │ (1) TLS handshake verifica cert firmado por Caddy Local CA    │
│     │     (root.crt instalado en trust store del OS/browser)         │
│     ▼                                                                │
│   [Ubuntu Server <IP_NEXO>]  (ufw: 22 LAN, 80, 443)                  │
│     │                                                                │
│     │ :80  ─── Caddy auto-redirect 308 ──→ :443                      │
│     │ :443 ──────────────────────────────────────┐                   │
│     ▼                                            ▼                   │
│   ┌──────────────────────────────────────────────────┐              │
│   │ caddy container (nexo-caddy)                     │              │
│   │  - lee /etc/caddy/Caddyfile (mounted)            │              │
│   │  - emite cert desde local CA en caddy_data       │              │
│   │  - reverse_proxy web:8000 por red interna compose│              │
│   └───────────────┬──────────────────────────────────┘              │
│                   │ http://web:8000  (red compose "default")         │
│                   ▼                                                  │
│   ┌──────────────────────────────────────────────────┐              │
│   │ web container (FastAPI + uvicorn)                │              │
│   │  - AuthMiddleware filtra /api/health como public │              │
│   │  - lifespan: schema_guard + cleanup_scheduler    │              │
│   │    + factor_auto_refresh + listen_loop (Phase 4) │              │
│   └───────────────┬──────────────────────────────────┘              │
│                   │ psycopg2 → db:5432 (red interna)                 │
│                   ▼                                                  │
│   ┌──────────────────────────────────────────────────┐              │
│   │ db container (postgres:16-alpine)                │              │
│   │  - SIN ports al host en prod (D-17)              │              │
│   │  - pgdata volume                                 │              │
│   │  - host cron 03:00 UTC → docker compose exec     │              │
│   │    pg_dump → /var/backups/nexo/*.sql.gz          │              │
│   └──────────────────────────────────────────────────┘              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure (nuevos artefactos)

```
analisis_datos/
├── caddy/
│   ├── Caddyfile          # dev (actual, no se toca)
│   └── Caddyfile.prod     # NUEVO — con hostname
├── docker-compose.yml             # dev (actual, no se toca)
├── docker-compose.prod.yml        # NUEVO — override prod
├── scripts/
│   ├── deploy.sh          # NUEVO — deploy idempotente
│   └── backup_nightly.sh  # NUEVO (opcional) — invocado por cron
├── .env.prod.example      # NUEVO — template prod
└── docs/
    └── DEPLOY_LAN.md      # NUEVO — runbook 1-2h recovery
```

### Anti-Patterns a Evitar

- **No ejecutar `git push --force` o `git filter-repo` en el host de prod**: CLAUDE.md lo prohíbe explícitamente sin autorización.
- **No montar `./tests:/app/tests` en prod**: El bind dev es para PDF baseline (Plan 03-02). En prod no hace falta y agrega superficie de ataque.
- **No hardcodear `<IP_NEXO>` en Caddyfile.prod**: el hostname es estable (`nexo.ecsmobility.local`); la IP sólo aparece en `.env.prod` y en el hosts-file del cliente.
- **No usar `restart: always`**: `unless-stopped` es idiomático — sobrevive a reboots pero respeta `docker compose down` manual.
- **No ejecutar migraciones Alembic en `deploy.sh`**: schema_guard en lifespan valida al arrancar (Phase 3 D-06). Hacer doble migración es redundante y error-prone.

---

## Topic 1: Caddy `tls internal` root CA extraction and distribution

### Path exacto del root CA

[VERIFIED: docs + issue caddyserver/caddy#7361] Dentro del container, ruta inmutable desde Caddy 2.0:

```
/data/caddy/pki/authorities/local/root.crt
```

(El volumen `caddy_data` en el compose actual mapea a `/data` dentro del container — la ruta es estable).

### Extracción al host

```bash
# Desde el host Ubuntu, tras primer `docker compose up -d caddy`:
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec caddy \
    cat /data/caddy/pki/authorities/local/root.crt > nexo-ca.crt

# Copiar nexo-ca.crt a cada equipo LAN (scp, USB, chat interno, etc.)
```

### Instalación por SO

**Windows 10/11 (admin elevation required):**

```powershell
# PowerShell como Administrador — instala en Trusted Root del equipo
certutil.exe -addstore -f "ROOT" C:\path\to\nexo-ca.crt

# Alternativa GUI: Doble-click nexo.ca.crt → "Install Certificate" → "Local Machine"
#                   → "Place all certificates in the following store" → "Trusted Root Certification Authorities"
```

**Ubuntu / Debian (incluido el propio servidor si se va a hacer `curl` local a HTTPS):**

```bash
sudo cp nexo-ca.crt /usr/local/share/ca-certificates/nexo-ca.crt
sudo update-ca-certificates
# Salida esperada: "1 added, 0 removed; done."
```

**macOS:**

```bash
sudo security add-trusted-cert -d -r trustRoot \
    -k /Library/Keychains/System.keychain nexo-ca.crt
```

**Firefox (usa su propio trust store, no el del OS):**

```
Menú → Configuración → Privacidad y seguridad → Certificados → Ver certificados…
→ Autoridades → Importar… → seleccionar nexo-ca.crt
→ marcar "Confiar en esta CA para identificar sitios web" → OK
```

### Verificación end-to-end

```bash
# Desde un equipo LAN con hosts-file + CA ya instalados:
openssl s_client -connect nexo.ecsmobility.local:443 -servername nexo.ecsmobility.local </dev/null 2>&1 | \
    grep -E "(subject|issuer|Verify return code)"

# Output esperado:
#   subject=CN=nexo.ecsmobility.local
#   issuer=CN=Caddy Local Authority - ECC Intermediate
#   Verify return code: 0 (ok)
```

**Lifetime:** [VERIFIED: pkg.go.dev/github.com/caddyserver/caddy/v2/modules/caddytls + caddy.community thread 13484] Root CA vive 10 años (3600d); intermediate 7d pero auto-renovado por Caddy cada 10 min. Los clientes instalan sólo el **root** — NO necesitan re-importar cuando el intermediate rota. Planificar re-distribución en ~2035.

---

## Topic 2: Caddyfile.prod con hostname + tls internal + redirect 80→443

### Comportamiento del redirect

[VERIFIED: caddyserver.com/docs/automatic-https] Cuando el Caddyfile declara un sitio HTTPS (p.ej. `nexo.ecsmobility.local { ... }`), Caddy **automáticamente**:

1. Abre un listener en :80.
2. Emite HTTP 308 redirect de `http://nexo.ecsmobility.local/*` → `https://nexo.ecsmobility.local/*`.
3. No se necesita directiva `redir` explícita.

**La sutileza:** El Caddyfile dev actual tiene `auto_https disable_redirects` en el bloque global. Copiarlo a prod **rompería** el redirect 80→443 (D-16 lo exige). Por tanto, `Caddyfile.prod` **omite** ese flag.

### `caddy/Caddyfile.prod` (snippet final)

```caddyfile
# Caddyfile de producción — Nexo LAN HTTPS
# Cambios vs Caddyfile dev:
#   1. Sin `auto_https disable_redirects` (queremos redirect 80->443).
#   2. Bloque con hostname explícito para que Caddy emita cert para
#      "nexo.ecsmobility.local" (el bloque `:443` del dev emitía para la IP).
#   3. Bloque `:443` conservado como fallback IP-directa durante transición.

nexo.ecsmobility.local {
    tls internal
    reverse_proxy web:8000

    # Logs estructurados al stdout del container; journald los captura.
    log {
        output stdout
        format json
    }
}

# Fallback IP-directa (visitantes que escriben https://<IP_NEXO>).
# Cert autofirmado genérico; browser muestra warning pero el acceso funciona.
# Retirar cuando DNS interno esté operativo (deferred).
:443 {
    tls internal
    reverse_proxy web:8000
}
```

### Verificación

```bash
# Validar sintaxis antes de arrancar:
docker compose -f docker-compose.yml -f docker-compose.prod.yml run --rm caddy \
    caddy validate --config /etc/caddy/Caddyfile

# Tras arrancar, comprobar cert emitido:
curl -vk https://nexo.ecsmobility.local 2>&1 | grep -E "subject|issuer"

# Comprobar redirect 80->443:
curl -sI http://nexo.ecsmobility.local | head -3
# Esperado:
#   HTTP/1.1 308 Permanent Redirect
#   Location: https://nexo.ecsmobility.local/
```

---

## Topic 3: docker-compose override file pattern (prod)

### Sintaxis de `ports: []` — landmine crítico

[VERIFIED: docs.docker.com/reference/compose-file/services + docker/compose#3729] Compose v2 fusiona listas por defecto. `ports: []` NO borra los ports del base. Soluciones:

1. **`!reset []`** (Compose v2.24+) — la forma canónica 2025-2026.
2. **`!override`** — reemplaza el valor completo.

### `docker-compose.prod.yml` (snippet final)

```yaml
# Override de producción. Uso:
#   docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
#
# Notas de herencia (Compose v2):
#   - `ports: !reset []` borra los ports heredados de dev (5433, 8001).
#     Compose v2.24+ soporta el tag YAML `!reset` nativamente.
#   - `volumes` se fusiona — montamos Caddyfile.prod ADEMÁS de lo de dev.
#     Para "quitar" el bind `./tests`, el override no puede reemplazar la lista
#     con `!reset` sin perder `./data` y `./informes`; por tanto también usamos
#     `!reset` y enumeramos los volumes que sí queremos en prod.
#   - `environment` se fusiona clave-a-clave (override gana si colisiona).
#   - `healthcheck` reemplaza completamente el de dev (tiene un único valor, no lista).

services:
  db:
    ports: !reset []      # cierra 5432 al host (DEPLOY-02)
    # healthcheck pg_isready se hereda del dev (D-24).

  web:
    ports: !reset []      # Caddy llega por red interna; no exponemos 8000
    volumes: !reset
      - ./data:/app/data
      - ./informes:/app/informes
      # NOTA: ./tests NO se monta en prod (es solo para PDF baseline dev).
    healthcheck:
      test: ["CMD", "curl", "-fs", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 20s   # schema_guard + lifespan jobs necesitan arrancar
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4g

  caddy:
    volumes: !reset
      - ./caddy/Caddyfile.prod:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
      - caddy_config:/config
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost/"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 256m
    # db.deploy.resources
    # (no es un servicio aparte; su override de limits se anota aquí para coherencia):

  # db resource limits (no se pueden declarar junto al ports: !reset porque
  # YAML no permite dos keys del mismo nombre; bloque completo):
  # NOTA de implementación: el planner debe consolidar TODO lo de `db:` en una
  # única sección del override — Compose fusionará con el service `db:` del base.
```

**Versión consolidada (sin comentarios de desarrollo, para copiar a PLAN.md):**

```yaml
services:
  db:
    ports: !reset []
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 2g

  web:
    ports: !reset []
    volumes: !reset
      - ./data:/app/data
      - ./informes:/app/informes
    healthcheck:
      test: ["CMD", "curl", "-fs", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 20s
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4g

  caddy:
    volumes: !reset
      - ./caddy/Caddyfile.prod:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
      - caddy_config:/config
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost/"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 256m
```

### Healthcheck curl en web container

El Dockerfile actual (imagen web) está basado en Python 3.11 (no alpine). `curl` suele estar disponible; si falta, alternativa portable:

```yaml
# Fallback si el container no tiene curl:
healthcheck:
  test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health').read()"]
```

El planner debe verificar `docker compose exec web which curl` antes de fijar la forma definitiva.

### Resource limits — sintaxis canónica

[VERIFIED: docs.docker.com/reference/compose-file/deploy] En Compose v2 puro (no Swarm) la forma canónica 2025 es `deploy.resources.limits`. El legacy `mem_limit` / `cpus` a nivel de servicio aún funciona pero está en modo compat y no es recomendado. **Usar `deploy.resources.limits` siempre.**

### start_period — valor seguro dado schema_guard

schema_guard corre síncrono en el lifespan (Phase 3). Ejecución observada: ~1-3s sobre schema ya migrado, hasta ~10s en primer arranque si `NEXO_AUTO_MIGRATE=true`. `start_period: 20s` deja margen de 2x sin ser excesivo. Si arranque primer deploy aparece `unhealthy` antes de `healthy`, subir a 30s.

---

## Topic 4: pg_dump nightly backup pattern

### Formato: `-Fc` (custom) es el estándar

[VERIFIED: postgresql.org/docs/18/app-pgdump] Custom format:
- Compresión nativa (no hace falta `| gzip`).
- `pg_restore` con `-j N` para paralelismo.
- Restore selectivo de tablas/schemas sin descomprimir.
- Archivo self-contained (incluye metadata de versión Postgres).

**Pero CONTEXT D-07 indica `| gzip` + extensión `.sql.gz`.** El plain SQL + gzip sigue siendo válido: se restaura con `zcat … | psql …`, operador puede `zcat … | less` para inspección manual. Dado D-12 (bus factor 2, admin sin conocimiento profundo Postgres), **plain SQL + gzip es la mejor opción operativa** — el runbook queda legible.

**Decisión recomendada al planner:** Respetar D-07 literal (plain SQL + gzip). Documentar `-Fc` como alternativa futura si el tamaño del dump se vuelve prohibitivo.

### Backup nightly — script canónico

`scripts/backup_nightly.sh`:

```bash
#!/usr/bin/env bash
# Backup nightly de Postgres Nexo. Invocado por cron.
# D-07, D-08, D-09: plain SQL + gzip, 7d retention, /var/backups/nexo/
set -euo pipefail
IFS=$'\n\t'

BACKUP_DIR="/var/backups/nexo"
PROJECT_DIR="/opt/nexo"       # <CHANGEME en el runbook al path real del clone>
STAMP="$(date -u +%Y%m%d-%H%M)"
OUT="${BACKUP_DIR}/nexo-${STAMP}.sql.gz"

mkdir -p "${BACKUP_DIR}"
cd "${PROJECT_DIR}"

# docker compose exec -T (sin tty) + uso de POSTGRES_USER/DB ya inyectados
# en el container. Password NO se pasa por CLI — se hereda vía env del
# contenedor, que ya la tiene desde el POSTGRES_PASSWORD del compose.
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec -T db \
    bash -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
    | gzip > "${OUT}.tmp"

# Atomic rename (evita archivo parcial si falla a mitad)
mv "${OUT}.tmp" "${OUT}"

# Rotación — borra archivos .sql.gz mayores de 7 días (D-09)
find "${BACKUP_DIR}" -maxdepth 1 -name "nexo-*.sql.gz" -type f -mtime +7 -delete

# Log cierre (cron captura stdout al mail local o /var/log/syslog)
echo "[$(date -u +%FT%TZ)] backup ok: ${OUT} ($(du -h "${OUT}" | cut -f1))"
```

### Cron entry (Ubuntu 24.04)

**Opción A (recomendada): `/etc/cron.d/nexo-backup`** — file-based, versionable:

```cron
# /etc/cron.d/nexo-backup
# Nightly pg_dump de Nexo. 03:00 UTC según D-07.
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

0 3 * * * root /opt/nexo/scripts/backup_nightly.sh >> /var/log/nexo-backup.log 2>&1
```

**Permisos de instalación:**

```bash
sudo install -m 0755 scripts/backup_nightly.sh /opt/nexo/scripts/backup_nightly.sh
sudo install -m 0644 config/nexo-backup.cron /etc/cron.d/nexo-backup
sudo touch /var/log/nexo-backup.log && sudo chmod 0640 /var/log/nexo-backup.log
```

**Verificar:**

```bash
# Ver si cron va a disparar el job
sudo systemctl status cron
sudo grep nexo-backup /var/log/syslog | tail -10

# Ejecución manual de prueba
sudo /opt/nexo/scripts/backup_nightly.sh
ls -lh /var/backups/nexo/
```

### Restore (documentado en DEPLOY_LAN.md, NO en deploy.sh)

```bash
# Restore manual, con los containers corriendo:
cd /opt/nexo
BACKUP=/var/backups/nexo/nexo-20260421-0300.sql.gz

# 1. Drop + recreate schema (nexo.*) para evitar conflictos:
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec -T db \
    bash -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "DROP SCHEMA IF EXISTS nexo CASCADE; CREATE SCHEMA nexo;"'

# 2. Restore
zcat "${BACKUP}" | docker compose -f docker-compose.yml -f docker-compose.prod.yml exec -T db \
    bash -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'

# 3. Verificar
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec -T db \
    bash -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\dt nexo.*"'
```

### Pre-deploy backup (invocado desde deploy.sh)

Convención recomendada para D-25 (discretion item): directorio SEPARADO para que no mezclarse con el rotation nightly ni cuenten hacia los 7d.

```bash
# Dentro de deploy.sh, antes del build:
PREDEPLOY_DIR="/var/backups/nexo/predeploy"
mkdir -p "${PREDEPLOY_DIR}"
HASH="$(git rev-parse --short HEAD)"
STAMP="$(date -u +%Y%m%d-%H%M)"
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec -T db \
    bash -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
    | gzip > "${PREDEPLOY_DIR}/predeploy-${HASH}-${STAMP}.sql.gz"
# Retención pre-deploy: últimos 30 días (deploys son raros):
find "${PREDEPLOY_DIR}" -name "predeploy-*.sql.gz" -mtime +30 -delete
```

---

## Topic 5: ufw rules para Ubuntu 24.04

### Secuencia segura (NO lockout)

```bash
# 1. Instalar (ya viene en Ubuntu Server 24.04; reinstalar por si acaso)
sudo apt update && sudo apt install -y ufw

# 2. PRIMERO la regla SSH — CRÍTICO: antes de `ufw enable`
#    Substituir <SUBNET_LAN> por la subred real ECS, p.ej. 192.168.1.0/24
sudo ufw allow from <SUBNET_LAN> to any port 22 proto tcp comment 'SSH from LAN'

# 3. HTTP y HTTPS (abiertos a toda la LAN; auth Phase 2 controla acceso)
sudo ufw allow 80/tcp  comment 'HTTP redirect to HTTPS'
sudo ufw allow 443/tcp comment 'HTTPS Nexo'

# 4. Defaults
sudo ufw default deny incoming
sudo ufw default allow outgoing

# 5. AHORA sí, enable (prompt "Proceed with operation (y|n)?" — contestar y)
sudo ufw enable

# 6. Verificar
sudo ufw status verbose
sudo ufw status numbered
```

**Esperado en `ufw status numbered`:**

```
[ 1] 22/tcp        ALLOW IN    192.168.1.0/24             # SSH from LAN
[ 2] 80/tcp        ALLOW IN    Anywhere                   # HTTP redirect to HTTPS
[ 3] 443/tcp       ALLOW IN    Anywhere                   # HTTPS Nexo
[ 4] 22/tcp (v6)   ALLOW IN    Anywhere (v6)              # (si la subred no es v6)
...
Default: deny (incoming), allow (outgoing), disabled (routed)
```

### Pitfall crítico: Docker bypassa ufw

[VERIFIED: docs.docker.com/engine/network/packet-filtering-firewalls + chaifeng/ufw-docker] Los `ports:` publicados por Docker van directamente a la chain `DOCKER-USER` de iptables, **sin pasar por ufw**. Consecuencia:

- `ufw deny 5432` **NO cierra** el puerto 5432 si `docker-compose.yml` tiene `ports: - "5432:5432"`.
- `ufw allow 443/tcp` es innecesario para que Docker acepte conexiones a 443 del caddy container — Docker ya abrió ese puerto directamente.

**Por qué sigue siendo correcto añadir las reglas ufw:**

1. Los puertos 80/443 del caddy container **siguen siendo funcionales** con ufw activo (Docker se impone sobre ufw en ese lado). No hay problema de bloqueo.
2. SSH y cualquier otro servicio del host (no Docker) SÍ pasan por ufw. Las reglas 22/80/443 protegen si mañana alguien levanta nginx nativo o cosas fuera de Docker.
3. `ufw default deny incoming` sigue siendo defensa en profundidad útil.

**Por qué Postgres 5432 queda cerrado en prod:**

- **No** por `ufw deny 5432` (que sería inoperante dado Docker).
- **Sí** por `db.ports: !reset []` en el override prod: Docker directamente no publica el puerto al host. Sin publicación, no hay regla iptables de DNAT hacia el container, y la red externa no puede llegar.

Verificación:

```bash
# 1. Confirmar que el puerto no está escuchando en el host
sudo ss -tlnp | grep :5432
# Esperado: sin output

# 2. Confirmar que el psql via exec sí funciona (red interna compose)
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec db \
    bash -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT version();"'

# 3. Desde un peer LAN:
nmap -Pn -p 22,80,443,5432 <IP_NEXO>
# Esperado:
#   22/tcp    open
#   80/tcp    open
#   443/tcp   open
#   5432/tcp  closed (o filtered si nmap recibe reset)
```

### Caveat: `ufw status` no muestra las reglas de Docker

El operador del runbook debe entender que el scope de `ufw status` excluye Docker. El DEPLOY_LAN.md debe incluir una nota explicativa para que nadie se confunda viendo "ufw status" sin puerto 443 y pensando que Caddy está mal.

---

## Topic 6: Client hosts-file instructions por SO

### Windows 10/11

```
Ruta: C:\Windows\System32\drivers\etc\hosts

Requiere admin elevation. Forma canónica:
  1. Botón Inicio → escribir "Bloc de notas"
  2. Click derecho → "Ejecutar como administrador"
  3. Archivo → Abrir → C:\Windows\System32\drivers\etc\hosts
     (si no ve el archivo, cambiar filtro a "Todos los archivos")
  4. Añadir al final:
       <IP_NEXO>   nexo.ecsmobility.local
  5. Guardar (Ctrl+S). Cerrar.
```

**Alternativa PowerShell como admin (para docs):**

```powershell
# PowerShell como Administrador
Add-Content -Path "$env:SystemRoot\System32\drivers\etc\hosts" -Value "`n<IP_NEXO>`tnexo.ecsmobility.local"
```

### Linux

```bash
echo "<IP_NEXO>   nexo.ecsmobility.local" | sudo tee -a /etc/hosts
```

### macOS

```bash
echo "<IP_NEXO>   nexo.ecsmobility.local" | sudo tee -a /etc/hosts
sudo dscacheutil -flushcache
sudo killall -HUP mDNSResponder
```

### Verificación cross-OS

```bash
# Todos los SO:
ping -c 1 nexo.ecsmobility.local
# Esperado: "PING nexo.ecsmobility.local (<IP_NEXO>) ..."

nslookup nexo.ecsmobility.local
# Nota: nslookup consulta DNS, no hosts-file — falla con NXDOMAIN pero ping funciona.
# El runbook debe aclarar que ping es la prueba correcta.

# Con la CA ya instalada y hosts-file:
curl -v https://nexo.ecsmobility.local/api/health
# Esperado: 200 OK con JSON {"ok": true, "services": {...}}
```

---

## Topic 7: `scripts/deploy.sh` — script completo

```bash
#!/usr/bin/env bash
# deploy.sh — Deploy idempotente Nexo (D-25, D-26)
# Uso: bash scripts/deploy.sh
#
# Salida:
#   0  = deploy OK (smoke pasa)
#   1  = deploy falló en cualquier step (git, build, up, smoke)
#
# No hay rollback automático — el operador decide tras leer el log.

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
git pull --ff-only || fail "git pull falló (posible divergencia). Resolver manualmente."

# 2. Pre-deploy backup (sólo si db ya estaba corriendo — primer deploy skip)
log "step 2/5: pre-deploy backup"
if $COMPOSE ps db --format json 2>/dev/null | grep -q '"State":"running"'; then
    mkdir -p "${PREDEPLOY_DIR}"
    HASH="$(git rev-parse --short HEAD)"
    STAMP="$(date -u +%Y%m%d-%H%M)"
    OUT="${PREDEPLOY_DIR}/predeploy-${HASH}-${STAMP}.sql.gz"
    $COMPOSE exec -T db bash -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
        | gzip > "${OUT}" || fail "pg_dump pre-deploy falló"
    log "  backup: ${OUT} ($(du -h "${OUT}" | cut -f1))"
    find "${PREDEPLOY_DIR}" -name "predeploy-*.sql.gz" -mtime +30 -delete 2>/dev/null || true
else
    log "  db no está corriendo — primer deploy, skip backup"
fi

# 3. Build imágenes (con --pull para refrescar bases apenas cambien)
log "step 3/5: build"
$COMPOSE build --pull || fail "build falló"

# 4. Up en background
log "step 4/5: up -d"
$COMPOSE up -d || fail "up -d falló"

# 5. Smoke test — /api/health via Caddy HTTPS
log "step 5/5: smoke test"
# -k (insecure): TLS internal; el host de deploy no necesariamente tiene la CA
#   instalada (deploy.sh corre como root del servidor). Alternativa más limpia:
#   apuntar a http://localhost:8000/api/health (bypass Caddy), pero eso no valida
#   el path completo HTTPS → Caddy → web.
# sleep 15: dar tiempo a start_period de web (20s en healthcheck), margen
#   conservador para que el smoke pille el container ya healthy.
sleep 15
if curl -fs -m 10 -k https://nexo.ecsmobility.local/api/health >/dev/null; then
    log "smoke OK"
else
    fail "smoke falló — rollback manual: git reset --hard HEAD~1 && bash scripts/deploy.sh"
fi

log "DONE"
```

**Permisos:**

```bash
chmod +x scripts/deploy.sh
```

**Idempotencia:**

Correr `bash scripts/deploy.sh` dos veces seguidas da:

- Pass 1: git pull (fast-forward), build (cache miss o hit según cambios), up (arranca o no-op).
- Pass 2: git pull (already up to date), build (todo cached), up (no-op si nada cambió; el `up -d` con imágenes idénticas no reinicia containers).

---

## Topic 8: Verificación de `/api/health` sin auth

### Estado verificado en el repo

[VERIFIED: api/routers/health.py + api/middleware/auth.py]

- **Endpoint:** `GET /api/health` (prefix `/api` añadido en `api/main.py:315`, prefix `/health` declarado en el router).
- **Público:** `api/middleware/auth.py:41` lo tiene en `_PUBLIC_PATHS`. AuthMiddleware lo deja pasar sin sesión.
- **Payload:** `{"ok": bool, "services": {"web": ..., "db": ..., "mes": ...}}`.
- **Coste:** El endpoint consulta `check_db_health` (Postgres — rápido) Y `mes_service.check_connection` (SQL Server remoto vía pyodbc — puede tardar 1-5s si la red MES está lenta).

**Landmine para el planner:**

`/api/health` llama a SQL Server MES. Un healthcheck Docker que falle porque MES está caído puede marcar `web` como `unhealthy` incorrectamente — sería un falso positivo (Nexo como app sigue funcionando: login, Postgres, auditoría, pipeline preflight). Opciones:

1. **Aceptarlo como síntoma legítimo:** Si MES está caído, Nexo queda parcialmente degradado y que aparezca `unhealthy` es señal útil.
2. **Crear endpoint `/api/health/liveness` que sólo compruebe web:** minimiza falsos positivos, pero añade superficie.
3. **Interpretar `ok=false` sin reinicio:** Cambiar el healthcheck a `curl -fs` (falla sólo en HTTP non-2xx) asegura que 200 con `"ok": false` no tira el container, pero sí señala degraded en `docker ps`.

**Recomendación:** Opción 1 (`curl -fs http://localhost:8000/api/health`). `/api/health` devuelve 200 incluso con MES caído (la lógica del router nunca lanza HTTP 4xx/5xx desde el exception handler — simplemente marca `services.mes.ok = false`). Por tanto `curl -fs` pasa siempre que el proceso uvicorn responda — exactamente lo que se quiere monitorizar.

**Confirmación con lectura del router:**

```python
# api/routers/health.py:12-32
@router.get("")
def health_check():
    services = {}
    services["web"] = {"ok": True, "msg": "Operativo"}
    db_ok, db_msg = check_db_health()
    services["db"] = {"ok": db_ok, "msg": db_msg}
    try:
        ok, msg, server, database = mes_service.check_connection()
        services["mes"] = {...}
    except Exception as exc:
        services["mes"] = {"ok": False, "msg": str(exc)}

    all_ok = all(s["ok"] for s in services.values())
    return {"ok": all_ok, "services": services}
```

Retorna siempre 200 (FastAPI default). `curl -fs` está OK.

---

## Topic 9: Ubuntu Server 24.04 Docker install (clean host)

Ver Topic "Standard Stack → Installation". Secuencia consolidada + systemd sanity:

```bash
# 1. Prerequisites
sudo apt update
sudo apt install -y ca-certificates curl git

# 2. Docker apt repo
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu noble stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update

# 3. Install Docker Engine + compose plugin + buildx
sudo apt install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin

# 4. Group membership (usar nombre de usuario real del operador, p.ej. "nexo-ops")
sudo usermod -aG docker "${USER}"
# El usuario debe hacer `logout` + `login` (o `newgrp docker`) para aplicar

# 5. Enable + start
sudo systemctl enable --now docker

# 6. Verificar
docker --version
docker compose version
sudo systemctl status docker --no-pager
docker run --rm hello-world
```

---

## Topic 10: `.env.prod.example` — estructura completa

Basado en `api/config.py` (fuente autoritativa de vars) + docker-compose.yml + CONTEXT D-27/D-28:

```bash
# =============================================================================
#  .env.prod.example — Template .env para producción Nexo (LAN)
#  --------------------------------------------------------------------------
#  INSTRUCCIONES:
#    1. Copiar este archivo a `.env` en el servidor de prod:
#         cp .env.prod.example .env
#    2. Reemplazar todos los <CHANGEME-*> con valores reales.
#    3. Para NEXO_SECRET_KEY y NEXO_PG_APP_PASSWORD, generar con:
#         python3 -c "import secrets; print(secrets.token_urlsafe(48))"
#    4. NUNCA commitear `.env` — ya está en `.gitignore`.
#  --------------------------------------------------------------------------
#  NOTA Mark-III: La compat layer OEE_* sigue activa. Definir SÓLO las NEXO_*;
#  si alguna app legacy lee OEE_*, añadirla explícitamente.
# =============================================================================

# ── Web server ────────────────────────────────────────────────────────────────
NEXO_HOST=0.0.0.0
NEXO_PORT=8000
NEXO_DEBUG=false
NEXO_LOG_SQL=false

# ── Auth (CRÍTICO — generar con secrets.token_urlsafe(48)) ────────────────────
# python3 -c "import secrets; print(secrets.token_urlsafe(48))"
NEXO_SECRET_KEY=<CHANGEME-GENERAR-CON-SECRETS-TOKEN-URLSAFE-48>
NEXO_SESSION_COOKIE_NAME=nexo_session
NEXO_SESSION_TTL_HOURS=12

# ── Postgres (interno a compose — host "db") ──────────────────────────────────
NEXO_PG_HOST=db
NEXO_PG_PORT=5432
NEXO_PG_USER=<CHANGEME-PG-OWNER-USER>
NEXO_PG_PASSWORD=<CHANGEME-PG-OWNER-PASSWORD>
NEXO_PG_DB=<CHANGEME-PG-DB-NAME>
# NEXO_PG_HOST_PORT no aplica en prod (ports cerrados al host).

# ── Postgres — rol app dedicado (Plan 02-04) ─────────────────────────────────
# Si no se define, el engine cae al owner. Recomendado en prod: rol separado.
NEXO_PG_APP_USER=nexo_app
NEXO_PG_APP_PASSWORD=<CHANGEME-PG-APP-PASSWORD>

# ── SQL Server APP (ecs_mobility) ─────────────────────────────────────────────
NEXO_APP_SERVER=<CHANGEME-IP-O-HOSTNAME-SQL>
NEXO_APP_PORT=1433
NEXO_APP_DB=ecs_mobility
NEXO_APP_USER=<CHANGEME-SQL-APP-USER>
NEXO_APP_PASSWORD=<CHANGEME-SQL-APP-PASSWORD>

# ── SQL Server MES (dbizaro, read-only) ───────────────────────────────────────
NEXO_MES_SERVER=<CHANGEME-IP-O-HOSTNAME-SQL>
NEXO_MES_PORT=1433
NEXO_MES_DB=dbizaro
NEXO_MES_USER=<CHANGEME-SQL-MES-USER>
NEXO_MES_PASSWORD=<CHANGEME-SQL-MES-PASSWORD>

# ── Branding ──────────────────────────────────────────────────────────────────
NEXO_APP_NAME=Nexo
NEXO_COMPANY_NAME=ECS Mobility
NEXO_LOGO_PATH=/static/img/brand/nexo/logo.png
NEXO_ECS_LOGO_PATH=/static/img/brand/ecs/logo.png

# ── Directorios de datos (volúmenes dentro del container) ─────────────────────
NEXO_DATA_DIR=/app/data
NEXO_INFORMES_DIR=/app/informes

# ── Observability Phase 4 ─────────────────────────────────────────────────────
# Retención query_log en días (0 = never cleanup). Default 90.
NEXO_QUERY_LOG_RETENTION_DAYS=90
# Auto-refresh threshold factor — trigger si factor_updated_at > N días. Default 60.
NEXO_AUTO_REFRESH_STALE_DAYS=60

# ── SMTP ──────────────────────────────────────────────────────────────────────
# TODO Mark-IV: habilitar cuando IT defina servidor SMTP corporativo.
# Mantener comentado para evitar que pydantic-settings o el código de
# notificaciones intente conectar a un servidor que no existe.
# NEXO_SMTP_HOST=<CHANGEME>
# NEXO_SMTP_PORT=587
# NEXO_SMTP_USER=<CHANGEME>
# NEXO_SMTP_PASSWORD=<CHANGEME>
# NEXO_SMTP_FROM=nexo@ecsmobility.com
```

**Verificación diff (DEPLOY-08 acceptance):**

```bash
diff -u .env.example .env.prod.example
# Esperado: diff limitado a
#   1. Valores reemplazados por <CHANGEME-*> o literales de prod (NEXO_HOST=0.0.0.0).
#   2. Comentarios # TODO Mark-IV en bloque SMTP.
#   3. NEXO_DEBUG=false en lugar del true dev.
#   4. Ausencia de NEXO_PG_HOST_PORT (no aplica).
```

---

## Topic 11: Landmines / gotchas específicos de este setup

Ver sección consolidada **Landmines** más abajo.

---

## Topic 12: Validation Architecture (Nyquist)

### Test Framework

| Property | Value |
|----------|-------|
| Framework | No hay framework de tests de infra tradicional (no pytest sobre infra). Validación = comandos shell explícitos en el runbook + smoke de `deploy.sh`. |
| Config file | — (validación manual + `deploy.sh`) |
| Quick run command | `bash scripts/deploy.sh` (smoke `curl -fs` a /api/health HTTPS) |
| Full suite command | Checklist de DEPLOY_LAN.md §"Validación post-deploy" — 8 comandos, uno por DEPLOY-* |
| Phase gate | Todos los 8 DEPLOY-* verificados empíricamente antes de `/gsd-verify-work` |

### Phase Requirements → Verification Commands Map

| Req ID | Behavior | Comando automatizable | Expected |
|--------|----------|-----------------------|----------|
| DEPLOY-01 | HTTPS via Caddy con hostname | `curl -sI https://nexo.ecsmobility.local` (desde peer LAN con CA) | `HTTP/2 200` sin warning TLS |
| DEPLOY-01 | cert emitido por Caddy local CA | `openssl s_client -connect nexo.ecsmobility.local:443 -servername nexo.ecsmobility.local </dev/null 2>&1 \| grep -E "issuer\|Verify"` | `issuer=CN=Caddy Local Authority...` y `Verify return code: 0 (ok)` |
| DEPLOY-02 | Postgres 5432 no reachable desde host | `sudo ss -tlnp \| grep :5432` | sin output |
| DEPLOY-02 | Postgres 5432 no reachable desde peer | `nmap -Pn -p 5432 <IP_NEXO>` (desde peer) | `5432/tcp closed` o `filtered` |
| DEPLOY-02 | psql via exec funciona | `docker compose -f docker-compose.yml -f docker-compose.prod.yml exec db bash -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1;"'` | `?column? = 1` |
| DEPLOY-03 | web healthcheck healthy | `docker inspect $(docker compose -f docker-compose.yml -f docker-compose.prod.yml ps -q web) --format '{{.State.Health.Status}}'` | `healthy` |
| DEPLOY-03 | caddy healthcheck healthy | `docker inspect $(docker compose -f docker-compose.yml -f docker-compose.prod.yml ps -q caddy) --format '{{.State.Health.Status}}'` | `healthy` |
| DEPLOY-03 | restart: unless-stopped | `docker inspect $(docker compose ... ps -q web) --format '{{.HostConfig.RestartPolicy.Name}}'` | `unless-stopped` |
| DEPLOY-04 | deploy.sh idempotente | Ejecutar 2x consecutivas: `bash scripts/deploy.sh; bash scripts/deploy.sh` | Ambas `DONE` exit 0 |
| DEPLOY-05 | Runbook end-to-end | Lectura manual: operador de respaldo sigue DEPLOY_LAN.md en máquina limpia | Nexo accesible en `<1-2h` |
| DEPLOY-06 | ufw sólo 22/80/443 | `sudo ufw status numbered` | Sólo 22 LAN, 80, 443 + deny default |
| DEPLOY-06 | nmap desde peer confirma | `nmap -Pn -p 22,80,443 <IP_NEXO>` | los 3 `open` |
| DEPLOY-07 | Peer LAN carga verde | Manual: desde laptop con hosts-file + CA → `https://nexo.ecsmobility.local` en browser | Padlock verde + página login |
| DEPLOY-08 | .env.prod.example sin secretos | `grep -E 'password\|secret\|key' .env.prod.example \| grep -v CHANGEME \| grep -v '^#'` | sin output (sólo literales como `NEXO_HOST=0.0.0.0`) |

### Sampling Rate

- **Per task commit:** Comandos del PLAN que cambien un archivo verifican el archivo in situ (sintaxis YAML/Caddyfile): `docker compose config` para YAML, `caddy validate` para Caddyfile.
- **Per wave merge:** `bash scripts/deploy.sh` completo en el servidor de prod (smoke incluido).
- **Phase gate:** Los 8 DEPLOY-* ejecutados desde peer LAN + inspección docker inspect. Gate final = `/gsd-verify-work` con el checklist humano.

### Wave 0 Gaps

- [x] No se requiere framework nuevo; validación = comandos shell + smoke en deploy.sh.
- [ ] **Servidor Ubuntu físico:** la fase requiere un host Ubuntu 24.04 asignado. Si no está físicamente disponible en el momento de ejecución, Phase 6 queda bloqueada en el paso de validación real. El código (override, Caddyfile, deploy.sh, docs) puede escribirse y committearse sin el host, pero los 8 DEPLOY-* no se cerrarán hasta que el operador corra el runbook en la máquina real.
- [ ] **Peer LAN para validación DEPLOY-07:** operador necesita un segundo equipo (laptop admin) con acceso LAN para la verificación del cert.

---

## Common Pitfalls

### Pitfall 1: Copiar `auto_https disable_redirects` del Caddyfile dev a prod
**Qué sale mal:** El redirect 80→443 se desactiva, usuarios que escriban `http://nexo...` ven `connection refused` (porque sólo hay listener en :443 con ese flag).
**Por qué:** El flag global inhibe la creación del listener :80 redirect por parte de `auto_https`.
**Cómo evitar:** `Caddyfile.prod` **NO** lleva esa línea. La redirect-hint para D-16 depende de no ponerla.
**Signos tempranos:** `curl -sI http://nexo.ecsmobility.local` retorna error en lugar de `308 Permanent Redirect`.

### Pitfall 2: `ports: []` en el override no resetea la lista
**Qué sale mal:** Postgres sigue publicando 5432 al host en prod → DEPLOY-02 falla silenciosamente.
**Por qué:** Compose v2 fusiona listas por defecto; sólo `!reset` las borra.
**Cómo evitar:** Usar `ports: !reset []` (Compose v2.24+).
**Signos tempranos:** `sudo ss -tlnp | grep :5432` devuelve una línea tras el deploy prod.

### Pitfall 3: MES caído marca web como unhealthy y reinicia el container en loop
**Qué sale mal:** Si SQL Server MES está inaccesible, `/api/health` retorna `{"ok": false, ...}` PERO con HTTP 200. El `curl -fs` pasa → NO reinicia. Sin embargo, si alguien cambia el healthcheck a `curl -fs ... && jq -e .ok`, MES caído tiraría el web.
**Por qué:** El endpoint está diseñado para reportar estado, no lanzar HTTP 5xx.
**Cómo evitar:** Dejar el healthcheck exactamente como `curl -fs http://localhost:8000/api/health` — NO añadir jq ni parsing.
**Signos tempranos:** `docker compose logs web | grep -i health` muestra reinicios frecuentes.

### Pitfall 4: `docker compose exec -T db pg_dump` escribiendo al `.tmp` sin atomic rename
**Qué sale mal:** Si el dump se corta a mitad (OOM, disk full), el archivo parcial `.sql.gz` queda en `/var/backups/nexo/` y la rotación `-mtime +7` lo trata como válido. El operador confiadamente usa un archivo corrupto durante restore.
**Por qué:** Atomic rename via `mv .tmp final` garantiza que sólo existe el archivo completo.
**Cómo evitar:** Script `backup_nightly.sh` escribe a `${OUT}.tmp` y hace `mv ${OUT}.tmp ${OUT}` al final.
**Signos tempranos:** Tamaños de backup aberrantes en `ls -lh /var/backups/nexo/`.

### Pitfall 5: `git pull --ff-only` falla por divergencia de `main` local vs remota
**Qué sale mal:** En un servidor de prod, si alguien hizo un commit manual al `main` local (hotfix debugging), el `git pull --ff-only` rechaza el merge y deploy.sh sale con exit 1.
**Por qué:** Diseño intencional — el servidor de prod no debe aceptar merges divergentes automáticamente.
**Cómo evitar:** El runbook debe instruir "no hacer git commit en el servidor". Cualquier cambio viene via merge a `main` en GitHub.
**Signos tempranos:** `git status` en el servidor muestra commits locales unpushed.

### Pitfall 6: Lifespan jobs no arrancan tras `docker compose up -d` porque web se crea pero no se reinicia
**Qué sale mal:** Si la imagen no cambió y `up -d` es no-op, los lifespan jobs (cleanup_scheduler, factor_auto_refresh, listen_loop) siguen corriendo en la instancia vieja — bien. Pero si hubo un `build` que cambió el código sin recrear el container, el código viejo sigue corriendo.
**Por qué:** `up -d` recrea containers sólo si detecta cambio en la imagen o config.
**Cómo evitar:** Con `build --pull` + `up -d` explícito, Compose recrea containers cuando la imagen tag cambia. Para forzar recreación: `up -d --force-recreate`.
**Signos tempranos:** `docker compose logs web | grep "lifespan startup complete"` no muestra timestamp reciente tras el deploy.

### Pitfall 7: Docker iptables rules persisten tras `docker compose down -v`
**Qué sale mal:** Al derribar el stack y levantarlo de nuevo, Docker reconstruye sus chains DOCKER/DOCKER-USER. Si el operador tuvo reglas customizadas en DOCKER-USER (p.ej. por ufw-docker), se pueden perder.
**Por qué:** Docker destruye y recrea sus chains al reiniciar dockerd (no al recrear containers).
**Cómo evitar:** No instalar ufw-docker en Phase 6 (sobre-ingeniería dado el modelo de confianza LAN). Si se instala en el futuro, persistir reglas con iptables-persistent.
**Signos tempranos:** Puertos Docker-published aparecen/desaparecen tras `systemctl restart docker`.

### Pitfall 8: pgdata volume persiste a `down -v`
**Qué sale mal:** Un operador bien-intencionado hace `docker compose down -v` (incluido el `-v` para limpiar volúmenes) y **borra `pgdata` de prod**. Sin backup previo del día = pérdida de todo lo creado desde el backup nightly.
**Por qué:** `-v` en Compose v2 elimina volumes definidos en el archivo.
**Cómo evitar:** El runbook nombra explícitamente "NO usar `docker compose down -v` en prod jamás — usar sólo `down` (sin `-v`)". El Makefile de prod podría omitir cualquier target que use `-v`.
**Signos tempranos:** El operador tipea `make clean` (que sí usa `down -v`) en lugar de `down`.

---

## Code Examples

### Caddyfile.prod completo (listo para copiar)

```caddyfile
# caddy/Caddyfile.prod — producción LAN
# No incluye `auto_https disable_redirects` para que Caddy inserte redirect 80->443.

nexo.ecsmobility.local {
    tls internal
    reverse_proxy web:8000

    log {
        output stdout
        format json
    }
}

# Fallback IP-directa (https://<IP_NEXO>). Conservado por D-06.
:443 {
    tls internal
    reverse_proxy web:8000
}
```

### docker-compose.prod.yml completo (listo para copiar)

```yaml
# docker-compose.prod.yml — override de producción
# Uso: docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
#
# Compose v2.24+: `!reset` borra listas heredadas (ports, volumes).

services:
  db:
    ports: !reset []
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 2g

  web:
    ports: !reset []
    volumes: !reset
      - ./data:/app/data
      - ./informes:/app/informes
    healthcheck:
      test: ["CMD", "curl", "-fs", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 20s
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4g

  caddy:
    volumes: !reset
      - ./caddy/Caddyfile.prod:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
      - caddy_config:/config
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost/"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 256m
```

### deploy.sh completo

Ver Topic 7. 80 líneas, permisos `0755`, ubicación `scripts/deploy.sh`.

### backup_nightly.sh completo

Ver Topic 4. ~30 líneas, permisos `0755`, ubicación `scripts/backup_nightly.sh` (repo) + `/opt/nexo/scripts/backup_nightly.sh` (host).

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `docker-compose` v1 (Python, `docker-compose` CLI) | `docker compose` v2 plugin (Go) | v2 GA 2022, v1 EoL julio 2023 | Todos los comandos en el runbook usan `docker compose` (con espacio) |
| Legacy `mem_limit` / `cpus` al nivel del service | `deploy.resources.limits` | Compose v2.x la consolida | Usar `deploy.resources.limits` en nuevo código |
| Hack para clear ports: `ports: null` o duplicar | `ports: !reset []` (tag YAML) | Compose v2.24 (dic 2023) | Reset explícito — sintaxis canónica 2025-2026 |
| Caddy `tls internal` + root CA manual | Idem (no cambia); alternativa futura: `smallstep-pki` o `step-ca` si se necesita PKI más robusta | — | Root CA Caddy está battle-tested en deploys internos desde 2020 |
| ufw sólo (pre-Docker) | ufw + reconocimiento de que Docker bypassa (DOCKER-USER) | Permanente | Runbook debe aclarar el scope |
| pg_dump plain SQL | `pg_dump -Fc` (custom) | Estándar desde Postgres 11+ | D-07 mantiene plain + gzip por bus factor; -Fc es la alternativa |

**Deprecated / outdated:**

- `docker-compose` (Python v1): no instalar jamás — pip `docker-compose` deprecado.
- `links:` entre servicios compose: reemplazado por la red default desde v2.
- `caddy_file`/`caddyfile` directive (singular): nunca existió esa sintaxis — es `import` o bloques por nombre.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | El container `web` tiene `curl` instalado (imagen Python 3.11-slim usada en Dockerfile) | Topic 3 healthcheck | Si falla: cambiar healthcheck a `python -c` one-liner (snippet provisto en Topic 3) |
| A2 | Compose v2.24+ está en uso en el servidor (requiere para `!reset`) | Topic 3 override | El paquete `docker-compose-plugin` de Ubuntu 24.04 + repo oficial suele venir v2.29+. Verificar con `docker compose version`. Si es <2.24, el planner debe usar `!override` o duplicar explícitamente |
| A3 | La subred LAN ECS es IPv4 /24. IPv6 no se considera | Topic 5 ufw | Si hay IPv6 LAN, las reglas ufw deben replicarse con `::/64` — documentar como futuro |
| A4 | `/api/health` responde en <5s incluso con MES degradado | Topic 8 | Si MES timeout > 5s (health timeout), healthcheck lo marca unhealthy falsamente. Aumentar `timeout` o documentar como conocido |
| A5 | El Dockerfile actual de `web` no tiene USER no-root definido; pg_dump via exec corre como root del container | Topic 4 backup | Si el Dockerfile cambia a USER unprivileged, `docker compose exec -T db bash -c 'pg_dump ...'` sigue funcionando porque db corre como postgres default. Sin riesgo |

**Nada crítico en el tabla de assumptions** — todos son knobs operativos fácilmente ajustables por el planner.

---

## Open Questions

1. **¿En qué directorio del servidor vive el clone del repo? `/opt/nexo/` (recomendado) o `/home/<user>/nexo/`?**
   - Lo que sabemos: El CONTEXT no lo fija.
   - Gap: cambiar en deploy.sh (`PROJECT_DIR`), en backup_nightly.sh, y en el cron entry.
   - Recomendación: El planner propone `/opt/nexo/` en DEPLOY_LAN.md (convención Linux para software operado por root/ops).

2. **¿Quién es el "admin de respaldo" concreto para D-12 (bus factor 2)?**
   - Lo que sabemos: "a designar" según CONTEXT.
   - Gap: No afecta al código; sí al texto del runbook.
   - Recomendación: DEPLOY_LAN.md tiene un placeholder `<ADMIN_BACKUP_NAME>` al inicio; el operador lo rellena.

3. **¿Hace falta mount bind para `caddy_data` en vez de volumen nombrado, para poder extraer el root.crt desde el host sin `exec`?**
   - Lo que sabemos: el compose actual usa volumen nombrado `caddy_data`. Funciona con `docker compose exec`.
   - Gap: bind mount simplificaría la extracción inicial pero cambia el patrón dev-prod.
   - Recomendación: mantener volumen nombrado; `docker compose exec caddy cat ...` es el patrón canónico.

4. **¿deploy.sh debe hacer `docker compose down` antes de `up -d` (full recreate) o sólo `up -d` (rolling)?**
   - Lo que sabemos: D-26 pide idempotencia. `up -d` es idempotente pero no reinicia si imagen no cambió.
   - Gap: Si un cambio sólo afecta a env vars (no imagen), `up -d` no recarga. Necesitaría `up -d --force-recreate`.
   - Recomendación: dejar `up -d` simple — cuando se cambian env vars de forma crítica, el operador ejecuta manualmente `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --force-recreate web`.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Ubuntu Server 24.04 host | DEPLOY-01..07 (todo) | **NOT YET** (por asignar) | — | Bloqueante — sin host no hay validación real |
| Docker CE + compose plugin | todo | Se instala como parte del runbook (DEPLOY-05) | docker 26+, compose v2.24+ | — |
| ufw | DEPLOY-06 | Viene por defecto en Ubuntu Server 24.04 | — | — |
| cron | D-07 backup | Viene por defecto en Ubuntu Server 24.04 | vixie-cron | — |
| Acceso SSH al host con sudo | Todo el runbook | Depende de IT config del servidor | — | — |
| IP estática LAN | D-03 | **NOT YET** asignada | — | Placeholder `<IP_NEXO>` en docs |
| Segundo equipo LAN (peer) | DEPLOY-07 validación cert | Laptop admin | — | — |

**Missing dependencies blocking execution:**
- Servidor Ubuntu físico no está asignado todavía. Phase 6 puede completarse todo el deliverable de código/docs, pero los 8 DEPLOY-* no se cierran hasta ejecución real en el host.

**Missing dependencies with fallback:**
- Subred LAN exacta — placeholder `<SUBNET_LAN>` en Caddyfile/ufw rules/docs.

---

## Validation Architecture

Ver Topic 12 (integrado arriba). Formato canónico del skeleton GSD:

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Bash smoke + manual checklist (no pytest sobre infra) |
| Config file | `scripts/deploy.sh` (smoke inline) + `docs/DEPLOY_LAN.md` §Validación |
| Quick run command | `bash scripts/deploy.sh` |
| Full suite command | Checklist DEPLOY_LAN.md §"Validación post-deploy" (8 items manual/scriptable) |

### Phase Requirements → Test Map
Ver tabla completa en Topic 12 arriba. Resumen: los 8 DEPLOY-* tienen comando shell verificable, la mayoría automatizables, excepto DEPLOY-05 (lectura humana del runbook) y DEPLOY-07 parcial (manual en browser).

### Sampling Rate
- **Per task commit:** `docker compose config` (valida YAML) y `docker compose -f docker-compose.yml -f docker-compose.prod.yml run --rm caddy caddy validate --config /etc/caddy/Caddyfile`.
- **Per wave merge:** `bash scripts/deploy.sh` completo en servidor prod.
- **Phase gate:** `/gsd-verify-work` con los 8 DEPLOY-* ejecutados/revisados + DEPLOY-05 validado por lectura del operador de respaldo.

### Wave 0 Gaps
- Servidor Ubuntu físico disponible (bloqueante para DEPLOY-01..07).
- Segundo equipo LAN para DEPLOY-07 validación cross-machine.
- Subred LAN y IP asignadas por IT (placeholders hasta entonces).

---

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V1 Architecture | yes | LAN-only (no exposición internet) — doc explícito en DEPLOY_LAN.md §Modelo de amenazas |
| V6 Cryptography | yes | TLS 1.2+ via Caddy (defaults modernos); root CA distribuida controladamente |
| V7 Error Handling | yes | `global_exception_handler` ya activo (NAMING-07); no fuga traceback en prod |
| V9 Communications | yes | HTTPS obligatorio + redirect 80→443 |
| V10 Malicious code | yes | `git pull --ff-only` en deploy.sh; no `--force` |
| V14 Configuration | yes | `.env.prod` con `<CHANGEME>`; secret key generado con `secrets.token_urlsafe(48)`; `.env` en `.gitignore`; DEBUG=false |

### Known Threat Patterns for Ubuntu + Docker + Caddy LAN

| Pattern | STRIDE | Mitigation |
|---------|--------|------------|
| SSH brute-force desde LAN | Spoofing | `ufw allow from <SUBNET_LAN>/24 to any port 22` (D-14); la rate-limiter de Phase 2 no aplica a SSH — fail2ban es una mejora futura |
| Postgres port exposed to LAN | Information Disclosure | `db.ports: !reset []` cierra a nivel Docker (D-17, DEPLOY-02); reforzado por Docker bypass de ufw |
| MitM en TLS con cert autofirmado aceptado a ciegas | Tampering | Distribución controlada del root CA (D-05) + verificación por openssl + warning educativo en el runbook |
| Credenciales .env leaked en el repo | Information Disclosure | `.env` en `.gitignore` (ya verificado); `.env.prod.example` con `<CHANGEME>`; gitleaks en CI (NAMING-09) |
| Docker root container pivot | Elevation of Privilege | Contenedores corren como usuarios no-root donde aplique (fuera del scope Phase 6 — mitigación futura en Phase 7 devex) |
| Backup `.sql.gz` con permisos laxos | Information Disclosure | `/var/backups/nexo/` con `chmod 0750 root:root` (o propietario del servicio nexo) |

### Landmines de seguridad específicas Phase 6

Ver Landmines sección consolidada abajo. Resaltar:
- Root CA extraction: sólo operador ops debe tenerla; no publicarla por email inseguro ni chat externo.
- `.env` nunca commiteado — verificar antes de cada push al servidor con `git status`.

---

## Project Constraints (from CLAUDE.md)

Directivas del proyecto que el planner DEBE respetar:

- **Conventional Commits con scope**: `feat(deploy): add docker-compose.prod.yml`, `docs(deploy): runbook DEPLOY_LAN.md`, `chore(deploy): add backup cron entry`. Scope = `deploy` para esta fase.
- **Idioma**: título commit en inglés, body en español si aporta. Runbook `docs/DEPLOY_LAN.md` en español.
- **No `--no-verify`, no force push**: deploy.sh usa `git pull --ff-only`, nunca `--force`. Sin `git push --force`.
- **Env vars naming**: `NEXO_*` primario, `OEE_*` fallback mantenido (compat layer de `api/config.py`). `.env.prod.example` usa sólo `NEXO_*`.
- **No emojis en archivos de código ni runbook** (coherencia con `common/coding-style.md` y CLAUDE.md). Sólo texto ASCII/UTF-8 en DEPLOY_LAN.md.
- **Service naming**: `nexo-*` prefix para containers (el `container_name` no se usa actualmente en el compose; Compose genera nombres `<project>-<service>-<n>`. El project name se puede fijar con `COMPOSE_PROJECT_NAME=nexo` en `.env`). Decisión recomendada: documentar en `.env.prod.example` `COMPOSE_PROJECT_NAME=nexo` para que los containers aparezcan como `nexo-web-1`, `nexo-db-1`, `nexo-caddy-1`.
- **`make up` / `make dev` MUST NOT start `mcp`**: ya garantizado vía `profiles: ["mcp"]` en docker-compose.yml:77. Prod simplemente no usa `--profile mcp`, no necesita cambios.
- **Decisiones cerradas Mark-III**: honrar todas las listadas en CLAUDE.md §"Qué NO hacer" — no rotar credenciales SQL, no renombrar repo, no exponer internet, no SMTP operativo, no 2FA.

---

## Landmines (consolidated)

### Landmine 1 (CRÍTICO): Caddy auto_https flag carryover

Copiar literal el Caddyfile dev a prod rompe DEPLOY-06 (redirect 80→443). El flag `auto_https disable_redirects` del dev existe porque el dev no tiene hostname estable. En prod **no** lo incluimos.

**Mitigación:** `Caddyfile.prod` escrito from-scratch, NO es `cp caddy/Caddyfile caddy/Caddyfile.prod + editar`. Mejor: un patrón claro en el plan — "crear nuevo, referenciar diffs contra dev".

### Landmine 2 (CRÍTICO): Compose ports fusion sin !reset

`ports: []` en el override **NO** cierra 5432 al host. DEPLOY-02 falla silenciosamente.

**Mitigación:** Usar `ports: !reset []`. Verificar Compose v2.24+ en el servidor. Si el servidor viene con Compose antiguo, alternativa: eliminar `db.ports` del base (riesgo de romper dev) o añadir bloque explícito con `!override`.

### Landmine 3 (ALTO): Docker bypassa ufw

El operador que nunca ha peleado esto piensa que `ufw deny 5432/tcp` cierra el puerto. **No lo cierra**.

**Mitigación:** Documentar en DEPLOY_LAN.md §"Firewall — alcance real" con 1 párrafo claro. La defensa efectiva es `db.ports: !reset []` en el override.

### Landmine 4 (ALTO): `/api/health` llama a SQL Server MES

Un MES caído NO tira el container (porque `/api/health` devuelve HTTP 200 con `"ok": false`), pero sí puede hacer el healthcheck lento (timeout SQL Server 5-15s). Si `healthcheck.timeout: 5s` es menor que el MES timeout, marca `unhealthy`.

**Mitigación:** `timeout: 5s` es el valor en D-22 — suficiente si la BD MES responde o rechaza rápido. Si en producción se observan `unhealthy` frecuentes con MES OK, subir a `timeout: 10s`.

### Landmine 5 (MEDIO): Root CA expira en ~10 años

No urgente, pero planificar. Caddy emite un NUEVO root CA cuando el actual expira. Los clientes tendrán que re-importar.

**Mitigación:** Añadir sección "Re-distribución periódica de la CA" en `DEPLOY_LAN.md` con fecha de expiración explícita (cron a verificar tras cada arranque de Caddy: `docker compose exec caddy openssl x509 -in /data/caddy/pki/authorities/local/root.crt -noout -enddate`).

### Landmine 6 (MEDIO): `docker compose down -v` borra pgdata

Si el operador confunde `down` con `down -v` = pérdida total de datos.

**Mitigación:** `Makefile.prod` target `make down` que llama SÓLO `down` sin `-v`. El runbook advierte explícitamente "no usar `-v` jamás en prod".

### Landmine 7 (MEDIO): Backup atomicity

`pg_dump ... | gzip > file.sql.gz` se queda con archivo parcial si el proceso muere mid-dump. Rotación confunde archivos corruptos con válidos.

**Mitigación:** `backup_nightly.sh` escribe a `.tmp` + atomic `mv` al final. Topic 4 incluye la implementación.

### Landmine 8 (BAJO): Lifespan jobs no se reinician si imagen no cambia

`up -d` no recrea containers si la imagen tag es idéntica. Cambios sólo de env var no aplican hasta `--force-recreate`.

**Mitigación:** Runbook documenta el caso "si cambias .env en prod, hacer `docker compose -f ... -f ... up -d --force-recreate web`". No alteramos `deploy.sh` para evitar recreación innecesaria en el path feliz.

### Landmine 9 (BAJO): `git pull --ff-only` falla por divergencia local

Si alguien hizo `git commit` en el host de prod para un hotfix → `deploy.sh` exit 1.

**Mitigación:** Runbook dice "no commitear en el servidor — siempre via merge PR en GitHub". Si pasa, el runbook documenta `git stash` + `git pull` + resolución manual.

### Landmine 10 (BAJO): Compose project name no fijado

Sin `COMPOSE_PROJECT_NAME`, los containers se nombran por el directorio (`analisis_datos-web-1`). Rompe convención CLAUDE.md `nexo-*`.

**Mitigación:** `.env.prod.example` incluye `COMPOSE_PROJECT_NAME=nexo` → containers serán `nexo-web-1`, `nexo-db-1`, `nexo-caddy-1`.

---

## Sources

### Primary (HIGH confidence)
- [Caddy Automatic HTTPS docs](https://caddyserver.com/docs/automatic-https) — redirect 80→443 behavior, tls internal
- [Caddy tls directive docs](https://caddyserver.com/docs/caddyfile/directives/tls) — `tls internal`
- [Caddy internal CA intermediate lifespan thread](https://caddy.community/t/internal-tls-intermediate-cert-lifespan/13484) — 10y root / 7d intermediate
- [Caddy pkg docs caddytls](https://pkg.go.dev/github.com/caddyserver/caddy/v2/modules/caddytls) — default CA paths
- [Docker Engine install on Ubuntu](https://docs.docker.com/engine/install/ubuntu/) — apt repo + compose plugin
- [Docker Compose services reference](https://docs.docker.com/reference/compose-file/services/) — ports, healthcheck, deploy.resources
- [Docker packet filtering firewalls](https://docs.docker.com/engine/network/packet-filtering-firewalls/) — DOCKER-USER chain, ufw bypass
- [PostgreSQL pg_dump docs](https://www.postgresql.org/docs/current/app-pgdump.html) — -Fc vs plain, compression
- Repo inspection:
  - `/home/eeguskiza/analisis_datos/caddy/Caddyfile` — dev Caddy config
  - `/home/eeguskiza/analisis_datos/docker-compose.yml` — dev compose base
  - `/home/eeguskiza/analisis_datos/api/config.py` — env var canonical source
  - `/home/eeguskiza/analisis_datos/api/routers/health.py` — /api/health endpoint
  - `/home/eeguskiza/analisis_datos/api/middleware/auth.py:41` — /api/health in `_PUBLIC_PATHS`
  - `/home/eeguskiza/analisis_datos/Makefile` — existing targets

### Secondary (MEDIUM confidence — cross-verified)
- [chaifeng/ufw-docker GitHub](https://github.com/chaifeng/ufw-docker) — ufw-docker integration pattern (para referenciar como mejora futura, NO instalar en Phase 6)
- [Ubuntu Docker install guide (Thomas-Krenn wiki)](https://www.thomas-krenn.com/en/wiki/Docker_Installation_under_Ubuntu_24.04) — secondary confirmation de apt repo
- [docker/compose issue #3729](https://github.com/docker/compose/issues/3729) — history de `!reset` tag para port override

### Tertiary (LOW confidence — needs validation in real deploy)
- Valor exacto de `start_period: 20s` — puede requerir tuning tras observar deploys reales.
- Comportamiento de `docker compose exec -T db pg_dump` con Postgres 16-alpine en volumen ext4/xfs — consistent en issue reports pero no testado en este servidor concreto.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — todo el stack está en uso en dev (Caddy, Postgres 16, Docker), el cambio es aditivo (override, Caddyfile.prod, script deploy).
- Architecture: HIGH — topología ya probada en dev, el diagrama refleja el estado real.
- Pitfalls: HIGH — los 10 landmines identificados vienen de docs oficiales + lecturas de código.
- Validation: MEDIUM — los comandos de validación están claros, pero el servidor Ubuntu físico no existe aún; ejecución real pendiente.
- Security: HIGH — modelo LAN + distribución controlada de root CA coherentes con CONTEXT y CLAUDE.md "no exposición internet".

**Research date:** 2026-04-21
**Valid until:** 2026-05-21 (30 días — stack maduro, Caddy + Docker + pg_dump cambian lento)

---

*Researched: 2026-04-21*
*Phase: 06-despliegue-lan-https*
*Ready for planning via `/gsd-plan-phase 6`*
