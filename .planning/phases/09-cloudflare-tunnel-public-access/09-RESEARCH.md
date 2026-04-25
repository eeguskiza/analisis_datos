# Phase 9: Cloudflare Tunnel + Public Access — Research

**Researched:** 2026-04-25
**Domain:** Cloudflare Zero Trust (Tunnel + Access) + Caddy v2 hardening + Docker Compose infra patterns
**Confidence:** HIGH (Cloudflare + Caddy claims verified against official docs and current 2025 releases; one Free-plan limitation discovered that contradicts CONTEXT.md and requires user resolution before planning)

---

## Summary

Phase 9 expone Nexo a internet vía túnel saliente `cloudflared` con email allowlist gateado por Cloudflare Access. Servidor sin puertos abiertos. Fallback LAN intacto. La research valida 8 de las 11 preguntas técnicas del orchestrator contra docs oficiales (Cloudflare One, Caddy v2, Mozilla Observatory). Confirma que la sintaxis Docker, el flujo OTP, los headers de seguridad y el routing SNI funcionan como D-07..D-20 asumen — con dos correcciones empíricas y **un hallazgo bloqueante**.

**Hallazgo bloqueante (BLOCKER-01):** D-17 / CLOUD-07 dice "subir HTML estático con branding Nexo al dashboard de Cloudflare Access". La research empírica contra docs oficiales confirma que **Custom Block Page Templates en Cloudflare Access requieren plan Pay-as-you-go o Enterprise — NO disponible en Free Plan**. Lo que SÍ está disponible en Free para ramas de denial es: (a) la página default de Cloudflare ("You don't have access — sign in with another email") y (b) **Redirect URL** a una URL externa que el operador controle. Esto necesita decisión del usuario antes de que el planner escriba 09-02. Hay 3 caminos viables sin pagar: redirigir a una página estática servida por la propia Caddy bajo `/access-denied` (público sin Access), redirigir a un Cloudflare Pages estático separado, o aceptar la página default y pasar a paid si la fricción de UX lo justifica.

**Correcciones a CONTEXT.md (no bloqueantes):**

1. **CORRECTION-01 (D-09):** El nombre de variable de entorno **canónico** que `cloudflared` lee es `TUNNEL_TOKEN`, no `CLOUDFLARE_TUNNEL_TOKEN`. Se puede usar el nombre custom `CLOUDFLARE_TUNNEL_TOKEN` en `.env` y mapearlo a `TUNNEL_TOKEN` en el bloque `environment:` del compose. O aceptar el nombre canónico y nombrar la variable en `.env` como `TUNNEL_TOKEN` directamente. Plan recomendado: mantener `CLOUDFLARE_TUNNEL_TOKEN` en `.env` por claridad de nombre + mapear a `TUNNEL_TOKEN` en compose.

2. **CORRECTION-02 (D-20):** Caddy v2.11.0 (Dic 2024) **cambió el comportamiento default** del Host header en `reverse_proxy`. La afirmación de D-20 "Caddy lo respeta" ya no es trivialmente cierta — la nueva default reescribe Host al hostport del upstream cuando upstream es HTTPS. Pero el SNI mismatch que preocupa a R-05 está controlado por la opción global `strict_sni_host`, **deshabilitada por defecto** salvo cuando hay client cert auth. Conclusión empírica: el routing actual (SNI=`caddy:443` interno + Host=`nexo.app` desde cloudflared) cae en el bloque `:443` catch-all de `Caddyfile.prod` ya existente y NO debería devolver 421. Sigue siendo prudente verificar empíricamente en CF-04 del smoke; la mitigación documental queda registrada.

**Primary recommendation:** Pivotar D-17 a estrategia "Redirect URL en Cloudflare Access apuntando a `https://nexo.app/access-denied` servido públicamente por Caddy en un bloque sin Access" o discutirlo con el operador antes de planning. Todo lo demás (D-01..D-16, D-18..D-29) está validado y listo para `/gsd-plan-phase 9`.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

29 decisiones D-01..D-29 cerradas (ver `.planning/phases/09-cloudflare-tunnel-public-access/09-CONTEXT.md` para texto completo). Resumen agrupado:

**Dominio + cuenta (D-01..D-06):**
- D-01/D-02: Dominio = `nexo.app` con 4 fallbacks (`nexo-app.com`, `usenexo.com`, `getnexo.com`, `nexoecs.com`)
- D-03: Registrar = Cloudflare Registrar (sin markup, evita propagación DNS)
- D-04: URL final = `https://nexo.app` (raíz directa, sin subdominio)
- D-05: Owner cuenta CF = Erik Eguskiza ownership personal
- D-06: Bus factor 1 mitigado con gestor passwords + recovery codes + segundo admin a designar

**Cloudflare Tunnel (D-07..D-12):**
- D-07: Servicio Docker en `docker-compose.prod.yml`, imagen `cloudflare/cloudflared:latest`
- D-08: Modo token-based remote-managed (config en dashboard CF, no en YAML)
- D-09: Token en `/opt/nexo/.env` perms 600, var `CLOUDFLARE_TUNNEL_TOKEN` (ver CORRECTION-01)
- D-10: Target = `https://caddy:443` (NO directo a `web:8000`; mantiene headers + simetría LAN/CF)
- D-11: TLS verify off (`noTLSVerify: true`) entre cloudflared y caddy interno
- D-12: Routing alternativo `cloudflared → web:8000` descartado

**Headers seguridad (D-13):**
- 5 headers en `Caddyfile.prod`: HSTS (preload), X-Frame-Options DENY, X-Content-Type-Options nosniff, Referrer-Policy, Permissions-Policy. Aplicados a ambas ramas LAN+CF. CSP fuera de scope.

**Cloudflare Access (D-14..D-18):**
- D-14: 1 Access App `nexo` cubriendo `nexo.app/*`
- D-15: Email allowlist explícita (inicial: `e.eguskiza@ecsmobility.com`)
- D-16: IdP = One-Time Passcode por email; cookie 24h
- **D-17: Custom denial page HTML al dashboard CF (ver BLOCKER-01)**
- D-18: Endpoint `/cdn-cgi/access/logout` documentado, no cableado en Phase 9

**LAN fallback (D-19..D-20):**
- D-19: Caddy sigue escuchando `:443` con `tls internal` para `nexo.ecsmobility.local`
- D-20: Túnel CF entra al mismo Caddy; SNI será `caddy:443` interno; mitigación documentada si 421 (ver CORRECTION-02)

**Smoke testing (D-21..D-23):**
- D-21: `tests/infra/deploy_smoke.sh` 8→11 checks (login + MES caído + cloudflared running)
- D-22: `tests/infra/cloudflare_smoke.sh` nuevo, 6 checks `[CF-01..CF-06]`
- D-23: Smoke externo manual + scripted

**Rollback (D-24..D-26):**
- D-24: Rollback = `docker compose stop cloudflared`. LAN intacta.
- D-25: Rollback dominio = no hay (transacción de compra)
- D-26: Rollback automático en `deploy.sh` diferido a phase futura

**ADR + CLAUDE.md (D-27..D-28):**
- D-27: ADR-001 con formato Nygard (Context/Decision/Alternatives/Consequences) — ya existe en `docs/decisions/`
- D-28: CLAUDE.md actualizado en sección "Despliegue" + "Qué NO hacer"

**Mobile detection (D-29):**
- Out of scope Phase 9. Sub-plan futuro Phase 8.

### Claude's Discretion

- Granularidad real de plans (sketch propone 3: 09-01 hardening + smoke, 09-02 cloudflared + docs, 09-03 activación manual; el planner decide si subdivide)
- Patrón de scripting de `cloudflare_smoke.sh` y `cloudflare-bootstrap.sh` (mantener consistencia con `deploy_smoke.sh` Phase 6 — sí pre-aprobado por research)
- Estructura interna de `docs/CLOUDFLARE_DEPLOY.md` (mirror de `docs/DEPLOY_LAN.md` 16 secciones — sí pre-aprobado por research)
- Estilo de la denial page HTML/JSX (Nexo branding via tokens.css; ya existe `static/img/brand/nexo/`)
- Nombrado exacto de Makefile targets (sketch dice `cf-up/cf-down/cf-status/cf-logs/cf-smoke-external` — coherente con patrón `prod-*` ya existente)
- Resolución de BLOCKER-01 (3 alternativas viables sin pagar; planner debe escoger una y documentarla, o escalar a operador en `/gsd-discuss-phase` rerun)
- Resolución de CORRECTION-01 (mapear `CLOUDFLARE_TUNNEL_TOKEN` a `TUNNEL_TOKEN` en compose, o renombrar la var del proyecto a `TUNNEL_TOKEN`)
- Healthcheck del container cloudflared: research recomienda usar el endpoint `/ready` con `--metrics 0.0.0.0:20241` (ver Pitfall 4)

### Deferred Ideas (OUT OF SCOPE)

- Layouts mobile-specific (Phase 8 sub-plan futuro)
- Rollback automático en `scripts/deploy.sh`
- CSP header (sub-plan futuro Mark-IV; requiere inventario inline assets)
- SSO con Google Workspace o Azure AD
- Sync de backups a NAS externo o S3
- WAF rules avanzadas Cloudflare (rate limiting agresivo, bot fight, geofencing)
- Custom Cloudflare Worker
- Tag de release v1.0.0 (vive en Phase 8 plan 08-11 o Mark-IV)
- Pruebas de carga / DDoS simulado
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CLOUD-01 | Dominio `nexo.app` (o fallback) en Cloudflare Registrar; nameservers CF; DNS público | Verified: CF Registrar disponible para `.app` TLD; coste wholesale ~$10-14/yr; HSTS preload obligatorio del TLD `.app` documentado |
| CLOUD-02 | Cuenta CF + 2FA + recovery codes + segundo admin <30d post-deploy | Verified: CF Free tier permite Zero Trust con 50 user cap; 2FA estándar en cuentas |
| CLOUD-03 | 5 security headers en Caddyfile.prod | Verified: sintaxis canónica `header { ... }` block; HSTS modifier 0 si presente / -20 si missing en Mozilla Observatory |
| CLOUD-04 | Servicio `cloudflared` en compose, target `https://caddy:443`, noTLSVerify | Verified: imagen oficial `cloudflare/cloudflared`, latest tag activo en 2025 (versión más reciente: 2025.10.0); `tunnel run` con token via env var canónico `TUNNEL_TOKEN` o flag `--token` |
| CLOUD-05 | Var `CLOUDFLARE_TUNNEL_TOKEN` en `.env.example` con procedencia | Verified: D-09 viable; **CORRECTION-01 — el nombre canónico que cloudflared lee es `TUNNEL_TOKEN`, requiere mapeo en compose** |
| CLOUD-06 | Access App `nexo` con email allowlist + OTP IdP + cookie 24h | Verified: OTP IdP free; 24h es **default** session duration (rango: immediate timeout → 1 mes); Free plan ≤50 users |
| CLOUD-07 | Custom denial page HTML al dashboard CF | **BLOCKER-01 — Custom Block Page Templates requieren plan Pay-as-you-go ($7/usr/mes) o Enterprise; NO disponible en Free.** Alternativas: (a) Redirect URL a página externa, (b) aceptar default CF page, (c) Cloudflare Pages estático |
| CLOUD-08 | `deploy_smoke.sh` 8→11 checks | Verified: archivo existe (94 líneas), patrón `[DEPLOY-XX] OK\|FAIL`, exit code = FAILS count, `set -uo pipefail` |
| CLOUD-09 | `cloudflare_smoke.sh` 6 checks externo | Verified: docs CF tienen `cloudflared tunnel ready` command + `/ready` endpoint para CF-06; CF-02..CF-05 son curl con flags estándar |
| CLOUD-10 | Makefile targets `cf-*` | Verified: patrón `prod-*` (7 targets) en líneas 142-173 de Makefile; mirror trivial |
| CLOUD-11 | `docs/CLOUDFLARE_DEPLOY.md` + ADR-001 + CLAUDE.md update + CHANGELOG | Verified: ADR-001 ya existe (`docs/decisions/ADR-001-cloudflare-tunnel.md`, 183 líneas, formato Nygard correcto, status "Proposed"); DEPLOY_LAN.md tiene 16 secciones para mirror; CHANGELOG.md Keep-a-Changelog 1.1.0 ya en repo |
| CLOUD-12 | LAN fallback verificado: smoke 11/11 con cloudflared corriendo Y parado | Verified: arquitectura aditiva pura; D-19 garantiza que parar cloudflared no toca el bloque `nexo.ecsmobility.local` ni `:443` de Caddyfile.prod |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| TLS público (edge) | Cloudflare Edge | — | Cloudflare termina TLS para `nexo.app` (cert real Google Trust Services o Let's Encrypt); cloudflared encapsula al edge sin que el servidor abra puertos |
| Email allowlist + OTP gating | Cloudflare Access | — | Capa de red previa a la app; rechaza requests antes de que toquen Caddy |
| TLS interno LAN | Caddy (`tls internal`) | — | Phase 6 ya cubre; sirve `nexo.ecsmobility.local` con CA local distribuida |
| Security headers (HSTS/XFO/CTO/RP/PP) | Caddy (`header` directive) | — | Caddy es el único choke point HTTP entre `cloudflared` ↔ `web` y entre LAN ↔ `web`; aplicar headers ahí cubre ambas ramas con un único cambio |
| Reverse proxy app HTTP | Caddy → `web:8000` | — | Sin cambios; Phase 6 ya configurado |
| App auth (argon2id + lockout + audit) | FastAPI middleware | Postgres `nexo.users` / `nexo.audit_log` | Segunda barrera detrás de Access; Phase 2 ya cubre |
| Túnel saliente conectividad | `cloudflared` container | Cloudflare Edge (4× redundant connections default) | Inicia conexión saliente; sin puertos en el host |
| Smoke verification | bash scripts (`tests/infra/`) | — | Patrón existente Phase 6 (`deploy_smoke.sh` 11 checks `[DEPLOY-XX]`); aditivo |
| DNS público | Cloudflare DNS (autoritativo) | — | Auto-managed; nameservers CF apuntan al edge |
| Logout endpoint | Cloudflare Access (`/cdn-cgi/access/logout`) | Nexo dropdown (Phase 8 cableado, fuera scope) | Endpoint provisto por CF, documentado en Phase 9 pero no integrado en UI |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cloudflare/cloudflared` | `latest` (resolves to `2025.10.0` o newer) [VERIFIED: Docker Hub] | Túnel saliente Docker hacia edge CF | Imagen oficial Cloudflare; única opción soportada para Cloudflare Tunnel; alpine-based, 27.4MB [CITED: hub.docker.com/r/cloudflare/cloudflared] |
| `caddy:2-alpine` | 2.x (ya pineado en `docker-compose.yml` línea 69) [VERIFIED: docker-compose.yml] | Reverse proxy + TLS termination LAN + security headers | Ya en stack Phase 6; v2.11.0+ tiene Host header default rewrite to upstream — relevante para D-20 [CITED: caddyserver.com/docs] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `curl` | system | Smoke checks externos (CF-01..CF-05) | En `cloudflare_smoke.sh`, mismo patrón que `deploy_smoke.sh` líneas 49-91 |
| `openssl s_client` | system | Verificar issuer real del cert público (CF-02) | Mismo patrón que `deploy_smoke.sh` línea 53-54 valida "Caddy Local Authority" — para CF-02 valida "Google Trust Services" o equivalente |
| `dig` | system | DNS público resuelve `nexo.app` (CF-01) | Verificación cross-resolver desde 4G |

### Alternatives Considered (rechazadas, ya en ADR-001)

| Instead of | Could Use | Tradeoff | Status |
|------------|-----------|----------|--------|
| Cloudflare Tunnel | Tailscale | TLS E2E sin tercero, pero requiere app cliente por dispositivo | Rejected ADR-001 §A |
| Cloudflare Tunnel | Port-forward 443 directo | Simple, pero peor postura de seguridad (sin DDoS/WAF/Access) | Rejected ADR-001 §C |
| OTP por email | Google Workspace / Azure AD SSO | Mejor UX, pero ECS no tiene Workspace gestionado | Deferred Mark-IV |
| Cloudflare Free | Pay-as-you-go ($7/usr/mes) | Custom block pages disponibles | Possible si BLOCKER-01 fuerza el cambio |
| `cloudflare/cloudflared:latest` | Pin a tag concreto (e.g. `2025.10.0`) | Reproducibilidad vs auto-updates | Decision: usar `:latest` pero documentar que CI smoke valida en cada deploy; rotación si breaking change |

**Installation (Docker Compose addition, no `npm install`):**

```yaml
# docker-compose.prod.yml (añadido bajo services:)
cloudflared:
  image: cloudflare/cloudflared:latest
  command: tunnel --no-autoupdate run
  environment:
    - TUNNEL_TOKEN=${CLOUDFLARE_TUNNEL_TOKEN}
    - TUNNEL_METRICS=0.0.0.0:20241  # Para healthcheck
  depends_on:
    caddy:
      condition: service_healthy
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "cloudflared", "tunnel", "--metrics", "127.0.0.1:20241", "ready"]
    interval: 30s
    timeout: 5s
    retries: 3
    start_period: 30s
```

**Version verification (verified 2026-04-25):**
- `cloudflare/cloudflared:latest` → resolves to `2025.10.0` (latest tag listed on Docker Hub Oct 2025) [VERIFIED: hub.docker.com/r/cloudflare/cloudflared]
- `caddy:2-alpine` → ya pineado a `2.x` en docker-compose.yml; v2.11.0+ tiene el cambio de Host header documentado [CITED: github.com/caddyserver/caddy/issues/7527]

## Architecture Patterns

### System Architecture Diagram

```
┌─────────────────┐                                    ┌──────────────────────┐
│ Browser móvil   │ HTTPS public                       │ Edge Cloudflare       │
│ (4G / WiFi)     │──────cert real Google TS──────────▶│ TLS termination       │
│ user@email.com  │                                    │ + Access gating       │
└─────────────────┘                                    │ (email allowlist+OTP) │
                                                       └──────────┬───────────┘
                                                                  │
                                                       Cloudflare Tunnel
                                                       (outbound from server,
                                                        4× redundant conns)
                                                                  │
                                                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Servidor Ubuntu LAN ECS (sin puertos abiertos a internet)                │
│                                                                          │
│    ┌──────────────┐   internal   ┌──────────┐    internal   ┌─────────┐ │
│    │ cloudflared  │──HTTPS:443──▶│  Caddy   │──HTTP:8000───▶│  web    │ │
│    │ (container)  │  noTLSVerify │ (tls     │  reverse_     │ FastAPI │ │
│    │ outbound     │              │  internal│  proxy        │ uvicorn │ │
│    │ tunnel       │              │  + headers)│  + headers  │  Nexo   │ │
│    └──────────────┘              └────▲─────┘                └─────────┘ │
│                                       │                                   │
│    LAN equipos ECS                    │ HTTPS:443                        │
│    (hosts-file +                      │ tls internal                     │
│     root CA)                          │ "nexo.ecsmobility.local"         │
│    ───────────────────────────────────┘                                   │
│                                                                          │
│    Postgres (nexo.users + audit_log)  ◀── argon2id auth segunda barrera  │
│    SQL Server MES (read-only, otra LAN)                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

**Flujo de denegación (Access policy reject):**
1. Usuario fuera de allowlist abre `https://nexo.app` → Cloudflare edge intercepta
2. Cloudflare muestra login Access; usuario mete email; CF verifica contra policy
3. Email NO está en allowlist → Cloudflare envía email NO (D-15 OTP behavior); página de login dice "A code has been emailed to you" pero no llega [CITED: developers.cloudflare.com/cloudflare-one/integrations/identity-providers/one-time-pin/]
4. Usuario no puede entrar nunca; tras N reintentos ve la denial page (default CF o redirect a URL custom — ver BLOCKER-01)

**Flujo de éxito:**
1. Usuario en allowlist abre `https://nexo.app` → Cloudflare edge intercepta
2. CF muestra login Access; usuario mete email autorizado
3. CF envía OTP de 6 dígitos al email (expira en 10 min); usuario lo mete
4. CF inyecta cookie `CF_Authorization` (24h) y forward request a través del túnel
5. cloudflared envía a `https://caddy:443` con SNI=`caddy` interno y Host header=`nexo.app`
6. Caddy aplica security headers + reverse_proxy a `web:8000`
7. Nexo middleware ve cookie de sesión Nexo (o redirige a `/login`); usuario hace login argon2id; `request.state.user` poblado; render página

### Recommended Project Structure (changes only — additive)

```
analisis_datos/
├── docker-compose.prod.yml          # MODIFY: añadir servicio cloudflared
├── caddy/Caddyfile.prod             # MODIFY: añadir bloque header (5 directives)
├── .env.example                     # MODIFY: añadir CLOUDFLARE_TUNNEL_TOKEN con comentario
├── CLAUDE.md                        # MODIFY: sección "Despliegue" + "Qué NO hacer"
├── CHANGELOG.md                     # MODIFY: añadir entry [Phase 9]
├── Makefile                         # MODIFY: añadir 5 cf-* targets (~10 líneas)
├── scripts/
│   └── cloudflare-bootstrap.sh      # NEW: validación pre-deploy CF
├── tests/infra/
│   ├── deploy_smoke.sh              # MODIFY: 8→11 checks (DEPLOY-09/10/11)
│   └── cloudflare_smoke.sh          # NEW: 6 checks CF-01..CF-06 desde fuera
├── static/
│   └── cloudflare-denial.html       # NEW (revisar tras BLOCKER-01): denial page Nexo
├── docs/
│   ├── CLOUDFLARE_DEPLOY.md         # NEW: runbook 16 secciones (mirror DEPLOY_LAN.md)
│   └── decisions/
│       └── ADR-001-cloudflare-tunnel.md  # ALREADY EXISTS (status: Proposed → Accepted)
└── api/                             # NO CHANGES (auth interna intacta)
└── nexo/                            # NO CHANGES
```

### Pattern 1: Token-based Remote-Managed Tunnel (D-08)

**What:** Cloudflared lee un token JWT que identifica el tunnel; toda la config (ingress rules, hostname mapping, Access apps) vive en el dashboard de Cloudflare, NO en YAML local.

**When to use:** Cuando se quiere "una sola fuente de verdad" para la config del túnel (vs locally-managed donde `config.yml` local es la fuente y se sube). Es el modo recomendado por Cloudflare desde 2022.

**Example:**

```yaml
# docker-compose.prod.yml — adicional a services:
cloudflared:
  image: cloudflare/cloudflared:latest
  command: tunnel --no-autoupdate run
  environment:
    - TUNNEL_TOKEN=${CLOUDFLARE_TUNNEL_TOKEN}
    - TUNNEL_METRICS=0.0.0.0:20241
  depends_on:
    caddy:
      condition: service_healthy
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "cloudflared", "tunnel", "--metrics", "127.0.0.1:20241", "ready"]
    interval: 30s
    timeout: 5s
    retries: 3
    start_period: 30s
```

**Source:** [VERIFIED: developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/configure-tunnels/run-parameters/] + [CITED: hub.docker.com/r/cloudflare/cloudflared]

**Notes:**
- `--no-autoupdate` recomendado en Docker para que el container no intente reiniciarse a sí mismo durante operación
- `command: tunnel --no-autoupdate run` (sin `--token` flag) — el token se lee de `TUNNEL_TOKEN` env var, lo cual evita exponer el token en `docker inspect` / process listing
- `TUNNEL_METRICS=0.0.0.0:20241` necesario para que `cloudflared tunnel ready` (healthcheck) pueda hablar con el endpoint
- `depends_on: caddy: condition: service_healthy` requiere que el `healthcheck` de caddy en el base compose esté arriba; `docker-compose.prod.yml` ya añade healthcheck a caddy (líneas 48-53)

### Pattern 2: Caddy Security Headers Block (D-13)

**What:** Bloque `header` con 5 directives aplicado al sitio. Sintaxis canónica multi-línea.

**Example:**

```caddyfile
# caddy/Caddyfile.prod (añadir DENTRO de cada bloque de sitio O usar como snippet)

(security_headers) {
    header {
        Strict-Transport-Security "max-age=31536000; includeSubDomains; preload"
        X-Frame-Options "DENY"
        X-Content-Type-Options "nosniff"
        Referrer-Policy "strict-origin-when-cross-origin"
        Permissions-Policy "geolocation=(), microphone=(), camera=()"
    }
}

nexo.ecsmobility.local {
    import security_headers
    tls internal
    reverse_proxy web:8000
    log {
        output stdout
        format json
    }
}

:443 {
    import security_headers
    tls internal
    reverse_proxy web:8000
}
```

**Source:** [VERIFIED: caddyserver.com/docs/caddyfile/directives/header]

**Por qué snippet `(security_headers)`:** Evita duplicar las 5 directivas en los 2 bloques (`nexo.ecsmobility.local` y `:443` catch-all). Caddy soporta named snippets — `(snippet_name) { ... }` define + `import snippet_name` lo expande. [CITED: caddyserver.com/docs/caddyfile/concepts]

**Quoted vs unquoted values:** Las docs oficiales muestran ambos estilos. Para HSTS con `;` y espacios, **quoted es más legible y seguro** — la regla práctica es: si el valor contiene espacios o caracteres especiales, citar. La research recomienda usar quoted para los 5 (consistencia + legibilidad + sin sorpresas con `caddy fmt`). [CITED: caddyserver.com/docs/caddyfile/directives/header §Examples]

### Pattern 3: Bash Smoke Check (mirror Phase 6 `deploy_smoke.sh`)

**What:** Cada check llama `check "<TAG>" "<msg>" "<cmd>"` que ejecuta `eval "${cmd}"` y cuenta fallos. Exit code = número de fails.

**Source:** [VERIFIED: tests/infra/deploy_smoke.sh líneas 30-40]

**Pattern (extracted):**

```bash
#!/usr/bin/env bash
set -uo pipefail
IFS=$'\n\t'

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

# Ejemplos para cloudflare_smoke.sh:
check "CF-01" "DNS público resuelve nexo.app" \
    "dig +short nexo.app | grep -qE '^[0-9]'"

check "CF-02" "cert real (Google Trust Services o equiv.) en nexo.app" \
    "echo | openssl s_client -connect nexo.app:443 -servername nexo.app 2>/dev/null | grep -qiE 'issuer.*(Google|Lets Encrypt)'"

check "CF-03" "request con JWT inválido es rechazado por Access (302 → login)" \
    "[ \"\$(curl -s -o /dev/null -w '%{http_code}' -H 'CF-Access-Jwt-Assertion: invalid' https://nexo.app/)\" = '302' ]"

check "CF-04" "GET sin headers redirige a Access login (cloudflareaccess.com)" \
    "curl -sI https://nexo.app/ | grep -qiE 'location:.*cloudflareaccess\\.com'"

check "CF-05" "headers de seguridad presentes (HSTS + X-Frame-Options + X-Content-Type-Options)" \
    "curl -sI https://nexo.app/ | grep -iE 'strict-transport-security|x-frame-options|x-content-type-options' | wc -l | grep -q '^[3-9]'"

check "CF-06" "tunnel info nexo-prod reporta connectorStatus HEALTHY (ejecutado en server)" \
    "docker compose exec -T cloudflared cloudflared tunnel --metrics 127.0.0.1:20241 ready"

echo "=== Fallos: ${FAILS} ==="
exit "${FAILS}"
```

**Notes:**
- CF-03 nota: el código real puede ser 302 a `https://*.cloudflareaccess.com/` (pre-Access login). Verificar empíricamente en deploy real; la lógica correcta es "no llega a la app".
- CF-06 corre desde el servidor (no externo) usando docker exec; la versión "externa" sólo puede verificar que la URL pública responde (cubierto por CF-04).

### Pattern 4: Makefile cf-* targets (mirror prod-* lines 142-173)

```makefile
# ── Cloudflare Tunnel (Phase 9) ──────────────────────────────────────────────
# Targets cf-* operan SOLO sobre el container cloudflared.
# El stack prod sigue arrancando con make prod-up (incluye cloudflared automáticamente
# por estar en docker-compose.prod.yml). Estos targets son para operar el túnel
# de forma independiente (rollback rápido D-24, debug, etc.).

cf-up: ## Arranca solo el container cloudflared (asume stack prod arriba)
	$(PROD_COMPOSE) up -d cloudflared
	@echo "  cloudflared arrancado. Verifica con 'make cf-status'."

cf-down: ## Para SOLO cloudflared (LAN fallback intacto, D-24)
	$(PROD_COMPOSE) stop cloudflared
	@echo "  cloudflared parado. LAN fallback sigue accesible en https://nexo.ecsmobility.local"

cf-status: ## Estado del container cloudflared + healthcheck
	$(PROD_COMPOSE) ps cloudflared
	@$(PROD_COMPOSE) exec -T cloudflared cloudflared tunnel --metrics 127.0.0.1:20241 ready && \
		echo "  tunnel HEALTHY" || echo "  tunnel UNHEALTHY"

cf-logs: ## Logs cloudflared en tiempo real
	$(PROD_COMPOSE) logs -f cloudflared

cf-smoke-external: ## Ejecuta cloudflare_smoke.sh (smoke desde fuera de la LAN, requiere ejecutar en máquina externa)
	bash tests/infra/cloudflare_smoke.sh
```

**Source pattern:** [VERIFIED: Makefile líneas 142-173 (prod-up/prod-down/etc)]

### Anti-Patterns to Avoid

- **Hardcoding token en `docker-compose.prod.yml`:** `command: tunnel run --token abc123...` filtra el token a `docker inspect`. Usar `TUNNEL_TOKEN` env var siempre.
- **`cloudflared` apuntando a `web:8000`:** Salta Caddy → no aplica security headers + asimetría LAN/CF (D-12 lo descarta). Apuntar a `https://caddy:443` siempre.
- **`tlsVerify: true` con cert interno de Caddy:** Caddy `tls internal` emite cert para `caddy` (Docker hostname); cloudflared no lo confiará. Aceptar `noTLSVerify: true` (D-11). Documentar que el tramo es Docker network (no expuesta).
- **CSP sin inventario inline:** Activar CSP sin haber inventariado todo el JS/CSS inline (Alpine, scripts en templates) rompe la app silenciosamente. D-13 explícitamente excluye CSP de Phase 9.
- **Olvidar redirect `:80` → `:443`:** Sin esto un browser que reciba HSTS pero pruebe HTTP cae en error duro. Caddy v2 lo hace por default (sin `auto_https off`). [CITED: Caddyfile.prod comentario línea 4]
- **Smoke externo desde la misma LAN:** Si el smoke se ejecuta desde el servidor, el DNS de `nexo.app` puede resolver al edge de CF pero la conexión no atraviesa internet → no detecta firewalls corporativos rotos. CF-01..CF-05 deben correrse desde 4G real.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Tunnel saliente HTTP/HTTPS | wireguard ad-hoc / OpenVPN custom | `cloudflare/cloudflared` Docker image oficial | TCP keepalive, reconexión automática, multiplexing 4 conns redundantes, métricas Prometheus built-in |
| Email OTP IdP | SMTP + tabla de OTPs en Postgres + endpoints custom | Cloudflare Access OTP IdP (free) | OTP delivery, expiración 10min, rate limiting, audit log = 0 código en Nexo |
| Healthcheck cloudflared | `curl` custom contra hostname interno | `cloudflared tunnel --metrics 127.0.0.1:20241 ready` | Comando oficial del binario, exit 0 si tunnel HEALTHY [CITED: github.com/cloudflare/cloudflared PR #1135] |
| Custom denial page hosting | Servir HTML desde Caddy con bypass de Access | **Sub-pendiente decisión BLOCKER-01** — Cloudflare Pages estático O bloque público en propio Caddy en path `/access-denied` | Free plan no permite custom HTML en CF dashboard; Cloudflare Pages incluye en Free; Caddy bloque público es trivial |
| Security headers | Custom middleware FastAPI | Caddy `header` directive | Caddy es el reverse proxy que ya termina TLS; añadir headers en Caddy aplica a TODA respuesta, no requiere modificar código Python |
| Smoke test con verificación issuer | parsear cert manualmente con `openssl x509` + grep | `openssl s_client | grep issuer` (patrón ya en deploy_smoke.sh línea 53-54) | Patrón ya validado en repo |
| Domain registration UI | sub-cuenta Namecheap / Google Domains | Cloudflare Registrar | wholesale price, mismo dashboard que Tunnel/Access, sin DNS propagation entre proveedores |

**Key insight:** Para esta phase, "no hand-roll" es trivialmente cierto — toda la complejidad operativa la absorben Cloudflare (Tunnel + Access + Registrar) y Caddy (TLS + headers). El código del proyecto se limita a 1 servicio nuevo en compose, ~5 líneas de header en Caddyfile, ~10 líneas de Makefile, ~80-120 líneas de bash en `cloudflare_smoke.sh`, y docs.

## Runtime State Inventory

> Phase 9 es **aditiva pura** (no rename, no migration). Esta sección se incluye para confirmar que NO hay state oculto que migrar.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — verified by inspection: `nexo.users`, `nexo.audit_log`, `nexo.query_log`, `nexo.login_attempts` no embeben strings de hostname/dominio que cambien con CF activación. La cookie de sesión Nexo es independiente de la cookie `CF_Authorization` (diferente dominio scope). | None |
| Live service config | El registro DNS de `nexo.app` apunta al edge CF (gestionado en dashboard, NO en git). Tunnel config (ingress, hostname mapping) vive en dashboard CF (D-08 token-based remote-managed) — la única referencia local es `TUNNEL_TOKEN` en `.env`. | Documentar en CLOUDFLARE_DEPLOY.md que la config del tunnel está en dashboard, NO en repo; rotación de token = nuevo `.env` + restart container |
| OS-registered state | None — el servidor Ubuntu sólo gana 1 container Docker más; cron de backup nightly de Phase 6 no toca cloudflared; `ufw` rules sin cambios (cloudflared usa outbound, no requiere INPUT rules). | None |
| Secrets/env vars | NEW: `CLOUDFLARE_TUNNEL_TOKEN` en `/opt/nexo/.env` (perms 600 ya forzados por Phase 6 D-13 del runbook DEPLOY_LAN.md). Note: existe `.env.prod.example` además de `.env.example` — verificar cuál actualizar (probable: ambos, con `.env.prod.example` siendo el canónico para deploy real). | Añadir `CLOUDFLARE_TUNNEL_TOKEN=` (vacío) + comentario procedencia en `.env.example` y `.env.prod.example`; rotación documentada en RUNBOOK |
| Build artifacts | None — cloudflared se pull al `make prod-up`, no se compila localmente. Imagen oficial Cloudflare cacheada en host docker. | None |

**Nothing found in Stored data, OS-registered state, Build artifacts categories — verified by direct inspection.**

## Common Pitfalls

### Pitfall 1: TUNNEL_TOKEN env var canonical name (CORRECTION-01)

**What goes wrong:** D-09 dice usar `CLOUDFLARE_TUNNEL_TOKEN`. Si el bloque `environment:` del compose pone `CLOUDFLARE_TUNNEL_TOKEN: ${CLOUDFLARE_TUNNEL_TOKEN}`, cloudflared NO lo lee — espera específicamente `TUNNEL_TOKEN`. El container arrancará pero fallará con "no token provided".

**Why it happens:** El binario cloudflared tiene una lista hard-coded de env vars que reconoce: `TUNNEL_TOKEN`, `TUNNEL_METRICS`, `TUNNEL_LOGFILE`, etc. (patrón estándar Cloudflare CLI tooling).

**How to avoid:** Mapear la variable del proyecto al nombre canónico en compose:

```yaml
environment:
  - TUNNEL_TOKEN=${CLOUDFLARE_TUNNEL_TOKEN}  # mapeo proyecto → cloudflared canónico
  - TUNNEL_METRICS=0.0.0.0:20241
```

**Warning signs:** `make cf-logs` muestra "no tunnel token provided" o el tunnel queda en estado `connectorStatus: REGISTERING` permanentemente.

**Source:** [VERIFIED: developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/configure-tunnels/run-parameters/]

### Pitfall 2: SNI mismatch + 421 Misdirected Request (CORRECTION-02 / R-05)

**What goes wrong:** D-20 anticipa que `cloudflared → https://caddy:443` con SNI=`caddy` (hostname Docker) puede provocar `421 Misdirected Request` desde Caddy si éste hace strict SNI matching contra el Host header (`nexo.app`).

**Why it happens:** Caddy v2 tiene una opción global `strict_sni_host` que cuando está activa devuelve 421 si Host ≠ SNI. Está **DESHABILITADA por default** salvo cuando hay `client_auth` configurado. [CITED: caddyserver.com/docs/caddyfile/options]

**How to avoid:**
- Confirmar que `Caddyfile.prod` NO tiene bloque global con `strict_sni_host` activado (verificado: el archivo actual no tiene global options block)
- El bloque catch-all `:443` actual (líneas 21-24 de Caddyfile.prod) absorbe el request del tunnel, ignora SNI mismatch, y reverse_proxy lo envía a web:8000
- Si en algún momento R-05 se materializa empíricamente: añadir bloque explícito `nexo.app { import security_headers; tls internal; reverse_proxy web:8000 }` para que Caddy tenga handling explícito por hostname

**Warning signs:** `make cf-logs` muestra cloudflared conectado pero CF-04 / CF-05 fallan con 421 en lugar de 302; `caddy logs` muestra "TLS handshake error" o "no matching site".

**Source:** [VERIFIED: caddyserver.com/docs/caddyfile/options] + [CITED: caddy.community/t/does-it-make-sense-to-offer-finer-control-for-strict-sni-host/33445]

### Pitfall 3: Caddy 2.11.0+ Host header default rewrite

**What goes wrong:** A partir de v2.11.0 Caddy reescribe automáticamente el Host header al hostport del upstream cuando upstream es HTTPS (`https://caddy:443` → Host=`caddy:443`). Apps que esperan ver `nexo.app` en el Host header (e.g., para generar URLs absolutas) podrían romperse.

**Why it happens:** Cambio de comportamiento por defecto introducido en Caddy 2.11.0 (Dec 2024) para mejorar compat con S3/upstream que requieren Host=upstream. [CITED: github.com/caddyserver/caddy/issues/7527]

**How to avoid:** Si Nexo (`web:8000`) genera URLs basándose en `request.headers["host"]`, añadir explícitamente al bloque caddy:

```caddyfile
nexo.ecsmobility.local {
    reverse_proxy web:8000 {
        header_up Host {host}
    }
}
```

**Verification step:** En el smoke (DEPLOY-09 nuevo o test integration), curl con `-H 'Host: nexo.app'` y verificar que la app vuelve URLs con `nexo.app` (no `caddy:443`). El test puede ser: `curl -k -H 'Host: nexo.app' https://localhost/api/health` desde dentro del container.

**Warning signs:** Toasts de redirect en login generan URLs `caddy:443/cambiar-password` que el browser no resuelve.

**Source:** [VERIFIED: github.com/caddyserver/caddy/issues/7527] + [CITED: caddyserver.com/docs/caddyfile/directives/reverse_proxy]

### Pitfall 4: Cloudflared healthcheck without TUNNEL_METRICS

**What goes wrong:** El healthcheck `cloudflared tunnel --metrics 127.0.0.1:20241 ready` falla con "connection refused" si el container no arrancó con el endpoint `/ready` en ese puerto. Por default cloudflared elige un puerto efímero entre 20241-20245.

**Why it happens:** Para que `cloudflared tunnel ready` (subcommand añadido en PR #1135) funcione, el proceso principal necesita exponer `/ready` en un puerto conocido. Sin `TUNNEL_METRICS=0.0.0.0:20241` env var, el puerto es no determinístico. [CITED: github.com/cloudflare/cloudflared/pull/1135]

**How to avoid:** Setear `TUNNEL_METRICS=0.0.0.0:20241` (o cualquier puerto fijo dentro del rango 20241-20245) y usar el mismo puerto en el healthcheck:

```yaml
environment:
  - TUNNEL_METRICS=0.0.0.0:20241
healthcheck:
  test: ["CMD", "cloudflared", "tunnel", "--metrics", "127.0.0.1:20241", "ready"]
```

**Warning signs:** `docker inspect <cloudflared> --format '{{.State.Health.Status}}'` queda en `unhealthy` o `starting` permanentemente.

**Source:** [VERIFIED: developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/monitor-tunnels/metrics/] + [CITED: github.com/cloudflare/cloudflared PR #1135]

### Pitfall 5: Token leak via `docker inspect`

**What goes wrong:** Si el compose pasa el token como CLI arg (`command: tunnel run --token ${CLOUDFLARE_TUNNEL_TOKEN}`), `docker inspect` y `ps -ef` lo exponen en process listing.

**Why it happens:** Process command line es leíble por cualquier usuario con acceso al host (incluso non-root vía `/proc`).

**How to avoid:** Pasar token vía env var, NO vía CLI arg:

```yaml
# RECOMENDADO
command: tunnel --no-autoupdate run
environment:
  - TUNNEL_TOKEN=${CLOUDFLARE_TUNNEL_TOKEN}
```

```yaml
# EVITAR (token visible en docker inspect)
command: tunnel --no-autoupdate run --token ${CLOUDFLARE_TUNNEL_TOKEN}
```

**Warning signs:** Auditoría de seguridad encuentra el token en `ps aux` o `cat /proc/<pid>/cmdline`.

**Source:** [VERIFIED: developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/configure-tunnels/run-parameters/]

### Pitfall 6: BLOCKER-01 — Custom denial HTML on Free Plan

**What goes wrong:** D-17 dice "subir HTML estático con branding Nexo al dashboard CF en Settings → Custom Pages → Forbidden". Las docs oficiales confirman que **esta funcionalidad requiere plan Pay-as-you-go ($7/usr/mes) o Enterprise**. Free plan solo permite (a) la página default de Cloudflare ("You don't have access — sign in with another email") y (b) **Redirect URL** a una URL externa.

**Why it happens:** Cloudflare reserva el "Custom Block Page Template" feature para tiers pagos.

**How to avoid:** El planner debe escoger UNA de estas 3 opciones antes de escribir 09-02 PLAN.md (o escalar al operador en `/gsd-discuss-phase` rerun):

**Opción A — Aceptar default page (CHEAPEST, FUNCIONA YA):**
- Pros: cero código adicional, cero coste
- Cons: branding Cloudflare, mensaje genérico ("sign in with another email"), no tiene email de contacto Erik
- Decisión: si la fricción es aceptable porque los rechazados se canalizan vía boca-a-boca a Erik

**Opción B — Redirect URL a página servida por Caddy en `/access-denied` (RECOMENDADO):**
- En Cloudflare Access dashboard: configurar la Access App con Block page = "Redirect URL" → `https://nexo.app/access-denied`
- En Caddyfile.prod: añadir bloque o ruta pública (sin Access) que sirva `static/cloudflare-denial.html` o un endpoint FastAPI público `/access-denied`
- **Subtlety:** El path `/access-denied` debe estar fuera del scope de Access App. La Access App está configurada como `nexo.app/*` (catch-all). Para que un path quede público, hay 2 maneras: (1) crear una segunda Access App con prioridad mayor con policy "Allow Everyone" para `/access-denied`, (2) usar un subdominio `denied.nexo.app` sin Access policy
- Pros: branding completo Nexo, mensaje custom, todo en repo
- Cons: requiere coordinación de Access policies (más complejo en dashboard)

**Opción C — Cloudflare Pages estático separado:**
- Crear un proyecto Cloudflare Pages (free) con HTML estático
- Configurar Access App Block page = Redirect URL → `https://nexo-denied.pages.dev`
- Pros: no toca Caddy, rama clara
- Cons: introduce un proyecto CF adicional que mantener

**How to avoid:** Decisión documentada en PLAN.md de 09-02 antes de escribir código. Recomendación research: **Opción B** porque mantiene todo en el repo y permite que el operador edite el mensaje sin tocar el dashboard CF.

**Warning signs:** Plan escrito asumiendo que `static/cloudflare-denial.html` se sube al dashboard; al ejecutar el operador no encuentra el botón en el panel Free.

**Source:** [VERIFIED: developers.cloudflare.com/cloudflare-one/reusable-components/custom-pages/access-block-page/] - "Custom Page Template only available on Pay-as-you-go and Enterprise plans"

### Pitfall 7: deploy_smoke.sh must work with cloudflared container PARADO

**What goes wrong:** CLOUD-12 / Success Criterion 6 exige que `deploy_smoke.sh` siga 11/11 OK incluso con `docker compose stop cloudflared` (verificación de independencia LAN fallback). Si `[DEPLOY-11]` queda como "cloudflared running" en lugar de "cloudflared running OR voluntarily stopped", parar cloudflared rompe el smoke LAN.

**Why it happens:** El check naive `docker inspect cloudflared --format '{{.State.Status}}' | grep -q running` falla cuando el container está stopped.

**How to avoid:** El check `[DEPLOY-11]` debe expresar "cloudflared está running Y healthy" como una validación de la rama CF, pero el smoke LAN completo (CLOUD-12) debe poder pasar sin él. Dos approaches:

**Approach A:** Saltar el check `[DEPLOY-11]` si cloudflared no está corriendo:
```bash
if $COMPOSE ps cloudflared --format json 2>/dev/null | grep -q '"State":"running"'; then
    check "DEPLOY-11" "..." "..."
else
    echo "[DEPLOY-11] SKIP: cloudflared no running (rama CF parada — LAN-only mode)"
fi
```

**Approach B:** Dos smokes diferentes — `deploy_smoke.sh` (10 checks LAN-only, NO incluye DEPLOY-11) + `deploy_smoke_full.sh` o flag `--with-cloudflared` (11 checks).

**Recomendación research:** Approach A — un solo script con SKIP graceful, alineado con el patrón Phase 6 donde `[DEPLOY-XX]` ya tienen exit semantics tolerantes (`set -uo pipefail` SIN `-e`).

**Warning signs:** `make cf-down && bash tests/infra/deploy_smoke.sh` exit code != 0 → el smoke acopla DEPLOY-11 al núcleo LAN.

### Pitfall 8: `.env.example` vs `.env.prod.example` — dual sources

**What goes wrong:** El repo tiene 2 archivos: `.env.example` (probablemente para dev) y `.env.prod.example` (canónico para deploy LAN, referenciado en `docs/DEPLOY_LAN.md`). CLOUD-05 dice añadir `CLOUDFLARE_TUNNEL_TOKEN` a `.env.example` — la realidad es que el operador en producción copia `.env.prod.example` → `/opt/nexo/.env`.

**How to avoid:** Añadir la variable a **ambos** archivos:
- `.env.example` (dev): vacío + comentario "solo aplica en prod, ignorar en dev"
- `.env.prod.example` (prod canónico): vacío + comentario procedencia ("Get from Cloudflare Zero Trust → Networks → Tunnels → nexo-prod → Install connector → token")

**Warning signs:** El operador lee CLOUDFLARE_DEPLOY.md, copia `.env.prod.example` → `.env`, pero la variable falta porque solo se añadió a `.env.example`.

**Source:** [VERIFIED: ls .env*] confirma 3 archivos: `.env`, `.env.example`, `.env.prod.example`

## Code Examples

Verified patterns from official sources:

### docker-compose.prod.yml addition (D-04 + D-08 + D-11)

```yaml
# Añadir bajo services: en docker-compose.prod.yml
# Source: https://hub.docker.com/r/cloudflare/cloudflared
#         https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/configure-tunnels/run-parameters/

services:
  # ... db, web, caddy ya existentes ...

  cloudflared:
    image: cloudflare/cloudflared:latest
    command: tunnel --no-autoupdate run
    environment:
      # CORRECTION-01: cloudflared lee TUNNEL_TOKEN, no CLOUDFLARE_TUNNEL_TOKEN.
      # Mapeamos la variable de proyecto al nombre canónico aquí.
      - TUNNEL_TOKEN=${CLOUDFLARE_TUNNEL_TOKEN}
      # TUNNEL_METRICS necesario para el healthcheck `cloudflared tunnel ready`
      - TUNNEL_METRICS=0.0.0.0:20241
    depends_on:
      caddy:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "cloudflared", "tunnel", "--metrics", "127.0.0.1:20241", "ready"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 30s
    deploy:
      resources:
        limits:
          cpus: "0.25"
          memory: 128m
```

### Caddyfile.prod with security headers + named snippet

```caddyfile
# caddy/Caddyfile.prod — produccion LAN + Cloudflare Tunnel
# Phase 6 (LAN HTTPS) + Phase 9 (CF Tunnel + headers)

# Snippet reusable de security headers (aplicado a ambas ramas LAN+CF)
# Source: https://caddyserver.com/docs/caddyfile/directives/header
(security_headers) {
    header {
        Strict-Transport-Security "max-age=31536000; includeSubDomains; preload"
        X-Frame-Options "DENY"
        X-Content-Type-Options "nosniff"
        Referrer-Policy "strict-origin-when-cross-origin"
        Permissions-Policy "geolocation=(), microphone=(), camera=()"
    }
}

# Rama LAN (D-19): equipos con hosts-file + root CA
nexo.ecsmobility.local {
    import security_headers
    tls internal
    reverse_proxy web:8000

    log {
        output stdout
        format json
    }
}

# Rama Cloudflare Tunnel (D-20): cloudflared → SNI=caddy:443 → este bloque catch-all
# También sirve fallback IP-directa para LAN.
:443 {
    import security_headers
    tls internal
    reverse_proxy web:8000
}
```

### .env.prod.example addition (CLOUD-05)

```bash
# .env.prod.example — Phase 9 addition

# ── Cloudflare Tunnel (Phase 9) ───────────────────────────────────────────────
# Token del tunnel "nexo-prod" en Cloudflare Zero Trust.
# Obtener desde: dash.cloudflare.com → Zero Trust → Networks → Tunnels →
#                nexo-prod → Install connector → token (string largo base64-like)
# Rotación: dashboard CF → Configure → Token → Refresh; pegar nuevo valor aquí
#           y ejecutar `make cf-down && make cf-up` (sin downtime LAN).
# NUNCA commitear este archivo con valor real (.env.prod.example tiene placeholder vacío).
CLOUDFLARE_TUNNEL_TOKEN=
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Cloudflare Tunnel "locally-managed" (config.yml en disco) | "Remote-managed" token-based (config en dashboard) | 2022 | D-08 alineado con state of the art; permite que el dashboard CF sea single source of truth |
| `cloudflared tunnel run --token <T>` (CLI arg) | `TUNNEL_TOKEN` env var en compose | Siempre disponible, recomendado para Docker | Pitfall 5 evita token leak via `docker inspect` |
| `cloudflared tunnel run --token-file <P>` | Disponible desde v2025.4.0 | Abr 2025 | Alternativa para secrets managers (vault, etc.); no necesario para nuestro caso |
| Custom Caddy build con `xcaddy` para security headers | `header { ... }` directive nativa | Siempre disponible en Caddy v2 | Sin necesidad de custom build |
| Cloudflare Access "Standalone" (cuenta separada) | Zero Trust unificado | 2022 | Cuenta normal CF + Zero Trust panel = misma cuenta; sin upgrade adicional |
| OTP IdP requiere config explícita por app | OTP IdP siempre disponible si Access App tiene Allow policy con email match | Siempre | Nada que añadir más allá de la policy |
| Caddy v2.10.x — Host header forwarded as-is | Caddy v2.11.0+ — Host header reescrito a upstream hostport si HTTPS upstream | Dic 2024 | Pitfall 3; verificar empíricamente impacto en URLs absolutas de Nexo |

**Deprecated/outdated:**
- `cloudflared` ejecutado como systemd service en host (mode pre-2022) — descartado por D-07 a favor de Docker container
- Cloudflare Argo Tunnel naming (pre-2021) — el producto se renombró a "Cloudflare Tunnel"; docs antiguas usan ambos nombres
- `tls insecure_skip_verify` en Caddy reverse_proxy con HTTPS upstream — sintaxis antigua; v2 usa `tls_insecure_skip_verify` dentro de `transport http { ... }` block (no aplica directamente aquí porque el proxy es Caddy → web HTTP, no HTTPS)

## Project Constraints (from CLAUDE.md)

Directivas extraídas de `./CLAUDE.md` que el planner DEBE honrar:

- **Pre-commit scoped a `^(api|nexo)/`**: los archivos nuevos `tests/infra/cloudflare_smoke.sh`, `scripts/cloudflare-bootstrap.sh`, `static/cloudflare-denial.html`, `docs/CLOUDFLARE_DEPLOY.md`, `docs/decisions/ADR-001-cloudflare-tunnel.md` quedan FUERA del scope de pre-commit hooks. CI tampoco les corre ruff/mypy. PASS por design.
- **Coverage gate ≥60%** (`pyproject.toml [tool.coverage.report].fail_under = 60` y `--cov-fail-under=60` en CI): Phase 9 NO añade código Python ejecutable (sólo bash + YAML + Markdown + 1 HTML estático opcional). El gate sigue cubierto por código Phase 1-8 existente. Plan no debe degradar.
- **Conventional Commits** (feat/fix/refactor/docs/test/chore/perf/ci/build/plan): los commits de Phase 9 son principalmente `feat:`, `docs:`, `test:`, `chore:`. Branch principal `feature/Mark-III` (no PR a `main` hasta cierre Mark-III).
- **No `git filter-repo`, no `git push --force` sobre `main`/`feature/Mark-III`**: aplica especialmente al token CF — si se commitea por error, rotar desde dashboard CF, NO reescribir historial.
- **No tocar carpeta `OEE/`**: sin riesgo, Phase 9 no la roza.
- **No instalar SMTP**: sin impacto, OTP por email lo gestiona Cloudflare (no requiere SMTP propio).
- **Idioma**: títulos commit en inglés, body en español si aporta contexto. Docs operativos (CLOUDFLARE_DEPLOY.md, denial.html messages) en español. ADR-001 ya escrito en español, mantener consistencia.
- **`.env` con perms 600** (Phase 6 D-13 del DEPLOY_LAN.md): `CLOUDFLARE_TUNNEL_TOKEN` hereda esta restricción; documentar en runbook.
- **Pre-commit hooks NO opcionales**: si añadir cf-bootstrap.sh u otros scripts dispara fallos de shellcheck (si está activo), corregir antes de commit, NO usar `--no-verify`.
- **Capa de compatibilidad `OEE_*` → `NEXO_*`** (Phase 1): no aplica a `CLOUDFLARE_TUNNEL_TOKEN` (variable nueva, no migración).

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework Python | pytest 8.x (ya instalado, configurado en `pyproject.toml`) |
| Framework infra | bash + curl + openssl + dig (mismo stack que Phase 6 `deploy_smoke.sh`) |
| Config file | `pyproject.toml` (pytest + coverage); `tests/infra/deploy_smoke.sh` standalone bash |
| Quick run command (Python) | `make test` (`pytest -q --cov=api --cov=nexo --cov-fail-under=60`) |
| Quick run command (infra LAN) | `bash tests/infra/deploy_smoke.sh` (requires server access + sudo) |
| Quick run command (infra CF) | `bash tests/infra/cloudflare_smoke.sh` (requires external network access — 4G tethering) |
| Full suite command | `make test && bash tests/infra/deploy_smoke.sh && bash tests/infra/cloudflare_smoke.sh` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CLOUD-01 | Domain `nexo.app` resuelve por DNS público | Manual (one-off post-purchase) | `dig +short nexo.app` | N/A — verifica una sola vez tras la compra |
| CLOUD-02 | Cuenta CF + 2FA + recovery codes | Manual + RUNBOOK checklist | N/A | N/A |
| CLOUD-03 | 5 security headers en respuesta HTTP | Integration smoke (Wave 0 + CI) | `curl -sI http://localhost:8001/ \| grep -iE 'hsts\|x-frame\|x-content-type\|referrer-policy\|permissions-policy' \| wc -l` ≥ 5 | ❌ Wave 0: NEW `tests/infra/test_caddyfile_security_headers.py` (parsea Caddyfile.prod + valida 5 directives) |
| CLOUD-04 | `cloudflared` container running + healthy | infra smoke (DEPLOY-11) | `bash tests/infra/deploy_smoke.sh` after `make cf-up` | ✅ extender `deploy_smoke.sh` (líneas 86-90) |
| CLOUD-05 | `CLOUDFLARE_TUNNEL_TOKEN` en `.env.example` | unit | NEW: `tests/infra/test_env_example_has_cf_token.py` | ❌ Wave 0 |
| CLOUD-06 | Access App + email allowlist + OTP | Manual external smoke (CF-03/CF-04) | `bash tests/infra/cloudflare_smoke.sh` | ❌ Wave 0: NEW `cloudflare_smoke.sh` |
| CLOUD-07 | Custom denial page (BLOCKER-01 pendiente decisión) | Manual external smoke + unit | curl a la URL post-denial; valida render Nexo branding | ❌ Wave 0: depende de decisión BLOCKER-01 |
| CLOUD-08 | `deploy_smoke.sh` 11 checks (no 8) | Regression test sobre el script | NEW `tests/infra/test_deploy_smoke_has_11_checks.py` (grep `[DEPLOY-` count) | ❌ Wave 0 |
| CLOUD-09 | `cloudflare_smoke.sh` 6 checks | unit (regression test) + manual (real run) | NEW `tests/infra/test_cloudflare_smoke_has_6_checks.py` | ❌ Wave 0 |
| CLOUD-10 | Makefile `cf-*` targets | unit | NEW `tests/infra/test_makefile_cf_targets.py` (grep `^cf-up\|^cf-down\|...`) | ❌ Wave 0 |
| CLOUD-11 | `docs/CLOUDFLARE_DEPLOY.md` + ADR + CLAUDE.md update | unit (presence + section structure) | NEW `tests/infra/test_cloudflare_docs_present.py` | ❌ Wave 0 |
| CLOUD-12 | LAN fallback intacto (smoke 11/11 con cloudflared running Y stopped) | Manual external (post-deploy) | `make cf-up && bash tests/infra/deploy_smoke.sh && make cf-down && bash tests/infra/deploy_smoke.sh` | ✅ scripts ya existirán post-Wave 0 |

### Sampling Rate

- **Per task commit:** `make test` (60s) + `make lint` (20s) — gates Python ya en CI (Phase 7)
- **Per wave merge:** `make test` + parsing tests `tests/infra/test_*.py` (los `_smoke.sh` requieren entorno real, no en CI)
- **Phase gate (manual antes de `/gsd-verify-work`):** ssh al servidor → `make cf-up && make cf-status && bash tests/infra/deploy_smoke.sh && make cf-down && bash tests/infra/deploy_smoke.sh && make cf-up`; desde 4G → `bash tests/infra/cloudflare_smoke.sh`

### Wave 0 Gaps

- [ ] `tests/infra/test_caddyfile_security_headers.py` — parsea `caddy/Caddyfile.prod` y valida los 5 directives (HSTS, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, Permissions-Policy) presentes en bloque `header { ... }` o snippet `(security_headers)` (CLOUD-03)
- [ ] `tests/infra/test_env_example_has_cf_token.py` — valida que `.env.example` y `.env.prod.example` contienen línea `CLOUDFLARE_TUNNEL_TOKEN=` con comentario procedencia (CLOUD-05)
- [ ] `tests/infra/test_deploy_smoke_has_11_checks.py` — extender existente `tests/infra/test_deploy_lan_doc.py` (24 tests Phase 6) para añadir 3 nuevos: validar que `deploy_smoke.sh` tiene 11 `[DEPLOY-*]` tags, no 8 (CLOUD-08)
- [ ] `tests/infra/test_cloudflare_smoke_has_6_checks.py` — valida que `cloudflare_smoke.sh` existe + tiene 6 `[CF-0X]` tags + es ejecutable (`stat -c '%a'` ends in `5` o `7`) (CLOUD-09)
- [ ] `tests/infra/test_makefile_cf_targets.py` — valida que Makefile tiene 5 targets cf-up/cf-down/cf-status/cf-logs/cf-smoke-external + cada uno con docstring `## ...` (CLOUD-10)
- [ ] `tests/infra/test_cloudflare_docs_present.py` — valida `docs/CLOUDFLARE_DEPLOY.md` exists + tiene N secciones (mirror de `test_deploy_lan_doc.py`); valida ADR-001 status cambia de "Proposed" a "Accepted" (sólo en plan 09-03 final); valida CLAUDE.md menciona "Cloudflare Tunnel" en sección "Despliegue" (CLOUD-11)
- [ ] `tests/infra/test_cloudflared_compose_service.py` — valida que `docker-compose.prod.yml` tiene servicio `cloudflared` con image `cloudflare/cloudflared`, `command` empieza con `tunnel`, `environment` mapea `TUNNEL_TOKEN`, `depends_on caddy` con `service_healthy`, healthcheck con `cloudflared tunnel ready`, `restart: unless-stopped` (CLOUD-04)
- [ ] **No new Python prod code → no new Python unit tests in `tests/api/` o `tests/nexo/`** — Phase 9 es 100% infra/config/docs. Coverage gate sigue cubierto por código existente.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | yes | OTP por email (Cloudflare) + argon2id (Nexo Phase 2) — capas independientes |
| V3 Session Management | yes | Cookie `CF_Authorization` 24h (Cloudflare Access) + cookie sesión Nexo HttpOnly+Secure (Phase 2 IDENT-04) |
| V4 Access Control | yes | RBAC interna Nexo (Phase 5 `can()` + 11 botones gateados + 8 GETs guarded) preservada; allowlist Access es additional |
| V5 Input Validation | yes (preservation) | Phase 2/4/5 Pydantic + sanitization; sin cambios |
| V6 Cryptography | no (gestionada por terceros) | TLS público termina en Cloudflare (TLS 1.3); cert real Google Trust Services o LE; HSTS preload obligatorio del TLD `.app` |
| V7 Error Handling and Logging | yes | Phase 1 NAMING-07 (`global_exception_handler` sin traceback) + Phase 2 IDENT-06 (audit_log append-only) preservados |
| V8 Data Protection | yes (con ADR-001 trade-off explícito) | Cloudflare termina TLS y técnicamente puede ver tráfico descifrado; aceptado para datos OEE industriales (ADR-001 §Negativas) |
| V9 Communications | yes | TLS 1.3 entre browser ↔ CF edge; tramo CF edge ↔ cloudflared encriptado (QUIC/HTTP2); tramo cloudflared ↔ caddy interno HTTPS con cert local (sin verify, red Docker no expuesta) |
| V10 Malicious Code | n/a | Sin nuevo código ejecutable |
| V11 Business Logic | n/a | Sin cambios |
| V12 Files and Resources | n/a | Sin cambios |
| V13 API and Web Service | yes | Headers de seguridad en Caddy aplican a `/api/*` también |
| V14 Configuration | yes | `TUNNEL_TOKEN` perms 600; `.gitignore` cubre `.env`; `gitleaks` (Phase 1 NAMING-09) en CI detectaría leak accidental |

### Known Threat Patterns for {Cloudflare Tunnel + Caddy + FastAPI stack}

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Token CF leaked vía `docker inspect` o git commit accidental | Information Disclosure | Pasar token como env var (no CLI arg, Pitfall 5); `.gitignore` cubre `.env`; gitleaks en CI; rotación 1-click desde dashboard CF |
| Brute force OTP por email no autorizado | Spoofing | Cloudflare diseñó behavior: emails fuera de allowlist NO reciben código pero ven misma página → atacante no puede enumerar emails válidos [VERIFIED: developers.cloudflare.com/cloudflare-one/integrations/identity-providers/one-time-pin/] |
| Bypass de Access — request directo al servidor con Host=nexo.app | Spoofing/Tampering | Imposible — el servidor no tiene puerto 443 abierto a internet. Sólo cloudflared puede entregar tráfico a Caddy. |
| Clickjacking de `nexo.app` desde otra origin | Tampering | `X-Frame-Options: DENY` + `Content-Security-Policy: frame-ancestors 'none'` (CSP fuera de scope Phase 9, X-Frame-Options DENY en D-13 cubre) |
| MIME sniff attack (browser interpreta JSON como HTML) | Tampering | `X-Content-Type-Options: nosniff` (D-13) |
| Downgrade attack HTTPS → HTTP | Tampering | HSTS preload (D-13) + TLD `.app` HSTS preload obligatorio del navegador |
| Referer leak a tracker externo | Information Disclosure | `Referrer-Policy: strict-origin-when-cross-origin` (D-13) |
| Browser API abuse (geolocation/camera/mic) | Tampering | `Permissions-Policy: geolocation=(), microphone=(), camera=()` (D-13) |
| Cloudflare ve tráfico descifrado | Information Disclosure | Aceptado en ADR-001 §Negativas para datos OEE industriales internos. Si se manejan datos significativamente más sensibles (RRHH/finanzas), reevaluar con ADR-002. |
| Replay de cookie `CF_Authorization` | Session Hijacking | Cookie HttpOnly+Secure+SameSite (Cloudflare default); 24h TTL; `/cdn-cgi/access/logout` para revocación inmediata (D-18) |
| DDoS o scan automático | DoS | Cloudflare DDoS protection incluido en Free; servidor sin puertos abiertos hace imposible scan directo |
| OTP email no llega (filtros antispam) | DoS / Availability | R-07 — documentar en RUNBOOK que sender es `noreply@notify.cloudflare.com`, whitelist requerida en filtros corporativos |

### Mozilla Observatory Verification (D-13 + Success Criterion 5)

Score baseline 100. Modificadores con los 5 headers de D-13 [VERIFIED: github.com/mozilla/http-observatory/blob/main/httpobs/docs/scoring.md]:

| Header | Status with D-13 | Modifier |
|--------|------------------|----------|
| Strict-Transport-Security (max-age 1yr + preload) | Present (≥6 months) | 0 |
| X-Frame-Options DENY | Implemented | 0 |
| X-Content-Type-Options nosniff | Present | 0 |
| Referrer-Policy strict-origin-when-cross-origin | Set to safe value | **+5 (bonus)** |
| Permissions-Policy (camera/mic/geo disabled) | Present | 0 (no penalty for missing; bonus only for specific features tested) |
| Content-Security-Policy | **NOT IMPLEMENTED** | **-25 (penalty)** |

**Score esperado:** 100 + 5 - 25 = **80** = **B+ (range 80-84)** ✅

**Conclusión:** D-13 + sin CSP = exactamente B+ (borderline B). Si quisiéramos A- o superior, añadir CSP es la única vía. Mark-III dejó CSP fuera de scope explícitamente — **B+ es alcanzable y suficiente para Success Criterion 5**.

**Warning:** Si Cloudflare añade headers automáticamente (e.g., `cf-ray`, `Server: cloudflare`) que tengan valores penalizadores (improbable), el score real podría diferir. Validar empíricamente post-deploy con [observatory.mozilla.org](https://observatory.mozilla.org) y documentar el resultado en VERIFICATION.md.

## Environment Availability

> Phase 9 es ejecución mixta: el repo (Linux/WSL dev) sólo necesita docker compose validation; el deploy real (servidor Ubuntu LAN) tiene los pre-requisitos de Phase 6 + nuevos.

| Dependency | Required By | Available (dev) | Available (server prod) | Version | Fallback |
|------------|------------|-----------------|------------------------|---------|----------|
| docker compose v2.24+ | docker-compose.prod.yml `!reset` tag (Phase 6) | ✓ (asumido) | ✓ (Phase 6 prerequisite) | 2.24+ | — |
| Acceso SSH al servidor | Activación 09-03 | ✓ (operador tiene) | n/a | — | — |
| Cuenta Cloudflare | D-05 | n/a | n/a (gestionado externamente) | — | Crear en 09-03 |
| Dominio `nexo.app` (o fallback) | D-01/D-02 | n/a | n/a (gestionado externamente) | — | 4 fallbacks documentados |
| Conexión saliente HTTPS desde el servidor a `*.cloudflare.com` | cloudflared tunnel | ? (asumido yes — Phase 6 servidor accede a internet salient para `apt update`, `docker pull`) | ? a confirmar | — | Si falla: revisar firewall corporativo egress |
| Conexión 4G / WiFi externa para smoke | CF-01..CF-05 | ✓ (operador tethering móvil) | n/a | — | — |
| `dig` + `openssl` + `curl` en máquina de smoke | cloudflare_smoke.sh | ✓ (Linux/macOS estándar) | ✓ | — | — |
| `cloudflared` CLI standalone | Healthcheck via `tunnel ready` | usado dentro del container, no requiere instalación host | usado dentro del container | image latest | — |

**Missing dependencies with no fallback:**
- Ninguno bloqueante en Wave 0. La activación real (09-03) depende de cuenta CF + dominio comprado, ambos gestionables el día del deploy.

**Missing dependencies with fallback:**
- `nexo.app` cogido → 4 fallbacks pre-aprobados (D-02)
- Conexión saliente bloqueada por firewall corporativo → revisar y abrir egress a `*.cloudflare.com:443` (highly improbable en LAN ECS estándar)

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | El `tls internal` cert que Caddy emite para `caddy:443` (Docker hostname) es aceptado por cloudflared con `noTLSVerify: true` sin warnings que rompan el túnel | D-11 / Pattern 1 | Bajo — `noTLSVerify` es flag explícito; Cloudflare docs confirman uso para self-signed |
| A2 | El bloque `:443` catch-all del `Caddyfile.prod` actual recibe correctamente el request del túnel (cloudflared envía SNI=`caddy:443` interno + Host=`nexo.app`) y responde 200 a `/api/health` sin requerir bloque explícito `nexo.app` | D-20 / CORRECTION-02 | Medio — Caddy v2.11 default Host rewrite añade incertidumbre; verificar empíricamente en CF-04 del smoke; mitigación documentada (añadir bloque `nexo.app` si 421) |
| A3 | El TLD `.app` está disponible en Cloudflare Registrar al momento de la compra (algunas TLD requieren plan paid) | D-03 | Bajo — `.app` es estándar y CF Registrar lo ha vendido desde 2018 |
| A4 | El servidor Ubuntu LAN ECS x300 tiene conectividad saliente a `*.cloudflare.com:443` sin filtrado corporativo | Environment Availability | Medio — depende de la red LAN; si bloqueada, abrir egress es trabajo de IT |
| A5 | El operador (Erik) tiene capacidad administrativa para ejecutar Cloudflare Registrar purchase (~14 €/yr personal expense aceptable según D-05) | D-05 | Bajo — confirmado en CONTEXT.md |
| A6 | Los gateways de email corporativo de ECS Mobility no bloquean `noreply@notify.cloudflare.com` | R-07 | Medio — primer login del primer usuario detecta esto; mitigación documentada en RUNBOOK |
| A7 | La vulnerabilidad teórica "Cloudflare termina TLS y puede ver tráfico" es aceptable para los datos manejados por Nexo (OEE industrial, datos de turnos, planificación interna) y NO se manejan datos significativamente más sensibles vía Nexo (RRHH/finanzas/PII regulada) | ADR-001 §Negativas | Alto si falso — reevaluar ADR-001 antes de deploy si Phase 4/5 expanded el dataset a algo más sensible. Verificar con operador en `/gsd-discuss-phase` rerun si dudas |
| A8 | Las 4 conexiones tunneling redundantes default de cloudflared son suficientes para el patrón de uso de Nexo (≤50 users concurrent, mayoría reads OEE no real-time) | Architecture | Bajo — Nexo no es high-throughput; CF Tunnel diseñado para self-hosted apps medianas |
| A9 | El operador (o admin de respaldo en bus factor 2) tiene capacidad de identity-provide en el dashboard CF para gestionar la allowlist sin contexto adicional | D-15 / D-06 | Bajo — UI dashboard CF es self-explanatory; CLOUDFLARE_DEPLOY.md documenta paso a paso |
| A10 | `cloudflared tunnel ready` exit code 0 es estable como signal de healthcheck (vs experimental subcommand que pueda romper en futuras versiones de cloudflared) | Pattern 1 / Pitfall 4 | Medio — el subcommand existe desde PR #1135 (mid-2024) y es ahora estándar; pin a tag específico (`2025.10.0`) si breaking change preocupa |
| A11 | BLOCKER-01 se resuelve con Opción B (Caddy `/access-denied` público) sin requerir paid plan CF | Pitfall 6 | Alto si la Access App `nexo.app/*` no permite excluir `/access-denied` del scope sin pagar — verificar en CLOUDFLARE_DEPLOY.md durante 09-02 que la sintaxis "Allow Everyone" sub-app es free |

**11 assumptions** — `[ASSUMED]` tags requieren confirmación del operador o validación empírica antes de execution. **A2, A7, A11** son los más arriesgados; los demás son operacionales y se descubren durante deploy.

## Open Questions

1. **BLOCKER-01: ¿Cómo se sirve la denial page sin pagar Access Pay-as-you-go? (D-17 / CLOUD-07)**
   - What we know: docs CF confirman que Custom Block Page Templates requieren paid plan. Free plan ofrece (a) default page o (b) Redirect URL.
   - What's unclear: cuál de las 3 alternativas (default/Redirect a Caddy/Cloudflare Pages estático) prefiere el operador.
   - Recommendation: planner escoge Opción B (Redirect a `https://nexo.app/access-denied` servido por Caddy con bloque público sin Access). Si alta complejidad técnica, recurrir al operador en `/gsd-discuss-phase` rerun antes de plan 09-02.

2. **¿Validación empírica del SNI mismatch (D-20 / R-05) puede hacerse antes del deploy real?**
   - What we know: Caddy default no fuerza strict_sni_host; el bloque `:443` catch-all captura el tráfico del túnel.
   - What's unclear: si la combinación específica (cloudflared 2025.x + Caddy 2.x + Caddyfile.prod actual) tiene comportamiento diferente.
   - Recommendation: dry run local con `cloudflared tunnel --hello-world` apuntando a Caddy de dev no es viable (requiere dominio real). Aceptar riesgo medio; el smoke CF-04 lo detecta inmediatamente; mitigación documentada (añadir bloque explícito `nexo.app` al Caddyfile).

3. **¿`make cf-up` debe arrancar SOLO cloudflared o todo el stack prod?**
   - What we know: Sketch dice "arranca solo cloudflared" (D-24 rollback semantics).
   - What's unclear: si el stack prod no está arriba, `make cf-up` falla por `depends_on caddy`. ¿Auto-arrancar todo, o fallar con mensaje claro?
   - Recommendation: fallar con mensaje claro ("Error: stack prod no arriba. Run `make prod-up` first."). Mantiene separation of concerns con prod-up de Phase 6.

4. **¿Pin `cloudflare/cloudflared` a tag específico o usar `:latest`?**
   - What we know: D-07 dice `:latest`. Imagen latest stable a 2026-04-25 es `2025.10.0`.
   - What's unclear: trade-off reproducibilidad vs auto-updates de seguridad de cloudflared.
   - Recommendation: usar `:latest` por simplicidad (cloudflared es trusted vendor, breaking changes raros), pero documentar en RUNBOOK que si breaking change ocurre, pin a último tag conocido bueno.

5. **¿La Access App con scope `nexo.app/*` se puede subdividir gratis para excluir `/access-denied` (necesario si Opción B de BLOCKER-01)?**
   - What we know: CF docs no son explícitas sobre límites de Access Apps en Free plan más allá de "≤50 users".
   - What's unclear: si crear una segunda Access App con scope `nexo.app/access-denied` y policy "Allow Everyone" cuenta contra algún límite de Free.
   - Recommendation: pre-validar el día del deploy creando Access App `denied` antes de finalizar; si falla, fallback a Cloudflare Pages estático (Opción C).

## Sources

### Primary (HIGH confidence)

- **[Cloudflare Tunnel Run Parameters](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/configure-tunnels/run-parameters/)** — token via `--token` CLI o `TUNNEL_TOKEN` env; `--no-autoupdate` recomendado para Docker; `--token-file` desde v2025.4.0
- **[Cloudflare One Time Pin IdP](https://developers.cloudflare.com/cloudflare-one/integrations/identity-providers/one-time-pin/)** — OTP IdP behavior, blocked emails no reciben código, login page genérica
- **[Cloudflare Access Block Page Custom](https://developers.cloudflare.com/cloudflare-one/reusable-components/custom-pages/access-block-page/)** — Custom Page Template **only available on Pay-as-you-go and Enterprise plans** (BLOCKER-01)
- **[Cloudflare Access Session Management](https://developers.cloudflare.com/cloudflare-one/access-controls/access-settings/session-management/)** — session duration default 24h, range immediate→1month, logout endpoint `/cdn-cgi/access/logout`
- **[Cloudflare Tunnel Metrics](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/monitor-tunnels/metrics/)** — `--metrics` flag default range 20241-20245; `0.0.0.0:PORT/metrics` en Docker
- **[Caddy Caddyfile header directive](https://caddyserver.com/docs/caddyfile/directives/header)** — sintaxis canónica `header { ... }` block; defer auto-enabled para deletions/defaults
- **[Caddy Global Options](https://caddyserver.com/docs/caddyfile/options)** — `strict_sni_host` deshabilitado por default salvo client_auth
- **[Caddy reverse_proxy](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy)** — Host header behavior change v2.11.0
- **[Mozilla HTTP Observatory Scoring](https://github.com/mozilla/http-observatory/blob/main/httpobs/docs/scoring.md)** — modifiers: HSTS 0/-20, XFO 0/-20, XCTO 0/-5, Referrer-Policy +5/0, CSP +5..+10/-25; B+ range 80-84
- **[GitHub cloudflare/cloudflared issue #7527 (Caddy 2.11 Host header)](https://github.com/caddyserver/caddy/issues/7527)** — breaking change Dic 2024
- **[GitHub cloudflare/cloudflared PR #1135 (tunnel ready)](https://github.com/cloudflare/cloudflared/pull/1135)** — readiness command añadido
- **[Docker Hub cloudflare/cloudflared](https://hub.docker.com/r/cloudflare/cloudflared)** — imagen oficial, latest tag = 2025.10.0 al 2026-04-25, alpine 27.4MB

### Secondary (MEDIUM confidence)

- **[Cloudflare Zero Trust Plans](https://www.cloudflare.com/plans/zero-trust-services/)** — Free plan 50 users cap confirmado
- **[Cloudflare One Account Limits](https://developers.cloudflare.com/cloudflare-one/account-limits/)** — applications 500, identity providers 50, rules per app 1000 (limits no Free-specific en doc)
- **[Cloudflare Tunnel Health Checks (community)](https://jacobhands.com/blog/cloudflare-tunnel-health-checks-using-docker-compose/)** — pattern docker-compose healthcheck con `tunnel ready`
- **[Caddy Hardening community guide](https://hackviser.com/tactics/hardening/caddy)** — security headers patterns

### Tertiary / Repo Inspection (HIGH for repo-specific patterns)

- `tests/infra/deploy_smoke.sh` — patrón `[DEPLOY-XX] OK\|FAIL`, `set -uo pipefail`, exit code = FAILS count
- `Makefile` líneas 142-173 — patrón `prod-*` 7 targets con docstring `## ...`
- `docker-compose.prod.yml` — `!reset []` y `!override` tags; healthcheck pattern para web/caddy
- `caddy/Caddyfile.prod` — sintaxis bloques sitio + catch-all `:443`; comentarios estilo dev/prod
- `docs/DEPLOY_LAN.md` — 16 secciones, 740 líneas, español sin emojis, placeholders `<IP_NEXO>` literales
- `docs/decisions/ADR-001-cloudflare-tunnel.md` — formato Nygard ya redactado, status "Proposed" → cambiar a "Accepted" en plan 09-03
- `CLAUDE.md` — sección "Despliegue" (línea ~26) y "Qué NO hacer" (línea ~152) target de update D-28
- `CHANGELOG.md` — Keep a Changelog 1.1.0; entry "[Phase 9]" añadible

## Metadata

**Confidence breakdown:**

- Standard stack: **HIGH** — Cloudflare official docs + Docker Hub verified; Caddy patterns canonical
- Architecture (D-07..D-12 + D-14..D-18): **HIGH** — modo token-based remote-managed es state of the art Cloudflare; OTP IdP free; Access self-hosted patrón estándar
- Headers + Mozilla Observatory (D-13): **HIGH** — modifiers exactos de Mozilla scoring docs; B+ matemáticamente verificado
- Caddy SNI/Host header (D-20 / Pitfalls 2-3): **MEDIUM** — Caddy v2.11 cambio de comportamiento documentado; comportamiento exacto en este stack requiere verificación empírica en CF-04
- BLOCKER-01 (D-17): **HIGH (factual claim)** — Custom Block Pages Free vs Paid es claim verificable; **el plan de mitigación (Opción B vs C)** es discrecional → pendiente decisión planner/operador
- CORRECTION-01 (TUNNEL_TOKEN env var): **HIGH** — Cloudflare docs explícitas
- CORRECTION-02 (strict_sni_host default off): **HIGH** — Caddy docs explícitas
- Smoke patterns: **HIGH** — repo inspection directo
- Project constraints from CLAUDE.md: **HIGH** — leído directamente

**Research date:** 2026-04-25
**Valid until:** 2026-05-25 (30 días para Cloudflare/Caddy stable claims; Cloudflare Free plan policies pueden cambiar — re-validar ≤50 users cap antes de deploy)

**Phase 9 readiness:** ✅ Ready for `/gsd-plan-phase 9` — con la caveat que **BLOCKER-01 (D-17 custom denial page Free plan)** debe resolverse antes o durante 09-02 PLAN.md drafting. Recomendación: planner escoge Opción B y procede; si rechaza, escalar a `/gsd-discuss-phase 9` rerun para decisión del operador.
