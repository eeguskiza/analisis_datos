# Phase 9: Cloudflare Tunnel + Public Access — Pattern Map

**Mapped:** 2026-04-25
**Files analyzed:** 16 (7 modified + 9 created)
**Analogs found:** 16 / 16 (100% — every file has a strong intra-repo analog)

## File Classification

| File | New/Mod | Role | Data Flow | Closest Analog | Match |
|------|---------|------|-----------|----------------|-------|
| `caddy/Caddyfile.prod` | modify | config (reverse proxy) | request-response | itself (lines 8-16) | exact |
| `docker-compose.prod.yml` | modify | config (orchestration) | event-driven | `caddy:` service block (lines 43-58) | exact |
| `.env.example` | modify | config (template) | static | existing `NEXO_*` lines | exact |
| `.env.prod.example` | modify | config (template) | static | existing `NEXO_*` lines | exact |
| `tests/infra/deploy_smoke.sh` | modify | test (bash smoke) | request-response | itself (lines 49-91) | exact |
| `Makefile` | modify | tooling | batch | `prod-up/down/logs/status/health` (lines 148-167) | exact |
| `CLAUDE.md` | modify | docs | static | itself ("Despliegue", "Qué NO hacer") | exact |
| `CHANGELOG.md` | modify | docs | static | Phase 7 entry (lines 59-67) | exact |
| `tests/infra/cloudflare_smoke.sh` | create | test (bash smoke) | request-response | `tests/infra/deploy_smoke.sh` | exact |
| `static/cloudflare-denial.html` | create | static asset | static | `templates/login.html` | role-match |
| `scripts/cloudflare-bootstrap.sh` | create | script (bash ops) | batch | `scripts/deploy.sh` + `scripts/backup_nightly.sh` | exact |
| `docs/CLOUDFLARE_DEPLOY.md` | create | docs (runbook) | static | `docs/DEPLOY_LAN.md` (740 lines, 16 sections) | exact |
| `tests/infra/test_caddyfile_security_headers.py` | create | test (pytest) | static | `tests/infra/test_caddyfile_prod.py` | exact |
| `tests/infra/test_deploy_smoke_has_11_checks.py` | create | test (pytest) | static | `tests/infra/test_deploy_lan_doc.py` (smoke section) | exact |
| `tests/infra/test_cloudflared_compose_service.py` | create | test (pytest) | static | `tests/infra/test_compose_override.py` | exact |
| `tests/infra/test_env_example_has_cf_token.py` | create | test (pytest) | static | `tests/infra/test_env_prod_example.py` | exact |
| `tests/infra/test_cloudflare_smoke_has_6_checks.py` | create | test (pytest) | static | `tests/infra/test_deploy_lan_doc.py` (smoke section) | exact |
| `tests/infra/test_makefile_cf_targets.py` | create | test (pytest) | static | `tests/infra/test_makefile_devex.py` | exact |
| `tests/infra/test_cloudflare_docs_present.py` | create | test (pytest) | static | `tests/infra/test_deploy_lan_doc.py` (doc section) | exact |

---

## Pattern Assignments

### `caddy/Caddyfile.prod` (modify)

**Closest analog:** itself, current shape (Phase 6).

**Pattern excerpt (lines 8-24, current state):**

```caddyfile
nexo.ecsmobility.local {
    tls internal
    reverse_proxy web:8000

    log {
        output stdout
        format json
    }
}

# Fallback IP-directa
:443 {
    tls internal
    reverse_proxy web:8000
}
```

**Adaptation:** add a top-level `(security_headers) { header ... }` snippet, then `import security_headers` inside both `nexo.ecsmobility.local { }` and `:443 { }` blocks. Headers per D-13: `Strict-Transport-Security "max-age=31536000; includeSubDomains; preload"`, `X-Frame-Options "DENY"`, `X-Content-Type-Options "nosniff"`, `Referrer-Policy "strict-origin-when-cross-origin"`, `Permissions-Policy "geolocation=(), microphone=(), camera=()"`. Add a third top-level block per D-17/BLOCKER-01:

```caddyfile
http://nexo.app:80, https://nexo.app:443 {
    handle /access-denied {
        root * /srv/static
        rewrite * /cloudflare-denial.html
        file_server
    }
    handle {
        import security_headers
        reverse_proxy web:8000
    }
}
```

The `/access-denied` path is intentionally public (no auth, no Cloudflare gating) and bind-mounted into Caddy at `/srv/static` via the compose override (see next file).

---

### `docker-compose.prod.yml` (modify)

**Closest analog:** the existing `caddy:` service block.

**Pattern excerpt (lines 43-58):**

```yaml
  caddy:
    volumes: !override
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

**Adaptation:** (a) add `- ./static/cloudflare-denial.html:/srv/static/cloudflare-denial.html:ro` to `caddy.volumes`. (b) add a new top-level service `cloudflared`. Mirror the caddy shape (image + restart + healthcheck + depends_on + deploy.resources). Per D-07/D-08/D-09 (CORRECTION-01):

```yaml
  cloudflared:
    image: cloudflare/cloudflared:latest
    restart: unless-stopped
    command: tunnel --no-autoupdate run
    environment:
      - TUNNEL_TOKEN=${CLOUDFLARE_TUNNEL_TOKEN}
    depends_on:
      caddy:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "cloudflared", "tunnel", "info"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 256m
```

Critical: env var inside container is `TUNNEL_TOKEN` (upstream convention). Project naming `CLOUDFLARE_TUNNEL_TOKEN` is preserved in `.env` and mapped explicitly. No port publishing — the tunnel is outbound-only.

---

### `.env.example` AND `.env.prod.example` (modify)

**Closest analog:** existing `NEXO_*` blocks with `# comment` headers.

**Adaptation:** append new section. Use `<CHANGEME-...>` placeholder convention (see `tests/infra/test_env_prod_example.py` `test_env_prod_example_no_real_secrets`):

```bash
# ── Cloudflare Tunnel (Phase 9) ──────────────────────────────────────────────
# Token del tunnel remote-managed. Se obtiene desde:
#   Cloudflare Zero Trust -> Networks -> Tunnels -> nexo-prod -> Install connector
# Pega el string completo despues de `--token`. La imagen oficial cloudflared
# lee `TUNNEL_TOKEN`; el compose mapea CLOUDFLARE_TUNNEL_TOKEN -> TUNNEL_TOKEN.
# Rotacion: 1-click desde el dashboard. Ver docs/CLOUDFLARE_DEPLOY.md.
CLOUDFLARE_TUNNEL_TOKEN=<CHANGEME-cf-tunnel-token>
```

In `.env.example` (dev) the value can be empty string — the dev stack does NOT spin up `cloudflared`. In `.env.prod.example` it must use the `<CHANGEME-...>` placeholder pattern to satisfy the `_no_real_secrets` regression test.

---

### `tests/infra/deploy_smoke.sh` (modify)

**Closest analog:** itself.

**Pattern excerpt (lines 30-40, the `check` helper):**

```bash
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
```

**Adaptation:** append three new `check` calls before `echo "=== Fallos: ${FAILS} ==="`. Tags `DEPLOY-09`, `DEPLOY-10`, `DEPLOY-11` per D-21:

```bash
# DEPLOY-09: POST /api/login con creds dummy responde 401 (no 500)
check "DEPLOY-09" "POST /api/login con creds dummy responde 401" \
    "[ \"\$(curl -s -m 10 -k -o /dev/null -w '%{http_code}' -X POST -d 'email=nope@nope&password=nope' https://nexo.ecsmobility.local/api/login)\" = '401' ]"

# DEPLOY-10: GET /api/health con MES caido responde 200 con ok:false
check "DEPLOY-10" "GET /api/health responde 200 (incluso con MES caido)" \
    "[ \"\$(curl -s -m 10 -k -o /dev/null -w '%{http_code}' https://nexo.ecsmobility.local/api/health)\" = '200' ]"

# DEPLOY-11: cloudflared container running + tunnel info exit 0
check "DEPLOY-11" "cloudflared container running + tunnel info OK" \
    "${COMPOSE} exec -T cloudflared cloudflared tunnel info 2>/dev/null"
```

Reuse the existing `set -uo pipefail`, `COMPOSE`, `FAILS` machinery — do NOT introduce `set -e` (Phase 6 invariant: smoke counts failures and reports total).

---

### `Makefile` (modify)

**Closest analog:** the `prod-*` block (lines 148-167).

**Pattern excerpt (lines 148-167):**

```makefile
PROD_COMPOSE = docker compose -f docker-compose.yml -f docker-compose.prod.yml

prod-up: ## Arranca stack prod (base + override).
	$(PROD_COMPOSE) up -d --build

prod-down: ## Para stack prod (SIN -v — pgdata persiste)
	$(PROD_COMPOSE) down

prod-logs: ## Logs stack prod en tiempo real
	$(PROD_COMPOSE) logs -f

prod-status: ## Estado de containers prod
	$(PROD_COMPOSE) ps

prod-health: ## Health check via /api/health del stack prod
	@$(PROD_COMPOSE) exec -T web curl -fs http://localhost:8000/api/health | python3 -m json.tool
```

**Adaptation:** add 5 `cf-*` targets that operate ONLY on the `cloudflared` service, mirroring shape and `## help-text` convention. Append to top-line `.PHONY:` (per `test_phony_declares_new_targets` style):

```makefile
# ── Cloudflare Tunnel (Phase 9) ──────────────────────────────────────────────

cf-up: ## Arranca solo cloudflared (requiere CLOUDFLARE_TUNNEL_TOKEN en .env)
	$(PROD_COMPOSE) up -d cloudflared

cf-down: ## Para cloudflared (LAN fallback intacto)
	$(PROD_COMPOSE) stop cloudflared

cf-status: ## Estado del cloudflared + tunnel info
	$(PROD_COMPOSE) ps cloudflared
	@$(PROD_COMPOSE) exec -T cloudflared cloudflared tunnel info || true

cf-logs: ## Logs cloudflared en tiempo real
	$(PROD_COMPOSE) logs -f cloudflared

cf-smoke-external: ## Smoke externo (6 checks CF-*) ejecutado desde fuera de la LAN
	bash tests/infra/cloudflare_smoke.sh
```

Add `cf-up cf-down cf-status cf-logs cf-smoke-external` to the existing `.PHONY:` line (line 1).

---

### `CLAUDE.md` (modify)

**Closest analog:** itself, sections "Despliegue:" and "Qué NO hacer".

**Adaptation per D-28:**

- "Despliegue: LAN interna ECS (Ubuntu Server 24.04). Sin exposición a internet." → "Despliegue: LAN interna ECS + público vía Cloudflare Tunnel con allowlist Access (ver `docs/decisions/ADR-001-cloudflare-tunnel.md`)."
- Remove the bullet "No comprar dominio público ni exponer Nexo a internet" from "Qué NO hacer".
- Add bullet under "Qué NO hacer": "No exponer puertos del servidor a internet directamente — todo tráfico público pasa obligatoriamente por Cloudflare Tunnel saliente. UFW debe seguir cerrando 80/443 al WAN."
- Add row under "Targets Makefile para prod": `make cf-up`, `make cf-down`, `make cf-status`, `make cf-logs`, `make cf-smoke-external`.

---

### `CHANGELOG.md` (modify)

**Closest analog:** Phase 7 entry (lines 59-67).

**Pattern excerpt (lines 59-67):**

```markdown
- **Phase 7 (Sprint 6) — DevEx hardening:** Pre-commit con hooks
  `ruff-check` + `ruff-format` + `mypy` scoped a `api/`+`nexo/` (hooks en
  `.pre-commit-config.yaml`). Config consolidada en `pyproject.toml`
  ...
```

**Adaptation:** append new entry under `[Unreleased] / ### Added`:

```markdown
- **Phase 9 (Sprint 7) — Cloudflare Tunnel + Public Access:** Acceso público a
  Nexo desde cualquier dispositivo via `https://nexo.app` con allowlist email
  Cloudflare Access (OTP, cookie 24h, Free tier). Servicio `cloudflared`
  remote-managed en `docker-compose.prod.yml` (target `https://caddy:443`,
  TLS verify off, noTLSVerify documentado). 5 headers de seguridad en Caddy
  (HSTS, X-Frame-Options, X-Content-Type-Options, Referrer-Policy,
  Permissions-Policy) aplicados a ambas ramas LAN+CF — Mozilla Observatory >= B+.
  Denial page branded en `static/cloudflare-denial.html` servida públicamente
  via Caddy (Free tier respetado, BLOCKER-01 resuelto). Smoke LAN ampliado de
  8 a 11 checks; smoke externo `tests/infra/cloudflare_smoke.sh` con 6 checks
  CF-*. Makefile `cf-up/down/status/logs/smoke-external`. Runbook
  `docs/CLOUDFLARE_DEPLOY.md` + ADR-001 en `docs/decisions/`. **Fallback LAN
  preservado** (`docker compose stop cloudflared` revierte sin pérdida).
```

---

### `tests/infra/cloudflare_smoke.sh` (create)

**Closest analog:** `tests/infra/deploy_smoke.sh` — copy verbatim then adapt.

**Pattern excerpt (deploy_smoke.sh lines 1-50):** see file above; reuse the `check()` helper, `set -uo pipefail`, `FAILS` counter, `[CHECK-NN] OK|FAIL` format, `exit "${FAILS}"`.

**Adaptation per D-22:** 6 checks `[CF-01..CF-06]`. Important: this script runs OUTSIDE the LAN (4G tethering, peer network), so it must NOT require docker/compose nor sudo for CF-01..CF-05. Only CF-06 may use SSH or be marked manual:

```bash
# CF-01: nexo.app resuelve por DNS público
check "CF-01" "nexo.app resuelve via DNS publico" \
    "dig +short nexo.app | grep -qE '^[0-9]+\\.'"

# CF-02: cert real de CF (issuer Google Trust Services o similar, NO Caddy Local)
check "CF-02" "TLS cert no es Caddy Local Authority" \
    "! openssl s_client -connect nexo.app:443 -servername nexo.app </dev/null 2>/dev/null | grep -q 'Caddy Local Authority'"

# CF-03: con header CF-Access-Jwt-Assertion invalido -> 302 a login
check "CF-03" "Access rechaza JWT invalido con redirect" \
    "[ \"\$(curl -s -m 10 -o /dev/null -w '%{http_code}' -H 'CF-Access-Jwt-Assertion: invalid' https://nexo.app/)\" = '302' ]"

# CF-04: GET sin headers redirige a cloudflareaccess.com
check "CF-04" "GET sin auth redirige a cloudflareaccess.com" \
    "curl -s -m 10 -I https://nexo.app/ | grep -iE '^location:.*cloudflareaccess\\.com'"

# CF-05: 5 headers de seguridad presentes
check "CF-05" "HSTS + X-Frame-Options + X-Content-Type-Options presentes" \
    "curl -s -m 10 -I https://nexo.app/access-denied | grep -iE '^strict-transport-security' && \
     curl -s -m 10 -I https://nexo.app/access-denied | grep -iE '^x-frame-options' && \
     curl -s -m 10 -I https://nexo.app/access-denied | grep -iE '^x-content-type-options'"

# CF-06: tunnel info HEALTHY (requiere SSH al servidor; si no disponible: skip)
check "CF-06" "cloudflared tunnel HEALTHY (skip si no hay SSH)" \
    "ssh -o ConnectTimeout=5 nexo-server 'cd /opt/nexo && docker compose -f docker-compose.yml -f docker-compose.prod.yml exec -T cloudflared cloudflared tunnel info' 2>/dev/null | grep -q HEALTHY"
```

CF-05 hits `/access-denied` (public, no auth) so it doesn't need a CF Access JWT to verify headers reach the response.

---

### `static/cloudflare-denial.html` (create)

**Closest analog:** `templates/login.html` (lines 1-50 head + body header).

**Pattern excerpt (login.html lines 1-27):**

```html
<!DOCTYPE html>
<html lang="es" class="h-full">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Entrar · {{ app_name }}</title>
  <link rel="stylesheet" href="/static/css/app.css">
  <script src="/static/js/tailwind.config.js"></script>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body ...>
  <div class="w-full max-w-7xl mx-auto ...">
    <header class="pt-1">
      <img src="{{ logo_path }}" alt="{{ app_name }}" ...>
      <img src="{{ ecs_logo_path }}" alt="{{ company_name }}" ...>
```

**Adaptation:** static HTML (NO Jinja — Caddy serves it raw). Hardcode logo paths (`/static/img/nexo-logo.svg`, `/static/img/ecs-logo.svg` — confirm against `docs/BRANDING.md`). Hardcode "Nexo" / "ECS Mobility". Replace login form with central card containing "Acceso restringido. Si crees que deberías tener acceso a Nexo, contacta con Erik Eguskiza – e.eguskiza@ecsmobility.com" (per D-17). Reuse `tokens.css` color classes (`text-heading`, `text-muted`, `bg-surface-base/88`, `border-surface-0/50`) for branding consistency. The file lives under `static/` (NOT `templates/`) because Caddy file_server serves it directly; the bind mount in `docker-compose.prod.yml` exposes it inside Caddy at `/srv/static/cloudflare-denial.html`.

---

### `scripts/cloudflare-bootstrap.sh` (create)

**Closest analog:** `scripts/deploy.sh` (lines 1-71) for log/fail helpers; `scripts/backup_nightly.sh` for shape and idempotence.

**Pattern excerpt (deploy.sh lines 13-24):**

```bash
set -euo pipefail
IFS=$'\n\t'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

COMPOSE="docker compose -f docker-compose.yml -f docker-compose.prod.yml"
LOG_PREFIX="[bootstrap $(date -u +%FT%TZ)]"

log() { echo "${LOG_PREFIX} $*"; }
fail() { echo "${LOG_PREFIX} ERROR: $*" >&2; exit 1; }
```

**Adaptation:** validation script (NOT a deploy). 4 steps: (1) `[ -f /opt/nexo/.env ]` and grep `^CLOUDFLARE_TUNNEL_TOKEN=` non-empty. (2) `$COMPOSE up -d cloudflared`. (3) Poll `$COMPOSE exec -T cloudflared cloudflared tunnel info` up to 30s until output contains `HEALTHY`. (4) Run `bash tests/infra/cloudflare_smoke.sh` against the local server (CF-01..CF-05, skip CF-06 since we ARE the server). Each step uses `log "step N/4: ..."` and `fail "..."` on error. Permissions: `chmod +x scripts/cloudflare-bootstrap.sh` (test_smoke_sh_is_executable analog applies).

---

### `docs/CLOUDFLARE_DEPLOY.md` (create)

**Closest analog:** `docs/DEPLOY_LAN.md` (740 lines, 16 numbered sections).

**Section structure of analog:**

```
## 1. Vista general
## 2. Prerrequisitos
## 3. Instalar Docker CE + compose plugin
## 4. Clonar el repo y configurar .env
## 5. Configurar el firewall (ufw)
## 6. Arrancar el stack Nexo
## 7. Extraer y distribuir la root CA de Caddy
## 8. Configurar hosts-file en cada equipo LAN
## 9. Avisos criticos (Landmines)
## 10. Configurar el cron de backup nightly
## 11. Validacion post-deploy (8 checks DEPLOY-*)
## 12. Operaciones rutinarias
## 13. Restaurar desde backup
## 14. Recuperacion desde cero (RTO 1-2h)
## 15. Mejoras futuras (deferred)
## 16. Glosario y referencias
## Apendice A: Comandos de un vistazo
## Apendice B: Secuencia mental
```

**Adaptation:** mirror this structure for Cloudflare. Suggested 14 sections + 2 apendices (~600-800 lines):

```
## 1. Vista general (LAN + CF coexistence diagram)
## 2. Prerrequisitos (Phase 6 deployed, server reachable)
## 3. Crear cuenta Cloudflare + 2FA + recovery codes (D-05, D-06)
## 4. Comprar dominio en CF Registrar (D-01, D-02 fallbacks)
## 5. Crear Tunnel `nexo-prod` remote-managed (D-07, D-08)
## 6. Configurar public hostname `nexo.app -> https://caddy:443` con noTLSVerify (D-10, D-11)
## 7. Crear Access App + email allowlist + OTP IdP (D-14, D-15, D-16)
## 8. Configurar Redirect URL on denial -> `nexo.app/access-denied` (D-17/BLOCKER-01)
## 9. Pegar token en /opt/nexo/.env (perms 600, D-09)
## 10. `make cf-up` y validacion `tunnel info HEALTHY`
## 11. Smoke externo desde 4G (`make cf-smoke-external`, manual + scripted, D-23)
## 12. Avisos criticos (Landmines CF: token leak, SNI 421, OTP filtrado, CF outage -> LAN fallback)
## 13. Operaciones rutinarias (anadir email a allowlist, rotar token, logout endpoint /cdn-cgi/access/logout)
## 14. Rollback / desactivacion (D-24: `docker compose stop cloudflared` -> LAN intacta)
## Apendice A: Comandos de un vistazo
## Apendice B: Troubleshooting (421 Misdirected, OTP no llega, edge incidente)
```

Reuse the placeholder convention (`<EMAIL_OWNER>`, `<TUNNEL_TOKEN_REDACTED>`, `<DOMAIN>` if `nexo.app` taken). No emojis (regression test analog: `test_doc_no_emoji`). Mention `make cf-up`, `make cf-smoke-external`, `tests/infra/cloudflare_smoke.sh`, `docs/decisions/ADR-001-cloudflare-tunnel.md` so doc-link tests pass.

---

### Pytest test files (create) — shared base pattern

**Closest analog:** `tests/infra/test_caddyfile_prod.py` (43 lines, simplest read+assert pattern).

**Pattern excerpt (lines 1-26):**

```python
"""Validaciones estaticas de caddy/Caddyfile.prod (Phase 6 / Plan 06-01).

No requiere Docker ni servidor Ubuntu real. Lee el archivo y comprueba
presencia/ausencia de literales criticos.
"""
from pathlib import Path

CADDYFILE_PROD = Path(__file__).resolve().parents[2] / "caddy" / "Caddyfile.prod"


def test_caddyfile_prod_exists():
    assert CADDYFILE_PROD.exists(), f"No existe {CADDYFILE_PROD}"


def test_caddyfile_prod_contains_hostname():
    content = CADDYFILE_PROD.read_text(encoding="utf-8")
    assert "nexo.ecsmobility.local" in content, (
        "Caddyfile.prod debe declarar el bloque ... (D-01, D-06)."
    )
```

**Per-file adaptations:**

- **`test_caddyfile_security_headers.py`** — same shape; assert each of the 5 headers is present in `Caddyfile.prod`: `Strict-Transport-Security`, `X-Frame-Options "DENY"`, `X-Content-Type-Options "nosniff"`, `Referrer-Policy`, `Permissions-Policy`. Plus `assert "/access-denied" in content` and `assert "max-age=31536000" in content`.

- **`test_deploy_smoke_has_11_checks.py`** — analog: `test_deploy_lan_doc.py` lines 201-213 (`test_smoke_sh_covers_deploy_requirements`). Adapt: assert `DEPLOY-01..DEPLOY-11` all appear in `tests/infra/deploy_smoke.sh`; assert `count of "[DEPLOY-" >= 11`.

- **`test_cloudflared_compose_service.py`** — analog: `test_compose_override.py` lines 1-110. Use `pyyaml` (already a dep — see test_compose_override pattern with `subprocess` + `json`). Read `docker-compose.prod.yml`, parse YAML, assert `services.cloudflared.image == "cloudflare/cloudflared:latest"`, `services.cloudflared.environment` contains `TUNNEL_TOKEN=${CLOUDFLARE_TUNNEL_TOKEN}`, `restart == "unless-stopped"`, `depends_on.caddy.condition == "service_healthy"`. CRITICAL: parse with `yaml.safe_load` only after stripping `!override` and `!reset` tags (use `yaml.add_constructor` or pre-process the text).

- **`test_env_example_has_cf_token.py`** — analog: `test_env_prod_example.py` lines 69-79 (`test_env_prod_example_has_all_required_nexo_vars`). Adapt: read `.env.example` AND `.env.prod.example`, regex `^CLOUDFLARE_TUNNEL_TOKEN=` in both. For `.env.prod.example` ALSO assert value matches `<CHANGEME-` pattern (mirrors `test_env_prod_example_no_real_secrets`).

- **`test_cloudflare_smoke_has_6_checks.py`** — same shape as `test_deploy_smoke_has_11_checks.py`. Assert `CF-01..CF-06` all appear in `tests/infra/cloudflare_smoke.sh`. Plus `bash -n` syntax check (analog: `test_smoke_sh_bash_syntax_valid` lines 190-198) and executable bit (analog: `test_smoke_sh_is_executable` lines 186-187).

- **`test_makefile_cf_targets.py`** — analog: `test_makefile_devex.py` (entire file, 141 lines). Reuse the `_make_dry_run` helper verbatim. Adapt: 5 tests, one per target (`cf-up`, `cf-down`, `cf-status`, `cf-logs`, `cf-smoke-external`). Each asserts `make -n <target>` exit 0 and stdout contains the expected core token (`docker compose ... up -d cloudflared`, `stop cloudflared`, `cloudflare_smoke.sh`, etc.). Plus `test_phony_declares_cf_targets` mirroring lines 112-125.

- **`test_cloudflare_docs_present.py`** — analog: `test_deploy_lan_doc.py` lines 25-178 (the doc validation half). Adapt: assert `docs/CLOUDFLARE_DEPLOY.md` exists, has >=300 lines, contains required keyword sections (`Tunnel`, `Access`, `allowlist`, `OTP`, `denial`, `rollback`, `LAN fallback`), references `make cf-up` and `tests/infra/cloudflare_smoke.sh` and `docs/decisions/ADR-001-cloudflare-tunnel.md`, has `<TUNNEL_TOKEN_REDACTED>` placeholder, and passes the no-emoji regex check (lines 141-147 of analog).

---

## Shared Patterns

### Bash smoke script skeleton

**Source:** `tests/infra/deploy_smoke.sh` lines 21-40 (set flags + check helper).
**Apply to:** `tests/infra/cloudflare_smoke.sh`, `scripts/cloudflare-bootstrap.sh`.

```bash
set -uo pipefail   # smoke: NO -e (we count failures)
IFS=$'\n\t'
FAILS=0
check() { ... echo "[${req}] OK|FAIL: ..."; FAILS=$((FAILS+1)); }
... checks ...
exit "${FAILS}"
```

For `cloudflare-bootstrap.sh` use `set -euo pipefail` (deploy-style: fail-fast) per `scripts/deploy.sh` line 13.

### Pytest static file validator skeleton

**Source:** `tests/infra/test_caddyfile_prod.py` (entire file, simplest variant).
**Apply to:** all 7 new pytest files.

```python
from pathlib import Path
TARGET = Path(__file__).resolve().parents[2] / "<relative>" / "<file>"

def test_target_exists():
    assert TARGET.exists()

def test_target_contains_X():
    content = TARGET.read_text(encoding="utf-8")
    assert "literal" in content, "explanatory message + decision id (D-XX)"
```

### Compose service block shape

**Source:** `docker-compose.prod.yml` `caddy:` block (lines 43-58).
**Apply to:** new `cloudflared:` service.

Pattern: `image | restart | command | environment | depends_on (service_healthy) | healthcheck | deploy.resources.limits`. Always set `restart: unless-stopped`; never publish ports for outbound-only services.

### Makefile target convention

**Source:** `Makefile` lines 148-167.
**Apply to:** 5 new `cf-*` targets.

Pattern: `target-name: ## help text` + `\t$(PROD_COMPOSE) <verb>`. Always add to top-line `.PHONY:` declaration.

### Doc placeholder + decision-id convention

**Source:** `docs/DEPLOY_LAN.md` (placeholders `<IP_NEXO>`, `<SUBNET_LAN>`, `<ADMIN_BACKUP_NAME>`).
**Apply to:** `docs/CLOUDFLARE_DEPLOY.md`.

Pattern: angle-bracket placeholders for operator-fill values (`<TUNNEL_TOKEN_REDACTED>`, `<EMAIL_OWNER>`, `<DOMAIN>`); inline decision IDs `(D-XX)` next to every prescriptive statement; no emojis (regression test enforces).

---

## No Analog Found

None — every file has an exact or near-exact analog inside this same repo. Phase 9 is fully a "stamp the existing pattern with CF-specific values" exercise. The planner can reference each analog file:line with high confidence.

---

## Metadata

**Analog search scope:** `caddy/`, `docker-compose*.yml`, `Makefile`, `scripts/`, `tests/infra/`, `templates/`, `docs/`, `CHANGELOG.md`, `CLAUDE.md`.
**Files scanned:** 14 (every cited analog read in this session).
**Pattern extraction date:** 2026-04-25.
