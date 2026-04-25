---
phase: 9
slug: cloudflare-tunnel-public-access
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-25
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution. Derived from `09-RESEARCH.md` § Validation Architecture (líneas 759-862).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework Python** | pytest 8.x (existente, configurado en `pyproject.toml`) |
| **Framework infra** | bash + curl + openssl + dig (mismo stack que Phase 6 `tests/infra/deploy_smoke.sh`) |
| **Config file** | `pyproject.toml` (pytest + coverage gate 60%); `tests/infra/*.sh` standalone bash |
| **Quick run command (Python)** | `make test` → `pytest -q --cov=api --cov=nexo --cov-fail-under=60` |
| **Quick run command (LAN smoke)** | `bash tests/infra/deploy_smoke.sh` (requires server access + sudo) |
| **Quick run command (CF smoke)** | `bash tests/infra/cloudflare_smoke.sh` (requires external network — 4G tethering) |
| **Full suite command** | `make test && bash tests/infra/deploy_smoke.sh && bash tests/infra/cloudflare_smoke.sh` |
| **Estimated runtime** | ~80s `make test` + ~30s LAN smoke + ~45s CF smoke = ~155s total |

---

## Sampling Rate

- **After every task commit:** Run `make test` (60s) + `make lint` (20s) — gates Python ya en CI desde Phase 7
- **After every plan wave:** Run `make test` + nuevos `tests/infra/test_*.py` parsing tests (los `_smoke.sh` requieren entorno real, no en CI)
- **Phase gate (manual antes de `/gsd-verify-work`):**
  1. SSH al servidor: `make cf-up && make cf-status`
  2. `bash tests/infra/deploy_smoke.sh` (espera 11/11)
  3. `make cf-down && bash tests/infra/deploy_smoke.sh` (espera 11/11 — verifica fallback LAN intacto, CLOUD-12)
  4. `make cf-up`
  5. Desde 4G ajeno: `bash tests/infra/cloudflare_smoke.sh` (espera 6/6)
- **Max feedback latency:** ~155s (full suite). Quick test gate: ~80s.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 09-01-W0-01 | 01 | 0 | CLOUD-03 / CLOUD-08 | T-09-04 (clickjacking) / T-09-05 (MIME sniff) / T-09-06 (downgrade) / T-09-07 (referer leak) | 5 security headers presentes en respuestas Caddy | unit | `pytest tests/infra/test_caddyfile_security_headers.py` | ❌ Wave 0 NEW | ⬜ pending |
| 09-01-W1-01 | 01 | 1 | CLOUD-03 | T-09-04..07 | `Caddyfile.prod` contiene snippet `(security_headers)` con HSTS+XFO+XCTO+Referrer-Policy+Permissions-Policy | unit | `pytest tests/infra/test_caddyfile_security_headers.py` | ✅ tras Wave 0 | ⬜ pending |
| 09-01-W1-02 | 01 | 1 | CLOUD-08 | — | `deploy_smoke.sh` tiene 11 `[DEPLOY-NN]` tags | unit | `pytest tests/infra/test_deploy_smoke_has_11_checks.py` | ❌ Wave 0 NEW | ⬜ pending |
| 09-01-W2-01 | 01 | 2 | CLOUD-03 | T-09-04..07 | LAN smoke pasa 11/11 con headers integrados | manual | `bash tests/infra/deploy_smoke.sh` (server) | ✅ scripts existirán | ⬜ manual |
| 09-02-W0-01 | 02 | 0 | CLOUD-04 | T-09-01 (token leak) | `docker-compose.prod.yml` define servicio `cloudflared` con shape correcta (image, command, environment TUNNEL_TOKEN, depends_on caddy healthy, healthcheck, restart) | unit | `pytest tests/infra/test_cloudflared_compose_service.py` | ❌ Wave 0 NEW | ⬜ pending |
| 09-02-W0-02 | 02 | 0 | CLOUD-05 | T-09-01 | `.env.example` y `.env.prod.example` contienen línea `CLOUDFLARE_TUNNEL_TOKEN=` con comentario procedencia | unit | `pytest tests/infra/test_env_example_has_cf_token.py` | ❌ Wave 0 NEW | ⬜ pending |
| 09-02-W0-03 | 02 | 0 | CLOUD-09 | T-09-03 (Access bypass) | `cloudflare_smoke.sh` existe + 6 `[CF-0X]` tags + ejecutable | unit | `pytest tests/infra/test_cloudflare_smoke_has_6_checks.py` | ❌ Wave 0 NEW | ⬜ pending |
| 09-02-W0-04 | 02 | 0 | CLOUD-10 | — | Makefile tiene 5 targets `cf-up cf-down cf-status cf-logs cf-smoke-external` con docstrings `## ...` | unit | `pytest tests/infra/test_makefile_cf_targets.py` | ❌ Wave 0 NEW | ⬜ pending |
| 09-02-W0-05 | 02 | 0 | CLOUD-11 | — | `docs/CLOUDFLARE_DEPLOY.md` existe + N secciones; ADR-001 exists; `CLAUDE.md` mentions "Cloudflare Tunnel" en sección Despliegue | unit | `pytest tests/infra/test_cloudflare_docs_present.py` | ❌ Wave 0 NEW | ⬜ pending |
| 09-02-W2-01 | 02 | 2 | CLOUD-04 | T-09-01 | Container `cloudflared` running + tunnel HEALTHY tras `make cf-up` | manual | `make cf-up && make cf-status` (server) | ✅ tras Wave 1 | ⬜ manual |
| 09-03-W3-01 | 03 | 3 | CLOUD-01 | — | DNS público resuelve `nexo.app` al edge CF | manual | `dig +short nexo.app` (post-purchase) | N/A | ⬜ manual |
| 09-03-W3-02 | 03 | 3 | CLOUD-02 | — | Cuenta CF con 2FA activo + recovery codes guardados | manual | RUNBOOK checklist | N/A | ⬜ manual |
| 09-03-W3-03 | 03 | 3 | CLOUD-06 | T-09-02 (OTP brute force) | Usuario en allowlist recibe OTP, autorizado entra; usuario fuera rechazado | manual external | `bash tests/infra/cloudflare_smoke.sh` desde 4G + intento manual con email no autorizado | ✅ tras Wave 1 | ⬜ manual |
| 09-03-W3-04 | 03 | 3 | CLOUD-07 | — | Usuario rechazado ve `/access-denied` con branding Nexo + mensaje "contacta con Erik" | manual external | curl `https://nexo.app/access-denied` desde 4G (público sin Access) | ✅ tras Wave 1 | ⬜ manual |
| 09-03-W3-05 | 03 | 3 | CLOUD-12 | — | LAN fallback intacto: smoke 11/11 OK con cloudflared parado | manual | `make cf-down && bash tests/infra/deploy_smoke.sh` | ✅ tras Wave 1 | ⬜ manual |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Wave 0 = stubs de tests Python que se crean ANTES de cada Wave 1 implementación, para que Wave 1 los haga pasar (TDD-light, no TDD estricto). Phase 9 es 100% infra/config/docs, NO añade código de aplicación → no hay tests Python en `tests/api/` o `tests/nexo/` nuevos. Todos los Wave 0 viven en `tests/infra/`.

- [ ] **Plan 01 — Caddy hardening:**
  - [ ] `tests/infra/test_caddyfile_security_headers.py` — parsea `caddy/Caddyfile.prod`, valida snippet `(security_headers)` con 5 directives (HSTS max-age=31536000+includeSubDomains+preload, X-Frame-Options DENY, X-Content-Type-Options nosniff, Referrer-Policy strict-origin-when-cross-origin, Permissions-Policy con camera/mic/geo) (CLOUD-03)
  - [ ] `tests/infra/test_deploy_smoke_has_11_checks.py` — extender `tests/infra/test_deploy_lan_doc.py` (24 tests Phase 6); valida que `deploy_smoke.sh` contiene 11 tags `[DEPLOY-NN]` (no 8); valida los 3 nuevos checks (login, MES caído, cloudflared running) presentes (CLOUD-08)

- [ ] **Plan 02 — Cloudflared service + bootstrap + docs:**
  - [ ] `tests/infra/test_cloudflared_compose_service.py` — valida que `docker-compose.prod.yml` tiene servicio `cloudflared` con: `image: cloudflare/cloudflared:latest`, `command:` empieza con `tunnel`, `environment:` mapea `TUNNEL_TOKEN: ${CLOUDFLARE_TUNNEL_TOKEN}`, `depends_on:` tiene `caddy` con `condition: service_healthy`, `healthcheck:` con `cloudflared tunnel ready`, `restart: unless-stopped` (CLOUD-04)
  - [ ] `tests/infra/test_env_example_has_cf_token.py` — `.env.example` y `.env.prod.example` ambos contienen línea `CLOUDFLARE_TUNNEL_TOKEN=` con comentario `# Get from Cloudflare Zero Trust → ...` (CLOUD-05)
  - [ ] `tests/infra/test_cloudflare_smoke_has_6_checks.py` — `tests/infra/cloudflare_smoke.sh` exists + 6 tags `[CF-0X]` + permisos ejecutable (`stat -c '%a'` ends 5 o 7) (CLOUD-09)
  - [ ] `tests/infra/test_makefile_cf_targets.py` — Makefile contiene targets `^cf-up:`, `^cf-down:`, `^cf-status:`, `^cf-logs:`, `^cf-smoke-external:` cada uno con docstring `## ...` para help target (CLOUD-10)
  - [ ] `tests/infra/test_cloudflare_docs_present.py` — `docs/CLOUDFLARE_DEPLOY.md` exists + tiene secciones (mirror `test_deploy_lan_doc.py`); `docs/decisions/ADR-001-cloudflare-tunnel.md` exists; `CLAUDE.md` contiene "Cloudflare Tunnel" en sección "Despliegue"; `CHANGELOG.md` tiene entry mencionando "Phase 9" o "Cloudflare Tunnel" (CLOUD-11)

- [ ] **Plan 03 — Activación producción:** Sin Wave 0 nuevo (es runbook-driven manual). Verifica los smoke tests existentes desde Wave 1.

**No new pytest fixtures or shared conftest changes** — todos los `test_*.py` nuevos son standalone, leen archivos del repo y validan estructura. Coverage gate sigue cubierto por código existente (Phase 9 no añade código `api/` o `nexo/`).

---

## Manual-Only Verifications

Estos checks NO se pueden automatizar en CI porque requieren cuenta Cloudflare real, dominio comprado, conexión externa, o interacción con el dashboard CF.

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Dominio resuelve por DNS público | CLOUD-01 | Requiere cuenta CF + compra real ($14/yr) | `dig +short nexo.app` debe devolver IP de CF (cualquier `104.*` o `172.*` típico de CF) |
| 2FA + recovery codes en cuenta CF | CLOUD-02 | Operación humana en dashboard | Operador valida en RUNBOOK checklist; toma screenshots de 2FA enabled + recovery codes guardados en gestor |
| OTP llega + usuario autorizado entra | CLOUD-06 | Requiere email allowlist real + IdP CF | Operador desde móvil en 4G abre `https://nexo.app`, mete email allowed, recibe código 6 dígitos en email, mete código, ve login Nexo |
| Email NO autorizado rechazado a denial page | CLOUD-06 / CLOUD-07 | Requiere email no listado real | Operador (o asistente) abre `https://nexo.app` desde email no listado, debe ser redirigido a `/access-denied` con branding Nexo + mensaje "contacta con Erik" |
| `connectorStatus: HEALTHY` durante 24h | Success Criteria 9 | Métrica temporal, no instantánea | Operador valida 24h post-activación: `make cf-status` o consulta dashboard CF tunnel "Status: Healthy" |
| Mozilla Observatory ≥ B+ | Success Criteria 4 | Servicio externo público | Operador navega a [observatory.mozilla.org](https://observatory.mozilla.org), ingresa `nexo.app`, valida grade ≥ B+ (esperado: 80 = B+ borderline según research) |
| Fallback LAN intacto con `cloudflared` parado | CLOUD-12 / Success Criteria 5 | Requiere cambio de estado runtime | `make cf-down`, comprobar `https://nexo.ecsmobility.local` desde LAN sigue OK; `bash tests/infra/deploy_smoke.sh` 11/11 |
| Latencia post-CF aceptable para luk4 timeline | R-08 | Métrica subjetiva en uso real | Operador navega Centro de Mando luk4 desde 4G (vía CF) y desde LAN directa; documenta diferencia subjetiva en VERIFICATION.md |

---

## Validation Sign-Off

- [ ] Todas las tareas tienen `<automated>` verify O dependencies Wave 0 declaradas
- [ ] Sampling continuity: ningún 3 tareas consecutivas sin verificación automatizada (cumplido — Wave 0 cubre cada plan)
- [ ] Wave 0 cubre todas las references MISSING (7 nuevos `test_*.py` listados arriba)
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s (full suite ~155s)
- [ ] `nyquist_compliant: true` set en frontmatter al cierre de la phase
- [ ] Manual verifications listadas con instrucciones reproducibles para el operador

**Approval:** pending (será approved 2026-MM-DD al cierre de plan 09-03)
