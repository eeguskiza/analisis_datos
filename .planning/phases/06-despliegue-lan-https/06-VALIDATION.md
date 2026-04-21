---
phase: 6
slug: despliegue-lan-https
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-21
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (ya operativo) + bash integration smoke scripts (nuevos) |
| **Config file** | `pyproject.toml` (pytest config existente) |
| **Quick run command** | `pytest -x -q tests/infra/ --timeout 30` |
| **Full suite command** | `pytest -q --timeout 120 && bash tests/infra/deploy_smoke.sh` |
| **Estimated runtime** | ~45-60 seconds (unit + config parse) + runbook manual post-deploy |

Phase 6 es infra-heavy: la mayoría de DEPLOY-* se valida por **presencia de
artefactos + idempotencia de comandos** (test de configuración, no de lógica
de app). El smoke E2E real se ejecuta en el Ubuntu de producción cuando esté
asignado — documentado como "Validación post-deploy" en el runbook.

---

## Sampling Rate

- **After every task commit:** `pytest -x -q tests/infra/` (validación config files)
- **After every plan wave:** Full suite + `docker compose -f docker-compose.yml -f docker-compose.prod.yml config` (sintaxis)
- **Before `/gsd-verify-work`:** Full suite green + smoke manual `bash scripts/deploy.sh --dry-run` si se añade flag dry-run
- **Max feedback latency:** 60 segundos (validación local) + ~10 min (smoke manual cuando hay Ubuntu disponible)

---

## Per-Task Verification Map

> Plan/task IDs se asignan durante planning — esta tabla se refinará tras
> `/gsd-plan-phase 6`. Skeleton inicial:

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 06-01-TBD | 01 | 1 | DEPLOY-01 | — | Caddyfile.prod valido, tls internal, hostname literal | config | `docker compose -f docker-compose.prod.yml config > /dev/null && grep -q 'nexo.ecsmobility.local' caddy/Caddyfile.prod` | ❌ W0 | ⬜ pending |
| 06-01-TBD | 01 | 1 | DEPLOY-02 | — | Prod override no publica 5432 | config | `docker compose -f docker-compose.yml -f docker-compose.prod.yml config \| yq '.services.db.ports' \| grep -v 5432` | ❌ W0 | ⬜ pending |
| 06-01-TBD | 01 | 1 | DEPLOY-03 | — | Healthchecks web/caddy presentes | config | `docker compose -f ... -f ... config \| yq '.services.web.healthcheck.test'` no vacío | ❌ W0 | ⬜ pending |
| 06-02-TBD | 02 | 2 | DEPLOY-04 | — | deploy.sh idempotente, set -euo pipefail | script | `bash -n scripts/deploy.sh && grep -q 'set -euo pipefail' scripts/deploy.sh` | ❌ W0 | ⬜ pending |
| 06-02-TBD | 02 | 2 | DEPLOY-04 | — | deploy.sh incluye pre-deploy backup + smoke curl | script | `grep -q 'pg_dump' scripts/deploy.sh && grep -q 'curl .*api/health' scripts/deploy.sh` | ❌ W0 | ⬜ pending |
| 06-03-TBD | 03 | 3 | DEPLOY-05 | — | DEPLOY_LAN.md cubre 5 escenarios del runbook | docs | `grep -qE 'Instalar Docker\|Configurar hosts\|Restaurar backup\|Distribuir root CA\|Verificar deploy' docs/DEPLOY_LAN.md` | ❌ W0 | ⬜ pending |
| 06-03-TBD | 03 | 3 | DEPLOY-06 | — | ufw rules correctas en runbook | docs | `grep -qE 'ufw allow.*22\|ufw allow 443\|ufw default deny' docs/DEPLOY_LAN.md` | ❌ W0 | ⬜ pending |
| 06-03-TBD | 03 | 3 | DEPLOY-08 | — | .env.prod.example cubre NEXO_* + SMTP TODO | config | `test -f .env.prod.example && grep -q '# TODO Mark-IV' .env.prod.example && grep -qE '^NEXO_SECRET_KEY=' .env.prod.example` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/infra/__init__.py` — nuevo directorio para tests de infra
- [ ] `tests/infra/test_compose_override.py` — parsea `docker compose config` y valida estructura prod (ports vacíos, healthchecks, resource limits)
- [ ] `tests/infra/test_caddyfile_prod.py` — valida que `caddy/Caddyfile.prod` existe, contiene hostname literal, tls internal
- [ ] `tests/infra/test_deploy_script.py` — valida sintaxis bash de `scripts/deploy.sh` (`bash -n`) + presencia de comandos clave (pg_dump, curl smoke)
- [ ] `tests/infra/test_env_prod_example.py` — valida estructura de `.env.prod.example` (NEXO_* presente, SMTP comentado, sin valores reales)
- [ ] `tests/infra/deploy_smoke.sh` — script bash que se ejecuta post-deploy en Ubuntu real (no en CI — validación manual documentada)

Dependencia: requiere `pip install pyyaml` (o `yq` instalado) para parsear compose.
Si pyyaml ya está en requirements.txt → Wave 0 es trivial.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Cert reconocido sin warnings tras instalar root CA | DEPLOY-01, DEPLOY-07 | Requiere browser real + equipo LAN con root CA instalada | DEPLOY_LAN.md §"Verificación desde equipo cliente" |
| Postgres 5432 realmente inaccesible desde LAN peer | DEPLOY-02 | Requiere 2 máquinas (Ubuntu + peer) | `nmap -p 5432 <IP_NEXO>` desde peer → debe ser filtered/closed |
| Runbook ejecutable por admin de respaldo | DEPLOY-05 | Requiere persona diferente al autor | Admin secundario sigue DEPLOY_LAN.md solo, sin contexto |
| ufw activo con reglas correctas | DEPLOY-06 | Requiere host Ubuntu físico | `sudo ufw status verbose` |
| URL green desde peer LAN tras setup hosts-file | DEPLOY-07 | Requiere peer real con hosts configurado | Navegador → https://nexo.ecsmobility.local → login page |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s for config tests
- [ ] `nyquist_compliant: true` set in frontmatter (pending — to set after planner wires tasks to this map)

**Approval:** pending
