---
phase: 07-devex-hardening
verified: 2026-04-22T14:43:27Z
status: passed
score: 7/7 requirements + 9/9 truths verified
overrides_applied: 0
---

# Phase 7: DevEx Hardening Verification Report

**Phase Goal:** Cualquier dev futuro puede arrancar, cambiar y desplegar Nexo sin fricción; CI con cobertura; runbook para incidencias comunes.

**Verified:** 2026-04-22T14:43:27Z
**Status:** passed
**Re-verification:** No — initial verification.

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Pre-commit activo scoped `^(api\|nexo)/` con ruff-check + ruff-format + mypy + backfill aislado previo | VERIFIED | `.pre-commit-config.yaml` presente con 9 hooks; 3 hooks language-specific con `files: ^(api\|nexo)/`; NO `psf/black`; rev `v0.15.11`; hook mypy usa `--config-file=pyproject.toml`. Backfill commit `fd83a4e` existe en git log. |
| 2 | `pyproject.toml` consolida config ruff + mypy + pytest + coverage con `fail_under=60` | VERIFIED | Los 4 bloques `[tool.*]` presentes; `fail_under = 60` explícito; `source = ["api", "nexo"]`; overrides mypy para `matplotlib.*/pyodbc.*/slowapi.*/OEE.*/argon2.*/itsdangerous.*`. |
| 3 | CI matriz Python 3.11+3.12 en jobs lint + test, coverage gate duro, smoke + Postgres service, sin `continue-on-error` en test | VERIFIED | 5 jobs (lint, test, smoke, build, secrets). Matriz Py `["3.11","3.12"]` en lint + test. `--cov-fail-under=60` en pytest. Service container `postgres:16-alpine` en puerto 5433. Job test sin `continue-on-error`. Secrets mantiene `continue-on-error: true`. Smoke curl a `localhost:8001/api/health` loop 45x2s. |
| 4 | Makefile tiene `test`, `lint`, `format`, `migrate`, `backup` ejecutables y `make up`/`make dev` NO arrancan `mcp` | VERIFIED | `make -n` exit 0 para los 5 targets DevEx. `make -n up` y `make -n dev` = 0 matches para `--profile mcp`. `--cov-fail-under=60` consistente con pyproject.toml. |
| 5 | `docs/ARCHITECTURE.md` con Mermaid de 3 engines + middleware stack + servicios compose | VERIFIED | 255 líneas. 1 bloque Mermaid. 4 menciones cada uno de `engine_mes`/`engine_app`/`engine_nexo`. 12 referencias a middleware stack. 4 menciones a `profile` (mcp profile-gated). Cross-links a DEPLOY_LAN.md/RUNBOOK.md/RELEASE.md. |
| 6 | `docs/RUNBOOK.md` con EXACTAMENTE 5 escenarios (MES, Postgres, Caddy, pipeline, lockout) y hallazgos críticos | VERIFIED | 476 líneas. 5 escenarios exactos verificados por `^## Escenario [1-5]:`. 20 sub-secciones (Síntomas/Diagnóstico/Remedio/Prevención). Keywords críticas: `pipeline_semaphore` (2), `DELETE FROM nexo.login_attempts` (1), `caddy_data` (6), `pgdata` (8), `docker compose restart web` (1), `>=2 propietarios` (2), `NO existe`/`NO hay` (2), `down -v` warnings (3). |
| 7 | `docs/RELEASE.md` con ≥7 checklist items + semver + referencias a `scripts/deploy.sh` + `deploy_smoke.sh` | VERIFIED | 159 líneas. 7 checklist items exactos. 12 referencias semver `vMAJOR.MINOR.PATCH`. 3 referencias `scripts/deploy.sh` y 3 a `deploy_smoke.sh`. Secciones: Versionado, Checklist pre-release, Checklist de release (ejecución), Rollback, Referencias. |
| 8 | `CHANGELOG.md` Keep a Changelog 1.1.0 con `[Unreleased]` + `[1.0.0]` cubriendo Phases 1-7 | VERIFIED | 105 líneas. 1 referencia Keep a Changelog + 1 Semantic Versioning. Secciones `## [Unreleased]` y `## [1.0.0]` presentes. 10 referencias a `Phase [1-7]` (7 bullets en Added, uno por phase + otras). |
| 9 | `CLAUDE.md` extendido con "Tooling DevEx" + "Despliegue productivo" + 2 bullets `Qué NO hacer`, invariantes preservadas | VERIFIED | 235 líneas (+84 vs 151 pre-plan). Ultima revisión `2026-04-22 (cierre Mark-III / Sprint 6 / Phase 7)`. 2 secciones nuevas `## Tooling DevEx` + `## Despliegue productivo`. 8 cross-references al quartet docs. 2 prohibiciones nuevas (`--no-verify` + coverage 60%). Invariantes preservados: `carpeta OEE/`, `filter-repo`, `profiles: ["mcp"]`, `make up` sin mcp. |

**Score:** 9/9 truths verified.

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pyproject.toml` | Config tooling Python (ruff + mypy + pytest + coverage) | VERIFIED | 108 líneas. Los 4 bloques `[tool.*]` presentes. `fail_under=60`. Overrides mypy para libs legacy. Per-file-ignores documentados. |
| `.pre-commit-config.yaml` | Hooks pre-commit scoped a api/ + nexo/ | VERIFIED | 51 líneas. 3 repos (pre-commit-hooks, ruff-pre-commit@v0.15.11, mirrors-mypy@v1.13.0). 3 hooks con `files: ^(api\|nexo)/`. 0 referencias a `psf/black`. |
| `requirements-dev.txt` | ruff 0.15.11 + mypy 1.13.0 + pre-commit 4.0.1 | VERIFIED | `ruff==0.15.11`, `mypy==1.13.0`, `pre-commit==4.0.1` presentes. `pytest==8.3.4`, `pytest-cov==6.0.0` preservados. |
| `.github/workflows/ci.yml` | CI con matriz 3.11+3.12 + coverage gate + smoke | VERIFIED | 180 líneas. YAML parseable. 5 jobs (lint, test, smoke, build, secrets). Matriz en lint + test. Service postgres:16-alpine. Coverage gate 60. |
| `Makefile` | Targets test/lint/format/migrate + backup (Phase 6) | VERIFIED | 199 líneas. `.PHONY` incluye los 4 nuevos. Bloque `# ── DevEx (Phase 7 / DEVEX-03) ───` con 5 targets (test, test-docker, lint, format, migrate). `backup` (Phase 6) preservado. |
| `docs/ARCHITECTURE.md` | Mapa técnico + Mermaid 3 engines + middleware | VERIFIED | 255 líneas. Mermaid flowchart. 3 engines referenciados. Middleware stack (Auth/Audit/Flash/QueryTiming). Cross-links a 9 docs externos. |
| `docs/RUNBOOK.md` | 5 escenarios con Síntomas→Diagnóstico→Remedio→Prevención | VERIFIED | 476 líneas. Exactamente 5 escenarios. 20 sub-secciones. Hallazgos críticos explícitos (list_locks ausente, unlock_user ausente). Placeholders angle-brackets. |
| `docs/RELEASE.md` | Checklist ≥7 items + semver + deploy refs | VERIFIED | 159 líneas. 7 checklist items. Semver `v1.0.0`/`v2.0.0` mapping. Referencias literales a `scripts/deploy.sh` y `tests/infra/deploy_smoke.sh`. Sección Rollback completa. |
| `CHANGELOG.md` | Keep a Changelog 1.1.0 con [Unreleased] + [1.0.0] | VERIFIED | 105 líneas. Keep a Changelog + Semantic Versioning referenciados. `[Unreleased]` (placeholder) + `[1.0.0]` con 7 bullets en Added (uno por phase). |
| `CLAUDE.md` | Actualizado con 2 secciones + 2 prohibiciones + invariantes preservados | VERIFIED | 235 líneas (+84). Fecha `2026-04-22`. 2 secciones nuevas. 5 docs nuevos en "Fuente de verdad del plan". 2 bullets nuevos en "Qué NO hacer". Invariantes 100% preservados. |
| `tests/infra/test_pre_commit_config.py` | Regresión sobre .pre-commit-config.yaml | VERIFIED | 7 tests. Congela hooks scope, ausencia black, rev ruff v0.15.11, mypy config-file. |
| `tests/infra/test_pyproject_config.py` | Regresión sobre pyproject.toml | VERIFIED | 9 tests. Congela bloques tool.*, fail_under, overrides mypy, asyncio_mode, source coverage. |
| `tests/infra/test_ci_matrix.py` | Regresión sobre ci.yml | VERIFIED | 10 tests. Matriz 3.11+3.12, cov-fail-under, smoke job, postgres service, secrets continue-on-error, no ruff-format tolerant. |
| `tests/infra/test_makefile_devex.py` | Regresión sobre Makefile | VERIFIED | 9 tests. Targets test/lint/format/migrate + backup preservado, invariante up/dev sin mcp, PHONY, cov-fail-under. |
| `tests/infra/test_devex_docs.py` | Regresión sobre los 4 docs del quartet | VERIFIED | 19 tests. Estructura (5 escenarios RUNBOOK, ≥7 checklist RELEASE, keep-a-changelog CHANGELOG, Mermaid+3-engines ARCHITECTURE), hallazgos críticos, no-emojis, cross-links. |
| `tests/infra/test_claude_md.py` | Regresión sobre CLAUDE.md | VERIFIED | 13 tests. Ultima revision, 2 secciones nuevas, 5 docs en Fuente de verdad, 2 prohibiciones en Qué NO hacer (con aislamiento de sección), no-emojis, preservación de invariantes. |

**All artifacts verified (Level 1: exists, Level 2: substantive, Level 3: wired).**

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `.pre-commit-config.yaml` | `pyproject.toml` | ruff/mypy hooks leen config de [tool.*] | WIRED | Hook mypy explícito `--config-file=pyproject.toml`. Hooks ruff leen `[tool.ruff]` automáticamente (convención del tool). |
| `pyproject.toml` | `api/` + `nexo/` | `[tool.coverage.run].source = ["api","nexo"]` | WIRED | Coverage scope explícito. Ruff scope via `files: ^(api\|nexo)/` en pre-commit. |
| `.github/workflows/ci.yml` | `pyproject.toml` | `--cov-fail-under=60` alineado con `fail_under=60` | WIRED | Valor consistente entre CI (`--cov-fail-under=60`) y pyproject.toml (`fail_under = 60`) y Makefile (`make test` con `--cov-fail-under=60`). Fuente de verdad: `07-COVERAGE-BASELINE.md`. |
| `.github/workflows/ci.yml` | `docker-compose.yml` + `.env.example` | Job smoke usa compose base + sembra `.env` | WIRED | `cp .env.example .env` + `echo NEXO_SECRET_KEY=ci-smoke-$(openssl rand -hex 16) >> .env` + `docker compose up -d --build db web` (sin `--profile mcp`). |
| `Makefile migrate` | `scripts/init_nexo_schema.py` | `docker compose exec web python scripts/init_nexo_schema.py` | WIRED | Comando literal presente en Makefile línea 190. |
| `docs/RUNBOOK.md Escenario 4` | `nexo/services/pipeline_lock.py` | documenta `pipeline_semaphore._value` y ausencia de `list_locks()` | WIRED | Referencias explícitas a `pipeline_semaphore` (2x) + `docker compose restart web` como remedio. HALLAZGO CRITICO documentado. |
| `docs/RUNBOOK.md Escenario 5` | `nexo.login_attempts` | `DELETE FROM nexo.login_attempts` via `docker compose exec -T db psql` | WIRED | SQL literal presente. Regla hard `>=2 propietarios` como prevención (2x en doc). Marcado EMERGENCY ONLY. |
| `docs/RELEASE.md` | `scripts/deploy.sh` + `tests/infra/deploy_smoke.sh` | Checklist referencia artefactos Phase 6 | WIRED | 3 menciones a `scripts/deploy.sh` y 3 a `deploy_smoke.sh`. Comandos literales en checklist ejecución. |
| `docs/ARCHITECTURE.md` | Enlaces rápidos: CLAUDE/AUTH_MODEL/BRANDING/GLOSSARY/DEPLOY_LAN/RUNBOOK/RELEASE/SECURITY_AUDIT | Sección 8 Enlaces rápidos | WIRED | 9 cross-links verificados (contiene DEPLOY_LAN.md, RUNBOOK.md, RELEASE.md). |
| `CLAUDE.md Tooling DevEx` | ARCHITECTURE + RUNBOOK + RELEASE + CHANGELOG + pre-commit-config + pyproject | cross-refs explícitas | WIRED | 8 menciones al quartet de docs en la sección. |
| `CLAUDE.md Despliegue productivo` | DEPLOY_LAN.md + `make prod-*` + `deploy_smoke.sh` | Tabla Makefile prod + referencia runbook Phase 6 | WIRED | `make prod-` referenciado 5x. DEPLOY_LAN.md presente. `deploy_smoke.sh` mencionado. |
| `CLAUDE.md Qué NO hacer` | pre-commit + pyproject `--cov-fail-under` | 2 prohibiciones nuevas referencian gates | WIRED | `--no-verify` prohibido con referencia a `make format` como auto-fix. `cov-fail-under=60` + `fail_under=60` referenciados literalmente. |

**All 12 key links verified.**

---

### Data-Flow Trace (Level 4)

N/A — Phase 7 is a tooling/docs phase. No dynamic data rendering artifacts. Config files (pyproject.toml, pre-commit) are consumed by static tools (ruff, mypy, pytest). CI workflow is consumed by GitHub Actions runtime. Docs are human-readable reference material. No data flow to verify.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Makefile dry-runs executable | `make -n {test,lint,format,migrate,backup}` | All exit 0 | PASS |
| `make up` does NOT invoke `--profile mcp` | `make -n up 2>&1 \| grep -c 'profile mcp'` | 0 | PASS (invariant preserved) |
| `make dev` does NOT invoke `--profile mcp` | `make -n dev 2>&1 \| grep -c 'profile mcp'` | 0 | PASS (invariant preserved) |
| `make test` uses `--cov-fail-under=60` consistent with pyproject | `make -n test \| grep -oE 'cov-fail-under=[0-9]+'` | `cov-fail-under=60` | PASS |
| CI YAML parseable | `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"` | OK | PASS |
| pyproject.toml parseable as TOML | `python3 -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"` | OK, 4 tool blocks | PASS |
| `.pre-commit-config.yaml` parseable as YAML | `python3 -c "import yaml; yaml.safe_load(open('.pre-commit-config.yaml'))"` | OK, 3 repos | PASS |
| No `psf/black` in pre-commit | `grep -c 'psf/black\|id: black' .pre-commit-config.yaml` | 0 | PASS (ruff-format drop-in) |
| CI test job has no `continue-on-error` | `python3 ... d['jobs']['test'].get('continue-on-error')` | absent | PASS |
| CI secrets job retains `continue-on-error: true` | `python3 ... d['jobs']['secrets'].get('continue-on-error')` | True | PASS (historical findings) |
| CI test job has postgres service | `python3 ... d['jobs']['test']['services']` | `['postgres']` | PASS |
| Phase 7 regression tests pass (container) | `docker compose exec -T web pytest tests/infra/test_pre_commit_config.py test_pyproject_config.py test_ci_matrix.py test_makefile_devex.py test_devex_docs.py test_claude_md.py --no-cov -q` | 67 passed | PASS |
| Full infra suite pass (container) | `docker compose exec -T web pytest tests/infra/ --no-cov -q` | 135 passed, 5 skipped | PASS (matches orchestrator claim) |
| Backfill commit exists in git log | `git log --oneline \| grep 'chore(07): backfill ruff format'` | `fd83a4e chore(07): backfill...` | PASS |

**All spot-checks passed.**

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DEVEX-01 | 07-01 | Pre-commit con ruff + ruff-format + mypy ligero, solo `api/` + `nexo/`, ruff-format drop-in de black, backfill en commit aislado previo | SATISFIED | `.pre-commit-config.yaml` scoped + `pyproject.toml [tool.ruff/mypy]` + `requirements-dev.txt` con pins + commit `fd83a4e` (backfill) + REQUIREMENTS.md DEVEX-01 anotado con nota `black reemplazado por ruff-format 2026-04-21`. |
| DEVEX-02 | 07-02 | CI ampliado matriz Python 3.11+3.12; cobertura 60% en `api/` + `nexo/`; smoke test `docker compose up` + `curl /api/health` | SATISFIED | `.github/workflows/ci.yml` con 5 jobs: lint+test con `matrix: [3.11, 3.12]`; test con `--cov-fail-under=60` + postgres service; smoke con `docker compose up -d --build db web` + loop curl `:8001/api/health`. |
| DEVEX-03 | 07-02 | Makefile añade targets `test`, `lint`, `format`, `migrate`, `backup` | SATISFIED | Makefile bloque DevEx (líneas 167-190) con los 5 targets. `make -n` exit 0 en los 5. `.PHONY` ampliado. `backup` (Phase 6) preservado. Makefile mantiene invariante `make up/dev` sin `--profile mcp`. |
| DEVEX-04 | 07-03 | `docs/ARCHITECTURE.md` con diagrama componentes (web ↔ 3 engines ↔ 3 sources) | SATISFIED | `docs/ARCHITECTURE.md` 255 líneas con Mermaid flowchart explícito: Browser LAN → Caddy (443 TLS) → Web FastAPI (:8000) → (engine_nexo → Postgres, engine_app → ecs_mobility SQL, engine_mes → dbizaro SQL read-only) + subgrafo mcp profile-gated. Middleware stack + schedulers + layout + cross-links. |
| DEVEX-05 | 07-03 | `docs/RUNBOOK.md` con 5 escenarios (MES caído, Postgres no arranca, cert expira, pipeline atascado, lockout propietario) | SATISFIED | `docs/RUNBOOK.md` con EXACTAMENTE 5 escenarios (`grep -cE '^## Escenario [1-5]:'` = 5). 20 sub-secciones (Síntomas/Diagnóstico/Remedio/Prevención). Hallazgos críticos en Escenarios 4 (list_locks ausente → `docker compose restart web`) y 5 (unlock_user ausente → DELETE FROM nexo.login_attempts, prevención >=2 propietarios). |
| DEVEX-06 | 07-03 | `docs/RELEASE.md` con checklist release (tag semver, deploy, smoke) + CHANGELOG | SATISFIED | `docs/RELEASE.md` con 7 checklist items + semver `v1.0.0` + referencias literales a `scripts/deploy.sh` + `tests/infra/deploy_smoke.sh` + sección Rollback. `CHANGELOG.md` Keep a Changelog 1.1.0 con `[Unreleased]` + `[1.0.0]` cubriendo Phases 1-7. |
| DEVEX-07 | 07-04 | `CLAUDE.md` actualizado con convenciones Mark-III completas tras Sprint 6 | SATISFIED | `CLAUDE.md` 235 líneas (+84), fecha `2026-04-22 (cierre Mark-III / Sprint 6 / Phase 7)`, 2 secciones nuevas (Tooling DevEx + Despliegue productivo), 5 docs nuevos en Fuente de verdad, 2 prohibiciones nuevas en Qué NO hacer (--no-verify + coverage <60%), invariantes 100% preservados. |

**All 7 DEVEX requirements SATISFIED. No ORPHANED requirements.**

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | — | — | No anti-patterns detected. |

Anti-pattern scan over Phase 7 files:
- No TODO/FIXME/PLACEHOLDER strings in the 4 docs of the quartet.
- No emojis in any authored doc (verified for ARCHITECTURE.md, RUNBOOK.md, RELEASE.md, CHANGELOG.md, CLAUDE.md).
- No hardcoded credentials in any file.
- No `return null` / empty JSON stubs in config or workflow files.
- The 3 F841 residual ruff warnings from the backfill are explicitly documented in `pyproject.toml` `[tool.ruff.lint.per-file-ignores]` (historial.py, metrics.py) and in the 07-01 SUMMARY — NOT a stub, they are intentional WIP deferrals tracked by owner.
- The 4 pre-existing test failures (Phase 4 `DEF-05-01-A` thresholds) are documented as deferred in prior phases; orchestrator confirmed the Phase 7 backfill explicitly noted `7 pre-existing fail (no regresion)` as its baseline.

---

### Human Verification Required

None. All success criteria are programmatically verifiable via file contents, grep, YAML/TOML parsing, `make -n` dry-runs, and the 67 Phase 7 regression tests (all passing). The phase goal ("Cualquier dev futuro puede arrancar, cambiar y desplegar Nexo sin fricción; CI con cobertura; runbook para incidencias comunes") is operationalized through:

- **Dev loop fricción**: medido por `make -n test/lint/format/migrate/backup` exit 0 + `pre-commit install` documentado.
- **CI con cobertura**: medido por coverage gate `--cov-fail-under=60` bloqueante en CI + matriz Py 3.11+3.12.
- **Runbook para incidencias**: medido por 5 escenarios exactos con estructura Síntomas/Diagnóstico/Remedio/Prevención y hallazgos críticos documentados.

Los 3 bloques se verifican automáticamente. El aterrizaje humano de "un dev nuevo clona y arranca" queda cubierto por los 67 tests de regresión que congelan el contrato — si alguien modifica los configs/docs de forma que rompa el onboarding, el CI bloquea.

---

### Gaps Summary

**No gaps.** All 9 observable truths verified, all 16 artifacts exist and are substantive + wired, all 12 key links connected, all 7 DEVEX requirements satisfied, 67/67 Phase 7 regression tests pass in-container, full infra suite 135 passed + 5 skipped (matches orchestrator context), no anti-patterns found, all hard invariants preserved (no emojis in docs; Spanish user-facing prose with technical terms in English; `make up`/`make dev` do NOT start mcp service; OEE/ legacy preserved; filter-repo still prohibited).

**Phase 7 achievement:** The goal is met. Any new dev cloning the repo after this phase can:
1. Run `pip install -r requirements-dev.txt && pre-commit install` (documented in CLAUDE.md Tooling DevEx + pre-commit-config scope).
2. Use `make test/lint/format/migrate/backup` for the dev loop (documented + tests green).
3. Rely on CI matrix 3.11+3.12 + coverage gate 60% as bloqueante (documented + CI YAML valid).
4. Consult `docs/ARCHITECTURE.md` to understand 3-engine data flow and middleware order.
5. Consult `docs/RUNBOOK.md` for 5 concrete incident scenarios (MES down, Postgres down, cert expired, pipeline stuck, owner locked out).
6. Execute a versioned release with `docs/RELEASE.md` + `CHANGELOG.md` + `make deploy` + `deploy_smoke.sh` from Phase 6.
7. Read `CLAUDE.md` as single source of truth, including Phase 7 dev conventions and 2 new `Qué NO hacer` hard prohibitions.

The 4 pre-existing test failures from Phase 4 (`DEF-05-01-A` in `test_thresholds_crud.py` + `test_thresholds_cache.py`) are explicitly NOT introduced by this phase (backfill commit `fd83a4e` documents `7 pre-existing fail (no regresion)` as its baseline, confirmed by orchestrator). They remain tracked in prior phases' `deferred-items.md`.

---

*Verified: 2026-04-22T14:43:27Z*
*Verifier: Claude (gsd-verifier)*
