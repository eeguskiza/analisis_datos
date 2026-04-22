---
phase: 07-devex-hardening
plan: 02
subsystem: infra
tags:
  - devex
  - ci
  - github-actions
  - makefile
  - coverage
  - smoke

# Dependency graph
requires:
  - phase: 07-devex-hardening
    plan: 01
    provides: "pyproject.toml con [tool.coverage.report].fail_under=60 + requirements-dev.txt con mypy/pre-commit + backfill ruff format ya aplicado (sin cual el paso a bloqueante de ruff format --check rompe el CI)."
provides:
  - "CI con matriz Python 3.11+3.12 en jobs lint y test."
  - "Coverage gate duro `--cov-fail-under=60` en CI (paridad con pyproject.toml)."
  - "Job `smoke` nuevo: docker compose up -d --build db web + loop 45x2s curl :8001/api/health."
  - "Service container postgres:16-alpine en el job test (aislado del compose del smoke, puerto host 5433)."
  - "Makefile extendido con targets test, lint, format, migrate (+ test-docker alias)."
  - "19 tests nuevos en tests/infra/ (10 ci_matrix + 9 makefile_devex) que congelan el contrato."
affects:
  - 07-03-runbook-release-claude-md
  - 07-04-claude-md-pre-commit-docs

# Tech tracking
tech-stack:
  added:
    - "GH Actions strategy.matrix.python-version Py 3.11+3.12"
    - "GH Actions service-container postgres:16-alpine:5433"
    - "actions/upload-artifact@v4 para coverage.xml por matrix entry"
  patterns:
    - "CI fail-fast: false en ambas matrices (ver ambos Python fails independientes)"
    - "smoke job usa compose base + .env sembrado desde .env.example + NEXO_SECRET_KEY random por run"
    - "Tear down con if: always() garantiza limpieza incluso en fallo"
    - "Makefile DevEx block scoped a api/ + nexo/ (paridad con pyproject.toml y CI)"

key-files:
  created:
    - "tests/infra/test_ci_matrix.py (10 tests)"
    - "tests/infra/test_makefile_devex.py (9 tests)"
  modified:
    - ".github/workflows/ci.yml (4 jobs → 5 jobs, matriz, gate, smoke, postgres service)"
    - "Makefile (+5 targets: test, test-docker, lint, format, migrate; .PHONY header ampliado)"

key-decisions:
  - "fail_under=60 en CI tomado de pyproject.toml (07-COVERAGE-BASELINE.md) — no se inventa N, se consume el baseline medido."
  - "Service container postgres en el job test en puerto host 5433 (paridad con docker-compose.yml NEXO_PG_HOST_PORT default), independiente del compose del smoke job."
  - "Smoke job arranca SOLO `db web` — NO `--profile mcp`. Invariante Phase 1 congelado en test_smoke_job_exists + ausencia literal en el workflow."
  - "Puerto curl en smoke = 8001 (NEXO_WEB_HOST_PORT default), NO 8000 (ese es el puerto interno del container)."
  - "Ruff format --check ahora SIN continue-on-error (backfill de 07-01 ya aplicado). Antes era tolerante porque el codigo no habia pasado por ruff format."
  - "Secrets job conserva continue-on-error: true (historical findings en SECURITY_AUDIT.md, rotacion ya ejecutada en Sprint 0 pero historial no limpiado hasta Mark-IV)."
  - "`make migrate` implementado como ALIAS directo de nexo-init (sin dependency de Make) para cumplir literal DEVEX-03."

patterns-established:
  - "CI matrix pattern: `strategy.matrix.python-version` con fail-fast: false; artifacts nombrados con ${{ matrix.python-version }} para trazabilidad."
  - "Smoke test pattern: loop 45x2s curl con exit 1 + dump de logs web/db en el fail path."
  - "Makefile regresion test pattern: `subprocess.run(['make', '-n', target])` + aserts sobre stdout; congela el contrato sin ejecutar receta real."

requirements-completed:
  - DEVEX-02
  - DEVEX-03

# Metrics
duration: ~5min
completed: 2026-04-22
---

# Phase 07 Plan 02: CI coverage gate + matriz + smoke + Makefile DevEx targets Summary

**Extension del CI (4 → 5 jobs) con matriz Python 3.11+3.12, coverage gate `--cov-fail-under=60` bloqueante, service container postgres:16-alpine en el job test, y nuevo job smoke que arranca `docker compose up -d --build db web` + curl a `:8001/api/health`. Simultaneamente, Makefile ampliado con 5 targets DevEx (test, test-docker, lint, format, migrate) y 19 tests nuevos que congelan ambos contratos.**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-04-22T14:01:32Z
- **Completed:** 2026-04-22T14:07:09Z
- **Tasks:** 2 (Task 1 CI matrix + smoke; Task 2 Makefile targets)
- **Files changed:** 2 modified (.github/workflows/ci.yml, Makefile) + 2 created (tests/infra/test_ci_matrix.py, tests/infra/test_makefile_devex.py)
- **Tests anadidos:** 19 (10 ci_matrix + 9 makefile_devex)

## Accomplishments

- **CI con matriz Python 3.11+3.12**: jobs `lint` y `test` duplicados por version de Python (fail-fast: false → vemos ambos fails independientes). Artifacts `coverage-3.11` y `coverage-3.12` publicados separados.
- **Coverage gate duro activado**: `--cov-fail-under=60` en el step `pytest` del job `test`. Valor consistente con `pyproject.toml [tool.coverage.report].fail_under=60` (fuente de verdad de 07-01).
- **`continue-on-error: true` retirado** del job `test` y del step `ruff format --check` del job `lint`. El CI vuelve a ser bloqueante de verdad tras el backfill de 07-01.
- **Job `smoke` nuevo**: arranca `docker compose up -d --build db web` (NO `--profile mcp` — invariante Phase 1 preservado), siembra `.env` desde `.env.example` + `NEXO_SECRET_KEY` random, hace loop 45x2s curl `localhost:8001/api/health`, dump de logs web+db en failure, `docker compose down` con `if: always()`.
- **Service container `postgres:16-alpine`** declarado en job `test` con health-cmd `pg_isready`, puerto host 5433 (paridad con `NEXO_PG_HOST_PORT` default del compose). Step `Init schema (Postgres service)` ejecuta `scripts/init_nexo_schema.py` contra el.
- **Makefile extendido** con bloque `# ── DevEx (Phase 7 / DEVEX-03) ───` y 5 nuevos targets: `test`, `test-docker`, `lint`, `format`, `migrate`. `backup` de Phase 6 intacto. `.PHONY` del header ampliado con los 4 nuevos (+ test-docker).
- **19 tests de regresion** nuevos en `tests/infra/`. Suite infra de configs ahora 35 tests (10 ci_matrix + 9 makefile_devex + 7 pre_commit + 9 pyproject), todos verdes.

## Task Commits

Cada task committed atomicamente en el worktree `agent-ac81f7e7`:

1. **Task 1: CI matrix + coverage gate + smoke job** — `ff27cb1` (ci) → `.github/workflows/ci.yml` + `tests/infra/test_ci_matrix.py`
2. **Task 2: Makefile DevEx targets** — `d06c8ae` (feat) → `Makefile` + `tests/infra/test_makefile_devex.py`

## CI jobs — estado final (continue-on-error policy)

| Job      | Py matrix         | continue-on-error | Bloquea merge? | Notas                                                    |
|----------|-------------------|-------------------|----------------|----------------------------------------------------------|
| `lint`   | 3.11 + 3.12       | (ninguno)         | Sí             | ruff check + ruff format --check + mypy (scoped api/ nexo/) |
| `test`   | 3.11 + 3.12       | (ninguno)         | Sí             | Service container postgres:16-alpine:5433 + --cov-fail-under=60 |
| `smoke`  | (sin matriz)      | (ninguno)         | Sí             | docker compose up -d --build db web + loop 45x2s curl :8001/api/health |
| `build`  | (sin matriz)      | (ninguno)         | Sí             | docker build nexo-web:ci + nexo-mcp:ci                   |
| `secrets`| (sin matriz)      | `true`            | No             | gitleaks — historical findings, rotacion ya ejecutada en Sprint 0 |

`--cov-fail-under` valor final: **60** (consistente con pyproject.toml).

## Files Created/Modified

### Created
- `tests/infra/test_ci_matrix.py` — 10 tests de regresion sobre ci.yml (matriz, cov-fail-under, smoke job, postgres service, invariantes, secrets continue-on-error).
- `tests/infra/test_makefile_devex.py` — 9 tests de regresion sobre Makefile (5 targets DevEx + conservacion backup + invariante mcp en up/dev + PHONY + cov-fail-under).

### Modified
- `.github/workflows/ci.yml` — Reescrito. 4 jobs → 5 jobs. De 91 lineas a 177 lineas. Todos los cambios documentados en commits `ff27cb1`.
- `Makefile` — `.PHONY` del header ampliado con test, test-docker, lint, format, migrate. Bloque `# ── DevEx (Phase 7 / DEVEX-03) ───` insertado entre `backup` (Phase 6) y `# ── Ayuda ──`.

## make -n exit codes (verificado en host WSL)

```
make -n test     → exit 0 (pytest -q --cov=api --cov=nexo --cov-report=term --cov-fail-under=60)
make -n lint     → exit 0 (ruff check api/ nexo/ ; ruff format --check api/ nexo/ ; mypy api/ nexo/)
make -n format   → exit 0 (ruff check --fix api/ nexo/ ; ruff format api/ nexo/)
make -n migrate  → exit 0 (docker compose exec web python scripts/init_nexo_schema.py)
make -n backup   → exit 0 (bash scripts/backup_nightly.sh)   [Phase 6, intacto]
make -n up  2>&1 | grep -c 'profile mcp'  → 0   [invariante Phase 1 OK]
make -n dev 2>&1 | grep -c 'profile mcp'  → 0   [invariante Phase 1 OK]
```

`make help` lista los 5 targets nuevos con su descripcion inline.

## Decisions Made

- **`--cov-fail-under=60`**: consistente con `pyproject.toml [tool.coverage.report].fail_under = 60`. No se inventa N — se consume el baseline medido en 07-COVERAGE-BASELINE.md. Si en plans futuros se suben tests, se sube N en pyproject.toml y aqui simultaneamente (un solo lugar).
- **Service container postgres en puerto host 5433**: paridad con `NEXO_PG_HOST_PORT` default del compose (docker-compose.yml:18). Permite que `scripts/init_nexo_schema.py` se ejecute en CI con las mismas env vars (`NEXO_PG_HOST=localhost NEXO_PG_PORT=5433`) que un dev con `make nexo-init-dev`.
- **Smoke curl a :8001, NO :8000**: el compose expone `${NEXO_WEB_HOST_PORT:-8001}:8000`. 8000 es el puerto interno del container. El test `test_smoke_job_exists` congela este detalle.
- **Smoke NO arranca mcp**: `docker compose up -d --build db web` explicita los servicios. El servicio `mcp` del compose tiene `profiles: ["mcp"]` → sin `--profile mcp` flag, no arranca. El comentario en el step documenta la intencion.
- **`make migrate` como alias DIRECTO de `nexo-init`**, sin dependency de Make (`migrate: nexo-init`). DEVEX-03 pide literal `make migrate` y un dev nuevo no deberia tener que conocer el legacy alias `nexo-init`. El contenido es identico: `docker compose exec web python scripts/init_nexo_schema.py`.
- **Ruff format --check ahora bloqueante**: el step del ci.yml pre-07-01 tenia `continue-on-error: true` porque el codigo legacy no habia pasado por `ruff format`. 07-01 ejecuto el backfill en commit aislado `fd83a4e`, por tanto ahora se puede eliminar el tolerant flag.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Regex `\b` en test_lint_job_runs_mypy fallaba en fin de string**

- **Found during:** Task 1 (Step C - GREEN verify)
- **Issue:** El test `test_lint_job_runs_mypy` usaba la regex `r"\bmypy\s+api/\s+nexo/\b"`. Python `re` trata `\b` como frontera word↔non-word. Tras `nexo/` viene `\n` (non-word); y `/` ya es non-word. La frontera no existe → la aserción fallaba tras el GREEN correcto del ci.yml (que sí contiene `mypy api/ nexo/`).
- **Fix:** Retirar el `\b` final. La regex queda `r"\bmypy\s+api/\s+nexo/"`; sigue siendo robusta porque `\b` inicial y el literal `api/\s+nexo/` son restrictivos.
- **Files modified:** `tests/infra/test_ci_matrix.py`
- **Verification:** `pytest tests/infra/test_ci_matrix.py -v` → 10/10 PASSED.
- **Committed in:** `ff27cb1` (mismo commit que la creacion del test, no merece commit separado porque el test nunca llego a publicarse en RED con la regex buggy — se corrigio en la misma sesion antes del commit atomic).

**2. [Rule 3 - Blocking] `make` no estaba instalado en el contenedor web**

- **Found during:** Task 2 (RED run)
- **Issue:** `tests/infra/test_makefile_devex.py` usa `subprocess.run(['make', '-n', ...])`. El contenedor `analisis_datos-web-1` (imagen `python:3.11-slim-bookworm`) no tiene `make` instalado, por lo que los tests fallaban con `FileNotFoundError` en lugar de la aserción esperada.
- **Fix:** `docker exec analisis_datos-web-1 apt-get install -y make`. Cambio efímero dentro del contenedor en ejecución, usado solo para validación en este sesión. NO se modifica el `Dockerfile` — en CI real la imagen base tambien es `python:3.11-slim-bookworm`, y los tests corren SIN make instalado (porque el step `pytest` en el job `test` de CI está previsto correr contra el runner Ubuntu, no dentro del contenedor web; Ubuntu 24.04 trae `make` por defecto). Si en algun momento se corre la suite completa dentro del contenedor, se añadira `make` al Dockerfile — no es este plan.
- **Files modified:** ninguno.
- **Verification:** tras el `apt-get install make`, `pytest tests/infra/test_makefile_devex.py -v` → 9/9 PASSED.
- **Committed in:** N/A.

---

**Total deviations:** 2 auto-fixed (1 bug en regex de test, 1 blocking environmental).
**Impact on plan:** Ninguna deviation afecta al alcance de DEVEX-02 o DEVEX-03.

## Issues Encountered

- **61 fallos pre-existentes en `tests/infra/`**: mismos que Plan 07-01 documento (root cause: `test_backup_script.py`, `test_deploy_script.py`, `test_env_prod_example.py`, `test_deploy_lan_doc.py`, `test_caddyfile_prod.py`, `test_compose_override.py` referencian artefactos de Phase 6 que viven en `feature/Mark-III` principal pero NO en el base del worktree = `33e2bcb` + `feature/Mark-II`). No son regresion del plan. Las pruebas `test_ci_matrix.py` + `test_makefile_devex.py` + `test_pre_commit_config.py` + `test_pyproject_config.py` corren a 35/35 GREEN.
- **Container tools gap**: el contenedor web carece de `make` y los workspaces de los agentes. El patron de `docker cp ... && docker exec ... pytest` desde el worktree es el camino real de ejecución, como hizo 07-01.

## Known Stubs

Ninguno. No se añade codigo de app nuevo en este plan; solo configuracion CI + Makefile + tests de regresion.

## Threat Flags

Ninguno nuevo. El threat_model del plan (T-07-06..T-07-10) ya contempla:
- Artifact privado del repo → paths revelables (mitigate, reevaluar en Mark-IV).
- fail-fast: false → gasto extra CI (accept).
- Smoke hang → timeout duro 45x2s + tear down if: always() (mitigate).
- `make migrate` con script idempotente documentado (accept).
- `.env` sembrado en CI desde `.env.example` + NEXO_SECRET_KEY random (mitigate).

## TDD Gate Compliance

Ambos tasks declaran `tdd="true"`. Gate sequence verificada:

### Task 1 (test_ci_matrix.py + ci.yml)
- **RED**: `tests/infra/test_ci_matrix.py` creado con 10 tests; ejecutado contra el ci.yml pre-07-02 (4 jobs, Py 3.11 only) → 8 FAIL, 2 PASS (los 2 asertos que ya pasaban antes: `test_ci_yml_exists_and_parseable` y `test_secrets_job_keeps_continue_on_error`). La premisa de RED se satisface: los aserts nuevos (matriz, gate, smoke, postgres service, ruff-format no tolerant, mypy) fallan antes de la edicion.
- **GREEN**: `ci.yml` reescrito con matriz + gate + smoke + postgres service → 10/10 PASS.
- **REFACTOR**: 1 bug en regex de test corregido inline (detalle en Deviations §1). 10/10 PASS final.
- Commit atomico: `ff27cb1`. RED y GREEN en el mismo commit (patron heredado de 07-01 Task 2).

### Task 2 (test_makefile_devex.py + Makefile)
- **RED**: `tests/infra/test_makefile_devex.py` creado con 9 tests; ejecutado contra el Makefile pre-07-02 → 6 FAIL (los 4 targets nuevos + PHONY + cov-fail-under), 3 PASS (backup, up-no-mcp, dev-no-mcp, que ya se satisfacian antes).
- **GREEN**: Makefile editado con el bloque DevEx + PHONY ampliado → 9/9 PASS.
- **REFACTOR**: no necesario.
- Commit atomico: `d06c8ae`.

Ambos (test + feat) agrupados por Task en commits atomicos para evitar publicar tests RED en history.

## User Setup Required

Ninguno. Los hooks del pre-commit siguen siendo responsabilidad del dev (documentado en 07-04 segun 07-01 SUMMARY). Nada nuevo aqui.

Para correr los nuevos Makefile targets, el dev necesita:
- `pip install -r requirements-dev.txt` (ruff, mypy, pytest-cov) — ya activo desde 07-01.
- `docker compose up -d` para que `make migrate` y `make test-docker` funcionen.

Para que el CI corra, requiere:
- El runner `ubuntu-24.04` ya trae `make`, `docker`, `docker compose`, `openssl`, `curl` por defecto. No hay deps externas nuevas.

## Next Plan Readiness

**Listo para Plan 07-03 (runbook-release-claude-md)**:
- `pyproject.toml`, `.pre-commit-config.yaml`, `ci.yml`, `Makefile` ya son fuentes de verdad coherentes. Plan 07-03 puede documentar el dev loop literalmente replicando los comandos: `make lint`, `make test`, `make format`, `make migrate`.
- Los tests de regresion (35 tests en tests/infra/ para configs) blindan que la documentacion no se desalinee del codigo: si alguien cambia `pyproject.toml` o `ci.yml` sin actualizar docs, los tests que congelan el contrato fallan antes del merge.
- No hay blockers para 07-03 ni para 07-04 (CLAUDE.md + docs).

## Self-Check

**Archivos creados presentes:**
- `tests/infra/test_ci_matrix.py` → FOUND
- `tests/infra/test_makefile_devex.py` → FOUND

**Archivos modificados presentes:**
- `.github/workflows/ci.yml` → FOUND (177 lineas, 5 jobs, matriz + gate + smoke + postgres service)
- `Makefile` → FOUND (PHONY ampliado + bloque DevEx con 5 targets)

**Commits en git log (este worktree):**
- `ff27cb1` (Task 1 ci matrix + smoke) → FOUND
- `d06c8ae` (Task 2 Makefile DevEx) → FOUND

**Verificaciones automatizadas (phase-level checks del PLAN.md):**
- `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"` → OK
- `grep -cE 'python-version.*3\.11' .github/workflows/ci.yml` → 2 (>=2 requerido)
- `grep -cE 'python-version.*3\.12' .github/workflows/ci.yml` → 2 (>=2 requerido)
- `grep -qE 'cov-fail-under=[0-9]+' .github/workflows/ci.yml` → exit 0
- `grep -qE '^\s+smoke:' .github/workflows/ci.yml` → exit 0
- `grep -q 'curl -fs http://localhost:8001/api/health' .github/workflows/ci.yml` → exit 0
- `grep -q 'postgres:16-alpine' .github/workflows/ci.yml` → exit 0
- `grep -q 'mypy api/ nexo/' .github/workflows/ci.yml` → exit 0
- `python3 -c "import yaml; d=yaml.safe_load(open('.github/workflows/ci.yml')); assert d['jobs']['secrets']['continue-on-error'] is True"` → OK
- `make -n test` → exit 0
- `make -n lint` → exit 0
- `make -n format` → exit 0
- `make -n migrate` → exit 0
- `make -n backup` → exit 0
- `make -n up  2>&1 | grep -c 'profile mcp'` → 0 (invariante Phase 1 OK)
- `make -n dev 2>&1 | grep -c 'profile mcp'` → 0 (invariante Phase 1 OK)
- `pytest tests/infra/test_ci_matrix.py tests/infra/test_makefile_devex.py tests/infra/test_pre_commit_config.py tests/infra/test_pyproject_config.py -q` (contenedor) → **35 passed**

## Self-Check: PASSED

---
*Phase: 07-devex-hardening*
*Completed: 2026-04-22*
