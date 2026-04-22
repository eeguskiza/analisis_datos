---
phase: 07-devex-hardening
plan: 01
subsystem: infra
tags:
  - devex
  - pre-commit
  - ruff
  - mypy
  - coverage
  - pytest
  - tooling

# Dependency graph
requires:
  - phase: 06-deploy-lan
    provides: "Infra foundation (Makefile targets, compose overrides) sobre la que DEVEX-03 extiende."
provides:
  - "pyproject.toml consolidando config de ruff + mypy + pytest + coverage."
  - ".pre-commit-config.yaml con hooks scoped a ^(api|nexo)/ (ruff-check + ruff-format + mypy)."
  - "requirements-dev.txt con ruff 0.15.11 + mypy 1.13.0 + pre-commit 4.0.1."
  - "16 tests de regresion sobre pyproject.toml + .pre-commit-config.yaml."
  - "Backfill commit aislado con format + auto-fix sobre 65 archivos api/+nexo/."
  - "Coverage baseline documentado: 60% (gate fijado en 60)."
affects:
  - 07-02-ci-coverage-gate
  - 07-03-makefile-devex
  - 07-04-runbook-release-claude-md

# Tech tracking
tech-stack:
  added:
    - "ruff 0.15.11 (bump desde 0.9.8)"
    - "mypy 1.13.0"
    - "pre-commit 4.0.1"
  patterns:
    - "tooling config consolidado en pyproject.toml (single source of truth)"
    - "hooks pre-commit scoped via `files: ^(api|nexo)/` por hook"
    - "fail_under gate locked al baseline medido, no a cifra aspiracional"
    - "tests de regresion congelan contrato de configs (anti-drift)"

key-files:
  created:
    - "pyproject.toml"
    - ".pre-commit-config.yaml"
    - "tests/infra/test_pre_commit_config.py"
    - "tests/infra/test_pyproject_config.py"
    - ".planning/phases/07-devex-hardening/07-COVERAGE-BASELINE.md"
  modified:
    - "requirements-dev.txt"
    - ".planning/REQUIREMENTS.md"
    - "api/**/*.py (65 files reformatted)"
    - "nexo/**/*.py (65 files reformatted)"

key-decisions:
  - "ruff-format sustituye a black — drop-in replacement, elimina friccion dual-format."
  - "fail_under=60 locked al baseline exacto medido, no a target aspiracional."
  - "3 F841 residuales del backfill se ignoran via per-file-ignores (historial.py, metrics.py) en lugar de refactorizar codigo fuera de scope Mark-III."
  - "E402 (api/main.py:327) se ignora globalmente: el handler se registra al final intencionalmente."
  - "Backfill en commit aislado (fd83a4e) antes de activar hooks, para que diffs futuros sean puramente semanticos."

patterns-established:
  - "Pre-commit scope pattern: cada hook language-specific declara `files: ^(api|nexo)/` individualmente; los hooks de higiene (trailing-whitespace, etc.) corren sobre todo el repo."
  - "Infra tests pattern: tests en `tests/infra/test_*_config.py` que parsean (yaml/tomllib) y hacen asserts explicitos sobre claves criticas — estilo identico a test_deploy_lan_doc.py de Phase 6."
  - "Coverage gate pattern: fail_under se locked al baseline real para evitar falsos greens. Se sube en plans sucesivos cuando se anaden tests, no antes."

requirements-completed:
  - DEVEX-01

# Metrics
duration: ~25min
completed: 2026-04-22
---

# Phase 07 Plan 01: DevEx tooling foundation Summary

**Consolidacion de tooling Python en pyproject.toml + pre-commit con hooks ruff/mypy scoped a api/+nexo/, bump ruff 0.9.8->0.15.11 + mypy 1.13.0 + pre-commit 4.0.1, backfill aislado sobre 65 archivos, y coverage gate locked en el baseline medido (60%).**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-04-22T15:38:00Z (aprox)
- **Completed:** 2026-04-22T16:03:00Z (aprox)
- **Tasks:** 3 (Task 0 baseline + Task 1 backfill + Task 2 configs+tests)
- **Files modified:** 71 (2 nuevas configs + 2 nuevos tests + 1 baseline doc + 65 archivos reformateados + 2 deps files + REQUIREMENTS)

## Accomplishments

- **Coverage baseline medido y documentado**: 60% TOTAL sobre `api/` + `nexo/` contra el contenedor `analisis_datos-web-1`. Decision locked: `fail_under=60` en pyproject.toml + `--cov-fail-under=60` en CI (para Plan 07-02).
- **Backfill aislado ejecutado en commit atomico**: 65 archivos de `api/` + `nexo/` reformateados con ruff-format 0.15.11, 11 auto-fixes aplicados por ruff-check, 3 F841 residuales identificados (gestionados via per-file-ignores). Diff 100% estilo, sin cambio de semantica. Tests funcionales sin regresion.
- **Configs consolidadas**: pyproject.toml con [tool.ruff] + [tool.mypy] + [tool.pytest.ini_options] + [tool.coverage.run] + [tool.coverage.report]. .pre-commit-config.yaml con 9 hooks totales (6 higiene + 2 ruff scoped + 1 mypy scoped).
- **Migracion black -> ruff-format documentada**: REQUIREMENTS.md DEVEX-01 tiene nota trailing 2026-04-21 con rationale.
- **16 tests de regresion** sobre las configs (7 pre-commit + 9 pyproject), todos verdes. Patron replicable para futuras configs.

## Task Commits

Cada task committed atomicamente en el branch `worktree-agent-abcb517c`:

1. **Task 0: Medir baseline cobertura** — `91617fa` (docs) → `.planning/phases/07-devex-hardening/07-COVERAGE-BASELINE.md`
2. **Task 1: Backfill ruff format + fix api/+nexo/** — `fd83a4e` (chore) → 65 archivos, 1792 insertions / 986 deletions
3. **Task 2: Configs + tests + bump deps** — `10f1627` (feat) → pyproject.toml + .pre-commit-config.yaml + requirements-dev.txt + 2 tests infra + REQUIREMENTS.md

## Files Created/Modified

### Created
- `pyproject.toml` — Config de tooling Python (ruff + mypy + pytest + coverage). 108 lineas, [tool.ruff/mypy/pytest/coverage.*] cubiertos.
- `.pre-commit-config.yaml` — Hooks pre-commit con scope ^(api|nexo)/. 52 lineas.
- `tests/infra/test_pre_commit_config.py` — 7 tests de regresion sobre .pre-commit-config.yaml.
- `tests/infra/test_pyproject_config.py` — 9 tests de regresion sobre pyproject.toml.
- `.planning/phases/07-devex-hardening/07-COVERAGE-BASELINE.md` — Baseline 60% + decision fail_under + top 10 modulos de menor cobertura.

### Modified
- `requirements-dev.txt` — bump ruff 0.9.8 -> 0.15.11, add mypy==1.13.0, add pre-commit==4.0.1.
- `.planning/REQUIREMENTS.md` — nota trailing DEVEX-01 con migracion black -> ruff-format.
- `api/**/*.py` + `nexo/**/*.py` — 65 archivos reformateados + 11 imports/variables auto-fixed.

## Decisions Made

- **Ruff-format sustituye a black (decision locked por planner en 07-RESEARCH.md, ejecutada aqui)**. Rationale: ruff-format es drop-in replacement de black, reduce un hook, evita fight entre formatters. DEVEX-01 dice literalmente "ruff + black" pero la interpretacion moderna 2026 es "ruff-check + ruff-format + mypy". Nota documentada en REQUIREMENTS.md.
- **fail_under=60**: baseline medido = 60% exacto → gate fijado en el piso medido para evitar falsos greens. No se sube a 80% (target aspiracional historico) hasta que plans futuros anadan tests.
- **3 F841 residuales tratados con per-file-ignores**: `api/routers/historial.py:67` (`dtos`) y `api/services/metrics.py:318` (`keys`) son variables asignadas en bloques WIP/comentados. En lugar de borrar codigo que el owner del modulo puede querer reactivar, se ignoran via `[tool.ruff.lint.per-file-ignores]`. Documentado inline en pyproject.toml.
- **E402 (api/main.py:327) ignorado globalmente**: el handler del exception dispatcher se registra al final del archivo intencionalmente (patron de registro tardio de Phase 5). Se ignora a nivel global en lugar de file-specific.
- **pre-commit-hooks `check-added-large-files` en 1024kb**: maxkb agresivo para impedir que matplotlib PDFs o PNGs de informes escapen al historial.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Residual F841 requerian per-file-ignores**
- **Found during:** Task 1 (Backfill)
- **Issue:** Tras `ruff check --fix`, 3 errores E402/F841 residuales bloqueaban el green de `ruff check` (requisito implicito de hooks futuros). El plan original no especificaba como tratar errores no auto-fixables.
- **Fix:** En Task 2 se anadieron:
  - `ignore = ["E402"]` global en `[tool.ruff.lint]` (justificado: api/main.py:327 es registro tardio intencional).
  - `[tool.ruff.lint.per-file-ignores]` con `"api/routers/historial.py" = ["F841"]` y `"api/services/metrics.py" = ["F841"]`.
- **Files modified:** `pyproject.toml`
- **Verification:** `ruff check api/ nexo/` ya no reporta errores tras aplicar la config de pyproject.toml. Los 16 tests de regresion pasan.
- **Committed in:** `10f1627` (Task 2 commit)

**2. [Rule 3 - Blocking] pre-commit no disponible localmente para validate-config**
- **Found during:** Task 2 (Step F — pre-commit install)
- **Issue:** El entorno local (WSL) no tiene `pip` instalado, impide `pip install -r requirements-dev.txt` + `pre-commit install` local.
- **Fix:** La validacion se ejecuto dentro del contenedor `analisis_datos-web-1`: `docker exec ... pip install pre-commit==4.0.1 && pre-commit validate-config` → exit 0. La instalacion local del hook (`pre-commit install` en el .git/hooks/) es responsabilidad del dev con pip funcional; se documentara en CLAUDE.md/README en Plan 07-04.
- **Files modified:** ninguno (solo validacion)
- **Verification:** `pre-commit validate-config` exit 0 dentro del contenedor.
- **Committed in:** N/A (no afecta archivos)

---

**Total deviations:** 2 auto-fixed (1 missing critical config, 1 blocking env).
**Impact on plan:** Ninguna deviation afecta al alcance. La deviation 1 es correctness (gate ruff verdor), la deviation 2 es entorno (la activacion real del hook es responsabilidad del dev con pip local).

## Issues Encountered

- **64 fallos pre-existentes en pytest**: cuando se corrio la suite completa tras el bump, 64 tests de infra fallaron con `FileNotFoundError`. Root cause: tests en `tests/infra/test_backup_script.py`, `test_deploy_script.py`, `test_env_prod_example.py`, `test_deploy_lan_doc.py`, etc., referencian artefactos creados en Phase 6 que viven en el branch `feature/Mark-III` principal (commit f07e80e) pero NO en el base de este worktree (commit `6e9109b`). Ver STATE del orchestrator: el worktree fue correctamente re-baseado a `6e9109b` al inicio (`worktree_branch_check`). Los tests que fallan no son regresion del bump — fallaban igual antes del Task 1 (verificado con `git stash` + rerun).
- **Coverage en el contenedor**: el contenedor `analisis_datos-web-1` en ejecucion monta la carpeta MAIN (/home/eeguskiza/analisis_datos), no este worktree. Para medir cobertura del worktree se uso un contenedor efimero con `docker run --rm -v <worktree>:/app`. Para validar pre-commit config se usa `docker cp <worktree>/pyproject.toml analisis_datos-web-1:/app/...` + `docker exec analisis_datos-web-1 pytest`. Documentado en 07-COVERAGE-BASELINE.md.

## Known Stubs

Ninguno. No se anade codigo app nuevo en este plan; solo tooling y tests.

## TDD Gate Compliance

Este plan no es `type: tdd` a nivel de plan, pero el Task 2 declara `tdd="true"` a nivel de task. Gate sequence:

- RED: `tests/infra/test_pre_commit_config.py` + `tests/infra/test_pyproject_config.py` creados y corridos con 16/16 fallos esperados (los configs aun no existian).
- GREEN: `pyproject.toml` + `.pre-commit-config.yaml` creados con la estructura que satisface los 16 asserts. Rerun → 16/16 pass.
- REFACTOR: no necesario; configs nacieron limpios.

Ambos (test + feat) se agruparon en un solo commit (`10f1627`) por atomicidad (anadir la config sin tests seria publicar una config sin verificacion; anadir tests sin la config seria RED publicado en history). Documentado aqui para trazabilidad.

## User Setup Required

None - no external service configuration required. Los devs que clonen tras este plan deben correr:

```bash
pip install -r requirements-dev.txt   # instala ruff/mypy/pre-commit
pre-commit install                     # activa hooks locales
```

Esa activacion se documentara formalmente en CLAUDE.md en Plan 07-04 (DEVEX-07).

## Next Phase Readiness

**Listo para Plan 07-02 (CI coverage gate + matriz 3.11/3.12 + smoke)**:
- `pyproject.toml` [tool.coverage.report].fail_under = 60 ya esta presente → `--cov-fail-under=60` en CI consume ese valor por inercia.
- `requirements-dev.txt` tiene ruff/mypy/pre-commit pinned → CI instala con `pip install -r requirements-dev.txt`.
- No hay blockers para 07-02, 07-03, 07-04 o el resto de Phase 7.

**Concerns menores para plans siguientes**:
- Plan 07-04 debe documentar `pre-commit install` como paso obligatorio en el onboarding (CLAUDE.md section "Setup dev").
- Si el coverage cae por debajo de 60% en PRs futuros, el gate en CI rebotara — se espera que sea un incentivo a escribir tests antes que un bloqueo.

## Self-Check

Verificacion de claims antes de cerrar:

**Archivos creados presentes:**
- `pyproject.toml` → FOUND
- `.pre-commit-config.yaml` → FOUND
- `tests/infra/test_pre_commit_config.py` → FOUND
- `tests/infra/test_pyproject_config.py` → FOUND
- `.planning/phases/07-devex-hardening/07-COVERAGE-BASELINE.md` → FOUND

**Commits en git log:**
- `91617fa` (Task 0 baseline) → FOUND
- `fd83a4e` (Task 1 backfill) → FOUND
- `10f1627` (Task 2 configs + tests) → FOUND

**Verificaciones automatizadas (phase-level checks del PLAN.md):**
- `test -f pyproject.toml` → exit 0
- `test -f .pre-commit-config.yaml` → exit 0
- `python3 -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"` → OK
- `python3 -c "import yaml; yaml.safe_load(open('.pre-commit-config.yaml'))"` → OK
- `grep -q 'ruff-format' .pre-commit-config.yaml` → exit 0
- `! grep -q 'psf/black' .pre-commit-config.yaml` → exit 0
- `grep -c 'files: \^(api|nexo)/' .pre-commit-config.yaml` → 3 (>=3 requerido)
- `grep -q 'ruff==0.15.11' requirements-dev.txt` → exit 0
- `grep -q 'mypy==1.13.0' requirements-dev.txt` → exit 0
- `grep -q 'pre-commit==4.0.1' requirements-dev.txt` → exit 0
- `git log --oneline -20 | grep -qE 'chore\(07\): backfill ruff format \+ fix on api/ \+ nexo/'` → exit 0
- `grep -A3 'DEVEX-01' .planning/REQUIREMENTS.md | grep -q 'black reemplazado por ruff-format'` → exit 0
- `pytest tests/infra/test_pre_commit_config.py tests/infra/test_pyproject_config.py -v` → 16 passed
- `pre-commit validate-config` → exit 0

## Self-Check: PASSED

---
*Phase: 07-devex-hardening*
*Completed: 2026-04-22*
