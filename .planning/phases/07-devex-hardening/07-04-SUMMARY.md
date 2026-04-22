---
phase: 07-devex-hardening
plan: 04
subsystem: devex
tags:
  - devex
  - claude-md
  - conventions
  - phase-wrap-up
  - closing

# Dependency graph
requires:
  - phase: 07-devex-hardening
    plan: 01
    provides: "pyproject.toml `--cov-fail-under=60` + requirements-dev.txt con pre-commit/mypy/ruff; referenciados literalmente en la nueva seccion 'Tooling DevEx' de CLAUDE.md."
  - phase: 07-devex-hardening
    plan: 02
    provides: "Matriz CI Python 3.11+3.12 + 5 jobs (lint/test/smoke/build/secrets) + Makefile DevEx (test/lint/format/migrate/backup); referenciados en la tabla Tooling DevEx + resumen de CI jobs."
  - phase: 07-devex-hardening
    plan: 03
    provides: "Quartet ARCHITECTURE.md + RUNBOOK.md + RELEASE.md + CHANGELOG.md; referenciados en Fuente de verdad + Tooling DevEx referencias de navegacion."
provides:
  - "CLAUDE.md actualizado tras Mark-III / Sprint 6 / Phase 7 con 2 secciones nuevas, 5 docs en 'Fuente de verdad', 2 prohibiciones en 'Que NO hacer'."
  - "tests/infra/test_claude_md.py con 13 tests de regresion sobre las 5 invariantes criticas del doc (T-07-16 del threat model)."
  - "Cierre formal de DEVEX-07 (ultimo requirement de Phase 7)."
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "CLAUDE.md extension pattern: anadir secciones sin reescribir (preserva decisiones previas literalmente). Las secciones nuevas van entre 'Flujo de trabajo GSD' y 'Que NO hacer'."
    - "Regression test heredado del patron test_deploy_lan_doc.py (Phase 6) + test_devex_docs.py (07-03): pathlib + regex sobre texto + asserts con mensaje claro."
    - "Aislamiento de seccion via regex `(?ms)^##\\s+<titulo>.*?(?=^##\\s|\\Z)` para enforzar que una regla aparezca en una seccion concreta, no en cualquier lugar del doc."

key-files:
  created:
    - "tests/infra/test_claude_md.py (182 lineas, 13 tests)"
  modified:
    - "CLAUDE.md (151 -> 235 lineas, delta +84; 2 secciones nuevas + 5 docs anadidos + 2 prohibiciones)"

key-decisions:
  - "Fecha 'Ultima revision': 2026-04-22 (dia de ejecucion, no 2026-MM-DD literal). Marcador 'cierre Mark-III / Sprint 6 / Phase 7'."
  - "Las 2 secciones nuevas ('Tooling DevEx' y 'Despliegue productivo') se insertan entre 'Flujo de trabajo GSD' y 'Que NO hacer' — orden narrativo: primero dev loop (Phase 7), despues deploy (Phase 6), despues invariantes. Esto contradice el orden historico de phases pero encaja con la manera en que un dev lee el doc (dev loop local primero, deploy despues)."
  - "5 docs nuevos en 'Fuente de verdad del plan' (ARCHITECTURE + RUNBOOK + RELEASE + DEPLOY_LAN + CHANGELOG) — DEPLOY_LAN.md se incluye aunque el plan solo pedia 4 porque ya era artefacto Phase 6 y no estaba listado, inconsistencia que se arregla de paso."
  - "Test de aislamiento de seccion: test_que_no_hacer_has_no_verify_rule usa regex de seccion para distinguir la entrada NUEVA en 'Que NO hacer' del bullet preexistente en 'Politica de commits'. Previene que alguien elimine la regla fuerte del listado de prohibiciones y deje solo la version suave de 'salvo decision explicita'."
  - "13 tests (superado el minimo de 12 requerido) — cubren las 5 invariantes criticas + 2 tests de integridad (no_emojis + preserves_existing_invariants) + 6 tests granulares dentro de las secciones nuevas. Ventana adicional contra relajacion silenciosa de reglas."
  - "Commit atomico unico (test + feat juntos) siguiendo patron heredado de 07-01 Task 2 y 07-02 Task 1/2 — evita publicar RED en history y mantiene atomicidad del change 'anadir regla + test que la congela'."

patterns-established:
  - "CLAUDE.md regression test: si alguien relaja una regla critica (borra la prohibicion de --no-verify, quita la cobertura 60%, desactualiza la fecha de 'Ultima revision'), el test correspondiente falla en CI y el PR se bloquea antes del merge."
  - "Seccion isolation regex `(?ms)^##\\s+<titulo>\\b(.*?)(?=^##\\s|\\Z)` — util para tests de documentos estructurados donde la ubicacion de una regla en una seccion concreta es significante."

requirements-completed:
  - DEVEX-07

# Metrics
duration: ~8min
completed: 2026-04-22
---

# Phase 07 Plan 04: CLAUDE.md + regression test — Summary

**Cierre formal de Phase 7 (DevEx hardening) y de Mark-III / Sprint 6. `CLAUDE.md` crece de 151 a 235 lineas (+84) con 2 secciones nuevas ('Tooling DevEx (Phase 7)' + 'Despliegue productivo (Phase 6)'), 5 docs anadidos al listado 'Fuente de verdad del plan' (ARCHITECTURE + RUNBOOK + RELEASE + DEPLOY_LAN + CHANGELOG), y 2 prohibiciones nuevas en 'Que NO hacer' (--no-verify + cobertura <60%). Blindado por 13 tests de regresion en `tests/infra/test_claude_md.py` que congelan las 5 invariantes criticas (T-07-16 del threat model).**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-04-22 (worktree agent-ae58bef0)
- **Completed:** 2026-04-22
- **Tasks:** 1 (Task 1: CLAUDE.md update + test de regresion)
- **Files changed:** 1 modified (CLAUDE.md) + 1 created (tests/infra/test_claude_md.py)
- **Tests anadidos:** 13

## Accomplishments

- **CLAUDE.md "Ultima revision" bumpeada** de `2026-04-18 (Sprint 0)` a `2026-04-22 (cierre Mark-III / Sprint 6 / Phase 7 — DevEx hardening + quartet docs + CI coverage gate + pre-commit activo)`.
- **Seccion nueva "## Tooling DevEx (Phase 7)"** entre "Flujo de trabajo GSD" y "Que NO hacer". Cubre: comandos de onboarding (`pip install -r requirements-dev.txt && pre-commit install`), tabla Makefile con 5 targets (test/lint/format/migrate/backup), resumen de los 5 jobs de CI con coverage gate 60% bloqueante, y referencias a ARCHITECTURE.md + RUNBOOK.md + RELEASE.md + CHANGELOG.md + .pre-commit-config.yaml + pyproject.toml.
- **Seccion nueva "## Despliegue productivo (Phase 6)"**. Cubre: stack LAN HTTPS con Caddy `tls internal`, hostname `nexo.ecsmobility.local` + hosts-file + root CA, tabla Makefile con 7 targets `make prod-*` + `make deploy` + `make backup`, referencia al runbook `docs/DEPLOY_LAN.md`, y al smoke externo `tests/infra/deploy_smoke.sh`.
- **Seccion "Fuente de verdad del plan" ampliada** con 5 bullets nuevos (el plan pedia 4; DEPLOY_LAN.md se anade porque era un artefacto Phase 6 que no estaba en el listado — de paso).
- **Seccion "Que NO hacer" ampliada** con 2 bullets nuevos al final: prohibicion explicita de `git commit --no-verify` + prohibicion de bajar cobertura por debajo del gate (60%). Ambos referencian los archivos (`pyproject.toml`, `.github/workflows/ci.yml`) donde el gate vive.
- **13 tests de regresion** en `tests/infra/test_claude_md.py` pasando en container (`analisis_datos-web-1`). Cubre: existencia, fecha bumpeada, 3 asserts sobre seccion Tooling DevEx (heading + pre-commit + make lint/format), cross-links quartet, 2 asserts sobre seccion Despliegue productivo (heading + DEPLOY_LAN.md), 2 prohibiciones nuevas (--no-verify dentro de la seccion "Que NO hacer" + cobertura 60% en cualquier parte del doc), 4 docs nuevos en Fuente de verdad, no-emojis, y preservacion de 4 invariantes pre-existentes (make up/mcp, OEE/, credenciales SQL Server, filter-repo).

## Delta medido

| Metrica | Antes | Despues | Delta |
|---------|-------|---------|-------|
| Lineas CLAUDE.md | 151 | 235 | **+84** (dentro del rango esperado +80 a +120) |
| Lineas tests/infra/test_claude_md.py | 0 | 182 | **+182** (test nuevo) |
| Tests en tests/infra/test_claude_md.py | 0 | 13 | **+13** |
| Secciones `## <heading>` en CLAUDE.md | 8 | 10 | **+2** (Tooling DevEx + Despliegue productivo) |
| Items en "Fuente de verdad del plan" | 8 docs | 13 docs | **+5** |
| Items en "Que NO hacer" | 10 bullets | 12 bullets | **+2** |

## Task Commits

1. **Task 1: CLAUDE.md update + regression test** — `8a22d15` (docs) → `CLAUDE.md` + `tests/infra/test_claude_md.py`. Atomic commit combinando test + feat (patron heredado de 07-01 Task 2 y 07-02 Task 1/2).

## Files Created/Modified

### Created

- `tests/infra/test_claude_md.py` (182 lineas, 13 tests): regresion sobre CLAUDE.md. Tests organizados por invariante con comentarios explicitos de justificacion (T-07-16 threat model). Patron pathlib + regex heredado de `test_deploy_lan_doc.py` (Phase 6) y `test_devex_docs.py` (Phase 7 / Plan 03).

### Modified

- `CLAUDE.md` (151 -> 235 lineas): +84 lineas. Cambios:
  1. Linea 7: fecha `Ultima revision` bumpeada.
  2. Lineas 37-41: 5 bullets anadidos en "Fuente de verdad del plan" (ARCHITECTURE/RUNBOOK/RELEASE/DEPLOY_LAN/CHANGELOG).
  3. Lineas 143-217: 2 secciones nuevas completas (Tooling DevEx + Despliegue productivo).
  4. Lineas 234-235: 2 bullets nuevos al final de "Que NO hacer".

**NO modificado** (preserva decisiones previas literal):
- Seccion "Que es Nexo" (linea 12-21)
- Seccion "Regeneracion de .planning/" (linea 52-68)
- Seccion "Decisiones cerradas de Mark-III" (linea 72-93)
- Seccion "Convenciones de naming" (linea 97-110)
- Seccion "Politica de commits" (linea 114-122) — incluido el bullet "No se aplica --no-verify salvo decision explicita" que sigue como politica de flujo; la prohibicion HARD adicional vive en "Que NO hacer".
- Seccion "Flujo de trabajo GSD" (linea 126-139)
- Los 10 bullets originales de "Que NO hacer" (linea 224-233)

## Decisiones Mark-III pre-existentes — lista de preservacion literal

Confirmacion explicita de que las decisiones siguientes NO fueron modificadas (verificado por `test_preserves_existing_invariants` + inspeccion manual):

- [x] **Producto "Nexo"** sin acento, capitalizado (Convenciones de naming bullet 1).
- [x] **Env vars `NEXO_*`** con compat `OEE_*` en Mark-III (Convenciones de naming bullet 3).
- [x] **SQL Server split** `NEXO_MES_*` (dbizaro read-only) + `NEXO_APP_*` (ecs_mobility), mismo host hoy (Decisiones cerradas bullet 1).
- [x] **Carpeta `OEE/`** no se renombra en Mark-III (Decisiones cerradas bullet 2 + Que NO hacer bullet 4).
- [x] **Repo GitHub** no se renombra (Decisiones cerradas bullet 3 + Que NO hacer bullet 5).
- [x] **Credenciales SQL Server** no se rotan en Sprint 0 (Decisiones cerradas bullet 4 + Que NO hacer bullet 1).
- [x] **Historial git / filter-repo** no se ejecuta sin autorizacion (Decisiones cerradas bullet 5 + Que NO hacer bullet 2).
- [x] **Postgres** casa de `nexo.*` / `cfg.*` se queda en `ecs_mobility` (Decisiones cerradas bullet 6 + Que NO hacer bullet 9).
- [x] **MCP `profiles: ["mcp"]`** — `make up` y `make dev` NO arrancan mcp (Decisiones cerradas bullet 7).
- [x] **Auth Sprint 1** (roles propietario/directivo/usuario + 5 dept + argon2id + lock 5/15 min) (Decisiones cerradas bullet 8 completo).
- [x] **Exposicion a internet descartada** — LAN-only (Decisiones cerradas bullet 10).
- [x] **No refactorizar OEE/**, **no SMTP**, **no 2FA/LDAP**, **no matplotlib replace** (Que NO hacer bullets 3, 6, 8, 10).
- [x] **Politica de commits** intacta: Conventional Commits, atomicidad, branch `feature/Mark-III`, `--no-verify` salvo decision explicita, no `git push --force`.
- [x] **Flujo GSD** intacto: `/gsd-progress`, `/gsd-plan-phase`, `/gsd-execute-phase`, `/gsd-verify-work`, `/gsd-map-codebase`; regla de oro `docs/` ↔ `.planning/` sync.

## Count cumulativo de tests/infra/ tras Phase 7 completo

Estimacion segun el plan: ~128. Real medido en container post-sync:

```
pytest tests/infra/ --no-cov -q
-> 135 passed, 5 skipped in 0.15s
```

Desglose:
- Phase 6 (tests Wave 0/2): ~73 tests — congelan compose overrides, Caddyfile, deploy.sh, backup_nightly.sh, env_prod_example, deploy_lan_doc.
- Plan 07-01: 16 tests (7 pre_commit + 9 pyproject).
- Plan 07-02: 19 tests (10 ci_matrix + 9 makefile_devex).
- Plan 07-03: 19 tests (devex_docs, cubre ARCHITECTURE + RUNBOOK + RELEASE + CHANGELOG).
- Plan 07-04: 13 tests (claude_md).
- **Total: 140 tests** (135 pass + 5 skipped por deps opcionales como `make` no disponible en container base sin apt install).

Nota: el count real (135+5=140) supera ligeramente la estimacion del plan (128) porque los tests de Plan 07-02 fueron 19 (no 11-12 como la estimacion inicial), y los de Plan 07-03 fueron 19 (no 12 minimo). Ambos casos documentados en sus respectivos SUMMARY.

## Decisions Made

- **Fecha "Ultima revision" = 2026-04-22** (fecha de ejecucion real). No se usa el literal `2026-MM-DD` del plan porque el plan espera que el executor ponga la fecha concreta.
- **Orden de las nuevas secciones**: Tooling DevEx ANTES de Despliegue productivo. Rationale: un dev que clona el repo pasa por el dev loop local (Tooling) antes de tocar prod (Despliegue). Esto contradice el orden cronologico de phases (Phase 6 antes de Phase 7) pero encaja mejor con el flujo mental del lector.
- **DEPLOY_LAN.md se anade a "Fuente de verdad"**: el plan solo pedia los 4 docs del quartet (ARCHITECTURE + RUNBOOK + RELEASE + CHANGELOG), pero DEPLOY_LAN.md ya era un artefacto Phase 6 critico que nunca habia aparecido en el listado. Se incluye como fix de paso (1 bullet extra, sin impacto).
- **Test de aislamiento de seccion** para la prohibicion --no-verify: el bullet pre-existente en "Politica de commits" ("No se aplica --no-verify salvo decision explicita") ya mencionaba "no-verify", lo que haria pasar un test `grep -q 'no-verify'` trivialmente. La entrada NUEVA en "Que NO hacer" es una regla HARD diferente ("los hooks no son opcionales"). El test `test_que_no_hacer_has_no_verify_rule` usa regex para aislar la seccion y enforza que la regla aparezca AHI, no solo en cualquier parte del doc. Previene regresion silenciosa.
- **Commit atomico unico** test + feat: el test no tiene vida propia sin la edicion de CLAUDE.md (los 10 de 13 tests fallarian antes del GREEN). Dividir en 2 commits publicaria RED en history. Patron heredado de 07-01 Task 2 y 07-02 Task 1/2.
- **No se usa `git commit --no-verify` localmente** en este commit: el worktree no tiene pre-commit instalado (Phase 7 es docs + tooling config; la activacion local del hook es responsabilidad del dev segun 07-01 SUMMARY). Pasa el flag `--no-verify` al commit por parity con 07-03 y por requisito explicito del orchestrator (parallel_execution).
- **`OEE/` no se testea directamente**: el test `test_preserves_existing_invariants` solo busca el literal "carpeta \`OEE/\`" como representante de la decision. Si alguien cambia el texto a "carpeta `OEE`" (sin slash) el test falla — rigidez intencional para detectar reescrituras minimalistas sin intencion de regresionar.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Test `test_que_no_hacer_has_no_verify_rule` inicial era insuficiente**

- **Found during:** Task 1 (Step A RED run inicial)
- **Issue:** El primer diseno del test usaba `grep -q 'no-verify' CLAUDE.md` (literalmente lo que pide el PLAN `grep -q 'no-verify' CLAUDE.md (en seccion Que NO hacer)`). El assert pasaba en RED ANTES de la edicion porque el literal `--no-verify` ya existia en la seccion "Politica de commits" con la politica suave ("salvo decision explicita"). El test entonces no distinguiria entre la entrada NUEVA en "Que NO hacer" (regla HARD) y el bullet preexistente en "Politica de commits". Riesgo: alguien borra la nueva prohibicion y el test sigue verde porque la vieja sigue ahi.
- **Fix:** Reemplazar el assert trivial por un aislamiento de seccion con regex `(?ms)^##\s+Qu[éeÉE]\s+NO\s+hacer\b(.*?)(?=^##\s|\Z)` que extrae el bloque y busca `no-verify` ahi dentro. Test RED confirmado tras el cambio, pasa verde tras la edicion.
- **Files modified:** `tests/infra/test_claude_md.py` (inline, mismo commit atomic).
- **Verification:** RED: 10/13 failed (esperado); GREEN: 13/13 passed.
- **Committed in:** `8a22d15` (mismo commit, Task 1 atomic).

---

**Total deviations:** 1 auto-fixed (test robustness).
**Impact on plan:** Ninguna. El cambio fortalece el gate de regresion sin alterar el scope.

## Issues Encountered

- **pytest no disponible en el host WSL**: igual que 07-01/02/03, se ejecuta en container `analisis_datos-web-1`. Patron: `docker cp <worktree-files> analisis_datos-web-1:/app/...` + `docker exec ... pytest`. Archivos sincronizados antes del test run: CLAUDE.md, tests/infra/, docs/ARCHITECTURE/RUNBOOK/RELEASE/DEPLOY_LAN.md, CHANGELOG.md, pyproject.toml, .pre-commit-config.yaml, .github/, Makefile, docker-compose.prod.yml, scripts/, caddy/, .env.prod.example, .gitignore, docker-compose.yml.
- **`make` no instalado en container**: `test_makefile_devex.py` usa `make -n` para validar targets. Sin `make`, ~9 tests fallarian con FileNotFoundError. Mitigacion: `docker exec ... apt-get install -y make` (efimero, se perdera al reiniciar el container; documentado en 07-02 SUMMARY). El container base `python:3.11-slim-bookworm` no lo incluye; el runner CI `ubuntu-24.04` si.
- **STATE.md modificado en working tree por el orchestrator**: idem 07-03. El orchestrator actualizo STATE.md antes de invocarme. La responsabilidad de STATE.md es del orchestrator tras completar la wave (instruccion literal de parallel_execution). NO se commitea desde este agent.

## Known Stubs

Ninguno. Todo el contenido anadido es texto completo y activado (no hay `<TODO>` ni placeholders no resueltos). El único elemento pendiente es la fecha concreta de "Ultima revision" que se fijo a 2026-04-22 (hoy).

## Threat Flags

Ninguno nuevo. El `threat_model` del plan (T-07-16, T-07-17, T-07-18) ya contempla:

- **T-07-16 (Tampering)**: PR que relaje las reglas silenciosamente — `mitigate` implementado via 13 tests de regresion + aislamiento de seccion para --no-verify.
- **T-07-17 (Information Disclosure)**: CLAUDE.md no contiene credenciales ni datos sensibles — `accept` heredado de Phase 1.
- **T-07-18 (Repudiation)**: commit message descriptivo con rationale de cada cambio; tests + doc en mismo commit para trazabilidad — `mitigate` implementado via Conventional Commits.

## TDD Gate Compliance

Task 1 declara `tdd="true"`. Gate sequence cumplida:

- **RED**: `tests/infra/test_claude_md.py` creado con 13 tests; ejecutado contra CLAUDE.md pre-edicion → **10 failed, 3 passed** (file_exists, no_emojis, preserves_existing_invariants). La premisa de RED se satisface: los 10 asserts nuevos fallan antes de la edicion.
- **GREEN**: CLAUDE.md editado con 4 cambios (header + 5 docs fuente verdad + 2 secciones nuevas + 2 bullets NO hacer) → **13/13 PASSED**.
- **REFACTOR**: no necesario en el GREEN, pero 1 refactor de test (deviation #1) se aplico antes del commit atomic: `test_que_no_hacer_has_no_verify_rule` reescrito para aislar seccion por regex en lugar de grep global.

Commit atomico unico (`8a22d15`) agrupa RED + GREEN + REFACTOR — patron heredado de 07-01 Task 2 y 07-02 Task 1/2 para evitar publicar RED en history.

## Estado de gates Phase 7 cerrables tras este plan

| Requirement | Plan | Estado |
|-------------|------|--------|
| DEVEX-01 (pyproject.toml + .pre-commit-config.yaml + ruff bump) | 07-01 | [x] Cerrado 2026-04-22 via commits 91617fa + fd83a4e + 10f1627 |
| DEVEX-02 (CI matriz Python 3.11+3.12 + cobertura 60%) | 07-02 | [x] Cerrado 2026-04-22 via commit ff27cb1 |
| DEVEX-03 (Makefile targets test/lint/format/migrate) | 07-02 | [x] Cerrado 2026-04-22 via commit d06c8ae |
| DEVEX-04 (docs/ARCHITECTURE.md) | 07-03 | [x] Cerrado 2026-04-22 via commit 99f89a0 |
| DEVEX-05 (docs/RUNBOOK.md con 5 escenarios) | 07-03 | [x] Cerrado 2026-04-22 via commit eb6a0c4 |
| DEVEX-06 (docs/RELEASE.md + CHANGELOG.md) | 07-03 | [x] Cerrado 2026-04-22 via commit 085f939 |
| **DEVEX-07 (CLAUDE.md actualizado + test regresion)** | **07-04** | **[x] Cerrado 2026-04-22 via commit 8a22d15 (este plan)** |

**Phase 7 cerrada.** Todos los requirements DEVEX-01..07 pasan de `Active` a `Validated` en REQUIREMENTS.md tras el run del orchestrator.

## User Setup Required

Ninguno. CLAUDE.md es un doc — ningún dev/IA tiene que activar nada. La documentacion sobre `pre-commit install` dirigida a futuros devs esta escrita en la seccion "Tooling DevEx"; el dev la lee y ejecuta ese comando cuando clona el repo.

## Next Phase Readiness

**No hay Next Phase en Mark-III.** Phase 7 era la ultima. Mark-III cierra con este plan.

Siguientes pasos (fuera de este plan, a responsabilidad del operador):

- Merge `feature/Mark-III` → `main`.
- Tag `v1.0.0` en `main` (seguir `docs/RELEASE.md`).
- Anuncio interno segun `docs/RELEASE.md` checklist ejecucion paso 7.
- Iniciar planning de Mark-IV (ingesta realtime, modulos nuevos, dashboards en streaming) cuando la direccion lo decida.

## Self-Check

Verificacion de claims antes de cerrar:

**Archivos creados presentes:**

- `tests/infra/test_claude_md.py` → FOUND (182 lineas, 13 tests)

**Archivos modificados presentes:**

- `CLAUDE.md` → FOUND (235 lineas, delta +84 vs 151 pre-plan)

**Commits en git log (este worktree):**

- `8a22d15` (Task 1: CLAUDE.md + test) → FOUND

**Verificaciones automatizadas (phase-level checks del PLAN.md):**

- `test -f CLAUDE.md` → exit 0
- `test -f tests/infra/test_claude_md.py` → exit 0
- `grep -qE '[ÚUú]ltima revisi[óo]n:\s*2026-0[4-9]' CLAUDE.md` → exit 0 (matches `2026-04-22`)
- `grep -q 'Tooling DevEx' CLAUDE.md` → exit 0
- `grep -q 'Despliegue productivo' CLAUDE.md` → exit 0
- `grep -q 'ARCHITECTURE.md' CLAUDE.md` → exit 0
- `grep -q 'RUNBOOK.md' CLAUDE.md` → exit 0
- `grep -q 'RELEASE.md' CLAUDE.md` → exit 0
- `grep -q 'CHANGELOG.md' CLAUDE.md` → exit 0
- `grep -q 'pre-commit install' CLAUDE.md` → exit 0
- `grep -cE 'make (lint|format|test|migrate)' CLAUDE.md` → 5 (>=3 requerido)
- `grep -qE 'cov-fail-under|60%' CLAUDE.md` → exit 0
- `grep -c 'make prod-' CLAUDE.md` → 5 (>=3 requerido)
- `grep -q 'DEPLOY_LAN.md' CLAUDE.md` → exit 0
- `grep -q 'no-verify' CLAUDE.md` → exit 0
- `grep -qE 'make up.*mcp|mcp.*make up|profiles.*mcp' CLAUDE.md` → exit 0 (invariante preservado)
- `grep -q 'carpeta \`OEE/\`' CLAUDE.md` → exit 0 (invariante preservado)
- `grep -q 'filter-repo' CLAUDE.md` → exit 0 (invariante preservado)
- `python3` emoji regex sobre CLAUDE.md → `ok` (sin emojis)
- `pytest tests/infra/test_claude_md.py -v --no-cov` (en container) → **13 passed in 0.02s**
- `pytest tests/infra/ --no-cov -q` (en container) → **135 passed, 5 skipped in 0.15s** (incluye suite completa Phase 6 + 7)

## Self-Check: PASSED

---
*Phase: 07-devex-hardening*
*Completed: 2026-04-22*
*Mark-III CLOSE.*
