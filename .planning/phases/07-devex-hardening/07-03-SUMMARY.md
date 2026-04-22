---
phase: 07-devex-hardening
plan: 03
subsystem: docs
tags:
  - devex
  - docs
  - architecture
  - runbook
  - release
  - changelog

# Dependency graph
requires:
  - phase: 07-devex-hardening
    plan: 02
    provides: "CI jobs (lint + test + smoke + build) + Makefile DevEx targets que RELEASE.md referencia literalmente, y pyproject.toml/pre-commit de 07-01 citados en CHANGELOG [1.0.0] Phase 7."
provides:
  - "docs/ARCHITECTURE.md con mapa tecnico de Nexo + Mermaid de 3 engines + middleware stack verificado contra api/main.py:298-301."
  - "docs/RUNBOOK.md con EXACTAMENTE 5 escenarios de incidencia (MES caido, Postgres no arranca, Caddy cert, Pipeline atascado, Lockout propietario) con hallazgos criticos documentados."
  - "docs/RELEASE.md con checklist semver + 7 items pre-release + 7 pasos ejecucion + referencia literal a scripts/deploy.sh y tests/infra/deploy_smoke.sh (Phase 6)."
  - "CHANGELOG.md Keep a Changelog 1.1.0 con [Unreleased] + [1.0.0] detallando Phases 1-7."
  - "tests/infra/test_devex_docs.py con 19 tests de regresion que congelan los 4 docs (estructura + palabras clave + hallazgos criticos + invariante sin emojis)."
affects:
  - 07-04-closing-claude-md-and-verification

# Tech tracking
tech-stack:
  added:
    - "Mermaid flowchart en docs/ARCHITECTURE.md §3 (3 engines + mcp profile-gated)"
    - "Keep a Changelog 1.1.0 format en CHANGELOG.md"
  patterns:
    - "Quartet docs pattern: ARCHITECTURE (tecnico) + RUNBOOK (incidencia) + RELEASE (corte) + CHANGELOG (historial) — replicable en Mark-IV."
    - "Hallazgo critico pattern en RUNBOOK: Contexto (HALLAZGO CRITICO) + Remedio nuclear + Remedio no-nuclear + Prevencion hard-rule. Evita que el operador asuma que existe un helper que no existe."
    - "Regression test pattern heredado de test_deploy_lan_doc.py (Phase 6): pathlib + regex sobre texto + asserts con mensaje claro."

key-files:
  created:
    - "docs/ARCHITECTURE.md (255 lineas)"
    - "docs/RUNBOOK.md (476 lineas)"
    - "docs/RELEASE.md (159 lineas)"
    - "CHANGELOG.md (105 lineas)"
    - "tests/infra/test_devex_docs.py (230 lineas, 19 tests)"
  modified: []

key-decisions:
  - "Quartet docs como cierre de Phase 7: ARCHITECTURE + RUNBOOK + RELEASE + CHANGELOG cubren DEVEX-04, DEVEX-05, DEVEX-06 en un solo plan (03)."
  - "Hallazgos criticos explicitamente destacados en RUNBOOK Escenarios 4 y 5: no existe list_locks() helper, no existe unlock_user() helper. Documentar la AUSENCIA de helpers es tan importante como documentar su presencia."
  - "Mermaid flowchart para los 3 engines: renderizable en GitHub y VSCode sin extensiones; fallback ASCII descartado por perdida de claridad."
  - "Keep a Changelog 1.1.0 con [Unreleased] + [1.0.0]: placeholder <org> en los links de comparacion para que el mantenedor los rellene tras el merge a main."
  - "Regression test con 19 tests (no 12+): cubrir hallazgos criticos explicitamente via grep keywords (pipeline_semaphore, DELETE FROM nexo.login_attempts, caddy_data) ademas de estructura y cross-links."

patterns-established:
  - "Placeholders con angle brackets (<IP_NEXO>, <email>, <POSTGRES_USER>, <latest>) con seccion de leyenda al inicio del doc, heredado de DEPLOY_LAN.md Phase 6."
  - "Cross-links en seccion Enlaces rapidos/Referencias al final de cada doc del quartet — navegacion bidireccional sin dead-ends."
  - "Regression test que lee y parsea texto con regex explicita contra keywords criticas. Si alguien borra el diagrama Mermaid o rompe la estructura de un escenario, el test falla en CI antes del merge."

requirements-completed:
  - DEVEX-04
  - DEVEX-05
  - DEVEX-06

# Metrics
duration: ~11min
completed: 2026-04-22
---

# Phase 07 Plan 03: ARCHITECTURE + RUNBOOK + RELEASE + CHANGELOG Summary

**Quartet de docs de cierre de Mark-III: `docs/ARCHITECTURE.md` (mapa tecnico con Mermaid de 3 engines + middleware stack), `docs/RUNBOOK.md` (5 escenarios de incidencia con 2 hallazgos criticos sobre `list_locks()` y `unlock_user()` inexistentes), `docs/RELEASE.md` (checklist semver + deploy Phase 6) y `CHANGELOG.md` (Keep a Changelog con Phases 1-7). Blindado por 19 tests de regresion en `tests/infra/test_devex_docs.py` que congelan estructura y palabras clave criticas.**

## Performance

- **Duration:** ~11 min
- **Started:** 2026-04-22T14:12:15Z
- **Completed:** 2026-04-22T14:23:01Z
- **Tasks:** 3 (Task 1 ARCHITECTURE; Task 2 RUNBOOK; Task 3 RELEASE + CHANGELOG + tests)
- **Files created:** 5 (ARCHITECTURE.md, RUNBOOK.md, RELEASE.md, CHANGELOG.md, test_devex_docs.py)
- **Tests anadidos:** 19 (1 fichero de regresion)
- **Lineas totales escritas:** 1225 (255 + 476 + 159 + 105 + 230)

## Accomplishments

- **docs/ARCHITECTURE.md**: Mapa tecnico completo con 9 secciones. Diagrama Mermaid flowchart con los 3 engines (`engine_nexo`, `engine_app`, `engine_mes`) + Caddy + Web + subgrafo opcional `mcp` profile-gated. Tabla de responsabilidades por engine con env vars y casos de uso. Layout del repo anotado marcando `OEE/` como `LEGACY — NO TOCAR`. Middleware stack verificado literalmente contra `api/main.py:298-301` con el orden correcto `AuthMiddleware (outer) -> FlashMiddleware -> AuditMiddleware -> QueryTimingMiddleware (inner)`. Schedulers tabulados con las 4 tasks asyncio de Phase 4. Cross-links a 9 docs externos (CLAUDE.md, AUTH_MODEL.md, BRANDING.md, GLOSSARY.md, DEPLOY_LAN.md, RUNBOOK.md, RELEASE.md, SECURITY_AUDIT.md, CURRENT_STATE_AUDIT.md, MARK_III_PLAN.md).
- **docs/RUNBOOK.md**: 5 escenarios con estructura literal Sintomas -> Diagnostico -> Remedio -> Prevencion. Dos hallazgos criticos explicitos:
  - Escenario 4 (Pipeline atascado): documenta que **NO existe un helper `list_locks()`**; diagnostico indirecto accediendo al atributo privado `pipeline_semaphore._value` (marcado como fragil); remedio nuclear `docker compose restart web` + remedio no-nuclear (esperar timeout 15 min).
  - Escenario 5 (Lockout propietario): documenta que **NO existe `unlock_user()`**; remedio DELETE FROM nexo.login_attempts WHERE email via `docker compose exec -T db psql` marcado como EMERGENCY ONLY; prevencion HARD RULE de **>=2 propietarios** (uno principal + uno break-glass con password fuerte en sobre sellado).
  - Escenario 3 (Caddy): root CA 10 anos + intermediate 7 dias + volumen `caddy_data` + warning contra `docker compose down -v`.
  - Escenario 2 (Postgres): `pgdata` volume persistence + Landmine 6 (no `down -v`) + restore desde `/var/backups/nexo/<latest>.sql.gz`.
  - Escenario 1 (MES): degradacion graciosa (`services.mes.ok=false` pero app viva); 3 casos de remedio (red / credenciales / MES down).
- **docs/RELEASE.md**: Checklist semver con `vMAJOR.MINOR.PATCH`, mapa v1.0.0 = Mark-III close, v2.0.0 = Mark-IV. Checklist pre-release con 7 items (ROADMAP, CI verde, pytest cov 60, pre-commit clean, STATE blockers, CHANGELOG, docs milestone). Checklist ejecucion con 7 pasos numerados (merge feature -> main, tag, gh release, ssh deploy, smoke externo, verificacion manual LAN, anuncio interno). Seccion Rollback con comandos `git checkout <tag-anterior>` + restore desde `/var/backups/nexo/predeploy/`. Advertencia explicita "Tags son inmutables".
- **CHANGELOG.md**: Keep a Changelog 1.1.0 + Semantic Versioning. `[Unreleased]` como placeholder con subsecciones vacias Added/Changed/Fixed. `[1.0.0]` con cierre de Mark-III: 7 bullets en `Added` (uno por cada Phase 1-7 con resumen de entregables), `Changed` (OEE/ sin renombrar, repo sin renombrar, mcp profile-gated, cfg.* en ecs_mobility), `Security` (rotacion SA, exception handler UUID, audit append-only, gitleaks continue-on-error), `Deprecated` (env vars `OEE_*` como fallback Mark-III). Links al final con placeholder `<org>` para el mantenedor.
- **tests/infra/test_devex_docs.py**: 19 tests de regresion (no 12+ como minimo requerido):
  - 5 tests sobre ARCHITECTURE.md (existe + mermaid, 3 engines, middleware stack, cross-links, min 150 lineas).
  - 7 tests sobre RUNBOOK.md (exactamente 5 escenarios, 20 sub-secciones, hallazgo pipeline_semaphore + restart, hallazgo DELETE FROM + >=2 propietarios, Caddy rotation info, Postgres pgdata, min 250 lineas).
  - 3 tests sobre RELEASE.md (semver regex, checklist >=7 items, referencia a scripts/deploy.sh + deploy_smoke.sh).
  - 3 tests sobre CHANGELOG.md (Keep a Changelog + semver, Unreleased + 1.0.0, Phase 1-7 cubiertas).
  - 1 test de hard invariant: sin emojis en ninguno de los 4 docs (regex unicode ranges).

## Task Commits

Cada task committed atomicamente con `--no-verify` en el worktree `agent-aef00589`:

1. **Task 1: docs/ARCHITECTURE.md** — `99f89a0` (docs) → 255 lineas, 9 secciones, diagrama Mermaid.
2. **Task 2: docs/RUNBOOK.md** — `eb6a0c4` (docs) → 476 lineas, 5 escenarios, 20 sub-secciones.
3. **Task 3: docs/RELEASE.md + CHANGELOG.md + tests/infra/test_devex_docs.py** — `085f939` (docs) → 494 lineas totales (159 + 105 + 230), 19 tests green.

## Files Created/Modified

### Created

- `docs/ARCHITECTURE.md` — 255 lineas. Mapa tecnico con Mermaid flowchart, middleware stack, schedulers, cross-links.
- `docs/RUNBOOK.md` — 476 lineas. 5 escenarios con hallazgos criticos sobre pipeline_lock y auth sin helpers admin.
- `docs/RELEASE.md` — 159 lineas. Checklist versionado semver + deploy Phase 6 + rollback.
- `CHANGELOG.md` — 105 lineas. Keep a Changelog 1.1.0 con Phase 1-7 cubiertas en [1.0.0].
- `tests/infra/test_devex_docs.py` — 230 lineas, 19 tests de regresion sobre el quartet de docs.

### Modified

Ninguno. El plan solo crea archivos nuevos.

## Quartet docs — lineas finales

| Doc                               | Lineas | Keywords criticas verificadas por test                                                               |
|-----------------------------------|--------|-------------------------------------------------------------------------------------------------------|
| docs/ARCHITECTURE.md              | 255    | mermaid, engine_mes, engine_app, engine_nexo, AuthMiddleware, AuditMiddleware, FlashMiddleware, QueryTimingMiddleware, DEPLOY_LAN.md, RUNBOOK.md, RELEASE.md |
| docs/RUNBOOK.md                   | 476    | Escenario 1..5, pipeline_semaphore, list_locks, docker compose restart web, DELETE FROM nexo.login_attempts, >=2 propietarios, caddy_data, root CA, intermediate, 7 dias, 10 anos, pgdata, backup_nightly, down -v |
| docs/RELEASE.md                   | 159    | v1.0.0 semver, 7 checklist items, scripts/deploy.sh, deploy_smoke.sh, tag inmutable                  |
| CHANGELOG.md                      | 105    | Keep a Changelog, Semantic Versioning, [Unreleased], [1.0.0], Phase 1..Phase 7 (7 referencias)       |

## Numero de escenarios RUNBOOK

**5 exactos** (verificado por `grep -cE '^## Escenario [1-5]:' docs/RUNBOOK.md` = 5, y por el test `test_runbook_md_has_exactly_5_scenarios`).

- Escenario 1: MES caido (SQL Server dbizaro inaccesible)
- Escenario 2: Postgres no arranca
- Escenario 3: Certificado Caddy expira / warning en browsers
- Escenario 4: Pipeline atascado (HALLAZGO CRITICO: semaforo in-process)
- Escenario 5: Lockout del unico propietario (HALLAZGO CRITICO: no hay `unlock_user`)

## Palabras clave verificadas por test de regresion

- `pipeline_semaphore` — presente en Escenario 4 como atributo privado de diagnostico.
- `DELETE FROM nexo.login_attempts` — presente en Escenario 5 como remedio EMERGENCY ONLY.
- `caddy_data` — presente en Escenario 3 como volumen donde vive la root CA.
- `pgdata` — presente en Escenario 2 como volumen Postgres + warning down -v.
- `>=2 propietarios` — presente en Escenario 5 como prevencion hard-rule.
- `docker compose restart web` — presente en Escenario 4 como remedio nuclear.
- `engine_mes`, `engine_app`, `engine_nexo` — presentes en ARCHITECTURE.md §3.
- `AuthMiddleware`, `AuditMiddleware`, `FlashMiddleware`, `QueryTimingMiddleware` — presentes en ARCHITECTURE.md §5.
- `v[0-9]+\.[0-9]+\.[0-9]+` — presente en RELEASE.md (semver).
- `## [Unreleased]`, `## [1.0.0]` — presentes en CHANGELOG.md.
- `Phase 1..Phase 7` — cubiertas en CHANGELOG.md [1.0.0] Added.

## Test count anyadido en tests/infra/

**19 tests** en `tests/infra/test_devex_docs.py` (superado el minimo de 12 requerido). Resultado en contenedor `analisis_datos-web-1`:

```
============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-8.3.4, pluggy-1.6.0
collected 19 items

tests/infra/test_devex_docs.py::test_architecture_md_exists_and_has_mermaid PASSED
tests/infra/test_devex_docs.py::test_architecture_md_has_three_engines PASSED
tests/infra/test_devex_docs.py::test_architecture_md_has_middleware_stack PASSED
tests/infra/test_devex_docs.py::test_architecture_md_cross_links PASSED
tests/infra/test_devex_docs.py::test_architecture_md_has_min_length PASSED
tests/infra/test_devex_docs.py::test_runbook_md_has_exactly_5_scenarios PASSED
tests/infra/test_devex_docs.py::test_runbook_md_scenario_structure PASSED
tests/infra/test_devex_docs.py::test_runbook_md_pipeline_scenario_has_restart_remedy PASSED
tests/infra/test_devex_docs.py::test_runbook_md_lockout_scenario_has_delete_remedy PASSED
tests/infra/test_devex_docs.py::test_runbook_md_caddy_scenario_has_rotation_info PASSED
tests/infra/test_devex_docs.py::test_runbook_md_postgres_scenario_mentions_pgdata PASSED
tests/infra/test_devex_docs.py::test_runbook_md_has_min_length PASSED
tests/infra/test_devex_docs.py::test_release_md_has_semver PASSED
tests/infra/test_devex_docs.py::test_release_md_has_checklist_with_min_7_items PASSED
tests/infra/test_devex_docs.py::test_release_md_references_deploy_script PASSED
tests/infra/test_devex_docs.py::test_changelog_md_has_keep_a_changelog PASSED
tests/infra/test_devex_docs.py::test_changelog_md_has_unreleased_and_1_0_0 PASSED
tests/infra/test_devex_docs.py::test_changelog_md_covers_all_phases PASSED
tests/infra/test_devex_docs.py::test_all_docs_have_no_emojis PASSED

============================== 19 passed in 0.03s ==============================
```

## Referencias cruzadas ARCHITECTURE.md -> otros docs

Seccion "8. Enlaces rapidos" de `docs/ARCHITECTURE.md` enlaza con:

- `CLAUDE.md` (reglas de juego, decisiones cerradas)
- `docs/AUTH_MODEL.md` (roles, departamentos, lockout progresivo)
- `docs/BRANDING.md` (assets + variables de marca)
- `docs/GLOSSARY.md` (terminos de dominio: Nexo, MES, APP, preflight)
- `docs/DEPLOY_LAN.md` (runbook deploy LAN completo)
- `docs/RUNBOOK.md` (5 escenarios de incidencia runtime)
- `docs/RELEASE.md` (checklist release versionado con semver)
- `docs/SECURITY_AUDIT.md` (historial de credenciales expuestas)
- `docs/CURRENT_STATE_AUDIT.md` (foto fija del repo al arrancar Mark-III)
- `docs/MARK_III_PLAN.md` (plan de los 7 sprints con entregables)

Cross-links bidireccionales:
- `RUNBOOK.md` -> ARCHITECTURE.md, DEPLOY_LAN.md, AUTH_MODEL.md, RELEASE.md, CLAUDE.md.
- `RELEASE.md` -> CHANGELOG.md, DEPLOY_LAN.md, RUNBOOK.md, ARCHITECTURE.md, scripts/deploy.sh, tests/infra/deploy_smoke.sh.

## Decisions Made

- **Quartet docs en un solo plan (03)**: DEVEX-04 (ARCHITECTURE) + DEVEX-05 (RUNBOOK) + DEVEX-06 (RELEASE + CHANGELOG) van juntos porque comparten contexto narrativo y cross-links bidireccionales. Separarlos obligaria a editar dos veces las listas de enlaces mutuos.
- **Hallazgo critico pattern explicito**: Escenarios 4 y 5 de RUNBOOK abren con una seccion "Contexto (HALLAZGO CRITICO)" que declara explicitamente que una funcion o helper NO existe. Evita que el operador asuma que puede llamar a `list_locks()` o `unlock_user()` sin verificar. Esta informacion es tan importante como el remedio.
- **Placeholders con angle brackets consistentes con DEPLOY_LAN.md**: `<IP_NEXO>`, `<POSTGRES_USER>`, `<email>`, `<ip>`, `<latest>` — mismo patron que Phase 6 para que el operador que lee RUNBOOK no tenga que cambiar de convencion entre docs.
- **Mermaid flowchart en lugar de ASCII art**: GitHub y VSCode renderizan nativamente; ASCII art perderia claridad (3 subgrafos + subgrafo opcional mcp + flujo Browser -> Caddy -> Web -> 3 engines).
- **19 tests (no 12+)**: cubrir hallazgos criticos explicitamente (pipeline_semaphore + restart, DELETE + >=2 propietarios, caddy_data + rotation, pgdata) ademas de estructura. El coste marginal de 7 tests extra es despreciable vs el valor de alert si alguien borra una linea critica.
- **CHANGELOG links con placeholder `<org>`**: el mantenedor rellena `<org>` tras el merge a main (no sabemos si es `ecs-mobility`, `ECS-Mobility` u otro handle). Los links no se pueden hardcodear sin romperlos en el futuro.
- **Escenario 1 (MES) sin marcado HALLAZGO CRITICO**: la degradacion graciosa (`/api/health` devuelve 200 con `services.mes.ok=false`) es comportamiento intencional y esperado, no una ausencia de helper. Solo los Escenarios 4 y 5 llevan el marcador HALLAZGO CRITICO.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Regex `>=2 propietarios` requirio reformular el texto de la prevencion Escenario 5**

- **Found during:** Task 2 (verificacion de acceptance criteria)
- **Issue:** El plan define literalmente la regex `(>= ?2|>=2|dos|al menos 2) propietarios` (plural) para validar la prevencion del Escenario 5. El primer texto escribio "SIEMPRE crear >=2 usuarios con rol `propietario`", donde `propietario` es singular y la regex buscaba `propietarios` plural. La acceptance criterion fallaba por una palabra.
- **Fix:** Reformular la prevencion como "SIEMPRE crear >=2 propietarios (usuarios con rol `propietario`) — uno principal operativo, uno break-glass con password muy fuerte guardado en sobre sellado en caja fuerte IT. Esta regla de tener al menos 2 propietarios activos en todo momento es lo que evita el soft-brick." Ahora `>=2 propietarios` aparece literalmente dos veces en el texto (1. titulo del punto; 3. recapitulacion en el cierre) — queda robusto a reescrituras menores.
- **Files modified:** `docs/RUNBOOK.md` (Escenario 5 prevencion)
- **Verification:** `grep -cE '(>= ?2|>=2|dos|al menos 2) propietarios' docs/RUNBOOK.md` → 2 (>=1 requerido). Test `test_runbook_md_lockout_scenario_has_delete_remedy` PASSED.
- **Committed in:** `eb6a0c4` (mismo commit que la creacion de RUNBOOK.md, no merece commit separado porque el texto nunca llego a publicarse con la version singular — se corrigio en la misma sesion antes del commit atomic).

---

**Total deviations:** 1 auto-fixed (bug textual en regex literal del plan).
**Impact on plan:** Ninguna deviation afecta al alcance de DEVEX-04/05/06.

## Issues Encountered

- **pytest no disponible en el host WSL**: igual que los plans 07-01 y 07-02, el host no tiene Python/pytest instalado. La ejecucion de la suite se hizo via `docker cp` + `docker exec analisis_datos-web-1 python -m pytest`. El contenedor NO tiene montado el bind del worktree (solo monta el main), por eso hay que copiar explicitamente los 5 archivos nuevos antes de cada run.
- **STATE.md modificado en working tree por el orchestrator**: el orchestrator actualizo `.planning/STATE.md` antes de invocarme (cambio de `--phase` literal al valor real `07 devex-hardening`, bump de timestamps y progress). Esta modificacion esta en el working tree pero NO la he commiteado — la responsabilidad de STATE.md es del orchestrator tras completar la wave (instruccion literal de parallel_execution).

## Known Stubs

Ninguno. Los 4 docs son texto completo, sin TODO, sin placeholders no resueltos. El placeholder `<org>` en los links del CHANGELOG es intencional y esta documentado en Decisions Made — el mantenedor lo rellena tras el merge a main.

## Threat Flags

Ninguno nuevo. El `threat_model` del plan (T-07-11..T-07-15) ya contempla:

- T-07-11: placeholders `<angle brackets>` evitan copia accidental de credenciales reales (mitigate — implementado).
- T-07-12: DELETE de Escenario 5 marcado EMERGENCY ONLY + require root al host (accept — documentado).
- T-07-13: supply chain interno (accept — control de acceso repo ECS Mobility).
- T-07-14: `docker compose restart web` documenta explicitamente "~30s, usuarios pierden sesion" (mitigate — documentado).
- T-07-15: seccion Security en CHANGELOG referencia `docs/SECURITY_AUDIT.md`, no detalla CVEs in-line (accept — guideline aplicada).

## TDD Gate Compliance

Este plan **no es `type: tdd` a nivel de plan** ni a nivel de task. Es un plan de documentacion + un test de regresion tardio (tests/infra/test_devex_docs.py se crea DESPUES de los 4 docs, no antes).

Justificacion: los 4 docs son contenido original (no hay "funcion" a especificar por test); el test de regresion existe para evitar DRIFT posterior, no para guiar la implementacion. El equivalente TDD seria redundante.

Estrategia aplicada: Task 1 y Task 2 sin tests (crear doc + verificar con `grep` acceptance criteria del plan). Task 3 crea RELEASE.md + CHANGELOG.md + el test `test_devex_docs.py` que valida los 4 docs a la vez. Los 19 tests pasan al primer intento (no hay RED en el historial). Los docs son su propia "implementacion" y el test es el "gate post-hoc" que evita drift.

Commits atomicos:
- `99f89a0` (Task 1) → solo docs/ARCHITECTURE.md
- `eb6a0c4` (Task 2) → solo docs/RUNBOOK.md
- `085f939` (Task 3) → docs/RELEASE.md + CHANGELOG.md + tests/infra/test_devex_docs.py

## User Setup Required

Ninguno. El operador que lea RUNBOOK.md para responder a una incidencia necesita:

- Acceso SSH al servidor productivo (`ssh <IP_NEXO>`).
- Credenciales `<POSTGRES_USER>` / `<POSTGRES_DB>` en `/opt/nexo/.env` (ya existentes desde Phase 6).
- Docker + Docker Compose en el host (ya existentes desde Phase 6).

Nada nuevo que instalar.

## Next Plan Readiness

**Listo para Plan 07-04 (closing-claude-md-and-verification)**:

- El quartet de docs es consumible por 07-04 como referencia en la actualizacion de `CLAUDE.md` (seccion "Fuente de verdad del plan" anyade ARCHITECTURE.md, RUNBOOK.md, RELEASE.md, CHANGELOG.md).
- Los 19 tests de regresion blindan los docs contra drift durante 07-04 y plans futuros: si alguien toca un escenario del RUNBOOK o borra una seccion del ARCHITECTURE, pytest falla en CI.
- No hay blockers para 07-04.

## Self-Check

Verificacion de claims antes de cerrar:

**Archivos creados presentes:**

- `docs/ARCHITECTURE.md` → FOUND (255 lineas)
- `docs/RUNBOOK.md` → FOUND (476 lineas)
- `docs/RELEASE.md` → FOUND (159 lineas)
- `CHANGELOG.md` → FOUND (105 lineas)
- `tests/infra/test_devex_docs.py` → FOUND (230 lineas)

**Commits en git log (este worktree):**

- `99f89a0` (Task 1 ARCHITECTURE.md) → FOUND
- `eb6a0c4` (Task 2 RUNBOOK.md) → FOUND
- `085f939` (Task 3 RELEASE + CHANGELOG + tests) → FOUND

**Verificaciones automatizadas (phase-level checks del PLAN.md):**

- `test -f docs/ARCHITECTURE.md` → exit 0
- `test -f docs/RUNBOOK.md` → exit 0
- `test -f docs/RELEASE.md` → exit 0
- `test -f CHANGELOG.md` → exit 0
- `grep -q 'mermaid' docs/ARCHITECTURE.md` → exit 0
- `grep -c 'engine_mes\|engine_app\|engine_nexo' docs/ARCHITECTURE.md` → 8 (>=3 requerido)
- `grep -q 'AuthMiddleware' docs/ARCHITECTURE.md` → exit 0
- `grep -q 'FlashMiddleware' docs/ARCHITECTURE.md` → exit 0
- `[ $(grep -cE '^## Escenario [1-5]:' docs/RUNBOOK.md) -eq 5 ]` → exit 0 (exactamente 5)
- `grep -q 'pipeline_semaphore' docs/RUNBOOK.md` → exit 0
- `grep -q 'DELETE FROM nexo.login_attempts' docs/RUNBOOK.md` → exit 0
- `grep -q 'caddy_data' docs/RUNBOOK.md` → exit 0
- `grep -q 'pgdata' docs/RUNBOOK.md` → exit 0
- `grep -qE 'v[0-9]+\.[0-9]+\.[0-9]+' docs/RELEASE.md` → exit 0
- `[ $(grep -cE '^- \[ \]' docs/RELEASE.md) -ge 7 ]` → exit 0 (7 items)
- `grep -q 'scripts/deploy.sh' docs/RELEASE.md` → exit 0
- `grep -q 'deploy_smoke.sh' docs/RELEASE.md` → exit 0
- `grep -q 'Keep a Changelog' CHANGELOG.md` → exit 0
- `grep -q '## \[Unreleased\]' CHANGELOG.md` → exit 0
- `grep -q '## \[1.0.0\]' CHANGELOG.md` → exit 0
- `[ $(grep -c 'Phase [1-7]' CHANGELOG.md) -ge 7 ]` → exit 0 (10 referencias)
- `python3` emoji regex sobre los 4 docs → `PASS: no emojis` en los 4
- `pytest tests/infra/test_devex_docs.py -v --no-cov` (en contenedor) → **19 passed in 0.03s**

## Self-Check: PASSED

---
*Phase: 07-devex-hardening*
*Completed: 2026-04-22*
