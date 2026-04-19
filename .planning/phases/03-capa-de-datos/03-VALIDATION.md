---
phase: 3
slug: capa-de-datos
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-19
revised: 2026-04-19
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Revision note (2026-04-19)

Flags `nyquist_compliant` and `wave_0_complete` flipped to `true` after plan-checker revision:
- **Nyquist**: every DATA-XX requirement has an `<automated>` verify command; no 3-consecutive-task gap without automation.
- **Wave 0**: all Wave 0 artifacts correctly distributed across plans (03-01 owns foundation tests + fixtures + conftest; 03-02 owns MES tests + PDF scripts **created BEFORE the checkpoint**; 03-03 owns APP/NEXO repo tests + meta-pyodbc).

Changes that unblocked the flip:
1. 03-02 Task 1/Task 2 swapped: scripts are created (Task 1) BEFORE the checkpoint (Task 2) that executes them — logical ordering restored.
2. 03-02 `files_modified` no longer overlaps with 03-03 on `api/services/pipeline.py` — pipeline.py is exclusively edited in 03-03 Task 4.7 (handoff documented in `<context_handoff>` of both plans).
3. 03-01 `schema_guard.verify` now accepts `critical_tables` kwarg — tests inject fake tables via kwarg instead of monkeypatching a module constant (no longer fragile).
4. 03-01 Task 2 grep verify split into two separate checks (broken `grep -q | head -1` pattern fixed).
5. 03-02 PDF baseline fallback strengthened: if operator defers, STATE.md TODO with 7-day deadline + SUMMARY.md gate-pending entry required; plan cannot be marked DONE until baseline executed in preprod.
6. DATA-05 scope clarification (CONTEXT.md D-01, 2026-04-19): ORM-only repos (RecursoRepo, CicloRepo, UserRepo, AuditRepo, etc.) are DATA-05-compliant WITHOUT `.sql` files — cited explicitly in 03-02 Task 3.5 (luk4) and 03-03 Task 3 (APP/NEXO).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.3.4 + pytest-cov 6.0.0 + httpx 0.28.1 (TestClient) |
| **Config file** | none — usa defaults + `tests/conftest.py` (marker `integration` ya registrado) |
| **Quick run command** | `docker compose exec web pytest tests/data/ -q -m "not integration"` |
| **Full suite command** | `docker compose exec web pytest tests/ -q` |
| **Estimated runtime** | ~30-60s quick, ~120-180s full (incluye `tests/auth/` ya verde) |

---

## Sampling Rate

- **After every task commit:** `docker compose exec web pytest tests/data/ -q -m "not integration"` (~10-30s).
- **After every plan wave:** `docker compose exec web pytest tests/data/ tests/auth/ -q` (verifica que el gate IDENT-06 sigue verde).
- **Before `/gsd-verify-work`:** Full suite + `python scripts/pdf_regression_check.py --fecha=2026-03-15` (success criterion #5).
- **Max feedback latency:** 60s para quick run.

---

## Per-Task Verification Map

> Mapeo derivado de RESEARCH §Validation Architecture. Cada DATA-* tiene al menos 1 test automatizado. Sin requirements en estado MISSING; todo va a Wave 0 porque `tests/data/` no existe todavía.

| Req ID | Plan | Wave | Behavior | Test Type | Automated Command | File Exists |
|--------|------|------|----------|-----------|-------------------|-------------|
| DATA-01 | 03-01 | 0 | 3 engines con pool configurado correcto | unit | `pytest tests/data/test_engines.py -x` | ❌ W0 (03-01 Task 3 crea) |
| DATA-01 | 03-01 | 1 | `engine_mes` apunta a `dbizaro` (DB_NAME()) | integration | `pytest tests/data/test_engines.py::test_engine_mes_database_is_dbizaro -x` | ❌ W0 (03-01 Task 3 crea) |
| DATA-02 | 03-02 | 0 | `MesRepository` con 5 métodos (firma + delegación) | unit | `pytest tests/data/test_mes_repository.py -x` | ❌ W0 (03-02 Task 4 crea) |
| DATA-02 | 03-02 | 1 | `extraer_datos_produccion` delega a `OEE.db.connector.extraer_datos` | unit (mock) | `pytest tests/data/test_mes_repository.py::test_extraer_datos_delega -x` | ❌ W0 (03-02 Task 4 crea) |
| DATA-03 | 03-03 | 0 | 6 repos APP con contrato session-inyectada | integration | `pytest tests/data/test_app_repository.py -x` | ❌ W0 (03-03 Task 5 crea) |
| DATA-04 | 03-03 | 0 | 3 repos NEXO (`UserRepo`, `RoleRepo`, `AuditRepo`) | integration | `pytest tests/data/test_nexo_repository.py -x` | ❌ W0 (03-03 Task 5 crea) |
| DATA-04 | 03-03 | 1 | `AuditRepo.append` no comitea internamente | integration | `pytest tests/data/test_nexo_repository.py::test_audit_repo_no_commits -x` | ❌ W0 (03-03 Task 5 crea) |
| DATA-05 | 03-01 | 0 | Loader carga `.sql` con `lru_cache` | unit | `pytest tests/data/test_sql_loader.py -x` | ❌ W0 (03-01 Task 3 crea) |
| DATA-05 | 03-02 | 1 | Comentario T3 presente en `extraer_datos_produccion.sql` | unit | `pytest tests/data/test_sql_loader.py::test_t3_comment_preserved -x` | ❌ W0 (03-02 Task 3 crea el .sql) |
| DATA-05 | 03-03 | 1 | ORM puro APP/NEXO compliance (no `.sql` files expected per D-01) | meta | `test -z "$(ls -A nexo/data/sql/app 2>/dev/null)" && test -z "$(ls -A nexo/data/sql/nexo 2>/dev/null)"` | ❌ W0 (03-03 Task 3 cumple por no-generación) |
| DATA-06 | 03-01 | 0 | `schema_guard.verify` falla con tabla faltante (via kwarg `critical_tables`) | integration | `pytest tests/data/test_schema_guard.py::test_verify_raises_when_table_missing -x` | ❌ W0 (03-01 Task 3 crea) |
| DATA-06 | 03-01 | 1 | `NEXO_AUTO_MIGRATE=true` auto-crea | integration | `pytest tests/data/test_schema_guard.py::test_auto_migrate_creates_missing -x` | ❌ W0 (03-01 Task 3 crea) |
| DATA-07 | 03-02/03 | 2 | `grep "import pyodbc" api/routers/` = 0 | smoke (meta) | `pytest tests/data/test_no_raw_pyodbc_in_routers.py -x` | ❌ W0 (03-03 Task 5 crea) |
| DATA-07 | 03-02 | 2 | `grep "dbizaro\\." api/` = 0 | smoke (meta) | `pytest tests/data/test_no_three_part_names.py -x` | ❌ W0 (03-02 Task 4 crea) |
| DATA-07 | 03-02/03 | 2 | Los 10 routers responden 200 bajo auth válida | integration | `pytest tests/data/test_routers_smoke.py -x` | ❌ W0 (03-03 Task 5 crea) |
| DATA-07 | 03-03 | 2 | `api/services/pipeline.py` sin pyodbc tras handoff atómico | meta | `grep -c "import pyodbc" api/services/pipeline.py \| grep -q "^0$"` | ❌ W0 (03-03 Task 4.7 garantiza) |
| DATA-08 | 03-01 | 0 | DTOs son frozen y fail con mutation | unit | `pytest tests/data/test_dto_immutable.py -x` | ❌ W0 (03-01 Task 3 crea) |
| DATA-09 | 03-02 | 1 | Query `centro_mando` sin `dbizaro.` (DB_NAME() = dbizaro) | integration | `pytest tests/data/test_mes_engine_context.py -x` | ❌ W0 (03-02 Task 4 crea) |
| DATA-10 | 03-01 | 0 | Fixtures `db_nexo`, `db_app`, `engine_mes_mock` funcionan | integration | `pytest tests/data/test_fixtures.py -x` | ❌ W0 (03-01 Task 3 crea) |
| DATA-11 | 03-01 | 0 | `engine_mes` pool: pool_recycle=3600, pool_pre_ping=True | unit | `pytest tests/data/test_engines.py::test_mes_pool_config -x` | ❌ W0 (03-01 Task 3 crea) |
| Success #5 | 03-02 | gate | PDF generation pre/post refactor → hash o size match | regression | `python scripts/pdf_regression_check.py --fecha=2026-03-15` | ❌ W0 (03-02 Task 1 crea script **ANTES del checkpoint** Task 2) |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Estos artefactos NO existen y deben ser creados antes de poder validar Phase 3. Distribuidos por plan (revisados tras revision 2026-04-19 para resolver overlap en `api/services/pipeline.py` y ordering de Task 1/2 en 03-02):

**En 03-01 (foundation):**
- [ ] `tests/data/__init__.py` — empty marker (Task 3)
- [ ] `tests/data/conftest.py` — fixtures `db_nexo`, `db_app`, `engine_mes_mock` + skipifs para SQL Server en CI (Task 3)
- [ ] `tests/data/test_engines.py` — DATA-01 + DATA-11 (Task 3)
- [ ] `tests/data/test_sql_loader.py` — DATA-05 (Task 3)
- [ ] `tests/data/test_schema_guard.py` — DATA-06 **using `critical_tables=` kwarg, not module monkeypatch** (Task 3)
- [ ] `tests/data/test_dto_immutable.py` — DATA-08 (Task 3)
- [ ] `tests/data/test_fixtures.py` — DATA-10 (auto-test del propio conftest) (Task 3)
- [ ] `Makefile` target `test-data` — arranca compose db + corre `pytest tests/data/` (Task 2; verify usa grep corregido)
- [ ] `.env.example` entry `NEXO_AUTO_MIGRATE=false` con comentario "solo dev" (Task 2; verify usa grep corregido)
- [ ] `nexo/data/schema_guard.py` — `verify(engine, critical_tables=CRITICAL_TABLES)` kwarg signature (Task 1)

**En 03-02 (capa MES) — Task 1 (Wave 1 code, PRIMERO, antes del checkpoint):**
- [ ] `scripts/gen_pdf_reference.py` — graba `tests/data/reference/pipeline_<fecha>.pdf` + `.sha256` (Task 1 — debe existir ANTES de Task 2 checkpoint)
- [ ] `scripts/pdf_regression_check.py` — comparador post-refactor hash+size (Task 1)
- [ ] `tests/data/reference/.gitkeep` (Task 1; los PDFs binarios pueden ir gitignored según peso)
- [ ] `nexo/data/repositories/mes.py` + `nexo/data/dto/mes.py` + `nexo/data/repositories/__init__.py` (Task 1)

**En 03-02 (capa MES) — Task 2 (checkpoint humano, DESPUÉS de Task 1):**
- [ ] Ejecución manual de `gen_pdf_reference.py --fecha=2026-03-15` antes de tocar routers. Si se difiere: TODO con deadline 7 días en STATE.md + nota en SUMMARY.md (fallback REFORZADO per revision).

**En 03-02 (capa MES) — Task 3-4 (durante/después del refactor):**
- [ ] 11 archivos `.sql` bajo `nexo/data/sql/mes/` (Task 3)
- [ ] `tests/data/test_mes_repository.py` — DATA-02 (Task 4)
- [ ] `tests/data/test_mes_engine_context.py` — DATA-09 smoke (DB_NAME() = dbizaro) (Task 4)
- [ ] `tests/data/test_no_three_part_names.py` — meta-test grep-based sobre `api/` (Task 4)
- [ ] `tests/data/test_bbdd_whitelist.py` — D-05 contract test (Task 4)

**NOTA DE HANDOFF (Blocker-1 revision)**: `api/services/pipeline.py` NO está en `files_modified` de 03-02. Propiedad exclusiva de 03-03 Task 4.7.

**En 03-03 (capa APP+NEXO) — durante el refactor:**
- [ ] `tests/data/test_app_repository.py` — DATA-03 (Task 5)
- [ ] `tests/data/test_nexo_repository.py` — DATA-04 (Task 5)
- [ ] `tests/data/test_routers_smoke.py` — DATA-07 (10 routers parametrizados) (Task 5)
- [ ] `tests/data/test_no_raw_pyodbc_in_routers.py` — meta-test final (Task 5)
- [ ] `api/services/pipeline.py` editado atómicamente en Task 4.7 (ORM migration via re-export + MES handoff de 03-02, Opción A o B documentada).
- [ ] DATA-05 compliance: 0 archivos `.sql` bajo `nexo/data/sql/{app,nexo}/` (ORM puro per D-01); verificación automatizada en Task 3.

**Framework install:** ninguno — pytest 8.3.4, httpx 0.28.1, pytest-cov 6.0.0 ya en `requirements-dev.txt`.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Pipeline OEE genera PDFs visualmente correctos contra MES real | Success #5 | Requiere SQL Server real con datos de fecha histórica; CI no tiene SQL Server | (1) Operador corre `make up` en preprod con `.env` real. (2) `python scripts/gen_pdf_reference.py --fecha=2026-03-15` ANTES de mergear 03-02. (3) Tras merge: `python scripts/pdf_regression_check.py --fecha=2026-03-15`. (4) Si hash difiere: revisar visualmente el PDF generado y comparar página a página con el de referencia. **Si se difiere**: STATE.md TODO con deadline 7 días + 03-02 SUMMARY.md gate-pending (fallback reforzado per revision). |
| `engine_mes` conecta contra dbizaro real con read-only-by-convention | DATA-01 | Smoke contra SQL Server real, no mockeable | Operador en preprod: `docker compose exec web python -c "from nexo.data.engines import engine_mes; from sqlalchemy import text; print(engine_mes.connect().execute(text('SELECT DB_NAME()')).scalar())"` debe imprimir `dbizaro`. |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (distribución entre 03-01/02/03 verificada post-revision; Task ordering en 03-02 corregido)
- [x] No watch-mode flags (pytest sin `-f` o `--watch`)
- [x] Feedback latency < 60s para quick run
- [x] `nyquist_compliant: true` set in frontmatter (flipped 2026-04-19 post-revision)
- [x] `wave_0_complete: true` set in frontmatter (flipped 2026-04-19 post-revision)
- [x] No file-overlap en `files_modified` entre planes del mismo wave (Blocker-1 fix: `api/services/pipeline.py` solo en 03-03)

**Approval:** green (flags flipped tras revision del plan-checker 2026-04-19). Plan listo para execute.
