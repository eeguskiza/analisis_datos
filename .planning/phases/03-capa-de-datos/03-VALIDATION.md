---
phase: 3
slug: capa-de-datos
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-19
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

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
| DATA-01 | 03-01 | 0 | 3 engines con pool configurado correcto | unit | `pytest tests/data/test_engines.py -x` | ❌ W0 |
| DATA-01 | 03-01 | 1 | `engine_mes` apunta a `dbizaro` (DB_NAME()) | integration | `pytest tests/data/test_engines.py::test_engine_mes_database_is_dbizaro -x` | ❌ W0 |
| DATA-02 | 03-02 | 0 | `MesRepository` con 5 métodos (firma + delegación) | unit | `pytest tests/data/test_mes_repository.py -x` | ❌ W0 |
| DATA-02 | 03-02 | 1 | `extraer_datos_produccion` delega a `OEE.db.connector.extraer_datos` | unit (mock) | `pytest tests/data/test_mes_repository.py::test_extraer_datos_delega -x` | ❌ W0 |
| DATA-03 | 03-03 | 0 | 6 repos APP con contrato session-inyectada | integration | `pytest tests/data/test_app_repository.py -x` | ❌ W0 |
| DATA-04 | 03-03 | 0 | 3 repos NEXO (`UserRepo`, `RoleRepo`, `AuditRepo`) | integration | `pytest tests/data/test_nexo_repository.py -x` | ❌ W0 |
| DATA-04 | 03-03 | 1 | `AuditRepo.append` no comitea internamente | integration | `pytest tests/data/test_nexo_repository.py::test_audit_repo_no_commits -x` | ❌ W0 |
| DATA-05 | 03-01 | 0 | Loader carga `.sql` con `lru_cache` | unit | `pytest tests/data/test_sql_loader.py -x` | ❌ W0 |
| DATA-05 | 03-02 | 1 | Comentario T3 presente en `extraer_datos_produccion.sql` | unit | `pytest tests/data/test_sql_loader.py::test_t3_comment_preserved -x` | ❌ W0 |
| DATA-06 | 03-01 | 0 | `schema_guard.verify` falla con tabla faltante | integration | `pytest tests/data/test_schema_guard.py::test_verify_raises_missing -x` | ❌ W0 |
| DATA-06 | 03-01 | 1 | `NEXO_AUTO_MIGRATE=true` auto-crea | integration | `pytest tests/data/test_schema_guard.py::test_auto_migrate_creates -x` | ❌ W0 |
| DATA-07 | 03-02/03 | 2 | `grep "import pyodbc" api/routers/` = 0 | smoke (meta) | `pytest tests/data/test_no_raw_pyodbc_in_routers.py -x` | ❌ W0 |
| DATA-07 | 03-02 | 2 | `grep "dbizaro\\." api/` = 0 | smoke (meta) | `pytest tests/data/test_no_three_part_names.py -x` | ❌ W0 |
| DATA-07 | 03-02/03 | 2 | Los 8 routers responden 200 bajo auth válida | integration | `pytest tests/data/test_routers_smoke.py -x` | ❌ W0 |
| DATA-08 | 03-01 | 0 | DTOs son frozen y fail con mutation | unit | `pytest tests/data/test_dto_immutable.py -x` | ❌ W0 |
| DATA-09 | 03-02 | 1 | Query `centro_mando` sin `dbizaro.` (DB_NAME() = dbizaro) | integration | `pytest tests/data/test_mes_engine_context.py -x` | ❌ W0 |
| DATA-10 | 03-01 | 0 | Fixtures `db_nexo`, `db_app`, `engine_mes_mock` funcionan | integration | `pytest tests/data/test_fixtures.py -x` | ❌ W0 |
| DATA-11 | 03-01 | 0 | `engine_mes` pool: pool_recycle=3600, pool_pre_ping=True | unit | `pytest tests/data/test_engines.py::test_mes_pool_config -x` | ❌ W0 |
| Success #5 | 03-02 | gate | PDF generation pre/post refactor → hash o size match | regression | `python scripts/pdf_regression_check.py --fecha=2026-03-15` | ❌ W0 (script nuevo) |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Estos artefactos NO existen y deben ser creados antes de poder validar Phase 3. Distribuidos por plan:

**En 03-01 (foundation):**
- [ ] `tests/data/__init__.py` — empty marker
- [ ] `tests/data/conftest.py` — fixtures `db_nexo`, `db_app`, `engine_mes_mock` + skipifs para SQL Server en CI
- [ ] `tests/data/test_engines.py` — DATA-01 + DATA-11
- [ ] `tests/data/test_sql_loader.py` — DATA-05
- [ ] `tests/data/test_schema_guard.py` — DATA-06
- [ ] `tests/data/test_dto_immutable.py` — DATA-08
- [ ] `tests/data/test_fixtures.py` — DATA-10 (auto-test del propio conftest)
- [ ] `Makefile` target `test-data` — arranca compose db + corre `pytest tests/data/`
- [ ] `.env.example` entry `NEXO_AUTO_MIGRATE=false` con comentario "solo dev"

**En 03-02 (capa MES) — primera tarea, ANTES del refactor:**
- [ ] `scripts/gen_pdf_reference.py` — graba `tests/data/reference/pipeline_<fecha>.pdf` + `.sha256`
- [ ] `tests/data/reference/.gitkeep` (los PDFs binarios pueden ir gitignored según peso)
- [ ] Ejecución de `gen_pdf_reference.py --fecha=2026-03-15` antes de tocar router pipeline

**En 03-02 (capa MES) — durante/después del refactor:**
- [ ] `tests/data/test_mes_repository.py` — DATA-02 + DATA-09
- [ ] `tests/data/test_mes_engine_context.py` — DATA-09 smoke (DB_NAME() = dbizaro)
- [ ] `tests/data/test_no_three_part_names.py` — meta-test grep-based
- [ ] `scripts/pdf_regression_check.py` — comparador post-refactor (hash con fallback a size ±5%)

**En 03-03 (capa APP+NEXO) — durante el refactor:**
- [ ] `tests/data/test_app_repository.py` — DATA-03
- [ ] `tests/data/test_nexo_repository.py` — DATA-04
- [ ] `tests/data/test_routers_smoke.py` — DATA-07 (los 8 routers parametrizados)
- [ ] `tests/data/test_no_raw_pyodbc_in_routers.py` — meta-test final

**Framework install:** ninguno — pytest 8.3.4, httpx 0.28.1, pytest-cov 6.0.0 ya en `requirements-dev.txt`.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Pipeline OEE genera PDFs visualmente correctos contra MES real | Success #5 | Requiere SQL Server real con datos de fecha histórica; CI no tiene SQL Server | (1) Operador corre `make up` en preprod con `.env` real. (2) `python scripts/gen_pdf_reference.py --fecha=2026-03-15` ANTES de mergear 03-02. (3) Tras merge: `python scripts/pdf_regression_check.py --fecha=2026-03-15`. (4) Si hash difiere: revisar visualmente el PDF generado y comparar página a página con el de referencia. |
| `engine_mes` conecta contra dbizaro real con read-only-by-convention | DATA-01 | Smoke contra SQL Server real, no mockeable | Operador en preprod: `docker compose exec web python -c "from nexo.data.engines import engine_mes; from sqlalchemy import text; print(engine_mes.connect().execute(text('SELECT DB_NAME()')).scalar())"` debe imprimir `dbizaro`. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (18 archivos + 1 Makefile entry + 1 .env entry)
- [ ] No watch-mode flags (pytest sin `-f` o `--watch`)
- [ ] Feedback latency < 60s para quick run
- [ ] `nyquist_compliant: true` set in frontmatter (cuando se cierre el plan checker)

**Approval:** pending (se aprueba al pasar el plan-checker en Phase 3 plan-phase).
