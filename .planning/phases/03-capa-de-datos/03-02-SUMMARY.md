# Summary — Plan 03-02: capa MES (MesRepository + refactor 5 routers + DATA-09)

**Phase**: 3 (Capa de datos — Sprint 2 Mark-III)
**Plan**: 03-02 of 03
**Ejecutado**: 2026-04-19
**Rama**: `feature/Mark-III`
**Modo**: `/gsd-execute-phase 3 --wave 2` (sequential, 03-02 primero)
**Total commits**: 10 (9 atómicos del plan + 1 fix out-of-plan + esta SUMMARY)

---

## Commits

| # | Hash | Tipo | Mensaje |
|---|------|------|---------|
| 1 | `2d48d99` | feat | create MesRepository + 5 MES DTOs + PDF regression scripts (DATA-02, DATA-08) |
| 2 | `ca7076c` | fix | patch PDF scripts — sys.path shim + correct run_pipeline contract (out-of-plan, Rule-1 deviation) |
| 3 | `e369624` | feat | add sql files for MES queries (DATA-05) — 12 archivos en `nexo/data/sql/mes/` |
| 4 | `d0467bb` | refactor | kill 3-part names in centro_mando via MesRepository (DATA-09) |
| 5 | `6427d09` | refactor | capacidad uses engine_mes + load_sql (no pyodbc) |
| 6 | `87604e2` | refactor | operarios via engine_mes + load_sql (no pyodbc) |
| 7 | `45a6ab3` | refactor | luk4 uses engine_app from nexo.data.engines (luk4.* lives in APP; text() inline per D-01 ORM scope) |
| 8 | `aa487d0` | refactor | bbdd via MesRepository.consulta_readonly + whitelist preserved (D-05) |
| 9 | `16cb671` | refactor | repoint api/services/db.py imports to MesRepository (pipeline.py untouched; handoff to 03-03) |
| 10 | `b255c8f` | test | MES repository + engine context + meta-tests + whitelist contract |
| 11 | TBD | docs | (esta SUMMARY + bind-mount tests/ en compose) |

---

## Hard gate — resultados

| # | Check | Resultado |
|---|-------|-----------|
| 1 | `MesRepository` + 5 DTOs (`ProduccionRow`, `EstadoMaquinaRow`, `CicloRealRow`, `CapacidadRow`, `OperarioRow`) importables | ✅ |
| 2 | 12 archivos `.sql` bajo `nexo/data/sql/mes/` | ✅ |
| 3 | `pytest tests/data/test_mes_repository.py` | ✅ 11 passed |
| 4 | `pytest tests/data/test_no_three_part_names.py` | ✅ 3 passed |
| 5 | `pytest tests/data/test_bbdd_whitelist.py` | ✅ 16 passed |
| 6 | `pytest tests/data/test_mes_engine_context.py` (DATA-09: DB_NAME() == dbizaro, 2-part `admuser.fmesmic` works) | ✅ 2 passed |
| 7 | `pytest tests/auth/` regression (IDENT-06 intacto) | ✅ 11 passed / 1 skipped |
| 8 | `grep -rE "dbizaro\.\w+" api/` | ✅ 0 lines |
| 9 | `grep "import pyodbc"` en los 5 routers refactorizados | ✅ 0 per router (bbdd.py mantiene pyodbc solo para metadata ops, permitido per D-05) |
| 10 | `api/services/pipeline.py` modificado en este plan | ✅ 0 líneas (handoff a 03-03 respetado) |
| 11 | NOTA T3 preservado en `extraer_datos_produccion.sql` | ✅ |
| 12 | Smoke HTTP 5 routers MES (sin cookie) | ✅ todos 401 (auth gate intacto, endpoints route) |
| 13 | **PDF regression check (success criterion #5)** | 🟡 **DEFERIDO** — ver "Riesgos abiertos" |

---

## Riesgos abiertos

### 🟡 PDF regression baseline diferido (success criterion #5)

**Qué pasó**: el baseline pre-refactor se grabó dentro del container (`/app/tests/data/reference/pipeline_2026-03-15.pdf`, sha256 `719943e96c6b2301...`, 89 PDFs nuevos generados, source `/app/informes/2026-04-19/GENERAL/GENERAL_oee_seccion.pdf`). Pero `tests/` NO estaba bind-mounted al host en `docker-compose.yml`. Cuando hicimos `docker compose up -d --build web` para meter los commits del refactor en la imagen, la nueva imagen reemplazó al container y el PDF baseline se perdió. El regression check `pdf_regression_check.py` no encuentra el archivo y falla con "ERROR: baseline ausente".

**Mitigación aplicada en este SUMMARY**: añadido `./tests:/app/tests` al bind-mount del servicio `web` en `docker-compose.yml`. A partir de ahora cualquier baseline grabado por `gen_pdf_reference.py` persiste en el host y sobrevive rebuilds.

**Análisis de riesgo del refactor**: bajo. El refactor 03-02 es mecánico:
- Mover queries inline a archivos `.sql` cargados con `load_sql()`.
- Reemplazar `pyodbc.connect()` directo por `engine_mes.connect()` o `MesRepository`.
- Eliminar 3-part names `dbizaro.admuser.X` por 2-part `admuser.X` (engine_mes ya tiene `DATABASE=dbizaro`).
- `api/services/pipeline.py` NO se tocó (handoff a 03-03 — D-04: `MesRepository` son wrappers delgados sobre `OEE.db.connector.*` que no se reescribe).

Por tanto el path de generación de PDFs es funcionalmente equivalente: pipeline → `OEE.db.connector.extraer_datos` (o vía `MesRepository` que delega) → mismo SQL → mismo DataFrame → mismos CSVs → mismos módulos OEE → mismos PDFs. El hash debería coincidir entre runs sucesivos del mismo input (ignorando timestamp jitter de matplotlib que produce WARN exit 1, aceptable per plan).

**TODO con deadline**: ver entrada en `.planning/STATE.md` "Deferred verifications".

```
- [ ] 03-02 PDF regression baseline check
  Deadline: 2026-04-26 (7 días desde merge)
  Operador: e.eguskiza@ecsmobility.com
  Comando (ahora con bind-mount, persiste rebuilds):
    docker compose up -d --build web
    docker compose exec -T web python scripts/gen_pdf_reference.py --fecha=2026-03-15
    docker compose exec -T web python scripts/pdf_regression_check.py --fecha=2026-03-15
  Criterio de aceptación:
    - exit 0 (hash idéntico) → cierre limpio
    - exit 1 (WARN, size ±5%, pages match) → aceptable; ojear 1-2 PDFs visualmente para confirmar
    - exit 2 (REGRESSION) → re-abrir 03-02, investigar diff y posiblemente revertir commits b255c8f..2d48d99
```

---

## Decisiones tomadas en ejecución

- **`api/services/db.py` — lazy import para romper ciclo**: cuando `api/services/db.py` se repointó a `MesRepository`, había riesgo de circular import (`api.services.db` ↔ `nexo.data.repositories.mes` ↔ `OEE.db.connector` ↔ `api.services.db.get_config`). Resuelto con import diferido dentro de la función, no a nivel módulo. Coherente con Pitfall #5 del RESEARCH.
- **`luk4.py` usa `engine_app`, no `engine_mes`**: las tablas `luk4.estado`, `luk4.tiempos_ciclo`, `luk4.alarmas`, `luk4.plano_zonas` viven en `ecs_mobility` (APP), no en `dbizaro` (MES). Refactor mantiene el `text()` inline en el handler porque las queries son específicas y no reusables. Per CONTEXT.md D-01 (clarificación DATA-05): ORM/text inline en routers no requiere `.sql` versionado si la query es local al router.
- **`bbdd.py` mantiene `pyodbc` para metadata ops**: `list_databases`, `list_tables`, `list_columns`, `preview` siguen usando pyodbc directo contra connection strings dinámicos (master, dbizaro, ecs_mobility). Solo `consulta_readonly` migró a `MesRepository`. Per D-05: whitelist anti-DDL queda en el router; el repo solo ejecuta SQL ya validada. Es la única excepción al "no pyodbc en routers".
- **Bind-mount `./tests:/app/tests` en compose**: añadido como mitigación al baseline perdido. No tiene efecto sobre operación normal — los tests siguen funcionando idéntico en CI y dev.
- **PDF baseline diferido**: el operador eligió cierre con gate-pending tras analizar el coste/beneficio (refactor mecánico sin tocar pipeline → bajo riesgo de regresión real; coste de re-grabar baseline + comparar = 10-15 min en preprod cuando haya tiempo).

---

## Deviations

- **[Rule 1 — bug en plan] PDF scripts asumían contrato incorrecto de `run_pipeline`**
  - Found during: Task 2 (operator-run baseline grab).
  - Issue: scripts pasaban `output_dir` a `run_pipeline()` pero la firma real es `run_pipeline(fecha_inicio, fecha_fin, modulos=None, source='db', recursos=None)` y devuelve un generator (SSE log lines). Además faltaba `sys.path` shim — `docker compose exec web python scripts/X.py` pone `/app/scripts` en `sys.path[0]`, no `/app`.
  - Fix: `scripts/gen_pdf_reference.py` y `scripts/pdf_regression_check.py` parcheados con (1) `sys.path.insert(0, _REPO_ROOT)`, (2) llamada correcta sin `output_dir`, (3) consumo del generator vía for-loop, (4) localización de PDFs nuevos vía diff `before/after` de `settings.informes_dir`.
  - Files modified: `scripts/gen_pdf_reference.py`, `scripts/pdf_regression_check.py`.
  - Commit: `ca7076c` (out-of-plan, fuera de la secuencia de tasks pero necesario para desbloquear Task 2).
  - Verificación: smoke real con MES — pipeline corrió completo, generó 89 PDFs, hash baseline registrado.

- **[Rule 1 — missing] bind-mount de `tests/` en compose**
  - Found during: Task 5 (operator-run regression check).
  - Issue: `docker-compose.yml` no bind-monta `tests/`. Al hacer `docker compose up -d --build web`, la nueva imagen reemplaza al container y los PDFs baseline grabados en `/app/tests/data/reference/` desaparecen.
  - Fix: añadido `./tests:/app/tests` al volumes del servicio `web`.
  - Files modified: `docker-compose.yml`.
  - Verificación: el operador tendrá baseline persistente en su próximo intento.

- **[Rule 2 — scope] luk4 usa text() inline en lugar de archivos `.sql`**
  - Found during: Task 3.5 (luk4 refactor).
  - Issue: el plan original mencionaba dos opciones (extraer a `.sql` vs mantener text() inline). Se eligió text() inline porque las queries son específicas del router (no reusables) y CONTEXT.md D-01 lo permite explícitamente para queries no compartidas.
  - Files modified: `api/routers/luk4.py`.
  - Documentado en commit `45a6ab3` con cita explícita de D-01.

- **[Rule 1 — circular import potencial] `api/services/db.py` lazy import**
  - Found during: Task 3g (repoint de `api/services/db.py`).
  - Issue: import directo de `MesRepository` en módulo top-level habría creado ciclo con `OEE/db/connector.py` → `api.services.db.get_config`.
  - Fix: import diferido dentro de la función que necesita el repo.
  - Files modified: `api/services/db.py`.
  - Commit: `16cb671`.
  - Verificación: `python -c "from api.services.db import ..."` y todos los smoke imports pasan.

**Total deviations:** 4 auto-fixed (3 Rule-1 + 1 Rule-2). **Impact:** nulo sobre comportamiento especificado. La estructura del compose ahora soporta el regression check correctamente.

---

## Archivos tocados

**Creados (15)**:
- `nexo/data/repositories/__init__.py`
- `nexo/data/repositories/mes.py`
- `nexo/data/dto/mes.py`
- `nexo/data/sql/mes/extraer_datos_produccion.sql` (con NOTA T3 header)
- `nexo/data/sql/mes/detectar_recursos.sql`
- `nexo/data/sql/mes/calcular_ciclos_reales.sql`
- `nexo/data/sql/mes/estado_maquina_live.sql`
- `nexo/data/sql/mes/centro_mando_fmesmic.sql` (2-part `admuser.fmesmic`, sin `dbizaro.`)
- `nexo/data/sql/mes/capacidad_piezas_linea.sql`
- `nexo/data/sql/mes/capacidad_ciclos_p10_180d.sql`
- `nexo/data/sql/mes/operarios_listar.sql`
- `nexo/data/sql/mes/operarios_ficha.sql`
- `nexo/data/sql/mes/bbdd_list_databases.sql`
- `nexo/data/sql/mes/bbdd_list_tables.sql`
- `nexo/data/sql/mes/bbdd_list_columns.sql`
- `scripts/gen_pdf_reference.py`
- `scripts/pdf_regression_check.py`
- `tests/data/reference/.gitkeep`
- `tests/data/test_mes_repository.py`
- `tests/data/test_mes_engine_context.py`
- `tests/data/test_no_three_part_names.py`
- `tests/data/test_bbdd_whitelist.py`

**Modificados (7)**:
- `api/routers/centro_mando.py` — MesRepository, kill 3-part names, bindparam expanding
- `api/routers/capacidad.py` — engine_mes + load_sql, no pyodbc
- `api/routers/operarios.py` — engine_mes + load_sql, no pyodbc
- `api/routers/luk4.py` — engine_app + text() inline
- `api/routers/bbdd.py` — MesRepository.consulta_readonly + whitelist preservado
- `api/services/db.py` — repoint a MesRepository (lazy import)
- `docker-compose.yml` — bind-mount `./tests:/app/tests`

**NO modificado (handoff a 03-03, verificado por git diff)**:
- `api/services/pipeline.py` — 0 líneas tocadas en este plan; ownership exclusivo de 03-03.

---

## Pre-condiciones para 03-03

- ✅ `nexo.data.repositories.mes.MesRepository` existe y funciona — 03-03 puede importarlo si lo necesita.
- ✅ `nexo.data.engines.{engine_mes, engine_app, engine_nexo}` operativos.
- ✅ `nexo.data.sql.loader.load_sql()` operativo.
- ✅ `nexo.data.dto.base.BaseRow` operativo.
- ✅ `nexo.data.schema_guard.verify()` wired en lifespan.
- ✅ `tests/data/conftest.py` provee `db_nexo`, `db_app`, `engine_mes_mock`.
- ✅ `api/services/db.py` repointed — facade para servicios externos.
- 🟡 `api/services/pipeline.py` PRISTINO — 03-03 lo modifica en Task 4.7 (atomic commit con ORM migration + MES handoff).

---

## Cierre de Plan 03-02

**Requirements cubiertos en este plan:**
- ✅ DATA-02 (`MesRepository` con 5 métodos)
- ✅ DATA-05 (queries en `.sql` versionados — parcial: MES queries cubiertas; APP/NEXO ORM-puros no requieren `.sql` per D-01)
- ✅ DATA-07 (5 routers MES sin pyodbc directo; bbdd.py mantiene pyodbc solo para metadata per D-05)
- ✅ DATA-08 (5 DTOs MES Pydantic frozen)
- ✅ DATA-09 (3-part `dbizaro.admuser.*` eliminados; 2-part funciona via engine_mes con DATABASE=dbizaro)

**Success criteria de Phase 3 cubiertos por 03-02:**
- ✅ §1 Routers MES no importan pyodbc (excepto bbdd.py para metadata, justificado).
- ✅ §2 Queries MES en `.sql` versionados.
- ✅ §3 Cross-database references `dbizaro.admuser.*` eliminadas.
- 🟡 §5 PDFs idénticos a Mark-II — diferido (ver "Riesgos abiertos"; refactor mecánico de bajo riesgo + bind-mount añadido para próximo intento).

**Siguiente:**
Plan 03-03 — capa APP + NEXO. Refactor de routers `historial`, `recursos`, `ciclos` + `nexo/services/auth.py` + `api/routers/{auditoria,usuarios}.py`. Migración de modelos ORM `api/database.py` → `nexo/data/models_app.py` y `nexo/db/models.py` → `nexo/data/models_nexo.py` con shims. Edit atómico de `api/services/pipeline.py` (Task 4.7).

---

*SUMMARY creado 2026-04-19 como parte de `/gsd-execute-phase 3 --wave 2`.*
*Plan 03-02 cerrado con gate diferido. PDF regression check pendiente para preprod (deadline 2026-04-26).*
