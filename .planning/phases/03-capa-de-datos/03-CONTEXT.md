# Phase 3: Capa de datos — Context

**Gathered:** 2026-04-19
**Status:** Ready for planning
**Source:** síntesis de `docs/MARK_III_PLAN.md` §Sprint 2, `.planning/REQUIREMENTS.md` DATA-01..DATA-11, `.planning/ROADMAP.md` Phase 3 + /gsd-discuss-phase sesión 2026-04-19.

<domain>
## Phase Boundary

**Objetivo**: eliminar SQL embebido en los 8 routers de dominio y aislar las tres fuentes de datos detrás de una capa de repositorios en `nexo/data/`.

**Entregable end-to-end**:
- `nexo/data/engines.py` con tres engines explícitos: `engine_mes` (dbizaro read-only, pool DATA-11), `engine_app` (ecs_mobility, reaprovecha el existente en `api/database.py`), `engine_nexo` (Postgres, movido desde `nexo/db/engine.py`).
- `nexo/data/sql/{mes,app,nexo}/*.sql` — un archivo por método de repo, cargados por `nexo/data/sql/loader.py` con `lru_cache`.
- `nexo/data/repositories/{mes,app,nexo}.py` con los 10 repos definidos en DATA-02/03/04.
- `nexo/data/dto/` con DTOs Pydantic `*Row` por cada tipo que cruza un endpoint.
- `nexo/data/schema_guard.py` ejecutado en `lifespan`; flag `NEXO_AUTO_MIGRATE=true` crea lo que falte.
- Los 8 routers (`bbdd`, `capacidad`, `operarios`, `centro_mando`, `luk4`, `historial`, `recursos`, `ciclos`) consumen repos; se elimina `import pyodbc` directo y todo SQL inline en Python.
- `nexo/services/auth.py`, `api/routers/auditoria.py`, `api/routers/usuarios.py` reemplazan queries inline por `UserRepo` / `RoleRepo` / `AuditRepo`.
- DATA-09 cierra: eliminación de 3-part names `dbizaro.admuser.*` vía engine dedicado (connection string con `DATABASE=dbizaro`).
- `tests/data/` con tests por repo contra Postgres real (`docker compose up db`) + mocks para `engine_mes`.

**Qué NO entra en esta phase** (ver ROADMAP.md / CLAUDE.md):
- Rename de `OEE/` a `modules/oee/` — diferido a Mark-IV (`docs/CLAUDE.md` lo reafirma).
- Refactor interno de los 4 módulos `OEE/{disponibilidad,rendimiento,calidad,oee_secciones}` — funcionan hoy, success criterion #5 exige PDFs idénticos.
- Mover `cfg.recursos`/`cfg.ciclos`/`cfg.contactos` a Postgres — permanecen en `ecs_mobility.cfg.*` (Power BI + túnel IoT los leen).
- Preflight/postflight de queries pesadas — Phase 4.
- UI por roles / split de `ajustes.html` — Phase 5.

</domain>

<decisions>
## Implementation Decisions

### D-01 — SQL loader (LOCKED)
- **Placeholders**: SQLAlchemy `text()` con params `:named`. DATA-05 dice "placeholders `?`" pero la implementación correcta es `:named` (soportado uniformemente por pyodbc/SQL Server y psycopg2/Postgres).
- **Layout**: `nexo/data/sql/<engine>/<method_name>.sql` — un archivo = una query canónica. Ej: `nexo/data/sql/mes/extraer_datos_produccion.sql`.
- **Loader**: `nexo/data/sql/loader.py:load_sql(name: str) -> str` con `@lru_cache`, lee vía `importlib.resources`.
- **IN-clauses dinámicos**: `text(...).bindparams(bindparam("codes", expanding=True))`. Sin Jinja.
- **Branching**: si una query necesita dos formas, dos archivos. `.sql` puros.
- **Comentarios preservados**: el filtro T3 que cruza medianoche (en la query MES actual) va como header `-- NOTA T3: ...` del archivo correspondiente.
- **Scope DATA-05** (clarificación): los archivos `.sql` versionados aplican a queries con `text()` (queries MES y queries APP/NEXO complejas o reutilizables). Las queries ORM puras vía `session.query(Model).filter(...)` NO requieren archivo `.sql` — el modelo declarativo SQLAlchemy ES la representación canónica. Repos APP/NEXO que solo hacen ORM (`RecursoRepo`, `CicloRepo`, `UserRepo`, etc.) cumplen DATA-05 sin generar archivos `.sql`. La intención del requisito es eliminar SQL hardcoded en strings de Python, no forzar `.sql` redundantes para ORM.

### D-02 — Repository shape (LOCKED)
- **Sesión inyectada, repos sin transacción**. Cada repo recibe `Session` (para `engine_app`/`engine_nexo`) o `Engine` (para `engine_mes` read-only) en `__init__`.
- Repos NO hacen `commit()` ni `rollback()`. Transacción pertenece al caller (router via `Depends(get_db_*)`, o service).
- Nuevas dependencies en `api/deps.py`: `get_db_app()`, `get_db_nexo()`, `get_engine_mes()`.
- **Retornos**:
  - DTOs Pydantic **frozen** en `nexo/data/dto/` para cualquier tipo que cruza un endpoint HTTP.
  - ORM entities (`NexoUser`, `NexoAuditLog`, `Recurso`, `Ciclo`, …) solo circulan intra-`nexo/services/`. Los routers nunca reciben ORM.
  - MES repo siempre devuelve DTOs (no hay ORM de SQL Server en Nexo).
- **Naming DTO**: sufijo `Row` (`ProduccionRow`, `RecursoRow`, `CapacidadRow`, `AuditLogRow`, `CicloRow`, `OperarioRow`, `EstadoMaquinaRow`).
- `AuditRepo.append(...)` solo hace INSERT; caller orquesta transacción. Coherente con gate IDENT-06 (rol `nexo_app` tiene INSERT/SELECT, no UPDATE/DELETE).

### D-03 — Plan breakdown (LOCKED) — 3 plans
- **03-01 foundation**:
  - Mueve `engine_nexo` de `nexo/db/engine.py` a `nexo/data/engines.py`. Shim en `nexo/db/engine.py` re-exporta para no romper `nexo/services/auth.py` hasta el refactor de `03-03`.
  - Añade `engine_mes` con `pool_recycle=3600`, `pool_pre_ping=True`, `timeout=15` (DATA-11). Read-only a nivel de código (SELECT-only) en Mark-III; enforcement a nivel DB usuario queda diferido.
  - Referencia a `engine_app` desde `api/database.py` (no se duplica el engine SQL Server; se expone como `engine_app` en `nexo/data/engines.py` vía re-export).
  - `nexo/data/sql/loader.py` + `nexo/data/dto/` base (módulos vacíos listos para rellenar).
  - `nexo/data/schema_guard.py` wired en `lifespan` de `api/main.py`. Comportamiento: al arrancar, inspecciona schema `nexo` (tablas críticas: `users`, `roles`, `departments`, `user_departments`, `permissions`, `sessions`, `login_attempts`, `audit_log`). Si falta tabla → falla arranque con mensaje claro. Si `NEXO_AUTO_MIGRATE=true` → crea las que faltan con `NexoBase.metadata.create_all(bind=engine_nexo)` y continúa. Columnas y tipos NO se validan en Mark-III (gap aceptable; si hay drift aparece como runtime error en la primera query).
  - `tests/data/conftest.py` + fixture de Postgres (docker compose `db` up en CI) + fixture de mock para `engine_mes`.
  - Sin tocar routers ni `nexo/services/auth.py` todavía.
- **03-02 capa MES** (puede correr en paralelo con 03-03 tras 03-01):
  - `nexo/data/repositories/mes.py` con `MesRepository` (5 métodos de DATA-02: `extraer_datos_produccion`, `detectar_recursos`, `calcular_ciclos_reales`, `estado_maquina_live`, `consulta_readonly`).
  - `MesRepository.extraer_datos_produccion` / `detectar_recursos` / `calcular_ciclos_reales` / `estado_maquina_live` son **wrappers delgados** que delegan a `OEE.db.connector.*` existente (ver D-04 abajo). No se reescribe la lógica del connector.
  - Refactor de 5 routers: `bbdd.py`, `capacidad.py`, `operarios.py`, `centro_mando.py`, `luk4.py`. Eliminar `import pyodbc` directo y 3-part names `dbizaro.admuser.*` (DATA-09).
  - Tests: `tests/data/test_mes_repository.py` con engine MES mockeado (no queremos depender de SQL Server en CI).
- **03-03 capa APP + NEXO** (paralelo con 03-02 tras 03-01):
  - `nexo/data/repositories/app.py` con `RecursoRepo`, `CicloRepo`, `EjecucionRepo`, `MetricaRepo`, `LukRepo`, `ContactoRepo` (DATA-03).
  - `nexo/data/repositories/nexo.py` con `UserRepo`, `RoleRepo`, `AuditRepo` (DATA-04).
  - Refactor de routers APP: `historial.py`, `recursos.py`, `ciclos.py`.
  - Refactor de `nexo/services/auth.py`, `api/routers/auditoria.py`, `api/routers/usuarios.py` para consumir `UserRepo`/`RoleRepo`/`AuditRepo`.
  - Tests: `tests/data/test_app_repository.py` + `tests/data/test_nexo_repository.py` contra Postgres real del compose.
- **Paralelismo**: 03-02 y 03-03 tocan repos y routers disjuntos, pueden paralelizarse. Ambos dependen de 03-01.
- **Reversibilidad**: cada plan = commit atómico. Si 03-02 rompe MES, se revierte sin afectar a 03-03.

### D-04 — Pipeline OEE (LOCKED) — wrapper superficial
- `OEE/db/connector.py` (847 LOC, núcleo del pipeline) **no se reescribe** en Phase 3. Se mantienen `extraer_datos`, `detectar_recursos`, `calcular_ciclos_reales`, `estado_maquina_live`, `_build_connection_string`, `load_config`, `save_config`, `detectar_driver`.
- `MesRepository.extraer_datos_produccion(...)` → delega a `OEE.db.connector.extraer_datos(...)`.
- `MesRepository.detectar_recursos(...)` → delega a `OEE.db.connector.detectar_recursos(...)`.
- `MesRepository.calcular_ciclos_reales(...)` → delega a `OEE.db.connector.calcular_ciclos_reales(...)`.
- `MesRepository.estado_maquina_live(...)` → delega a `OEE.db.connector.estado_maquina_live(...)`.
- `api/services/pipeline.py` y `api/services/db.py` cambian el import (`OEE.db.connector` → `nexo.data.repositories.mes.MesRepository`). La lógica del pipeline no se toca.
- **Risk guard** (regression check success criterion #5): antes del refactor, generar un PDF con fecha conocida y hashear. Tras el refactor, repetir con mismos inputs y comparar hash. Si no hay fecha reproducible disponible, se compara count/total de páginas + tamaño en bytes. Gate en 03-02 antes de cerrar el plan.

### D-05 — `/bbdd` explorer (LOCKED)
- Whitelist anti-DDL/DML se queda en `api/routers/bbdd.py` (el router valida input del usuario; la validación es concern del transporte, no del repo).
- `MesRepository.consulta_readonly(sql: str, database: str) -> list[dict]` recibe la SQL **ya validada** y ejecuta contra `engine_mes` (o un engine puntual para el `database` solicitado si el explorer permite cambiar de catalog — ver código actual de `bbdd._get_conn_string`).
- Las ops de metadata del explorer (`list_databases`, `list_tables`, `list_columns`, `preview`) quedan inline en `bbdd.py` usando `engine_mes` directamente. Son específicas de UI, no se reusan desde otros routers, no justifican método de repo.
- DDL del SQL de usuario: el check actual (`re.match` anti-`INSERT|UPDATE|DELETE|DROP|...`) se conserva; se testea con un unit test que inyecte SQL malicioso y verifique rechazo (test explícito nuevo en 03-02).

### D-06 — Retrofit de routers (LOCKED)
- Orden de refactor dentro de cada plan: empezar por el más simple del grupo, escalar.
  - 03-02: `centro_mando.py` (2 queries) → `capacidad.py` (3 queries) → `operarios.py` (6 queries) → `luk4.py` → `bbdd.py` (más complejo, 10+ queries + SQL libre).
  - 03-03: `ciclos.py` → `recursos.py` → `historial.py` → luego `nexo/services/auth.py` + `auditoria.py` + `usuarios.py`.
- Cada router refactorizado pasa smoke HTTP antes de continuar con el siguiente.

### D-07 — schema_guard comportamiento (LOCKED)
- **Valida**: existencia de tablas críticas del schema `nexo`. NO valida columnas ni tipos (gap aceptable en Mark-III).
- **Flag `NEXO_AUTO_MIGRATE`**:
  - `false` (default): si falta tabla → `RuntimeError("Schema guard: missing table nexo.X, run `make nexo-init`")` y el lifespan aborta.
  - `true`: si falta tabla → `NexoBase.metadata.create_all(bind=engine_nexo)` y log `WARN: auto-migrated N tables`. Usar solo en dev / primera instalación.
- **Scope**: solo schema `nexo`. No valida `ecs_mobility.cfg.*` (fuera de control de Nexo) ni `dbizaro` (read-only).
- Ubicación: `nexo/data/schema_guard.py`. Llamado desde `api/main.py:lifespan()` antes de `init_db()`.

### D-08 — Tests DB strategy (LOCKED, consistente con DATA-10)
- Postgres real vía `docker compose up db` para tests APP+NEXO (fixture de sesión que hace rollback, no truncate).
- Mocks para `engine_mes` (no queremos SQL Server en CI; queries MES se testean con fixtures de respuesta grabadas).
- `tests/data/conftest.py` provee `db_nexo`, `db_app`, `engine_mes_mock`.
- CI: `Makefile` target `make test-data` arranca compose, corre `pytest tests/data/`, apaga compose. GitHub Actions futuro (Phase 7) replicará.

### Claude's Discretion
- Orden exacto de refactor dentro de 03-02 y 03-03 (empezar por simpler; planner concreta).
- Forma de los DTOs (campos, validadores) — copiar del shape actual de dict que devuelven los routers.
- Naming interno del loader (`load_sql` vs `sql`) — planner decide.
- Si `lru_cache` del loader se invalida en test — probablemente no, los `.sql` son estáticos.
- Si se graba un PDF de referencia para el hash guard de D-04 — planner elige fecha histórica con datos estables.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Contratos y decisiones de producto
- `docs/MARK_III_PLAN.md` §Sprint 2 — plan detallado de capa de datos con riesgos y estimaciones.
- `docs/AUTH_MODEL.md` — contrato de auth consumido por `UserRepo` / `RoleRepo` / `AuditRepo`.
- `docs/DATA_MIGRATION_NOTES.md` — notas de migración de datos, caminos muertos (`data/oee.db`).
- `docs/OPEN_QUESTIONS.md` — decisiones cerradas sobre split SQL Server y Postgres.
- `docs/GLOSSARY.md` — términos MES/APP/Nexo usados en naming de engines y tablas.
- `CLAUDE.md` — reglas del repo, reafirma que `OEE/` no se renombra en Phase 3.

### Requisitos y roadmap
- `.planning/REQUIREMENTS.md` §DATA-01..DATA-11 — requisitos trazables.
- `.planning/ROADMAP.md` §Phase 3 — goal + 6 success criteria + dependencias.
- `.planning/PROJECT.md` §Key Decisions — constraints de producto (Postgres vs SQL Server).

### Estado del codebase (proyección de /gsd-map-codebase)
- `.planning/codebase/ARCHITECTURE.md` — layers actuales, puntos de integración.
- `.planning/codebase/STRUCTURE.md` — directorios, incluyendo `nexo/` ya creado en Phase 2.
- `.planning/codebase/CONVENTIONS.md` — estilo del proyecto (naming, async patterns).

### Cierre de Phase 2 (precondiciones de Phase 3)
- `.planning/phases/02-identidad-auth-rbac-audit/02-CONTEXT.md` — decisiones de auth.
- `.planning/phases/02-identidad-auth-rbac-audit/02-01-SUMMARY.md` — `engine_nexo` creado en `nexo/db/engine.py` (se mueve en 03-01).
- `.planning/phases/02-identidad-auth-rbac-audit/02-04-SUMMARY.md` — rol `nexo_app` con GRANT SELECT/INSERT (el engine Postgres ya conecta con privilegios limitados; `AuditRepo.append` solo puede INSERT).

### Código que se refactoriza
- `OEE/db/connector.py` — 847 LOC, núcleo MES, se envuelve con wrappers delgados en `MesRepository` (D-04).
- `api/database.py` — engine SQL Server (`engine`, `SessionLocal`) se re-expone como `engine_app` desde `nexo/data/engines.py`. Modelos ORM `Recurso`, `Ciclo`, `Ejecucion`, `DatosProduccion`, `Contacto`, `MetricaOEE`, `ReferenciaStats`, `IncidenciaResumen`, `InformeMeta` se mueven a `nexo/data/models_app.py` en 03-03.
- `nexo/db/engine.py` — contiene `engine_nexo`, se mueve a `nexo/data/engines.py` en 03-01 (shim re-export durante Phase 3, remove en Phase 4+).
- `nexo/db/models.py` — 8 modelos ORM del schema `nexo` (NexoUser, NexoRole, NexoDepartment, etc.); se mueven a `nexo/data/models_nexo.py` en 03-03.
- `api/routers/{bbdd,capacidad,operarios,centro_mando,luk4,historial,recursos,ciclos}.py` — consumidores del refactor.
- `nexo/services/auth.py` + `api/routers/{auditoria,usuarios}.py` — consumen `UserRepo` / `RoleRepo` / `AuditRepo` en 03-03.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `nexo/db/engine.py` — `engine_nexo` + `SessionLocalNexo` ya creados en 02-01. Se mueven a `nexo/data/engines.py` en 03-01.
- `nexo/db/models.py` — 8 modelos ORM del schema `nexo` ya definidos. Ya consumidos por `nexo/services/auth.py` y `api/routers/{auditoria,usuarios}.py`.
- `api/database.py:engine` — engine SQL Server ya existe con `pool_pre_ping=True, pool_size=5, max_overflow=10, pool_recycle=3600`. Se re-expone como `engine_app`.
- `OEE/db/connector.py` — 7 funciones públicas del módulo MES (`extraer_datos`, `detectar_recursos`, `calcular_ciclos_reales`, `estado_maquina_live`, `_build_connection_string`, `load_config`, `save_config`). Todas envueltas por `MesRepository`.
- `api/services/db.py` y `api/services/pipeline.py` — consumidores de `OEE.db.connector`; solo cambia el import tras el refactor.
- Gate de regression: `tests/test_oee_calc.py` y `tests/test_oee_helpers.py` (30 tests) validan la lógica OEE pura; no cubren `extraer_datos` pero sirven de safety net para `oee_secciones`.

### Established Patterns
- SQLAlchemy 2.0 — múltiples engines coexisten (ya hay dos en runtime: `engine` SQL Server + `engine_nexo` Postgres). Añadir un tercero (`engine_mes`) es patrón conocido.
- `Depends(get_db)` — ya usado en routers APP (`centro_mando.py`, `luk4.py`, etc.) para sesión SQL Server. Se replica el patrón con `get_db_app`, `get_db_nexo`, `get_engine_mes`.
- `text()` + bindparams — ya usado en `centro_mando.py` con placeholders dinámicos estilo `:ct0, :ct1, ...` generados via string interpolation. Se sustituye por `bindparam("codes", expanding=True)`.
- ORM queries internas en `nexo/services/auth.py` (`session.query(NexoUser).filter(...)`) — se mueven a `UserRepo` con la misma semántica.

### Integration Points
- `api/main.py:lifespan()` — ya llama `init_db()`. Se añade `schema_guard.verify()` antes. Si falla, el `yield` no se ejecuta y la app no arranca (FastAPI maneja el error en stdout).
- `api/deps.py` — ya tiene globals Jinja2 y `current_user`. Se añaden dependencies de DB.
- `Makefile` — targets `nexo-init`, `nexo-owner`, `nexo-verify`, `nexo-app-role`, `nexo-smoke` existen. Se añade `test-data` (arranca compose + pytest + apaga compose).
- `docker-compose.yml` — servicio `db` ya expone Postgres en host:5433. Fixture de tests usa host:5433 directo o `docker compose exec db`.

### Riesgos específicos
- **Success criterion #5 (PDFs idénticos a Mark-II)**: el wrapper delgado en D-04 lo protege, pero la validación exige un hash/count reproducible. Planner debe elegir una fecha histórica con datos estables y hashear páginas o bytes del PDF.
- **`engine_mes` en CI**: no hay SQL Server en CI. Tests MES van con mock (DATA-10). Integration real solo se corre en dev/preprod.
- **`NEXO_AUTO_MIGRATE=true` en prod**: peligroso (schema changes silenciosos). Solo se usa en dev/primer deploy; doc claro en `scripts/init_nexo_schema.py` y Makefile.
- **Imports circulares**: `nexo/data/repositories/nexo.py` importa de `nexo/data/engines.py`. `nexo/services/auth.py` importa de `nexo/data/repositories/nexo.py`. Evitar que `nexo/data/engines.py` importe de `nexo/services/*`.
- **Shim `nexo/db/engine.py`**: durante la transición 03-01, se deja un re-export `from nexo.data.engines import engine_nexo, SessionLocalNexo` para que `nexo/services/auth.py` siga funcionando hasta 03-03. Shim se elimina en 03-03.
- **3-part names en `centro_mando.py`**: DATA-09 exige eliminarlos. El SQL actual usa `FROM dbizaro.admuser.fmesmic` — tras mover a `engine_mes` (DATABASE=dbizaro en connection string), se reescribe como `FROM admuser.fmesmic`. Smoke test verifica que sigue devolviendo las mismas filas.

</code_context>

<specifics>
## Requisitos trazables (REQUIREMENTS.md)

- **DATA-01** → Plan 03-01 (engines.py).
- **DATA-02** → Plan 03-02 (MesRepository con 5 métodos).
- **DATA-03** → Plan 03-03 (RecursoRepo, CicloRepo, EjecucionRepo, MetricaRepo, LukRepo, ContactoRepo).
- **DATA-04** → Plan 03-03 (UserRepo, RoleRepo, AuditRepo).
- **DATA-05** → Plan 03-01 (loader) + Plans 03-02/03-03 (archivos `.sql` por método).
- **DATA-06** → Plan 03-01 (schema_guard + lifespan + NEXO_AUTO_MIGRATE).
- **DATA-07** → Plans 03-02 y 03-03 (refactor de 8 routers, 0 `import pyodbc` y 0 SQL inline fuera de archivos `.sql`).
- **DATA-08** → Plan 03-01 (DTO scaffolding) + Plans 03-02/03-03 (DTOs por repo).
- **DATA-09** → Plan 03-02 (kill `dbizaro.admuser.*`).
- **DATA-10** → Plan 03-01 (fixtures) + Plans 03-02/03-03 (tests por repo).
- **DATA-11** → Plan 03-01 (`engine_mes` con pool_recycle=3600, pool_pre_ping=True, timeout=15).

## Success criteria (ROADMAP.md Phase 3)

1. Routers no importan `pyodbc` directamente; todas las consultas pasan por repositorios.
2. Queries residen en `.sql` versionados bajo `nexo/data/sql/`; no hay SQL hardcoded en Python.
3. Cross-database references `dbizaro.admuser.*` eliminadas.
4. `schema_guard` en lifespan: si falta `nexo.users` (o cualquier tabla crítica), el arranque falla con mensaje claro.
5. Pipeline OEE sigue generando PDFs idénticos a Mark-II tras el refactor (sin regresión en el núcleo de cálculo).
6. Tests de repositorios pasan en CI contra un Postgres dedicado.

</specifics>

<deferred>
## Deferred Ideas

- Validación de columnas + tipos en `schema_guard` — Mark-IV (drift detector completo).
- `engine_mes` con usuario read-only real a nivel DB (hoy solo a nivel código) — Mark-IV.
- Rename `OEE/` → `modules/oee/` — Mark-IV (CLAUDE.md lo confirma).
- Refactor interno de los 4 módulos OEE (`disponibilidad`, `rendimiento`, `calidad`, `oee_secciones`) — fuera de scope Mark-III por diseño.
- Mover `cfg.*` / `oee.*` / `luk4.*` a Postgres — Power BI + IoT lo impiden, Mark-V+ si llegara a plantearse.
- Migración a Alembic para `schema_guard` — Mark-IV DevEx.
- Sustituir matplotlib por otro motor de PDF — solo si Phase 4 preflight demuestra que es inviable.
- CI con GitHub Actions ejecutando `make test-data` — Phase 7 (DevEx hardening).
- Unificar `data/ecs-logo.png` con `static/img/brand/ecs/logo.png` — nota de 02-01, sin prisa; puede hacerse en 03-01 como cleanup oportunista.

</deferred>

---

*Phase: 03-capa-de-datos*
*Context gathered: 2026-04-19 via /gsd-discuss-phase (user delegó decisiones, 4 áreas aceptadas en bloque).*
