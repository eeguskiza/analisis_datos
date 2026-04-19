# Phase 3: Capa de datos — Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisiones canónicas en CONTEXT.md — este log preserva las alternativas consideradas.

**Date:** 2026-04-19
**Phase:** 03-capa-de-datos
**Areas discussed:** SQL loader + `.sql` layout, Repository shape, Routers refactor — plan breakdown, Pipeline OEE + `/bbdd` edge cases
**Mode:** usuario delegó decisiones ("elige tu por que habia un plan de 7 fases no se como vamos"). 4 áreas aceptadas en bloque.

---

## Gate inicial — CONTEXT.md gate

| Option | Description | Selected |
|--------|-------------|----------|
| Run discuss-phase first (Recommended) | Ejecutar /gsd-discuss-phase 3 antes del planner. | ✓ |
| Continue without context | Planear desde ROADMAP + REQUIREMENTS + research solo. | |

**User's choice:** Run discuss-phase first.

---

## Selección de gray areas (multiSelect)

| Option | Description | Selected |
|--------|-------------|----------|
| SQL loader + `.sql` layout | Placeholders `?` vs `:named`, Jinja, granularidad. | ✓ |
| Repository shape (sesión + retornos) | Inyección, DTOs vs ORM, transacciones. | ✓ |
| Routers refactor — plan breakdown | Monolítico vs 3 plans por engine. | ✓ |
| Pipeline OEE + `/bbdd` explorer | Wrapper superficial vs rewrite, edge cases. | ✓ |

**User's choice:** las 4. Nota: "elige tu por que habia un plan de 7 fases no se como vamos".

---

## Area 1 — SQL loader + `.sql` layout

**Presented recommendation:**
- Placeholders: SQLAlchemy `text()` con `:named` (corrige DATA-05 que menciona `?`).
- Layout: `nexo/data/sql/<engine>/<method>.sql` — un archivo por método.
- Loader: `lru_cache` + `importlib.resources`.
- IN-clauses: `bindparam("codes", expanding=True)`.
- Sin Jinja en los `.sql`.

**Alternatives considered (not picked):**
- `?` posicionales a pelo vía pyodbc — rompe en psycopg2; menos legible.
- Jinja para branching dinámico — introduce ambigüedad SQL/texto + vector de inyección si no se sanitiza.
- `dbt` / SQL as templates — overkill para 30 queries.
- Un único archivo `.sql` por router — rompe el 1:1 método/archivo y complica testing.

**User's choice:** Accept recommendation.

---

## Area 2 — Repository shape (sesión + retornos)

**Presented recommendation:**
- Sesión inyectada por `Depends(get_db_*)`, repos sin transacción.
- DTOs Pydantic frozen en `nexo/data/dto/` para cualquier tipo que cruce un endpoint.
- ORM entities solo intra-`nexo/services/`.
- Naming `*Row` (ProduccionRow, RecursoRow, etc.).
- `AuditRepo.append` solo INSERT (coherente con gate IDENT-06 del rol `nexo_app`).

**Alternatives considered (not picked):**
- Repo autogestiona sesión (`@contextmanager def session()`) — oculta el lifecycle, dificulta testing.
- Repo orquesta transacción (commit interno) — quita control al caller, rompe atomicidad en servicios que tocan varios repos.
- Devolver ORM entities al router — fuga del modelo interno al wire format; acopla router al schema.
- Dict/tuple raw sin DTO — sin validación tipada, difícil mantener.

**User's choice:** Accept recommendation.

---

## Area 3 — Routers refactor — plan breakdown

**Presented recommendation (LOCKED):**
- **03-01 foundation**: engines.py + loader + DTO scaffolding + schema_guard + fixtures. Sin tocar routers.
- **03-02 capa MES**: MesRepository + 5 routers MES (bbdd, capacidad, operarios, centro_mando, luk4). Cierra DATA-09.
- **03-03 capa APP + NEXO**: RecursoRepo/CicloRepo/... + UserRepo/RoleRepo/AuditRepo + refactor de `historial/recursos/ciclos` + `nexo/services/auth.py` + `auditoria/usuarios` routers.
- Paralelismo: 03-02 y 03-03 pueden ir a la vez tras 03-01.

**Alternatives considered (not picked):**
- 1 plan monolítico — commit gigante, reversibilidad imposible, review infernal.
- 8 plans (1 por router) — overhead de infra (cada uno necesita engines/loader) + mucha coordinación.
- Split por tabla (1 plan por repo) — 10 plans, demasiado granular, overhead GSD.
- Split por dominio funcional (auth vs dashboard vs pipeline) — cruza engines y complica el gate DATA-09.

**User's choice:** Accept recommendation.

---

## Area 4 — Pipeline OEE + `/bbdd` explorer

**Presented recommendation:**
- Pipeline: `OEE/db/connector.py` NO se reescribe; `MesRepository.extraer_datos_produccion` delega al connector existente. `api/services/pipeline.py` cambia solo el import.
- Risk guard: antes de refactor en 03-02, hash de PDF de una fecha conocida. Post-refactor: mismo input → mismo hash.
- `/bbdd`: whitelist anti-DDL se queda en el router; `MesRepository.consulta_readonly` recibe SQL ya validada. Ops metadata (`list_databases`, `list_tables`, etc.) se quedan inline en `bbdd.py` contra `engine_mes`.

**Alternatives considered (not picked):**
- Reescribir `extraer_datos` dentro del repo — riesgo alto de romper success criterion #5, coste alto.
- Eliminar `OEE/db/connector.py` completamente — quitaría el núcleo MES funcional; diferido a Mark-IV.
- Mover whitelist al repo — mezcla transport validation con data access.
- Método de repo por cada op metadata (`list_tables`, `list_columns`) — no reusable, solo consumido por el explorer.

**User's choice:** Accept recommendation.

---

## Claude's Discretion

- Orden exacto de refactor de routers dentro de 03-02 y 03-03.
- Shape concreto de DTOs (campos, validadores Pydantic).
- Nombre final del loader (`load_sql` vs `sql` vs `get_sql`).
- Fecha histórica con datos estables para el hash guard del pipeline.
- Si el shim en `nexo/db/engine.py` se elimina en 03-01 o se difiere al último plan.

## Deferred Ideas

- Validación de columnas + tipos en `schema_guard` → Mark-IV.
- `engine_mes` con usuario read-only a nivel DB → Mark-IV.
- Rename de carpeta `OEE/` → Mark-IV (CLAUDE.md lo confirma).
- Refactor interno de los 4 módulos OEE → out of scope Mark-III.
- Mover `cfg.*` / `oee.*` a Postgres → Power BI/IoT lo impiden.
- Alembic → Mark-IV DevEx.
- Sustituir matplotlib → solo si Phase 4 preflight lo demuestra.
- CI con GitHub Actions ejecutando tests de repos → Phase 7.
- Unificar `data/ecs-logo.png` con `static/img/brand/ecs/logo.png` → cleanup oportunista en 03-01.
