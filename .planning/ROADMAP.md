# Roadmap: Nexo (Mark-III)

## Overview

Refactor estructural de "OEE Planta / analisis_datos" hacia **Nexo**, plataforma
interna de ECS Mobility. Estrategia strangler-fig en el mismo repo con historial
preservado. Mark-III cubre 7 sprints de trabajo focalizado (29-30 días, 10-12
semanas naturales) que van de rebrand + higiene a despliegue LAN productivo con
auth, RBAC, audit, capa de datos saneada y preflight para consultas pesadas.

Derivado de `docs/MARK_III_PLAN.md` con la reordenación `0 → 1 (identidad) →
2 (datos) → 3 (consultas) → 4 (UI) → 5 (deploy) → 6 (devex)`. Nomenclatura
de phases GSD: Phase N = Sprint (N-1) del plan.

## Phases

- [ ] **Phase 1: Naming + Higiene + CI** — Sprint 0: rebrand Nexo, limpieza, CI mínimo, exception handler sin traceback, audit de historial
- [ ] **Phase 2: Identidad (auth + RBAC + audit)** — Sprint 1: login, roles propietario/directivo/usuario + departamentos, middleware de auth y audit append-only
- [ ] **Phase 3: Capa de datos** — Sprint 2: repositorios, `.sql` versionados, schema_guard, separación `engine_mes` / `engine_app` / `engine_nexo`
- [ ] **Phase 4: Consultas pesadas** — Sprint 3: preflight + postflight + aprobación asíncrona + umbrales editables
- [ ] **Phase 5: UI por roles** — Sprint 4: sidebar y páginas condicionadas al rol, split de `ajustes.html`
- [ ] **Phase 6: Despliegue LAN HTTPS** — Sprint 5: `docker-compose.prod.yml`, Caddy con LE DNS-01 o cert interno, firewall, runbook de deploy
- [ ] **Phase 7: DevEx hardening** — Sprint 6: pre-commit, CI ampliado con cobertura, `docs/ARCHITECTURE.md`, `docs/RUNBOOK.md`, `docs/RELEASE.md`

## Phase Details

### Phase 1: Naming + Higiene + CI
**Goal**: Repo arrancable con marca "Nexo", higiene mínima (Zone.Identifier, archivos residuales, requirements pineados), CI en GitHub Actions, handler de excepciones que no filtra traceback, audit de historial git documentado.
**Depends on**: Nothing (first phase)
**Requirements**: NAMING-01, NAMING-02, NAMING-03, NAMING-04, NAMING-05, NAMING-06, NAMING-07, NAMING-08, NAMING-09, NAMING-10, NAMING-11, NAMING-12, NAMING-13, NAMING-14, NAMING-15
**Success Criteria** (what must be TRUE):
  1. `make build && make up && make health` devuelven OK con contenedores llamados `nexo-*`
  2. Título FastAPI, sidebar, README dicen "Nexo"; metadata OpenAPI consistente
  3. `.env.example` cubre `NEXO_*` (MES + APP + web + PG) y mantiene compat `OEE_*` durante Mark-III
  4. `.github/workflows/ci.yml` ejecuta lint/test/build/secrets en push a `feature/Mark-III` y PRs
  5. Forzar un 500 en desarrollo devuelve `{"error_id": "<uuid>", "message": "Internal error"}` — sin traceback en el body
  6. `docs/SECURITY_AUDIT.md` lista credenciales expuestas del historial (sin valores literales)
  7. `CLAUDE.md`, `docs/AUTH_MODEL.md`, `docs/GLOSSARY.md` existen con el contenido acordado; `docs/MARK_III_PLAN.md` y `docs/OPEN_QUESTIONS.md` actualizados
**Plans**: 1 plan (Sprint 0 se ejecuta como una secuencia de 12 commits atómicos especificados en el plan de arranque)

Plans:
- [ ] 01-01: Sprint 0 — 12 commits atómicos (junk, gitignore, odbc move, oee.db decision, env rename + split, UI rebrand, exception handler fix, requirements pin, CI workflow, docs core, security audit doc, MARK_III_PLAN/OPEN_QUESTIONS sync)

### Phase 2: Identidad (auth + RBAC + audit)
**Goal**: Toda request con `/api/*` o HTML autenticada; roles propietario/directivo/usuario con departamentos; cada acción registrada en tabla append-only a nivel BBDD.
**Depends on**: Phase 1 (env vars estables `NEXO_*`, exception handler seguro, Postgres configurable)
**Requirements**: IDENT-01, IDENT-02, IDENT-03, IDENT-04, IDENT-05, IDENT-06, IDENT-07, IDENT-08, IDENT-09, IDENT-10
**Success Criteria** (what must be TRUE):
  1. Navegar a cualquier URL sin sesión redirige a `/login`; `curl /api/health` sin cookie responde 401 (excepto `/api/health`, decisión a tomar en Phase 2)
  2. Un usuario con rol "propietario" ve y gestiona `/ajustes/usuarios` y `/ajustes/auditoria`
  3. Un usuario con rol "usuario" sólo ve operaciones básicas dentro de sus departamentos; sin acceso a ajustes
  4. 5 intentos de login fallidos lockean la cuenta + IP durante 15 minutos
  5. Cada request autenticada genera una fila en `nexo.audit_log`; `DELETE` desde el rol app falla con error de permiso
  6. Panel `/ajustes/auditoria` filtra por user/fecha/path/status y exporta CSV
**Plans**: TBD (se definen en `/gsd-plan-phase 2`)

### Phase 3: Capa de datos
**Goal**: Eliminar queries SQL embebidas en routers; aislar `engine_mes` (dbizaro read-only), `engine_app` (ecs_mobility), `engine_nexo` (Postgres); `schema_guard` valida el esquema al arrancar.
**Depends on**: Phase 2 (`engine_nexo` ya existe; users/audit dependen de Postgres)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, DATA-07, DATA-08, DATA-09, DATA-10, DATA-11
**Success Criteria** (what must be TRUE):
  1. Routers no importan `pyodbc` directamente; todas las consultas pasan por repositorios
  2. Queries residen en `.sql` versionados bajo `nexo/data/sql/`; no hay SQL hardcoded en Python
  3. Cross-database references `dbizaro.admuser.*` eliminadas
  4. `schema_guard` en lifespan: si falta `nexo.users` (o cualquier tabla crítica), el arranque falla con mensaje claro
  5. Pipeline OEE sigue generando PDFs idénticos a Mark-II tras el refactor (sin regresión en el núcleo de cálculo)
  6. Tests de repositorios pasan en CI contra un Postgres dedicado
**Plans**: TBD

### Phase 4: Consultas pesadas
**Goal**: Preflight estima coste antes de ejecutar pipeline/queries caras; postflight mide y alerta; aprobación asíncrona para rojos.
**Depends on**: Phase 2 (auth para user_id), Phase 3 (repositorios exponen timing hooks)
**Requirements**: QUERY-01, QUERY-02, QUERY-03, QUERY-04, QUERY-05, QUERY-06, QUERY-07, QUERY-08
**Success Criteria** (what must be TRUE):
  1. Disparar un pipeline con muchos recursos/días muestra toast amber "esto tardará ~X min, ¿continuar?"
  2. Una query en `/bbdd` con coste estimado > `block_ms` abre flujo de aprobación y no ejecuta hasta que propietario aprueba
  3. `nexo.query_log` contiene estimated_ms y actual_ms para cada ejecución; alertas WARNING si divergen > 50%
  4. `/ajustes/limites` permite editar umbrales por endpoint sin tocar código
  5. Pipeline no congela la UI: otras páginas responden mientras matplotlib genera PDFs (vía `asyncio.to_thread`)
**Plans**: TBD

### Phase 5: UI por roles
**Goal**: Sidebar y páginas muestran sólo lo que el rol del usuario puede ver; split de `ajustes.html`.
**Depends on**: Phase 2 (roles), Phase 4 (páginas de ajustes existen)
**Requirements**: UIROL-01, UIROL-02, UIROL-03, UIROL-04, UIROL-05
**Success Criteria** (what must be TRUE):
  1. Login como propietario muestra todas las entradas del sidebar
  2. Login como usuario de "producción" muestra pipeline/historial/capacidad/recursos/ciclos; oculta `/bbdd` y `/ajustes/*`
  3. Login como directivo de "ingeniería" muestra los módulos de su departamento
  4. `/ajustes` hub lleva a 6 sub-páginas separadas (conexión, SMTP, usuarios, auditoría, límites, solicitudes)
  5. Botones "Ejecutar pipeline", "Borrar ejecución", "Sincronizar recursos" ocultos si el user no tiene permiso
**Plans**: TBD

### Phase 6: Despliegue LAN HTTPS
**Goal**: Nexo corriendo en servidor Ubuntu Server 24.04 (i5 7ª gen, 16 GB, SSD 1 TB), `nexo.ecsmobility.com` resuelto por DNS interno, HTTPS (LE DNS-01 o cert interno documentado), sin exposición internet.
**Depends on**: Phase 1 (nombres estables), Phase 2 (auth en producción), Phase 4 (preflight en producción)
**Requirements**: DEPLOY-01, DEPLOY-02, DEPLOY-03, DEPLOY-04, DEPLOY-05, DEPLOY-06, DEPLOY-07, DEPLOY-08
**Success Criteria** (what must be TRUE):
  1. Desde otro equipo LAN: `https://nexo.ecsmobility.com` carga con cert reconocido (LE válido o interno firmado aceptado por la política IT)
  2. Puerto 5432 no escuchable desde el host Ubuntu (`ss -tlnp | grep 5432` vacío); sí desde `docker compose exec db psql`
  3. `scripts/deploy.sh` en el servidor ejecuta `git pull + build + up -d` sin intervención
  4. `docs/DEPLOY_LAN.md` permite reinstalar desde cero a otro admin sin contexto
  5. `ufw status` muestra sólo 22, 80, 443 permitidos
**Plans**: TBD

### Phase 7: DevEx hardening
**Goal**: Cualquier dev futuro puede arrancar, cambiar y desplegar Nexo sin fricción; CI con cobertura; runbook para incidencias comunes.
**Depends on**: Todos los anteriores estables (no introduce código nuevo de app, sólo herramientas)
**Requirements**: DEVEX-01, DEVEX-02, DEVEX-03, DEVEX-04, DEVEX-05, DEVEX-06, DEVEX-07
**Success Criteria** (what must be TRUE):
  1. `git commit` dispara pre-commit (ruff/black/mypy) y rechaza código con issues críticos
  2. CI pasa en Python 3.11 y 3.12; cobertura ≥ 60% en `api/` y `nexo/`
  3. `make test && make lint && make format` funcionan sin configuración adicional
  4. `docs/ARCHITECTURE.md` tiene diagrama actualizado de los 3 engines y los servicios compose
  5. `docs/RUNBOOK.md` cubre los 5 escenarios acordados
  6. `docs/RELEASE.md` define checklist reproducible para un release versionado
**Plans**: TBD

## Progress

**Execution Order:**
Phases ejecutan en orden estricto: 1 → 2 → 3 → 4 → 5 → 6 → 7

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Naming + Higiene + CI | 0/1 | Not started | - |
| 2. Identidad | 0/? | Not started | - |
| 3. Capa de datos | 0/? | Not started | - |
| 4. Consultas pesadas | 0/? | Not started | - |
| 5. UI por roles | 0/? | Not started | - |
| 6. Despliegue LAN HTTPS | 0/? | Not started | - |
| 7. DevEx hardening | 0/? | Not started | - |
