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

- [x] **Phase 1: Naming + Higiene + CI** — Sprint 0: rebrand Nexo, limpieza, CI mínimo, exception handler sin traceback, audit de historial ✓ 2026-04-18
- [ ] **Phase 2: Identidad (auth + RBAC + audit)** — Sprint 1: login, roles propietario/directivo/usuario + departamentos, middleware de auth y audit append-only
- [x] **Phase 3: Capa de datos** — Sprint 2: repositorios, `.sql` versionados, schema_guard, separación `engine_mes` / `engine_app` / `engine_nexo` ✓ 2026-04-19
- [x] **Phase 4: Consultas pesadas** — Sprint 3: preflight + postflight + aprobación asíncrona + umbrales editables ✓ 2026-04-20
- [x] **Phase 5: UI por roles** — Sprint 4: sidebar y páginas condicionadas al rol, split de `ajustes.html` ✓ 2026-04-20
- [ ] **Phase 6: Despliegue LAN HTTPS** — Sprint 5: `docker-compose.prod.yml`, Caddy con LE DNS-01 o cert interno, firewall, runbook de deploy
- [ ] **Phase 7: DevEx hardening** — Sprint 6: pre-commit, CI ampliado con cobertura, `docs/ARCHITECTURE.md`, `docs/RUNBOOK.md`, `docs/RELEASE.md`
- [ ] **Phase 8: Rediseño UI (modo claro moderno)** — Sprint 7: rediseño visual completo manteniendo Centro de Mando; sidebar collapsible/drawer; animaciones; tema claro; ventana a ventana con propuestas; secciones nuevas si aplican

## Phase Details

### Phase 1: Naming + Higiene + CI
**Goal**: Repo arrancable con marca "Nexo", higiene mínima (Zone.Identifier, archivos residuales, requirements pineados), CI en GitHub Actions, handler de excepciones que no filtra traceback, audit de historial git documentado.
**Depends on**: Nothing (first phase)
**Requirements**: NAMING-01, NAMING-02, NAMING-03, NAMING-04, NAMING-05, NAMING-06, NAMING-07, NAMING-08, NAMING-09, NAMING-10, NAMING-11, NAMING-12, NAMING-13, NAMING-14, NAMING-15, NAMING-16, NAMING-17
**Success Criteria** (what must be TRUE):
  1. `make build && make up && make health` devuelven OK con contenedores llamados `nexo-*`; `make up` y `make dev` **no** arrancan el servicio `mcp`
  2. Título FastAPI, sidebar, README dicen "Nexo"; metadata OpenAPI consistente
  3. `.env.example` cubre `NEXO_*` (MES + APP + web + PG + branding) y mantiene compat `OEE_*` durante Mark-III
  4. `.github/workflows/ci.yml` ejecuta lint/test/build/secrets en push a `feature/Mark-III` y PRs
  5. Forzar un 500 en desarrollo devuelve `{"error_id": "<uuid>", "message": "Internal error"}` — sin traceback en el body
  6. `docs/SECURITY_AUDIT.md` lista credenciales expuestas del historial (sin valores literales)
  7. `CLAUDE.md`, `docs/AUTH_MODEL.md`, `docs/GLOSSARY.md`, `docs/BRANDING.md`, `docs/DATA_MIGRATION_NOTES.md` existen con el contenido acordado; `docs/MARK_III_PLAN.md` y `docs/OPEN_QUESTIONS.md` actualizados
  8. `docker compose --profile mcp up` arranca el contenedor `mcp`; sin el flag, no arranca
  9. Templates (`base.html`, favicon, Notification icon de `app.js`) leen el logo desde las variables `NEXO_LOGO_PATH` / `NEXO_ECS_LOGO_PATH` en vez de rutas hardcoded; logos físicos ubicados bajo `static/img/brand/nexo/` y `static/img/brand/ecs/`
**Plans**: 1 plan (Sprint 0 se ejecuta como una secuencia de 13 commits atómicos especificados en `.planning/1-sprint-0-naming-and-hygiene/PLAN.md`)

Plans:
- [x] 01-01: Sprint 0 — 13 commits atómicos + 1 feedback operador (audit historial GATE, junk, gitignore, odbc move, oee.db decision, env rename + split, UI rebrand + logos vars, exception handler fix, mcp profile, requirements pin, CI workflow, docs core, MARK_III_PLAN/OPEN_QUESTIONS sync, sidebar tweak) ✓ 2026-04-18

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
**Plans**: 3 plans

Plans:
- [x] 03-01-PLAN.md — Foundation: engines + loader + schema_guard + DTO skeleton + fixtures (DATA-01, DATA-05, DATA-06, DATA-08, DATA-10, DATA-11) ✓ 2026-04-18
- [x] 03-02-PLAN.md — Capa MES: MesRepository + 5 routers refactor + PDF regression gate + kill 3-part names (DATA-02, DATA-05, DATA-07, DATA-09) ✓ 2026-04-19 (PDF regression diferido a preprod, deadline 2026-04-26)
- [x] 03-03-PLAN.md — Capa APP + NEXO: repos + routers APP + auth.py/auditoria.py/usuarios.py refactor (DATA-03, DATA-04, DATA-07, DATA-08, DATA-10) ✓ 2026-04-19

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
**Plans**: 4 plans

Plans:
- [x] 04-01-PLAN.md — Foundation: 3 tablas (query_log, query_thresholds, query_approvals) + ORM + DTOs + repos skeleton + schema_guard extendido + seeds + thresholds_cache skeleton + Wave 0 tests (QUERY-01, QUERY-02) ✓ 2026-04-20
- [x] 04-02-PLAN.md — Preflight + Middleware + asyncio.to_thread: preflight.py + pipeline_lock.py + query_timing.py + refactor 4 routers (pipeline/bbdd/capacidad/operarios) + modales amber/red Alpine (QUERY-03, QUERY-04, QUERY-05, QUERY-07, QUERY-08) ✓ 2026-04-20
- [x] 04-03-PLAN.md — Approval flow: approvals.py service + /api/approvals/* router + páginas /mis-solicitudes + /ajustes/solicitudes + badge sidebar HTMX + cleanup_scheduler + job TTL 7d (QUERY-06) ✓ 2026-04-20
- [x] 04-04-PLAN.md — Observability UI + LISTEN/NOTIFY + learning: listen_loop completo + /ajustes/limites CRUD + /ajustes/rendimiento Chart.js + factor_learning helper + query_log retention job + factor_auto_refresh mensual (QUERY-02 complete, QUERY-07 complete, D-19, D-20) ✓ 2026-04-20

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
**Plans**: 5 plans (Wave 1 fundamentos → Wave 2 sidebar/menus → Wave 3 botones sensibles → Wave 4 ajustes split → Wave 5 verificación manual E2E).

Plans:
- [x] 05-01: Wave 1 — extract `can()` helper + trampoline `require_permission` + register `can` as Jinja global ✓ 2026-04-20
- [x] 05-02: Wave 2 — sidebar/menu filtering por permiso (UIROL-02) ✓ 2026-04-20
- [x] 05-03: Wave 3 — forbidden UX + flash pipeline (FlashMiddleware + 403 redirect+toast, JSON contract estable) (UIROL-02) ✓ 2026-04-20
- [x] 05-04: Wave 4 — split de `ajustes.html` en 6 sub-páginas + hub (UIROL-03) ✓ 2026-04-20
- [x] 05-05: Wave 5 — button gating UIROL-04 + HTML GET hardening Pitfall 4 + smoke manual UIROL-05 ✓ 2026-04-20

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
**Plans**: 3 plans

Plans:
- [x] 06-01-PLAN.md — Foundation: caddy/Caddyfile.prod + docker-compose.prod.yml + .env.prod.example + Wave 0 tests infra (DEPLOY-01, DEPLOY-02, DEPLOY-03, DEPLOY-08)
- [x] 06-02-PLAN.md — Automation: scripts/deploy.sh + scripts/backup_nightly.sh + Makefile prod-* targets + script tests (DEPLOY-04)
- [ ] 06-03-PLAN.md — Runbook + smoke: docs/DEPLOY_LAN.md + tests/infra/deploy_smoke.sh + test_deploy_lan_doc.py (DEPLOY-05, DEPLOY-06, DEPLOY-07)

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

### Phase 8: Rediseño UI (modo claro moderno)
**Goal**: Rediseño visual completo de la aplicación manteniendo el Centro de Mando como ancla. Look&feel moderno en tema claro, fácil de leer, no sobrecargado. Sidebar colapsable tipo drawer/popup en lugar de barra lateral siempre visible. Animaciones sutiles donde aporten. Discussion window-by-window con propuestas visuales antes de tocar código; propuestas de secciones nuevas que mejoren el flujo si se detectan gaps.
**Depends on**: Phase 5 (permisos y filtrado estables — el rediseño NO puede regresionar la capa de autorización), Phase 6 (despliegue existente para validar en LAN antes/después), Phase 7 (DevEx para iterar rápido durante el rediseño)
**Requirements**: UIREDO-01, UIREDO-02, UIREDO-03, UIREDO-04, UIREDO-05, UIREDO-06, UIREDO-07, UIREDO-08
**Success Criteria** (what must be TRUE):
  1. Tema claro aplicado consistentemente en todas las páginas rediseñadas (paleta definida en `docs/BRANDING.md` actualizado; variables CSS tokenizadas)
  2. Sidebar convertida en drawer/popup colapsable: por defecto colapsado en desktop, siempre drawer en móvil; animación de apertura/cierre fluida (≤200ms, respetando `prefers-reduced-motion`)
  3. Centro de Mando (`/`) mantiene su estructura actual (datos, layout de tarjetas) pero hereda el nuevo sistema de tokens/tema claro
  4. Cada pantalla rediseñada (pipeline, bbdd, capacidad, operarios, recursos, ciclos, historial, datos, ajustes + sub-páginas) se discute individualmente con al menos 2 propuestas visuales antes de implementar
  5. Permisos/RBAC de Phase 5 intactos: `can()` + `require_permission` + 11 botones gated + 8 HTML GETs guarded siguen funcionando en el UI rediseñado
  6. Secciones nuevas propuestas (si las hay) validadas contra scope Mark-III antes de añadir al roadmap — nada se añade silenciosamente
  7. Transición sin regresión funcional: cada página rediseñada pasa su smoke test de Phase anterior antes de mergear
  8. Accesibilidad mínima: contraste AA, focus visible, navegación por teclado funcional en drawer + modales
**Plans**: TBD (se discute ventana a ventana; el plan-phase emite 1 plan atómico por pantalla o grupo de pantallas)

## Progress

**Execution Order:**
Phases ejecutan en orden estricto: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Naming + Higiene + CI | 1/1 | Complete | 2026-04-18 |
| 2. Identidad | 4/4 | Complete | 2026-04-19 |
| 3. Capa de datos | 3/3 | Complete | 2026-04-19 |
| 4. Consultas pesadas | 4/4 | Complete | 2026-04-20 |
| 5. UI por roles | 5/5 | Complete | 2026-04-20 |
| 6. Despliegue LAN HTTPS | 0/? | Not started | - |
| 7. DevEx hardening | 0/? | Not started | - |
| 8. Rediseño UI (modo claro moderno) | 0/? | Not started | - |
