# Nexo

## What This Is

Plataforma interna de ECS Mobility que centraliza OEE (Overall Equipment Effectiveness),
calidad, trazabilidad y futuros módulos de planta. Sucesora de la app actual
"OEE Planta / analisis_datos": mismo repo, mismo stack, mismo equipo. El cambio
a "Nexo" es rebrand + refactor estructural (Mark-III), no reescritura.

Actualmente en operación: FastAPI + Jinja2 + Alpine + Tailwind (runtime CDN) +
Postgres 16 + SQL Server via pyodbc + Docker Compose + Caddy reverse proxy.
Despliegue en LAN interna de ECS, sin exposición a internet.

## Core Value

Consultar datos reales de producción de la planta (MES IZARO / `dbizaro`) y
generar informes OEE fiables por máquina, turno y sección **sin bloquear al
operario y sin filtrar información entre departamentos**. Si todo lo demás
falla, la pipeline OEE tiene que seguir generando PDFs contra la BD MES.

## Requirements

### Validated

Inferidas del codebase existente (rama `feature/Mark-II`, commit `f07e80e`,
ver `.planning/codebase/` y `docs/CURRENT_STATE_AUDIT.md`).

- ✓ Pipeline OEE: extracción de `fmesdtc+fmesddf+fmesinc+fprolof` de `dbizaro`, cálculo de disponibilidad/rendimiento/calidad/oee_secciones, generación PDF con matplotlib — existing
- ✓ Centro de mando / panel LUK4 con estado en tiempo real de pabellón 5 (tablas `luk4.estado`, `luk4.tiempos_ciclo`, `luk4.alarmas`, `luk4.plano_zonas`) — existing
- ✓ Historial de ejecuciones con regeneración de informes desde `oee.datos` persistidos — existing
- ✓ Cálculo de capacidad teórica vs real (P10 de ciclos a 180 días) — existing
- ✓ CRUD de recursos, ciclos y contactos en `ecs_mobility.cfg.*` — existing
- ✓ Consulta de fichajes y operarios (`admuser.fmesope`) — existing
- ✓ Explorador BBDD con endpoint `query` SELECT-only (whitelist anti-DDL/DML) — existing
- ✓ Envío de informes por email a contactos registrados (contactos OK; SMTP **roto** — sección `smtp` ausente de `data/db_config.json`) — existing
- ✓ Servidor MCP read-only con 15 tools wrappers sobre la API, para uso de Claude Code en desarrollo — existing
- ✓ Despliegue Docker Compose (web + db + caddy + mcp) con Caddy HTTPS autofirmado en LAN — existing

### Active

Alcance Mark-III (milestone actual). Cada ítem mapea a una phase del ROADMAP.

- [ ] Rebrand + higiene + CI mínimo + exception handler sin fuga de traceback (Phase 1 / Sprint 0)
- [ ] Identidad: auth + RBAC con roles propietario/directivo/usuario + departamentos + audit append-only (Phase 2 / Sprint 1)
- [ ] Capa de datos: repositorios + `.sql` versionados + schema_guard + separación `engine_mes` / `engine_app` / `engine_nexo` (Phase 3 / Sprint 2)
- [ ] Consultas pesadas: preflight + postflight + umbrales editables + aprobación asíncrona (Phase 4 / Sprint 3)
- [ ] UI por roles: sidebar + páginas condicionadas al rol, split de `ajustes.html` (Phase 5 / Sprint 4)
- [ ] Despliegue LAN con HTTPS válido (Let's Encrypt DNS-01 o cert autofirmado) en servidor Ubuntu Server 24.04 (Phase 6 / Sprint 5)
- [ ] DevEx hardening: pre-commit, CI ampliado con cobertura, runbook, release checklist (Phase 7 / Sprint 6)

### Out of Scope

Explícitamente excluido de Mark-III (ver `docs/MARK_III_PLAN.md` §"Qué queda EXPLÍCITAMENTE fuera").

- Ingesta realtime vía OPC-UA/MQTT — Mark-IV+
- Módulos nuevos de calidad y trazabilidad — Mark-IV+
- Dashboards en streaming — Mark-IV+
- Exposición a internet (dominio público) — decisión de producto: Nexo es LAN-only
- Refactor interno de los 4 módulos OEE (`disponibilidad`, `rendimiento`, `calidad`, `oee_secciones`) — funciona hoy, refactor no prioritario
- Sustituir matplotlib por WeasyPrint / ReportLab / Playwright — sólo si preflight demuestra que matplotlib es inviable
- CRUD de `plantillas.html` enganchado al pipeline — experimental, no requerido
- Microservicios — scope contradice strangler-fig
- Rotación de credenciales SQL Server en Sprint 0 — diferida (decisión explícita del operador)
- Configuración SMTP — pendiente de decisión de infraestructura
- Compra de dominio público — LAN-only cierra esta puerta
- 2FA — diferido a Mark-IV
- LDAP / Active Directory — diferido (usuarios locales en Postgres para Mark-III)
- Reescritura de historial git (filter-repo) — sólo tras revisar `docs/SECURITY_AUDIT.md` con el operador
- Rename de la carpeta `OEE/` a `modules/oee/` — diferido a Phase 3 / Sprint 2
- Rename del repo en GitHub (`analisis_datos`) — diferido
- Mover `cfg.recursos` / `cfg.ciclos` / `cfg.contactos` a Postgres — datos de negocio compartidos; Power BI y túnel IoT ya los leen; migración rompería integraciones externas
- MCP en stack de producción por defecto — aparcado en `profiles: ["mcp"]`; sólo arranca con `docker compose --profile mcp up`

## Context

**Estado del codebase al arrancar Mark-III** (ver `docs/CURRENT_STATE_AUDIT.md` para detalle):

- ~10 361 LOC Python + ~4 957 LOC Jinja2 + 421 LOC JS + 103 LOC CSS
- 14 routers FastAPI, todos públicos sin auth
- SQL hardcoded en Python (no hay ficheros `.sql` versionados)
- `SECTION_MAP` duplicado en 4 sitios, `determinar_turno` duplicado en 3 módulos OEE
- `.env` con credenciales SQL Server en claro (en `.gitignore`), `.env:Zone.Identifier` trackeado (residuo WSL), `test_email.py` y `server.py` redundantes en raíz
- `global_exception_handler` devuelve traceback completo al navegador (fuga de información)
- Postgres 16 en compose pero **sin usar** (camino muerto en `effective_database_url`)
- 30 tests unitarios puros sobre `OEE/oee_secciones/main.py`; cobertura real estimada <15%; sin GitHub Actions
- 3 ficheros >800 LOC: `OEE/db/connector.py` (847), `OEE/disponibilidad/main.py` (911), `OEE/oee_secciones/main.py` (1 624)
- Servicios compose: `web`, `db` (PG sin consumidores), `caddy` (autofirmado), `mcp` (stdio)

**Reconocimiento previo:**
- `docs/CURRENT_STATE_AUDIT.md` — foto fija autoritativa del repo en `f07e80e`
- `docs/OPEN_QUESTIONS.md` — 12 preguntas + 5 decisiones bloqueantes (todas resueltas en esta sesión; ver Key Decisions)
- `docs/MARK_III_PLAN.md` — plan detallado de los 7 sprints con entregables, riesgos y estimaciones
- `.planning/codebase/` — mapa estructurado generado por `/gsd-map-codebase` el 2026-04-18

**Equipo:** Erik (IT + dev) + operador de planta (feedback cualitativo). Sin otros devs activos hoy.

## Constraints

- **Tech stack**: FastAPI + Jinja2 + Alpine + Tailwind + Postgres 16 + SQL Server via pyodbc + Docker Compose + Caddy — sin cambios en Mark-III. Strangler-fig en el mismo repo; historial conservado.
- **Despliegue**: LAN interna ECS (Ubuntu Server 24.04, i5 7ª gen, 16 GB, SSD 1 TB). Sin exposición internet. DNS interno resuelve `nexo.ecsmobility.com`.
- **BBDD SQL Server**: dos BDs (`dbizaro` MES read-only, `ecs_mobility` propia) **en la misma instancia** (`192.168.0.4:1433`). Confirmado por cross-database references de 3-part names en código. Nexo prepara env vars separadas (`NEXO_MES_*` / `NEXO_APP_*`) para futuro split, pero apuntan al mismo host y credenciales hoy.
- **Postgres 16**: casa nueva para `nexo.*` (users, roles, audit_log, query_log). `cfg.*` y `oee.*` se quedan en `ecs_mobility` (son datos de negocio ya consumidos por Power BI y el túnel IoT; moverlos es riesgo fuera de scope).
- **Timeline**: estimación orientativa **10-12 semanas de trabajo focalizado** (29-30 días focalizados, ~3 por semana). **Sin fecha objetivo ni hito externo.** El operador marca el ritmo.
- **Seguridad**: `.env` con permisos 600 en producción; credenciales NO rotadas en Sprint 0 (decisión explícita, diferida). Cualquier credencial encontrada en historial git se documenta en `docs/SECURITY_AUDIT.md` pero no se reescribe historial sin autorización.
- **Backward compatibility**: `api/config.py` acepta ambos prefijos `NEXO_*` y `OEE_*` durante Mark-III (compat removida en Mark-IV).

## Key Decisions

Las 5 decisiones bloqueantes de `docs/OPEN_QUESTIONS.md` se cerraron al arrancar
Sprint 0. El modelo de roles se definió en la misma sesión.

| Decisión | Rationale | Outcome |
|----------|-----------|---------|
| Instancias SQL Server: misma instancia, env vars separadas `NEXO_MES_*` / `NEXO_APP_*` apuntando al mismo host/credenciales | Preparar el código para un eventual split sin duplicar esfuerzo en Mark-IV; hoy conviven en la misma instancia confirmado por 3-part names en `centro_mando.py` | — Pending |
| Credenciales SQL Server: no rotar en Sprint 0 | El operador decide rotar tras revisar `docs/SECURITY_AUDIT.md`; Sprint 0 sólo reestructura env vars | — Pending |
| CI mínimo en Sprint 0 (no en Sprint 7) | Una hora de coste, retorno inmediato: cada PR a Mark-III se valida automáticamente | — Pending |
| `global_exception_handler` sin fuga de traceback en Sprint 0 (no en Sprint 1) | Fuga de información independiente de auth; no tiene sentido arrastrarla 6 días más | — Pending |
| Audit de historial git para buscar credenciales commiteadas (`docs/SECURITY_AUDIT.md`) | Pre-requisito para decidir sobre filter-repo; usuario decide después de ver el informe | — Pending |
| `filter-repo` (reescritura de historial): NO en Sprint 0; decisión manual post-audit | Reescribir historial sin revisar hallazgos rompe commits y referencias externas | — Pending |
| MCP server aparcado en `docker-compose.yml` bajo `profiles: ["mcp"]`; `make up` y `make dev` **no** lo arrancan; sólo `docker compose --profile mcp up` | En LAN con Nexo en producción el MCP no aporta; se conserva el código para uso local de desarrollo. Cambio efectivo en `docker-compose.yml` ejecutado dentro de Sprint 0 (commit 9). | ✓ Closed |
| Postgres = casa de `nexo.users`, `nexo.roles`, `nexo.permissions`, `nexo.audit_log`, `nexo.query_log`, `nexo.login_attempts` | Aislar identidad y observabilidad de la BD de negocio (ecs_mobility) y de la BD read-only (dbizaro) | — Pending |
| `cfg.recursos`, `cfg.ciclos`, `cfg.contactos` permanecen en `ecs_mobility.cfg.*` | Datos de negocio compartidos: Power BI y túnel IoT ya los leen; migrarlos rompería integraciones externas | ✓ Closed |
| Modelo de auth: 1 rol de nivel (propietario/directivo/usuario) + N departamentos (rrhh/comercial/ingenieria/producción/gerencia) | Granularidad suficiente para ECS sin complejidad de RBAC full-matrix; simplificable si no escala | — Pending |
| Passwords: argon2id, min 12 chars, cambio obligatorio primer login, sin expiración forzada | Compromiso entre seguridad y fricción para uso interno; sin expiración evita tickets de soporte por passwords caducadas | — Pending |
| Bloqueo progresivo: 5 intentos fallidos → 15 min lock de `(user, IP)` | Más simple que el escalado 3→30s / 5→5min / 10→manual de MARK_III_PLAN.md; mismo efecto defensivo | — Pending |
| Sin 2FA, sin LDAP en Mark-III | LAN-only + equipo pequeño; añadir complejidad sin valor inmediato | — Pending |
| Reorden de sprints: `0 → 1 (identidad antes de datos) → 2 (datos) → ...` respecto del plan original `0 → 1 (datos) → 2 (auth) → 3 (audit) → ...` | Evita 3-5 días operando app pública en LAN durante el refactor de datos; auth + audit se funden por dependencia de sesión | — Pending |
| Carpeta `OEE/` no se renombra a `modules/oee/` en Mark-III (diferido a Sprint 2) | Evita churn masivo; rename coordinado con movimiento de SQL a `.sql` cuando ya exista capa de repositorios | — Pending |
| Nombre del repo GitHub `analisis_datos` no se cambia en Mark-III | Costoso (hooks, CI, clones de todos), beneficio puramente cosmético, diferido | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

**Sync rule (Nexo-specific, ver `CLAUDE.md`):**
Este archivo es **proyección derivada** de `docs/MARK_III_PLAN.md`,
`docs/OPEN_QUESTIONS.md` y `docs/CURRENT_STATE_AUDIT.md`. No editar a mano.
Si el plan cambia, se edita el doc fuente y se regenera `.planning/`.
`STATE.md` y artefactos de fase (`PLAN.md`, `SUMMARY.md`, `VERIFICATION.md`)
son runtime de GSD — éstos sí los mutan los comandos GSD durante la ejecución.

---
*Last updated: 2026-04-18 after closing Mark-III open questions (MCP, cfg tables, timeline, regen policy)*
