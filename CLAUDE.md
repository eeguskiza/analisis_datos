# CLAUDE.md — Guía para IAs y devs nuevos en Nexo

Este archivo es el mapa que le damos a cualquier IA (Claude Code, Copilot,
Gemini, etc.) o dev humano que entre a tocar este repo. Lo actualizamos con
cada sprint de Mark-III.

Última revisión: 2026-04-22 (cierre Mark-III / Sprint 6 / Phase 7 —
DevEx hardening + quartet docs + CI coverage gate + pre-commit activo).

---

## Qué es Nexo

Plataforma interna de ECS Mobility. Sucesora del monolito "OEE Planta /
analisis_datos". Mismo repo, mismo stack, mismo equipo. Cambio de nombre +
refactor estructural en Mark-III, no reescritura.

Stack: FastAPI + Jinja2 + Alpine + Tailwind (CDN) + Postgres 16 +
SQL Server via pyodbc + Docker Compose + Caddy.

Despliegue: LAN interna ECS (Ubuntu Server 24.04). Sin exposición a internet.

---

## Fuente de verdad del plan

**Documentos autoritativos (editables a mano):**

- `docs/MARK_III_PLAN.md` — plan de los 7 sprints con entregables, riesgos y estimaciones.
- `docs/OPEN_QUESTIONS.md` — preguntas del reconocimiento + decisiones cerradas.
- `docs/CURRENT_STATE_AUDIT.md` — foto fija del repo al arrancar Mark-III (rev. `f07e80e`).
- `docs/AUTH_MODEL.md` — modelo de roles, departamentos, política de passwords, bloqueo progresivo.
- `docs/BRANDING.md` — assets de marca (Nexo + ECS Mobility), variables de entorno, convenciones de uso.
- `docs/GLOSSARY.md` — términos de dominio (Nexo, MES, APP, rol, departamento, audit_log, preflight, postflight).
- `docs/SECURITY_AUDIT.md` — hallazgos del audit de historial git (generado en Sprint 0).
- `docs/DATA_MIGRATION_NOTES.md` — decisiones de migración (p. ej. `data/oee.db` en Sprint 0).
- `docs/ARCHITECTURE.md` — mapa técnico del repo (3 engines `engine_mes` / `engine_app` / `engine_nexo`, middleware stack, schedulers, layout del paquete). Entregable Phase 7.
- `docs/RUNBOOK.md` — 5 escenarios de incidencia runtime (MES caído, Postgres no arranca, certificado Caddy expira, pipeline atascado, lockout del único propietario) con hallazgos críticos sobre helpers que NO existen. Entregable Phase 7.
- `docs/RELEASE.md` — checklist reproducible de release versionado (semver `vMAJOR.MINOR.PATCH` + deploy + smoke externo + rollback). Entregable Phase 7.
- `docs/DEPLOY_LAN.md` — runbook de instalación y recuperación del servidor LAN (Ubuntu + Docker + root CA + hosts-file + cron backup). Entregable Phase 6.
- `CHANGELOG.md` — Keep a Changelog 1.1.0 con historial Mark-III completo (Phases 1-7).

**Artefactos de ejecución (runtime de GSD):**

- `.planning/PROJECT.md`, `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md` — proyección GSD de los `docs/`.
- `.planning/STATE.md` — estado de la sesión (fase actual, progreso).
- `.planning/<phase>/PLAN.md`, `.planning/<phase>/SUMMARY.md`, `.planning/<phase>/VERIFICATION.md` — generados durante `/gsd-execute-phase`.
- `.planning/codebase/` — mapa del repo (generado con `/gsd-map-codebase`).

---

## Regeneración de .planning/

La fuente de verdad humana es `docs/MARK_III_PLAN.md`,
`docs/OPEN_QUESTIONS.md` y `docs/CURRENT_STATE_AUDIT.md`.

Los archivos `.planning/PROJECT.md`, `REQUIREMENTS.md` y `ROADMAP.md`
son proyección derivada. No se editan a mano. Si cambian los docs
fuente, se regenera la proyección solicitándolo: **"Claude, regenera
`.planning/` desde `docs/`"**.

Los archivos `.planning/STATE.md`, `.planning/<phase>/PLAN.md` y
`.planning/<phase>/VERIFICATION.md` son estado runtime de GSD.
Los gestiona el skill durante ejecución. Tampoco se editan a mano.

**Condición de guarda:** si antes de abrir una fase nueva los `docs/` y
los `.planning/` de proyección han divergido, regenerar antes de
avanzar.

---

## Decisiones cerradas de Mark-III

Ver tabla completa en `.planning/PROJECT.md` sección "Key Decisions".
Resumen de las más estructurales:

- **Naming**: plataforma = "Nexo". Env vars `OEE_*` → `NEXO_*` con capa de compat durante Mark-III. SQL Server separado en `NEXO_MES_*` (dbizaro read-only) y `NEXO_APP_*` (ecs_mobility), mismo host y credenciales hoy.
- **Carpeta `OEE/`**: no se renombra en Mark-III (diferido a Sprint 2 coordinado con capa de repositorios).
- **Repo GitHub** `analisis_datos`: no se renombra (coste alto, beneficio cosmético).
- **Credenciales SQL Server**: no se rotan en Sprint 0. Decisión se toma tras revisar `docs/SECURITY_AUDIT.md`.
- **Historial git**: `filter-repo` **no** se ejecuta sin autorización explícita del operador.
- **Postgres**: casa de `nexo.users`, `nexo.roles`, `nexo.permissions`, `nexo.audit_log`, `nexo.query_log`, `nexo.login_attempts`. `cfg.recursos`/`cfg.ciclos`/`cfg.contactos` se quedan en `ecs_mobility.cfg.*`.
- **MCP** (`mcp/` service): aparcado en `profiles: ["mcp"]` de compose. `make up` y `make dev` no lo arrancan. Sólo con `docker compose --profile mcp up`.
- **Auth** (Sprint 1):
  - Roles: `propietario` / `directivo` / `usuario`.
  - Departamentos: `rrhh` / `comercial` / `ingenieria` / `produccion` / `gerencia`.
  - Usuario tiene 1 rol + N departamentos.
  - Propietario = acceso global, ignora departamento. Único que gestiona usuarios y ve audit completo.
  - Passwords: argon2id, min 12 chars, cambio obligatorio primer login, **sin expiración forzada**.
  - Bloqueo progresivo: **5 intentos fallidos → 15 min lock de `(user, IP)`**.
  - Sin 2FA, sin LDAP en Mark-III.
- **Timeline**: estimación orientativa 10-12 semanas de trabajo focalizado. Sin fecha objetivo ni hito externo.
- **Exposición a internet**: descartada. LAN-only.

---

## Convenciones de naming

- **Producto**: "Nexo" (sin acento, capitalizado).
- **Empresa**: "ECS Mobility" (en textos largos), "ECS" (en abreviaciones).
- **Env vars**: prefijo `NEXO_*` para todas las nuevas. `OEE_*` se acepta como fallback durante Mark-III.
  - SQL Server (MES, read-only): `NEXO_MES_SERVER`, `NEXO_MES_PORT`, `NEXO_MES_USER`, `NEXO_MES_PASSWORD`, `NEXO_MES_DB`.
  - SQL Server (app, ecs_mobility): `NEXO_APP_SERVER`, `NEXO_APP_PORT`, `NEXO_APP_USER`, `NEXO_APP_PASSWORD`, `NEXO_APP_DB`.
  - Postgres (casa nueva): `NEXO_PG_USER`, `NEXO_PG_PASSWORD`, `NEXO_PG_DB`.
  - Web: `NEXO_HOST`, `NEXO_PORT`, `NEXO_DEBUG`.
  - Branding: `NEXO_APP_NAME`, `NEXO_COMPANY_NAME`, `NEXO_LOGO_PATH`, `NEXO_ECS_LOGO_PATH`.
- **Schemas / Tablas**: `cfg.*` y `oee.*` y `luk4.*` se mantienen en `ecs_mobility` (ya los leen Power BI y túnel IoT). Nuevas tablas Nexo en Postgres bajo schema `nexo.*`.
- **Módulos Python nuevos**: paquete raíz `nexo/` (a crear en Sprint 2). El paquete `OEE/` se mantiene con ese nombre hasta Mark-IV.
- **Servicios compose**: `web`, `db` (Postgres), `caddy`, `mcp`. Prefijos de contenedor `nexo-*`.
- **MCP**: ID interno `nexo-mcp` (renombrado desde `oee-planta`). Usuarios con `.claude.json` antiguo deben actualizar.

---

## Política de commits

- **Formato**: Conventional Commits. `<type>: <descripción corta>` + body opcional.
- **Tipos**: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`, `build`, `plan`.
- **Idioma**: inglés en el título (`docs: close open questions`); body en español si aporta contexto.
- **Atomicidad**: un commit = un cambio coherente. No mezclar rename + fix + feature en el mismo commit.
- **Branch principal de Mark-III**: `feature/Mark-III`. PRs a `main` sólo al cierre de la milestone.
- **No se aplica `--no-verify` salvo decisión explícita.** Pre-commit entra en Sprint 6 (Phase 7).
- **No se reescribe historial publicado** (`git push --force`) salvo decisión explícita con `filter-repo`.

---

## Flujo de trabajo GSD

Este proyecto usa [GSD (Get Shit Done)](https://github.com/gsd). Comandos
relevantes:

- `/gsd-progress` — ver estado actual (fase, plan, próximo paso).
- `/gsd-plan-phase N` — generar `PLAN.md` para una fase.
- `/gsd-execute-phase N` — ejecutar los plans de una fase con commits atómicos.
- `/gsd-verify-work` — verificar que la fase satisface sus requirements.
- `/gsd-map-codebase` — refrescar `.planning/codebase/`.

**Regla de oro**: antes de abrir una fase nueva, verificar que `docs/`
(fuente de verdad) y `.planning/` (proyección) no han divergido. Si han
divergido, regenerar `.planning/` conversacionalmente.

---

## Tooling DevEx (Phase 7)

Pre-commit + ruff + ruff-format + mypy + coverage gate están activos desde
Phase 7. Cada dev tras clonar debe correr una sola vez:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

Tras `pre-commit install`, cada `git commit` dispara los hooks scoped a
`api/` + `nexo/` (ruff-check, ruff-format, mypy) + hooks de higiene sobre
todo el repo. `OEE/` queda fuera del scope (decisión Mark-III).

**Targets Makefile para el dev loop**:

| Target | Propósito |
|--------|-----------|
| `make test` | `pytest -q --cov=api --cov=nexo --cov-fail-under=60` (gate bloqueante) |
| `make lint` | `ruff check api/ nexo/` + `ruff format --check api/ nexo/` + `mypy api/ nexo/` |
| `make format` | `ruff check --fix api/ nexo/` + `ruff format api/ nexo/` (auto-fix seguro) |
| `make migrate` | `docker compose exec web python scripts/init_nexo_schema.py` |
| `make backup` | `bash scripts/backup_nightly.sh` (cron-ready) |

**CI** (`.github/workflows/ci.yml`) corre 5 jobs: `lint` y `test` sobre
matriz Python 3.11 + 3.12 con coverage gate `--cov-fail-under=60`
bloqueante; `smoke` (arranca `docker compose up -d --build db web` + curl
a `:8001/api/health`); `build` (docker build nexo-web + nexo-mcp); y
`secrets` (gitleaks, único `continue-on-error: true` por hallazgos
históricos ya rotados en Sprint 0).

**Referencias de navegación**:

- `docs/ARCHITECTURE.md` — cómo está hecho Nexo por dentro (3 engines,
  middleware stack, schedulers, layout del repo).
- `docs/RUNBOOK.md` — qué hacer cuando algo se rompe (5 escenarios con
  hallazgos críticos sobre helpers que NO existen).
- `docs/RELEASE.md` — cómo cortar un release (semver + deploy + smoke +
  rollback).
- `CHANGELOG.md` — historial Mark-III completo (Keep a Changelog 1.1.0).
- `.pre-commit-config.yaml` — hooks scoped a `^(api|nexo)/`.
- `pyproject.toml` — single source of truth de config ruff + mypy +
  pytest + coverage.

---

## Despliegue productivo (Phase 6)

Deploy LAN HTTPS con Caddy `tls internal`. Hostname
`nexo.ecsmobility.local` resuelto via hosts-file en cada cliente + root
CA interna distribuida manualmente (Windows `certutil`, Linux
`update-ca-certificates`, macOS `add-trusted-cert`). Sin exposición a
internet — LAN-only.

**Targets Makefile para prod** (usan `docker-compose.yml` +
`docker-compose.prod.yml`):

| Target | Propósito |
|--------|-----------|
| `make prod-up` | Arranca stack prod (requiere `/opt/nexo/.env` con secretos reales, 600 perms) |
| `make prod-down` | Para stack prod (SIN `-v` — `pgdata` persiste; Landmine 6) |
| `make prod-logs` | Logs stack prod en tiempo real |
| `make prod-status` | Estado de containers prod |
| `make prod-health` | Health check via `/api/health` (MES caído devuelve 200 con `ok:false`) |
| `make deploy` | `scripts/deploy.sh` (git pull + build + up + smoke con pre-deploy backup atómico) |
| `make backup` | `scripts/backup_nightly.sh` (pg_dump + rotación 7d en `/var/backups/nexo/`) |

**Runbook completo de instalación y recuperación**: `docs/DEPLOY_LAN.md`
(Docker install + clone + ufw + root CA por SO + hosts-file por SO +
cron nightly + restore zcat|psql + 7 landmines numeradas; RTO 1-2h).

**Smoke externo post-deploy**: `bash tests/infra/deploy_smoke.sh` (11
checks `[DEPLOY-XX] OK|FAIL`, exit code = número de fallos, usable desde
cron o peer LAN con CA + hosts-file instalados).

---

## Qué NO hacer

Decisiones explícitas del operador que **no** se tocan sin preguntar:

- No rotar credenciales SQL Server en Sprint 0.
- No ejecutar `git filter-repo` ni `git push --force` sobre `main`/`feature/Mark-III`.
- No refactorizar los 4 módulos de `OEE/` (`disponibilidad`, `rendimiento`, `calidad`, `oee_secciones`) en Mark-III.
- No renombrar la carpeta `OEE/` hasta Sprint 2 (Phase 3).
- No renombrar el repo en GitHub.
- No configurar SMTP operativo en Mark-III (pendiente de decisión de infraestructura).
- No comprar dominio público ni exponer Nexo a internet.
- No implementar 2FA ni LDAP en Mark-III.
- No mover `cfg.recursos`/`cfg.ciclos`/`cfg.contactos` de `ecs_mobility.cfg.*` a Postgres.
- No sustituir matplotlib por otro motor de PDFs salvo que preflight demuestre que es inviable.
- No commitear con `git commit --no-verify` (pre-commit entró en Phase 7). Si un hook falla: inspeccionar con `git diff`, dejar que `make format` auto-fixee lo que pueda, `git add` y volver a commitear. Los hooks no son opcionales.
- No bajar la cobertura por debajo del gate configurado en `pyproject.toml` (`[tool.coverage.report].fail_under = 60`) ni en `.github/workflows/ci.yml` (`--cov-fail-under=60`). CI bloquea PRs que regresionen cobertura; si tu PR añade lógica nueva, añade tests antes de abrirlo.
