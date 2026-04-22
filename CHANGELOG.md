# Changelog

All notable changes to Nexo documented here.

El formato sigue [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
y el proyecto adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

## [1.0.0] - 2026-MM-DD

**Cierre de Mark-III.** Primera release productiva de Nexo como plataforma
interna de ECS Mobility. Sucesora del monolito "OEE Planta" manteniendo el
mismo stack; cambio de nombre y refactor estructural, no reescritura.

### Added

- **Phase 1 (Sprint 0) — Naming + Higiene + CI:** Rebrand completo
  `OEE Planta` -> `Nexo`. Env vars `OEE_*` -> `NEXO_*` con capa de compat.
  Exception handler sin traceback leak (solo `error_id` UUID al cliente).
  CI minimo en GitHub Actions (lint + test). Audit de credenciales expuestas
  en historial git (docs/SECURITY_AUDIT.md).
- **Phase 2 (Sprint 1) — Identidad (auth + RBAC + audit):** Auth con
  argon2id, minimo 12 chars, cambio obligatorio primer login. RBAC con roles
  `propietario`/`directivo`/`usuario` + 5 departamentos (`rrhh`, `comercial`,
  `ingenieria`, `produccion`, `gerencia`). Audit log append-only en
  `nexo.audit_log`. Lockout progresivo 5 intentos / 15 min por tupla
  `(user, IP)`. Sin 2FA, sin LDAP.
- **Phase 3 (Sprint 2) — Capa de datos:** Repositorios en
  `nexo/data/repositories/{app,mes,nexo}.py`. SQL versionado en
  `nexo/data/sql/` con loader. Tres engines SQLAlchemy separados:
  `engine_nexo` (Postgres `nexo.*`), `engine_app` (SQL Server
  `ecs_mobility`), `engine_mes` (SQL Server `dbizaro` read-only).
  `schema_guard` al arrancar valida tablas `nexo.*` criticas.
- **Phase 4 (Sprint 3) — Consultas pesadas:** Preflight + postflight para
  queries con coste alto. Flujo de aprobacion asincrono con TTL 7d. Umbrales
  editables en `/ajustes/limites` con LISTEN/NOTIFY para invalidar cache.
  Pipeline con `asyncio.Semaphore(3)` + timeout soft 15 min. Schedulers
  `approvals_cleanup`, `query_log_cleanup`, `factor_auto_refresh` sobre
  `nexo.query_log` (retencion 90d).
- **Phase 5 (Sprint 4) — UI por roles:** Sidebar filtrado por permisos
  via helper `can()`. `/ajustes` partido en 6 sub-paginas (`limites`,
  `usuarios`, `roles`, `departamentos`, `aprobaciones`, `auditoria`).
  11 botones + 8 GETs HTML gateados por permiso. FlashMiddleware para UX
  toast entre requests.
- **Phase 6 (Sprint 5) — Despliegue LAN HTTPS:** Caddy `tls internal` con
  root CA interna redistribuida manualmente a clientes LAN. Targets
  `make prod-up/down/logs/health/deploy`. Scripts `scripts/deploy.sh`
  (deploy atomico con pre-backup) y `scripts/backup_nightly.sh`
  (retencion 7d en `/var/backups/nexo/`). Runbook operativo
  `docs/DEPLOY_LAN.md` (740 lineas, 16 secciones). Smoke post-deploy
  `tests/infra/deploy_smoke.sh` con 11 checks LAN.
- **Phase 7 (Sprint 6) — DevEx hardening:** Pre-commit con hooks
  `ruff-check` + `ruff-format` + `mypy` scoped a `api/`+`nexo/` (hooks en
  `.pre-commit-config.yaml`). Config consolidada en `pyproject.toml`
  (ruff 0.15.11, mypy 1.13.0). CI con matriz Python 3.11 + 3.12, coverage
  gate `--cov-fail-under=60` bloqueante, job smoke con `docker compose up`
  + curl `/api/health`, service container `postgres:16-alpine`. Makefile
  con 5 targets DevEx (`test`, `test-docker`, `lint`, `format`, `migrate`)
  + `backup` de Phase 6 intacto. Docs `ARCHITECTURE.md`, `RUNBOOK.md`
  (5 escenarios), `RELEASE.md` (checklist versionado) y este CHANGELOG.

### Changed

- Carpeta `OEE/` mantiene nombre en Mark-III (rename diferido a Mark-IV
  coordinado con capa de repositorios). Modulos legacy `OEE/disponibilidad`,
  `OEE/rendimiento`, `OEE/calidad`, `OEE/oee_secciones` no se refactorizan.
- Repo GitHub mantiene nombre `analisis_datos` (rename diferido por coste
  alto vs beneficio cosmetico; decision cerrada CLAUDE.md).
- `make up` y `make dev` NO arrancan el servicio `mcp` (queda detras de
  `profiles: ["mcp"]` en `docker-compose.yml`). Solo arranca con
  `docker compose --profile mcp up`.
- Tablas `cfg.recursos`, `cfg.ciclos`, `cfg.contactos` se quedan en
  `ecs_mobility.cfg.*` (ya las leen Power BI y el tunel IoT). Solo las
  tablas `nexo.*` nuevas (users, roles, permissions, audit_log, query_log,
  login_attempts) viven en Postgres.

### Security

- Rotacion del password SA de SQL Server ejecutada en Sprint 0
  (ver `docs/SECURITY_AUDIT.md` para contexto; historial git NO se limpia
  con `git filter-repo` en Mark-III — decision cerrada CLAUDE.md).
- Exception handler de FastAPI devuelve solo `error_id` UUID al cliente,
  nunca el traceback (Phase 1). El traceback completo queda en los logs del
  servidor con correlacion por `error_id`.
- Audit log en Postgres con `GRANT SELECT, INSERT` (append-only). El usuario
  aplicacion no puede hacer UPDATE/DELETE sobre `nexo.audit_log`.
- Pre-commit + CI con `gitleaks` (Phase 7) — job `secrets` del CI con
  `continue-on-error: true` por findings historicos; se hara bloqueante en
  Mark-IV tras limpieza de historial.

### Deprecated

- Env vars `OEE_*` — aceptadas como fallback durante Mark-III via
  pydantic-settings. Eliminacion planificada para Mark-IV cuando todos los
  clientes internos hayan migrado a `NEXO_*`.

[Unreleased]: https://github.com/<org>/analisis_datos/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/<org>/analisis_datos/releases/tag/v1.0.0
