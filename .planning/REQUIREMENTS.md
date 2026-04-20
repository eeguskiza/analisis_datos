# Requirements: Nexo (Mark-III)

**Defined:** 2026-04-18
**Core Value:** Consultar datos reales de producción (MES IZARO) y generar informes OEE fiables por máquina/turno/sección sin bloquear al operario y sin filtrar información entre departamentos.

**Fuente de verdad**: este archivo se deriva de `docs/MARK_III_PLAN.md`
(entregables verificables de cada sprint) + `docs/OPEN_QUESTIONS.md` (decisiones
cerradas) + las decisiones de auth tomadas en la sesión de arranque de Sprint 0.

## Scope Mark-III (v1 requirements)

Cada requirement se mapea a exactamente una phase del ROADMAP.

### NAMING — Phase 1 / Sprint 0 (rebrand + higiene + CI + exception handler + audit)

- [ ] **NAMING-01**: Título FastAPI, sidebar, README y metadata OpenAPI dicen "Nexo" (no "OEE Planta"); banner/favicon/sidebar coherentes
- [ ] **NAMING-02**: Env vars renombradas `OEE_*` → `NEXO_*` con capa de compatibilidad (`api/config.py` acepta ambos prefijos durante Mark-III); SQL Server separadas en `NEXO_MES_*` y `NEXO_APP_*` apuntando al mismo host/credenciales
- [ ] **NAMING-03**: `docker-compose.yml` actualizado con prefijos `NEXO_*` y service names de Nexo; MCP ID renombrado de `oee-planta` a `nexo-mcp` (documentando breakage en `.claude.json` externos)
- [ ] **NAMING-04**: Archivos residuales eliminados del tracking: `.env:Zone.Identifier`, `test_email.py`, `server.py`; `.gitignore` añade patrón `*:Zone.Identifier`
- [ ] **NAMING-05**: Decisión sobre `data/oee.db` documentada: si contiene datos reales (>1 MB o rows significativas), snapshot a `data/backups/oee_db_snapshot.sqlite` (fuera de tracking) + nota en `docs/DATA_MIGRATION_NOTES.md`; si es residuo, borrado directo con justificación
- [ ] **NAMING-06**: `install_odbc.sh` ubicado en `scripts/` (mover si no está ahí)
- [ ] **NAMING-07**: `global_exception_handler` en `api/main.py` deja de devolver traceback al navegador; cada error genera UUID, traceback al log server-side, cliente recibe `{"error_id": "<uuid>", "message": "Internal error"}`
- [ ] **NAMING-08**: `requirements.txt` con versiones pineadas exactas (`package==x.y.z`); `requirements-dev.txt` creado con ruff, pytest, pytest-cov, gitleaks; `docker build` sigue funcionando tras el pin
- [ ] **NAMING-09**: `.github/workflows/ci.yml` con jobs `lint` (ruff check + format), `test` (pytest, non-blocking en Sprint 0), `build` (docker build sin push), `secrets` (gitleaks); triggers `push` a `feature/Mark-III` y `main` + PRs a ambas
- [ ] **NAMING-10**: Audit de historial git completado; `docs/SECURITY_AUDIT.md` lista credenciales expuestas (commit hash + archivo + tipo, sin valor literal); `filter-repo` **no** se ejecuta
- [ ] **NAMING-11**: `CLAUDE.md` creado en raíz con: decisiones cerradas, glosario, convenciones de naming, política de commits, regla de sync `.planning/` ↔ `docs/`
- [ ] **NAMING-12**: `docs/AUTH_MODEL.md` creado con modelo de roles (propietario/directivo/usuario) + departamentos (rrhh/comercial/ingeniería/producción/gerencia) + política de passwords (argon2id, 12 chars, cambio primer login, sin expiración) + bloqueo progresivo (5 → 15 min `user+IP`)
- [ ] **NAMING-13**: `docs/GLOSSARY.md` creado con términos: Nexo, MES, APP, módulo, rol, departamento, propietario, directivo, usuario, audit_log, preflight, postflight
- [ ] **NAMING-14**: `docs/MARK_III_PLAN.md` y `docs/OPEN_QUESTIONS.md` actualizados con las 5 decisiones bloqueantes resueltas + lagunas pendientes (SMTP, dominio, Ubuntu detalles, backups, LDAP futuro)
- [ ] **NAMING-15**: Post-Sprint-0 la app arranca con `make dev` y `docker build` sigue funcionando; `make health` devuelve OK
- [ ] **NAMING-16**: Servicio `mcp` en `docker-compose.yml` movido a `profiles: ["mcp"]`; `make up` y `make dev` **no** arrancan `mcp`; sólo `docker compose --profile mcp up`. `README.md` documenta cómo arrancarlo
- [ ] **NAMING-17**: Assets de marca estructurados en `static/img/brand/nexo/` y `static/img/brand/ecs/`; `docs/BRANDING.md` documenta uso, variables (`NEXO_LOGO_PATH`, `NEXO_ECS_LOGO_PATH`, `NEXO_APP_NAME`, `NEXO_COMPANY_NAME`) y templates (base.html, favicon, Notification) leen de esas variables en vez de rutas hardcoded

### IDENT — Phase 2 / Sprint 1 (auth + RBAC + audit)

- [ ] **IDENT-01**: Página `/login` funcional con bloqueo progresivo `5 intentos fallidos → 15 min lock de (user, IP)`
- [ ] **IDENT-02**: Tabla `nexo.users` en Postgres con hash argon2id, role, departments[], active, last_login, must_change_password
- [ ] **IDENT-03**: Tabla `nexo.roles` (propietario/directivo/usuario) y `nexo.departments` (rrhh/comercial/ingeniería/producción/gerencia) y `nexo.permissions` mapeando rol+dept → acciones por módulo
- [ ] **IDENT-04**: Middleware FastAPI redirige HTML a `/login` y devuelve `401` JSON para `/api/*` no autenticadas; cookie HttpOnly + Secure
- [ ] **IDENT-05**: Dependency `require_permission("modulo:accion")` aplicada a cada router; propietario ignora departamento (global); directivo y usuario filtrados por departamentos del user
- [ ] **IDENT-06**: Tabla `nexo.audit_log` append-only a nivel BBDD (`REVOKE UPDATE, DELETE` al rol app); middleware registra cada request autenticada con sanitización de campos sensibles (passwords SQL, etc.)
- [ ] **IDENT-07**: Panel `/ajustes/auditoria` con filtros (user, fecha, path, status) + export CSV (visible sólo a propietario en Mark-III; filtros por departamento en Mark-IV)
- [ ] **IDENT-08**: `/ajustes/usuarios` CRUD de usuarios, asignación de rol y departamentos (sólo propietario)
- [ ] **IDENT-09**: Primer login obliga a cambio de password (min 12 chars argon2id); flag `must_change_password` en user
- [ ] **IDENT-10**: `global_exception_handler` sin fuga de traceback (verificar que no se regresó tras cambios del Sprint 1)

### DATA — Phase 3 / Sprint 2 (repositorios + schema_guard)

- [ ] **DATA-01**: Paquete `nexo/data/engines.py` con `engine_mes` (dbizaro read-only, pool dedicado), `engine_app` (ecs_mobility, pool existente), `engine_nexo` (Postgres, creado en Sprint 1)
- [ ] **DATA-02**: `nexo/data/repositories/mes.py` con `MesRepository` (`extraer_datos_produccion`, `detectar_recursos`, `calcular_ciclos_reales`, `estado_maquina_live`, `consulta_readonly`)
- [ ] **DATA-03**: `nexo/data/repositories/app.py` con `RecursoRepo`, `CicloRepo`, `EjecucionRepo`, `MetricaRepo`, `LukRepo`, `ContactoRepo`
- [ ] **DATA-04**: `nexo/data/repositories/nexo.py` con `UserRepo`, `RoleRepo`, `AuditRepo` (consumidos por IDENT-*)
- [ ] **DATA-05**: `nexo/data/sql/` con queries versionadas en `.sql` (subdirectorios `mes/`, `app/`, `nexo/`); loader con placeholders `?`; comentario del filtro T3 cruza medianoche preservado
- [ ] **DATA-06**: `nexo/data/schema_guard.py` ejecutado en lifespan; falla arranque si falta tabla/columna crítica; flag `NEXO_AUTO_MIGRATE=true` crea lo que falte
- [ ] **DATA-07**: Routers consumen repositorios, no pyodbc directo ni SQL inline; queries duplicadas eliminadas de `bbdd.py`, `capacidad.py`, `operarios.py`, `centro_mando.py`, `luk4.py`, `historial.py`, `recursos.py`, `ciclos.py`
- [ ] **DATA-08**: DTOs Pydantic en `nexo/data/dto/` (`ProduccionRow`, `RecursoRow`, `CapacidadRow`, etc.)
- [ ] **DATA-09**: Cross-database references de 3-part names (`dbizaro.admuser.fmesmic`) eliminadas; engine MES apunta al catalog correcto vía `USE dbizaro` o connection string
- [ ] **DATA-10**: Tests `tests/data/` por repositorio con BD Postgres dedicada (fixture `docker compose up db` en CI); mocks para `engine_mes`
- [ ] **DATA-11**: `engine_mes` con `pool_recycle=3600`, `pool_pre_ping=True`, connection timeout 15s

### QUERY — Phase 4 / Sprint 3 (preflight + postflight)

- [x] **QUERY-01**: Tabla `nexo.query_log` con `(id, ts, user_id, endpoint, params_json, estimated_ms, actual_ms, rows, status)` — Plan 04-01 ✓ 2026-04-20
- [x] **QUERY-02**: Tabla `nexo.query_thresholds` con `(endpoint, warn_ms, block_ms, updated_at, updated_by)` editable desde UI — Plan 04-01 ✓ 2026-04-20 (schema + seeds; UI /ajustes/limites en Plan 04-04)
- [x] **QUERY-03**: `nexo/services/preflight.py` con `estimate_cost(endpoint, params) -> Estimation(ms, level, reason)`; heurística inicial `n_recursos × n_días × factor_por_modulo`; aprendizaje desde `nexo.query_log` — Plan 04-02 ✓ 2026-04-20 (estimador puro + 13 unit tests; learning via botón manual de /ajustes/limites en Plan 04-04)
- [x] **QUERY-04**: Endpoints devuelven `Estimation` antes de ejecutar; `green`=directo, `amber`=confirmación UI, `red`=requiere aprobación propietario — Plan 04-02 ✓ 2026-04-20 (4 routers gatean por level: 428 amber, 403 red; modal AMBER/RED Alpine en pipeline.html + bbdd.html)
- [x] **QUERY-05**: `nexo/middleware/query_timing.py` mide `time.monotonic()` y escribe `actual_ms` en `query_log`; alerta `logging.WARNING` si `actual > warn_ms × 1.5` — Plan 04-02 ✓ 2026-04-20 (middleware innermost + D-17 slow status + log.warning con ratio)
- [x] **QUERY-06**: Flujo approval asíncrono: `/ajustes/solicitudes` (propietario aprueba), usuario re-dispara con `force=true` — Plan 04-03 ✓ 2026-04-20 (6 funciones service + 7 endpoints router + 2 templates + badge HTMX 30s + job cleanup Mon 03:05 + TTL 7d; PC-04-07 xfail removido; _APPROVALS_AVAILABLE activado en 4 routers)
- [x] **QUERY-07**: `/ajustes/limites` edita umbrales; preflight aplicado en `pipeline/run`, `bbdd/query`, `capacidad` (rango > 90 días), `operarios` (rango > 90 días) — Plan 04-02 ✓ 2026-04-20 (preflight aplicado a los 4 endpoints; UI /ajustes/limites en Plan 04-04)
- [x] **QUERY-08**: Pipeline OEE corre en `asyncio.to_thread` para no bloquear worker uvicorn (matplotlib sigue síncrono internamente pero no congela el resto de la UI) — Plan 04-02 ✓ 2026-04-20 (asyncio.Semaphore(3) + asyncio.to_thread + wait_for(timeout=900) con landmine soft-timeout documentado)

### UIROL — Phase 5 / Sprint 4 (UI por roles)

- [ ] **UIROL-01**: `request.state.user` inyectado en templates vía `Jinja2Templates` globals o dependency en `pages.py`
- [ ] **UIROL-02**: `base.html` filtra `nav_items` según permisos; propietario ve todo; directivo ve módulos de sus departamentos; usuario ve subset reducido
- [ ] **UIROL-03**: `templates/ajustes.html` dividido en `ajustes_conexion.html`, `ajustes_smtp.html`, `ajustes_usuarios.html`, `ajustes_auditoria.html`, `ajustes_limites.html`, `ajustes_solicitudes.html`, + hub `ajustes.html`
- [ ] **UIROL-04**: `static/js/app.js` oculta botones sensibles ("Ejecutar pipeline", "Borrar ejecución", "Sincronizar recursos") según permiso del usuario cargado en contexto
- [ ] **UIROL-05**: Verificación manual E2E: login como propietario/directivo (un departamento)/usuario (un departamento), capturar sidebar y páginas accesibles de cada rol

### DEPLOY — Phase 6 / Sprint 5 (LAN HTTPS)

- [ ] **DEPLOY-01**: `docker-compose.prod.yml` (o `profiles: [prod]`) con Caddyfile usando `nexo.ecsmobility.com` — Let's Encrypt DNS-01 si se controla el DNS público, fallback a `tls internal` documentado
- [ ] **DEPLOY-02**: Postgres sin publicar puerto 5432 al host en prod; acceso sólo vía `docker compose exec db psql`
- [ ] **DEPLOY-03**: Healthchecks añadidos a servicios `web` y `caddy`; `restart: unless-stopped` consistente en todos los servicios
- [ ] **DEPLOY-04**: `scripts/deploy.sh` implementa `git pull && docker compose --profile prod build --pull && docker compose --profile prod up -d`
- [ ] **DEPLOY-05**: `docs/DEPLOY_LAN.md` con: instalación Docker + plugin compose en Ubuntu Server 24.04, DNS interno (A record), DNS-01 registros, rotación SMTP/SQL, backup `pgdata`, plan de recuperación
- [ ] **DEPLOY-06**: Firewall Ubuntu (`ufw`): 22 (red interna), 443, 80 (redirect opcional); deny all else
- [ ] **DEPLOY-07**: Verificación desde otro equipo LAN: `https://nexo.ecsmobility.com` carga con cert válido (LE) o autofirmado documentado en DEPLOY_LAN.md
- [ ] **DEPLOY-08**: `.env.prod.example` con claves de producción sin valores reales

### DEVEX — Phase 7 / Sprint 6 (hardening)

- [ ] **DEVEX-01**: Pre-commit con ruff + black + mypy ligero (sólo `api/` y `nexo/`, OEE excluido); black + ruff --fix ejecutados previamente en commit aislado para no contaminar diffs
- [ ] **DEVEX-02**: CI ampliado con matriz Python 3.11 y 3.12; cobertura mínima 60% en `api/` y `nexo/`; smoke test `docker compose up` + `curl /api/health`
- [ ] **DEVEX-03**: Makefile añade targets `make test`, `make lint`, `make format`, `make migrate`, `make backup`
- [ ] **DEVEX-04**: `docs/ARCHITECTURE.md` con diagrama de componentes (web ↔ engine_nexo ↔ Postgres, web ↔ engine_app ↔ ecs_mobility, web ↔ engine_mes ↔ dbizaro)
- [ ] **DEVEX-05**: `docs/RUNBOOK.md` con procedimiento para 5 escenarios: MES caído, Postgres no arranca, certificado expira, pipeline atascado, lockout de propietario
- [ ] **DEVEX-06**: `docs/RELEASE.md` con checklist de release (tag semver, deploy, smoke test)
- [ ] **DEVEX-07**: `CLAUDE.md` actualizado con convenciones Mark-III completas tras Sprint 6

## v2 Requirements (Mark-IV+)

Diferidas explícitamente. No se planifican en Mark-III.

### Streaming / Realtime

- **RT-01**: Ingesta OPC-UA / MQTT desde PLCs
- **RT-02**: Dashboards en streaming (reemplazar polling HTMX 30s del badge de conexión)

### Módulos nuevos

- **MOD-01**: Módulo de calidad (más allá del cálculo OEE actual)
- **MOD-02**: Módulo de trazabilidad de pieza a orden

### Refactor interno

- **REF-01**: Refactor de los 4 módulos OEE para aceptar iterables de filas en lugar de leer CSVs del disco
- **REF-02**: Sustituir matplotlib por WeasyPrint / ReportLab / Playwright para PDFs
- **REF-03**: Rename de carpeta `OEE/` a `modules/oee/`
- **REF-04**: CRUD de `plantillas.html` enganchado al pipeline

### Seguridad y Auth avanzada

- **SEC-01**: 2FA (TOTP, WebAuthn)
- **SEC-02**: Integración LDAP / Active Directory
- **SEC-03**: Filtros por departamento en panel de auditoría (visibles a directivo)
- **SEC-04**: Rotación automatizada de credenciales SQL Server
- **SEC-05**: Reescritura de historial git si `docs/SECURITY_AUDIT.md` lo justifica

### Infra

- **INF-01**: Exposición a internet con dominio público
- **INF-02**: Monitorización (Prometheus + Grafana) y alertas
- **INF-03**: SMTP configurado y operativo

## Out of Scope

Exclusiones definitivas en el alcance Mark-III — no confundir con v2.

| Feature | Reason |
|---------|--------|
| Microservicios | Contradice la estrategia strangler-fig; ratio complejidad/beneficio malo para equipo de 1-2 devs |
| Rotación de credenciales SQL Server en Sprint 0 | Decisión explícita del operador: diferida hasta revisar `docs/SECURITY_AUDIT.md` |
| Configuración SMTP operativa | Pendiente de decisión de infraestructura (qué servidor, qué from); mantiene estado actual "roto hasta que alguien rellene el JSON" |
| Rename del repo GitHub | Coste alto (hooks, CI, clones de todos), beneficio cosmético |
| Rename de carpeta `OEE/` en Mark-III | Diferido a Sprint 2 para evitar churn durante Sprint 0 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| NAMING-01 | Phase 1 | Pending |
| NAMING-02 | Phase 1 | Pending |
| NAMING-03 | Phase 1 | Pending |
| NAMING-04 | Phase 1 | Pending |
| NAMING-05 | Phase 1 | Pending |
| NAMING-06 | Phase 1 | Pending |
| NAMING-07 | Phase 1 | Pending |
| NAMING-08 | Phase 1 | Pending |
| NAMING-09 | Phase 1 | Pending |
| NAMING-10 | Phase 1 | Pending |
| NAMING-11 | Phase 1 | Pending |
| NAMING-12 | Phase 1 | Pending |
| NAMING-13 | Phase 1 | Pending |
| NAMING-14 | Phase 1 | Pending |
| NAMING-15 | Phase 1 | Pending |
| NAMING-16 | Phase 1 | Pending |
| NAMING-17 | Phase 1 | Pending |
| IDENT-01 | Phase 2 | Pending |
| IDENT-02 | Phase 2 | Pending |
| IDENT-03 | Phase 2 | Pending |
| IDENT-04 | Phase 2 | Pending |
| IDENT-05 | Phase 2 | Pending |
| IDENT-06 | Phase 2 | Pending |
| IDENT-07 | Phase 2 | Pending |
| IDENT-08 | Phase 2 | Pending |
| IDENT-09 | Phase 2 | Pending |
| IDENT-10 | Phase 2 | Pending |
| DATA-01 | Phase 3 | Pending |
| DATA-02 | Phase 3 | Pending |
| DATA-03 | Phase 3 | Pending |
| DATA-04 | Phase 3 | Pending |
| DATA-05 | Phase 3 | Pending |
| DATA-06 | Phase 3 | Pending |
| DATA-07 | Phase 3 | Pending |
| DATA-08 | Phase 3 | Pending |
| DATA-09 | Phase 3 | Pending |
| DATA-10 | Phase 3 | Pending |
| DATA-11 | Phase 3 | Pending |
| QUERY-01 | Phase 4 / Plan 04-01 | Complete (2026-04-20) |
| QUERY-02 | Phase 4 / Plan 04-01 | Partial (schema + seeds complete; UI in Plan 04-04) |
| QUERY-03 | Phase 4 / Plan 04-02 | Complete (2026-04-20) |
| QUERY-04 | Phase 4 / Plan 04-02 | Complete (2026-04-20) |
| QUERY-05 | Phase 4 / Plan 04-02 | Complete (2026-04-20) |
| QUERY-06 | Phase 4 / Plan 04-03 | Complete (2026-04-20) |
| QUERY-07 | Phase 4 / Plan 04-02 | Complete (2026-04-20) (UI /ajustes/limites en 04-04) |
| QUERY-08 | Phase 4 / Plan 04-02 | Complete (2026-04-20) |
| UIROL-01 | Phase 5 | Pending |
| UIROL-02 | Phase 5 | Pending |
| UIROL-03 | Phase 5 | Pending |
| UIROL-04 | Phase 5 | Pending |
| UIROL-05 | Phase 5 | Pending |
| DEPLOY-01 | Phase 6 | Pending |
| DEPLOY-02 | Phase 6 | Pending |
| DEPLOY-03 | Phase 6 | Pending |
| DEPLOY-04 | Phase 6 | Pending |
| DEPLOY-05 | Phase 6 | Pending |
| DEPLOY-06 | Phase 6 | Pending |
| DEPLOY-07 | Phase 6 | Pending |
| DEPLOY-08 | Phase 6 | Pending |
| DEVEX-01 | Phase 7 | Pending |
| DEVEX-02 | Phase 7 | Pending |
| DEVEX-03 | Phase 7 | Pending |
| DEVEX-04 | Phase 7 | Pending |
| DEVEX-05 | Phase 7 | Pending |
| DEVEX-06 | Phase 7 | Pending |
| DEVEX-07 | Phase 7 | Pending |

**Coverage:**
- v1 requirements: 66 total
- Mapped to phases: 66
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-18*
*Last updated: 2026-04-18 after closing open questions (added NAMING-16 MCP profile, NAMING-17 brand assets)*
