# Phase 2: Identidad (auth + RBAC + audit) — Context

**Gathered:** 2026-04-18
**Status:** Ready for research (planning skipped discuss, CONTEXT derivado de docs autoritativos)
**Source:** síntesis de `docs/AUTH_MODEL.md`, `docs/MARK_III_PLAN.md` §Sprint 1, `docs/OPEN_QUESTIONS.md` decisiones resueltas

<domain>
## Phase Boundary

**Objetivo**: toda request (`/api/*` o HTML) va autenticada; roles propietario/directivo/usuario + departamentos controlan qué puede hacer cada quién; cada acción queda registrada en una tabla append-only.

**Entregable end-to-end**:
- Tablas Postgres `nexo.users`, `nexo.roles`, `nexo.departments`, `nexo.user_departments`, `nexo.permissions`, `nexo.login_attempts`, `nexo.sessions`, `nexo.audit_log`.
- Login/logout funcional con bloqueo progresivo 5 intentos → 15 min lock `(user, IP)`.
- Middleware que redirige HTML a `/login` y devuelve 401 JSON para `/api/*` no autenticadas.
- Dependency `require_permission("modulo:accion")` aplicada a cada router existente (14 routers).
- Audit log append-only con middleware que graba cada request autenticada (body sanitizado).
- UI: `/login`, `/ajustes/usuarios` (CRUD, sólo propietario), `/ajustes/auditoria` (filtros + export CSV, sólo propietario en Mark-III).
- Seed inicial: al menos 1 usuario con rol `propietario` creado vía script de bootstrap interactivo.

**Qué NO entra en esta phase** (ver ROADMAP.md):
- UI condicionada por rol (sidebar filtrado) — Phase 5 / Sprint 4.
- Refactor de capa de datos (repositorios) — Phase 3 / Sprint 2.
- 2FA, LDAP — Mark-IV.
- Email de recuperación de contraseña — SMTP out of scope.
</domain>

<decisions>
## Implementation Decisions

Todas las decisiones autoritativas están en `docs/AUTH_MODEL.md`. Resumen aquí sólo para lectura rápida. **Ante cualquier conflicto, `docs/AUTH_MODEL.md` gana.**

### Modelo de roles (LOCKED — docs/AUTH_MODEL.md)
- 1 rol de nivel por usuario: `propietario` | `directivo` | `usuario`.
- N departamentos por usuario: subconjunto de `{rrhh, comercial, ingenieria, produccion, gerencia}`.
- `propietario` ignora departamento (global). Único que gestiona usuarios y ve audit completo.
- `directivo` ve sus departamentos. Audit filtrado por departamento diferido a Mark-IV.
- `usuario` sólo operaciones básicas dentro de sus departamentos.

### Passwords (LOCKED)
- Argon2id vía `argon2-cffi`.
- Parámetros OWASP 2024: `time_cost=3, memory_cost=65536, parallelism=4`.
- Mínimo 12 caracteres, sin reglas de complejidad adicionales.
- Cambio obligatorio al primer login (flag `must_change_password` en `nexo.users`).
- **Sin expiración forzada.**

### Bloqueo progresivo (LOCKED)
- 5 intentos fallidos consecutivos → 15 min lock sobre tupla `(user, IP)`.
- Purga al login exitoso.
- Login rate-limited adicionalmente por IP para cubrir ataque distribuido.

### Sesión
- Cookie HttpOnly + Secure + SameSite=Lax.
- Duración 12 horas con sliding expiration.
- **Claude's Discretion**: mecanismo concreto — cookie firmada con `itsdangerous` (stateful, requiere tabla `nexo.sessions`) vs JWT stateless. Decisión a tomar en el planner o delegar al researcher para pesar pros/cons. Default sugerido: **cookie firmada con tabla `nexo.sessions`** porque permite revocación inmediata (logout forzado) que JWT puro no da.

### Audit log (LOCKED)
- Tabla `nexo.audit_log`: `(id, ts, user_id, ip, method, path, status, details_json)`.
- Append-only a nivel BBDD: `REVOKE UPDATE, DELETE` al rol app sobre esa tabla. Sólo un rol admin de BBDD puede purgar.
- Middleware registra cada request autenticada. Whitelist de campos a grabar en `details_json`; endpoints sensibles (`/api/conexion/config` por ej.) se graban con `details_json = null` o campos redactados.

### Schema Postgres (LOCKED)
- Todas las tablas bajo schema `nexo`.
- `cfg.*`, `oee.*`, `luk4.*` permanecen en SQL Server (`ecs_mobility`) — no se tocan.

### Retrofit de permisos
- **LOCKED**: los 3 roles base (`propietario` / `directivo` / `usuario`) están fijados. No se reinventa la granularidad en este sprint.
- **LOCKED**: mapping modulo→departamentos se define en código (tabla `nexo.permissions` o dict en `nexo/services/auth.py`). Decisión de implementación abierta para el planner.
- **Claude's Discretion**: orden de retrofit en los 14 routers. Sugerencia: arrancar por los más sencillos (`health`, `centro_mando` readonly) para validar el dependency, luego escalar.

### Herramientas
- **LOCKED**: scripts Python + SQLAlchemy + `docker compose exec` (no MCP con write, no Alembic en Mark-III). Ver `docs/PROJECT.md` Key Decisions y conversación del 2026-04-18.
- Bootstrap del primer `propietario`: script interactivo `scripts/create_propietario.py` o `scripts/init_nexo_schema.py` con modo `--create-owner` que pide email + password al correrse.

### Estabilidad y compat
- `engine_nexo` (Postgres) es **nuevo** en Phase 2. `engine_app` (SQL Server / `ecs_mobility`) sigue siendo el de `api/database.py` actual, intacto.
- No se toca `OEE/db/connector.py` (MES) en esta phase.
- `global_exception_handler` ya está cerrado en Sprint 0 (commit 8 Phase 1). Verificar que no se regresa.

### Alcance visual UI
- `/login`, `/cambiar-password`, `/ajustes/usuarios`, `/ajustes/auditoria` son nuevas.
- `templates/base.html` añade nombre de usuario + logout en topbar.
- **Claude's Discretion**: diseño concreto de `/ajustes/usuarios` y `/ajustes/auditoria` (tablas con Alpine, exportar CSV lado cliente o servidor, paginación).
</decisions>

<code_context>
## Existing Code Insights

### Qué existe y tocamos sin romper
- `api/config.py` — ya tiene campos `pg_user/pg_password/pg_db` pineados a Postgres local. Añadir `secret_key` (para firmar cookies) y `session_cookie_name` en esta phase.
- `api/database.py` — engine SQL Server (`_mssql_creator`). Introducimos `engine_nexo` apuntando a Postgres como SEGUNDO engine. Pueden convivir (SQLAlchemy 2.0 soporta múltiples engines).
- `api/main.py` — añade middleware de auth antes del de audit. Orden: request → auth_middleware (401 o inject user) → audit_middleware (logea) → router.
- `api/deps.py` — globals Jinja2 ya tiene `app_name`, `company_name`, etc. Añadir `current_user` al contexto.
- Los 14 routers existentes reciben `dependencies=[Depends(require_permission(...))]`. Retrofit mecánico.

### Qué no existe y hay que crear
- Paquete `nexo/` no existe. El PLAN del Sprint 2 lo crea. Decisión: en Phase 2, ¿creamos ya `nexo/data/repositories/nexo.py` con `UserRepo`/`RoleRepo`/`AuditRepo` o esperamos a Phase 3? **Propuesta (Claude's Discretion abierta para el planner)**: introducir la estructura mínima `nexo/services/auth.py` + `nexo/db/models.py` ahora; el refactor completo de repositorios (`nexo/data/repositories/*`) llega en Phase 3.

### Riesgos específicos (recordatorio)
- **Sanitización de body en audit**: `/api/conexion/config` acepta passwords SQL. Middleware de audit debe tener whitelist o los sensibles los graba vacíos.
- **Orden de middlewares**: auth antes de audit (audit necesita user_id).
- **`engine_nexo` no existe** al arrancar en dev sin el script: el primer `make up` tras Phase 2 requiere correr `scripts/init_nexo_schema.py` antes del arranque de la app. Documentar.
- **Datos de planta no se mueven**: `cfg.recursos`, `cfg.ciclos`, `cfg.contactos` en SQL Server.
</code_context>

<specifics>
## Requisitos trazables (de REQUIREMENTS.md)

- **IDENT-01** — Login con bloqueo 5 → 15 min `(user, IP)`.
- **IDENT-02** — `nexo.users` con argon2id + must_change_password.
- **IDENT-03** — `nexo.roles` + `nexo.departments` + `nexo.user_departments` + `nexo.permissions`.
- **IDENT-04** — Middleware redirige HTML a `/login`, devuelve 401 para `/api/*`.
- **IDENT-05** — `Depends(require_permission(...))` aplicado a los 14 routers.
- **IDENT-06** — `nexo.audit_log` append-only (REVOKE UPDATE, DELETE).
- **IDENT-07** — `/ajustes/auditoria` con filtros + export CSV.
- **IDENT-08** — `/ajustes/usuarios` CRUD (sólo propietario).
- **IDENT-09** — Primer login obliga cambio de password.
- **IDENT-10** — Exception handler sigue sin filtrar traceback (regression check).

## Success criteria (de ROADMAP.md)

1. Navegar sin sesión redirige a `/login`; `curl /api/health` sin cookie responde 401 (excepción posible a `/api/health` — decisión del planner).
2. `propietario` ve y gestiona `/ajustes/usuarios` y `/ajustes/auditoria`.
3. `usuario` sólo ve operaciones básicas dentro de sus departamentos; sin ajustes.
4. 5 intentos fallidos lockean cuenta + IP durante 15 min.
5. Cada request autenticada genera fila en `nexo.audit_log`; `DELETE` desde rol app falla con error de permiso.
6. `/ajustes/auditoria` filtra por user/fecha/path/status y exporta CSV.

</specifics>

<deferred>
## Deferred Ideas

- 2FA (TOTP, WebAuthn) — Mark-IV.
- LDAP / Active Directory — Mark-IV.
- Audit filtrado por departamento visible a directivo — Mark-IV.
- Flujo "olvidé contraseña" por email — depende de SMTP (Out of Scope Mark-III).
- Migración a Alembic — propuesta para Mark-IV DevEx.
- MCP con write access — descartado; scripts + `docker compose exec` cubre el caso.
</deferred>

<canonical_refs>
## Canonical References

- `docs/AUTH_MODEL.md` — contrato autoritativo de auth.
- `docs/MARK_III_PLAN.md` — Sprint 1 section.
- `.planning/REQUIREMENTS.md` — IDENT-01..IDENT-10.
- `.planning/ROADMAP.md` — Phase 2 details + success criteria.
- `.planning/PROJECT.md` — Key Decisions.
- `.planning/codebase/ARCHITECTURE.md` — layout actual de api/, OEE/.
- `.planning/phases/01-naming-higiene-ci/01-01-SUMMARY.md` — cierre Phase 1 (compat layer OEE_*/NEXO_*, engine_nexo no existe, exception handler ya cerrado).

</canonical_refs>
