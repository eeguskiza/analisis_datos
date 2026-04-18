# Nexo · Mark-III — Plan de sprints

Rediseño del monolito OEE actual a **Nexo**, plataforma interna de ECS que
integra OEE, calidad, trazabilidad y futuros módulos. Mark-III se limita a:
auth + roles + RBAC, auditoría, refactor de capa de datos, sistema de
consultas pesadas, refactor UI por roles y despliegue LAN con HTTPS.

Estrategia: **strangler fig** en el mismo repo, historial conservado.
Stack sin cambios (FastAPI + Jinja2 + Alpine + Tailwind + Postgres 16 +
SQL Server via pyodbc + Docker Compose).

Las estimaciones son **días de trabajo focalizado**, no días de calendario.

> **Actualizado 2026-04-18 — sesión de arranque Sprint 0.**
>
> - 5 decisiones bloqueantes de `docs/OPEN_QUESTIONS.md` cerradas. Resumen inline en ese doc; tabla completa en `.planning/PROJECT.md` "Key Decisions".
> - Alcance de **Sprint 0** ampliado con tres piezas que no estaban (o estaban en Sprint 1) en el bosquejo inicial: **audit de historial git**, **fix del `global_exception_handler`** y **mover `mcp` a profile compose**. Detalle en la sección Sprint 0 más abajo.
> - **Sprint 1** reemplaza los roles `admin / analyst / viewer` por el modelo `propietario / directivo / usuario + departamentos` acordado. Ver `docs/AUTH_MODEL.md` (autoritativo). El bloqueo progresivo se simplifica a **5 intentos → 15 min** lock `(user, IP)`.
> - **Sin fecha objetivo ni hito externo.** Estimación orientativa 10-12 semanas focalizadas.

---

## Desviación respecto al plan base (justificada)

El plan que trajiste era: `0 naming → 1 datos → 2 auth → 3 audit → 4
consultas pesadas → 5 UI → 6 despliegue → 7 DevEx`.

Propongo dos ajustes:

1. **Sprint 2 (auth) se adelanta a Sprint 1 (datos)**, reordeno a `0 → 2 → 1`.
   Razón: la capa de datos es tocar 4 K+ líneas de módulos OEE y queries
   hardcodeadas en 5 routers. Durante ese refactor vas a querer **poder
   autenticar ya** para que el panel admin esté detrás de login y para que
   los logs muestren quién provocó cada error. Si dejas auth para después
   del refactor de datos, te pasas 3-5 días operando una app pública en
   LAN mientras tocas lo más delicado.
2. **Sprint 3 (audit) se funde con Sprint 2 (auth)** en un único sprint
   "Identidad + Audit". Razón: el middleware de audit depende de la sesión
   autenticada para saber quién registra. Partirlos obliga a hacer dos
   middlewares casi idénticos. Lo junto y el "sprint" resultante es 5-6
   días, no 3+3.

Reorden final:

| # | Nombre | Alias del plan original |
|---|---|---|
| 0 | Naming + estructura + higiene | Sprint 0 |
| 1 | Identidad: auth + RBAC + audit | Sprint 2 + Sprint 3 |
| 2 | Capa de datos: repositorios + schema_guard | Sprint 1 |
| 3 | Consultas pesadas: preflight + postflight | Sprint 4 |
| 4 | UI por roles | Sprint 5 |
| 5 | Despliegue LAN con HTTPS | Sprint 6 |
| 6 | DevEx hardening | Sprint 7 |

Si prefieres mantener el orden original tal cual, decídmelo — es reversible.

---

## Sprint 0 — Naming + estructura + higiene

**Objetivo**: repo arrancable con marca "Nexo", higiene básica (residuos,
Zone.Identifier, requirements pineados, CI mínimo), exception handler que
**no** filtra traceback, audit del historial git documentado, y un
`CLAUDE.md` que le explique a cualquier dev futuro cómo navegar el código.

**Entregable verificable** (versión refinada 2026-04-18):
- Repo compila con `make build && make up` y `/api/health` responde OK. `make dev` arranca también. `mcp` **no** arranca por default (está en `profiles: ["mcp"]`).
- Título FastAPI dice "Nexo", sidebar dice "NEXO", README renombrado, metadata OpenAPI consistente.
- `.env.example` completo con nueva nomenclatura `NEXO_*` (split `NEXO_MES_*` + `NEXO_APP_*`) y variables de branding. `api/config.py` acepta compat `OEE_*` durante Mark-III.
- `.env:Zone.Identifier`, `test_email.py` y `server.py` eliminados del tracking. `.gitignore` añade `*:Zone.Identifier` y residuos comunes.
- `install_odbc.sh` movido a `scripts/` si no estaba ahí.
- Decisión sobre `data/oee.db` documentada (snapshot o borrado) en `docs/DATA_MIGRATION_NOTES.md`.
- `global_exception_handler` de `api/main.py` devuelve `{"error_id": "<uuid>", "message": "Internal error"}` — nunca traceback.
- Servicio `mcp` en `docker-compose.yml` bajo `profiles: ["mcp"]`. `README.md` documenta `docker compose --profile mcp up`.
- `.github/workflows/ci.yml` con jobs `lint` (ruff), `test` (pytest, no bloqueante), `build` (docker build), `secrets` (gitleaks). Trigger push a `feature/Mark-III` + `main` y PRs a ambas. **No bloqueante** como status check en Sprint 0.
- `requirements.txt` con versiones pineadas exactas (`package==x.y.z`); `requirements-dev.txt` con ruff/pytest/pytest-cov/gitleaks.
- `docs/SECURITY_AUDIT.md` lista credenciales expuestas en el historial (sin valores literales). **No se ejecuta `filter-repo`**.
- `CLAUDE.md`, `docs/AUTH_MODEL.md`, `docs/GLOSSARY.md`, `docs/BRANDING.md`, `docs/DATA_MIGRATION_NOTES.md` creados. `docs/MARK_III_PLAN.md` y `docs/OPEN_QUESTIONS.md` actualizados con las decisiones cerradas.
- Assets de marca estructurados en `static/img/brand/{nexo,ecs}/`; templates consumen `NEXO_LOGO_PATH` / `NEXO_ECS_LOGO_PATH` desde contexto, no hardcoded.

**Rotación de credenciales SQL Server**: **NO entra en Sprint 0**. Decisión diferida hasta revisar `docs/SECURITY_AUDIT.md`.

**13 commits atómicos** (detalle en `.planning/1-sprint-0-naming-and-hygiene/PLAN.md`):

1. `chore: audit git history for leaked credentials` — GATE: sólo genera `SECURITY_AUDIT.md`. No modifica código.
2. `chore: remove tracked junk files` — `.env:Zone.Identifier`, `test_email.py`, `server.py`.
3. `chore: update .gitignore patterns` — `*:Zone.Identifier` y residuos.
4. `chore: move install_odbc.sh to scripts/` — si no está ahí.
5. `chore: handle data/oee.db` — decisión documentada: inspeccionar; si residuo, borrar; si datos reales, snapshot + documentar.
6. `refactor: rename OEE_* env vars to NEXO_*, split MES/APP` — `api/config.py`, `docker-compose.yml`, `.env.example`. Compat `OEE_*` activa.
7. `refactor: update UI titles and metadata to Nexo` — títulos visibles, OpenAPI, compose service names, uso de `NEXO_LOGO_PATH` y `NEXO_ECS_LOGO_PATH` en templates.
8. `fix: global_exception_handler no longer leaks tracebacks` — UUID al cliente, traceback server-side.
9. `chore: move mcp service to docker-compose profile` — `profiles: ["mcp"]` + documentar en README.
10. `build: pin requirements.txt, add requirements-dev.txt` — versiones exactas; dev deps aisladas.
11. `ci: add GitHub Actions workflow` — 4 jobs: lint, test (no bloqueante), build, secrets.
12. `docs: add GLOSSARY, DATA_MIGRATION_NOTES (verify CLAUDE.md, AUTH_MODEL, BRANDING alignment)` — los tres primeros ya existen de la sesión de arranque.
13. `docs: sync MARK_III_PLAN and OPEN_QUESTIONS with Sprint 0 outcomes` — actualización final con resultado efectivo del sprint (hashes, decisiones finales).

**Riesgos específicos**:
- Cambiar prefijo `OEE_*` a `NEXO_*` rompe el `.env` actual en el servidor LAN si ya hay uno. **Mitigación**: `api/config.py` acepta ambos prefijos durante Mark-III.
- El prefijo `OEE_` aparece también en `docker-compose.yml`. Actualizar al mismo tiempo.
- Renombrar el servicio MCP (ID `oee-planta` → `nexo-mcp`) invalida cualquier `.claude.json` externo apuntando al ID viejo. Documentarlo en README.
- Mover assets de logo rompe las rutas hardcoded en `templates/base.html`, `static/js/app.js`, `api/main.py`. Resolverlo en el mismo commit (7) que actualiza templates y usa las variables de contexto.
- El audit de historial (commit 1) puede descubrir credenciales expuestas. Se documenta pero **no** se actúa sin autorización explícita.

**Estimación**: **2-3 días** (ampliado frente a 2 originales por audit + exception handler + mcp profile + logos).

**Dependencias**: ninguna.

**Ejecutado**: 2026-04-18, rama `feature/Mark-III`, via `/gsd-execute-phase 1 --interactive` checkpoint-por-hito.

Commits del sprint:

1. `8774deb` — chore: audit git history for leaked credentials (Gate 1).
2. `dec03d1` — chore: remove tracked junk files.
3. `bfe35e7` — chore: update .gitignore patterns (`*:Zone.Identifier`).
4. `7babe5d` — chore: move install_odbc.sh to scripts/.
5. `151d83c` — chore: handle data/oee.db (backup offline + borrado).
6. `10f8164` — refactor: rename OEE_* env vars to NEXO_*, split MES/APP.
7. `f5d0ce7` — refactor: update UI titles and metadata to Nexo.
8. `dc5b7df` — fix: global_exception_handler no longer leaks tracebacks.
9. `f689212` — chore: move mcp service to docker-compose profile.
10. `72d658c` — build: pin requirements.txt, add requirements-dev.txt.
11. `1edf57a` — ci: add GitHub Actions workflow (lint, test, build, secrets).
12. `4889811` — docs: add GLOSSARY, DATA_MIGRATION_NOTES (verify CLAUDE.md, AUTH_MODEL, BRANDING alignment).
13. *(este commit)* — docs: sync MARK_III_PLAN and OPEN_QUESTIONS with Sprint 0 outcomes.

Desviaciones respecto al plan:
- Commit 2 amplió scope para editar `Dockerfile` (eliminar `COPY server.py .`) por dependencia directa del delete de `server.py`. Sin ello el build rompía.
- Commit 5 tomó el camino "preservar backup offline" (no "borrar directo") porque la inspección encontró 1 433 filas reales en `datos_produccion`, 10 ejecuciones, 58 ciclos. Justificado en `docs/DATA_MIGRATION_NOTES.md`.
- Commit 7 añadió inyección de `window.NEXO_CONFIG` en `base.html` para que `static/js/app.js` pueda leer `appName`, `logoPath`, etc. sin hardcoded strings.
- H1 y H2 del audit (commit 1) resueltos por el operador rotando la password SA antes del cierre del sprint; ver `docs/SECURITY_AUDIT.md` sección Operator Review.

Hallazgos fuera del alcance propuestos para sprints posteriores:
- `showToast` duplicado entre `base.html` (Alpine) y `app.js` (DOM append) — limpieza cosmética, Mark-IV.
- `data/ecs-logo.png` sigue separado de `static/img/brand/ecs/logo.png` porque lo consume matplotlib vía `api/config.py:logo_filename`. Unificación en Sprint 2.
- `pandas==3.0.2` pineado: la columna major 3.x introduce cambios de API respecto a 2.x. Verificar que `api/services/pipeline.py` y módulos OEE no tocaron algo que 3.x rompe.
- Tests actuales (`tests/test_oee_calc.py`, `tests/test_oee_helpers.py`): no se han ejecutado en CI todavía (el workflow existe pero es no-bloqueante). Ejecutar manualmente al primer push.

---

## Sprint 1 — Identidad: auth + RBAC + audit

**Objetivo**: toda request con `/api/*` o HTML autenticada; roles con
permisos granulares; cada acción registrada en una tabla append-only.

> Modelo de auth autoritativo: ver `docs/AUTH_MODEL.md`. Los bullets de
> abajo se ajustan al modelo definitivo (`propietario / directivo /
> usuario` + departamentos, bloqueo `5 → 15 min`), que sustituye al
> bosquejo `admin / analyst / viewer` del borrador inicial.

**Entregable verificable**:
- Página `/login` funcional con bloqueo progresivo **5 intentos fallidos → 15 min** lock sobre `(user, IP)`. Tabla `nexo.login_attempts` con TTL y purga al login exitoso.
- Tabla `nexo.users` en Postgres: `id`, `email`, `password_hash` (argon2id), `role_id`, `active`, `last_login`, `must_change_password`, `created_at`, `updated_at`.
- Tabla `nexo.roles` (semilla: `propietario`, `directivo`, `usuario`) y `nexo.departments` (semilla: `rrhh`, `comercial`, `ingenieria`, `produccion`, `gerencia`).
- Tabla `nexo.user_departments` (N:M entre users y departments).
- Tabla `nexo.permissions` con mapping `rol × modulo × accion` (ejemplo: `usuario:pipeline:read`, `directivo:admin:write`). Propietario ignora departamento.
- Middleware FastAPI que redirige HTML a `/login` y devuelve `401` JSON para `/api/*` no autenticadas. Cookie HttpOnly + Secure + SameSite=Lax.
- Dependency `Depends(require_permission("modulo:accion"))` aplicada a cada router.
- Tabla `nexo.audit_log` append-only: `(id, ts, user_id, ip, method, path, status, details_json)`. **Append-only** a nivel BBDD: `REVOKE UPDATE, DELETE` del rol app; sólo un rol admin de BBDD puede purgar.
- Middleware que registra **cada request autenticada**.
- Panel `/ajustes/auditoria` con filtros (user, fecha, path, status) + export CSV (visible sólo a `propietario` en Mark-III).
- `/ajustes/usuarios` CRUD de usuarios + asignación de rol + asignación de departamentos (sólo `propietario`).
- Primer login obliga a cambio de password (flag `must_change_password`).
- `global_exception_handler` **ya está arreglado en Sprint 0** — verificar que no se regresó tras cambios de este sprint.

**Archivos/módulos tocados**:
- `api/routers/auth.py` (nuevo), `api/routers/admin.py` (nuevo).
- `api/services/auth.py` (argon2, tokens de sesión en cookie HttpOnly
  Secure, verificación), `api/services/audit.py` (nuevo).
- `api/middleware/auth_middleware.py` (nuevo),
  `api/middleware/audit_middleware.py` (nuevo).
- `api/database.py` — añadir modelos User, Role, Permission, AuditLog
  contra Postgres (nuevo engine, `engine_nexo`). Engine existente `mssql`
  sigue para `cfg/oee/luk4`.
- Nueva migración Postgres (ver Sprint 2 para el mecanismo; en Sprint 1
  usamos `scripts/init_postgres.py` como hoy se usa `create_oee_db.py`).
- Cada router existente: añadir `dependencies=[Depends(require_permission(...))]`.
- `templates/login.html` (nuevo), `templates/ajustes_usuarios.html` (nuevo),
  `templates/ajustes_auditoria.html` (nuevo).
- `templates/base.html` — mostrar nombre de usuario + logout en topbar.
- `api/main.py` — reescribir `global_exception_handler` para que no
  devuelva traceback al navegador (fuga de info).

**Riesgos específicos**:
- El `global_exception_handler` ya está cerrado en Sprint 0. Verificar en este sprint que ningún cambio de auth lo regresa.
- El middleware de audit captura el body de POST/PUT. Esos bodies pueden contener credenciales SQL (endpoint `/api/conexion/config`). Sanitizar en el servicio de audit: whitelist de campos a grabar, o `details_json` vacío para endpoints marcados como "sensibles".
- La política de bloqueo progresivo (5 → 15 min) requiere tabla `nexo.login_attempts` con TTL y purga al login exitoso.
- El login también tiene que estar rate-limited por IP (excepción al "no rate limit" general), porque sin eso un atacante distribuido puede probar 10 K usuarios en paralelo.
- **Retro-fit de permisos** en los 14 routers es una pasada mecánica larga. Los 3 roles base (`propietario` / `directivo` / `usuario`) + departamentos están fijados en `docs/AUTH_MODEL.md` — trabajar con ese contrato, no reinventar la granularidad en el sprint.

**Estimación**: **6 días**.

**Dependencias**: Sprint 0 (prefijo env vars estable antes de añadir
`NEXO_SECRET_KEY`, `NEXO_SESSION_COOKIE`...).

---

## Sprint 2 — Capa de datos: repositorios + schema_guard

**Objetivo**: eliminar queries SQL embebidas en routers; aislar acceso a
`dbizaro` (MES read-only) y a `ecs_mobility` (BD propia). `schema_guard`
que valida al arrancar que las tablas esperadas existen con las columnas
esperadas.

**Entregable verificable**:
- `nexo/data/` (nuevo paquete) con sub-módulos:
  - `nexo/data/engines.py` — dos engines: `engine_mes` (dbizaro,
    read-only, conexión dedicada), `engine_app` (ecs_mobility, pool
    actual), `engine_nexo` (Postgres — ya creado en Sprint 1).
  - `nexo/data/repositories/mes.py` — `MesRepository` con métodos
    `extraer_datos_produccion`, `detectar_recursos`,
    `calcular_ciclos_reales`, `estado_maquina_live`, `consulta_readonly`.
  - `nexo/data/repositories/app.py` — `RecursoRepo`, `CicloRepo`,
    `EjecucionRepo`, `MetricaRepo`, `LukRepo`, `ContactoRepo`.
  - `nexo/data/repositories/nexo.py` — `UserRepo`, `RoleRepo`, `AuditRepo`.
- `nexo/data/sql/` — queries movidas a ficheros `.sql` versionados, con
  sufijo por repo (`mes/extract_produccion.sql`, `mes/capacidad.sql`,
  etc.). Los repositorios los cargan con un loader simple que soporta
  placeholders `?`.
- `nexo/data/schema_guard.py` — al arrancar (lifespan), comprueba que cada
  tabla/columna que el código va a tocar existe en la instancia. Falla el
  arranque si el esquema no está listo; loguea con detalle qué falta.
- Routers actuales consumen repositorios (no pyodbc, no SQL inline).
- Borrado del SQL duplicado en `bbdd.py`, `capacidad.py`, `operarios.py`,
  `centro_mando.py`, `luk4.py`, `historial.py`, `recursos.py`, `ciclos.py`.
- `OEE/db/connector.py` se vacía: lo que queda se mueve a
  `nexo/data/repositories/mes.py` y el módulo se elimina.
- DTOs Pydantic explícitos en `nexo/data/dto/`: `ProduccionRow`, `RecursoRow`,
  `CapacidadRow`, etc. Los módulos OEE siguen aceptando CSV (no los tocamos
  en Mark-III), pero el pipeline llama a repo.extraer_datos → DTOs → escribe
  CSV temporales igual que hoy.
- Queries contra dbizaro ya no usan 3-part names; van contra `engine_mes`
  apuntado al catalog correcto.

**Archivos/módulos tocados**:
- Todo lo indicado arriba. **5 routers pierden entre 100-500 líneas cada
  uno**. Es el sprint más grande de código, probablemente 3 K líneas
  movidas o reescritas.
- **No tocar** los 4 módulos de `OEE/disponibilidad/rendimiento/calidad/oee_secciones`
  salvo para que acepten el logo desde `settings` en lugar de path hardcoded.
  Mark-IV los refactoriza.
- Tests nuevos: `tests/data/` con tests por repositorio, usando una BD
  Postgres dedicada a tests y mocks para `engine_mes` (la BBDD IZARO real
  está en producción y no tocamos tests contra ella).

**Riesgos específicos**:
- **Cross-database references**: `routers/centro_mando.py` hoy hace
  `FROM dbizaro.admuser.fmesmic` desde el engine de `ecs_mobility`. Al
  separar engines, hay que decidir: ¿el repo MES recibe el ct_code y
  devuelve piezas, y luego el orquestador lo junta con recursos? (más limpio).
  Sí — en el sprint se reescribe esto.
- **El SQL de `extraer_datos` tiene lógica de T3 cruzando medianoche**
  que está bien pensada. Al moverla a un `.sql`, conservar un comentario
  al inicio explicando el filtro de `fecha_fin+1 AND dt080<600`. Sin ese
  contexto el siguiente dev lo "simplifica" y rompe el último turno.
- **schema_guard al arrancar**: la primera vez post-deploy el guard
  encontrará cosas nuevas (`nexo.users`, etc.). Diseñar el guard para
  que *liste lo que falta* y lo cree si hay flag `NEXO_AUTO_MIGRATE=true`,
  o que haga `print` y salga con código 1 si falta algo crítico.
- **Pool de `engine_mes`**: hoy cada router abre y cierra pyodbc. Con
  pool, hay que pensar reciclaje (`pool_recycle=3600`), `pool_pre_ping`,
  y timeout más corto (15s) porque MES es read-only y no queremos que una
  query larga bloquee el pool.
- **Tests con BD Postgres real** requieren fixture `docker compose up db`
  en CI. Añadirlo al workflow.

**Estimación**: **7-8 días**. Sprint más grande.

**Dependencias**: Sprint 0 (naming), Sprint 1 (engine_nexo ya existe).

---

## Sprint 3 — Consultas pesadas: preflight + postflight

**Objetivo**: antes de ejecutar una operación cara (pipeline OEE, capacidad
de 180 días, exploración BBDD con query grande, export CSV masivo), estimar
coste y bloquear/avisar. Después, registrar métricas para ajustar.

**Entregable verificable**:
- Tabla `nexo.query_log`: `(id, ts, user_id, endpoint, params_json,
  estimated_ms, actual_ms, rows, status)`.
- Tabla `nexo.query_thresholds`: `(endpoint, warn_ms, block_ms,
  updated_at, updated_by)`. Editable desde UI en **Ajustes**.
- `nexo/services/preflight.py` con función
  `estimate_cost(endpoint, params) -> Estimation`. Para pipeline:
  estimación por `n_recursos × n_días × factor_por_modulo`; para queries:
  `EXPLAIN` o fallback a `SET STATISTICS IO ON` no — demasiado frágil.
  Mejor **heurística simple + aprendizaje del postflight**.
- Flujo UI:
  - Endpoint devuelve `Estimation(ms, level, reason)` antes de ejecutar.
  - `level="green"`: ejecuta sin aviso.
  - `level="amber"`: muestra toast "esto tardará ~X min, ¿continuar?"
    (sólo aviso, no bloqueo).
  - `level="red"`: no ejecuta. Botón "Solicitar aprobación admin". Admin
    ve la solicitud en `/ajustes/solicitudes`, aprueba, el usuario re-dispara.
- `nexo/middleware/query_timing.py` mide `time.monotonic()` antes/después
  y escribe en `query_log`. Si `actual > warn_ms * 1.5`, emite alerta en
  `logging.WARNING` + notificación en panel admin.
- Sin bloqueo duro en ningún endpoint salvo `/login` (ya rate-limited
  en Sprint 1).
- Umbrales editables en `/ajustes/limites`.

**Archivos/módulos tocados**:
- `api/routers/pipeline.py`: endpoint `POST /preflight` que devuelve la
  Estimation; endpoint `POST /run` acepta `force=true` si venía de approval.
- `api/routers/bbdd.py`: endpoint `POST /query` mete el preflight encima
  de la validación SELECT-only.
- `api/routers/capacidad.py`, `operarios.py` — preflight si el rango > 90 d.
- `nexo/services/preflight.py`, `nexo/services/query_timing.py`,
  `nexo/services/approvals.py` (flujo amber/red).
- `templates/ajustes_limites.html`, `templates/ajustes_solicitudes.html`.
- `static/js/app.js` — mostrar modal amber/red y desbloqueo tras approval.

**Riesgos específicos**:
- La estimación para pipeline es difícil sin datos históricos. Primer mes
  postflight "aprende", después refinamos factores. Documentar que las
  primeras semanas los umbrales son orientativos.
- La aprobación asíncrona (user solicita, admin aprueba, user re-dispara)
  es UX delicado. Probar con un admin y un analyst reales antes de cerrar
  el sprint.
- El middleware de timing **se suma** al de audit y al de auth. Orden
  importa: auth → audit → timing. Cuidado con el tamaño del request body
  si se captura dos veces.
- Pipeline con matplotlib **bloquea el worker**. Aunque el preflight
  diga "ámbar", si aceptas, el resto de la UI se congela hasta que termine.
  Considerar meter el pipeline en `asyncio.to_thread` aquí, no en Sprint 5.

**Estimación**: **4 días**.

**Dependencias**: Sprint 1 (auth para saber el usuario, audit para
`query_log`). Sprint 2 (repositorios expuestos para timing).

---

## Sprint 4 — UI por roles

**Objetivo**: la sidebar y las páginas muestran sólo lo que el rol del
usuario puede ver. Los endpoints ya estaban protegidos en Sprint 1; aquí
se adapta la UI para no mostrar botones inútiles.

**Entregable verificable**:
- El contexto `request.state.user` está disponible en cualquier template
  (dependency en `pages.py` o middleware que lo inyecta en
  `Jinja2Templates` globals).
- `base.html` filtra `nav_items` según permisos del user. El admin ve
  todo; un viewer ve Centro Mando, Historial, Capacidad.
- 3 roles iniciales con UI verificada:
  - **admin**: ve todo, incluyendo `/ajustes/auditoria`,
    `/ajustes/usuarios`, `/ajustes/limites`, `/bbdd`.
  - **analyst**: ve pipeline, historial, capacidad, recursos, ciclos,
    operarios, datos. No ve `/ajustes/*` ni `/bbdd`.
  - **viewer**: ve centro_mando, historial, capacidad. Sin edición.
- `static/js/app.js` — los botones "Ejecutar pipeline", "Borrar ejecución",
  "Sincronizar recursos" comprueban permiso antes de mostrarse.
- Reorganizar `templates/ajustes.html`: splitearlo en `ajustes_conexion.html`,
  `ajustes_smtp.html`, `ajustes_usuarios.html`, `ajustes_auditoria.html`,
  `ajustes_limites.html`, `ajustes_solicitudes.html`, y un `ajustes.html`
  hub que los enlaza.
- Test E2E manual: login como admin, analyst, viewer; capturar pantalla
  de cada sidebar.

**Archivos/módulos tocados**:
- `templates/base.html` (filtro de nav_items), todos los templates del
  sidebar.
- `api/routers/pages.py` — inyecta user en ctx.
- `api/deps.py` — expone `current_user()` dependency global.
- Nuevos templates de ajustes.
- Revisión de `static/js/app.js` para eliminar el `renderOeeDashboard`
  con innerHTML y **pasarlo a Alpine** si queda tiempo. (Opcional, puede
  quedar para Mark-IV).

**Riesgos específicos**:
- Jinja2 + Alpine mezclan condicionales en dos niveles (servidor y cliente).
  Decidir: **la visibilidad la decide Jinja en el servidor**, Alpine sólo
  interactividad. Eso obliga a un full reload tras login para que la UI
  se ajuste al rol — aceptable, es LAN.
- Los 331 líneas del partial `_partials/mapa_pabellon.html` podrían
  necesitar que ciertos controles (edición de zonas) dependan de rol.
- No hay tests E2E automatizados hoy. En Sprint 6 (DevEx) podemos meter
  Playwright; aquí es visual/manual.

**Estimación**: **4 días**.

**Dependencias**: Sprint 1 (roles), Sprint 3 (las páginas de ajustes
existen).

---

## Sprint 5 — Despliegue LAN con HTTPS

**Objetivo**: montar Nexo en el servidor LAN (i5 7ª gen, 16 GB, SSD 1TB,
Ubuntu Server 24.04) con `nexo.ecsmobility.com` resuelto por DNS interno,
HTTPS con Let's Encrypt vía DNS-01 (si controlamos el DNS público del
dominio), sin exposición a internet.

**Entregable verificable**:
- `docker-compose.prod.yml` (o `profiles:` en el actual) que:
  - Cambia `caddy/Caddyfile` para usar `nexo.ecsmobility.com` con
    Let's Encrypt DNS-01 (proveedor depende de quién hostea el DNS del
    dominio).
  - Cierra el puerto Postgres al host (no expone 5432).
  - Quita el servicio `mcp` del perfil prod (si se decide así en OPEN_QUESTIONS).
  - Añade `healthcheck` a web y caddy.
  - `restart: unless-stopped` consistente.
- Script `scripts/deploy.sh` que hace `git pull && docker compose
  --profile prod build --pull && docker compose --profile prod up -d`.
- Documentación `docs/DEPLOY_LAN.md` con:
  - Instalación de Docker + plugin compose en Ubuntu Server 24.04.
  - Configuración DNS interno (A record `nexo.ecsmobility.com` →
    IP LAN del servidor).
  - Registros necesarios en el DNS público del dominio para DNS-01 (el
    reto se publica, el cert se obtiene, el dominio **no resuelve
    públicamente a la IP LAN** — sólo existe en el DNS interno).
  - Rotación de password SQL y SMTP documentada.
  - Backup de `pgdata` (volumen Docker) y de `ecs_mobility` (SQL Server
    está fuera del servidor Nexo, ya tiene su backup).
  - Plan de recuperación básico: reinstalar compose + `git clone` + `cp
    .env.prod .env` + `deploy.sh`.
- Firewall Ubuntu (`ufw`) configurado: permite 22 (SSH desde red interna),
  443 (HTTPS), 80 (redirect a 443 opcional). Deny all else.
- Verificación desde otro equipo LAN: `https://nexo.ecsmobility.com` carga
  con cert válido.

**Archivos/módulos tocados**:
- `caddy/Caddyfile` (nuevo bloque producción con `tls email@dominio
  dns_provider_config`).
- `docker-compose.yml` con profiles o un `docker-compose.prod.yml`.
- `scripts/deploy.sh` (nuevo).
- `docs/DEPLOY_LAN.md` (nuevo).
- `.env.prod.example` (claves de producción sin valores).

**Riesgos específicos**:
- **DNS-01 requiere controlar el DNS público** de `ecsmobility.com`. Si
  no lo controla ECS directamente (está en un registrador externo), o
  no hay API del registrador, hay que caer a **cert interno/autofirmado**
  que es lo que Caddy hace hoy. Confirmarlo antes del sprint.
- El certificado de la máquina actual es `tls internal` (autofirmado).
  Cambiar a Let's Encrypt mid-sprint es seguro; volver atrás también.
- Postgres publicado al host hoy (5432) es un riesgo menor en LAN, pero
  al cerrarlo perdemos `make db-shell` desde el host. Alternativa: usar
  `docker compose exec db psql` (ya lo hace `make db-shell`).
- El i5 de 7ª con 16 GB es **justo** para tener web+db+caddy+matplotlib
  ejecutando pipelines grandes en paralelo. Monitorizar RAM en las
  primeras semanas.
- El dominio `.ecsmobility.com` requiere que el WHOIS lo valide como tuyo
  antes de emitir cert — comprobar ACME client puede hacerlo.

**Estimación**: **3 días**.

**Dependencias**: Sprint 0 (nombres estables para el Caddyfile), Sprint 1
(para que no desplegemos en producción una app sin auth), Sprint 3 (para
que el preflight esté en producción).

---

## Sprint 6 — DevEx hardening

**Objetivo**: cualquier dev (tú + yo + un Claude futuro) puede arrancar,
cambiar y desplegar Nexo sin fricción.

**Entregable verificable**:
- Pre-commit con ruff, black, mypy ligero (solo `api/` y `nexo/`, OEE queda
  fuera).
- CI ampliado: matriz Python 3.11 y 3.12, pytest + cobertura min 60% en
  `api/` y `nexo/`, smoke test que levanta docker compose y pega a `/api/health`.
- `make` targets nuevos: `make test`, `make lint`, `make format`,
  `make migrate`, `make backup`.
- `docs/ARCHITECTURE.md` con diagrama de componentes (web ↔ engine_nexo ↔
  Postgres, web ↔ engine_app ↔ ecs_mobility, web ↔ engine_mes ↔ dbizaro).
- `docs/RUNBOOK.md` con "qué hacer cuando X falla" para los 5 escenarios
  más probables (MES caído, Postgres no arranca, certificado expira,
  pipeline atascado, lockout admin).
- `docs/RELEASE.md` con checklist de release (tag, deploy, smoke).
- `CLAUDE.md` actualizado con cambios del sprint completo.

**Archivos/módulos tocados**:
- `.pre-commit-config.yaml`, `.github/workflows/ci.yml` (ampliado),
  `Makefile`, `docs/*.md`.

**Riesgos específicos**:
- Pre-commit sobre el código existente va a reformatear mucho. **Correr
  black + ruff --fix sobre main primero, en un commit único y aislado**,
  para que los diffs futuros sean señal y no ruido.
- Mypy sobre código con `pd.read_sql` + `pyodbc` es ruidoso. Configurarlo
  en modo "sólo errores críticos" (`--warn-unused-ignores`, `--no-strict`).
- Cobertura del 60% requiere escribir tests de routers que no tuvimos
  tiempo de escribir en Sprint 2. Ajustar al alza o a la baja según
  tiempo real.

**Estimación**: **3 días**.

**Dependencias**: todos los anteriores estables (no introduce código
nuevo, sólo herramientas).

---

## Totales y orden

| Sprint | Nombre | Días focalizados |
|---|---|---:|
| 0 | Naming + higiene | 2 |
| 1 | Identidad: auth + RBAC + audit | 6 |
| 2 | Datos: repositorios + schema_guard | 7-8 |
| 3 | Consultas pesadas | 4 |
| 4 | UI por roles | 4 |
| 5 | Despliegue LAN HTTPS | 3 |
| 6 | DevEx hardening | 3 |
| | **Total** | **29-30 días focalizados** |

Calendario realista (asumiendo 3 días focalizados por semana entre reuniones
y otras tareas): **10-12 semanas naturales**, algo más si hay imprevistos.

---

## Qué queda EXPLÍCITAMENTE fuera (recordatorio)

- Ingesta realtime vía OPC-UA/MQTT.
- Módulos nuevos (calidad, trazabilidad).
- Dashboards en streaming.
- Exposición a internet.
- Refactor de los 4 módulos OEE (`disponibilidad/rendimiento/calidad/oee_secciones`).
- Sustituir matplotlib por WeasyPrint/ReportLab/Playwright.
- CRUD de `plantillas.html` enganchado al pipeline.
- Microservicios.

Todo eso queda para Mark-IV o posterior.
