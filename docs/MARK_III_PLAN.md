# Nexo · Mark-III — Plan de sprints

Rediseño del monolito OEE actual a **Nexo**, plataforma interna de ECS que
integra OEE, calidad, trazabilidad y futuros módulos. Mark-III se limita a:
auth + roles + RBAC, auditoría, refactor de capa de datos, sistema de
consultas pesadas, refactor UI por roles y despliegue LAN con HTTPS.

Estrategia: **strangler fig** en el mismo repo, historial conservado.
Stack sin cambios (FastAPI + Jinja2 + Alpine + Tailwind + Postgres 16 +
SQL Server via pyodbc + Docker Compose).

Las estimaciones son **días de trabajo focalizado**, no días de calendario.

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

**Objetivo**: repo arrancable con marca "Nexo", higiene básica (secrets,
Zone.Identifier, requirements pineados, CI mínimo), y un CLAUDE.md que le
explique a cualquier dev futuro cómo navegar el código.

**Entregable verificable**:
- Repo compila con `make build && make up` y `/api/health` responde OK.
- Título FastAPI dice "Nexo", sidebar dice "NEXO", README renombrado.
- `.env.example` completo (incluye SMTP y nueva nomenclatura `NEXO_*`).
- Password SQL rotada, `.env:Zone.Identifier` eliminado del repo.
- `CI_basic.yml` en GitHub Actions que corre pytest + ruff en cada push.
- `CLAUDE.md` en raíz con mapa del repo para IAs / devs nuevos.

**Archivos/módulos tocados**:
- `api/main.py` (title), `api/config.py` (prefijo env `NEXO_` + compat
  `OEE_`), `README.md`, `docs/` (ya arranca esta rama).
- `templates/base.html` (marca, título, favicon), `static/css/app.css`,
  `static/js/app.js` (strings).
- `.env.example` (añadir SMTP, cambiar prefijos), `.gitignore` (añadir
  `*.Zone.Identifier`).
- `requirements.txt` (pinear versiones exactas: `fastapi==X.Y.Z`, etc.),
  añadir `httpx`, `pytest`, `ruff`.
- `.github/workflows/ci.yml` (nuevo).
- Borrar `.env:Zone.Identifier` del repo, `data/oee.db`, `test_email.py`,
  `server.py` (redundante con `CMD` del Dockerfile y `make dev`).
- `CLAUDE.md` (nuevo).

**Riesgos específicos**:
- Cambiar prefijo `OEE_*` a `NEXO_*` rompe el `.env` actual que está en
  disco en el servidor LAN si ya hay uno. **Mitigación**: `api/config.py`
  acepta ambos prefijos durante Mark-III (lee `NEXO_DB_SERVER` y cae a
  `OEE_DB_SERVER` si el primero está vacío). Eliminamos la compat en
  Mark-IV.
- El prefijo `OEE_` aparece en el `docker-compose.yml` también. Actualizar
  allí al mismo tiempo.
- Renombrar el servicio MCP (ID `oee-planta` → `nexo`) invalida cualquier
  config `.claude.json` que los usuarios tengan apuntando al ID viejo.
  Documentarlo en el README.

**Estimación**: **2 días**.

**Dependencias**: ninguna.

---

## Sprint 1 — Identidad: auth + RBAC + audit

**Objetivo**: toda request con `/api/*` o HTML autenticada; roles con
permisos granulares; cada acción registrada en una tabla append-only.

**Entregable verificable**:
- Página `/login` funcional con bloqueo progresivo: 3 intentos → 30s, 5 →
  5 min, 10 → lockout manual. Contador por `(user, IP)` para no bloquear
  a un usuario desde todas partes por culpa de una IP concreta.
- Tabla `nexo.users` en Postgres (hash argon2, rol, activo, last_login).
- Tabla `nexo.roles` y `nexo.permissions` (RBAC simple: rol = conjunto de
  permisos, permiso = `modulo:accion`, ejemplo `oee:read`, `admin:write`).
- Middleware FastAPI que redirige HTML a `/login` y devuelve 401 JSON para
  `/api/*` no autenticadas.
- Dependency `Depends(require_permission("oee:read"))` aplicada a cada
  router.
- Tabla `nexo.audit_log` append-only: `(id, ts, user_id, ip, method, path,
  status, details_json)`. **Append-only** a nivel de BBDD: `REVOKE UPDATE,
  DELETE` del rol app sobre esa tabla; sólo un rol admin puede purgar.
- Middleware que registra **cada request autenticada**, incluido admin.
- Panel `/ajustes/auditoria` con filtros (user, fecha, path, status) +
  export CSV.
- `/ajustes/usuarios` CRUD de usuarios y roles (sólo rol admin).

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
- El `global_exception_handler` actual devuelve traceback HTML. Si lo
  dejas y alguien fuerza un 500 en `/login`, filtras secretos. **Hay que
  reescribirlo en este sprint, no en Sprint 0**.
- El middleware de audit captura el body de POST/PUT. Esos bodies pueden
  contener credenciales SQL (endpoint `/api/conexion/config`). Sanitizar
  en el servicio de audit: whitelist de campos a grabar, o `details_json`
  vacío para endpoints marcados como "sensibles".
- La política de bloqueo progresivo requiere tabla `nexo.login_attempts`
  con limpieza periódica (cronjob o purga al login exitoso).
- El login también tiene que estar rate-limited (excepción al "no rate
  limit" que me diste), porque sin eso el bloqueo es por usuario pero un
  atacante puede probar 10 K usuarios en paralelo.
- **Retro-fit de permisos** en los 14 routers es un pasada mecánica larga.
  Propón 2-3 roles base iniciales (admin, analyst, viewer) antes de
  empezar; sin eso te pierdes en granularidad.

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
