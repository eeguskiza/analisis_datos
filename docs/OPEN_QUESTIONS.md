# Nexo · Mark-III — Preguntas abiertas

Respuestas extraídas del código actual (rama `feature/Mark-II`, commit
`f07e80e`). Cuando no se puede deducir del repo, se indica explícitamente.

Las decisiones **🚦 BLOQUEANTES** hay que tomarlas antes de arrancar Sprint 0
para no pintar planes sobre suposiciones. Las demás se pueden resolver durante
el sprint correspondiente.

---

## SQL Server y BBDD

### 1. ¿La conexión a SQL Server parametriza la BD o está hardcoded a `dbizaro`?

**Sí, parametrizada — pero a medias.**

- El "catalog" viaja dentro del dict `cfg` en `OEE/db/connector.py`
  (`_build_connection_string` usa `cfg["database"]`) y lo fija por request
  el llamante. `services/db.get_config()` inyecta `izaro_db` (por defecto
  `dbizaro`) desde `.env` si el JSON está vacío.
- `routers/bbdd.py._get_conn_string(database=...)` permite pasar el catalog
  explícito por llamada — se usa para listar BBDD y explorar tablas.
- El `engine` SQLAlchemy (`api/database.py`) está atado a `ecs_mobility`
  vía el creator pyodbc. **Para leer `cfg/oee/luk4` siempre apunta a
  `ecs_mobility`.**
- Hay queries con **prefijo de 3 partes**: `dbizaro.admuser.fmesmic` en
  `routers/centro_mando.py:51,55`. Eso implica asumir cross-database en la
  misma instancia. Si algún día dbizaro vive en otra instancia, esto rompe.

**Para soportar `ecs_mobility` e `izaro_db` en paralelo limpiamente hay que:**
- Separar dos engines: `engine_app` (ecs_mobility, pool SQLAlchemy) y
  `engine_mes` (dbizaro, read-only, pool dedicado).
- Quitar los prefijos de 3 partes (`dbizaro.admuser.fmesmic` → `admuser.fmesmic`
  ejecutado contra `engine_mes`).
- Unificar el resto de pyodbc sueltos (`bbdd.py`, `capacidad.py`,
  `operarios.py`, `centro_mando.py`) para que pasen por el repositorio
  correspondiente.

### 2. ¿`dbizaro` y `ecs_mobility` están en la misma instancia o en instancias separadas?

**Misma instancia — confirmado por el código, no por env separada.**

Argumentos:
- `routers/centro_mando.py` usa nombres de 3 partes (`dbizaro.admuser.fmesmic`)
  desde el engine conectado a `ecs_mobility`. Cross-database references en
  SQL Server **sólo funcionan dentro de la misma instancia** (hay linked
  servers, pero no está configurado).
- `.env` y `api/config.py` tienen un único bloque `OEE_DB_SERVER/PORT`
  (`192.168.0.4:1433`) con dos BDs distintas (`OEE_DB_NAME=ecs_mobility`
  y `OEE_IZARO_DB=dbizaro`). **No hay variables separadas para server/port
  de IZARO**.

Variables que lo definen:
- `OEE_DB_SERVER`, `OEE_DB_PORT`, `OEE_DB_USER`, `OEE_DB_PASSWORD` — instancia.
- `OEE_DB_NAME` — BD propia (ecs_mobility).
- `OEE_IZARO_DB` — BD externa (dbizaro).

🚦 **Decisión bloqueante:** ¿Mark-III mantiene la asunción "misma instancia"
o abre la puerta a instancias separadas desde el principio? Si sí, hay que
duplicar el bloque de env vars (`NEXO_MES_SERVER/PORT/USER/PASSWORD` vs
`NEXO_APP_SERVER/PORT/USER/PASSWORD`) y quitar los prefijos de 3 partes.

### 3. ¿Qué driver ODBC se usa hoy, qué versión, cómo se instala?

- **Driver preferido**: `ODBC Driver 18 for SQL Server` (el
  `OEE/db/connector.detectar_driver` lo elige por defecto).
- Fallbacks: Driver 17, 13, SQL Server Native Client 11.0, genérico.
- Paquete: `msodbcsql18` del repo oficial de Microsoft.
- Instalación:
  - **En Docker**: `Dockerfile` añade el repo de packages.microsoft.com
    (auto-detecta arch amd64/arm64) y hace `ACCEPT_EULA=Y apt-get install
    msodbcsql18`.
  - **En host Ubuntu**: `install_odbc.sh` hace lo mismo para el usuario,
    pero con repo `packages.microsoft.com/ubuntu/...`.

Observación: `install_odbc.sh` asume Ubuntu y `arch=amd64`, el Dockerfile
detecta arch. Si el despliegue LAN es Ubuntu 24.04 x86_64, ambos están
bien; si el servidor futuro es ARM, sólo sirve el Dockerfile.

---

## Módulo OEE

### 4. Valoración de acoplamiento OEE ↔ datos: **4 / 10**

- **Lo que separa**: el cálculo "puro" (`convertir_raw_a_metricas`,
  `clasificar_incidencia`, `determinar_turno`, etc.) vive en
  `OEE/oee_secciones/main.py` y es **testeable sin BD** (los 30 tests
  unitarios lo cubren).
- **Lo que pega**: los 4 módulos (`disponibilidad`, `rendimiento`,
  `calidad`, `oee_secciones`) reciben `data_dir` y `output_dir` y **leen
  CSVs desde disco con `listar_csv_por_seccion`**. También leen `ciclos.csv`
  desde disco. No aceptan un repositorio ni un iterador de filas; están
  acoplados al filesystem.
- **El pipeline** (`api/services/pipeline.py`) hace de "adaptador":
  extrae de IZARO → guarda en `oee.datos` → escribe CSVs temporales →
  llama a los módulos OEE → mueve PDFs → guarda metricas en `oee.metricas`.
  Es el único pegamento entre BD y módulos.

Por eso la nota es 4: el núcleo matemático está limpio, pero el contrato
de los módulos es "dame ficheros", no "dame datos". Sprint 1 (repositorios)
tiene que decidir si:
- (a) los módulos OEE aceptan ahora un iterable de filas (refactor grande,
  4.7 K líneas tocadas), o
- (b) el pipeline mantiene el adaptador "BD → CSV temporal → módulo OEE"
  tal como está hoy, y sólo cambiamos el origen de los datos.

Mi recomendación: **(b)** en Mark-III, dejar (a) para Mark-IV. El módulo
OEE funciona, no merece la pena tocarlo sólo por ortodoxia.

### 5. ¿Qué se hace en Python vs en SQL? ¿Se puede mover más a SQL?

- **En SQL** (`_SQL_TEMPLATE` de `OEE/db/connector.py`): joins entre
  `fmesdtc/fmesddf/fmesinc/fprolof`, agregación de malas/recuperadas
  por lote, decodificación del campo `dt110` a proceso, filtro por rango
  de fechas con lógica especial para T3 de la última fecha (trae
  `<06:00` del día siguiente).
- **En Python** (módulos OEE y `services/metrics.py`): cálculo de
  solapamientos con turnos, clasificación de incidencias (regex),
  determinación de turno por hora, agregación por máquina/turno/día,
  cálculo de disponibilidad/rendimiento/calidad/OEE, generación de gráficas
  con matplotlib.

¿Se puede mover más a SQL? **Poco valor**:
- El solapamiento turno↔registro lo puede hacer SQL con un CTE, pero los
  turnos de cambio (T3 cruza medianoche) añaden ramas que hoy ya están
  resueltas en Python y bien testeadas.
- La clasificación de incidencias usa regex con acentos — SQL Server
  tiene `LIKE` pero para esto Python es más claro y fácil de cambiar.

Lo que **sí conviene mover**: nada. El balance actual es razonable.
Lo que **conviene quitar de Python**: el `pd.read_sql` del pipeline que
trae millones de filas a un DataFrame; usar cursor iterativo directo.

---

## UI

### 6. Vistas con menú + contenido, y plantilla base

- Todas las páginas actuales extienden `templates/base.html`. **Sí hay
  plantilla base común**, no se reimplementa.
- `base.html` define sidebar + topbar + main + toast system. El sidebar
  usa `nav_items` hardcodeado como lista literal en Jinja (9 entradas).
- Cada página define `{% block title %}`, `{% block page_title %}` y
  `{% block content %}`. Algunas redefinen `{% block topbar %}` (luk4, por
  ejemplo, la acorta).
- Partial reutilizable: `templates/_partials/mapa_pabellon.html` — 331
  líneas que se incluyen 4 veces en `luk4.html` (P5, P4, P3, P2).

Vistas que hoy renderizan contenido real (11):
`luk4` (Centro Mando, ruta `/`), `pipeline` (Análisis), `historial`, `capacidad`,
`recursos`, `ciclos_calc` (Calcular Ciclos), `operarios`, `datos`, `bbdd`,
`ajustes`. Y la `datos` redirige a `historial`.

Templates sin ruta directa: `ciclos.html` (CRUD ciclos, accesible sólo
escribiendo URL), `informes.html` (sustituida por redirect), `plantillas.html`
(experimental).

### 7. Alpine vs JS plano

- **Alpine** es el estándar en todos los templates: formularios, modales,
  wizards, toggles, tabs, tooltips. `recursos.html` y `ciclos_calc.html`
  llevan la carga más densa.
- **JS plano en `static/js/app.js`** (421 líneas) hace tres cosas que no
  pasan por Alpine:
  1. Handler global del badge de conexión (HTMX `afterRequest`).
  2. `runPipeline` (fetch + streaming SSE + parser de mensajes DONE/ERROR).
  3. **`renderOeeDashboard`** — dibuja el panel post-ejecución con Chart.js
     generando `innerHTML` a mano. Esto **duplica** patrones que se podrían
     haber hecho con un `x-data` Alpine, pero no es exactamente el mismo
     código (no hay "dos renders del mismo dashboard"), simplemente es el
     único lugar donde se usa JS plano para construir DOM.
- Además, hay **un `showToast` duplicado**: `base.html` lo implementa con
  Alpine (`toastSystem()`), `app.js` lo implementa con DOM append. Ambos
  conviven.

Duplicación real: `SECTION_MAP` en `app.js` duplica el de `database.py` y
`connector.py`.

---

## Configuración y secretos

### 8. Secrets en `.env` (claves, sin valores)

Claves presentes en `.env` real:
`OEE_HOST, OEE_PORT, OEE_DEBUG, OEE_DB_SERVER, OEE_DB_PORT, OEE_DB_NAME,
OEE_DB_USER, OEE_DB_PASSWORD, OEE_IZARO_DB, OEE_PG_USER, OEE_PG_PASSWORD,
OEE_PG_DB`.

Observaciones:
- `OEE_DB_PASSWORD` contiene la credencial real del usuario `sa` en texto
  plano en disco. `.env` está en `.gitignore` (bien) — pero conviene
  **rotarla en Sprint 0** dando por hecho que el fichero pudo ser visto
  por cualquiera que haya tenido acceso al equipo.
- `OEE_PG_*` no se usa (Postgres no está enganchado al código). Si Mark-III
  hace que Postgres sea la "casa de Nexo", estas sí importan.
- **Faltan en `.env`**: SMTP (server, port, email, password, from). Hoy la
  única forma de configurar SMTP es editar `data/db_config.json` a mano, y
  ese JSON actual **no tiene sección `smtp`** → el email está inoperativo.

¿`.env` protegido basta para Mark-III? **Sí, con matices**:
- Para LAN-only con servidor dedicado, `.env` con permisos 600 es aceptable.
- Añadir rotación trimestral documentada.
- Mover secretos a **Docker secrets** o **SOPS + age** sólo si se publica
  fuera de LAN, lo cual Mark-III explícitamente no hace.
- 🚦 Decisión: en Sprint 0 creo que basta con (a) rotar la password SQL,
  (b) ajustar permisos del `.env` a 600, (c) revisar que `.env:Zone.Identifier`
  sea removido del repo, y (d) añadir SMTP al `.env.example`.

### 9. Config UI: ¿`data/db_config.json` a BBDD o se queda como archivo?

Hoy `db_config.json` contiene:
- Credenciales de conexión a IZARO (duplicadas con `.env`).
- `uf_code`, `referencia_campo` — filtros de consulta.
- `recursos[]` — 67 entradas sincronizadas bidireccionalmente con
  `cfg.recursos` (via `api/routers/recursos._sync_to_config`).
- `smtp` (ausente hoy, esperado por `services/email`).

Recomendación para Mark-III:
- **Credenciales SQL**: a `.env` únicamente. Quitar del JSON.
- **`recursos[]`**: eliminar del JSON, fuente única = `cfg.recursos` en BD.
  `OEE/db/connector.extraer_datos` necesita `cfg["recursos"]` — adaptarlo
  para que lo reciba por parámetro, o consultarlo en BD.
- **`uf_code`, `referencia_campo`**: mover a tabla `cfg.ajustes` en Postgres
  local (nueva casa de Nexo) o dejarlas en `.env` como variables.
- **`smtp`**: mover a `.env` (server, port, from, user, password).

Resultado final: **`data/db_config.json` desaparece** en Mark-III.

---

## MCP

### 10. ¿Qué hay en `mcp/`, se expone en producción, entra en Mark-III?

- **Qué hay**: un servidor MCP (`mcp/server.py`, 272 líneas) que expone a
  Claude Code 15 tools read-only, todas wrappers HTTP sobre la API FastAPI
  (`get_health`, `get_connection_status`, `bizaro_list_databases`,
  `bizaro_list_tables`, `bizaro_columns`, `bizaro_preview`, `bizaro_query`
  — con validación SELECT-only —, `bizaro_buscar_rutas`, `get_capacidad`,
  `get_ciclos`, `get_recursos`, `list_reports`, `list_executions`,
  `get_execution_detail`).
- **Está activo**: el `docker-compose.yml` lo arranca siempre (servicio
  `mcp`, sin puertos expuestos, sólo stdio en el container). Se conecta a
  `web:8000` por variable `OEE_API_URL`.
- **En compose de producción**: sí, pero sin puerto → en el servidor LAN
  corre "ocioso" hasta que alguien haga `docker exec -i ... python server.py`.
  Consume memoria mínima pero está ahí.
- **Entra en Mark-III**: recomiendo **aparcar** el servicio en compose para
  producción (no aporta a la LAN, sólo es útil para desarrollo local) y
  mantener el código en el repo para que cada dev lo arranque local. Es
  una decisión pequeña; si quieres conservarlo en prod, no rompe nada pero
  tampoco ayuda.

---

## Tests y CI

### 11. ¿Pasan los tests? ¿Cobertura? ¿GitHub Actions?

- **Pasan o no**: **no se puede determinar del repo en este entorno** —
  `pytest` no está instalado en el host donde corre esta auditoría y no
  hay acceso a pip. El código de los tests es puro (dict → dict), y por
  inspección visual no detecto errores evidentes. La funcionalidad que
  testean (`convertir_raw_a_metricas`, `determinar_turno`, etc.) existe
  en `OEE/oee_secciones/main.py` con las firmas esperadas.
- **Cobertura aproximada**: muy baja. Sólo se cubren ~20 funciones del
  módulo `oee_secciones`, y ninguna de las 80+ funciones del resto del
  código (routers, pipeline, conector, services). Estimación: **<15% real**,
  aunque dentro del núcleo de cálculo OEE la cobertura sí es buena.
- **GitHub Actions**: **no existe**. No hay `.github/` ni ficheros `.yml`
  de CI. Tampoco GitLab CI, Jenkinsfile, etc.

🚦 **Decisión para Mark-III**: ¿añadimos un CI mínimo (GitHub Actions con
pytest + ruff) desde Sprint 0, o lo dejamos para Sprint 7 (DevEx hardening)?
Mi recomendación: **Sprint 0**, aunque sea un único workflow que instale
deps y corra pytest. El coste es una hora, el retorno es que cada PR a
Mark-III se valida automáticamente.

---

## PDFs

### 12. ¿Cómo se generan los PDFs? ¿Cuello de botella?

- **Librería**: `matplotlib` + `matplotlib.backends.backend_pdf.PdfPages`.
  Confirmado en `OEE/disponibilidad/main.py`, `OEE/rendimiento/main.py`,
  `OEE/calidad/main.py`, `OEE/oee_secciones/main.py`. No se usa ReportLab,
  ni WeasyPrint, ni Playwright.
- **Tamaños**: `OEE/informes/2025-12-04/LINEAS/LINEAS_oee_seccion.pdf`
  y similares — decenas de KB por recurso/día/módulo. Un informe completo
  (4 módulos × N recursos) puede ser del orden de 20-50 PDFs.
- **Bloqueante**: sí, síncrono, corre en el proceso del worker uvicorn.
  `MPLBACKEND=Agg` (en Dockerfile) evita el servidor X, pero la generación
  sigue siendo **CPU-bound** y **mantiene el worker ocupado minutos**.

**Para el sistema de consultas pesadas (Sprint 4)**:
- **Preflight**: multiplicar `n_recursos × n_días × factor_fijo_por_módulo`
  da una estimación razonable. Sin datos históricos, empezar con factor
  heurístico `~10s/recurso/día/módulo` y refinar.
- **Postflight**: medir `time.monotonic()` antes/después de cada módulo y
  guardar en `nexo.query_log`. Si el promedio reciente supera umbral,
  alerta.
- **Bloqueo real**: un informe grande ocupa el worker entero y congela el
  resto de endpoints si `workers=1`. Considerar `workers=2` o meter los
  PDFs en un `asyncio.to_thread` o un pool separado **como parte de
  Sprint 4**, no antes.

El cuello de botella **no es SQL**, es **matplotlib**. Por eso el preflight
no debe medir coste en SQL Server sino en **tiempo de ejecución estimado
del pipeline**.

---

## Resumen de decisiones bloqueantes antes de Sprint 0

1. 🚦 **Separación de instancias SQL Server**: ¿seguimos asumiendo "misma
   instancia para dbizaro y ecs_mobility" (más simple, coincide con hoy),
   o abrimos dos bloques de env vars ahora para no duplicar esfuerzo en
   Mark-IV?
2. 🚦 **Rotación de password SQL + limpieza del `.env:Zone.Identifier`
   trackeado**: confirmar que se hace en Sprint 0 antes de empezar
   cualquier refactor.
3. 🚦 **CI mínimo (pytest + ruff) desde Sprint 0 o lo dejamos para Sprint 7**.
4. 🚦 **MCP en compose de producción**: lo dejamos o lo aparcamos.
5. 🚦 **Postgres de Nexo = "casa" para users/audit/cache**: confirmado en tus
   reglas iniciales. Pregunta derivada: ¿`cfg.recursos`, `cfg.ciclos`,
   `cfg.contactos` se quedan en `ecs_mobility.cfg` (donde están hoy) o
   se mueven a Postgres? Mi lectura es que **se quedan**: son datos de
   negocio, no estado interno de la app. Pero confirmar explícitamente.

Cualquier "sí/no" sobre estas 5 cosas me vale para cerrar el plan.
