# Nexo · Mark-III — Auditoría del estado actual

Foto fija del repo `analisis_datos` (rama `feature/Mark-II`, commit `f07e80e`,
2026-04-18). Es el punto de partida para el rediseño **Nexo / Mark-III**.

No propone cambios. Describe qué hay, cómo está acoplado y dónde duele.

---

## 1. Inventario de código

### 1.1 Árbol resumido

```
analisis_datos/
├── server.py              # entry-point (uvicorn wrapper, 13 líneas)
├── test_email.py          # script one-shot de prueba SMTP, fuera de tests/
├── Dockerfile             # python:3.11-slim-bookworm + ODBC Driver 18
├── docker-compose.yml     # web + db(postgres) + caddy + mcp
├── Makefile               # up/down/build/rebuild/logs/status/dev/clean
├── install_odbc.sh        # instala msodbcsql18 en Ubuntu
├── requirements.txt       # 8 deps, todas con >= sin pin duro
├── .env / .env.example    # credenciales SQL Server + PG + web
├── ERRORES.md             # registro manual de bugs conocidos (LUK4)
├── README.md              # doc de uso diario, obsoleta en varios puntos
│
├── api/                   # FastAPI (backend) ── ver 1.2
├── OEE/                   # módulos de cálculo OEE + conector IZARO
├── templates/             # Jinja2 (15 páginas + 1 partial)
├── static/                # app.css (103) + app.js (421) + 5 imágenes
├── scripts/               # create_oee_db / migrate_schemas / extract_2025
├── mcp/                   # servidor MCP para Claude Code, en compose
├── caddy/                 # Caddyfile (HTTPS LAN, cert autofirmado)
├── data/                  # db_config.json + CSVs + logo + SQLite residual
├── docs/                  # ECS-Pabellon5.pdf + pabellon5_L4_v2.html
├── informes/              # PDFs generados (gitignored)
└── tests/                 # 2 ficheros pytest, solo funciones puras OEE
```

### 1.2 `api/` — FastAPI

Factory en `api/main.py`. Monta `/static` y 14 routers, todos con prefijo
`/api/` salvo `pages.py` que sirve HTML en la raíz.

| Módulo | LOC | Qué hace | Crítico |
|---|---:|---|---|
| `main.py` | 88 | App factory, lifespan, error handler HTML, router wiring | sí |
| `config.py` | 71 | Pydantic-settings con prefijo `OEE_`. Construye `effective_database_url` (mssql) y paths | sí |
| `database.py` | 299 | Engine SQLAlchemy con `creator=_mssql_creator`, modelos ORM de `cfg/oee`, `init_db()` siembra desde CSV/JSON | sí |
| `deps.py` | 9 | Templates Jinja2 compartido | sí |
| `models.py` | 67 | Pydantic DTOs (Recurso, Pipeline, Ciclo…) | sí |
| `routers/pages.py` | 118 | 10 rutas HTML que renderizan los templates | sí |
| `routers/centro_mando.py` | 167 | `/api/dashboard/summary` con caché 15 min consultando `dbizaro.admuser.fmesmic` | sí |
| `routers/luk4.py` | 446 | Panel pabellón 5: estado, tiempos ciclo, piezas por turno, alarmas, zonas del plano | sí |
| `routers/capacidad.py` | 235 | Capacidad teórica vs real con P10 de ciclos de 180 días | sí |
| `routers/bbdd.py` | 575 | Explorador genérico: list DBs/tables/columns, `query` SELECT-only, `buscar-rutas` heurístico | sí |
| `routers/recursos.py` | 223 | CRUD recursos + autodetectar CTs desde IZARO + reglas heurísticas de nombrado | sí |
| `routers/ciclos.py` | 159 | CRUD ciclos + `calcular` (KDE mode) + `live` (estado máquina) | sí |
| `routers/operarios.py` | 264 | Consulta de fichajes `admuser.fmesope` + `fmesdtc` para ficha por operario | sí |
| `routers/historial.py` | 177 | Lista y detalle de ejecuciones + `tendencias` OEE diario + regenerar | sí |
| `routers/pipeline.py` | 41 | POST SSE `/api/pipeline/run` | sí |
| `routers/datos.py` | 42 | Igual que pipeline pero sin módulos OEE (para Power BI) | dup |
| `routers/conexion.py` | 74 | GET/PUT config MES (incluye password en claro en GET) | sí |
| `routers/email.py` | 97 | CRUD contactos + `/test` + `/enviar` con adjuntos | sí |
| `routers/informes.py` | 58 | List + servir/descargar PDF | sí |
| `routers/plantillas.py` | 87 | CRUD JSON en `data/report_templates/`, no enganchado al pipeline | experimental |
| `routers/health.py` | 32 | Salud de web/db/MES | sí |
| `services/db.py` | 93 | Wrapper fino sobre `OEE.db.connector` | sí |
| `services/pipeline.py` | 431 | Orquesta extracción→CSV→módulos OEE→PDFs→BD, genera SSE | sí |
| `services/metrics.py` | 362 | Cálculo OEE interactivo desde `oee.datos` | sí |
| `services/turnos.py` | 77 | `TURNOS` constante + helpers (T1 06-14, T2 14-22, T3 22-06) | sí |
| `services/informes.py` | 80 | Recorre `informes/` y sirve árbol PDF | sí |
| `services/email.py` | 76 | SMTP con `db_config.json['smtp']` | sí |

**Probable código muerto / experimental**:
- `routers/plantillas.py` — la propia UI dice "se implementará en una futura actualización".
- `routers/datos.py` — solapa con `routers/pipeline.py` con `modulos=[]`.
- `test_email.py` en raíz — script one-shot, no test real.
- `OEE/utils/cycles_registry.py` y `excel_import.py` — flujo `source=excel` del
  pipeline, probablemente no usado en producción hoy.
- `data/oee.db` — SQLite residual de la versión anterior (hoy todo está en
  SQL Server `ecs_mobility`).

### 1.3 `OEE/` — núcleo de cálculo + conector

| Archivo | LOC | Qué hace |
|---|---:|---|
| `OEE/db/connector.py` | 847 | Conexión a `dbizaro`, extracción de `fmesdtc+fmesddf+fmesinc+fprolof`, cálculo de ciclos reales (contadores `zmeshva` con KDE) y estado "live" |
| `OEE/disponibilidad/main.py` | 911 | Genera PDFs de disponibilidad por recurso (matplotlib + PdfPages) |
| `OEE/rendimiento/main.py` | 687 | Genera PDFs de rendimiento, lee `ciclos.csv` |
| `OEE/calidad/main.py` | 599 | Genera PDFs de calidad |
| `OEE/oee_secciones/main.py` | 1624 | Genera PDFs agregados por sección, contiene el core `convertir_raw_a_metricas` usado por los tests |
| `OEE/utils/data_files.py` | 42 | Enumera CSV por sección |
| `OEE/utils/excel_import.py` | 105 | Convierte XLSX → CSV (source=excel) |
| `OEE/utils/cycles_registry.py` | 68 | Autocompleta `ciclos.csv` con pares (máquina, ref) nuevos |

Los 4 módulos OEE leen CSV de `data/recursos/<SECCION>/` y escriben PDFs
vía `matplotlib.backends.backend_pdf.PdfPages`. Son **síncronos**, bloqueantes
y pesados (minutos por ejecución).

### 1.4 Totales de líneas

| Lenguaje | LOC |
|---|---:|
| Python (api + OEE + services + scripts + tests + mcp + server/test_email) | ~10 361 |
| HTML/Jinja2 (templates + partials) | ~4 957 |
| JavaScript (static/js) | 421 |
| CSS (static/css) | 103 |

Los archivos **>800 líneas** (umbral del rule común de estilo): `OEE/db/connector.py`
(847), `OEE/disponibilidad/main.py` (911), `OEE/oee_secciones/main.py` (1624).
Son los candidatos a trocear.

---

## 2. Capa de datos actual

### 2.1 Cómo se conecta a SQL Server

Dos vías coexisten:

1. **SQLAlchemy + pool** (`api/database.py`):
   ```python
   engine = create_engine(
       "mssql+pyodbc://",
       creator=_mssql_creator,       # pyodbc.connect(...) directo
       pool_pre_ping=True, pool_size=5, max_overflow=10, pool_recycle=3600,
   )
   ```
   La URL es un placeholder; toda la conexión real la hace el creator a
   **`ecs_mobility`** (BD propia de la app) con `ODBC Driver 18 for SQL Server`,
   `Encrypt=yes; TrustServerCertificate=yes`, timeout 10 s.
   Se usa en: `luk4.py`, `centro_mando.py`, `historial.py`, routers que leen
   tablas de `cfg/oee/luk4`.

2. **`pyodbc.connect()` directo** (`OEE/db/connector.py`, `routers/bbdd.py`,
   `routers/capacidad.py`, `routers/operarios.py`): abre y cierra conexión
   por petición, sin pool. Permite cambiar la BD (catalog) por request.
   Se usa para consultar **`dbizaro`** (IZARO / MES).

Driver preferido: `ODBC Driver 18 for SQL Server`, con fallback a 17/13/Native/SQL Server.
Instalado en `Dockerfile` (debian bookworm) y en host via `install_odbc.sh`
(Ubuntu). No hay reconexión explícita — depende de `pool_pre_ping`.

### 2.2 Queries en código vs archivos

Todo el SQL vive **hardcoded en Python**. No hay archivos `.sql` versionados.

Queries relevantes (todas contra `admuser.*` de `dbizaro` salvo LUK4):
- `OEE/db/connector.py:_SQL_TEMPLATE` — `fmesdtc + fmesddf + fmesinc + fprolof`
  (extracción principal, con placeholders parametrizados).
- `OEE/db/connector.py` — `zmeshva` (contadores), `fmesrec`, y fallback sobre
  `fmesdtc` para ciclos reales.
- `routers/centro_mando.py` — `fmesmic` (estado actual, SQL con nombre de
  3 partes: `dbizaro.admuser.fmesmic`).
- `routers/capacidad.py` — `fmesdtc + fmesrec + fprolof` (P10 de ciclos a 180 d).
- `routers/operarios.py` — `fmesope + fmesdtc + fmesrec + fprolof`.
- `routers/bbdd.py` — exploración genérica y endpoint `query` SELECT-only con
  whitelist de palabras clave (rechaza INSERT/UPDATE/DELETE/DDL y multi-statement).
- `routers/luk4.py` — `luk4.estado`, `luk4.tiempos_ciclo`, `luk4.alarmas`,
  `luk4.plano_zonas` (todas en **`ecs_mobility`**).

### 2.3 Mapa de tablas IZARO tocadas hoy

| Tabla `admuser.*` | Usada por | Propósito |
|---|---|---|
| `fmesdtc` | connector, capacidad, operarios, bbdd | Partes de producción (tiempo, cantidad, CT, proceso) |
| `fmesddf` | connector | Defectos (malas/recuperadas) — LEFT JOIN con dtc |
| `fmesinc` | connector | Incidencias (texto libre) — LEFT JOIN con dtc |
| `fprolof` | connector, capacidad, operarios | Lote/orden → referencia (`lo030`) |
| `fmesrec` | connector (detectar), capacidad, operarios | Catálogo de recursos (nombre CT) |
| `fmesope` | operarios | Catálogo de operarios |
| `fmesmic` | centro_mando | Microparadas / estado instantáneo |
| `zmeshva` | connector | Contadores de piezas por CT (cálculo ciclos reales) |
| `fproope`, `fprorut`, `fprorut_edm` | bbdd `/explorar-rutas-detalle` | Exploración de rutas/operaciones (investigación) |

El README menciona `fmesdtc/fmesddf/fmesinc/fprolof` — **confirmado**, más
`fmesrec`, `fmesope`, `fmesmic`, `zmeshva`, y las tres `f*rut*` para investigación.

### 2.4 Esquema Postgres local

**No se usa.** El servicio `db: postgres:16-alpine` del compose arranca,
pero ningún código de la app habla con Postgres:
- `settings.database_url` se construye con `mssql+pyodbc`.
- `engine` ignora la URL y conecta vía creator pyodbc a `ecs_mobility`.
- La env var `OEE_DATABASE_URL=postgresql://...` que inyecta el compose
  nunca se lee por el engine (hay un camino muerto en `effective_database_url`).

No hay Alembic ni migraciones declarativas. El estado "migrado" de la BD
vive en **scripts idempotentes manuales**:
- `scripts/create_oee_db.py` — crea BD `ecs_mobility`, esquemas `cfg/oee/luk4`,
  tablas, índices y sinónimos en `dbo` para compatibilidad con el túnel IoT.
- `scripts/migrate_schemas.py` — mueve tablas de `dbo` a `cfg/oee/luk4` y
  crea sinónimos.
- `scripts/extract_2025.py` — one-shot de importación 2025.

Tablas vivas en **`ecs_mobility`**:
- `cfg.ciclos`, `cfg.recursos`, `cfg.contactos`.
- `oee.ejecuciones`, `oee.datos`, `oee.metricas`, `oee.referencias`,
  `oee.incidencias`, `oee.informes`.
- `luk4.estado`, `luk4.tiempos_ciclo`, `luk4.alarmas`, `luk4.plano_zonas`.
- Sinónimos `dbo.luk4_estado`, `dbo.luk4_tiempos_ciclo`, `dbo.alarmas_luk4`,
  `dbo.plano_zonas` — el túnel IoT escribe con nombres antiguos.

---

## 3. Auth y audit actuales

- **Auth**: ninguna. No hay router `auth`, no hay middleware de sesión, no
  hay login, no hay decorador que exija usuario. **Todos los endpoints son
  públicos** y la UI no pregunta credenciales. El único "login" conceptual
  es la password del SQL Server, que se configura en Ajustes y se guarda
  en texto plano en `data/db_config.json`.
- **Audit**: no existe. Sólo `logging.basicConfig(level=INFO)` a stdout y
  un handler global de excepciones que devuelve HTML con el traceback al
  navegador (fuga de información). No hay persistencia de acciones, ni
  tabla `audit_log`, ni middleware que registre quién hizo qué cuándo.
- **Handler de errores**: `global_exception_handler` en `api/main.py`
  imprime el traceback completo en la respuesta HTTP — hay que cerrarlo
  en Mark-III.

---

## 4. Compose, Makefile y DevEx

### 4.1 Servicios en `docker-compose.yml`

| Servicio | Imagen / build | Puertos | Healthcheck | Depende |
|---|---|---|---|---|
| `db` | `postgres:16-alpine` | 5432:5432 | `pg_isready`, 5s/3s/5 reintentos | — |
| `web` | build: `.` | 8000:8000 (+ expose 8000) | **no** | `db` healthy |
| `caddy` | `caddy:2-alpine` | 80:80, 443:443 | **no** | `web` |
| `mcp` | build: `./mcp` | (ninguno expuesto) | **no** | `web` |

Observaciones:
- Postgres publica 5432 al host sin password fuerte (`oee/oee` por defecto)
  y **no lo consume nadie**. Es coste sin beneficio hoy.
- `web` no tiene healthcheck; `caddy` tampoco. Una caída de web no fuerza
  reinicio de caddy.
- No hay `restart: unless-stopped` en `mcp` (sí lo tienen los otros).
- `caddy/Caddyfile` emite cert autofirmado (`tls internal`) — aceptable
  para LAN hasta conseguir DNS-01.

### 4.2 Makefile

17 targets. Todos parecen funcionales salvo dependencias del SO:
- `make ip` usa `ipconfig getifaddr en0` (macOS) con fallback a `hostname -I`,
  funciona en Linux.
- `make dev` usa `ss` y `lsof` para detectar puerto libre — vale.
- `make db-shell` entra en Postgres que **no se usa** — expectativa incorrecta.
- `make health` llama a `/api/health` — correcto.
- `make clean` borra `data/oee.db` (SQLite residual) y volúmenes — OK.

### 4.3 `.env.example` vs `.env` real

Diff de **claves** (no de valores): idénticas. Ambos cubren:
`OEE_HOST, OEE_PORT, OEE_DEBUG, OEE_DB_SERVER, OEE_DB_PORT, OEE_DB_NAME,
OEE_DB_USER, OEE_DB_PASSWORD, OEE_IZARO_DB, OEE_PG_USER, OEE_PG_PASSWORD,
OEE_PG_DB`.

`OEE_DATABASE_URL` está comentado en `.example` y ausente en `.env` real.
El `.env.example` usa `OEE_DB_USER=tu_usuario` — placeholder claro.

---

## 5. UI y rutas

### 5.1 Rutas FastAPI (HTML)

Todas en `api/routers/pages.py`, sin prefijo `/api`:

| Ruta | Template | Page key |
|---|---|---|
| `/` | `luk4.html` | dashboard |
| `/pipeline` | `pipeline.html` | pipeline |
| `/informes` | redirect 301 → `/historial` | — |
| `/recursos` | `recursos.html` | recursos |
| `/historial` | `historial.html` | historial |
| `/ciclos-calc` | `ciclos_calc.html` | ciclos_calc |
| `/operarios` | `operarios.html` | operarios |
| `/datos` | `datos.html` | datos |
| `/bbdd` | `bbdd.html` | bbdd |
| `/capacidad` | `capacidad.html` | capacidad |
| `/ajustes` | `ajustes.html` | ajustes |

Hay templates sin ruta directa: `ciclos.html`, `informes.html`, `plantillas.html`.
`ciclos.html` parece no enlazarse desde nav; `plantillas.html` se declara como
"futura actualización". `informes.html` está redireccionado desde `/informes`.

### 5.2 Rutas FastAPI (API)

Agrupadas por prefijo, todas con `/api/`:

- `/api/health` — salud general.
- `/api/dashboard/summary`, `/api/dashboard/refresh` — centro de mando.
- `/api/conexion/status|config|explorar` — MES/IZARO.
- `/api/recursos` + `/row` + `/detectar` + `/auto-detectar`.
- `/api/pipeline/run` (SSE).
- `/api/informes` + `/pdf/*` + `/download/*` + `DELETE /{fecha}`.
- `/api/ciclos` + `/row` + `/calcular/{nombre}` + `/live/{nombre}` + `/sync-csv`.
- `/api/historial` + `/{id}` + `/{id}/metrics` + `/{id}/regenerar` +
  `/{id}` (DELETE) + `/tendencias`.
- `/api/email/contactos` + `/test` + `/enviar`.
- `/api/operarios` + `/{codigo}`.
- `/api/bbdd/databases|tables|columns|preview|schema|query|buscar-rutas|explorar-rutas-detalle`.
- `/api/datos/extraer` (SSE).
- `/api/luk4/status|turno-detail|zonas`.
- `/api/capacidad`.
- `/api/plantillas` (CRUD JSON en disco).

### 5.3 Jinja2 / navegación

`templates/base.html` define layout único con sidebar + topbar:
- Sidebar: lista de `nav_items` hardcodeada con 9 ítems visibles:
  Centro Mando, Datos, Análisis, Historial, Capacidad,
  Recursos, Calcular Ciclos, Operarios, BBDD, Ajustes.
- El badge de conexión se refresca cada 30 s con HTMX contra
  `/api/conexion/status`.
- **Toast system** montado con Alpine en `base.html`.

Todas las demás páginas extienden `base.html` con `{% extends %}`. Hay un
partial: `_partials/mapa_pabellon.html` (331 líneas) que reutiliza `luk4.html`
para los 4 pabellones.

### 5.4 Alpine.js vs JS plano

Cada template usa Alpine con densidad muy alta. Conteo de patrones
`x-data|Alpine|@click|x-show|x-if|x-for` por template:

```
ciclos_calc.html  43     historial.html  42     pipeline.html   29
bbdd.html         25     capacidad.html  22     recursos.html   55
plantillas.html   21     operarios.html  12     ajustes.html    12
datos.html        11     luk4.html       11     ciclos.html     18
base.html          9     informes.html   26
```

`static/js/app.js` (421 líneas) contiene:
- Handler del badge de conexión (HTMX afterRequest).
- `runPipeline(formData)` — fetch + SSE + stream reader.
- `renderOeeDashboard()` — render manual de KPIs/tabla/chart con Chart.js
  (DOM innerHTML a mano, no Alpine). **Hay duplicación conceptual**: cada
  template tiene su `x-data` con lógica de pantalla, pero el dashboard OEE
  se dibuja 100% en JS plano en `app.js`.
- `showToast` duplicado (uno en `base.html` como Alpine, otro en `app.js`
  como DOM append).
- Solicitud de permisos de `Notification` al cargar la página.

CDNs externos (cargados en cada request): TailwindCSS runtime,
htmx@2.0.4, htmx-ext-sse@2.2.2, alpinejs@3.14.8, chart.js@4.4.7. Ningún
bundler, ningún `package.json`.

---

## 6. Deuda técnica visible

### 6.1 Archivos sospechosos

- **`.env:Zone.Identifier`** — residuo WSL, **trackeado en git**. Basura
  que debería borrarse y añadirse al `.gitignore`.
- **`.env` con credenciales reales trackeables**: el fichero actual en
  disco contiene `OEE_DB_USER=sa` y `OEE_DB_PASSWORD=<valor en claro>` —
  está en `.gitignore` (OK), pero conviene **rotar la contraseña** como
  si hubiese sido expuesta, porque el que tenga el equipo la ve.
- **`test_email.py`** en raíz — script one-shot suelto, no es un test.
- **`data/oee.db`** — SQLite residual, trackeado y con 176 KB. No se usa
  (todo está en SQL Server). `make clean` lo borra pero el repo lo trae.
- **`data/export_*.csv`** — tres CSV grandes (94 KB, 630 KB, 84 KB) de
  exportaciones puntuales, trackeados.
- **`data/db_config.json`** contiene 67 recursos y **no tiene sección
  `smtp`**. El `email.py` lee `cfg["smtp"]` y devolverá vacío siempre —
  el envío por email está roto hasta que alguien rellene el JSON manualmente.
  `.gitignore` excluye `data/db_config.json` (OK), pero eso significa que
  **el único lugar con SMTP es fuera del control de versiones**.
- **`server.py` en raíz** — wrapper de `uvicorn.run`, redundante con el
  `CMD` del Dockerfile y con `make dev`.

### 6.2 `requirements.txt`

```
matplotlib>=3.5.0    pandas>=1.4.0      pyodbc>=5.0.0
fastapi>=0.109.0     uvicorn[standard]>=0.27.0
jinja2>=3.1.0        python-multipart>=0.0.6
pydantic-settings>=2.0.0
sqlalchemy>=2.0.0    psycopg2-binary>=2.9.0
```

Problemas:
- Todo con `>=`, **ningún pin duro**. Un `docker build` dentro de 6 meses
  puede traer breaking changes.
- `psycopg2-binary` incluido sin necesidad (Postgres no se usa hoy).
- Falta `httpx` (lo usa el MCP), `pytest` (lo necesitan los tests). MCP
  tiene su propio `requirements.txt` separado.
- No hay herramientas de calidad (black, ruff, mypy, bandit).

### 6.3 Duplicidades y configs dispersas

- `SECTION_MAP` (luk1..omr → LINEAS, t48 → TALLADORAS) está en **4 sitios**:
  `api/database.py`, `OEE/db/connector.py`, `OEE/utils/excel_import.py`,
  `static/js/app.js`. Cambiar un recurso requiere tocar los 4.
- `TURNOS` está centralizado en `api/services/turnos.py` — bien — pero las
  funciones `determinar_turno` viven duplicadas en `OEE/oee_secciones/main.py`,
  `OEE/rendimiento/main.py`, `OEE/calidad/main.py`.
- `INCIDENCIA_DISPONIBILIDAD_PATTERNS` idéntico en `disponibilidad/main.py`
  y `oee_secciones/main.py`.
- `_build_connection_string` se **reimplementa** en `api/routers/bbdd.py` y
  en `scripts/create_oee_db.py` y en `scripts/migrate_schemas.py`, con
  variaciones sutiles, en lugar de reutilizar `OEE.db.connector._build_connection_string`.
- Config de conexión MES vive a la vez en `.env`, `data/db_config.json` y en
  memoria (`api/config.py`). `services/db.get_config` "inyecta" desde settings
  si el JSON está vacío — complejidad sin ganancia clara.
- Handler de excepciones devuelve **traceback completo** al navegador
  (`api/main.py:44-57`). Fuga de información evidente.

### 6.4 Tests

- Framework: **pytest**. Tests en `tests/test_oee_calc.py` (197 líneas) y
  `tests/test_oee_helpers.py` (99 líneas). Total ~30 tests unitarios puros
  sobre funciones del `OEE/oee_secciones/main.py`.
- Cobertura: **sólo el core del cálculo OEE**. Ningún test para routers,
  servicios, conector a BD, SSE, pipeline, email, luk4, capacidad, operarios.
- No he podido ejecutarlos en esta sesión: `pytest` no está instalado en
  el entorno del host ni en `requirements.txt`. Los tests son puros (dict
  in → dict out), deberían pasar — pero no puedo confirmarlo ahora mismo.
- **No hay CI**: no existe `.github/` ni `.gitlab-ci.yml` ni similares.

### 6.5 Comentario sobre el estilo de commits

Los 30 últimos mensajes son tipo "Arreglos", "mejoras", "cosas", "Best",
"vivo", "Detodo". Hace imposible reconstruir por qué se tocó algo. No es
un problema técnico, pero sí un riesgo en Mark-III si no se adopta
conventional commits desde el Sprint 0.

---

## 7. Riesgos de migración

### 7.1 Naming `OEE` / `OEE Planta` / `analisis_datos`

Aparece como marca en:
- `title` de FastAPI: "ECS Mobility — Centro de Mando" (v2.0.0) — aquí ya
  no dice OEE, **bien**.
- Muchos templates: `{% block title %}... OEE Planta ...{% endblock %}` en
  `plantillas.html`, `ciclos.html`, `pipeline.html`, etc.
- `README.md`: "OEE Planta — Informes de producción".
- `api/routers/email.py:90` — cuerpo del mail dice "OEE Planta (automático)".
- Nombres de carpeta `OEE/` y módulo `oee_secciones/`, tablas `oee.*`.
- Prefijo `OEE_` en todas las env vars (`OEE_DB_*`, `OEE_PG_*`, `OEE_DATA_DIR`).
- CSS/JS tienen comentarios "OEE Planta".
- Dominio/ID interno del MCP: `oee-planta`.
- Path del proyecto: `/home/eeguskiza/analisis_datos/`.

Rename coordinado necesario: títulos en HTML, subject de la app FastAPI,
readme, navegación, comentarios de banner, env vars (`OEE_*` → `NEXO_*` con
capa de compatibilidad), MCP ID. Los **nombres de BD y de schemas**
(`ecs_mobility`, `oee.*`, `luk4.*`, `cfg.*`) es razonable mantenerlos
porque ya están en el tunel IoT y en Power BI; pero el concepto "OEE" como
producto desaparece.

### 7.2 Acoplamiento fuerte entre negocio y datos

- **`OEE/disponibilidad`, `OEE/rendimiento`, `OEE/calidad`, `OEE/oee_secciones`**
  leen **directamente del sistema de ficheros** (`data/recursos/<SECCION>/*.csv`).
  Todos los PDFs pasan por `data/ciclos.csv` en disco, no por BD. Pasar
  esto a un repositorio significa **cambiar la firma** de los 4 módulos
  o meter un adaptador "repository → CSV temporal" equivalente al actual
  `api/services/pipeline._sync_ciclos_to_csv`. Puntos de dolor: 847+911+
  687+599+1624 = **4 668 líneas de código basado en CSVs**.

- **Queries SQL embebidas** en routers:
  - `routers/capacidad.py:54-129` — 2 queries grandes de capacidad.
  - `routers/luk4.py` — 6 queries largas.
  - `routers/centro_mando.py:45-59` — query con `dbizaro.admuser.fmesmic`
    hardcoded (3-part name → asume que dbizaro y ecs_mobility están en la
    misma instancia).
  - `routers/operarios.py` — 4 queries grandes.
  - `OEE/db/connector.py:_SQL_TEMPLATE` — query principal con placeholders.
  Pasarlas a `.sql` versionados requiere un loader.

- **Cross-database reference hardcodeada**: `routers/centro_mando.py` y
  otros asumen que `dbizaro` y `ecs_mobility` **están en la misma instancia
  SQL Server**. Si Mark-III las separa en instancias distintas, estas queries
  rompen.

- **`init_db()` siembra BD al arrancar** leyendo CSV/JSON desde disco. Pasar
  eso a migraciones versionadas es un sprint por sí solo.

- **`db_config.json`** es fuente de verdad para recursos, credenciales,
  uf_code, referencia_campo y smtp, y se sincroniza bidireccionalmente con
  `cfg.recursos` (SQLAlchemy). Romper ese puente rompe recursos Y conexión.

### 7.3 Otros riesgos menores

- PDFs con **matplotlib** bloquean el hilo — un SSE que tarda varios minutos.
  Preflight en Sprint 4 tendrá que estimarlo por nº de recursos × nº días.
- La UI está montada con Alpine + HTMX + Chart.js + Tailwind **runtime**
  desde CDN. Cualquier refactor UI por roles (Sprint 5) tendrá que decidir
  si mantiene el stack runtime o introduce build, porque si no, añadir
  componentes por rol es puro copy/paste en Jinja.
- El `global_exception_handler` actual rompe cualquier flujo API JSON porque
  devuelve HTML. Al añadir auth hay que reescribirlo.

---

## 8. Resumen ejecutivo del estado

- **Monolito FastAPI ya existente**, ~10 K líneas Python y ~5 K Jinja2,
  sin auth, sin audit, con queries SQL embebidas en routers y un flujo PDF
  con matplotlib que pasa por CSVs en disco.
- **Dos BBDD en una instancia SQL Server**: `dbizaro` (IZARO, read-only) y
  `ecs_mobility` (app — tablas propias en esquemas `cfg/oee/luk4`). Postgres
  está en compose pero es **coste sin uso**.
- **Naming OEE** cubre títulos, templates, prefijo env var, MCP ID y README;
  BD y esquemas pueden mantenerse.
- **Deuda clara**: credenciales en texto plano, handler que devuelve tracebacks,
  requirements sin pinear, SECTION_MAP duplicado en 4 sitios, tests solo del
  cálculo puro, sin CI.
- **Puntos fuertes**: separación `api/` vs `OEE/` razonable, tests unit
  del núcleo OEE, `services/turnos.py` ya centralizado, MCP funcionando
  con whitelist de SELECT, Caddy HTTPS listo para LAN.
