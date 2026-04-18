# Phase 1 · Sprint 0 — Naming + Higiene + CI

**Phase**: 1 of 7
**Milestone**: Mark-III
**Slug**: `1-naming-higiene-ci`
**Planned**: 2026-04-18
**Estimación**: 2-3 días de trabajo focalizado

---

## Objetivo

Dejar el repo con marca **Nexo**, higiene mínima, CI en GitHub Actions,
`global_exception_handler` que no filtra traceback, audit del historial git
documentado y el servicio MCP aparcado en un profile de compose — todo sin
tocar lógica de negocio ni crear auth.

---

## Precondiciones (antes de `/gsd-execute-phase 1`)

1. **Estado git**: rama `feature/Mark-III` limpia; commits previos de
   scaffolding (07b220b, 08f6af2, 2457cdb) aplicados.
2. **Logos ya ubicados**: `static/img/brand/nexo/logo.png` y
   `static/img/brand/ecs/logo.png` existen (verificado en scaffolding
   Paso 2). Si por cualquier motivo no están, **parar** y avisar antes
   de tocar templates.
3. **Docker funcional**: el host tiene `docker`, `docker compose`, y
   `make build` construye sin errores antes de empezar.
4. **Acceso a red interna** para resolver `192.168.0.4:1433` si algún
   commit toca configuración que requiere validar la conexión SQL Server.
   No es bloqueante para la mayoría de los commits.
5. **Suite de tests actual funciona**: `pytest` en local pasa los
   ~30 tests existentes (sin tocarlos). Si no pasan pre-commit, documentar
   cuáles fallan y seguir — Sprint 0 no arregla tests.

---

## Lo que entra en esta fase (13 commits atómicos)

Orden estricto. Cada commit debe dejar el repo en estado arrancable
(`make dev`). Si uno rompe el arranque, revertir/ajustar **antes** de
avanzar al siguiente.

### Commit 1 — `chore: audit git history for leaked credentials` 🚦 GATE 1

**Qué hace**: genera `docs/SECURITY_AUDIT.md`. NO modifica código ni
borra archivos. NO ejecuta `git filter-repo`. NO fuerza push.

**Acciones**:
- Ejecutar `git log -p -- '.env*' 'data/db_config*' '*.ini' '*.cfg'` y variantes (`*.yml`, `*.yaml`, `*.json` con secretos conocidos).
- Filtrar output buscando patrones: `password=`, `PASSWORD=`, `api_key`, `DB_PASSWORD`, `sa` user con password, cadenas entre comillas tras `=` en secciones `[credentials]`, etc.
- Documentar cada hallazgo en `docs/SECURITY_AUDIT.md`: commit hash + archivo + tipo de credencial + si sigue siendo válida (si la sabes). **NO copiar valor literal al doc.**
- Marcar cada hallazgo como "expuesto, rotar cuando sea viable".

**Archivos tocados**: crea `docs/SECURITY_AUDIT.md` (nuevo).

**Criterio de hecho**:
- `docs/SECURITY_AUDIT.md` existe con al menos la estructura (aunque haya 0 hallazgos, el doc explicita que se auditó y no se encontró nada).
- `git log -p` no se invoca con redirección a un archivo trackeado (evitar meter la salida cruda en el repo).

**GATE 1 — comportamiento del skill de ejecución**:
- Si aparecen credenciales expuestas → **parar**, presentar el informe, pedir al operador que decida si ejecutar `filter-repo` manualmente (fuera de este sprint). El skill **nunca** ejecuta `filter-repo` ni `git push --force` por sí solo.
- Si el informe está limpio → avanzar al commit 2.

---

### Commit 2 — `chore: remove tracked junk files`

**Qué hace**: elimina archivos residuales del tracking.

**Acciones**:
- `git rm .env:Zone.Identifier` (residuo WSL).
- `git rm test_email.py` (script one-shot en raíz, no es test).
- `git rm server.py` (wrapper de uvicorn.run redundante con Dockerfile CMD y `make dev`).

**Archivos tocados**: ninguno además de los borrados.

**Criterio de hecho**:
- `git ls-files | grep -E '(Zone.Identifier|test_email\.py|server\.py)'` → vacío.
- `make dev` sigue arrancando (`server.py` no se invocaba desde Makefile).

---

### Commit 3 — `chore: update .gitignore patterns`

**Qué hace**: añade patrones para evitar que los residuos vuelvan.

**Acciones**:
- Añadir `*:Zone.Identifier` al `.gitignore`.
- Añadir otros residuos comunes si no están: `*.pyc`, `__pycache__/`, `.venv/`, `*.swp`, `.DS_Store`.

**Archivos tocados**: `.gitignore`.

**Criterio de hecho**:
- `git check-ignore .env:Zone.Identifier` (simulado) debería matching.
- No se borran patterns ya existentes.

---

### Commit 4 — `chore: move install_odbc.sh to scripts/`

**Qué hace**: mueve el script de instalación ODBC al directorio `scripts/`.

**Acciones**:
- Verificar si ya está en `scripts/install_odbc.sh`. Si está, commit vacío (no se crea — saltar commit).
- Si está en raíz, `git mv install_odbc.sh scripts/install_odbc.sh`.
- Revisar `Dockerfile`, `docker-compose.yml`, `README.md`, `Makefile` por si referencian `install_odbc.sh` sin prefijo; actualizar si es el caso.

**Archivos tocados**: `install_odbc.sh` → `scripts/install_odbc.sh`; posible update de Dockerfile/README.

**Criterio de hecho**:
- `ls scripts/install_odbc.sh` existe.
- `ls install_odbc.sh` no existe (en raíz).
- `make build` sigue construyendo (si Dockerfile lo referenciaba, update hecho).

---

### Commit 5 — `chore: handle data/oee.db`

**Qué hace**: decide y aplica qué hacer con el SQLite residual `data/oee.db`
(176 KB según `docs/CURRENT_STATE_AUDIT.md`).

**Acciones**:
1. Inspeccionar el SQLite: `sqlite3 data/oee.db ".tables"` + `.schema` + conteo de filas por tabla.
2. Decisión:
   - Si **residuo o vacío** (< 1 MB + 0 rows o tablas irrelevantes): `git rm data/oee.db` + documentar en `docs/DATA_MIGRATION_NOTES.md` qué se inspeccionó y por qué es residuo.
   - Si **datos reales** (> 1 MB o tablas con rows significativas): exportar a `data/backups/oee_db_snapshot.sqlite` (añadir `data/backups/` al `.gitignore`), `git rm data/oee.db`, y documentar en `docs/DATA_MIGRATION_NOTES.md` qué hay y por qué se preserva offline.
3. Crear `docs/DATA_MIGRATION_NOTES.md` con:
   - Fecha de la inspección.
   - Tablas encontradas + conteo aproximado de rows.
   - Decisión tomada (borrado / backup).
   - Hash del commit anterior para que el blob siga en historial si se quiere recuperar.

**Archivos tocados**: `data/oee.db` (borrado); `docs/DATA_MIGRATION_NOTES.md` (nuevo); posible update de `.gitignore` si se crea `data/backups/`.

**Criterio de hecho**:
- `data/oee.db` no está en `git ls-files`.
- `docs/DATA_MIGRATION_NOTES.md` contiene justificación razonada.
- Si hay backup offline, la ruta está documentada y NO trackeada.

---

### Commit 6 — `refactor: rename OEE_* env vars to NEXO_*, split MES/APP`

**Qué hace**: migra el prefijo de variables de entorno + separa SQL Server
en dos bloques lógicos (MES read-only + APP ecs_mobility).

**Acciones**:
- Actualizar `api/config.py`:
  - Renombrar campos Pydantic con prefijo `NEXO_*`.
  - Split: `mes_server/port/user/password/db` (NEXO_MES_*) y `app_server/port/user/password/db` (NEXO_APP_*).
  - **Capa de compatibilidad**: para cada campo nuevo, si la env var `NEXO_*` está vacía o no definida, caer a la `OEE_*` equivalente. Ejemplo: `NEXO_MES_SERVER` ← fallback ← `OEE_DB_SERVER`.
- Actualizar `docker-compose.yml`:
  - Reemplazar `OEE_*` por `NEXO_*` en las secciones `environment:` de `web`, `mcp`, y `db` si aplica.
- Actualizar `.env.example`:
  - Añadir todas las variables `NEXO_*` (MES + APP + web + PG + branding).
  - Dejar las viejas `OEE_*` comentadas con nota "deprecadas, compat activa durante Mark-III".
- Los módulos que leen de `settings.*` no requieren cambios (la config centralizada oculta el rename).

**Archivos tocados**: `api/config.py`, `docker-compose.yml`, `.env.example`.

**Criterio de hecho**:
- `docker compose config` muestra las nuevas variables expandidas sin errores.
- `python -c "from api.config import settings; print(settings.mes_server, settings.app_server)"` devuelve los valores actuales (vía compat OEE_*).
- `make dev` arranca y `/api/health` responde OK.

**Nota**: este commit **no rota credenciales**. `.env` en disco del servidor LAN sigue con sus valores `OEE_*` actuales y la compat los lee.

---

### Commit 7 — `refactor: update UI titles and metadata to Nexo`

**Qué hace**: rebrand efectivo de la capa de presentación + cableado de
variables de branding.

**Acciones**:
- Añadir a `api/config.py` (si no se hizo en commit 6): `app_name`, `company_name`, `logo_path`, `ecs_logo_path` leyendo de `NEXO_APP_NAME`, `NEXO_COMPANY_NAME`, `NEXO_LOGO_PATH`, `NEXO_ECS_LOGO_PATH`.
- Inyectar esas 4 variables en el contexto global de Jinja2 vía `Jinja2Templates` globals o dependency en `api/deps.py`.
- Actualizar `templates/base.html`:
  - `<title>{% block title %}{{ app_name }}{% endblock %}` (era `ECS Mobility`).
  - `<h1>`/sidebar: texto "NEXO" en lugar de "ECS MOBILITY". Logo `<img src="{{ logo_path }}">` (Nexo).
  - Añadir footer con `<img src="{{ ecs_logo_path }}">` + `© {{ company_name }}`.
  - Fallback del `<img>` en `onerror`: pasar de "ECS" a "NEXO".
- Actualizar `static/js/app.js`:
  - `Notification('ECS Mobility', ...)` → usar `{{ company_name }}` si se puede inyectar por template; si no, dejar como "ECS Mobility" (empresa) y sólo cambiar el icon a la variable injectada vía `data-*` attributes o `window.NEXO_CONFIG`.
  - Actualizar `renderOeeDashboard` si tiene strings visibles "OEE Planta".
- Actualizar `api/main.py`:
  - `FastAPI(title="Nexo", description=..., version="3.0.0")` (era "ECS Mobility — Centro de Mando" v2.0.0).
  - Favicon usa `settings.logo_path` (Nexo, no ECS).
- Actualizar `README.md`: título "Nexo — Informes de produccion" + nota de rebrand ECS Mobility.
- Actualizar templates que digan "OEE Planta" hardcoded: `plantillas.html`, `ciclos.html`, `pipeline.html`, body de mails en `api/routers/email.py`, etc. Buscar con `grep -rn "OEE Planta" templates/ api/`.
- Renombrar ID MCP: `mcp/server.py` o metadata del compose service `mcp` de `oee-planta` → `nexo-mcp`. Actualizar `mcp/Dockerfile` y `mcp/` si referencian el ID. Documentar en README que usuarios con `.claude.json` apuntando a `oee-planta` deben actualizar.

**Archivos tocados**: `api/config.py`, `api/main.py`, `api/deps.py`, `templates/base.html`, `templates/*` (varios), `static/js/app.js`, `static/css/app.css` (si hay strings en comments), `README.md`, `mcp/server.py`, `docker-compose.yml` (mcp service name), `api/routers/email.py`.

**Criterio de hecho**:
- `make dev` + navegador a `/`: sidebar dice NEXO con logo de Nexo; footer muestra logo ECS + "© ECS Mobility"; `<title>` es "Nexo".
- `curl /api/docs` (OpenAPI): title es "Nexo".
- `grep -rn "OEE Planta" templates/ api/` devuelve 0 o sólo comentarios no-visibles.

---

### Commit 8 — `fix: global_exception_handler no longer leaks tracebacks`

**Qué hace**: cierra la fuga de información del handler de excepciones.

**Acciones**:
- Reescribir `global_exception_handler` en `api/main.py`:
  - Generar UUID por error: `error_id = str(uuid.uuid4())`.
  - Loguear traceback completo server-side con `error_id`: `logger.exception("Unhandled error %s", error_id)`.
  - Si la request es HTML (Accept header o `/api/` prefix), devolver una página HTML minimal: "Internal error. ID: {error_id}. Contacta con soporte."
  - Si es `/api/*` JSON, devolver `{"error_id": error_id, "message": "Internal error"}` con status 500.
  - Nunca incluir traceback ni detalles de excepción en el body.
- Verificar que no hay otros handlers que filtren info:
  - `grep -rn "traceback.format_exc\|tb_str\|traceback.print" api/` — si aparecen otros handlers o endpoints que devuelven traceback en el body, corregirlos (decisión del user: "si encuentras algo grave no contemplado, repórtalo pero NO lo arregles sin consultar salvo que sea trivial y del mismo tipo que ya estoy arreglando"). Si son del mismo tipo, arreglar; si no, reportar.

**Archivos tocados**: `api/main.py`; posible update de otros handlers si tipo mismo.

**Criterio de hecho**:
- Forzar un 500 en dev (endpoint `/api/health/__crash__` temporal, o navegar a un endpoint que dispare una divisón por cero inyectada en debug) → response tiene `error_id` UUID, sin traceback.
- `logger` (uvicorn stdout en docker compose logs web) muestra el traceback completo con el UUID.

---

### Commit 9 — `chore: move mcp service to docker-compose profile`

**Qué hace**: aparca el MCP server en un profile de compose.

**Acciones**:
- Editar `docker-compose.yml`, servicio `mcp`: añadir `profiles: ["mcp"]`.
- `make up` y `make dev` siguen sin flag y por tanto **no** arrancan `mcp`.
- Actualizar `README.md` sección "Arranque":
  - Indicar que MCP está aparcado.
  - Documentar cómo arrancarlo manualmente: `docker compose --profile mcp up -d mcp` o `docker compose --profile mcp up`.
  - Nota breve sobre cuándo se usa MCP (inspección local desde Claude Code).

**Archivos tocados**: `docker-compose.yml`, `README.md`.

**Criterio de hecho**:
- `make up` no arranca el contenedor `mcp`: `docker compose ps` tras `make up` no lo lista.
- `docker compose --profile mcp up -d mcp` sí lo arranca.
- `make down` sigue parando todo (profile también si está up).

---

### Commit 10 — `build: pin requirements.txt, add requirements-dev.txt`

**Qué hace**: pinea deps y separa las herramientas de desarrollo.

**Acciones**:
- `requirements.txt`: reemplazar cada `>=` por `==x.y.z` con la versión actualmente instalada en el container de producción. Pasos:
  - `docker compose run --rm web pip freeze | grep -Ei 'fastapi|uvicorn|pydantic|sqlalchemy|pyodbc|pandas|matplotlib|jinja2|httpx|psycopg2|python-multipart' > /tmp/pinned.txt` (o equivalente en local si no hay docker).
  - Trasladar versiones al `requirements.txt` conservando el orden y los comentarios.
- `requirements-dev.txt` (nuevo):
  - `ruff==<latest>`
  - `pytest==<latest>`
  - `pytest-cov==<latest>`
  - `gitleaks` — nota: gitleaks es binario, se usa desde CI; no va en `requirements-dev.txt` salvo que se use el wrapper pip `gitleaks-python`. Dejar comentario.
- `Dockerfile`: no cambia (sigue haciendo `pip install -r requirements.txt`). Devs locales corren `pip install -r requirements-dev.txt` si quieren tools.

**Archivos tocados**: `requirements.txt`, `requirements-dev.txt` (nuevo).

**Criterio de hecho**:
- `make build` (o `docker build .`) construye sin errores con las versiones pineadas.
- `make dev` arranca y `/api/health` responde OK tras el build.
- `pip install -r requirements-dev.txt` (local) instala ruff y pytest.

---

### Commit 11 — `ci: add GitHub Actions workflow`

**Qué hace**: añade CI mínimo.

**Acciones**:
- Crear `.github/workflows/ci.yml` con:
  - Trigger: `push` a `feature/Mark-III` y `main`, `pull_request` a ambas.
  - Job `lint`:
    - `ruff check .` + `ruff format --check .`
  - Job `test`:
    - Install `requirements.txt` + `requirements-dev.txt`.
    - `pytest -q` — **`continue-on-error: true`** porque Sprint 0 es no bloqueante. Reporta pass/fail en el log pero no falla el workflow.
  - Job `build`:
    - `docker build -t nexo-web:ci .` — sin push.
  - Job `secrets`:
    - `zricethezav/gitleaks-action@v2` (o `gitleaks/gitleaks-action`) — run `gitleaks detect` sobre el repo. Si falla (encuentra secretos), mostrar output; **non-blocking** en Sprint 0.
- Status checks **no bloqueantes** para merges: no configurar branch protection rules en este commit; eso es decisión del operador en Sprint 7.

**Archivos tocados**: `.github/workflows/ci.yml` (nuevo).

**Criterio de hecho**:
- YAML válido: `yamllint .github/workflows/ci.yml` (si está disponible) o `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"` sin errores.
- Tras `git push`, el workflow se dispara en GitHub Actions (Erik lo verifica manualmente en la pestaña Actions).

---

### Commit 12 — `docs: add GLOSSARY, DATA_MIGRATION_NOTES (verify CLAUDE.md/AUTH_MODEL/BRANDING)`

**Qué hace**: completa la documentación core de Mark-III.

**Acciones**:
- Crear `docs/GLOSSARY.md` con los términos:
  - **Nexo**: plataforma interna de ECS Mobility, sucesora de OEE Planta.
  - **MES**: Manufacturing Execution System. En Nexo se refiere a IZARO (BD `dbizaro`).
  - **APP**: referencia a `ecs_mobility` (BD propia de Nexo en SQL Server).
  - **Módulo**: unidad funcional del producto (pipeline, historial, bbdd, capacidad, operarios, recursos, ciclos, luk4, ajustes).
  - **Rol**: nivel de acceso del usuario (propietario / directivo / usuario). Ver `docs/AUTH_MODEL.md`.
  - **Departamento**: área funcional (rrhh / comercial / ingenieria / produccion / gerencia). Ver `docs/AUTH_MODEL.md`.
  - **Propietario / Directivo / Usuario**: los tres roles.
  - **`audit_log`**: tabla append-only en Postgres (`nexo.audit_log`) con historial de requests autenticadas.
  - **Preflight**: estimación de coste de una consulta/operación antes de ejecutarla. Sprint 3.
  - **Postflight**: medición del coste real tras ejecutar + alerta si supera umbral. Sprint 3.
- `docs/DATA_MIGRATION_NOTES.md` ya se creó en commit 5. Verificar que está completo.
- Verificar que `CLAUDE.md`, `docs/AUTH_MODEL.md`, `docs/BRANDING.md` (creados en scaffold Paso 1 y Paso 2) siguen alineados con los commits 1-11 de este sprint. Si algún commit cambió algo que contradice los docs, alinear aquí.

**Archivos tocados**: `docs/GLOSSARY.md` (nuevo); `CLAUDE.md`, `docs/AUTH_MODEL.md`, `docs/BRANDING.md`, `docs/DATA_MIGRATION_NOTES.md` (verificación, posibles updates menores).

**Criterio de hecho**:
- Los 6 archivos de `docs/` + `CLAUDE.md` en raíz existen.
- No hay contradicciones entre `docs/AUTH_MODEL.md` y lo implementado hasta ahora (en Sprint 0 no hay auth, sólo contrato).

---

### Commit 13 — `docs: sync MARK_III_PLAN and OPEN_QUESTIONS with Sprint 0 outcomes`

**Qué hace**: actualiza los docs autoritativos con los hashes finales del
sprint y las decisiones reales tomadas.

**Acciones**:
- `docs/MARK_III_PLAN.md`:
  - En la sección "Sprint 0", añadir un footer "Ejecutado: [fecha]. Commits: [hash 1] … [hash 13]".
  - Si durante el sprint se tomó alguna decisión distinta de lo planeado, documentarla.
- `docs/OPEN_QUESTIONS.md`:
  - Confirmar el estado de las 5 decisiones bloqueantes (ya marcadas RESUELTAS en Paso 1).
  - Añadir los hallazgos del audit de historial si hubo credenciales reales expuestas (no el valor, sólo el hecho): linkear a `docs/SECURITY_AUDIT.md`.
  - Si surgió alguna laguna nueva durante Sprint 0 (p. ej. algo que Erik descubre al inspeccionar `data/oee.db`), documentarla en "preguntas que quedan abiertas".

**Archivos tocados**: `docs/MARK_III_PLAN.md`, `docs/OPEN_QUESTIONS.md`.

**Criterio de hecho**:
- Ambos docs referencian los hashes de los 13 commits del sprint.
- `git log --oneline feature/Mark-III` es consistente con lo documentado en los docs.

---

## Gates duros

### GATE 1 — Audit de historial (tras commit 1)

- Si `docs/SECURITY_AUDIT.md` lista credenciales expuestas con estado "expuesta, rotar cuando sea viable", el skill **para**, presenta el informe completo al operador y pregunta cómo proceder.
- **NO se ejecuta `git filter-repo` bajo ninguna circunstancia dentro de este sprint.** Esa decisión se toma fuera de `/gsd-execute-phase 1`.
- Si el informe está limpio (0 hallazgos) o el operador da visto bueno explícito, avanzar al commit 2.

### GATE 2 — Integridad de arranque (tras cada commit del 2 al 13)

Después de cada commit, el skill ejecuta (en backgroud o interactivo):

```
make build                       # Dockerfile sigue siendo válido
docker compose config            # compose YAML sigue siendo válido
make up && sleep 5 && make health   # /api/health responde OK
make down
```

- Si falla el build → revertir el commit, reportar el error, parar.
- Si falla el health check → idem.
- Los tests (`pytest`) NO son parte del GATE 2 (son no bloqueantes en Sprint 0).

---

## Lo que NO entra en esta fase

- **Auth** (login, sesiones, RBAC, roles, departamentos) → Phase 2 / Sprint 1.
- **Audit_log** (tabla + middleware) → Phase 2 / Sprint 1.
- **Refactor de capa de datos** (repositorios, schema_guard, engines separados) → Phase 3 / Sprint 2.
- **Estructura `modules/` nueva** → Phase 3 / Sprint 2.
- **Rename de la carpeta `OEE/`** → Phase 3 / Sprint 2.
- **Refactor interno de los 4 módulos OEE** (`disponibilidad`, `rendimiento`, `calidad`, `oee_secciones`) → Mark-IV.
- **Rotación de credenciales SQL Server** → diferida hasta revisar `docs/SECURITY_AUDIT.md`.
- **Reescritura de historial git** (`filter-repo`) → fuera de scope; decisión manual del operador.
- **SMTP operativo** → Out of Scope Mark-III.
- **Compra de dominio público / exposición internet** → Out of Scope (LAN-only).
- **Rename del repo GitHub `analisis_datos`** → Out of Scope.
- **Unificación de `data/ecs-logo.png` con `static/img/brand/ecs/logo.png`** → Phase 3 / Sprint 2.
- **Pre-commit hooks** → Phase 7 / Sprint 6.
- **Cobertura de tests ≥ 60%** → Phase 7 / Sprint 6.
- **2FA, LDAP** → Mark-IV+.

---

## Criterios de verificación (para `VERIFICATION.md` al cierre)

El skill GSD generará `VERIFICATION.md` en este mismo directorio tras
completar los 13 commits. Debe cubrir:

1. **Los 13 commits** con hash, resumen de cambios y resultado (✓ / ✗).
2. **Estado del audit de historial**: resumen (limpio / N hallazgos), referencia a `docs/SECURITY_AUDIT.md`.
3. **Decisión tomada sobre `data/oee.db`** con justificación: borrado directo vs. backup offline + razón.
4. **`make dev` arranca OK** tras los 13 commits (verificación post-sprint). `/api/health` devuelve OK.
5. **Suite de tests actual**: passing / failing con listado de los que fallan. Los tests no se modifican en Sprint 0.
6. **Workflow CI**: YAML válido; primer run en GitHub Actions referenciado (Erik lo verifica).
7. **Hallazgos fuera del alcance**: cosas descubiertas durante Sprint 0 que no se abordaron y se proponen para sprints posteriores.
8. **Desviaciones del PLAN** con justificación: si algún commit tuvo que partirse, reordenarse o ajustarse.

---

## Reglas de comportamiento durante ejecución

- **Español peninsular**, directo, sin fluff, en commit messages en inglés (título) + español (body si aporta contexto).
- **Commits atómicos**: un commit = un cambio coherente.
- **No tocar lógica de negocio.** Ninguna función de cálculo OEE se toca. Ningún router hace lo que no hacía antes (salvo el exception handler, que es fuga).
- **Si algo del plan entra en conflicto con el código real, parar y preguntar.** Mejor desbloquear manualmente que improvisar.
- **Hallazgos fuera del alcance**: reportar pero NO arreglar sin consultar, salvo que sea trivial y del mismo tipo que ya estás arreglando (ejemplo: otro handler que también filtra traceback → arreglar en el mismo commit 8).
- **No `--no-verify`**, no `--force`, no `filter-repo`.
- **Verificar sintaxis** después de cada rename de env vars: import Python, render de templates, validez de YAML de compose. No basta `sed` a ciegas.
- **Tras cada commit**, verificar GATE 2 (arranque). Si rompe, revertir **antes** del siguiente commit.

---

## Estimación

2-3 días de trabajo focalizado. Commits cortos (1, 2, 3, 4, 9) son de
minutos; commits largos (6, 7, 10) son de horas cada uno.

---

*PLAN escrito: 2026-04-18. Fase 1 de 7 del milestone Mark-III.*
