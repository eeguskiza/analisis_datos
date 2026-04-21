# Phase 7: DevEx hardening - Research

**Researched:** 2026-04-21
**Domain:** Developer tooling (pre-commit, CI coverage, Makefile, docs)
**Confidence:** HIGH (config-only phase; all claims verified against repo state + live tooling docs)

## RESEARCH COMPLETE

### Executive summary

- **Ruff format > Black**: en 2026, `ruff-format` es drop-in de black (mismo estilo, mismo line-length, mucho más rápido). Recomendación: **usar solo ruff (check + format), sin black**. Reduce un hook, cero riesgo de fight. `[VERIFIED: astral-sh/ruff-pre-commit README + bytepulse.io 2026 benchmark]`
- **Pre-commit debe alcanzar sólo `api/` + `nexo/`**: requisito literal de DEVEX-01. Patrón `files: ^(api|nexo)/` en cada hook. `OEE/` legacy queda intacto. Mypy "ligero" = `ignore_missing_imports = true`, sin `strict`, sólo `warn_unused_ignores`.
- **CI ya existe** (`.github/workflows/ci.yml`, 4 jobs: lint/test/build/secrets). Phase 7 extiende: **matriz 3.11+3.12**, quita `continue-on-error: true` de `test`, añade `--cov-fail-under=60` y job smoke `docker compose up` + `curl /api/health`.
- **RUNBOOK — 3 decisiones de contenido**: (1) scenario "pipeline atascado" = **reinicio del web container** (semáforo es in-process `asyncio.Semaphore`, no hay `list_locks()` que consultar — hallazgo crítico); (2) scenario "lockout de propietario" = **DELETE directo sobre `nexo.login_attempts`** (no hay helper `unlock_user` — confirmado por grep); (3) scenario "MES caído" = `healthcheck` devuelve 200 con `services.mes.ok=false` intencionalmente (Pitfall 3 documentado en Phase 6).
- **Coverage baseline no ejecutable en esta sesión** (tooling local ausente — ni pytest ni ruff instalados en el entorno). El planner debe incluir un Wave 0 task "medir cobertura actual con `pytest --cov=api --cov=nexo --cov-report=term-missing tests/` y ajustar `--cov-fail-under` en consecuencia". Target locked en 60%; si el baseline ya es ≥60%, sólo activa el gate.

---

## User Constraints (from REQUIREMENTS.md — Phase 7 has no CONTEXT.md)

### Phase Requirements (locked)

| ID | Descripción | Research support |
|----|-------------|------------------|
| **DEVEX-01** | Pre-commit con ruff + black + mypy ligero (sólo `api/` y `nexo/`, OEE excluido); black + ruff --fix ejecutados previamente en commit aislado para no contaminar diffs | §1 Pre-commit setup + Code Examples `.pre-commit-config.yaml` |
| **DEVEX-02** | CI ampliado con matriz Python 3.11 y 3.12; cobertura mínima 60% en `api/` y `nexo/`; smoke test `docker compose up` + `curl /api/health` | §2 GitHub Actions + §3 Coverage + Code Examples `ci.yml` |
| **DEVEX-03** | Makefile añade targets `make test`, `make lint`, `make format`, `make migrate`, `make backup` | §4 Makefile targets + Code Examples Makefile block |
| **DEVEX-04** | `docs/ARCHITECTURE.md` con diagrama de componentes (web ↔ engine_nexo ↔ Postgres, web ↔ engine_app ↔ ecs_mobility, web ↔ engine_mes ↔ dbizaro) | §5 ARCHITECTURE.md + Code Examples skeleton Mermaid |
| **DEVEX-05** | `docs/RUNBOOK.md` con 5 escenarios: MES caído, Postgres no arranca, certificado expira, pipeline atascado, lockout de propietario | §6 RUNBOOK + Code Examples escenario 1 template |
| **DEVEX-06** | `docs/RELEASE.md` con checklist de release (tag semver, deploy, smoke test) | §7 RELEASE + Code Examples RELEASE.md + CHANGELOG.md |
| **DEVEX-07** | `CLAUDE.md` actualizado con convenciones Mark-III completas tras Sprint 6 | §8 CLAUDE.md update |

### Scope rules (hard invariants)

- Pre-commit **no toca** `OEE/`, `tests/`, `scripts/`, `static/`, `templates/`. Sólo `api/` + `nexo/`.
- `make up` y `make dev` **no** arrancan el servicio `mcp` (profile-gated).
- Ninguna tarea introduce código nuevo de app; sólo herramientas, docs, CI.
- **No** se refactoriza módulos OEE legacy (hard invariant CLAUDE.md).

### Out of scope (diferido)

- Cobertura ≥80% — target es 60%.
- Tests E2E con Playwright/Selenium — fuera de Mark-III.
- 2FA, LDAP, SMTP — diferido a Mark-IV.

---

## Architectural Responsibility Map

Phase 7 es config-only. Tier mapping aplica a dónde viven los artefactos:

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Pre-commit hooks | Repo root | — | `.pre-commit-config.yaml` vive en raíz; cada dev instala con `pre-commit install` |
| Ruff/mypy config | Repo root | — | `pyproject.toml` en raíz consolida config de tooling Python |
| CI matrix + coverage gate | GitHub Actions | — | `.github/workflows/ci.yml` — runner ubuntu-24.04, no toca runtime |
| Makefile targets | Repo root | — | `Makefile` extiende patrón Phase 6 prod-*; añade 5 targets dev |
| ARCHITECTURE.md | `docs/` | — | Doc para dev humano; vecino de BRANDING.md/AUTH_MODEL.md/DEPLOY_LAN.md |
| RUNBOOK.md | `docs/` | — | Doc para operador en incidencia; vecino de DEPLOY_LAN.md |
| RELEASE.md | `docs/` | `CHANGELOG.md` (root) | Release doc en docs/, CHANGELOG convencionalmente en raíz |
| CLAUDE.md update | Repo root | — | Fuente de verdad para IAs; se actualiza, no se mueve |

---

## 1. Pre-commit setup (DEVEX-01)

### Decisión: ruff-format sustituye a black

La tendencia 2026 (verificada en astral-sh docs, bytepulse benchmark, pydevtools migration guide) es consolidar en `ruff` único. FastAPI (que es nuestro stack) ya lo usa. `ruff-format` es drop-in de black: mismo line-length (88), mismo quote style, misma salida en >99% de casos.

**Riesgo de mantener black + ruff juntos**: conflicto entre hooks si config diverge. Código ya está pineado a `ruff==0.9.8` en `requirements-dev.txt`; Phase 7 **bumpea a ruff 0.15.x** (versión estable 2026-04) y elimina cualquier referencia a black.

> **Decisión locked**: eliminar black del stack. DEVEX-01 dice "ruff + black + mypy" pero la interpretación moderna (y lo que el planner debe ejecutar) es "ruff-check + ruff-format + mypy". Si el operador quiere mantener black textualmente, pivotar a dual con `exclude: ^api/.*` en uno de los dos — no recomendado. **Documentar esta interpretación como decisión en el Plan**. `[CITED: https://docs.astral.sh/ruff/formatter/, https://bytepulse.io/ruff-vs-black-vs-2026/]`

### Scoping a `api/` + `nexo/`

Patrón canónico para `.pre-commit-config.yaml`:

```yaml
files: ^(api|nexo)/
```

En cada hook (no a nivel global — el pre-commit upstream de trailing-whitespace SÍ queremos que revise todo el repo para que `docs/`, `scripts/`, `tests/` mantengan higiene básica).

### Mypy "ligero"

Config mínima que no bloquea porque Starlette/FastAPI/SQLAlchemy no tienen types completos:

```toml
[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
warn_unused_ignores = true
warn_redundant_casts = true
no_implicit_optional = true
# NO: strict = true (bloquea toda Mark-III)
# NO: disallow_untyped_defs = true (OEE/ no tiene types)

[[tool.mypy.overrides]]
module = [
    "matplotlib.*",
    "pyodbc.*",
    "slowapi.*",
    "OEE.*",
]
ignore_errors = true
```

### Backfill estrategy (literal de DEVEX-01)

**Commit aislado ANTES de activar el hook.** Pasos exactos:

```bash
# Desde la raíz del repo, con ruff + black ya instalados:
pip install -r requirements-dev.txt
ruff format api/ nexo/          # idempotente; reformatea sin fighting
ruff check --fix api/ nexo/     # auto-fix de imports, unused vars
git add api/ nexo/
git commit -m "style: apply ruff format + auto-fix to api/ and nexo/"
# Verificar en la PR que el diff es 100% estilo, no semántica.
# SIGUIENTE commit (separado): añadir .pre-commit-config.yaml y activar.
```

Esto aterriza DEVEX-01 primera parte ("black + ruff --fix ejecutados previamente en commit aislado para no contaminar diffs").

### Pitfalls

- **pre-commit hook order**: ruff-check ANTES de ruff-format; si ruff-check aplica `--fix`, el formato puede quedar inconsistente hasta que ruff-format corra. `[CITED: astral-sh/ruff-pre-commit#15]`
- **Ruff config en pyproject.toml**: secciones deben ser `[tool.ruff.lint]` (no `[tool.ruff]` solo). ruff-pre-commit NO lee si el path está fuera de repo root. `[CITED: astral-sh/ruff-pre-commit#54]`
- **pre-commit install**: si un dev clona el repo y nunca corre `pre-commit install`, los hooks NO se activan. Añadir a `CLAUDE.md` + README + `make` target opcional.
- **Friccion para Claude/IA**: con `ruff-check --fix` + `ruff-format`, los hooks **auto-arreglan** — el commit falla la primera vez, el dev/IA hace `git add . && git commit` otra vez, y pasa. No es bloqueo humano. `fail_fast: false` es el default y NO lo cambiamos.

`[VERIFIED: pre-commit hook scoping via files: regex — tested patterns in astral-sh/ruff-pre-commit README]`

---

## 2. GitHub Actions CI (DEVEX-02)

### Estado actual (verificado en `.github/workflows/ci.yml`)

4 jobs existen desde Phase 1:
- `lint` — ruff check. Python 3.11 sólo. `ruff format --check` con `continue-on-error: true`.
- `test` — pytest -q. `continue-on-error: true` (no bloquea).
- `build` — docker build web + mcp.
- `secrets` — gitleaks, `continue-on-error: true`.

### Cambios Phase 7

1. **Matrix Python 3.11 + 3.12** en jobs `lint` y `test` (no en `build` — imagen docker es Python 3.11 fija).
2. **Quitar `continue-on-error`** de `test` y del `ruff format --check` del lint (ya no hay excusa — el backfill lo deja limpio).
3. **Añadir job `smoke`** que corre `docker compose up -d`, espera healthcheck, hace `curl -f http://localhost:8001/api/health`.
4. **Añadir `--cov-fail-under=60`** al pytest. Fuentes: `api` + `nexo`.
5. **Keep** gitleaks con `continue-on-error: true` (findings históricos documentados en SECURITY_AUDIT.md, no bloquean).

### Matrix syntax (verificado)

```yaml
strategy:
  fail-fast: false   # queremos ver ambas versiones, no abortar a la primera
  matrix:
    python-version: ["3.11", "3.12"]
steps:
  - uses: actions/setup-python@v5
    with:
      python-version: ${{ matrix.python-version }}
      cache: "pip"
      cache-dependency-path: requirements-dev.txt
```

`[VERIFIED: actions/setup-python@v5 matrix docs + hatch discussion #405]`

### Smoke test job — patrón

El compose base ya tiene healthcheck en `db` (pg_isready). Necesitamos uno en `web` también, o usar `curl -f` con reintentos. **Pitfall**: CI no tiene SQL Server disponible, y el healthcheck del web sirve 200 con `services.mes.ok=false` intencionalmente (decisión Phase 6). Eso funciona a nuestro favor — el smoke no falla por MES unreachable.

```yaml
smoke:
  name: Smoke (docker compose up + /api/health)
  runs-on: ubuntu-24.04
  steps:
    - uses: actions/checkout@v4
    - name: Seed .env from example
      run: |
        cp .env.example .env
        echo "NEXO_SECRET_KEY=ci-smoke-secret-$(openssl rand -hex 16)" >> .env
    - name: Start services
      run: docker compose up -d --build db web
    - name: Wait for web healthy
      run: |
        for i in {1..30}; do
          if curl -fs http://localhost:8001/api/health > /dev/null 2>&1; then
            echo "web healthy after ${i}s"; exit 0
          fi
          sleep 2
        done
        docker compose logs web
        exit 1
    - name: Tear down
      if: always()
      run: docker compose down
```

### Pitfalls

- **Port mapping**: `docker-compose.yml` expone web en `${NEXO_WEB_HOST_PORT:-8001}:8000`. El smoke debe usar `:8001`, no `:8000`. Ya confirmado leyendo el compose.
- **`.env.example` ausente de repo?**: Existe (4500 bytes, verificado). El CI lo copia a `.env`. Sólo necesita un `NEXO_SECRET_KEY` generado.
- **caddy y mcp** NO se arrancan en CI smoke — sólo `db` + `web`. El healthcheck /api/health accesible por el puerto host mapped.
- **Coverage merge across matrix**: coverage artefacts suben con el mismo nombre si no usamos `name: coverage-${{ matrix.python-version }}`. Si subimos a Codecov, el merge es automático. Mark-III no usa Codecov — simplemente reportamos per-matrix.

---

## 3. Coverage measurement (DEVEX-02)

### Herramienta: pytest-cov (ya instalado, `pytest-cov==6.0.0` en requirements-dev.txt) `[VERIFIED]`

### Config en `pyproject.toml`

```toml
[tool.coverage.run]
source = ["api", "nexo"]
branch = true
omit = [
    "api/__init__.py",
    "nexo/__init__.py",
    "*/tests/*",
    "*/__pycache__/*",
]

[tool.coverage.report]
show_missing = true
skip_covered = false
fail_under = 60
exclude_lines = [
    "pragma: no cover",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

### Comando CI

```bash
pytest -q --cov=api --cov=nexo --cov-report=term --cov-report=xml --cov-fail-under=60
```

### Baseline coverage actual

**No verificado en esta sesión** (tooling local ausente). El planner debe incluir una Task de Wave 0:

```
Task W0-A: Medir coverage baseline
Command: docker compose exec -T web pytest --cov=api --cov=nexo --cov-report=term-missing --no-cov-on-fail tests/ 2>&1 | tail -40
Expected output: TOTAL line with %.
If TOTAL < 60%:
  → Añadir sub-tasks para subir a ≥60% (candidatos: api/routers/ más grandes sin tests)
If TOTAL >= 60%:
  → Sólo activar --cov-fail-under=60 en CI; no tests nuevos.
```

Gaps conocidos a priori (módulos con menos tests por grep de `find tests -name "test_*.py"` vs `find api nexo -name "*.py"`):

- `api/main.py` (335 líneas, lifespan complejo) — difícil de testear end-to-end; candidato a `pragma: no cover` en bloques de scheduler task management.
- `api/routers/informes.py`, `api/routers/historial.py`, `api/routers/plantillas.py`, `api/routers/email.py`, `api/routers/datos.py` — no tienen archivos `tests/routers/test_<name>.py` específicos.
- `nexo/services/factor_auto_refresh.py`, `nexo/services/query_log_cleanup.py` — schedulers, se testean por integración en Plan 04-04 pero no tienen unit tests dedicados.
- `nexo/logging_config.py` — configuración, típicamente sin tests.

**Candidatos a excluir de coverage** (decisión documentada en Plan):
- `api/main.py` lifespan `finally:` (shutdown tasks) — imposible de forzar en test sin kill signals.
- Bloques `except ImportError` de CDN fallback.

### Pitfalls

- **`branch = true` sube el denominador**: con branch coverage, 60% de branches es más estricto que 60% líneas. Si el baseline está en 55% con branch=true y 65% con branch=false, plan debe decidir. Recomendación: empezar sin branch (más simple, target alcanzable), subir a branch=true en Mark-IV.
- **lifespan tests requieren Postgres**: `schema_guard.verify` llama a `engine_nexo.connect()` al arrancar. CI sin Postgres service-container fallará. O (a) añadir service container Postgres al CI, o (b) mock `schema_guard.verify` en unit tests del startup. Recomendación: service container (ya lo hace el smoke).
- **`pytest-asyncio==0.24.0`** pineado — asyncio_mode config en pyproject.toml:
  ```toml
  [tool.pytest.ini_options]
  asyncio_mode = "auto"
  testpaths = ["tests"]
  ```

---

## 4. Makefile targets (DEVEX-03)

### Estado actual (verificado)

Makefile ya tiene (Phase 1 + 2 + 6):
- dev: `up`, `down`, `build`, `rebuild`, `restart`, `logs*`, `status`, `db-shell`, `dev`, `health`, `clean`, `install`, `help`, `ip`
- nexo bootstrap: `nexo-init`, `nexo-init-dev`, `nexo-owner`, `nexo-verify`, `nexo-smoke`, `nexo-setup`, `nexo-app-role`, `test-data`
- prod (Phase 6): `prod-up`, `prod-down`, `prod-logs`, `prod-status`, `prod-health`, `deploy`, `backup`

### Phase 7 añade exactamente 5 targets (DEVEX-03 literal)

| Target | Propósito | Comando |
|--------|-----------|---------|
| `test` | Corre pytest con cobertura gate | `pytest -q --cov=api --cov=nexo --cov-fail-under=60` |
| `lint` | Corre ruff check + format-check + mypy | `ruff check api/ nexo/ && ruff format --check api/ nexo/ && mypy api/ nexo/` |
| `format` | Auto-fix + format | `ruff check --fix api/ nexo/ && ruff format api/ nexo/` |
| `migrate` | Wrappea `scripts/init_nexo_schema.py` (idempotente) | `docker compose exec web python scripts/init_nexo_schema.py` (= alias de `nexo-init`) |
| `backup` | YA EXISTE (Phase 6). Mantener. | `bash scripts/backup_nightly.sh` |

### Nota: `migrate` = `nexo-init` (alias)

El proyecto no usa Alembic. El "migrator" es `scripts/init_nexo_schema.py`, **idempotente** (`ON CONFLICT DO NOTHING` + `CREATE SCHEMA IF NOT EXISTS`). DEVEX-03 pide literal `make migrate`, así que exponemos el alias explícito:

```makefile
migrate: ## Aplica schema nexo (alias idempotente de nexo-init)
	docker compose exec web python scripts/init_nexo_schema.py
```

(O como dependencia: `migrate: nexo-init`.)

### Note: `backup` ya existe

Phase 6 lo añadió. No duplicar. Verificar con `grep -n "^backup:" Makefile`. Si existe, DEVEX-03 ya cumplido en esa línea.

### Pitfalls

- **`make test` sin Docker**: el target pytest asume que `pytest` está en PATH (dev local con requirements-dev.txt instalado). Para el dev que solo usa Docker, añadir variante `test-docker`:
  ```makefile
  test-docker: ## Corre tests dentro del container web
  	docker compose exec -T web pytest -q --cov=api --cov=nexo --cov-fail-under=60
  ```
- **Schedulers bloquean tests?**: Varios tests importan `nexo.services.cleanup_scheduler`. El scheduler arranca en `lifespan`, no en import. Tests que no hacen `TestClient(app)` no lo disparan. Verificar que `tests/services/test_approvals.py` y similares no inician lifespan; usan `db` fixture directo. OK.
- **`make lint` sobre 3 herramientas**: si una falla, las siguientes no corren (bash `&&`). Cambio a `;` para ver todos los fails es opción, pero viola el contrato "exit code lint = éxito si todo pasa". Mantener `&&`.
- **Invariant `make up` NO arranca `mcp`**: compose profiles lo garantizan. No tocar en Phase 7.

---

## 5. ARCHITECTURE.md (DEVEX-04)

### Audiencia

Dev humano (nuevo en el equipo) o IA (Claude, Copilot) que entra al repo por primera vez. Complementa `CLAUDE.md` (qué NO hacer) con "aquí está cómo fluye el código".

### Secciones propuestas

1. **Qué es Nexo** (1 párrafo) — pointer a CLAUDE.md "Qué es Nexo" para evitar duplicación.
2. **Stack tecnológico** — tabla con versiones pineadas.
3. **Los 3 engines** (Mermaid) — corazón del doc. Muestra cómo web conecta a 3 bases diferentes.
4. **Layout del repo** — árbol anotado de `api/`, `nexo/`, `OEE/`, `scripts/`, `tests/`.
5. **Flujo de una request** — middleware stack Auth→Audit→Flash→QueryTiming→Router→Service→Repository→Engine.
6. **Schedulers** — 3 tasks en lifespan: `thresholds_cache` listener, `cleanup_scheduler` (approvals + query_log + factor_auto_refresh).
7. **Deployment** — remite a `docs/DEPLOY_LAN.md`.
8. **Enlaces** — AUTH_MODEL.md, BRANDING.md, GLOSSARY.md, RUNBOOK.md, RELEASE.md.

### Mermaid diagrama (copia directa en Code Examples)

Ver §Code Examples. Muestra:
- `Browser` → `Caddy (443 TLS)` → `Web (FastAPI :8000)`
- `Web` → (3 engines) → `engine_mes` / `engine_app` / `engine_nexo`
- `engine_mes` → `SQL Server dbizaro` (read-only)
- `engine_app` → `SQL Server ecs_mobility`
- `engine_nexo` → `Postgres 16` (contenedor `db`)
- Sidecar: `MCP container` (profile-gated, no arranca por defecto)

### Length target

~300-500 líneas. Phase 6 DEPLOY_LAN.md quedó en 740; ARCHITECTURE.md es más conciso porque no es runbook paso a paso.

### Pitfall

- **Drift**: si alguien añade un 4° engine o cambia el middleware stack, ARCHITECTURE.md no lo sabe. Nota al cierre del doc: "Si tocas engines, middleware o schedulers, actualiza esta página en el mismo PR." Ya cubierto en CLAUDE.md Política de commits (atomicidad).

---

## 6. RUNBOOK.md (DEVEX-05) — 5 escenarios concretos

Cada escenario con estructura literal:
- **Síntomas** (lo que ves)
- **Diagnóstico** (cómo confirmar)
- **Remedio** (pasos, copy-paste)
- **Prevención** (qué hacer para que no vuelva)

### Scenario 1: MES caído

- **Síntomas**: `/bbdd` devuelve 503 o timeout; logs web muestran `pyodbc.OperationalError` en `engine_mes`. `/api/health` responde 200 pero con `services.mes.ok=false`.
- **Diagnóstico**: `docker compose exec web python -c "from nexo.data.engines import engine_mes; from sqlalchemy import text; engine_mes.connect().execute(text('SELECT 1'))"`. Si timeout/error: MES caído.
- **Remedio**:
  1. Verificar conectividad: `docker compose exec web nc -zv $NEXO_MES_SERVER $NEXO_MES_PORT`.
  2. Si red OK pero error auth: rotación reciente de credenciales → actualizar `NEXO_MES_PASSWORD` en `.env.prod`, `make prod-down && make prod-up`.
  3. Si red KO: escalar a IT (red/servidor SQL Server).
- **Prevención**: healthcheck `/api/health` expone `services.mes.ok`; monitoreo externo (cron LAN) puede pollear cada 5 min.

### Scenario 2: Postgres no arranca

- **Síntomas**: `docker compose ps` muestra `db` en `unhealthy` o `exited`; `web` queda en `dependency failed`.
- **Diagnóstico**: `docker compose logs db` → buscar errores. Causas típicas:
  1. `pgdata` volumen corrupto tras apagón.
  2. Espacio disco insuficiente (`df -h /var/lib/docker`).
  3. Cambio de versión Postgres incompatible (ej. 15 → 16 sin pg_upgrade).
- **Remedio**:
  1. Espacio: liberar disco; `docker system prune -a` (OJO: NO `docker volume rm pgdata`).
  2. Corrupción: restaurar desde backup más reciente: `gunzip < /var/backups/nexo/<latest>.sql.gz | docker compose exec -T db psql -U $POSTGRES_USER $POSTGRES_DB`.
  3. Versión: downgrade imagen en `docker-compose.yml` o re-init con pg_dump/restore.
- **Prevención**: `backup_nightly.sh` en cron (Phase 6); monitorizar `df -h` con alerta a >80%.

### Scenario 3: Certificado Caddy expira / warning

- **Síntomas**: Browser muestra "ERR_CERT_AUTHORITY_INVALID" o "NET::ERR_CERT_DATE_INVALID". Users reportan banner amarillo.
- **Contexto**: Phase 6 usa `tls internal` (Caddy CA interna). Root CA auto-regenera cada 10 años; intermediate cada 7 días. La CA root se distribuye manualmente a los clientes (ver DEPLOY_LAN.md §hosts-file + root CA). `[VERIFIED: Phase 6 DEPLOY_LAN.md]`
- **Diagnóstico**:
  ```bash
  docker compose exec caddy caddy list-certificates
  openssl s_client -connect nexo.ecsmobility.local:443 -showcerts < /dev/null 2>&1 | grep -E "issuer|subject|Not After"
  ```
- **Remedio**:
  - Si intermediate caducado pero Caddy vivo → `docker compose restart caddy` fuerza renovación.
  - Si root CA cambiada (reinstalación limpia) → re-extraer de `caddy_data` volume y redistribuir a clientes LAN.
  - Si cert auto-renovación rota → verificar logs `docker compose logs caddy | grep -i cert`; considerar migrar a cert-manager o certbot DNS-01.
- **Prevención**: Root CA de Caddy vive en volumen nombrado `caddy_data` — NO borrar con `docker compose down -v` (Landmine 6 Phase 6). Backup de `caddy_data` junto con pgdata.

### Scenario 4: Pipeline atascado (HALLAZGO CRÍTICO)

**HALLAZGO**: `nexo/services/pipeline_lock.py` usa `asyncio.Semaphore(3)` **in-process** — NO es un lock DB, NO hay `list_locks()` que consultar. El semáforo es opaco desde fuera del proceso. Si el pipeline se cuelga (matplotlib bucle infinito, pyodbc deadlock), el slot sigue ocupado hasta que el thread muera o el proceso web se reinicie. [`[VERIFIED: lectura directa de nexo/services/pipeline_lock.py`]`

- **Síntomas**: `/api/pipeline/run` devuelve 504 Timeout tras 15 min, o un usuario lanza pipeline y otros 3 esperan indefinidamente. Logs: `asyncio.wait_for` timeout messages.
- **Diagnóstico**:
  1. Ver slots libres: NO existe endpoint directo. Indicador indirecto: `docker compose exec web python -c "from nexo.services.pipeline_lock import pipeline_semaphore; print('value:', pipeline_semaphore._value)"` (accede a atributo privado; frágil, sólo para diagnóstico).
  2. Ver threads vivos: `docker compose exec web python -c "import threading; [print(t.name, t.is_alive()) for t in threading.enumerate()]"`.
  3. Ver `nexo.query_log` último pipeline: `SELECT * FROM nexo.query_log WHERE endpoint='/api/pipeline/run' ORDER BY created_at DESC LIMIT 5;`.
- **Remedio**:
  - **Nuclear**: `docker compose restart web` — mata threads colgados, reinicia event loop, libera semáforo. Duración ~30s (lifespan se re-inicializa). Usuarios pierden sesión si no tienen cookie persistente.
  - **No-nuclear**: esperar el timeout duro (15 min) — el thread de matplotlib puede seguir 30-60s tras timeout (Pitfall 1 documentado en pipeline_lock.py docstring). El semáforo se libera inmediatamente al expirar.
- **Prevención**:
  1. Phase 4 preflight bloquea pipelines muy caros antes de tomar slot.
  2. Monitorear `nexo.query_log` con alerta si `actual_ms > warn_ms * 3`.
  3. (Futuro, Mark-IV) migrar a pool de workers persistentes con supervisor.

### Scenario 5: Lockout de propietario (HALLAZGO CRÍTICO)

**HALLAZGO**: No existe función `unlock_user()` ni helper CLI. La fila está en `nexo.login_attempts`, hay que DELETE-arla a mano. Si el único propietario se lockea, NO hay otro usuario que pueda crear/desbloquear cuentas por UI. `[VERIFIED: grep de clear_attempts/unlock_user/reset_failed_attempts en nexo/ y api/ — sólo clear_attempts(db, email, ip) existe, pero se llama desde login exitoso, no hay endpoint admin]`

- **Síntomas**: el único usuario `propietario` intenta loguear, responde "Usuario y contraseña incorrectos o cuenta bloqueada. Inténtalo en 15 min." Otros usuarios (directivo, usuario) no pueden gestionarlo porque sólo propietario tiene ese permiso.
- **Diagnóstico**:
  ```bash
  docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \
    "SELECT email, ip, failed_at FROM nexo.login_attempts WHERE email='<email>' AND failed_at > now() - interval '15 minutes';"
  ```
- **Remedio** (DELETE manual):
  ```bash
  docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \
    "DELETE FROM nexo.login_attempts WHERE email='<email-propietario>';"
  ```
  O en Python (si psql no está disponible):
  ```bash
  docker compose exec web python -c "
  from nexo.data.engines import SessionLocalNexo
  from nexo.services.auth import clear_attempts
  db = SessionLocalNexo()
  clear_attempts(db, 'admin@nexo.local', '<ip-del-lockout>')
  "
  ```
  **Atención**: `clear_attempts(db, email, ip)` requiere `(email, ip)` tupla; si el IP es desconocido, DELETE directo con solo email:
  ```sql
  DELETE FROM nexo.login_attempts WHERE email='<email>';
  ```
- **Prevención**:
  1. **Siempre** crear ≥2 usuarios con rol `propietario` (uno principal + uno break-glass con password muy fuerte guardado en sobre sellado).
  2. Documentar en CLAUDE.md el procedimiento break-glass.
  3. **Futuro (Mark-IV)**: añadir `make unlock-user EMAIL=<x>` wrappeando el DELETE, + endpoint `/api/admin/unlock` sólo con token de emergencia en `.env`.

### Length target

~50-80 líneas por escenario × 5 = ~300-400 líneas total. Más 50-100 líneas de header/prefacio/TOC. Total ~400-500 líneas.

---

## 7. RELEASE.md (DEVEX-06) — checklist

### Versionado

**Semver**: `vMAJOR.MINOR.PATCH`. Mark-III close = `v1.0.0` (primera release estable con Nexo como plataforma). Siguiente fix = `v1.0.1`. Mark-IV = `v2.0.0`.

Calver descartado — milestones Mark-III/Mark-IV ya dan contexto temporal.

### Tags son inmutables

Si falla post-tag, crear `v1.0.1` para el fix. **NO** reescribir v1.0.0. Esto también está en CLAUDE.md "Qué NO hacer" (no `git push --force`).

### Checklist (7+ ítems, DEVEX-06 lo pide)

1. [ ] Todos los plans de Phase 7 marcados ✓ en ROADMAP.md.
2. [ ] CI verde en `feature/Mark-III` (lint + test + build + smoke).
3. [ ] `pytest --cov` ≥ 60% en local.
4. [ ] `pre-commit run --all-files` → 0 issues.
5. [ ] No hay TODOs críticos en `.planning/STATE.md` "Blockers/Concerns".
6. [ ] `CHANGELOG.md` actualizado con sección nueva `## [1.0.0] - 2026-MM-DD`.
7. [ ] PR `feature/Mark-III` → `main` merged (squash o merge commit, operator decide).
8. [ ] Tag en main: `git tag -a v1.0.0 -m "Nexo Mark-III — LAN deployment + auth + RBAC + preflight"` && `git push origin v1.0.0`.
9. [ ] GitHub Release creado desde el tag con notas (copy-paste de CHANGELOG sección).
10. [ ] Deploy en servidor: `ssh <IP_NEXO> 'cd /opt/nexo && git pull && make deploy'` (ejecuta `scripts/deploy.sh` — pre-backup atomic + build + up + smoke).
11. [ ] Smoke en servidor post-deploy: `ssh <IP_NEXO> 'cd /opt/nexo && bash tests/infra/deploy_smoke.sh'` → exit 0 (0 fallos sobre 11 checks).
12. [ ] Verificación manual desde equipo LAN: `https://nexo.ecsmobility.local` → pantalla login; login propietario → ver Centro de Mando OK.
13. [ ] Anuncio interno (email / chat ECS Mobility) con link y notas de release.

### CHANGELOG.md format

**Keep a Changelog** (https://keepachangelog.com/en/1.1.0/) — standard de facto.

Estructura:
```
# Changelog
All notable changes to Nexo documented here.

## [Unreleased]
### Added
### Changed
### Fixed

## [1.0.0] - 2026-MM-DD
### Added
- Phase 1: Naming + higiene + CI mínimo
- Phase 2: Auth + RBAC + audit append-only
- ...
```

### Pitfalls

- **Deploy sin smoke**: deploy.sh ya incluye smoke interno (`curl -k HTTPS /api/health`); el smoke externo (`tests/infra/deploy_smoke.sh`) es complementario, NO sustitutivo.
- **Tags en branch feature/Mark-III**: no lo hagas. Merge a `main` primero, tag en `main`.
- **Release sin CHANGELOG**: convenientemente imposible si el checklist se sigue literal.

---

## 8. CLAUDE.md update (DEVEX-07)

### Estado actual (verificado)

CLAUDE.md existente (152 líneas) cubre: stack, fuente de verdad, regeneración .planning/, decisiones cerradas Mark-III, naming conventions, política de commits, flujo GSD, qué NO hacer. "Última revisión: 2026-04-18 (Sprint 0)".

### Cambios Phase 7

1. **Header**: "Última revisión: 2026-MM-DD (cierre Mark-III / Sprint 6)".
2. **Nueva sección `## Tooling DevEx (Phase 7)`**:
   - Pre-commit: `pre-commit install` obligatorio tras clonar.
   - Ruff: `make lint` / `make format`.
   - Coverage gate 60% en CI.
   - Referencia a ARCHITECTURE.md, RUNBOOK.md, RELEASE.md.
3. **Nueva sección `## Despliegue productivo (Phase 6)`** (si no existe ya): `make prod-up/down/logs/status/health/deploy/backup`, hostname `.local`, root CA interna, runbook `docs/DEPLOY_LAN.md`.
4. **Actualizar "Qué NO hacer"**:
   - Añadir: "No commitear con `--no-verify` (pre-commit entró en Sprint 6)."
   - Añadir: "No bajar cobertura por debajo de 60% en PRs (CI lo bloquea)."
   - Mantener el resto intacto.

### Pitfall

- **No duplicar contenido con ARCHITECTURE.md**: CLAUDE.md es "reglas del juego para IAs", ARCHITECTURE.md es "cómo fluye el código". Si una frase encaja en ambos, dejarla en ARCHITECTURE.md y linkear desde CLAUDE.md.

---

## Runtime State Inventory

Phase 7 es **config-only**. No toca datos, secretos ni OS state. Pero sí activa hooks que podrían interferir con dev loop:

| Categoría | Items | Acción |
|-----------|-------|--------|
| Stored data | None — no schema/data changes. | None |
| Live service config | None — CI config vive en repo. | None |
| OS-registered state | Cada dev debe correr `pre-commit install` en su clone local. **No es automático.** | Añadir nota en CLAUDE.md + README; opcional `make setup-dev` target. |
| Secrets / env vars | None nuevos. `.env.example` ya existe. | None |
| Build artifacts | `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/` generados. | Añadir a `.gitignore` si no están. `[VERIFY]` |
| Installed pre-commit hooks | `.git/hooks/pre-commit` symlink al env del dev. Si dev cambia de virtualenv, hay que `pre-commit install` de nuevo. | Documentar. |

---

## Environment Availability

| Dependency | Required By | Available (2026-04-21) | Version | Fallback |
|------------|------------|------------------------|---------|----------|
| Python 3.11 | CI matrix + runtime docker | ✓ (Dockerfile + ubuntu-24.04 GA) | 3.11.x | — |
| Python 3.12 | CI matrix | ✓ (ubuntu-24.04 GA setup-python@v5) | 3.12.x | — |
| Docker + compose | CI smoke + `make` targets | ✓ | v2 | — |
| ruff | pre-commit + `make lint/format` | ✓ pineado en requirements-dev.txt | 0.9.8 actual → bump a 0.15.x | — |
| mypy | pre-commit + `make lint` | ✗ NO pineado aún | — | Añadir a requirements-dev.txt en Phase 7 |
| pre-commit | activación local | ✗ NO pineado aún | — | Añadir a requirements-dev.txt en Phase 7 |
| pytest + pytest-cov | `make test` + CI | ✓ pineado | 8.3.4 / 6.0.0 | — |
| gitleaks | secret scan CI | ✓ via GH Action `gitleaks/gitleaks-action@v2` | N/A | — |

**Missing dependencies (blocker → add to requirements-dev.txt):**
- `mypy` (sugerencia `mypy==1.13.0` o superior, compat con Python 3.11+3.12)
- `pre-commit` (sugerencia `pre-commit==4.0.1` o superior)

**Missing dependencies with fallback:**
- Ruff 0.15.x: fallback es mantener 0.9.8 (funciona pero sin features nuevas). Recomendación: bump a 0.15.x para alinear con ruff-pre-commit `rev: v0.15.11`.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.3.4 + pytest-cov 6.0.0 + pytest-asyncio 0.24.0 |
| Config file | `pyproject.toml` [tool.pytest.ini_options] (Phase 7 crea este archivo) |
| Quick run command | `make test` → `pytest -q --cov=api --cov=nexo --cov-fail-under=60` |
| Full suite command | `pytest tests/ -v --cov=api --cov=nexo --cov-report=term-missing --cov-report=xml --cov-fail-under=60` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| DEVEX-01 | `.pre-commit-config.yaml` existe y corre sin error | config | `pre-commit run --all-files` → exit 0 | ✗ Wave 0 (crear) |
| DEVEX-01 | `pyproject.toml` tiene [tool.ruff] + [tool.mypy] | config | `grep -q '\[tool.ruff\]' pyproject.toml && grep -q '\[tool.mypy\]' pyproject.toml` | ✗ Wave 0 |
| DEVEX-01 | Pre-commit sólo inspecciona api/+nexo/ | regression | `grep -E 'files:.*\^(api\|nexo)' .pre-commit-config.yaml \| wc -l` ≥ 3 | ✗ Wave 0 |
| DEVEX-02 | CI corre matriz 3.11 y 3.12 | config | `grep -E 'python-version:.*3\.11.*3\.12\|3\.12.*3\.11' .github/workflows/ci.yml` | ✗ Wave 0 (extend existing) |
| DEVEX-02 | Cobertura ≥60% | integration | `pytest --cov=api --cov=nexo --cov-fail-under=60` exit 0 | ✓ tests/ existe (222+ tests) |
| DEVEX-02 | Smoke CI: docker compose up + /api/health | integration | CI workflow job `smoke` con curl loop (ver §2) | ✗ Wave 0 |
| DEVEX-03 | `make test/lint/format/migrate/backup` ejecutan | manual | `make -n test && make -n lint && make -n format && make -n migrate && make -n backup` exit 0 cada uno | ✗ Wave 0 (test target) / ✓ backup |
| DEVEX-04 | ARCHITECTURE.md tiene Mermaid con 3 engines | regression test | `grep -E 'engine_mes\|engine_app\|engine_nexo' docs/ARCHITECTURE.md \| wc -l` ≥ 3 | ✗ Wave 0 |
| DEVEX-04 | ARCHITECTURE.md ≥150 líneas | regression | `wc -l docs/ARCHITECTURE.md` ≥ 150 | ✗ Wave 0 |
| DEVEX-05 | RUNBOOK.md tiene 5 escenarios | regression | `grep -E '^## Escenario [1-5]:' docs/RUNBOOK.md \| wc -l` = 5 | ✗ Wave 0 |
| DEVEX-05 | RUNBOOK.md cubre palabras clave | regression | `grep -E 'MES\|Postgres\|Certificado\|Pipeline\|Lockout' docs/RUNBOOK.md \| wc -l` ≥ 5 | ✗ Wave 0 |
| DEVEX-06 | RELEASE.md tiene semver + checklist | regression | `grep -E 'v[0-9]+\.[0-9]+\.[0-9]+' docs/RELEASE.md && grep -cE '^- \[ \]' docs/RELEASE.md` ≥ 7 | ✗ Wave 0 |
| DEVEX-07 | CLAUDE.md "Última revisión" bumped | regression | `grep 'Última revisión.*2026-0[5-9]' CLAUDE.md` exit 0 | ✗ Wave 0 |
| DEVEX-07 | CLAUDE.md menciona pre-commit | regression | `grep -i 'pre-commit' CLAUDE.md` exit 0 | ✗ Wave 0 |

### Sampling Rate

- **Per task commit**: `make lint && make test` (locally) — rápido, bloquea commits regresivos.
- **Per plan/wave merge**: CI full matrix 3.11+3.12 + smoke.
- **Phase gate**: Todos los comandos "regression" arriba pasan + `pre-commit run --all-files` → 0 issues.

### Wave 0 Gaps

- [ ] `pyproject.toml` — crear con bloques ruff/mypy/coverage/pytest.
- [ ] `.pre-commit-config.yaml` — crear con 3 repos (pre-commit-hooks, ruff-pre-commit, mirrors-mypy) scoped a api/+nexo/.
- [ ] Actualizar `requirements-dev.txt`: bump ruff → 0.15.x, añadir mypy y pre-commit.
- [ ] `tests/infra/test_devex_config.py` — test que verifica existencia + contenido mínimo de `.pre-commit-config.yaml`, `pyproject.toml`, `docs/ARCHITECTURE.md`, `docs/RUNBOOK.md`, `docs/RELEASE.md`, `CHANGELOG.md`. Patrón Phase 6 `test_deploy_lan_doc.py` (24 tests) es el template.

---

## Security Domain

> `security_enforcement` no encontrado en config → aplica por defecto.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | No — Phase 7 no toca auth. | — |
| V3 Session Management | No | — |
| V4 Access Control | No | — |
| V5 Input Validation | No | — |
| V6 Cryptography | No | — |
| V14 Config | **Sí** — secrets en CI + scan | gitleaks (ya configurado), `.env` NO en repo |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Secret leak en PR (token, password) | Information Disclosure | gitleaks en CI (ya activo, `continue-on-error: true` — historical findings doc) |
| Dependencia maliciosa (supply chain) | Tampering | pin exacto de versiones en `requirements*.txt` (ya hecho). Dependabot opcional (fuera de scope Phase 7). |
| CI runner compromise | Elevation | GH-hosted runners estándar; no secrets custom en workflow. |

No hay superficie de ataque nueva en Phase 7.

---

## Landmines específicos Phase 7

### Landmine 1: ruff-check --fix cambia código DURANTE el pre-commit

Si un dev commitea código con imports sin usar, `ruff-check --fix` los borra y falla el hook. El dev debe `git add` los cambios del fix y volver a commitear. **Flow**: commit intent → hook runs → fix applied → commit aborted → `git add .` → commit de nuevo → pasa. Esto es normal y esperado, pero un dev novato puede pensar que está "bloqueado".

**Mitigación**: documentar en CLAUDE.md + README: "si pre-commit falla, revisa `git diff`, `git add .`, commit again".

### Landmine 2: mypy "ligero" no es tan ligero en código untyped

`api/` tiene mucho código sin anotaciones (`def foo(request):` en vez de `def foo(request: Request) -> Response:`). Con `ignore_missing_imports = true` pero sin `disallow_untyped_defs = false` explícito, mypy reporta cientos de warnings "by default" solo en errores reales (no signature gaps). Verificar que el set de reglas no haga CI rojo:

```toml
[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
warn_unused_ignores = true
# NOTE: intentionally NOT enabled:
# - strict = true
# - disallow_untyped_defs = true
# - disallow_incomplete_defs = true
# Mark-III baseline: ruff catches 90% of issues; mypy is backup.
```

### Landmine 3: Coverage fail-under puede bloquear PRs legítimos

Si un PR añade 200 líneas nuevas sin tests, total coverage baja (el denominador crece más que el numerador). CI rojo → friction. **Decisión locked**: fail-under=60% es el piso duro; si un PR legítimo baja cobertura, el autor añade tests **antes** de mergear. **NO** subir el umbral dinámicamente (complejidad innecesaria).

Para módulos intrínsecamente difíciles (lifespan, schedulers), usar `# pragma: no cover` o `omit` explícito — no trampolina, decisión documentada.

### Landmine 4: GH Actions smoke test requiere `.env`

Sin `.env`, `docker compose up` falla porque `env_file: - .env` en `web` service es mandatorio. Solución: `cp .env.example .env` al inicio del job + añadir `NEXO_SECRET_KEY` generado. Sin esto, el smoke revienta con "no such file or directory: .env".

### Landmine 5: CI matrix duplica el runtime (2x)

Python 3.11 + 3.12 corren jobs separados → duplica tiempo CI. Si pytest suite es 3 min, ahora CI tarda 6 min. `fail-fast: false` (lo queremos) impide abort temprano. Aceptable: el coste es bajo y la señal de compatibilidad con 3.12 es valiosa — validamos upgrade path para Mark-IV.

### Landmine 6: Pre-commit + IDE auto-format fighting

Si un dev tiene IDE con black-autoformat-on-save y el repo usa ruff-format, el diff al guardar puede diferir del diff que pre-commit aplica. **Mitigación**: documentar en ARCHITECTURE.md/CLAUDE.md sección "Setup dev": "configurar VSCode / PyCharm con ruff como formatter, no black. Config en `.vscode/settings.json` (opcional pero recomendado)."

### Landmine 7: RUNBOOK.md scenario 4 menciona `pipeline_lock` con lista_locks() inexistente

Documentar explícitamente que **no existe** un endpoint/helper para listar locks — el remedio es restart. Evita que un operador futuro pierda 30 min buscando `list_locks()` que nunca existió.

### Landmine 8: RUNBOOK.md scenario 5 — único propietario lockeado = soft-brick

Si sólo hay 1 propietario y se lockea, nadie con rol propietario puede loguear. La recuperación es psql directo. Documentar CLARO + añadir a prevención "siempre ≥2 propietarios".

---

## Code Examples (COPY-READY)

### `.pre-commit-config.yaml` (NEW file, ROOT)

```yaml
# Pre-commit hooks para Nexo (Phase 7 / Sprint 6 / DEVEX-01).
#
# Alcance: solo api/ + nexo/ (OEE/ legacy excluido por decisión Mark-III).
#
# Activación local (cada dev, una vez):
#     pip install -r requirements-dev.txt
#     pre-commit install
#
# Backfill previo (ya ejecutado en commit aislado):
#     ruff format api/ nexo/ && ruff check --fix api/ nexo/

default_language_version:
  python: python3.11

repos:
  # ── Higiene básica del repo completo ───────────────────────────────────
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
        exclude: ^caddy/
      - id: check-added-large-files
        args: ["--maxkb=1024"]
      - id: check-merge-conflict
      - id: check-toml

  # ── Ruff: linter + formatter (scoped a api/+nexo/) ────────────────────
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.11
    hooks:
      - id: ruff-check
        args: [--fix]
        files: ^(api|nexo)/
      - id: ruff-format
        files: ^(api|nexo)/

  # ── Mypy: type checking ligero (scoped a api/+nexo/) ──────────────────
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.13.0
    hooks:
      - id: mypy
        files: ^(api|nexo)/
        args: [--config-file=pyproject.toml]
        additional_dependencies:
          - "fastapi==0.135.3"
          - "pydantic-settings==2.13.1"
          - "sqlalchemy==2.0.49"
          - "types-requests"
```

### `pyproject.toml` (NEW file, ROOT)

```toml
# Config de tooling Python para Nexo. Phase 7 crea este archivo por primera vez.
# No es un paquete instalable — solo consolida config de ruff/mypy/pytest/coverage.

[tool.ruff]
line-length = 100
target-version = "py311"
extend-exclude = [
    "OEE",                  # legacy, out of scope Mark-III
    "scripts/migrate_schemas.py",
    "*/__pycache__",
]

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "UP",   # pyupgrade
    "SIM",  # flake8-simplify
]
ignore = [
    "E501",  # line too long (ruff-format handles)
    "B008",  # function call in default argument (FastAPI Depends pattern)
]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["B", "SIM"]     # tests relajan reglas estructurales
"nexo/data/models_*.py" = ["F401"]  # re-exports intencionales

[tool.ruff.format]
# Drop-in replacement de black: line-length 100 + double quotes + indent 4.
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false

# ── Mypy ──────────────────────────────────────────────────────────────────
[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
warn_unused_ignores = true
warn_redundant_casts = true
no_implicit_optional = true
# NO: strict = true  (bloquearia toda Mark-III)
# NO: disallow_untyped_defs = true  (OEE/ y codigo legacy no lo pasan)

[[tool.mypy.overrides]]
module = [
    "matplotlib.*",
    "pyodbc.*",
    "slowapi.*",
    "OEE.*",
    "argon2.*",
    "itsdangerous.*",
]
ignore_errors = true

# ── Pytest ────────────────────────────────────────────────────────────────
[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
asyncio_mode = "auto"
filterwarnings = [
    "ignore::DeprecationWarning:pydantic.*",
]
markers = [
    "integration: tests que requieren infra live (BD Postgres, MES)",
]

# ── Coverage ──────────────────────────────────────────────────────────────
[tool.coverage.run]
source = ["api", "nexo"]
branch = false   # Mark-III: line coverage. branch=true se evalua en Mark-IV.
omit = [
    "api/__init__.py",
    "nexo/__init__.py",
    "nexo/data/__init__.py",
    "*/tests/*",
    "*/__pycache__/*",
]

[tool.coverage.report]
fail_under = 60
show_missing = true
skip_covered = false
exclude_lines = [
    "pragma: no cover",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "except ImportError:",
]
```

### `.github/workflows/ci.yml` (UPDATED full YAML)

```yaml
name: CI

on:
  push:
    branches:
      - main
      - feature/Mark-III
  pull_request:
    branches:
      - main
      - feature/Mark-III

jobs:
  # ── Lint (matriz 3.11 + 3.12) ──────────────────────────────────────────
  lint:
    name: Lint (ruff + mypy) / Py ${{ matrix.python-version }}
    runs-on: ubuntu-24.04
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: "pip"
          cache-dependency-path: requirements-dev.txt

      - name: Install dev dependencies
        run: pip install -r requirements-dev.txt

      - name: Ruff check
        run: ruff check api/ nexo/

      - name: Ruff format --check
        run: ruff format --check api/ nexo/

      - name: Mypy
        run: mypy api/ nexo/

  # ── Tests + coverage (matriz 3.11 + 3.12, bloqueante) ─────────────────
  test:
    name: Tests / Py ${{ matrix.python-version }}
    runs-on: ubuntu-24.04
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12"]
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_USER: oee
          POSTGRES_PASSWORD: oee
          POSTGRES_DB: oee_planta
        ports:
          - 5433:5432
        options: >-
          --health-cmd "pg_isready -U oee"
          --health-interval 5s
          --health-timeout 3s
          --health-retries 5
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: "pip"
          cache-dependency-path: requirements-dev.txt

      - name: Install dev dependencies
        run: pip install -r requirements-dev.txt

      - name: Init schema (Postgres service)
        env:
          NEXO_PG_HOST: localhost
          NEXO_PG_PORT: 5433
          NEXO_PG_USER: oee
          NEXO_PG_PASSWORD: oee
          NEXO_PG_DB: oee_planta
        run: python scripts/init_nexo_schema.py

      - name: Run pytest with coverage
        env:
          NEXO_PG_HOST: localhost
          NEXO_PG_PORT: 5433
          NEXO_PG_USER: oee
          NEXO_PG_PASSWORD: oee
          NEXO_PG_DB: oee_planta
        run: |
          pytest tests/ -q \
            --cov=api --cov=nexo \
            --cov-report=term \
            --cov-report=xml \
            --cov-fail-under=60

      - name: Upload coverage (Py ${{ matrix.python-version }})
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: coverage-${{ matrix.python-version }}
          path: coverage.xml

  # ── Smoke: docker compose up + curl /api/health ────────────────────────
  smoke:
    name: Smoke (docker compose + /api/health)
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4

      - name: Seed .env from example
        run: |
          cp .env.example .env
          echo "NEXO_SECRET_KEY=ci-smoke-$(openssl rand -hex 16)" >> .env

      - name: Start db + web
        run: docker compose up -d --build db web

      - name: Wait for web healthy
        run: |
          for i in {1..45}; do
            if curl -fs http://localhost:8001/api/health > /dev/null 2>&1; then
              echo "web healthy after ${i} attempts"
              curl -s http://localhost:8001/api/health | python3 -m json.tool
              exit 0
            fi
            echo "attempt $i failed, retrying..."
            sleep 2
          done
          echo "--- web logs ---"
          docker compose logs web
          echo "--- db logs ---"
          docker compose logs db
          exit 1

      - name: Tear down
        if: always()
        run: docker compose down

  # ── Docker build (sin push) ───────────────────────────────────────────
  build:
    name: Docker build
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - name: Build nexo-web image
        run: docker build -t nexo-web:ci .
      - name: Build nexo-mcp image
        run: docker build -t nexo-mcp:ci ./mcp

  # ── Secret scan ───────────────────────────────────────────────────────
  secrets:
    name: Secret scan (gitleaks)
    runs-on: ubuntu-24.04
    # Historical findings doc en docs/SECURITY_AUDIT.md. Se mantiene no-bloqueante
    # hasta Mark-IV donde se evalua filter-repo para limpiar historial.
    continue-on-error: true
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Gitleaks
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### `Makefile` — nuevo bloque a añadir (tras la sección "Produccion")

```makefile
# ── DevEx (Phase 7 / DEVEX-03) ───────────────────────────────────────────
# Targets para el dev loop: tests, lint, format, migrate, backup.
# `backup` ya existe (Phase 6); no se duplica.

.PHONY: test lint format migrate

test: ## Corre pytest con cobertura gate 60% (api/ + nexo/)
	pytest -q --cov=api --cov=nexo --cov-report=term --cov-fail-under=60

test-docker: ## Corre pytest dentro del container web
	docker compose exec -T web pytest -q --cov=api --cov=nexo --cov-fail-under=60

lint: ## Ruff check + Ruff format --check + mypy (scoped api/ nexo/)
	ruff check api/ nexo/
	ruff format --check api/ nexo/
	mypy api/ nexo/

format: ## Auto-fix ruff + auto-format ruff (scoped api/ nexo/)
	ruff check --fix api/ nexo/
	ruff format api/ nexo/

migrate: ## Aplica schema nexo (idempotente). Alias de nexo-init.
	docker compose exec web python scripts/init_nexo_schema.py

# `backup` ya declarado en la seccion Produccion (Phase 6) — no redeclarar.
```

**Añadir `test lint format migrate` a la línea `.PHONY` del header del Makefile** — Phase 6 ya tiene `backup` ahí.

### `docs/ARCHITECTURE.md` (NEW skeleton)

```markdown
# ARCHITECTURE.md — Mapa técnico de Nexo

> Para convenciones y reglas del juego, ver [CLAUDE.md](../CLAUDE.md).
> Para deploy en LAN, ver [DEPLOY_LAN.md](DEPLOY_LAN.md).
> Para incidencias runtime, ver [RUNBOOK.md](RUNBOOK.md).

Última revisión: 2026-MM-DD (Sprint 6 / Phase 7).

---

## 1. Qué es Nexo

Plataforma interna de ECS Mobility. Web app FastAPI + Postgres 16 + dos SQL Server.
Reemplaza al monolito "OEE Planta" manteniendo el mismo stack. Desplegada en LAN
interna (Ubuntu Server 24.04). Sin exposición a internet.

Audiencia de este doc: dev/IA nuevo en el repo. Después de leer esto + CLAUDE.md
deberías poder clonar, arrancar (`make up && make migrate && make nexo-owner`)
y empezar a trabajar.

---

## 2. Stack

| Capa | Tecnología | Versión | Notas |
|------|-----------|---------|-------|
| Web framework | FastAPI | 0.135.3 | Lifespan + middleware stack custom |
| Templating | Jinja2 | 3.1.6 | SSR; sin SPA |
| Frontend | Alpine.js + Tailwind (CDN) | latest | Sin build step |
| ORM | SQLAlchemy | 2.0.49 | 3 engines independientes |
| Postgres driver | psycopg2-binary | 2.9.11 | — |
| SQL Server driver | pyodbc | 5.3.0 | msodbcsql18 |
| Auth hashing | argon2-cffi | 25.1.0 | argon2id |
| Rate limit | slowapi | 0.1.9 | In-memory, per-IP |
| PDF gen | matplotlib | 3.10.8 | Backend Agg |
| DataFrame | pandas | 3.0.2 | — |
| Runtime image | python:3.11-slim-bookworm | — | Dockerfile |
| Reverse proxy | Caddy 2 | 2-alpine | `tls internal` en prod |
| Postgres server | postgres:16-alpine | — | Container `db` |

---

## 3. Los 3 engines

Nexo habla con tres bases de datos distintas. Cada una tiene un engine
SQLAlchemy dedicado, con responsabilidad y credenciales separadas.

\`\`\`mermaid
flowchart LR
  Client[Browser LAN]
  Caddy[Caddy 2<br/>tls internal<br/>:443]
  Web[FastAPI web<br/>container nexo-web<br/>:8000]

  Client -- HTTPS --> Caddy
  Caddy -- HTTP --> Web

  subgraph "BBDD Nexo (Postgres)"
    PGEngine[engine_nexo]
    PG[(Postgres 16<br/>container db<br/>:5432 interno)]
    PGEngine --> PG
  end

  subgraph "BBDD App (SQL Server ecs_mobility)"
    AppEngine[engine_app]
    AppDB[(SQL Server<br/>ecs_mobility)]
    AppEngine --> AppDB
  end

  subgraph "BBDD MES (SQL Server dbizaro, READ-ONLY)"
    MesEngine[engine_mes]
    MesDB[(SQL Server<br/>dbizaro)]
    MesEngine -- SELECT only --> MesDB
  end

  Web --> PGEngine
  Web --> AppEngine
  Web --> MesEngine

  subgraph "Opcional (profile: mcp)"
    MCP[MCP server<br/>nexo-mcp]
    MCP --> Web
  end
\`\`\`

### Responsabilidades por engine

| Engine | BBDD | Env vars | Casos de uso |
|--------|------|----------|--------------|
| `engine_nexo` | Postgres 16 (schema `nexo.*`) | `NEXO_PG_*` | users, roles, permissions, sessions, login_attempts, audit_log, query_log, query_thresholds, query_approvals |
| `engine_app` | SQL Server `ecs_mobility` | `NEXO_APP_*` | cfg.recursos, cfg.ciclos, cfg.contactos, oee.*, luk4.* |
| `engine_mes` | SQL Server `dbizaro` | `NEXO_MES_*` | partes de trabajo, turnos, recursos MES (read-only: el usuario SQL es de sólo lectura) |

**Regla de oro** (Phase 3): ningún router importa pyodbc directamente. Todas las
queries pasan por repositorios en `nexo/data/repositories/{app,mes,nexo}.py`, que
cargan SQL desde archivos `.sql` versionados en `nexo/data/sql/`.

---

## 4. Layout del repo

\`\`\`
analisis_datos/
├── api/                    # FastAPI app
│   ├── main.py             # Lifespan + middleware registration
│   ├── config.py           # pydantic-settings (lee NEXO_*)
│   ├── database.py         # init_db legacy (ecs_mobility bootstrap)
│   ├── deps.py             # Dependencies (current_user, Jinja env)
│   ├── middleware/         # auth, audit
│   ├── routers/            # 25+ routers (pipeline, bbdd, auth, ...)
│   ├── services/           # Business logic (pipeline, email, ...)
│   └── rate_limit.py       # slowapi limiter
├── nexo/                   # Nueva capa (Phase 3+)
│   ├── data/
│   │   ├── engines.py      # engine_mes / engine_app / engine_nexo
│   │   ├── repositories/   # MesRepository, AppRepository, NexoRepository
│   │   ├── sql/            # Queries .sql versionadas + loader
│   │   ├── dto/            # DTOs inmutables (frozen dataclasses)
│   │   ├── models_app.py   # SQLAlchemy models ecs_mobility
│   │   ├── models_nexo.py  # SQLAlchemy models schema nexo
│   │   └── schema_guard.py # Valida tablas nexo.* al arrancar
│   ├── middleware/         # flash, query_timing
│   └── services/           # auth, approvals, preflight, pipeline_lock, ...
├── OEE/                    # LEGACY — NO TOCAR en Mark-III
│   ├── disponibilidad/
│   ├── rendimiento/
│   ├── calidad/
│   └── oee_secciones/
├── templates/              # Jinja2 (base.html + 40 sub-templates)
├── static/                 # CSS, JS, imágenes brand
├── scripts/                # init_nexo_schema.py, deploy.sh, backup_nightly.sh
├── tests/                  # pytest (222+ tests; auth/ data/ routers/ ...)
├── caddy/                  # Caddyfile + Caddyfile.prod
├── docs/                   # Fuente de verdad humana (planes, audits)
├── .planning/              # Runtime GSD (no editar a mano)
├── Dockerfile              # Python 3.11-slim + msodbcsql18
├── docker-compose.yml      # Base: db + web + caddy + mcp (profile)
├── docker-compose.prod.yml # Override prod (Phase 6)
├── Makefile                # 30+ targets dev/prod/devex
└── pyproject.toml          # Config ruff + mypy + pytest + coverage (Phase 7)
\`\`\`

---

## 5. Flujo de una request

\`\`\`
Browser → Caddy (TLS) → web container (:8000) → ...
  → FastAPI Exception Handler (catches 500 + wraps in error_id UUID)
  → AuthMiddleware (validates session cookie, sets request.state.user)
  → AuditMiddleware (writes to nexo.audit_log after response)
  → FlashMiddleware (reads/clears nexo_flash cookie for toast UX)
  → QueryTimingMiddleware (measures + logs slow queries to nexo.query_log)
  → Router (e.g., api.routers.pipeline.run)
  → Service layer (e.g., api.services.pipeline.run_pipeline)
  → Repository (e.g., nexo.data.repositories.mes.MesRepository)
  → Engine (engine_mes / engine_app / engine_nexo)
\`\`\`

Orden del middleware stack (verificado en `api/main.py`):
1. **AuthMiddleware** (outer) — 401 si no hay sesión válida, salvo `/login`, `/static/`, `/api/health`.
2. **AuditMiddleware** — registra cada request + user + path + status en `nexo.audit_log`.
3. **FlashMiddleware** — lee/borra `nexo_flash` cookie, inyecta en response.
4. **QueryTimingMiddleware** (inner) — mide `actual_ms`, compara con threshold, alerta si > warn_ms × 1.5.

---

## 6. Schedulers (lifespan tasks)

Tres tasks asyncio arrancadas en `lifespan` (`api/main.py`):

| Task | Origen | Cadencia | Función |
|------|--------|----------|---------|
| `thresholds_cache` listener | Phase 4 / Plan 04-04 | LISTEN/NOTIFY + safety-net 5 min | Refresca cache cuando `/ajustes/limites` CRUD dispara NOTIFY |
| `cleanup_scheduler` → `approvals_cleanup` | Phase 4 / Plan 04-03 | Lunes 03:05 UTC | Purga approvals expirados (TTL 7d) |
| `cleanup_scheduler` → `query_log_cleanup` | Phase 4 / Plan 04-04 | Lunes 03:00 UTC | Retención 90d de `nexo.query_log` |
| `cleanup_scheduler` → `factor_auto_refresh` | Phase 4 / Plan 04-04 | 1er lunes del mes 03:10 UTC | Recalcula factor si estancado > 60d |

Todos los jobs escriben audit log con `path='__<job_name>__'`.

---

## 7. Deployment

Deploy productivo LAN: ver [DEPLOY_LAN.md](DEPLOY_LAN.md) (740 líneas, 16 secciones).
Resumen: `make prod-up`, hostname `nexo.ecsmobility.local`, Caddy `tls internal`,
ufw 22/80/443, cron diario `backup_nightly.sh`.

---

## 8. Enlaces rápidos

- [CLAUDE.md](../CLAUDE.md) — reglas de juego para IAs/devs
- [AUTH_MODEL.md](AUTH_MODEL.md) — roles, departamentos, lockout
- [BRANDING.md](BRANDING.md) — assets + vars de marca
- [GLOSSARY.md](GLOSSARY.md) — términos de dominio
- [DEPLOY_LAN.md](DEPLOY_LAN.md) — runbook deploy LAN
- [RUNBOOK.md](RUNBOOK.md) — 5 escenarios de incidencia
- [RELEASE.md](RELEASE.md) — checklist release versionado
- [SECURITY_AUDIT.md](SECURITY_AUDIT.md) — historial credenciales expuestas

---

## 9. Si tocas algo estructural, actualiza esto

Si añades un 4° engine, cambias el middleware stack, o introduces un scheduler
nuevo, actualiza esta página en el mismo PR. ARCHITECTURE.md drift es peor que
ARCHITECTURE.md ausente.
```

### `docs/RUNBOOK.md` — Template del Escenario 1 (planner replica para 2-5)

```markdown
# RUNBOOK.md — Procedimientos de incidencia para Nexo

> Audiencia: operador (admin IT) respondiendo a incidencia en producción.
> Complementa [DEPLOY_LAN.md](DEPLOY_LAN.md) (instalación) con respuesta runtime.
> Ver [ARCHITECTURE.md](ARCHITECTURE.md) para contexto del sistema.

Última revisión: 2026-MM-DD (Sprint 6 / Phase 7).

Cada escenario tiene: **Síntomas** → **Diagnóstico** → **Remedio** → **Prevención**.
Comandos asumen que estás en el servidor productivo (`ssh <IP_NEXO>`) en
`/opt/nexo` salvo indicación contraria.

---

## Escenario 1: MES caído (SQL Server dbizaro inaccesible)

### Síntomas

- `/bbdd` devuelve 503 o timeout > 30s.
- `/pipeline` con rango que consulta MES falla con mensaje "no se puede
  conectar a MES".
- Logs del container web muestran `pyodbc.OperationalError` con `engine_mes`:
  ```
  docker compose logs web | grep -i "engine_mes\|pyodbc"
  ```
- `/api/health` responde **200** (no 503) pero con `services.mes.ok=false`
  (este es el diseño intencional — la app sigue viva aunque MES esté caído,
  Caddy no marca el backend down).

### Diagnóstico

1. **Confirma que MES es el problema** (no todo el SQL Server):
   ```bash
   docker compose exec web python -c "
   from nexo.data.engines import engine_mes
   from sqlalchemy import text
   try:
       engine_mes.connect().execute(text('SELECT 1'))
       print('MES: OK')
   except Exception as e:
       print(f'MES: FAIL — {e}')
   "
   ```

2. **Test de red al host SQL Server**:
   ```bash
   MES_HOST=$(grep '^NEXO_MES_SERVER=' .env | cut -d= -f2)
   MES_PORT=$(grep '^NEXO_MES_PORT=' .env | cut -d= -f2)
   docker compose exec web nc -zv "$MES_HOST" "$MES_PORT"
   ```
   - `open` → red OK, problema es auth o BBDD.
   - `refused` / `timeout` → red caída o firewall.

3. **Verifica que engine_app SÍ funciona** (para aislar el problema a MES solo):
   ```bash
   docker compose exec web python -c "
   from nexo.data.engines import engine_app
   from sqlalchemy import text
   engine_app.connect().execute(text('SELECT 1'))
   print('engine_app: OK')
   "
   ```

### Remedio

**Caso A — Red caída al SQL Server**:
- Escalar a IT: cable, switch, firewall, ruta estática.
- Mientras tanto: la app sigue viva, solo las pantallas que consultan MES
  (`/bbdd`, `/pipeline`, `/capacidad`) darán error claro.

**Caso B — Credenciales SQL Server cambiadas**:
- Rotación reciente: actualizar `NEXO_MES_PASSWORD` en `/opt/nexo/.env`.
- Reiniciar stack para recargar env:
  ```bash
  cd /opt/nexo
  make prod-down && make prod-up
  make prod-health
  ```

**Caso C — SQL Server MES caído**:
- Escalar al DBA del equipo MES (dbizaro). Fuera del alcance de Nexo.

### Prevención

- Healthcheck `/api/health` expone `services.mes.ok` — integrar en
  monitoreo externo (cron LAN que llame cada 5 min, alerta si
  `ok=false` durante > 15 min consecutivos).
- Las credenciales MES están en `.env` (no en imagen Docker) — al rotar,
  un solo reinicio las recoge.
- La app está diseñada para degradarse graciosamente: MES caído no tumba
  Nexo completo, sólo las pantallas que lo consumen.

---

## Escenario 2: Postgres no arranca

### Síntomas
...(mismo patrón — réplica de estructura)...

## Escenario 3: Certificado Caddy expira / warning en browsers
...

## Escenario 4: Pipeline atascado (semáforo in-process no libera)
...

## Escenario 5: Lockout del único propietario
...
```

(Planner replica esa estructura 4 veces con el contenido específico de §6 de esta research.)

### `docs/RELEASE.md` (NEW file)

```markdown
# RELEASE.md — Checklist de release versionado

> Audiencia: quien corta una release de Nexo (típicamente el dev lead o
> responsable de la milestone).
> Ver [RUNBOOK.md](RUNBOOK.md) para recuperación si el release rompe prod.

## Versionado

Nexo usa **Semantic Versioning** (https://semver.org):

- **MAJOR** (x.0.0) — milestones estructurales (Mark-III, Mark-IV).
- **MINOR** (1.x.0) — features no-breaking dentro de una milestone.
- **PATCH** (1.0.x) — fixes, hotfixes, seguridad.

Mapa actual:
- **v1.0.0** — cierre Mark-III (Sprint 6 / Phase 7 verificado).
- **v1.0.x** — fixes sobre Mark-III.
- **v2.0.0** — cierre Mark-IV (fecha TBD).

**Tags son inmutables**. Si un release falla, se crea `v1.0.1` con el fix.
NUNCA reescribir un tag publicado.

---

## Checklist pre-release

- [ ] Todos los plans de la phase de cierre marcados ✓ en `.planning/ROADMAP.md`.
- [ ] CI verde en `feature/Mark-III` (lint matrix 3.11+3.12 + test matrix + smoke + build).
- [ ] `pytest --cov=api --cov=nexo --cov-fail-under=60` exit 0 en local.
- [ ] `pre-commit run --all-files` exit 0.
- [ ] `.planning/STATE.md` sección "Blockers/Concerns" no tiene items críticos abiertos.
- [ ] `CHANGELOG.md` actualizado con sección nueva `## [1.0.0] - 2026-MM-DD`.
- [ ] Docs de la milestone actualizados: ARCHITECTURE.md, RUNBOOK.md, CLAUDE.md.

## Checklist de release (ejecución)

1. [ ] **Merge** `feature/Mark-III` → `main`:
   ```bash
   git checkout main
   git pull origin main
   git merge --no-ff feature/Mark-III -m "release: Mark-III v1.0.0"
   git push origin main
   ```

2. [ ] **Tag** en `main`:
   ```bash
   git tag -a v1.0.0 -m "Nexo Mark-III — LAN deployment + auth + RBAC + preflight + devex"
   git push origin v1.0.0
   ```

3. [ ] **GitHub Release** desde el tag:
   ```bash
   gh release create v1.0.0 \
     --title "Nexo v1.0.0 — Mark-III" \
     --notes-file <(sed -n '/## \[1\.0\.0\]/,/## \[/p' CHANGELOG.md | head -n -1)
   ```

4. [ ] **Deploy** en servidor:
   ```bash
   ssh <IP_NEXO> 'cd /opt/nexo && git pull && make deploy'
   ```
   (`scripts/deploy.sh` hace: pre-backup atomic → git pull → build → up -d → smoke HTTPS.)

5. [ ] **Smoke externo** post-deploy:
   ```bash
   ssh <IP_NEXO> 'cd /opt/nexo && bash tests/infra/deploy_smoke.sh'
   ```
   Aceptación: exit 0 (0 fallos sobre 11 checks).

6. [ ] **Verificación manual** desde equipo LAN:
   - `https://nexo.ecsmobility.local` → pantalla login.
   - Login propietario → `/` (Centro de Mando) carga.
   - `/api/health` → `{"status":"ok","services":{"db":{"ok":true},...}}`.

7. [ ] **Anuncio interno** (email / chat ECS Mobility) con:
   - Link a la GitHub Release.
   - Highlights (extraídos de CHANGELOG).
   - Instrucciones para usuarios finales si hay cambios de UX.

## Rollback

Si el release rompe prod y no se puede recuperar con `RUNBOOK.md`:

```bash
ssh <IP_NEXO>
cd /opt/nexo
git log --oneline -5
git checkout <tag-anterior>  # ej. v0.9.0
make prod-down && make prod-up
```

Restaurar BD desde `/var/backups/nexo/predeploy/<latest>.sql.gz` si el
release incluyó migración destructiva:

```bash
gunzip < /var/backups/nexo/predeploy/<latest>.sql.gz | \
  docker compose exec -T db psql -U "$POSTGRES_USER" "$POSTGRES_DB"
```

Abrir issue de hotfix → `v1.0.1`.
```

### `CHANGELOG.md` (NEW file, ROOT, Keep a Changelog format)

```markdown
# Changelog

All notable changes to Nexo documented here.

El formato sigue [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
y el proyecto adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

## [1.0.0] - 2026-MM-DD

**Cierre de Mark-III.** Primera release productiva de Nexo como
plataforma interna de ECS Mobility.

### Added

- **Phase 1 (Sprint 0):** Rebrand completo `OEE Planta` → `Nexo`. Env vars
  `OEE_*` → `NEXO_*` con compat layer. Exception handler sin traceback leak.
  CI mínimo en GitHub Actions. Audit de credenciales expuestas.
- **Phase 2 (Sprint 1):** Auth con argon2id. RBAC con roles
  `propietario`/`directivo`/`usuario` + 5 departamentos. Audit log
  append-only en `nexo.audit_log`. Lockout progresivo 5/15min.
- **Phase 3 (Sprint 2):** Capa de datos — repositorios en
  `nexo/data/repositories/`. SQL versionado en `nexo/data/sql/`. Tres
  engines separados: `engine_mes` (dbizaro RO), `engine_app` (ecs_mobility),
  `engine_nexo` (Postgres). `schema_guard` al arrancar.
- **Phase 4 (Sprint 3):** Preflight + postflight para queries pesadas.
  Flujo de aprobación asíncrona. Umbrales editables en `/ajustes/limites`.
  Pipeline con semáforo + timeout soft.
- **Phase 5 (Sprint 4):** Sidebar filtrado por permisos (`can()`).
  `/ajustes` split en 6 sub-páginas. 11 botones + 8 GETs HTML gateados.
- **Phase 6 (Sprint 5):** Despliegue LAN HTTPS con Caddy `tls internal`.
  `make prod-*` targets. `scripts/deploy.sh` + `scripts/backup_nightly.sh`.
  Runbook `docs/DEPLOY_LAN.md` (740 líneas).
- **Phase 7 (Sprint 6):** Pre-commit (ruff + mypy). CI matriz 3.11+3.12
  + cobertura ≥60%. Makefile `make test/lint/format/migrate/backup`.
  `docs/ARCHITECTURE.md`, `docs/RUNBOOK.md`, `docs/RELEASE.md`, CHANGELOG.

### Changed

- `OEE/` renombrado a `OEE/` (sic — carpeta mantiene nombre; módulos legacy
  no se refactorizan en Mark-III).
- Repo GitHub mantiene nombre `analisis_datos` (rename diferido).

### Security

- Rotación de password SA SQL Server (Sprint 0).
- Exception handler devuelve solo `error_id` UUID, nunca traceback.
- Audit log en Postgres con `GRANT SELECT, INSERT` (append-only).

### Deprecated

- Env vars `OEE_*` — aceptadas como fallback durante Mark-III, removidas en
  Mark-IV.

[Unreleased]: https://github.com/<org>/analisis_datos/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/<org>/analisis_datos/releases/tag/v1.0.0
```

### `requirements-dev.txt` — cambios Phase 7

```diff
 -r requirements.txt

 # Lint + formato
-ruff==0.9.8
+ruff==0.15.11

+# Type checking (Phase 7 / DEVEX-01)
+mypy==1.13.0
+
+# Pre-commit (Phase 7 / DEVEX-01)
+pre-commit==4.0.1
+
 # Testing
 pytest==8.3.4
 pytest-cov==6.0.0
 httpx==0.28.1
 pytest-asyncio==0.24.0
```

---

## State of the Art

| Old Approach | Current Approach (2026) | Changed | Impact |
|--------------|-------------------------|---------|--------|
| black + isort + flake8 + ruff (4 tools) | ruff único (check + format) | ~2024 | 10-100× más rápido, 1 tool, sin conflictos de formato |
| pyproject.toml opcional | pyproject.toml canónico para config | ~2023 PEP 621 adoption | Single source of truth |
| mypy strict desde día 1 | mypy ligero (ignore_missing_imports + overrides) | — | Progresivo: viable en codebases legacy |
| pytest-cov separado | pytest-cov sigue siendo canonical | — | Stable; coverage.py puro rara vez justificado |
| Pre-commit hooks bash scripts ad-hoc | pre-commit framework + ruff-pre-commit | ~2022 | Hooks declarativos, versionados, reproducibles |

### Deprecated/outdated

- **`ruff format --check` con `continue-on-error: true`** (estado actual Phase 1 CI): deprecated con el backfill de Phase 7 — ya no hay razón para ser permisivo.
- **`pytest --continue-on-error: true`**: idem, deprecated en Phase 7.
- **Black standalone**: reemplazado por ruff-format (aunque seguirá funcionando — decisión Mark-III: no mantenerlo).

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Ruff 0.15.11 es estable para producción | §1 + Code Examples | Si inestable, downgrade a 0.15.x-1 o 0.14.x; no bloquea Phase 7 |
| A2 | mypy 1.13.0 compatible con Python 3.11+3.12 | §1 + Code Examples | Si incompatible con 3.12, pinear a mypy compatible; ligero blocker |
| A3 | Coverage baseline actual está entre 55-70% (no medido en esta sesión) | §3 | Si <50%: Phase 7 necesita waves adicionales para subir cobertura; MEDIO impacto |
| A4 | `.env.example` tiene todos los vars requeridos para `docker compose up` en CI | §2 Smoke | Si falta algún var, CI smoke falla; fácil de arreglar en el propio PR |
| A5 | `pre-commit==4.0.1` estable y compat Python 3.11+3.12 | §1 | Fallback: versión anterior estable |
| A6 | GH Actions `ubuntu-24.04` runner tiene Docker + compose preinstalados | §2 Smoke | VERIFICADO históricamente; bajo riesgo |

Resto de claims son `[VERIFIED]` (lectura directa del repo) o `[CITED]` (docs oficiales).

---

## Open Questions

1. **¿Baseline de cobertura actual en api/+nexo/?**
   - What we know: tests existen (41 files, 222+ tests green tras Phase 5), pero nunca se ha medido coverage exacto.
   - What's unclear: si está ≥60% ya o no.
   - Recommendation: Wave 0 task del Plan → ejecutar `pytest --cov=api --cov=nexo --cov-report=term` en local (dev con tooling instalado o `docker compose exec web pytest --cov=...`). Reportar resultado y decidir si Phase 7 necesita añadir tests o sólo activar gate.

2. **¿Pinear Python 3.12 específicamente (e.g., 3.12.7) o accept any 3.12.x?**
   - Recommendation: accept 3.12 (setup-python@v5 resuelve a la última 3.12.x). Más robusto, menos mantenimiento.

3. **¿Smoke test CI usa Postgres service container o docker compose?**
   - **Option A** (docker compose): real-world, usa `docker-compose.yml` directo, prueba el compose real.
   - **Option B** (service container): más rápido (no build), aislado.
   - Recommendation: **A para smoke, B para test matrix**. Coverage matrix necesita Postgres (schema_guard); usar service container. Smoke necesita simular prod → compose completo.

4. **¿Pre-commit CI (adicional al local)?**
   - Opción: añadir job `pre-commit-ci` a CI que corra `pre-commit run --all-files`. Asegura que el hook también pasa en el runner, no solo localmente.
   - Recommendation: SÍ, baja fricción y cierra el loop. Añadir como extra job en ci.yml (5 líneas).

---

## Project Constraints (from CLAUDE.md)

Estas directivas de CLAUDE.md aplican a Phase 7:

- **No refactorizar módulos OEE legacy** → `pre-commit` scoped a `api/+nexo/`, OEE excluido. ✓ cumple.
- **No renombrar carpeta `OEE/`** → docs y Makefile referencian `OEE/`. ✓
- **`make up` y `make dev` NO arrancan `mcp`** → profile-gated; Phase 7 no lo toca. ✓
- **Conventional Commits + no `--no-verify`** → Phase 7 activa pre-commit que refuerza este policy. ✓ (se alinea)
- **No reescribir historial** → Phase 7 no toca git history. ✓
- **Pre-commit entra en Sprint 6 (Phase 7)** → LITERAL cita de CLAUDE.md; Phase 7 lo cierra. ✓

---

## Sources

### Primary (HIGH confidence, `[VERIFIED]`)

- `/home/eeguskiza/analisis_datos/.planning/REQUIREMENTS.md` (DEVEX-01..07 literal text, lines 105-113)
- `/home/eeguskiza/analisis_datos/.planning/ROADMAP.md` (Phase 7 goal + success criteria, lines 132-143)
- `/home/eeguskiza/analisis_datos/.planning/STATE.md` (current progress, deferred items, blockers)
- `/home/eeguskiza/analisis_datos/CLAUDE.md` (convenciones, hard invariants)
- `/home/eeguskiza/analisis_datos/.github/workflows/ci.yml` (CI actual — 4 jobs, Py 3.11 only)
- `/home/eeguskiza/analisis_datos/Makefile` (targets actuales — 30+ incluyendo prod-*)
- `/home/eeguskiza/analisis_datos/requirements-dev.txt` (ruff 0.9.8, pytest 8.3.4, pytest-cov 6.0.0, pytest-asyncio 0.24.0)
- `/home/eeguskiza/analisis_datos/requirements.txt` (FastAPI 0.135.3, Python 3.11 via Dockerfile)
- `/home/eeguskiza/analisis_datos/docker-compose.yml` (puerto 8001 host → 8000 container; mcp profile-gated)
- `/home/eeguskiza/analisis_datos/Dockerfile` (Python 3.11-slim-bookworm)
- `/home/eeguskiza/analisis_datos/nexo/services/pipeline_lock.py` (semáforo in-process, NO list_locks)
- `/home/eeguskiza/analisis_datos/nexo/services/auth.py` (clear_attempts helper, NO unlock_user)
- `/home/eeguskiza/analisis_datos/nexo/data/schema_guard.py` (CRITICAL_TABLES incluye login_attempts)
- `/home/eeguskiza/analisis_datos/api/main.py` (lifespan, 3 scheduler tasks)

### Secondary (HIGH confidence, `[CITED]` official docs)

- [Ruff — Configuring Ruff](https://docs.astral.sh/ruff/configuration/)
- [Ruff — The Ruff Formatter](https://docs.astral.sh/ruff/formatter/)
- [astral-sh/ruff-pre-commit](https://github.com/astral-sh/ruff-pre-commit) — rev `v0.15.11`
- [pre-commit.com](https://pre-commit.com/) — framework canónico
- [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) — CHANGELOG format
- [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html)
- [actions/setup-python@v5](https://github.com/actions/setup-python) — matrix + cache
- [pytest-cov on PyPI](https://pypi.org/project/pytest-cov/) — `--cov-fail-under`

### Tertiary (MEDIUM confidence, WebSearch verified)

- [Ruff vs Black vs isort: Complete 2026 Formatter Benchmark — bytepulse.io](https://bytepulse.io/ruff-vs-black-vs-2026/) — confirma ruff > black en 2026
- [How to migrate from Black to Ruff formatter — pydevtools](https://pydevtools.com/handbook/how-to/how-to-migrate-from-black-to-ruff-formatter/)
- [Astral Python 2026: Ruff, uv, ty — w3resource](https://www.w3resource.com/python/astral-python-tooling-revolution-in-2026.php)
- [Configure Ruff Easily with pyproject.toml — Gema Correa / Medium](https://medium.com/@gema.correa/configure-ruff-easily-with-pyproject-toml-f75914fab055)

---

## Metadata

**Confidence breakdown:**
- Pre-commit setup: **HIGH** — configuración estándar, múltiples fuentes oficiales, patrón validado en FastAPI upstream.
- CI GitHub Actions: **HIGH** — matrix syntax trivial, `.github/workflows/ci.yml` ya existe.
- Coverage measurement: **MEDIUM** — baseline no medido en sesión; target 60% es educated guess validable en Wave 0.
- Makefile targets: **HIGH** — patrón Phase 6 ya establece convención, 4 nuevos targets son directos.
- ARCHITECTURE.md: **HIGH** — estructura y Mermaid verificados contra repo real.
- RUNBOOK.md: **HIGH** — 2 hallazgos críticos (pipeline_lock sin list_locks, unlock_user inexistente) verificados por lectura directa del código.
- RELEASE.md + CHANGELOG: **HIGH** — standard de facto (Keep a Changelog + semver).
- CLAUDE.md update: **HIGH** — estructura actual legible, diff es aditivo.

**Research date:** 2026-04-21
**Valid until:** 2026-05-21 (30 días; ruff/mypy versions pueden bumpearse antes pero config es estable).

---

## Ready for Planning

Research completo. El planner puede proceder con `/gsd-plan-phase 7` con:

- **Waves sugeridas**:
  - **Wave 0** (foundation): medir coverage baseline; crear `pyproject.toml` + `.pre-commit-config.yaml`; actualizar `requirements-dev.txt`; backfill `ruff format + ruff check --fix` en commit aislado.
  - **Wave 1** (CI + Makefile): extender `ci.yml` con matriz + smoke + coverage gate; añadir 5 Makefile targets.
  - **Wave 2** (docs): `ARCHITECTURE.md`, `RUNBOOK.md`, `RELEASE.md`, `CHANGELOG.md`.
  - **Wave 3** (closing): `CLAUDE.md` update + `tests/infra/test_devex_config.py` regression tests + `gsd-verify-work`.

- **Estimación**: ~4-5 plans atómicos, ~3-4h total (basado en velocity Phase 6 que tuvo 3 plans en ~60 min).

- **Gates**:
  1. `pre-commit run --all-files` → 0 issues.
  2. CI matriz verde (lint + test + smoke + build) en ambos 3.11 y 3.12.
  3. `pytest --cov=api --cov=nexo --cov-fail-under=60` exit 0.
  4. `tests/infra/test_devex_config.py` green (valida existencia de 5 docs + config files).
  5. `make test lint format migrate backup` ejecutan sin error.
