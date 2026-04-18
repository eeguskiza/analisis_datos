# Summary — Plan 01-01: Sprint 0

**Phase**: 1 (Naming + Higiene + CI)
**Plan**: 01-01
**Ejecutado**: 2026-04-18
**Rama**: `feature/Mark-III`
**Modo**: `/gsd-execute-phase 1 --interactive` con checkpoint por hito
**Total commits**: 14 (13 del plan + 1 feedback operador post-commit 13)

---

## Commits del sprint

| # | Hash | Tipo | Mensaje corto |
|---|------|------|---------------|
| 1 | `8774deb` | chore (gate 1) | audit git history for leaked credentials |
| 2 | `dec03d1` | chore | remove tracked junk files (incluye ajuste al Dockerfile) |
| 3 | `bfe35e7` | chore | update .gitignore patterns (`*:Zone.Identifier`) |
| 4 | `7babe5d` | chore | move install_odbc.sh to scripts/ |
| 5 | `151d83c` | chore | handle data/oee.db (backup offline + borrado) |
| 6 | `10f8164` | refactor | rename OEE_* env vars to NEXO_*, split MES/APP |
| 7 | `f5d0ce7` | refactor | update UI titles and metadata to Nexo |
| 8 | `dc5b7df` | fix | global_exception_handler no longer leaks tracebacks |
| 9 | `f689212` | chore | move mcp service to docker-compose profile |
| 10 | `72d658c` | build | pin requirements.txt, add requirements-dev.txt |
| 11 | `1edf57a` | ci | add GitHub Actions workflow (lint, test, build, secrets) |
| 12 | `4889811` | docs | add GLOSSARY, DATA_MIGRATION_NOTES (verify CLAUDE.md/AUTH_MODEL/BRANDING) |
| 13 | `263b706` | docs | sync MARK_III_PLAN and OPEN_QUESTIONS with Sprint 0 outcomes |
| 14 | `8feacb0` | style | sidebar: drop 'NEXO' text, enlarge logo, keep 'by ECS Mobility' below (**out-of-plan**) |

---

## Gates

### 🚦 Gate 1 — Audit del historial (tras commit 1)

- **Resultado**: 4 hallazgos documentados en `docs/SECURITY_AUDIT.md`.
  - H1 🔴 `data/db_config.json @ b0e80b9` — password SA real commiteada.
  - H2 🟠 `docker-compose.yml @ f22d689, d1b4339` — `POSTGRES_PASSWORD` (mismo valor SA).
  - H3 🟡 `.env.example @ d1b4339, 03f3992` — placeholders esperados.
  - H4 🟡 `data/db_config.example.json @ 3007dc5` — placeholder.
  - 5 × `:Zone.Identifier` — ruido NTFS/WSL.
- **Acción operador durante el sprint**: confirmó H1 real y **rotó la password SA** en SQL Server. H1 y H2 pasan a RESOLVED.
- **`filter-repo`**: no ejecutado. Al quedar la credencial muerta, la limpieza del historial queda cosmética y se difiere.
- **Pass**: sí, con intervención del operador.

### 🚦 Gate 2 — Integridad de arranque (tras cada commit del 2 al 13)

Validaciones aplicadas:
- `ast.parse` de archivos Python modificados (api/main.py, api/config.py, api/deps.py, api/routers/email.py, mcp/server.py): **OK**.
- `yaml.safe_load(docker-compose.yml)`: **OK** tras cada commit que lo tocó.
- `docker compose config --quiet`: **OK** (commits 6, 9).
- `docker compose --profile mcp config --services`: incluye `mcp`; sin profile lo excluye (commit 9).
- `new Function(static/js/app.js)`: **OK** (commits 2, 7).
- `docker run --rm analisis_datos-web:latest pip freeze`: confirma que el pin del commit 10 matches la imagen activa.
- `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`: **OK** (commit 11).

Ningún commit rompió la integridad. **Pass**.

---

## Decisiones ejecutadas durante el sprint

- **Rotación de password SA**: ejecutada por operador durante cierre del audit (reducción de riesgo de H1/H2). Decisión originalmente diferida; adelantada al ver el hallazgo real.
- **`data/oee.db`**: backup offline (`data/backups/oee_db_snapshot.sqlite`) + borrado. Inspección reveló 1 433 filas en `datos_produccion`, no residuo. Hash git de recuperación = `c611b78`.
- **Dockerfile tocado en commit 2**: delete de `server.py` obligaba a quitar `COPY server.py .`. Alineado con atomicidad.
- **Compat layer `OEE_* ↔ NEXO_*`**: `api/config.py` usa `pydantic.AliasChoices`. Lectura: `NEXO_*` preferente → `OEE_*` fallback. Se retira en Mark-IV.
- **`window.NEXO_CONFIG` en `base.html`**: inyectado para que `static/js/app.js` lea branding sin hardcoded strings.
- **`ruff format --check` en CI**: `continue-on-error: true` en Sprint 0. Se hará bloqueante en Sprint 6 tras el commit masivo de formato.
- **Logo del sidebar (post-commit 13)**: operador pidió quitar el texto "NEXO" y agrandar el logo. Aplicado como commit 14 out-of-plan.

---

## Desviaciones vs PLAN original

| Desviación | Causa | Impacto |
|------------|-------|---------|
| Commit 2 toca `Dockerfile` | Sin eliminar `COPY server.py .` el build rompe | Coherente con el delete, commit sigue atómico |
| Commit 5 preserva backup | Inspección encontró 1 433 filas reales | Prescrito por el PLAN para este caso |
| Commit 7 inyecta `window.NEXO_CONFIG` | Necesario para que `app.js` lea branding sin imports nuevos | Ampliación menor, misma filosofía |
| Commit 12 sólo añade GLOSSARY | CLAUDE.md + AUTH_MODEL + BRANDING + DATA_MIGRATION ya existían (sesión de arranque + commit 5) | Scope reducido, propósito preservado |
| Commit 14 adicional (out-of-plan) | Feedback del operador sobre el sidebar durante ejecución | Ajuste visual aislado, commit atómico |

---

## Archivos tocados (resumen agregado)

- **Creados**: `.github/workflows/ci.yml`, `docs/SECURITY_AUDIT.md`, `docs/DATA_MIGRATION_NOTES.md`, `docs/GLOSSARY.md`, `requirements-dev.txt`, `data/backups/oee_db_snapshot.sqlite` (no trackeado).
- **Movidos**: `install_odbc.sh` → `scripts/install_odbc.sh`.
- **Eliminados**: `.env:Zone.Identifier`, `test_email.py`, `server.py`, `data/oee.db`.
- **Modificados**: `.gitignore`, `Dockerfile`, `docker-compose.yml`, `Makefile`, `README.md`, `requirements.txt`, `mcp/requirements.txt`, `mcp/README.md`, `mcp/server.py`, `api/main.py`, `api/config.py`, `api/deps.py`, `api/routers/email.py`, `templates/base.html`, `templates/ciclos.html`, `templates/plantillas.html`, `static/js/app.js`, `static/css/app.css`, `docs/MARK_III_PLAN.md`, `docs/OPEN_QUESTIONS.md`.

---

## Hallazgos fuera del alcance — propuestas para sprints posteriores

- **`showToast` duplicado** entre `templates/base.html` (implementación Alpine) y `static/js/app.js` (DOM append). Unificar en Mark-IV o cuando UI por roles (Sprint 4) toque `app.js`.
- **`data/ecs-logo.png`** sigue separado de `static/img/brand/ecs/logo.png`. Lo consume matplotlib vía `api/config.py:logo_filename`. Unificación cuando el refactor de config en Sprint 2 toque estos paths.
- **`pandas==3.0.2`**: la mayor 3.x tiene cambios de API frente a 2.x. `api/services/pipeline.py` y módulos OEE no han sido re-validados explícitamente con 3.x. Añadir a la lista de verificación de Sprint 2 (capa de datos) o correr la suite existente en CI.
- **Tests pytest actuales no se han ejecutado durante este sprint**. El workflow `ci.yml` existe y no los bloquea; tras el push, Erik verifica manualmente la primera corrida en GitHub Actions.
- **`ruff check .` sin reglas configuradas**: usará defaults de ruff. Crear `pyproject.toml` con `[tool.ruff]` y `[tool.ruff.lint]` en Sprint 6 junto con pre-commit.

---

## Próximos pasos

1. **Erik** verifica la primera corrida del workflow `ci.yml` en GitHub Actions tras el push.
2. **Erik** pega (si aún no lo hizo) las variables `NEXO_*` definitivas en `.env.example` (yo no tengo acceso por denyList).
3. **Siguiente fase**: `/gsd-plan-phase 2` para generar el PLAN.md de Phase 2 (Identidad — auth + RBAC + audit) o `/gsd-discuss-phase 2` si quieres una ronda de clarificación sobre implementación antes de planear.

---

*SUMMARY creado 2026-04-18 al cierre del Sprint 0.*
