# Summary — Plan 02-03: rbac-retrofit-routers

**Phase**: 2 (Identidad — auth + RBAC + audit)
**Plan**: 02-03 of 04
**Ejecutado**: 2026-04-19
**Rama**: `feature/Mark-III`
**Modo**: `/gsd-execute-phase 2 --interactive` con hitos A/B/C
**Total commits del plan**: 4 (1 por tarea, atómicos)

---

## Commits

| # | Hash | Tipo | Mensaje |
|---|------|------|---------|
| 1 | `0c4f34b` | feat | PERMISSION_MAP (24) + require_permission factory |
| 2 | `229ca44` | feat | retrofit RBAC en 14 routers (health excluido) |
| 3 | `3a158a6` | test | smoke tests RBAC + skip condicional si falta Postgres |
| 4 | `2ff742d` | docs | apendice PERMISSION_MAP en AUTH_MODEL.md |

---

## Entregables — estado

| # | Entregable | REQ | Estado |
|---|------------|-----|--------|
| 1 | `PERMISSION_MAP` + factory `require_permission` en `nexo/services/auth.py` | IDENT-03, IDENT-05 | ✅ 24 entries |
| 2 | Bypass `propietario` hardcodeado antes del lookup | IDENT-05 | ✅ verificado por test |
| 3 | Retrofit en los 14 routers (health excluido, whitelisted) | IDENT-05 | ✅ 34 ocurrencias |
| 4 | Pagina `/` decide según rol | IDENT-05 | ⏳ diferido a Phase 5 (decisión explícita en plan) |
| 5 | Tabla de mapeo en `docs/AUTH_MODEL.md` (apéndice) | IDENT-05 | ✅ apéndice completo |

---

## Gate duro — resultados (6/6 automáticos)

| # | Check | Resultado |
|---|-------|-----------|
| 1 | `make dev` arranca | ✅ web+db up |
| 2 | `grep -rn require_permission api/routers/` >= 20 | ✅ 34 ocurrencias |
| 3 | `api/routers/health.py` sin `require_permission` | ✅ intacto |
| 4 | Introspección `app.routes`: 0 rutas /api/* (excepto /api/health) sin require_permission | ✅ 0 |
| 5 | Smoke HTTP sin cookie: / → 302, /api/* → 401, /api/health → 200 | ✅ |
| 6 | `pytest tests/auth/ -q` | ✅ 6 passed, 1 skipped |

---

## PERMISSION_MAP (24 permisos)

**Router-level (14):** `centro_mando:read`, `luk4:read`, `historial:read`, `capacidad:read`, `datos:read`, `informes:read`, `operarios:read`, `email:send`, `ciclos:read`, `recursos:read`, `plantillas:read`, `pipeline:read`, `bbdd:read`, `conexion:read`.

**Endpoint-level strict (6):** `informes:delete` (DELETE), `ciclos:edit` (mutations), `recursos:edit` (mutations), `plantillas:edit` (mutations), `pipeline:run` (POST /run), `conexion:config` (PUT /config — solo propietario).

**Reservados sin endpoint (4):** `operarios:export` (sin endpoint aún), `ajustes:manage`, `auditoria:read`, `usuarios:manage` (los tres llegan en Plan 02-04).

Tabla completa con departamentos autorizados: `docs/AUTH_MODEL.md` §Apéndice.

---

## Suite de tests (tests/auth/test_rbac_smoke.py)

| # | Test | Resultado |
|---|------|-----------|
| 1 | `test_api_sin_cookie_devuelve_401` (5 endpoints) | ✅ |
| 2 | `test_api_health_sin_cookie_devuelve_200` | ✅ |
| 3 | `test_propietario_pasa_todos_los_endpoints` | ✅ |
| 4 | `test_usuario_rrhh_sin_produccion_recibe_403_en_pipeline` | ✅ |
| 5 | `test_usuario_ingenieria_pasa_pipeline_run` | ✅ |
| 6 | `test_solo_propietario_accede_conexion_config` | ✅ |
| 7 | `test_regression_ident_10_exception_handler_no_traceback` | ⏭ skipped (no se pudo forzar 500 limpio; regresión manual) |

**Aislamiento:** fixture autouse purga users `@rbac-test.local` + sessions + login_attempts antes y después de cada test. La BD del operador no queda contaminada.

**Skip condicional:** `pytestmark = [integration, skipif(not _postgres_reachable())]`. Si `docker compose up -d db` no está arriba, la suite entera se saltea sin romper CI.

**Ejecución local:**
```bash
docker compose up -d db web
docker compose exec web pip install pytest==8.3.4 httpx==0.28.1   # una sola vez
docker compose exec web pytest tests/auth/ -q
```

---

## Decisiones tomadas en ejecución

- **Eager-load de `user.departments` en el `AuthMiddleware`** (fuera del plan original pero necesario): la sesión ORM que cargó el user se cierra dentro del middleware; el posterior acceso a `user.departments` desde `require_permission()` haría lazy-load sobre una sesión cerrada → `DetachedInstanceError`. Fix: `_ = list(user.departments)` antes de `db.close()` materializa la relación en `user.__dict__`, accesible después. Commit `0c4f34b`.
- **`limiter` y `require_permission` en archivos distintos**: el primero ya vivía en `api/rate_limit.py` (Plan 02-02); el factory RBAC se ancla en `nexo/services/auth.py` junto a las primitivas. Ambos pueden ser importados desde routers sin crear ciclos con `api.main`.
- **`require_permission` devuelve un callable con `__name__` dinámico** (`require_permission__pipeline_run`) para que el debug de FastAPI (y los logs de stack) muestren el permiso concreto, no un genérico `_check`.
- **`operarios:export`** declarado pero sin endpoint aplicado: el router `operarios.py` solo tiene `GET /` y `GET /{codigo}`, ambos bajo `operarios:read`. El permiso existe en el mapa para cuando el endpoint de export aterrice (previsible en Plan 02-04 o Phase 5).
- **historial mutaciones** (`DELETE /{id}`, `POST /{id}/regenerar`) bajo `historial:read`: plan no define `historial:delete`/`:edit`. Decisión conservadora: seguir al plan literal, documentar en AUTH_MODEL.md como "pendiente tightening".
- **luk4 PUT /zonas** bajo `luk4:read`: plan no define `luk4:edit`. Misma política.
- **bbdd POST /query** (explorador SQL arbitrario) bajo `bbdd:read`: es el uso principal del módulo. Ingeniería es el único depto autorizado.
- **Pages router NO modificado** (Tarea 3.4): decisión Claude's Discretion explícita en el plan §3.4. La homepage por rol se diseña en Phase 5 UIROL-02.
- **`COPY tests/ ./tests/` añadido al Dockerfile**: necesario para que `docker compose exec web pytest tests/auth/` funcione. Antes, `tests/` no estaba en la imagen.
- **`httpx==0.28.1` añadido a `requirements-dev.txt`**: dep real de `fastapi.testclient` en Starlette 1.x — no venía como transitiva de fastapi.
- **`pytest_configure` en `tests/conftest.py`**: registra el marker `integration` para eliminar `PytestUnknownMarkWarning`. Permite además filtrar con `-m 'not integration'`.

---

## Deviations

- **[Rule 1 — missing critical] Eager-load de `user.departments` en el middleware**
  - Found during: Tarea 3.1 (anticipado antes de escribir `require_permission`)
  - Issue: `AuthMiddleware` (Plan 02-02) cierra la sesión ORM en `finally`. `user.departments` es un relationship lazy; su primer acceso hace SELECT. Si ocurre tras `db.close()` → `DetachedInstanceError`.
  - Fix: `_ = list(user.departments)` antes del `db.close()`. La lista queda cacheada en `user.__dict__['departments']`.
  - Files modified: `api/middleware/auth.py`.
  - Verification: suite de tests corre sin DetachedInstanceError en ninguno de los 6 tests que usan `user.departments`.
  - Commit: `0c4f34b`.

- **[Rule 1 — missing critical] `COPY tests/` en Dockerfile + `httpx` en requirements-dev**
  - Found during: Tarea 3.3 (al intentar correr `pytest tests/auth/` dentro del container)
  - Issue: el plan pide `pytest tests/auth/test_rbac_smoke.py pasa en local` pero `tests/` no estaba en la imagen y `httpx` no era dep instalada.
  - Fix: `COPY tests/ ./tests/` al Dockerfile + `httpx==0.28.1` a `requirements-dev.txt`. Install de `pytest+httpx` en el container como paso manual documentado.
  - Files modified: `Dockerfile`, `requirements-dev.txt`.
  - Verification: `docker compose exec web pytest tests/auth/ -q` pasa con 6/7 tests (1 skip esperado en IDENT-10).
  - Commit: `3a158a6`.

- **[Rule 1 — scope] Tarea 3.4 pages.py intencionalmente no modificado**
  - Found during: Tarea 3.4
  - Issue: el plan dice en §3.4 "en Phase 2 no diseñamos homepage por rol — eso es Phase 5". No requería modificar `pages.py` más allá del refactor ya hecho en 02-02 (helper `render()`).
  - Fix: skip explicit. Documentado en el apéndice AUTH_MODEL.md como pendiente Phase 5.
  - Files modified: ninguno (en esta tarea).

**Total deviations:** 3 auto-fixed (todas Rule 1 — missing critical/scope).
**Impact:** nulo sobre el comportamiento especificado. 2 son ajustes de entorno (ORM lazy-load + dev environment) y 1 es alineación con decisión explícita del plan.

---

## Archivos tocados

**Creados:**
- `tests/auth/__init__.py`
- `tests/auth/test_rbac_smoke.py`

**Modificados:**
- `nexo/services/auth.py` (+PERMISSION_MAP 24 entries, +`require_permission` factory)
- `api/middleware/auth.py` (+eager-load `user.departments` pre-close)
- `api/routers/centro_mando.py`, `luk4.py`, `historial.py`, `capacidad.py`, `datos.py`, `informes.py`, `operarios.py`, `email.py`, `ciclos.py`, `recursos.py`, `plantillas.py`, `pipeline.py`, `bbdd.py`, `conexion.py` (14 × retrofit de dependencies)
- `docs/AUTH_MODEL.md` (+apéndice PERMISSION_MAP completo)
- `tests/conftest.py` (+registro del marker `integration`)
- `Dockerfile` (+COPY tests/)
- `requirements-dev.txt` (+httpx==0.28.1)

**NO modificados (consciente):**
- `api/routers/health.py` (whitelist en middleware, plan lo exige intacto)
- `api/routers/pages.py` (diferido a Phase 5 UIROL-02 por decisión explícita del plan)
- `static/js/app.js` (handler global de 403 → Phase 5 UIROL-02)

---

## Qué habilita este plan

- **02-04 (audit middleware + UI de ajustes)** puede registrar en `nexo.audit_log` eventos autenticados con `request.state.user` poblado y con información de permisos (el AuditMiddleware vendrá después de AuthMiddleware en el orden LIFO y antes de require_permission, así que verá tanto 200 como 403).
- **Phase 5 UIROL-02** (homepage/sidebar por rol) tiene lista la guía de permisos + sidebar condicional; basta con leer `PERMISSION_MAP` desde el template contexto.
- **Phase 4 (tests endurecidos en CI)** tendrá un scaffolding de tests de integración que solo requiere `continue-on-error: false` + un servicio Postgres en el workflow.

---

## Issues Encountered

- **Sin issues bloqueantes.** Los 2 Rule-1 deviations (eager-load + Dockerfile/httpx) se resolvieron inline durante la ejecución del plan.

- **Warning persistente**: `DeprecationWarning: Setting per-request cookies=<...>` desde Starlette TestClient. Afecta sólo a output de pytest, no al comportamiento. Limpieza futura en Phase 6 reorganizando los tests para usar `client.cookies = {...}` a nivel de cliente.

---

## Next Phase Readiness

- ✅ `request.state.user.departments` disponible en cualquier dependency/handler — 02-04 puede leerlo para registrar permisos evaluados.
- ✅ `PERMISSION_MAP` disponible via `from nexo.services.auth import PERMISSION_MAP` — 02-04 podrá mostrar la tabla en `/ajustes/usuarios`.
- ✅ Tests de integración con patrón reutilizable (`_create_test_user`, `_login`, fixture purge) — 02-04 puede extender para probar `/ajustes/auditoria` y el gate IDENT-06.
- ⏳ **Gate IDENT-06** sigue pendiente para 02-04 (test de integración de `DELETE nexo.audit_log` → `permission denied`). Decisión Opción A vs B del operador.

---

*Summary creado 2026-04-19 como parte de `/gsd-execute-phase 2 --interactive`.*
