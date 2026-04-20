---
phase: 04-consultas-pesadas
fixed_at: 2026-04-20
review_path: .planning/phases/04-consultas-pesadas/04-REVIEW.md
iteration: 1
findings_in_scope: 6
fixed: 6
skipped: 13
status: complete
---

# Phase 4: Code Review Fix Report

**Fixed at:** 2026-04-20
**Source review:** `.planning/phases/04-consultas-pesadas/04-REVIEW.md`
**Iteration:** 1
**Scope:** 1 CRITICAL + 5 HIGH (MEDIUM/LOW deferred por instrucciones del operador)

**Summary:**
- Findings in scope: 6 (1 CRITICAL + 5 HIGH)
- Fixed: 6
- Deferred to Phase 4.x polish: 13 (7 MEDIUM + 6 LOW)
- Test suite baseline: 140 passed, 1 pre-existing SQL Server failure (SSR no alcanzable)
- Test suite post-fix: 141 passed, 1 pre-existing failure, 22 skipped → **+1 test nuevo (H-05), sin regresiones**

## Finding status table

| ID    | Severity | Status    | Commit / Reason |
|-------|----------|-----------|-----------------|
| CR-01 | CRITICAL | fixed     | `a709035` — QueryTiming middleware ignora responses 4xx/5xx (salvo 504) si el router no marcó `query_executed=True` |
| H-01  | HIGH     | fixed     | `872064a` — `_audit_approval_event` helper escribe `__approval_{approved\|rejected\|cancelled}__` en `audit_log` |
| H-02  | HIGH     | fixed     | `c1f3361` — `CreateApprovalBody.endpoint` valida contra `ALLOWED_ENDPOINTS` frozenset |
| H-03  | HIGH     | fixed     | `882056a` — `/ajustes/solicitudes` precarga `users_map` y template renderiza email |
| H-04  | HIGH     | fixed     | `08725e0` — `_assert_allowed_endpoint` aplicado a PUT/POST `/api/thresholds/{endpoint:path}` |
| H-05  | HIGH     | fixed     | `f9a8849` — `_canonical_json` ordena `recursos`/`modulos` antes de serializar + nuevo test |
| M-01  | MEDIUM   | deferred  | User instruction: MEDIUM/LOW fuera de scope |
| M-02  | MEDIUM   | deferred  | User instruction |
| M-03  | MEDIUM   | deferred  | User instruction |
| M-04  | MEDIUM   | deferred  | User instruction |
| M-05  | MEDIUM   | deferred  | User instruction |
| M-06  | MEDIUM   | deferred  | User instruction |
| M-07  | MEDIUM   | deferred  | User instruction |
| L-01  | LOW      | deferred  | User instruction |
| L-02  | LOW      | deferred  | User instruction |
| L-03  | LOW      | deferred  | User instruction |
| L-04  | LOW      | deferred  | User instruction |
| L-05  | LOW      | deferred  | User instruction |
| L-06  | LOW      | deferred  | User instruction |

## Fixed Issues

### CR-01: QueryTiming middleware writes misleading `query_log` rows for gate rejections

**Commit:** `a709035`
**Files modified:**
- `nexo/middleware/query_timing.py`
- `api/routers/pipeline.py`
- `api/routers/bbdd.py`
- `api/routers/capacidad.py`
- `api/routers/operarios.py`

**Applied fix:**
- Middleware: añadido short-circuit que no persiste fila cuando el router no marcó `request.state.query_executed=True` Y la response es 4xx/5xx (excepto 504 que sí viene de ejecución real iniciada).
- Except-path: también respeta el flag — excepciones en el gate no se loggean; excepciones durante ejecución sí se loggean como `status='error'`.
- Routers: `request.state.query_executed = True` se setea después de pasar el gate (preflight + whitelist cuando aplica) y antes de ejecutar la query.

**Impacto:**
- `/ajustes/rendimiento` ya no cuenta rejections 403/428/503 como "errors" ni infla `n_slow` / `avg_actual_ms`.
- Intentos bloqueados (SQL DDL antes de whitelist, amber sin force, red sin approval) ya no quedan como shadow-log en `params_json`.

**Verificación:** `tests/middleware/test_query_timing.py` (4 tests) sigue verde. El test `test_timing_writes_row_for_bbdd_query` valida que el whitelist-pass → ejecución fallida sigue escribiendo fila (`query_executed=True` se setea antes de que el handler intente conectar SQL Server).

---

### H-01: Approval state mutations not audited

**Commit:** `872064a`
**Files modified:**
- `api/routers/approvals.py`

**Applied fix:**
- Nuevo helper `_audit_approval_event(db, event, approval_id, actor_user_id, request)` que escribe fila en `nexo.audit_log` con:
  - `path='__approval_{approved|rejected|cancelled}__'`
  - `details_json={approval_id, actor_user_id, endpoint, estimated_ms, requested_by}`
- Llamado desde `approve`, `reject`, `cancel` tras la mutación, antes del redirect.
- Best-effort (mismo patrón que `approvals_cleanup.run_once` lines 57-59): si el audit falla, la mutación ya está commiteada y no se tumba.

**Notas:**
- Eventos `expired` ya estaban auditados via `approvals_cleanup.run_once` — no se toca.
- Consume queda trazado vía `query_log.approval_id` — no requiere fila adicional en `audit_log` (sería redundante con el middleware genérico + el query_log row).

---

### H-02: `/api/approvals` endpoint accepts arbitrary strings

**Commit:** `c1f3361`
**Files modified:**
- `nexo/services/thresholds_cache.py`
- `api/routers/approvals.py`

**Applied fix:**
- Nuevo export `nexo.services.thresholds_cache.ALLOWED_ENDPOINTS` como `frozenset[str]` canónico con los 4 endpoints con preflight/postflight.
- `CreateApprovalBody` añade `@field_validator("endpoint")` que rechaza cualquier string fuera de la allowlist con `ValueError` → Pydantic convierte en 422.
- `estimated_ms` ahora `Field(ge=0)` (rechaza negativos).

**Single source of truth:** `ALLOWED_ENDPOINTS` quedará reusado en H-04.

---

### H-03: `/ajustes/solicitudes` renders `user#{id}` instead of email

**Commit:** `882056a`
**Files modified:**
- `api/routers/approvals.py`
- `templates/ajustes_solicitudes.html`

**Applied fix:**
- `page_solicitudes` precarga `users_map = {u.id: u.email for u in UserRepo(db).list_all()}` y lo pasa al contexto.
- Template usa `{{ users_map.get(s.user_id) or 'user#%d' % s.user_id }}` — fallback seguro si el user fue eliminado (dangling FK improbable por FK constraint, pero documentado).
- Aplica a tabla "Pendientes" y "Histórico 30d".

**Opción elegida:** option (b) del review (precargar dict en el router) — diff más pequeño que outerjoin en el repo.

---

### H-04: `/api/thresholds/{endpoint:path}` accepts any path

**Commit:** `08725e0`
**Files modified:**
- `api/routers/limites.py`

**Applied fix:**
- Importa `ALLOWED_ENDPOINTS` de `thresholds_cache` (introducido en H-02).
- Nuevo helper `_assert_allowed_endpoint(endpoint)` que devuelve 404 si el endpoint no está en la allowlist, con mensaje que lista los endpoints válidos.
- Llamado al principio de PUT `/api/thresholds/{endpoint:path}` y POST `/api/thresholds/{endpoint:path}/recalibrate`.

**Beneficio secundario:** `/recalibrate` ya no dispara `compute_factor` (query expensive) para endpoints inexistentes — 404 antes de tocar BD.

---

### H-05: `consume_approval` params equality fragile to list ordering

**Commit:** `f9a8849`
**Files modified:**
- `nexo/services/approvals.py`
- `tests/services/test_approvals.py`

**Applied fix:**
- Nuevo constante `_CANONICAL_SET_FIELDS = frozenset({"recursos", "modulos"})` en `approvals.py`.
- `_canonical_json` ahora ordena los valores de esos campos antes de serializar, solo si son listas de primitivos comparables (str/int/float/bool). Si la lista contiene dicts u objetos mezclados, deja el orden original (evita `TypeError`).
- Otros campos (p.ej. `columns` en bbdd/query que sí tiene orden semántico) se dejan intactos.

**Test añadido:** `test_consume_approval_tolerates_reordered_recursos` — crea approval con `recursos=["A","B"]`/`modulos=["oee","rendimiento"]`, consume con el orden inverso, espera `consumed` sin 403.

**Side effect:** el test existente `test_create_canonicalizes_params` usa keys `{"a","b"}` (no son `recursos`/`modulos`) → sigue verde.

## Skipped / Deferred Issues

Todos los findings MEDIUM (M-01..M-07) y LOW (L-01..L-06) quedan diferidos a Phase 4.x polish por instrucción explícita del operador ("Focus on 1 CRITICAL + 5 HIGH items only"). No hay "trivially fixable" incluidos en esta sesión.

Los hallazgos "checked but clean" del REVIEW.md (SQL injection, XSS, secrets, auth/authz, asyncio cancellation, semaphore leak, CAS atomicity, query_log append-only, HTMX CSRF) siguen siendo correctos tras los fixes aplicados — ninguno de los fixes los alteró.

## Test results

**Baseline (pre-fix):** 140 passed, 1 failed (pre-existing SQL Server timeout), 22 skipped.

**Post-fix:** 141 passed, 1 failed (mismo pre-existing SQL Server timeout), 22 skipped.
- +1 test nuevo: `test_consume_approval_tolerates_reordered_recursos` (H-05).
- Sin regresiones.

Comando ejecutado:
```
NEXO_PG_HOST=localhost NEXO_PG_PORT=5433 NEXO_PG_USER=oee NEXO_PG_PASSWORD=oee \
NEXO_PG_DB=oee_planta NEXO_PG_APP_USER= NEXO_PG_APP_PASSWORD= \
NEXO_SECRET_KEY=testsecretkeytestsecretkeytestsecretkey \
pytest tests/data tests/services tests/routers tests/middleware tests/integration -q
```

El test fallido (`test_run_red_with_valid_approval_executes`) depende de SQL Server dbizaro disponible vía pyodbc; la misma falla aparecía antes de aplicar los fixes y está fuera de scope de este pase (pre-existing, requiere conectividad de producción).

---

_Fixed: 2026-04-20_
_Fixer: gsd-code-fixer (Opus 4.7 1M)_
_Iteration: 1_
