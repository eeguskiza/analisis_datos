---
phase: 04-consultas-pesadas
type: review
depth: standard
status: issues_found
reviewed: 2026-04-20
critical: 1
high: 5
medium: 7
low: 6
files_reviewed: 22
---

# Phase 4: Code Review Report — consultas-pesadas

**Reviewed:** 2026-04-20
**Depth:** standard (per-file + cross-cutting concerns on concurrency/state/security)
**Files reviewed:** 22 of the 38 source files in scope (high-value targets: services, middleware, data layer, routers, key templates).
**Status:** issues_found

## Summary

Phase 4 delivers a coherent preflight/postflight pipeline with credible state-machine semantics, good LISTEN/NOTIFY hygiene and a defensible CAS consume. The core security-sensitive flows (approvals CAS, authorization gates, semaphore+timeout, audit/postflight) are implemented correctly. The Alpine modals render identifiers via Jinja auto-escape, no SQL injection via threshold CRUD, and the only free-form SQL path (`/api/bbdd/query`) keeps its whitelist.

However, the review surfaces:

- **1 CRITICAL** bug: on 4xx/5xx from gate logic or approval consumption, the QueryTiming middleware re-reads `request.state` and silently emits a postflight row for a path the user was NOT authorized to execute, with misleading `status='error'` classification and a potential data-leak of rejected SQL in `params_json`. Compounded by the fact that `_persist` uses the router-populated snapshot even when the request never executed — misleading dashboards and inflating `nexo.query_log` with false-negatives.
- **5 HIGH** bugs: approval mutations are NOT audited (audit silence on a sensitive state machine); `/api/approvals` accepts any arbitrary `endpoint` string (no allowlist — pollutes the approval table and risks workflow confusion); `approvals.py` renders `user#{id}` instead of the email, hiding the actor on the propietario screen (and breaks accountability in audit correlation); missing endpoint allowlist in `/api/thresholds/{endpoint:path}` PUT/POST (anyone with `limites:manage` can create "phantom" threshold rows via a mistyped path, and `/recalibrate` silently accepts any path the `path:` converter catches); `consume_approval` uses a DB-side timestamp (`now() AT TIME ZONE 'UTC'`) but the `params_json` equality check compares against a client-side `json.dumps(sort_keys=True)` and is susceptible to false failures when Python dict field order changes across requests (e.g. `recursos` ordering).
- **7 MEDIUM** issues: function size/nesting in `api/routers/capacidad.py::capacidad` (>200 lines, nested dict mutation, deep nesting); Chart.js / Tailwind loaded from public CDNs with no Subresource Integrity (SRI) despite LAN-only deployment; thresholds cache global mutables (`_loaded_at_global`, `_cache`) protected by `Lock` but with data-race between `needs_reload` check and reload (`full_reload` runs unlocked outside the CAS); `pipeline/run` re-declares `Depends(require_permission("pipeline:run"))` both at router level and as parameter (double auth check, minor perf nit but semantic duplication); cleanup_scheduler triggers can fire twice if the loop wakes <60s late AND the next iteration re-computes `_seconds_until` before the minute-before-target window closes (off-by-one vs `_TRIGGER_TOLERANCE_S`); `notify_changed` opens a fresh psycopg2 connection per edit without retry/backoff (every save is one round-trip that can fail the HTTP request); `pipeline/run` request state `estimated_ms` / `params_json` is populated but `approval_id` only populated for red path — postflight correctly logs, but amber+force runs lose the link back to an approval in the log (semantic mismatch with D-15 audit aspiration).
- **6 LOW** issues: minor style/typing gaps, pyflakes-level dead imports, and docstring drift (see below).

## Findings by severity

### CRITICAL

#### CR-01 — QueryTiming middleware writes misleading `query_log` rows for gate rejections and may leak user SQL in `params_json`

**File:** `nexo/middleware/query_timing.py:94-164`

**Code excerpt:**

```python
async def dispatch(self, request: Request, call_next) -> Response:
    ...
    t0 = time.monotonic()
    try:
        response = await call_next(request)
        actual_ms = int((time.monotonic() - t0) * 1000)
        estimated_ms = getattr(request.state, "estimated_ms", None)
        approval_id  = getattr(request.state, "approval_id",  None)
        params_snapshot = getattr(request.state, "params_json", None)
        ...
        query_status = _classify_status(
            actual_ms=actual_ms,
            endpoint_key=endpoint_key,
            http_status=response.status_code,  # 403/428/504 all pass through here
        )
        _persist(
            user_id=user.id,
            endpoint=endpoint_key,
            params_json=params_snapshot,      # ← contains user SQL even when gate rejected
            estimated_ms=estimated_ms,
            actual_ms=actual_ms,
            rows=None,
            status=query_status,              # 'error' for 403/428 — "failed query"
            approval_id=approval_id,
            ip=...,
        )
```

**Issue (two-fold):**

1. **Mis-classification in `query_log.status`.** Routers populate `request.state.estimated_ms` / `params_json` BEFORE the gate check (see `bbdd.py:560-563`, `pipeline.py:130-136`, `capacidad.py:88-91`, `operarios.py:111-114`). When the gate rejects (403 red-sin-approval, 428 amber-sin-force, 503 approvals-unavailable), `call_next` returns a non-2xx and `_classify_status` stamps `status='error'`. The log reads as "the query failed at runtime" when in reality the query **never executed**. That pollutes `/ajustes/rendimiento` (n_slow, avg_actual_ms over 0-cost rejected rows), poisons `factor_learning.compute_factor` (which filters by `status IN ('ok','slow')` — so rejected rows are excluded from the factor, fine, BUT the `n_runs` counter on `/ajustes/limites` counts both statuses and includes these noop rejections, misleading the "Recalibrar" button eligibility).
2. **Data exposure on rejected SQL.** For `/api/bbdd/query`, `params_snapshot` is the full user-supplied SQL (`params_json_str = json.dumps({"sql": sql, ...})`). The whitelist `_validate_sql` runs AFTER the preflight gate (intentional per the docstring, for UX). A user who submits a payload that hits `block_ms` threshold gets a 403 reject BEFORE `_validate_sql` — but `query_log` still stores the full SQL, including any tokens the user typed for later "would-have-failed-whitelist" queries (DDL attempts, malformed SQL). That fact is not a secret, but it turns `nexo.query_log.params_json` into a shadow log of blocked/attempted queries visible to any propietario reviewing `/ajustes/rendimiento`. Worse, combined with issue 1, those queries appear as "error" runs rather than "rejected-by-policy" runs.

**Root cause:** the gate logic runs inside the handler (where `state.params_json` is already set), and the middleware unconditionally persists on any response ≥ 400 that passed through the `_TIMED_PATHS` dispatch. There is no signal for "was the query actually executed?".

**Concrete fix:**

Introduce a "executed" marker set by the router AFTER the gate passes (and AFTER `consume_approval` for red). Only persist when the marker is set, or when the status is `timeout`/`ok`/`slow`. Example:

```python
# In each router, right after consume_approval / amber force / green path:
request.state.query_executed = True

# In middleware:
executed = bool(getattr(request.state, "query_executed", False))
if not executed and response.status_code >= 400:
    # Gate rejection — don't pollute query_log.
    return response
```

Alternatively, classify `status='rejected'` (new enum value) and emit a separate log row with `params_json=None` (to avoid storing user SQL that was blocked). Either fix MUST land before Phase 4 verification.

---

### HIGH

#### H-01 — Approval state mutations (approve/reject/cancel/expire/consume) are NOT written to `nexo.audit_log`

**File:** `api/routers/approvals.py:129-202`, `nexo/services/approvals.py:159-186`, `nexo/services/approvals_cleanup.py:30-67`

**Excerpt:**

```python
# api/routers/approvals.py:129
def approve(approval_id: int, request: Request, db: DbNexo) -> RedirectResponse:
    user = getattr(request.state, "user", None)
    ...
    svc.approve(db, approval_id, user.id)   # ← mutates status to 'approved'
    logger.info("approval approved id=%d by=%s", approval_id, user.email)  # only stdout
    return RedirectResponse("/ajustes/solicitudes?ok=approved", status_code=303)
```

**Issue:** The approval flow (D-15) is the security-critical gate for red-level queries — a propietario approving a request lets a user bypass the block threshold and execute a potentially multi-hour pipeline. None of `approve`, `reject`, `cancel`, or `expire_stale` calls `AuditRepo.append`. The only audit trail is the generic request-level audit middleware (which records HTTP method/path/status but NOT the `approval_id` or the resulting state transition). `query_log_cleanup` and `factor_auto_refresh` DO audit their cleanup runs — the inconsistency is glaring.

Compare with `approvals_cleanup.py:44` (`path='__cleanup_approvals__'` audit entry) — the system audits the *cleanup job* but not the *decision*.

**Fix:** Add audit calls inside `svc.approve`/`reject`/`cancel` OR in the router after the service call, before the redirect. Suggested path values: `__approval_approved__`, `__approval_rejected__`, `__approval_cancelled__`. Details JSON should include `{approval_id, user_id, endpoint, estimated_ms}`. The middleware already captures the HTTP fact; these rows add the state transition context required to reconstruct "who approved what, when".

---

#### H-02 — `POST /api/approvals` accepts arbitrary `endpoint` strings with no allowlist

**File:** `api/routers/approvals.py:57-96`

**Excerpt:**

```python
class CreateApprovalBody(BaseModel):
    endpoint: str     # ← no validator, no Literal, no field_validator
    params: dict
    estimated_ms: int

@router.post("/api/approvals")
async def create(body: CreateApprovalBody, ...) -> dict:
    approval_id = svc.create_approval(
        db,
        user_id=user.id,
        endpoint=body.endpoint,           # stored as-is
        params=body.params,
        estimated_ms=body.estimated_ms,
    )
```

**Issue:** A user can POST `{"endpoint": "../admin/wipe_db", "params": {"confirm": true}, "estimated_ms": 0}` and it will persist as a `pending` row. The propietario UI (`ajustes_solicitudes.html:41`) renders `{{ s.endpoint }}` (auto-escaped by Jinja — no XSS) but the **propietario could approve an arbitrary-endpoint row**, creating a row with `status='approved'` that does not correspond to any real executable route. Since consumption is also checked per-endpoint (routers only consume for their own endpoint key), this does NOT allow privilege escalation — but it allows `query_approvals` table pollution, confuses dashboards, and signals weak input validation at a critical boundary.

**Fix:**

```python
from typing import Literal

class CreateApprovalBody(BaseModel):
    endpoint: Literal["pipeline/run", "bbdd/query", "capacidad", "operarios"]
    params: dict
    estimated_ms: int = Field(ge=0)
```

Validate the same allowlist that lives in `QueryTimingMiddleware._TIMED_PATHS.values()` — they must agree.

---

#### H-03 — `/ajustes/solicitudes` renders `user#{id}` instead of the user's email, obscuring who requested and who approved

**File:** `templates/ajustes_solicitudes.html:40, 91`, `api/routers/approvals.py:207-242`, `nexo/data/repositories/nexo.py:643-671`

**Excerpt:**

```jinja
{% for s in pending %}
  <tr>
    <td>#{{ s.id }}</td>
    <td>user#{{ s.user_id }}</td>            ← shows integer id, not email
    <td>{{ s.endpoint }}</td>
    ...
```

**Issue:** The propietario needs to decide whether to approve an operation that can burn 15+ minutes of compute. They see `user#7` in the grid, not the requesting user's email. This:
- Breaks D-06/D-13 UX (propietario cannot correlate request to the human without opening Postgres).
- Reduces accountability in post-hoc review (the audit log *does* track `user_id` but with email lookup missing on screen).
- Creates asymmetry with `/ajustes/auditoria` (which renders email via `AuditRepo.list_filtered`'s outerjoin).

`ApprovalRepo.list_pending` / `list_recent_non_pending` returns `QueryApprovalRow` DTOs that only carry `user_id` — the fix is either (a) add an `outerjoin` with `NexoUser` in the repo (following the pattern at `repositories/nexo.py:196-215`), or (b) preload a `{user_id: email}` dict from `UserRepo.list_all()` in the router and hand it to the template.

**Fix:** option (b) is the smaller diff:

```python
# api/routers/approvals.py::page_solicitudes
users_map = {u.id: u.email for u in UserRepo(db).list_all()}
return render(..., {"pending": pending, "historico": historico,
                    "users_map": users_map, ...})
```

And in the template: `{{ users_map.get(s.user_id, 'user#%d' % s.user_id) }}`.

---

#### H-04 — `PUT /api/thresholds/{endpoint:path}` and `/recalibrate` accept any `endpoint` string (path converter) without allowlist check

**File:** `api/routers/limites.py:88-214`

**Excerpt:**

```python
@router.put("/api/thresholds/{endpoint:path}")
def update(endpoint: str, body: UpdateThresholdBody, ...) -> dict:
    repo = ThresholdRepo(db)
    current = repo.get(endpoint)
    if current is None:
        raise HTTPException(status_code=404, detail=...)
```

**Issue:** `path:` converter allows slashes (`pipeline/run` is the common case). The router correctly 404s if the row doesn't exist — so a hostile/typoed call like `/api/thresholds/anything/../secret` ends up as a read-miss. **BUT**: (1) it routes any sub-path through this handler, competing with other routes, and (2) `/recalibrate` emits a `NOTIFY` on any string that *does* exist, and more importantly triggers a full `compute_factor` scan of `query_log` (expensive for a propietario-only action — rate-limit territory). There is no allowlist, no rate-limit, no endpoint normalization.

**Fix:** Add a guard at the top:

```python
_ALLOWED_ENDPOINTS = frozenset(["pipeline/run", "bbdd/query", "capacidad", "operarios"])

def _assert_allowed(endpoint: str) -> None:
    if endpoint not in _ALLOWED_ENDPOINTS:
        raise HTTPException(404, f"Endpoint {endpoint!r} no es editable")
```

Same fix applies to `approvals.CreateApprovalBody.endpoint` (H-02). Keep one canonical source of truth (`nexo.services.thresholds_cache.ALLOWED_ENDPOINTS`).

---

#### H-05 — `consume_approval` params equality check is susceptible to Python dict ordering / list ordering drift → false 403s

**File:** `nexo/services/approvals.py:38-47, 125-156`, `nexo/data/repositories/nexo.py:733-782`

**Excerpt:**

```python
def _canonical_json(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)
# ...
def consume(self, *, approval_id, user_id, current_params_json) -> ...:
    sql = text("... WHERE ... AND params_json = :pj RETURNING ...")
```

```python
# api/routers/pipeline.py:_build_pipeline_params
return {
    "fecha_desde": ..., "fecha_hasta": ...,
    "modulos": list(req.modulos),      # ← order-sensitive
    "source": req.source,
    "recursos": list(recursos),         # ← order-sensitive, Optional[None]→[]
    ...
}
```

**Issue:** `_canonical_json` sorts dict keys but leaves list element order intact. Pydantic does NOT normalize list order; so two logically equivalent payloads (same recursos, same modulos, in different JSON array order) will hash to different canonical strings. Flow:

1. User POSTs `/api/approvals` with `recursos: ["A", "B"]` → stored canonical JSON has `"recursos":["A","B"]`.
2. Propietario approves.
3. User clicks "Ejecutar ahora" → frontend re-builds the form, may serialize `recursos: ["B", "A"]` (alphabetic sort on the UI side, or different form-field order on `<select multiple>`).
4. Router re-computes `_canonical_json` → `"recursos":["B","A"]`.
5. `ApprovalRepo.consume` `WHERE params_json = :pj` → 0 rows → falls through to the diagnostic path → 403 "Parámetros cambiaron".

User sees a confusing error and must re-request approval. The mitigation is fragile: depends on the frontend rebuilding the exact same payload it sent.

**Fix:** canonicalize inside `_canonical_json` by sorting list members that the contract declares as "sets" (recursos, modulos). Either:

```python
def _canonical_json(obj: dict) -> str:
    normalized = {
        k: sorted(v) if isinstance(v, list) and k in {"recursos", "modulos"} else v
        for k, v in obj.items()
    }
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False)
```

…or document the ordering contract in `PipelineRequest` and enforce it in a Pydantic `field_validator` (so both the approval-creation and approval-consumption requests produce the same canonical order). Add a unit test covering list-order drift (`test_consume_approval_tolerates_reordered_recursos`).

---

### MEDIUM

#### M-01 — `api/routers/capacidad.py::capacidad` is 200+ lines with deep nesting and repeated dict mutation — exceeds the <50 line / <4-level guideline

**File:** `api/routers/capacidad.py:60-258`

**Issue:** Function spans nearly 200 lines, mixes preflight gate, SQL fetching, pandas-less data reshaping, P10 computation, and output assembly. Deep nesting (5 levels inside the `for key, entry in base.items()` block at 183-212). Hard to test and hard to reason about. Similar (smaller) issue in `api/routers/operarios.py::ficha_operario` (~160 lines, inline SQL, repeated pattern of "fetch rows → reshape to list").

**Fix:** Extract 3 helpers: `_fetch_real_production(conn, fi, ff)`, `_compute_cycle_baselines(conn, pairs)`, `_assemble_output(base, cycles)`. Same refactor is warranted on `operarios.py`. No behavior change needed; pure readability/testability win.

---

#### M-02 — Chart.js / Tailwind / Alpine / HTMX loaded from public CDNs without Subresource Integrity

**File:** `templates/base.html:7-11`

**Excerpt:**

```html
<script src="https://cdn.tailwindcss.com"></script>
<script defer src="https://unpkg.com/htmx.org@2.0.4"></script>
<script defer src="https://unpkg.com/htmx-ext-sse@2.2.2/sse.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.8/dist/cdn.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
```

**Issue:** Nexo is LAN-only (per `CLAUDE.md`), so a classic CDN-compromise XSS vector is limited, BUT the browser still fetches from public internet on first load. If the LAN gateway allows egress to `cdn.jsdelivr.net` and `unpkg.com`, a supply-chain swap at those CDNs executes inside the propietario's session — with full authorization. SRI hashes eliminate that class of attack.

**Fix:** Add `integrity="sha384-..."` + `crossorigin="anonymous"` to each `<script>` tag; pin the hashes (use https://www.srihash.org/). Alternative: vendor the JS under `static/js/vendor/` and serve locally. Vendoring is the cleaner LAN answer.

---

#### M-03 — Thresholds cache has a TOCTOU between the stale check and `full_reload`

**File:** `nexo/services/thresholds_cache.py:148-169`

**Excerpt:**

```python
def get(endpoint: str) -> Optional[ThresholdEntry]:
    global _loaded_at_global
    now = datetime.now(timezone.utc)
    needs_reload = (
        _loaded_at_global is None
        or (now - _loaded_at_global).total_seconds() > FALLBACK_REFRESH_SECONDS
    )
    if needs_reload:
        full_reload()    # ← runs OUTSIDE _cache_lock

    with _cache_lock:
        return _cache.get(endpoint)
```

**Issue:** The read of `_loaded_at_global` and the decision to `full_reload` happen unlocked. Under high-concurrency load on a single worker with many threads (uvicorn `--workers 1` but multiple thread pool threads from `asyncio.to_thread`), N threads may all see `needs_reload=True` and each trigger a `full_reload` — N round-trips to Postgres, N full scans, last-writer-wins. Not a correctness bug (idempotent) but 100-1000× unnecessary load at the 5-min boundary.

**Fix:** Guard the staleness decision with a singleflight:

```python
_reload_lock = Lock()

def get(endpoint: str) -> Optional[ThresholdEntry]:
    ...
    if needs_reload:
        with _reload_lock:
            # Re-check under lock — another thread may have already reloaded.
            if (_loaded_at_global is None
                    or (now - _loaded_at_global).total_seconds() > FALLBACK_REFRESH_SECONDS):
                full_reload()
```

---

#### M-04 — `cleanup_scheduler` double-trigger risk on same wake-up when jobs are scheduled close together

**File:** `nexo/services/cleanup_scheduler.py:97-140`

**Issue:** The tolerance window `_TRIGGER_TOLERANCE_S = 60` and the 3 jobs spaced at 03:00/03:05/03:10 (300s apart) work today. But the logic `abs(sec_qlog - next_sec) < 60` runs after `await asyncio.sleep(next_sec)`; if the sleep over-shoots by >60s (rare on a busy loop with GIL contention, but possible on a swapping host), the tolerance window slips and a job can be SKIPPED for that week. Secondary concern: if two jobs fall within <60s of each other (say, a future D-20 change to make them 03:00/03:00:30), they ALL fire on the first wake-up (correct), but the `next_sec` recomputation at the top of the next iteration is done BEFORE the just-fired jobs are marked as "ran this week" — so the next iteration would sleep until next week for that DOW (correct) but the logic is fragile to maintenance.

**Fix:** Track last-run timestamps per job and skip if `last_run_dt > (now - 7d)`. More robust than relying on tolerance math. `apscheduler` exists precisely for this; adopting it is a modest dep add.

---

#### M-05 — `thresholds_cache.notify_changed` fails open on NOTIFY errors (routers swallow via generic 500 handler)

**File:** `nexo/services/thresholds_cache.py:172-213`, `api/routers/limites.py:125, 201`

**Issue:** Each edit from `/ajustes/limites` opens a *fresh* psycopg2 connection for `pg_notify`, with no retry/backoff. If Postgres is momentarily unreachable (network blip, VACUUM lockout), the PUT handler raises, user sees 500, and the UPDATE *was already committed* via `ThresholdRepo.update`. The fallback 5-min cache safety-net in `get()` will eventually catch up, but the propietario sees "error" on what was actually a successful save.

**Fix:** Wrap `notify_changed` in a best-effort try/except and log-and-continue:

```python
try:
    thresholds_cache.notify_changed(endpoint)
except Exception:
    logger.exception("notify_changed failed for %s — cache will refresh within 5min", endpoint)
```

(`factor_auto_refresh.run_once` already does this at line 108-114 — apply the same pattern in the PUT/recalibrate handlers.)

---

#### M-06 — `pipeline/run` has duplicate `Depends(require_permission("pipeline:run"))` at router and parameter level

**File:** `api/routers/pipeline.py:98-107`

**Excerpt:**

```python
@router.post(
    "/run",
    dependencies=[Depends(require_permission("pipeline:run"))],   # 1st call
)
async def run(
    req: PipelineRequest,
    request: Request,
    db: DbNexo,
    user=Depends(require_permission("pipeline:run")),              # 2nd call
):
```

**Issue:** `require_permission("pipeline:run")` returns a fresh closure function per call, so FastAPI treats these as two distinct dependencies and executes both — two permission lookups, two potential exceptions, two audit entries (if audit traps raise). The same pattern appears in `bbdd.py:519`, `capacidad.py:69`, `operarios.py:82`. No security bug, but mild performance hit (~µs) and confusing to readers.

**Fix:** memoize the factory or pass the `user` from the router dep list:

```python
_DEPS_PIPELINE_RUN = Depends(require_permission("pipeline:run"))
@router.post("/run", dependencies=[_DEPS_PIPELINE_RUN])
async def run(req: PipelineRequest, request: Request, db: DbNexo, user=_DEPS_PIPELINE_RUN):
    ...
```

(Same `Depends` instance → FastAPI de-dupes.)

---

#### M-07 — `approval_id` on amber+force runs is not populated in `query_log`

**File:** `api/routers/pipeline.py:162-173`, `bbdd.py:588-597`, `capacidad.py:113-123`, `operarios.py:136-146`

**Issue:** When the level is `red`, the router does `consume_approval(...)` and sets `request.state.approval_id = approval.id`. When the level is `amber` with `force=true`, the router skips consumption (correct per D-05) — but also leaves `request.state.approval_id` unset. A clean amber run, therefore, has `approval_id=NULL` in `query_log`, which is correct. HOWEVER, the issue surfaces in edge-cases: a user clicks "Ejecutar ahora" from `/mis-solicitudes` for an *amber* approval (bug path: the UI only renders "Ejecutar ahora" for approved rows, but a user could construct the URL `?approval_id=X` manually on a `/pipeline` page, skipping the modal, and the backend would still evaluate amber+force=true WITHOUT consuming the approval). So the approval stays in `status='approved'` indefinitely until it expires — single-use invariant violated for amber.

**Fix:** In the routers, when `force=True` AND `approval_id is not None` AND level is `amber`, still call `consume_approval` (or at least `mark_consumed_amber`) to maintain the single-use invariant. Simpler: amber doesn't need approval, so reject `approval_id` on amber:

```python
elif est.level == "amber":
    if not req.force:
        raise HTTPException(428, ...)
    if req.approval_id is not None:
        # Amber doesn't use approvals — refuse to silently discard it.
        raise HTTPException(400, "approval_id no se usa en nivel amber")
```

---

### LOW

#### L-01 — `schema_guard.py` imports from legacy shim `nexo.db.models` instead of canonical `nexo.data.models_nexo`

**File:** `nexo/data/schema_guard.py:28`, note at line 26-27.

**Fix:** Migrate import to `nexo.data.models_nexo` once all Phase 2/3 consumers are done. The shim is documented (`models_nexo.py:1-8`), but Phase 4 added three new tables to `models_nexo.py` and the shim path is now routinely updated from two locations.

---

#### L-02 — Inconsistent `decided_by` column usage for reject

**File:** `nexo/data/repositories/nexo.py:693-708`

**Issue:** `ApprovalRepo.reject` stores `decided_by` in `approved_by` column — docstring says it's intentional (simmetric semantics), but schema calls the column `approved_by`. Readers of the DB without reading the docstring will misinterpret it. Minor clarity issue — rename column or add a computed `decided_by` view.

---

#### L-03 — `_fallback_*_factor_ms` floats used with `int(...)` cast — silent truncation

**File:** `nexo/services/preflight.py:48-50, 107, 127, 164`

**Issue:** `estimated = int(n_recursos * n_dias * factor)` truncates toward zero — small rounding bias on the boundary ms. Not a bug per se, but consider `round()` for consistency with the median-computed factor.

---

#### L-04 — Unused variable `now` in `cleanup_scheduler.cleanup_loop`

**File:** `nexo/services/cleanup_scheduler.py:111`

**Issue:** `now = datetime.now(timezone.utc)` is captured but only used for `now.day <= 7` in the factor_auto_refresh block. Rename or move it inline to reduce scope.

---

#### L-05 — `humanize_ms` falls back to `~?` on NaN — no unit test but trivial; add one

**File:** `static/js/app.js:435-447`

**Fix:** Add a Playwright/Jest unit test covering: `NaN`, `null`, `undefined`, 500, 12000, 125000, 4000000. Lack of test = regression risk on future refactor.

---

#### L-06 — `api/routers/rendimiento.py::_parse_iso_datetime` accepts `YYYY-MM-DD` via length==10 check — fragile

**File:** `api/routers/rendimiento.py:52-64`

**Issue:** `if len(value) == 10` is a magic heuristic. Better: try parsing as `date` first, fall back to `datetime.fromisoformat`. Edge case: `2026-04-20` vs `2026-4-20` (9 chars) — latter is a valid ISO-8601 "reduced precision" that `datetime.fromisoformat` (Python 3.11+) accepts but `len() == 10` rejects.

---

## Summary table

| Severity | Count | Blockers? |
|----------|-------|-----------|
| CRITICAL | 1     | Yes — CR-01 must fix before merge / Phase 4 verification |
| HIGH     | 5     | Strongly recommended before Sprint 3 closure |
| MEDIUM   | 7     | Address opportunistically in Phase 4.x polish |
| LOW      | 6     | Backlog |
| **Total**| **19**|           |

## Top 3 findings

1. **CR-01** — QueryTiming middleware persists bogus `status='error'` rows with full user SQL in `params_json` when the gate rejects. Data-quality bug in `/ajustes/rendimiento` + latent SQL-exposure risk. Single biggest fix-or-rollback blocker.
2. **H-01** — Approval state mutations are not audited. A red-level query is the most sensitive thing a propietario can authorize; the absence of `__approval_approved__` / `__approval_rejected__` audit entries makes post-incident forensics impossible.
3. **H-05** — CAS `params_json` equality check is fragile to list ordering drift, causing false 403 on "Ejecutar ahora" flows and forcing user to re-request approval. Trivial fix in `_canonical_json`, high UX impact.

## Checked but clean (per category)

- **SQL injection:** `api/routers/bbdd.py::_validate_sql` whitelist remains intact; `rendimiento.py` uses bind params (`:ep`, `:df`, `:dt`); `capacidad.py`, `operarios.py` use bind params only; only identifiers embedded in f-strings (schema/table) are pre-sanitized with `^[\w]+$` regex.
- **XSS on approval flow templates:** Jinja autoescape on; no `|safe` filters; Alpine `x-text` bindings render safely; no `innerHTML` assignment from user-controlled data.
- **Secrets leakage:** `init_nexo_schema.py:269` logs `pg_user:***@host` — no password in log; `thresholds_cache.notify_changed` uses `settings.effective_pg_user/pg_password` — no logging of credentials; no hardcoded secrets in any reviewed file.
- **Authentication/authorization:** all new endpoints use `require_permission` or explicit `request.state.user` + `HTTPException(401)` fall-through. Propietario-only endpoints (`limites.py`, `rendimiento.py`, `approvals.py` approve/reject/count) correctly use empty-list permissions (`ajustes:*` / `rendimiento:read` / `aprobaciones:manage`).
- **asyncio cancellation:** `pipeline_lock.run_with_lock` correctly wraps `asyncio.to_thread` in `wait_for`; `cleanup_scheduler.cleanup_loop` re-raises `CancelledError`; `thresholds_cache.start_listener` uses `stop_event` + `stop_event.wait(timeout=...)` for interruptible backoff. Shutdown sequence in `api/main.py:86-104` is correct (stop_event → cancel → await).
- **Semaphore slot leak:** `async with pipeline_semaphore` in `run_with_lock` (pipeline_lock.py:99-103) correctly releases on exception (wait_for raises TimeoutError after cancellation is propagated; `async with` guarantees release).
- **CAS atomicity:** `ApprovalRepo.consume` uses single `UPDATE ... WHERE consumed_at IS NULL AND status='approved' ... RETURNING` — concurrent consume attempts either succeed once (returning row) or return None (no row matched); both callers handle None. Postgres row-level lock holds across the UPDATE.
- **`nexo.query_log` append-only:** no UPDATE/DELETE in `QueryLogRepo`; retention via `query_log_cleanup.run_once` uses parametrized `DELETE WHERE ts < :cutoff`.
- **HTMX CSRF on badge poll:** `GET /api/approvals/count` is idempotent, propietario-only, and returns only a rendered count span — no state mutation, acceptable for GET.

---

_Reviewed: 2026-04-20_
_Reviewer: gsd-code-reviewer (standard depth)_
_Files inspected: nexo/services/{approvals, pipeline_lock, thresholds_cache, cleanup_scheduler, preflight, factor_learning, factor_auto_refresh, query_log_cleanup, approvals_cleanup}.py; nexo/middleware/query_timing.py; nexo/data/{repositories/nexo, models_nexo, dto/query, schema_guard}.py; api/main.py; api/models.py; api/routers/{approvals, limites, rendimiento, pipeline, bbdd, capacidad, operarios}.py; templates/{ajustes_solicitudes, mis_solicitudes, ajustes_limites, ajustes_rendimiento, base, pipeline}.html; static/js/app.js; scripts/init_nexo_schema.py._
