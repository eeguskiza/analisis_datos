# Phase 4: Consultas pesadas — Pattern Map

**Mapped:** 2026-04-19
**Files analyzed:** 27 new/modified (8 tests + 7 new services/middleware + 4 router mods + 3 new routers + 5 new templates + 1 JS + models/DTO/repo extensions + main.py/deps.py/env)
**Analogs found:** 24 / 27 (3 files have no exact analog — listed in §No Analog Found)

Canonical analog files (by source path):

- `nexo/data/models_nexo.py` — ORM declarative Base + Column shape
- `nexo/data/repositories/nexo.py` — UserRepo / AuditRepo (paginated + streaming queries)
- `nexo/data/dto/nexo.py` — Pydantic v2 frozen Row DTOs
- `nexo/data/dto/base.py` — `ROW_CONFIG` helper
- `nexo/data/schema_guard.py` — critical_tables + auto-migrate fallback
- `nexo/services/auth.py` — service layer (module-level singletons + pure functions + `require_permission` factory)
- `api/middleware/audit.py` — Starlette `BaseHTTPMiddleware` + body sanitization + best-effort persistence
- `api/middleware/auth.py` — request.state.user population + public path whitelist
- `api/routers/auditoria.py` — propietario-only router with paginated + filtered table + CSV export
- `api/routers/usuarios.py` — propietario-only CRUD form-based router
- `api/routers/pipeline.py` — async SSE streaming endpoint (current minimal form)
- `api/routers/pages.py` — Jinja2 HTML page router with `require_permission`
- `api/routers/capacidad.py` / `api/routers/operarios.py` — date-range endpoints (retrofit targets)
- `api/routers/bbdd.py` — validator-before-execute pattern (whitelist)
- `api/main.py` — lifespan + middleware chain LIFO + router registration
- `api/deps.py` — `Annotated[...]` DB session deps + `render()` helper
- `api/models.py` — flat Pydantic request/response models
- `templates/ajustes_auditoria.html` — filter form + paginated table + CSV button
- `templates/ajustes_usuarios.html` — Alpine modal (create/edit/reset) with x-data/x-show
- `templates/ajustes.html` — hub with navigation cards
- `templates/base.html` — CDN scripts (Chart.js 4.4.7 already loaded) + sidebar nav_items
- `tests/data/test_nexo_repository.py` — integration tests with `db_nexo` fixture + contract tests via `inspect.getsource`
- `tests/data/test_schema_guard.py` — schema_guard kwarg-injection test pattern
- `tests/data/conftest.py` — `_postgres_reachable` + `db_nexo` fixture with rollback
- `tests/auth/test_rbac_smoke.py` — TestClient + cookie-based auth smoke tests

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `nexo/data/models_nexo.py` (extend: QueryLog, QueryThreshold, QueryApproval) | model/ORM | write-heavy (CRUD) | `nexo/data/models_nexo.py` (self, existing classes) | exact (additive) |
| `nexo/data/dto/query.py` (new) | DTO | transform | `nexo/data/dto/nexo.py` | exact |
| `nexo/data/repositories/nexo.py` (extend: QueryLogRepo, ThresholdRepo, ApprovalRepo) | repository | mixed (R + append-only W + CAS) | `AuditRepo` (same file) + `UserRepo` | exact |
| `nexo/data/schema_guard.py` (update CRITICAL_TABLES) | config | read | self (kwarg injection pattern) | exact |
| `nexo/services/preflight.py` (new) | service | transform (pure) | `nexo/services/auth.py` (pure function module) | role-match |
| `nexo/services/thresholds_cache.py` (new, with LISTEN listener) | service | in-memory cache + async pub/sub | `nexo/services/auth.py` module-level `_ph`/`_serializer` singleton | partial (no direct analog for LISTEN) |
| `nexo/services/approvals.py` (new, CAS consume) | service | write (CAS + state machine) | `nexo/services/auth.py` session mgmt (create/revoke/expire) | role-match |
| `nexo/services/pipeline_lock.py` (new) | service | concurrency primitive | `nexo/services/auth.py` module-level `_ph` singleton pattern | partial |
| `nexo/services/query_log_cleanup.py` (new) | service/job | batch write | (none; first scheduled job) | no analog |
| `nexo/services/approvals_cleanup.py` (new) | service/job | batch write | (none; first scheduled job) | no analog |
| `nexo/services/factor_auto_refresh.py` (new) | service/job | transform + write | (none; first scheduled job) | no analog |
| `nexo/middleware/__init__.py` (new dir) | package init | — | `api/middleware/__init__.py` (empty marker) | exact |
| `nexo/middleware/query_timing.py` (new) | middleware | request/response wrap + write | `api/middleware/audit.py` | exact |
| `api/routers/pipeline.py` (modify: +preflight; accept force+approval_id) | router | request-response + streaming | self + `api/routers/bbdd.py` validator pattern | exact |
| `api/routers/bbdd.py` (modify: inject preflight before whitelist) | router | request-response | self (validator pattern) | exact |
| `api/routers/capacidad.py` (modify: preflight if >90d) | router | request-response | self (date-range) | exact |
| `api/routers/operarios.py` (modify: preflight if >90d) | router | request-response | self (date-range) | exact |
| `api/routers/approvals.py` (new) | router | CRUD | `api/routers/usuarios.py` | exact |
| `api/routers/limites.py` (new) | router | CRUD + cache-invalidate | `api/routers/usuarios.py` + threshold-specific | role-match |
| `api/routers/rendimiento.py` (new) | router | read + aggregates | `api/routers/auditoria.py` (filter + paginate + export) | exact |
| `api/routers/pages.py` (extend: /ajustes/{limites,solicitudes,rendimiento}, /mis-solicitudes) | router | HTML rendering | self (existing handlers) | exact |
| `templates/ajustes.html` (extend links) | template | — | self | exact |
| `templates/ajustes_limites.html` (new) | template | form submit | `templates/ajustes_usuarios.html` (form + Alpine) | role-match |
| `templates/ajustes_solicitudes.html` (new) | template | table + actions + polling | `templates/ajustes_auditoria.html` + `ajustes_usuarios.html` | exact |
| `templates/ajustes_rendimiento.html` (new) | template | filters + table + chart | `templates/ajustes_auditoria.html` | role-match (add chart) |
| `templates/mis_solicitudes.html` (new) | template | table + cancel button | `templates/ajustes_auditoria.html` | role-match |
| `static/js/app.js` (extend: modal amber/red + humanize_ms) | frontend | — | `templates/ajustes_usuarios.html` inline Alpine component `usuariosPanel()` | role-match |
| `tests/data/test_schema_query_log.py` (new) | test | — | `tests/data/test_nexo_repository.py` | exact |
| `tests/data/test_schema_guard_extended.py` (new) | test | — | `tests/data/test_schema_guard.py` | exact |
| `tests/services/test_preflight.py` (new, unit) | test | — | no services/test dir yet; shape from `tests/data/test_nexo_repository.py` | partial (need new dir) |
| `tests/services/test_thresholds_cache.py` (new) | test | — | `tests/data/test_nexo_repository.py` + contract tests via `inspect.getsource` | partial |
| `tests/services/test_approvals.py` (new, CAS semantics) | test | — | `tests/auth/test_audit_append_only.py` | partial |
| `tests/services/test_pipeline_lock.py` (new, asyncio) | test | — | (none; first asyncio semaphore test) | no analog |
| `tests/middleware/test_query_timing.py` (new) | test | — | `tests/auth/test_rbac_smoke.py` (TestClient + cookie) | role-match |
| `tests/routers/test_preflight_endpoints.py` (new) | test | — | `tests/auth/test_rbac_smoke.py` | exact |
| `tests/routers/test_approvals_api.py` (new) | test | — | `tests/auth/test_rbac_smoke.py` | exact |
| `api/main.py` (modify: +middleware, +lifespan tasks) | bootstrap | — | self | exact |
| `api/deps.py` (modify: +thresholds cache dep optional) | deps | — | self (existing Annotated aliases) | exact |
| `.env.example` (extend env vars) | config | — | (user-owned; new vars per D-10/D-14/D-18/D-20) | no analog |

---

## Pattern Assignments

### `nexo/data/models_nexo.py` — extend with QueryLog / QueryThreshold / QueryApproval (model, CRUD)

**Analog:** `nexo/data/models_nexo.py` (self; existing classes `NexoAuditLog`, `NexoLoginAttempt`)

**Imports pattern** (lines 25-42):
```python
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship
```

**`_utcnow` helper + NEXO_SCHEMA constant** (lines 44-52):
```python
NEXO_SCHEMA = "nexo"


class NexoBase(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
```

**Append-only log table pattern (closest to `query_log`)** — `NexoAuditLog`, lines 169-180:
```python
class NexoAuditLog(NexoBase):
    __tablename__ = "audit_log"
    __table_args__ = {"schema": NEXO_SCHEMA}

    id = Column(Integer, primary_key=True)
    ts = Column(DateTime(timezone=True), nullable=False, default=_utcnow, index=True)
    user_id = Column(Integer, ForeignKey("nexo.users.id", ondelete="SET NULL"), nullable=True)
    ip = Column(String(64), nullable=False)
    method = Column(String(10), nullable=False)
    path = Column(String(500), nullable=False)
    status = Column(Integer, nullable=False)
    details_json = Column(Text, nullable=True)  # whitelist-sanitized
```

**Composite-index pattern (closest to `approvals` single-use CAS)** — `NexoLoginAttempt`, lines 154-164:
```python
class NexoLoginAttempt(NexoBase):
    __tablename__ = "login_attempts"
    __table_args__ = (
        Index("ix_login_attempts_email_ip", "email", "ip"),
        {"schema": NEXO_SCHEMA},
    )

    id = Column(Integer, primary_key=True)
    email = Column(String(200), nullable=False, index=True)
    ip = Column(String(64), nullable=False)
    failed_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, index=True)
```

**Adaptation notes for the 3 new tables:**
- `NexoQueryLog` — copy shape of `NexoAuditLog`; add `endpoint: String(50)`, `params_json: Text`, `estimated_ms: Integer`, `actual_ms: Integer`, `rows: Integer nullable`, `status: String(20)`, `approval_id: ForeignKey(nexo.query_approvals.id) nullable`. Index on `ts desc`, `endpoint`, `user_id`, `status`.
- `NexoQueryThreshold` — small lookup table, PK on `endpoint: String(50)`; columns `warn_ms`, `block_ms`, `factor_ms`, `factor_updated_at`, `updated_at`, `updated_by: FK users.id nullable`.
- `NexoQueryApproval` — single-use CAS table; columns `id: uuid/Integer PK`, `user_id: FK users.id`, `endpoint`, `params_json`, `estimated_ms`, `status: String(20)` (pending/approved/rejected/expired/cancelled/consumed), `created_at`, `ttl_days`, `decided_by: FK users.id nullable`, `decided_at`, `consumed_at`, `consumed_run_id`, `cancelled_at`. Index on `(user_id, status)`, `(status, created_at)` for expiration job.
- All tables use `DateTime(timezone=True)` + `_utcnow` default (per research §Pitfall 3 — `datetime.utcnow()` deprecated).
- Export via `__all__` append.

Add to the module-level `__all__` list (line 183): `"NexoQueryLog"`, `"NexoQueryThreshold"`, `"NexoQueryApproval"`.

---

### `nexo/data/dto/query.py` — new (DTO, transform)

**Analog:** `nexo/data/dto/nexo.py`

**Imports + frozen config pattern** (lines 1-16 from `nexo.py`):
```python
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class UserRow(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    id: int
    email: str
    role: str
```

**Alternative: reuse shared `ROW_CONFIG`** (from `nexo/data/dto/base.py`, lines 17-24):
```python
from pydantic import ConfigDict

# ``frozen=True`` → inmutabilidad (mutation lanza ValidationError).
# ``from_attributes=True`` → permite Row.model_validate(orm_entity).
ROW_CONFIG: ConfigDict = ConfigDict(frozen=True, from_attributes=True)
```

**Adaptation notes:**
- Import `ROW_CONFIG` from `nexo.data.dto.base` (consistent with `app.py` pattern).
- Create `QueryLogRow`, `QueryThresholdRow`, `QueryApprovalRow` using `model_config = ROW_CONFIG`.
- Add non-Row DTOs: `Estimation(BaseModel)` for preflight output (fields: `endpoint: str`, `estimated_ms: int`, `level: Literal["green","amber","red"]`, `reason: str`, `breakdown: str`, `factor_used_ms: Optional[int]`, `warn_ms: int`, `block_ms: int`). `Estimation` is a response DTO, not a Row; keep `frozen=True` still (immutability principle from common/coding-style.md).
- Pattern for tuple fields when frozen (lines 15-24 `UserRow.departments: tuple[str, ...] = ()`) applies if any `list` fields are needed.

---

### `nexo/data/repositories/nexo.py` — extend with QueryLogRepo / ThresholdRepo / ApprovalRepo (repository, mixed)

**Analog:** `nexo/data/repositories/nexo.py` (self — `AuditRepo`)

**Repo class shape + constructor + session injection** (lines 114-121):
```python
class AuditRepo:
    """Audit log repository. IDENT-06 compat: SOLO INSERT/SELECT desde
    este repo; DB tambien bloquea UPDATE/DELETE al rol ``nexo_app``
    (defense-in-depth).
    """

    def __init__(self, db: Session):
        self._db = db
```

**Append-only INSERT without commit (caller orchestrates)** — `AuditRepo.append`, lines 123-147. This is the canonical pattern for `QueryLogRepo.append`:
```python
def append(
    self,
    *,
    user_id: int | None,
    ip: str,
    method: str,
    path: str,
    status: int,
    details_json: str | None,
) -> None:
    """INSERT fila en nexo.audit_log. Caller orquesta commit.

    NO comitea aqui. El middleware de audit (o quien llame a este
    metodo) es responsable del commit.
    """
    self._db.add(NexoAuditLog(
        user_id=user_id,
        ip=ip,
        method=method,
        path=path,
        status=status,
        details_json=details_json,
    ))
```

**Paginated filtered list returning DTOs** — `AuditRepo.list_filtered`, lines 175-220. This is the canonical pattern for `QueryLogRepo.list_filtered` (used by `/ajustes/rendimiento`):
```python
def list_filtered(
    self,
    *,
    user_email: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    path: str | None = None,
    status: int | None = None,
    page: int = 1,
    limit: int = 100,
) -> list[AuditLogRow]:
    stmt = select(NexoAuditLog, NexoUser).outerjoin(
        NexoUser, NexoUser.id == NexoAuditLog.user_id
    )
    if user_email:
        stmt = stmt.where(NexoUser.email == user_email)
    if date_from:
        stmt = stmt.where(NexoAuditLog.ts >= date_from)
    ...
    stmt = (
        stmt.order_by(NexoAuditLog.ts.desc())
        .limit(limit)
        .offset((page - 1) * limit)
    )

    rows = self._db.execute(stmt).all()
    return [
        AuditLogRow.model_validate({...})
        for log, user in rows
    ]
```

**Memory-safe streaming iterator** — `AuditRepo.iter_filtered`, lines 222-261. Applies to `QueryLogRepo.iter_filtered` if CSV export of query_log is added:
```python
stmt = stmt.order_by(NexoAuditLog.ts.desc()).execution_options(
    yield_per=500
)
for log, user in self._db.execute(stmt):
    yield AuditLogRow.model_validate({...})
```

**`count_filtered` pattern** — lines 149-173:
```python
from sqlalchemy import func
stmt = select(func.count()).select_from(NexoAuditLog).outerjoin(...)
... apply filters ...
return self._db.execute(stmt).scalar_one() or 0
```

**Adaptation notes for new repos:**
- `QueryLogRepo.append(**fields)` copies `AuditRepo.append` shape exactly — no commit, caller orchestrates. The middleware persister in `nexo/middleware/query_timing.py` will open its own Session and commit (pattern from `api/middleware/audit.py` below).
- `QueryLogRepo.list_filtered` / `count_filtered` / `iter_filtered` mirror `AuditRepo` trio but filter on `endpoint`, `status ∈ {ok,slow,error,timeout,approved_run}`, `user_id`, `ts range`.
- `QueryLogRepo.summary(endpoint, date_from, date_to) -> list[SummaryRow]` uses `select(func.count(), func.avg(actual_ms), func.percentile_cont(0.95)...)` grouped by endpoint — SQL aggregates, Postgres-specific.
- `ThresholdRepo.get(endpoint) -> ThresholdRow | None` / `list_all() -> list[ThresholdRow]` / `update(endpoint, warn_ms, block_ms, factor_ms, updated_by) -> None`. For the UPDATE, use the same pattern as `api/routers/usuarios.py` `editar` (lines 194-206): mutate ORM attributes + `db.commit()` in caller.
- `ApprovalRepo.consume(approval_id, user_id, run_id) -> ApprovalRow | None` — uses the CAS pattern **specific to this phase** (not analog in existing repos):
  ```sql
  UPDATE nexo.query_approvals
     SET consumed_at = now(), consumed_run_id = :run_id
   WHERE id = :approval_id
     AND user_id = :user_id
     AND status = 'approved'
     AND consumed_at IS NULL
  RETURNING *
  ```
  Implement with `self._db.execute(text(...).bindparams(...)).scalar_one_or_none()` — research §Pattern 4 "Single-use CAS" confirms this is the reference implementation; no existing analog in codebase for UPDATE ... RETURNING.
- **IMPORTANT — contract test parity:** `QueryLogRepo.append` must have NO `.commit()` / `.flush()` in source (matching `tests/data/test_nexo_repository.py::test_audit_repo_append_source_has_no_commit` lines 67-77). Copy that test verbatim for `QueryLogRepo`.

---

### `nexo/data/schema_guard.py` — update CRITICAL_TABLES (config, read)

**Analog:** `nexo/data/schema_guard.py` (self)

**Critical tables tuple** (lines 33-42):
```python
CRITICAL_TABLES: tuple[str, ...] = (
    "users",
    "roles",
    "departments",
    "user_departments",
    "permissions",
    "sessions",
    "login_attempts",
    "audit_log",
)
```

**Adaptation notes:**
- Append `"query_log"`, `"query_thresholds"`, `"query_approvals"` to the tuple.
- No other changes needed — `verify()` signature stays stable (kwarg injection already works per `test_schema_guard.py` lines 39-55).

---

### `nexo/services/preflight.py` — new (service, transform — pure)

**Analog:** `nexo/services/auth.py` (module-level pure functions; no I/O inside)

**Module-level singleton pattern** (lines 47-53 from `auth.py`):
```python
_ph = PasswordHasher(
    time_cost=RFC_9106_LOW_MEMORY.time_cost,
    memory_cost=RFC_9106_LOW_MEMORY.memory_cost,
    ...
)
```

**Pure-function pattern** — `hash_password`/`verify_password`, lines 56-67:
```python
def hash_password(password: str) -> str:
    """Devuelve el hash Argon2id PHC string del password en claro."""
    return _ph.hash(password)


def verify_password(stored_hash: str, password: str) -> bool:
    try:
        _ph.verify(stored_hash, password)
        return True
    except VerifyMismatchError:
        return False
```

**Factory-returns-dependency pattern** — `require_permission`, lines 232-268 (use if preflight is wrapped as a FastAPI Dependency for routers):
```python
def require_permission(permission: str):
    async def _check(request: Request) -> NexoUser:
        user = getattr(request.state, "user", None)
        if user is None:
            raise HTTPException(status_code=401, detail="Not authenticated")
        if user.role == "propietario":
            return user
        allowed = PERMISSION_MAP.get(permission, [])
        user_depts = {d.code for d in user.departments}
        if not user_depts.intersection(allowed):
            raise HTTPException(status_code=403, detail=f"Permiso requerido: {permission}")
        return user

    _check.__name__ = f"require_permission__{permission.replace(':', '_')}"
    return _check
```

**Adaptation notes:**
- `estimate_cost(endpoint: str, params: dict) -> Estimation` is pure (no DB calls) — reads factor + thresholds from in-memory `thresholds_cache` (imported module). Depends on `Estimation` DTO from `nexo/data/dto/query.py`.
- No singletons needed (stateless); module-level dispatch dict `_ESTIMATORS: dict[str, Callable[[dict], tuple[int, str]]]` mapping endpoint → heuristic function is idiomatic (cf. `PERMISSION_MAP` dict at line 200).
- Classification function `_level(estimated_ms, warn_ms, block_ms) -> Literal["green","amber","red"]` is a pure helper; keep <50 lines (common/coding-style.md).
- No `raise HTTPException` from the service layer — return `Estimation` and let the router decide (cf. `auth.hash_password` doesn't raise; it returns).

---

### `nexo/services/thresholds_cache.py` — new (service, in-memory cache + async pub/sub)

**Analog:** `nexo/services/auth.py` (module-level singletons + helper functions)

**Module-level singleton pattern + private `_` prefix** (lines 47-53, 77):
```python
_ph = PasswordHasher(...)
_serializer = URLSafeTimedSerializer(settings.secret_key, salt="nexo.session.v1")
```

**Adaptation notes:**
- Module-level dict `_CACHE: dict[str, ThresholdRow] = {}` guarded by `_LOCK: threading.Lock` (not asyncio.Lock — readers are sync from middleware).
- `get(endpoint: str) -> ThresholdRow | None` — returns in-memory row; if `now() - updated_at > 5min`, triggers a best-effort refresh (D-19 safety net).
- `full_reload(db: Session) -> None` / `reload_one(db: Session, endpoint: str) -> None` — read from `ThresholdRepo` and update `_CACHE`.
- `async def listen_loop() -> None` — wraps a blocking psycopg2 `LISTEN nexo_thresholds_changed` loop via `await asyncio.to_thread(...)`. Pattern to copy is **from RESEARCH Pattern 1** (the research document specifies this — no direct codebase analog). The loop lives in `lifespan` as `asyncio.create_task(thresholds_cache.listen_loop())` (see main.py adaptation below).
- Tests: `tests/services/test_thresholds_cache.py` imports only the pure functions (`full_reload`, `reload_one`) with a synthetic `ThresholdRepo` mock. LISTEN/NOTIFY path is integration (`pytest.mark.integration`, skip if Postgres down).

---

### `nexo/services/approvals.py` — new (service, CAS + state machine)

**Analog:** `nexo/services/auth.py` session management (`create_session`, `revoke_session`, `revoke_all_sessions`)

**Session create pattern** (lines 95-103):
```python
def create_session(db: Session, user_id: int) -> tuple[str, str]:
    raw = secrets.token_urlsafe(SESSION_TOKEN_BYTES)
    expires = datetime.now(timezone.utc) + timedelta(hours=settings.session_ttl_hours)
    row = NexoSession(user_id=user_id, token=raw, expires_at=expires)
    db.add(row)
    db.commit()
    ...
```

**Session revocation pattern** (lines 122-131):
```python
def revoke_session(db: Session, raw_token: str) -> None:
    db.execute(delete(NexoSession).where(NexoSession.token == raw_token))
    db.commit()


def revoke_all_sessions(db: Session, user_id: int) -> None:
    db.execute(delete(NexoSession).where(NexoSession.user_id == user_id))
    db.commit()
```

**Adaptation notes:**
- `create_approval(db, user_id, endpoint, params_json, estimated_ms, ttl_days) -> int` mirrors `create_session` shape (UUID or integer PK, INSERT, commit, return id).
- `approve(db, approval_id, decided_by) -> None`, `reject(db, approval_id, decided_by) -> None`, `cancel(db, approval_id, user_id) -> None` — each does a filtered UPDATE; commit at caller or within (match `revoke_session` pattern where commit is local).
- `consume(db, approval_id, user_id, run_id) -> ApprovalRow | None` — the single-use CAS using `UPDATE ... WHERE consumed_at IS NULL RETURNING *` (see §repository above; logic lives in `ApprovalRepo.consume`, the service calls it).
- `expire_stale(db, now) -> int` — for the cleanup job; returns count of rows moved to `expired`. Pattern to copy: `SQLAlchemy delete().where(...)` from `revoke_all_sessions`, but with `update().where(status='pending' AND created_at < cutoff).values(status='expired')`.

---

### `nexo/services/pipeline_lock.py` — new (service, concurrency primitive)

**Analog:** `nexo/services/auth.py` module-level singletons (closest shape for "one instance per process")

**Singleton pattern** (line 47 + 77):
```python
_ph = PasswordHasher(...)
_serializer = URLSafeTimedSerializer(settings.secret_key, salt="nexo.session.v1")
```

**Adaptation notes:**
- No direct analog for `asyncio.Semaphore` in the codebase. Create:
  ```python
  import asyncio, os
  MAX = int(os.environ.get("NEXO_PIPELINE_MAX_CONCURRENT", "3"))
  TIMEOUT = float(os.environ.get("NEXO_PIPELINE_TIMEOUT_SEC", "900"))
  _pipeline_semaphore = asyncio.Semaphore(MAX)
  ```
- Export `_pipeline_semaphore` for direct `async with` in `api/routers/pipeline.py`, plus a helper `async def run_with_lock(fn, /, *args, **kwargs)` that wraps `asyncio.wait_for(asyncio.to_thread(fn, *args, **kwargs), timeout=TIMEOUT)`.
- **Env-var parsing pattern** — copy from `nexo/data/schema_guard.py` lines 45-47:
  ```python
  def _auto_migrate_enabled() -> bool:
      return os.environ.get("NEXO_AUTO_MIGRATE", "").lower() in {"1", "true", "yes"}
  ```
- Tests: `tests/services/test_pipeline_lock.py` uses `pytest-asyncio` (`@pytest.mark.asyncio`); no existing analog — first asyncio test in repo.

---

### `nexo/services/query_log_cleanup.py`, `approvals_cleanup.py`, `factor_auto_refresh.py` — new (service/job, batch write)

**Analog:** None. First scheduled jobs in the codebase.

**Closest shapes to copy from:**
- `nexo/services/auth.py` `clear_attempts` (lines 162-170) — DELETE with filter + commit:
  ```python
  def clear_attempts(db: Session, email: str, ip: str) -> None:
      db.execute(
          delete(NexoLoginAttempt).where(
              NexoLoginAttempt.email == email,
              NexoLoginAttempt.ip == ip,
          )
      )
      db.commit()
  ```
- `AuditRepo.append` for writing to `audit_log` with `path="__cleanup__"` (D-10 requirement).

**Adaptation notes:**
- Each cleanup module exposes:
  - `async def run_once(db_factory) -> int` — does one pass, returns rows affected.
  - `async def run_loop() -> None` — sleeps until next Monday 03:00 (or 03:05 / 03:10), calls `run_once`, logs to `audit_log` via `AuditRepo.append(user_id=None, path="__cleanup__", method="DELETE", details_json=json.dumps({"rows_deleted": N, "cutoff_ts": ...}))`, repeat.
- Env var reading pattern: follow `nexo/data/schema_guard.py` `_auto_migrate_enabled` idiom (os.environ + default).
- Scheduling: `asyncio.create_task(query_log_cleanup.run_loop())` in `api/main.py` `lifespan` (research §Architecture confirms no APScheduler — plain loop with `asyncio.sleep(seconds_until_next_monday)`).

---

### `nexo/middleware/__init__.py` — new (package init)

**Analog:** `api/middleware/__init__.py` (empty marker)

**Content:** empty file, just makes the directory importable.

---

### `nexo/middleware/query_timing.py` — new (middleware, request/response wrap + write)

**Analog:** `api/middleware/audit.py`

**Imports + logger + Starlette BaseHTTPMiddleware** (lines 20-32):
```python
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from nexo.db.engine import SessionLocalNexo
from nexo.db.models import NexoAuditLog

logger = logging.getLogger("nexo.audit")
```

**Whitelist / constants at module level** (lines 38-64):
```python
_REDACTED_ENDPOINTS: frozenset[str] = frozenset({...})
_SENSITIVE_FIELDS: frozenset[str] = frozenset({...})
_MAX_DETAILS_CHARS = 4096
```

**Dispatch body + user extraction + best-effort persistence** (lines 103-151) — the canonical analog for `QueryTimingMiddleware.dispatch`:
```python
class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        user = getattr(request.state, "user", None)

        # Request publica (sin sesion): no auditamos.
        if user is None:
            return await call_next(request)

        # Capturar body ANTES de llamar al handler; FastAPI 0.135+ lo cachea.
        details: str | None = None
        if request.method in ("POST", "PUT", "PATCH"):
            ...

        response = await call_next(request)

        try:
            db = SessionLocalNexo()
            try:
                db.add(
                    NexoAuditLog(
                        ts=datetime.now(timezone.utc),
                        user_id=user.id,
                        ip=request.client.host if request.client else "unknown",
                        method=request.method,
                        path=request.url.path,
                        status=response.status_code,
                        details_json=details,
                    )
                )
                db.commit()
            finally:
                db.close()
        except Exception:
            # Un fallo del log no debe tumbar la response.
            logger.exception(
                "Error escribiendo nexo.audit_log (user_id=%s, path=%s)",
                user.id,
                request.url.path,
            )

        return response
```

**Adaptation notes:**
- Replace `AuditMiddleware` class with `QueryTimingMiddleware`. Swap `NexoAuditLog` with `NexoQueryLog` in `_persist`.
- Add `_TIMED_PATHS: frozenset[str] = frozenset({"/api/pipeline/run", "/api/bbdd/query", "/api/capacidad", "/api/operarios"})` (short-circuit for non-timed paths — cf. research Pattern 1).
- Insert `t0 = time.monotonic()` BEFORE `await call_next(request)` and compute `actual_ms = int((time.monotonic() - t0) * 1000)` after response — new primitive not in audit.py.
- **Error handling** — copy the `try/except: logger.exception` pattern verbatim (lines 143-149): a failed log MUST NOT tumble the response.
- **Commit discipline** — the middleware opens its OWN `SessionLocalNexo()` and commits, because middleware runs outside the FastAPI `Depends` ciclo (cf. `api/middleware/auth.py` line 132 same pattern). This is different from repos/routers where caller commits.
- `request.state.user` is already populated by `AuthMiddleware` when this middleware runs (see `api/main.py` middleware registration order below).
- Additional fields read from `request.state`: `estimated_ms`, `approval_id`, `params_json` (set by the router before reaching here — see router adaptations).

---

### `api/routers/pipeline.py` — modify: add `POST /preflight`, accept `force` + `approval_id` (router, request-response + streaming)

**Analog:** `api/routers/pipeline.py` (self, current form lines 1-49) + `api/routers/bbdd.py` (for validator-before-execute shape)

**Current pipeline router shape** (full file, lines 1-49):
```python
router = APIRouter(
    prefix="/pipeline",
    tags=["pipeline"],
    dependencies=[Depends(require_permission("pipeline:read"))],
)


@router.post(
    "/run",
    dependencies=[Depends(require_permission("pipeline:run"))],
)
def run(req: PipelineRequest):
    def event_stream():
        for msg in run_pipeline(...):
            yield f"data: {msg}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", ...)
```

**Validator-before-execute pattern from `api/routers/bbdd.py`** (lines 48-80 — how preflight should be injected):
```python
def _validate_sql(sql: str) -> None:
    """Valida que ``sql`` sea SELECT/WITH puro y sin multi-statement.

    Raises HTTPException(400) si rechaza; None si acepta.
    """
    if not sql or not sql.strip():
        raise HTTPException(400, "Falta el campo 'sql'")
    ...
```

**Adaptation notes:**
- Add `POST /pipeline/preflight` endpoint — accepts `PipelineRequest`, calls `preflight.estimate_cost("pipeline/run", params_dict)`, returns the `Estimation` DTO. Dependency: `require_permission("pipeline:read")`.
- Extend `PipelineRequest` in `api/models.py` with `force: bool = False` and `approval_id: Optional[int] = None` (flat Pydantic, consistent with existing shape at lines 34-39).
- Modify `run()` handler:
  1. If `req.force and req.approval_id`: call `ApprovalRepo.consume(approval_id, user_id, run_id=None)` — if returns None → `HTTPException(403, "Invalid or expired approval")`.
  2. If not forced: call `preflight.estimate_cost("pipeline/run", params)`. If `level != "green"` → `HTTPException(409, detail={"estimation": est.model_dump()})` so the client opens the modal.
  3. If green (or force+approval valid): `request.state.estimated_ms = est.estimated_ms; request.state.approval_id = req.approval_id; request.state.params_json = json.dumps(params)` — so `QueryTimingMiddleware` picks them up.
  4. `from nexo.services.pipeline_lock import _pipeline_semaphore, run_with_lock` — wrap execution in `async with _pipeline_semaphore: await run_with_lock(run_pipeline_sync, ...)`.
  5. Change the handler signature from `def run(...)` to `async def run(request: Request, req: PipelineRequest)` — async is required for semaphore + `asyncio.to_thread`.
- SSE streaming unchanged — `event_stream()` becomes an async generator that `yield`s after each thread-step message.

---

### `api/routers/bbdd.py` — modify: inject preflight before whitelist (router, request-response)

**Analog:** `api/routers/bbdd.py` (self, validator pattern `_validate_sql` at lines 48-80)

**Pattern — validator chain before execution** (lines 48-80, current shape):
```python
def _validate_sql(sql: str) -> None:
    if not sql or not sql.strip():
        raise HTTPException(400, "Falta el campo 'sql'")
    sql_stripped = sql.lstrip(" \t\r\n;(")
    first_word = sql_stripped.split(None, 1)[0].upper() if sql_stripped else ""
    if first_word not in {"SELECT", "WITH"}:
        raise HTTPException(400, "Solo se permiten SELECT/WITH")
    ...
```

**Adaptation notes:**
- In the `POST /query` handler (not shown above but at bottom of the file — line ~400+), call `preflight.estimate_cost("bbdd/query", {"sql": sql, "database": database})` BEFORE `_validate_sql`. Rationale: preflight is user-facing feedback; whitelist is security. Per research §Endpoint Retrofit, run preflight first so user sees cost in modal even if SQL is then rejected.
- Actually **verify order decision in research** — research suggests preflight comes first (§pipeline of discussions in §Architecture Diagram lines 221-227): `preflight → whitelist → execute`.
- If `level == "red"` without `force`+`approval_id` → `HTTPException(409, {"estimation": ...})`. Modal UI catches 409 and shows red modal.
- Store `request.state.estimated_ms`, `approval_id`, `params_json` for the middleware (same as pipeline).

---

### `api/routers/capacidad.py`, `api/routers/operarios.py` — modify: preflight if `rango_dias > 90` (router, request-response)

**Analog:** `api/routers/capacidad.py` (self — current form lines 42-50 is the hook):
```python
@router.get("")
def capacidad(
    engine_mes: EngineMes,
    fecha_inicio: date = Query(...),
    fecha_fin: date = Query(...),
):
    """Devuelve capacidad vs fabricado por referencia en el periodo."""
    if fecha_fin < fecha_inicio:
        raise HTTPException(400, "fecha_fin < fecha_inicio")

    fi = fecha_inicio.strftime("%Y-%m-%d")
    ff = fecha_fin.strftime("%Y-%m-%d")
```

**Adaptation notes:**
- Right after the `fecha_fin < fecha_inicio` guard, add:
  ```python
  rango_dias = (fecha_fin - fecha_inicio).days
  if rango_dias > 90:
      est = preflight.estimate_cost("capacidad", {"fecha_desde": fi, "fecha_fin": ff, "rango_dias": rango_dias, ...})
      if est.level != "green":
          if not (force and approval_id_is_valid):
              raise HTTPException(409, detail={"estimation": est.model_dump()})
      request.state.estimated_ms = est.estimated_ms
  ```
- Add `force: bool = Query(False)` and `approval_id: Optional[int] = Query(None)` parameters to the handler signature.
- `operarios.py` adaptation is identical; endpoint key is `"operarios"`.
- Rango ≤90d → short-circuit (no preflight, no `request.state.estimated_ms`). `QueryTimingMiddleware` checks `endpoint_key in _TIMED_PATHS` but the middleware skips entries absent from `_TIMED_PATHS` OR when user_id is None. To avoid writing query_log for sub-90d requests, gate the write on `request.state.estimated_ms is not None` (new field in middleware).

---

### `api/routers/approvals.py` — new (router, CRUD)

**Analog:** `api/routers/usuarios.py`

**Router prefix + auth dependency** (lines 44-48):
```python
router = APIRouter(
    prefix="/ajustes/usuarios",
    tags=["ajustes"],
    dependencies=[Depends(require_permission("usuarios:manage"))],
)
```

**Form-driven POST + validation + DB mutation + redirect** (lines 158-220 `editar`):
```python
@router.post("/{user_id}/editar")
async def editar(
    user_id: int,
    request: Request,
    db: DbNexo,
    role: str = Form(...),
    departments: list[str] = Form(default=[]),
    active: str = Form(default="off"),
):
    user = db.get(NexoUser, user_id)
    if user is None:
        raise HTTPException(404, "Usuario no encontrado")

    ... validations ...

    user.role = role
    user.departments = list(new_depts)
    user.active = new_active
    db.commit()
    ...
    return RedirectResponse(
        f"/ajustes/usuarios?ok=usuario-editado:{user.email}", status_code=303
    )
```

**Serialize ORM → dict for template** (lines 56-65):
```python
def _serialize_user(u: NexoUser) -> dict:
    return {
        "id": u.id,
        "email": u.email,
        ...
    }
```

**Adaptation notes:**
- Router prefix: `/api/approvals`. Dependency: `require_permission("aprobaciones:manage")` for propietario-only endpoints (and `authenticated` for user self-serve endpoints). Add `"aprobaciones:manage": []` to `PERMISSION_MAP` in `nexo/services/auth.py` (empty list = propietario-only per AUTH_MODEL).
- Endpoints per D-13/D-14/D-15/D-16:
  - `POST /api/approvals` — create new approval (user-initiated from red modal). Uses `approvals.create_approval(...)`.
  - `GET /api/approvals/count` — returns HTML fragment `<span>(3)</span>` for HTMX badge (propietario-only). Use `HTMLResponse` directly, not `render()`.
  - `GET /ajustes/solicitudes` — HTML page (propietario); uses `render("ajustes_solicitudes.html", request, {...})`.
  - `POST /api/approvals/{id}/approve` — propietario only. `ApprovalRepo.approve` + commit + `RedirectResponse(303)`.
  - `POST /api/approvals/{id}/reject` — propietario only.
  - `POST /api/approvals/{id}/cancel` — ownership check: `approval.user_id == request.state.user.id` (cf. `usuarios.editar` line 174 `is_self = current_user.id == user.id`) — if not, `HTTPException(403)`.
- All write ops: mutate ORM + `db.commit()` inline, like `editar` lines 194-206. No ORM commit in the repo (consistent with `AuditRepo.append` discipline).

---

### `api/routers/limites.py` — new (router, CRUD + cache-invalidate)

**Analog:** `api/routers/usuarios.py`

**Pattern:** same as `approvals.py` (propietario-only CRUD over `nexo.query_thresholds`).

**Adaptation notes:**
- Prefix `/ajustes/limites`. Dependency `require_permission("limites:manage")`. Add `"limites:manage": []` to `PERMISSION_MAP`.
- `GET /ajustes/limites` → HTML page (template `ajustes_limites.html`). Shows all thresholds + factor + "Recalcular desde últimos 30 runs" button (D-04).
- `POST /api/thresholds/{endpoint}` → UPDATE row, commit, then emit `NOTIFY nexo_thresholds_changed, '<endpoint>'`. Pattern for NOTIFY (no codebase analog):
  ```python
  db.execute(text("NOTIFY nexo_thresholds_changed, :channel"), {"channel": endpoint})
  db.commit()
  ```
- `POST /api/thresholds/{endpoint}/recalibrate` → computes `factor_nuevo = median(actual_ms / (n_recursos * n_dias))` from last 30 query_log rows — uses `QueryLogRepo.iter_filtered(endpoint="pipeline/run", status IN ('ok','slow'))` then Python median (stdlib `statistics.median`). Returns preview JSON; the client POSTs again with `confirm=true` to persist.

---

### `api/routers/rendimiento.py` — new (router, read + aggregates)

**Analog:** `api/routers/auditoria.py` (filter + paginate + structure)

**Router shape + parse helpers + filtered query** (lines 41-146):
```python
router = APIRouter(
    prefix="/ajustes/auditoria",
    tags=["ajustes"],
    dependencies=[Depends(require_permission("auditoria:read"))],
)


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if len(value) == 10:
            return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


@router.get("", response_class=HTMLResponse)
async def listar(
    request: Request,
    db: DbNexo,
    user_email: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    path: Optional[str] = Query(None),
    status: Optional[int] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1, le=500),
):
    repo = AuditRepo(db)

    email_norm = user_email.strip().lower() if user_email else None
    dt_from = _parse_iso_datetime(date_from)
    dt_to = _parse_date_to_end_of_day(date_to)

    total = repo.count_filtered(...)
    dtos = repo.list_filtered(..., page=page, limit=limit)

    serialized = [
        {"id": r.id, "ts": r.ts, ...}
        for r in dtos
    ]
    total_pages = (total + limit - 1) // limit if total else 1

    return render("ajustes_auditoria.html", request, {...})
```

**Adaptation notes:**
- Prefix `/ajustes/rendimiento`. Dependency `require_permission("rendimiento:read")` (add to PERMISSION_MAP with `[]` for propietario-only).
- Two endpoints per D-11:
  - `GET /ajustes/rendimiento` → HTML page.
  - `GET /api/rendimiento/summary?endpoint=X&date_from=Y&date_to=Z` → JSON with aggregates (`n_runs`, `avg_estimated_ms`, `avg_actual_ms`, `divergencia_%`, `p95_actual_ms`, `n_slow`). Uses `QueryLogRepo.summary(...)` (new method with SQL `func.avg`, `func.percentile_cont(0.95)`).
  - `GET /api/rendimiento/timeseries?endpoint=X&days=N` → JSON list `[{ts, estimated_ms, actual_ms}, ...]` for Chart.js.
- Reuse `_parse_iso_datetime` / `_parse_date_to_end_of_day` verbatim (copy from `auditoria.py` lines 48-73).
- CSV export optional (deferred per CONTEXT line 322 "exportar CSV… nice-to-have").

---

### `api/routers/pages.py` — extend (router, HTML rendering)

**Analog:** `api/routers/pages.py` (self, lines 106-111 `ajustes_page`):
```python
@router.get(
    "/ajustes",
    dependencies=[Depends(require_permission("ajustes:manage"))],
)
def ajustes_page(request: Request):
    return render("ajustes.html", request, _common_extra("ajustes"))
```

**Adaptation notes:**
- Add handlers for:
  - `GET /ajustes/limites` → `render("ajustes_limites.html", request, _common_extra("ajustes"))`.
  - `GET /ajustes/solicitudes` → `render("ajustes_solicitudes.html", ...)` with `require_permission("aprobaciones:manage")`.
  - `GET /ajustes/rendimiento` → already handled inside `api/routers/rendimiento.py` (moved there).
  - `GET /mis-solicitudes` → `render("mis_solicitudes.html", ...)`; depends only on auth (no specific permission — all users see their own).
- Use `_common_extra("ajustes")` for sidebar highlight.
- If the rendimiento router owns its HTML page (via `response_class=HTMLResponse`), then pages.py doesn't need a `/ajustes/rendimiento` handler — keep it consistent with `auditoria.py` which owns its own page.

---

### `templates/ajustes.html` — extend (add links)

**Analog:** `templates/ajustes.html` (self, lines 8-34 navigation cards)

**Navigation card pattern** (lines 9-21):
```html
<a href="/ajustes/usuarios"
   class="group bg-white rounded-2xl border border-surface-200 shadow-sm p-5 hover:border-brand-400 hover:shadow-md transition-all">
  <div class="flex items-start justify-between">
    <div>
      <h3 class="font-bold text-gray-800 group-hover:text-brand-700 transition-colors">Usuarios</h3>
      <p class="text-xs text-gray-500 mt-1">CRUD de usuarios del sistema. ...</p>
    </div>
    <svg class="w-5 h-5 ..."><path .../></svg>
  </div>
</a>
```

**Adaptation notes:**
- Add 3 more cards following the exact same markup: `/ajustes/limites`, `/ajustes/solicitudes`, `/ajustes/rendimiento`. Each gets a title + 1-line description.
- Keep the 2-column grid (`grid-cols-1 sm:grid-cols-2`) — adding cards increments rows automatically.

---

### `templates/ajustes_limites.html` — new (template, form submit)

**Analog:** `templates/ajustes_usuarios.html`

**Alpine-powered panel with `x-data`/`x-init`** (lines 6-7):
```html
<div x-data="usuariosPanel()" x-init="init()">
```

**Table row + edit button opening modal** (lines 47-114): full table structure; for `ajustes_limites.html` one row per endpoint (pipeline/run, bbdd/query, capacidad, operarios) with `[Editar]` button opening modal.

**Modal with `@click.outside="closeModal()"`** (lines 117-204): copy this modal shape for the "Editar umbrales" form.

**Alpine component at bottom** (lines 240-299): `usuariosPanel()` with `showForm`, `form`, `openCreate()`, `openEdit()`, `closeModal()` — copy shape, rename to `limitesPanel()`.

**Feedback banner pattern** (lines 9-19):
```html
{% if error %}
  <div class="mb-4 px-4 py-3 rounded-lg bg-red-50 border border-red-200 text-red-700 text-sm">
    {{ error }}
  </div>
{% endif %}
```

**Adaptation notes:**
- Form posts to `/api/thresholds/<endpoint>` (UPDATE).
- Include "Recalcular factor" button that POSTs to `/api/thresholds/<endpoint>/recalibrate` with `confirm=false` first (preview), then `confirm=true` (persist). Use Alpine `fetch()` + modal preview.

---

### `templates/ajustes_solicitudes.html`, `templates/mis_solicitudes.html` — new (template, table + actions)

**Analog:** `templates/ajustes_auditoria.html` (filters + paginated table) + `templates/ajustes_usuarios.html` (action buttons + confirm dialog)

**Filter form** from `ajustes_auditoria.html` (lines 8-64):
```html
<form method="get" action="/ajustes/auditoria"
      class="bg-white rounded-xl shadow-sm border border-surface-200 p-4 mb-4">
  <div class="grid grid-cols-1 md:grid-cols-5 gap-3 text-sm">
    <div>
      <label class="block text-xs font-medium text-gray-600 mb-1">Email</label>
      <input type="email" name="user_email" value="{{ filters.user_email }}" ...>
    </div>
    ...
  </div>
</form>
```

**Status badge with conditional colors** (lines 94-102):
```html
<span class="text-[10px] px-2 py-0.5 rounded-full font-mono font-medium
       {% if r.status < 300 %}bg-green-50 text-green-700
       {% elif r.status < 400 %}bg-blue-50 text-blue-700
       {% elif r.status < 500 %}bg-yellow-50 text-yellow-700
       {% else %}bg-red-50 text-red-700{% endif %}">
  {{ r.status }}
</span>
```

**Form-submit inline button with confirm** (from `ajustes_usuarios.html` lines 100-107):
```html
<form method="post" action="/ajustes/usuarios/{{ u.id }}/desactivar" class="inline"
      onsubmit="return confirm('Desactivar a {{ u.email }}? ...');">
  <button type="submit" class="text-xs text-red-700 hover:text-red-800 underline underline-offset-2">
    Desactivar
  </button>
</form>
```

**HTMX polling** — no codebase analog yet; use `hx-get="/api/approvals/count" hx-trigger="every 30s" hx-swap="innerHTML"` per research §Pattern 4 and D-13. HTMX is already loaded in `base.html` line 8.

**Adaptation notes:**
- `ajustes_solicitudes.html`: status-mapped badges (pending=amber, approved=green, rejected=red, expired=gray, cancelled=gray, consumed=blue). Action buttons: `[Aprobar]`, `[Rechazar]` (both POST form inline with confirm).
- `mis_solicitudes.html`: same table but no approve/reject; only `[Cancelar]` button on `pending` rows (ownership enforced server-side per D-16).
- Both templates extend `base.html`, use `{% block content %}`, no special CDN changes needed.

---

### `templates/ajustes_rendimiento.html` — new (template, filters + table + chart)

**Analog:** `templates/ajustes_auditoria.html` (same filter + table shape)

**Same filter form structure as `ajustes_auditoria.html`**; swap filter fields to: user (dropdown), endpoint (dropdown), status (green/amber/red/slow/timeout/approved_run), rango temporal (7d/30d/90d/custom).

**Chart.js integration** — Chart.js is already loaded in `base.html` line 11:
```html
<script defer src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
```

**Adaptation notes:**
- No existing chart in the codebase. Add a `<canvas id="timeseriesChart" height="200">` below the summary table.
- Client-side Alpine component fetches `/api/rendimiento/timeseries?endpoint=X&days=N` on init and on filter change, then `new Chart(ctx, {type:'line', data:{labels, datasets:[{label:'Estimado', data:estData},{label:'Real', data:actData}]}})`.
- Fallback (D-12): if `Chart` undefined (CDN failure), hide canvas and keep the table visible. Add `x-show="typeof Chart !== 'undefined'"` on the chart container.

---

### `static/js/app.js` — extend (frontend: modal amber/red + humanize_ms)

**Analog:** inline Alpine component in `templates/ajustes_usuarios.html` lines 240-299 (`usuariosPanel()`)

**Alpine component factory pattern** (lines 241-299 from `ajustes_usuarios.html`):
```javascript
function usuariosPanel() {
  return {
    showForm: false,
    showReset: false,
    mode: 'create',
    formAction: '',
    form: { id: null, email: '', ..., active: true },
    init() {
      {% if open_create %}
      this.openCreate();
      {% endif %}
    },
    openCreate() {
      this.mode = 'create';
      this.formAction = '/ajustes/usuarios/crear';
      this.form = { ... };
      this.showForm = true;
    },
    openEdit(user) { ... },
    closeModal() {
      this.showForm = false;
      this.showReset = false;
    },
  };
}
```

**Modal markup with Alpine** (from `ajustes_usuarios.html` lines 117-204):
```html
<div x-show="showForm" x-cloak x-transition.opacity
     class="fixed inset-0 z-40 flex items-center justify-center bg-black/40 p-4">
  <div class="bg-white rounded-xl shadow-2xl w-full max-w-lg p-6"
       @click.outside="closeModal()">
    ...
    <div class="flex justify-end gap-2 pt-2">
      <button type="button" @click="closeModal()"
              class="px-4 py-2 rounded-lg border border-surface-300 text-gray-700 hover:bg-surface-50 text-sm">
        Cancelar
      </button>
      <button type="submit"
              class="px-4 py-2 rounded-lg bg-brand-600 hover:bg-brand-700 text-white text-sm font-medium">
        Confirmar
      </button>
    </div>
  </div>
</div>
```

**Adaptation notes:**
- Register as `Alpine.data('preflightModal', () => ({ ... }))` so pages using heavy endpoints can `x-data="preflightModal"` globally.
- Expose two methods: `openAmber(estimation, executeCallback)` and `openRed(estimation, endpoint, params)`. Amber → `[Continuar]` calls `executeCallback(force=true)` (+ no approval); Red → `[Solicitar aprobación]` does `fetch('/api/approvals', {method:'POST', body: JSON.stringify({endpoint, params, estimated_ms: estimation.estimated_ms})})` → toast.
- `humanize_ms(ms)` utility: pure function, returns "3 min 20s" / "45s" / "2h 10 min". Pattern: plain JS, no library.
- Entry trigger: client wrapping of the relevant fetches (pipeline/bbdd/capacidad/operarios). Frontend intercepts 409 with `{estimation}` payload → opens the matching modal (amber vs red by `estimation.level`).

---

### `api/main.py` — modify: add middleware + lifespan tasks

**Analog:** `api/main.py` (self)

**Lifespan pattern** (lines 27-49):
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa la BBDD al arrancar."""
    # 1. Schema guard
    schema_guard.verify(engine_nexo)

    # 2. init_db legacy
    try:
        init_db()
        logger.info("Base de datos inicializada OK")
    except Exception as exc:
        logger.error(f"Error inicializando BD: {exc}")
        logger.error(traceback.format_exc())
    yield
```

**Middleware LIFO registration** (lines 138-139):
```python
# ORDEN LIFO (research §Pitfall 1):
# Starlette ejecuta los middlewares en orden INVERSO al registro. El ultimo
# ``add_middleware`` es el primero en procesar (outer).
#
# Cadena deseada (outer → inner):
#   Request → AuthMiddleware → AuditMiddleware → handler → ...

app.add_middleware(AuditMiddleware)    # inner — segundo en ejecutar
app.add_middleware(AuthMiddleware)     # outer — primero en ejecutar
```

**Router registration pattern** (lines 153-172):
```python
from api.routers import auth as auth_router, auditoria as auditoria_router, pages, ...

app.include_router(health.router, prefix="/api")
app.include_router(auth_router.router)  # /login, /logout
app.include_router(usuarios_router.router)  # /ajustes/usuarios/*
app.include_router(auditoria_router.router)  # /ajustes/auditoria
app.include_router(pages.router)
app.include_router(pipeline.router, prefix="/api")
```

**Adaptation notes:**
- In `lifespan`, after `schema_guard.verify(engine_nexo)`:
  ```python
  # 3. Thresholds cache + LISTEN listener
  from nexo.services import thresholds_cache
  with SessionLocalNexo() as db:
      thresholds_cache.full_reload(db)
  listen_task = asyncio.create_task(thresholds_cache.listen_loop())

  # 4. Cleanup schedulers
  cleanup_tasks = [
      asyncio.create_task(query_log_cleanup.run_loop()),
      asyncio.create_task(approvals_cleanup.run_loop()),
      asyncio.create_task(factor_auto_refresh.run_loop()),
  ]
  ```
- After `yield`, cancel tasks: `listen_task.cancel(); [t.cancel() for t in cleanup_tasks]`; `await asyncio.gather(..., return_exceptions=True)`.
- Middleware chain per research Pattern 1 + §Middleware Order (lines 365-373 of RESEARCH):
  ```python
  from nexo.middleware.query_timing import QueryTimingMiddleware
  app.add_middleware(QueryTimingMiddleware)  # innermost — closest to handler
  app.add_middleware(AuditMiddleware)
  app.add_middleware(AuthMiddleware)          # outermost — runs first
  ```
- Include new routers after existing ones:
  ```python
  from api.routers import approvals as approvals_router, limites as limites_router, rendimiento as rendimiento_router
  app.include_router(approvals_router.router, prefix="/api")
  app.include_router(limites_router.router)  # /ajustes/limites + /api/thresholds
  app.include_router(rendimiento_router.router)  # /ajustes/rendimiento + /api/rendimiento
  ```

---

### `api/deps.py` — modify: add optional thresholds cache dep

**Analog:** `api/deps.py` (self, lines 96-100 `Annotated` aliases)

**Annotated dependency pattern** (lines 96-100):
```python
# Aliases ``Annotated`` (PEP 593) — firmas de router más limpias:
#   def endpoint(db: DbApp, engine_mes: EngineMes): ...
DbApp = Annotated[Session, Depends(get_db_app)]
DbNexo = Annotated[Session, Depends(get_db_nexo)]
EngineMes = Annotated[Engine, Depends(get_engine_mes)]
```

**Adaptation notes (optional, only if preflight is invoked via Dependency):**
- Add:
  ```python
  from nexo.services import thresholds_cache as _tc

  def get_thresholds() -> "ThresholdsCache":
      return _tc  # module-as-cache

  Thresholds = Annotated[ThresholdsCache, Depends(get_thresholds)]
  ```
- Routers can accept `cache: Thresholds` if they want cleaner signatures; otherwise they `from nexo.services import thresholds_cache` directly (either is idiomatic in this codebase — `auditoria.py` does direct import for `SessionLocalNexo`, `auth.py` does direct import for its singletons).

---

### `tests/data/test_schema_query_log.py` — new (integration test)

**Analog:** `tests/data/test_nexo_repository.py`

**Fixture use + pytestmark** (lines 14-19):
```python
import pytest

from nexo.data.dto.nexo import AuditLogRow, RoleRow, UserRow
from nexo.data.repositories.nexo import AuditRepo, RoleRepo, UserRepo

pytestmark = pytest.mark.integration
```

**Contract test via `inspect.getsource`** (lines 67-77):
```python
def test_audit_repo_append_source_has_no_commit():
    """Contract test T-03-03-01: AuditRepo.append NO debe contener .commit()
    ni .flush() en su source."""
    src = inspect.getsource(AuditRepo.append)
    assert ".commit()" not in src, f"AuditRepo.append contiene .commit(): {src}"
    assert ".flush()" not in src
    assert "db.commit" not in src
```

**Session-in-transaction test** (lines 80-108):
```python
def test_audit_repo_append_does_not_commit(db_nexo):
    repo = AuditRepo(db_nexo)
    before_new = len(list(db_nexo.new))
    repo.append(user_id=None, ip="127.0.0.1", method="GET", path="/_test_", status=200, details_json=None)
    after_new = len(list(db_nexo.new))
    assert after_new == before_new + 1
    assert db_nexo.in_transaction()
```

**DTO assertion pattern** (lines 39-47):
```python
def test_user_repo_list_all_returns_dtos(db_nexo):
    repo = UserRepo(db_nexo)
    users = repo.list_all()
    assert isinstance(users, list)
    assert all(isinstance(u, UserRow) for u in users)
```

**Adaptation notes:**
- Import `QueryLogRepo`, `ThresholdRepo`, `ApprovalRepo`, and corresponding DTOs.
- Copy the three test styles: DTO assertion, `inspect.getsource` no-commit, session-new delta.
- For `ApprovalRepo.consume` (CAS), add a test that calls it twice and asserts second call returns None (single-use invariant).
- `db_nexo` fixture comes from `tests/data/conftest.py` lines 60-73 — rollback at end; skip if Postgres down via `_postgres_reachable`.

---

### `tests/data/test_schema_guard_extended.py` — new

**Analog:** `tests/data/test_schema_guard.py` (full file lines 1-87)

**kwarg-injection pattern** (lines 39-54):
```python
def test_verify_raises_when_table_missing(monkeypatch):
    monkeypatch.delenv("NEXO_AUTO_MIGRATE", raising=False)
    with pytest.raises(RuntimeError) as exc_info:
        schema_guard.verify(
            engine_nexo,
            critical_tables=("users", "__nonexistent_table__"),
        )
    assert "__nonexistent_table__" in str(exc_info.value)
```

**Adaptation notes:**
- Test that verify accepts the 3 new tables (`"query_log"`, `"query_thresholds"`, `"query_approvals"`) by default (`CRITICAL_TABLES` now includes them).
- Same kwarg-injection pattern; no monkeypatching `CRITICAL_TABLES`.

---

### `tests/services/test_preflight.py` — new (unit test)

**Analog:** no `tests/services/` dir yet. Closest shape is `tests/data/test_nexo_repository.py` BUT preflight is pure — no DB fixture needed.

**Unit test shape (pure functions, no integration mark)** — distilled from `tests/data/test_nexo_repository.py` minus `db_nexo` fixture. Don't use `pytestmark = pytest.mark.integration` because preflight has no I/O.

**Adaptation notes:**
- Create `tests/services/__init__.py` (empty) + `tests/services/test_preflight.py`.
- Tests: `test_estimate_cost_pipeline_returns_ms()`, `test_level_green_when_under_warn()`, `test_level_amber_when_between_warn_and_block()`, `test_level_red_when_over_block()`, `test_breakdown_string_includes_factors()`.
- Mock thresholds_cache via monkeypatch:
  ```python
  def test_estimate_cost_green(monkeypatch):
      monkeypatch.setattr("nexo.services.thresholds_cache.get", lambda ep: FakeThreshold(warn_ms=120_000, block_ms=600_000, factor_ms=2000))
      est = preflight.estimate_cost("pipeline/run", {"n_recursos": 2, "n_dias": 5})
      assert est.level == "green"
      assert est.estimated_ms == 2 * 5 * 2000
  ```

---

### `tests/services/test_thresholds_cache.py` — new

**Analog:** `tests/data/test_nexo_repository.py` (DB-backed) + `tests/data/test_schema_guard.py` (skipif pattern)

**Postgres skip pattern** (lines 20-26 `test_schema_guard.py`):
```python
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _postgres_reachable(),
        reason="Postgres no arriba: docker compose up -d db",
    ),
]
```

**Adaptation notes:**
- Two test groups:
  1. Unit tests (no Postgres): `full_reload` with mock `ThresholdRepo`, `get()` returns None for unknown, `get()` returns cached row after reload.
  2. Integration (Postgres): `test_notify_triggers_reload` — uses `db_nexo` + a side-thread to issue NOTIFY, then polls `cache.get(endpoint).updated_at` for <1s change.
- Use `_postgres_reachable` from `tests/data/conftest.py` lines 36-43.

---

### `tests/services/test_approvals.py` — new (CAS semantics)

**Analog:** `tests/auth/test_audit_append_only.py` (DB permission test pattern; not shown but references `nexo_app` role restrictions)

**Adaptation notes:**
- `test_consume_returns_none_on_second_call(db_nexo)` — creates approval, calls `consume()` twice, asserts second returns None.
- `test_consume_fails_if_user_id_mismatch(db_nexo)` — approval for user A, consume called by user B → None.
- `test_consume_fails_if_status_not_approved(db_nexo)` — status='pending' → None.
- Use `db_nexo` fixture from `tests/data/conftest.py`.

---

### `tests/services/test_pipeline_lock.py` — new (asyncio)

**Analog:** None. First asyncio-specific test in repo.

**Adaptation notes:**
- Use `@pytest.mark.asyncio` (add `pytest-asyncio` to requirements if not present — verify).
- `test_semaphore_limits_to_max()` — spawn 5 coroutines, assert only 3 are running concurrently (use `asyncio.Event` to probe).
- `test_timeout_raises()` — `await asyncio.wait_for(asyncio.to_thread(lambda: time.sleep(2)), timeout=0.1)` raises `TimeoutError`.
- No DB fixture needed.

---

### `tests/middleware/test_query_timing.py` — new

**Analog:** `tests/auth/test_rbac_smoke.py` (TestClient + cookie + integration + skipif Postgres)

**TestClient + cookie fixture** (lines 69-73):
```python
@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app, follow_redirects=False) as c:
        yield c
```

**Login + cookie extraction** (lines 134-147):
```python
def _login(client: TestClient, email: str, password: str = TEST_PASSWORD) -> str:
    r = client.post("/login", data={"email": email, "password": password},
                    headers={"Accept": "text/html"})
    assert r.status_code == 303
    cookie = r.cookies.get("nexo_session")
    assert cookie
    return cookie
```

**Integration skipif** (lines 54-60):
```python
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _postgres_reachable(),
        reason="Postgres no disponible — requiere `docker compose up -d db`",
    ),
]
```

**Adaptation notes:**
- Create `tests/middleware/__init__.py` + `test_query_timing.py`.
- Test: after a request to `/api/bbdd/query`, a row is present in `nexo.query_log` with matching `endpoint`, `user_id`, `actual_ms > 0`.
- Test: `slow` status gets written when `actual_ms > warn_ms * 1.5` (simulate by setting `warn_ms=1` in a test-only override).
- Mirror `_create_test_user` pattern from `tests/auth/test_rbac_smoke.py` lines 104-131.

---

### `tests/routers/test_preflight_endpoints.py`, `test_approvals_api.py` — new

**Analog:** `tests/auth/test_rbac_smoke.py` (TestClient + cookie + RBAC smoke)

**Adaptation notes:**
- Reuse `_create_test_user` + `_login` helpers. Consider refactoring them into `tests/auth/conftest.py` (but out of scope for this phase — copy-paste is fine per common/coding-style.md DRY at this level is excessive).
- `test_preflight_green_returns_estimation()`, `test_red_without_force_returns_409()`, `test_red_with_invalid_approval_returns_403()`, `test_red_with_valid_approval_executes()`.

---

## Shared Patterns

### Authentication (RBAC)

**Source:** `nexo/services/auth.py` lines 232-268 (`require_permission` factory)
**Apply to:** ALL new routers (`approvals.py`, `limites.py`, `rendimiento.py`) and extended endpoints in `pipeline.py`, `bbdd.py`, `capacidad.py`, `operarios.py`.

**Concrete excerpt:**
```python
router = APIRouter(
    prefix="/ajustes/solicitudes",
    tags=["ajustes"],
    dependencies=[Depends(require_permission("aprobaciones:manage"))],
)
```

**Propietario bypass** — the factory short-circuits propietario role without consulting the map:
```python
if user.role == "propietario":
    return user
```

**Add to PERMISSION_MAP** (`nexo/services/auth.py` line 200):
```python
"aprobaciones:manage": [],   # propietario only
"limites:manage":      [],
"rendimiento:read":    [],
```

---

### Error Handling

**Source:** `api/middleware/audit.py` lines 143-149 (best-effort persistence)
**Apply to:** `nexo/middleware/query_timing.py` and all cleanup jobs.

**Excerpt:**
```python
try:
    db.commit()
except Exception:
    # Un fallo del log no debe tumbar la response.
    logger.exception(
        "Error escribiendo nexo.audit_log (user_id=%s, path=%s)",
        user.id,
        request.url.path,
    )
```

**Source:** `api/main.py` lines 81-109 (global exception handler sanitizes output)
**Apply to:** never log raw tracebacks to user-facing responses in new routers. `HTTPException` with sanitized detail is the pattern.

---

### Database Session Management

**Source:** `api/deps.py` lines 73-100 (yield-pattern generators + `Annotated` aliases)
**Apply to:** all new routers (`approvals.py`, `limites.py`, `rendimiento.py`) — use `DbNexo` parameter, not `SessionLocalNexo()` inline. Middleware and background jobs are the ONLY place to open sessions manually (cf. `api/middleware/audit.py` line 127 — outside the Depends ciclo).

**Router excerpt:**
```python
from api.deps import DbNexo, render

@router.get("")
async def listar(request: Request, db: DbNexo, ...):
    repo = QueryLogRepo(db)
    rows = repo.list_filtered(...)
    return render("ajustes_rendimiento.html", request, {"rows": rows})
```

**Middleware / background excerpt:**
```python
db = SessionLocalNexo()
try:
    db.add(...)
    db.commit()
finally:
    db.close()
```

---

### Environment Variable Parsing

**Source:** `nexo/data/schema_guard.py` lines 45-47
**Apply to:** `nexo/services/pipeline_lock.py`, cleanup jobs, `.env.example`.

**Excerpt:**
```python
def _auto_migrate_enabled() -> bool:
    return os.environ.get("NEXO_AUTO_MIGRATE", "").lower() in {"1", "true", "yes"}
```

**Convention for ints/floats:**
```python
NEXO_PIPELINE_MAX_CONCURRENT = int(os.environ.get("NEXO_PIPELINE_MAX_CONCURRENT", "3"))
NEXO_PIPELINE_TIMEOUT_SEC = float(os.environ.get("NEXO_PIPELINE_TIMEOUT_SEC", "900"))
```

Document each new var in `.env.example`:
- `NEXO_QUERY_LOG_RETENTION_DAYS=90` (0 = forever)
- `NEXO_APPROVAL_TTL_DAYS=7`
- `NEXO_PIPELINE_MAX_CONCURRENT=3`
- `NEXO_PIPELINE_TIMEOUT_SEC=900`
- `NEXO_AUTO_REFRESH_STALE_DAYS=60`

---

### UTC-aware Timestamps

**Source:** `nexo/data/models_nexo.py` lines 27, 51-52
**Apply to:** ALL new models + services + middleware.

**Excerpt:**
```python
from datetime import datetime, timezone

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

# In models:
ts = Column(DateTime(timezone=True), nullable=False, default=_utcnow, index=True)
```

Never use `datetime.utcnow()` (deprecated in Python 3.12 per research §Pitfall 3).

---

### Frozen DTOs (Pydantic v2)

**Source:** `nexo/data/dto/base.py` lines 17-24 (`ROW_CONFIG`)
**Apply to:** `nexo/data/dto/query.py`, including `Estimation`.

**Excerpt:**
```python
from nexo.data.dto.base import ROW_CONFIG

class QueryLogRow(BaseModel):
    model_config = ROW_CONFIG
    id: int
    ts: datetime
    endpoint: str
    ...
```

`frozen=True` enforces common/coding-style.md "Immutability (CRITICAL)" and surfaces mutation attempts as `ValidationError`.

---

### HTMX Polling (new convention for this phase)

No pre-existing codebase analog. Pattern from research Pattern 4 (approvals badge) and D-13:
```html
<span hx-get="/api/approvals/count"
      hx-trigger="every 30s"
      hx-swap="innerHTML"></span>
```

Server returns either `<span class="badge">(3)</span>` or empty string. HTMX is already loaded in `base.html` line 8.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `nexo/services/thresholds_cache.py` (LISTEN loop) | service | async pub/sub | No existing LISTEN/NOTIFY consumer in the codebase. Pattern comes from RESEARCH §Pattern 1 + `asyncio.to_thread` wrapping of psycopg2 sync LISTEN. |
| `nexo/services/pipeline_lock.py` | service | concurrency primitive | First `asyncio.Semaphore` + `asyncio.to_thread` usage. Research Pattern 3 is canonical. |
| `nexo/services/query_log_cleanup.py`, `approvals_cleanup.py`, `factor_auto_refresh.py` | service/job | scheduled batch | First scheduled jobs in the repo. Planner implements as `async def run_loop()` with `asyncio.sleep(until_next_monday)` — research §Architecture confirms NO APScheduler. |
| `tests/services/test_pipeline_lock.py` | test | — | First asyncio test; requires `pytest-asyncio`. Verify it's in `requirements*.txt`; if not, add to planner task. |
| `.env.example` diff (5 new vars) | config | — | User-owned file; no planner-owned analog. Follow the existing `NEXO_*` prefix convention per `CLAUDE.md` naming rules. |

Planner should use RESEARCH.md §Pattern 1 (LISTEN/NOTIFY wrapping), §Pattern 3 (Semaphore + to_thread + wait_for), and §Factor Learning Algorithm for the corresponding implementations.

---

## Metadata

**Analog search scope:**
- `nexo/data/*`
- `nexo/services/*`
- `api/middleware/*`
- `api/routers/*`
- `api/{main,deps,models}.py`
- `templates/*.html`
- `tests/{data,auth}/*`

**Files scanned (read or grep'd):** 27
**Pattern extraction date:** 2026-04-19
**Research doc consulted:** `.planning/phases/04-consultas-pesadas/04-RESEARCH.md` §Pattern 1-4, §Architecture, §Standard Stack
**Context doc consulted:** `.planning/phases/04-consultas-pesadas/04-CONTEXT.md` §Decisions D-01..D-20
