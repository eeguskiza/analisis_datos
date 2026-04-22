---
phase: 08-redise-o-ui-modo-claro-moderno
plan: 03
type: execute
wave: 3
depends_on: [08-02]
files_modified:
  - nexo/data/models_nexo.py
  - nexo/data/schema_guard.py
  - nexo/data/sql/nexo/migration_add_users_nombre.sql
  - nexo/data/repositories/nexo.py
  - api/routers/usuarios.py
  - templates/ajustes_usuarios.html
  - tests/auth/test_users_nombre_column.py
  - tests/routers/test_usuarios_nombre_form.py
autonomous: true
gap_closure: false
requirements: [UIREDO-02, UIREDO-06]
tags: [db, migration, users, nombre, ajustes-usuarios, nexo, phase-8]
user_setup: []

must_haves:
  truths:
    - "`nexo.users` has a new `nombre VARCHAR(120) NULL` column, nullable with no default, added via an idempotent migration."
    - "Existing rows are backfilled with a sensible display name derived from the email local-part (`e.eguskiza` → `E.eguskiza`)."
    - "`schema_guard` in lifespan verifies the column is present; if missing and `NEXO_AUTO_MIGRATE=true`, it runs the migration automatically (reusing the Phase 3 pattern)."
    - "`NexoUser` ORM model declares `nombre: Mapped[str | None]`."
    - "`/ajustes/usuarios` create + edit form has an optional `nombre` input labelled `Nombre (opcional)` per D-24 + D-26; the backend router persists it (empty string normalised to NULL)."
    - "`api/deps.py` continues to expose `current_user` via `render()` — since `NexoUser.nombre` is an ORM column, it's available on every authenticated request without additional plumbing."
    - "`base.html` top bar display-name logic from Plan 08-02 resolves to `user.nombre` when populated, else `email.split('@')[0]|capitalize` — the fallback works for legacy pre-migration users."
  artifacts:
    - path: "nexo/data/models_nexo.py"
      provides: "NexoUser.nombre column added to the ORM model"
      contains: "nombre = Column(String(120), nullable=True)"
    - path: "nexo/data/sql/nexo/migration_add_users_nombre.sql"
      provides: "Idempotent migration: ALTER TABLE nexo.users ADD COLUMN IF NOT EXISTS nombre VARCHAR(120); UPDATE ... backfill"
      contains: "ADD COLUMN IF NOT EXISTS nombre"
    - path: "nexo/data/schema_guard.py"
      provides: "schema_guard extended to check `users.nombre` column existence"
      contains: "('users', 'nombre')"
    - path: "api/routers/usuarios.py"
      provides: "Create/edit POST handlers accept `nombre: str | None = Form(None)` and persist it"
      contains: "nombre=nombre.strip() if nombre and nombre.strip() else None"
    - path: "templates/ajustes_usuarios.html"
      provides: "Form field `Nombre (opcional)` added to the user create + edit modal/form"
      contains: "Nombre <span class=\"text-muted font-normal\">(opcional)</span>"
  key_links:
    - from: "templates/base.html (from Plan 08-02)"
      to: "nexo.users.nombre via current_user"
      via: "Jinja expression current_user.nombre if current_user.nombre else email.split('@')[0]|capitalize"
      pattern: "current_user\\.nombre"
    - from: "api/routers/usuarios.py"
      to: "nexo/data/repositories/nexo.py (UserRepo)"
      via: "create_user / update_user signatures accept `nombre` kwarg"
      pattern: "def (create_user|update_user)"
---

<objective>
Migrate `nexo.users` to include a `nombre VARCHAR(120) NULL` column so
the new chrome (Plan 08-02) and the landing screen (Plan 08-04) can
greet users by their first name instead of falling back to the email
local-part.

This plan ships the database migration, ORM model update, schema_guard
extension, backfill for existing rows, form field + backend handler in
`/ajustes/usuarios`, and a tight regression test that proves the
column exists, the backfill worked, and the form round-trips correctly.

Purpose: Plan 08-02's top bar already reads `current_user.nombre` with
a fallback; this plan makes the fallback the exception, not the rule.
Plan 08-04 (/bienvenida) consumes the same attribute.

Output: 1 new SQL migration file, 2 edited ORM / service files, 1
edited router, 1 edited template, 2 new test files. All Phase 5 tests
continue to pass. The app continues to serve propietarios, directivos,
usuarios as before — nothing regresses.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/08-redise-o-ui-modo-claro-moderno/08-CONTEXT.md
@.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md
@.planning/phases/08-redise-o-ui-modo-claro-moderno/08-02-SUMMARY.md
@nexo/data/models_nexo.py
@nexo/data/schema_guard.py
@nexo/data/repositories/nexo.py
@api/routers/usuarios.py
@templates/ajustes_usuarios.html
@templates/base.html
@api/deps.py
@CLAUDE.md

<interfaces>
<!-- Existing NexoUser ORM model (from nexo/data/models_nexo.py:78-100). -->

```python
class NexoUser(NexoBase):
    __tablename__ = "users"
    __table_args__ = {"schema": NEXO_SCHEMA}

    id = Column(Integer, primary_key=True)
    email = Column(String(200), nullable=False, unique=True, index=True)
    password_hash = Column(String(200), nullable=False)
    role = Column(String(20), nullable=False)
    active = Column(Boolean, nullable=False, default=True)
    must_change_password = Column(Boolean, nullable=False, default=True)
    last_login = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    # Plan 08-03 adds:
    # nombre = Column(String(120), nullable=True)
    departments = relationship(...)
    sessions = relationship(...)
```

<!-- schema_guard checks — from Phase 3 pattern. -->

`nexo/data/schema_guard.py` already has a list of (table, column) tuples
consulted at lifespan startup. Plan 03-01 + 04-01 established the
pattern:

```python
REQUIRED_COLUMNS: list[tuple[str, str]] = [
    ("users", "email"),
    ("users", "password_hash"),
    ...
]
```

This plan appends `("users", "nombre")` to that list.

<!-- Existing /ajustes/usuarios router signature (from api/routers/usuarios.py) -->

Look for the existing `create_user` / `update_user` POST handlers.
They take `email`, `role`, `departments` (list), `must_change_password`.
Plan 08-03 adds `nombre: str | None = Form(None)`.
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add nombre column to NexoUser ORM model + SQL migration + schema_guard entry</name>
  <read_first>
    - `nexo/data/models_nexo.py` full file — confirm the exact class block for `NexoUser`.
    - `nexo/data/schema_guard.py` full file — the REQUIRED_COLUMNS list + the migration-driver loop.
    - `nexo/data/sql/nexo/` directory — pattern for existing migration SQL files (Phase 3 + Phase 4 references).
    - `api/config.py` — the `NEXO_AUTO_MIGRATE` flag.
  </read_first>
  <action>
### Part A — Edit `nexo/data/models_nexo.py`

Add a `nombre` column to `NexoUser` directly under `email` so callers
see it early in the attribute list. Use the exact idiom already used in
the file (no Python 3.11+ `Mapped[...]` syntax — stay consistent).

```python
# Insert after the `email = Column(...)` line at line ~83
nombre = Column(String(120), nullable=True)
```

### Part B — Create `nexo/data/sql/nexo/migration_add_users_nombre.sql`

Idempotent migration. The `IF NOT EXISTS` makes re-runs safe. The
backfill uses `regexp_replace` to capitalize the first letter of the
email local-part.

```sql
-- Phase 8 / Plan 08-03 — add nombre column to nexo.users
-- Idempotent: safe to run multiple times.
-- Backfill maps e.g. 'e.eguskiza@ecsmobility.com' -> 'E.eguskiza'

ALTER TABLE nexo.users
  ADD COLUMN IF NOT EXISTS nombre VARCHAR(120);

-- Backfill only rows where nombre is still NULL (post-migration idempotent).
UPDATE nexo.users
   SET nombre = (
           UPPER(LEFT(split_part(email, '@', 1), 1))
           || SUBSTRING(split_part(email, '@', 1) FROM 2)
       )
 WHERE nombre IS NULL;
```

### Part C — Extend `nexo/data/schema_guard.py`

Append the new tuple to the REQUIRED_COLUMNS list. If schema_guard
supports auto-applying SQL migration files via `NEXO_AUTO_MIGRATE=true`,
also register the migration file under `MIGRATION_FILES` (or the
equivalent mechanism the existing module uses — read the module first
to learn the pattern). If the mechanism does not exist yet, add a
minimal block that reads `nexo/data/sql/nexo/migration_add_users_nombre.sql`
and executes it when the column is missing AND
`settings.nexo_auto_migrate` is True.

Keep the module's existing logging, error messages, and Spanish error
output consistent.

### Part D — Smoke-test the migration locally

```bash
# In a running dev environment with Postgres up:
docker compose exec -T db psql -U oee -d oee_planta -c \
  "SELECT column_name, data_type, is_nullable \
     FROM information_schema.columns \
    WHERE table_schema='nexo' AND table_name='users' AND column_name='nombre';"
```

Expected: one row showing `nombre | character varying | YES`.

If `NEXO_AUTO_MIGRATE=false`, run the migration file manually:

```bash
docker compose exec -T db psql -U oee -d oee_planta \
  < nexo/data/sql/nexo/migration_add_users_nombre.sql
```
  </action>
  <acceptance_criteria>
    - `grep -c "nombre = Column(String(120), nullable=True)" nexo/data/models_nexo.py` returns 1.
    - `test -f nexo/data/sql/nexo/migration_add_users_nombre.sql` returns 0.
    - `grep -c "ADD COLUMN IF NOT EXISTS nombre" nexo/data/sql/nexo/migration_add_users_nombre.sql` returns 1.
    - `grep -c "('users', 'nombre')" nexo/data/schema_guard.py` returns 1.
    - `ruff check nexo/` exit 0, `mypy nexo/` exit 0.
    - Running the app locally (`make dev` or `docker compose up web`) succeeds without `schema_guard` raising an "missing column" exception when the migration has been applied.
  </acceptance_criteria>
  <verify>
    <automated>grep -q "nombre = Column(String(120), nullable=True)" nexo/data/models_nexo.py &amp;&amp; test -f nexo/data/sql/nexo/migration_add_users_nombre.sql &amp;&amp; grep -q "ADD COLUMN IF NOT EXISTS nombre" nexo/data/sql/nexo/migration_add_users_nombre.sql &amp;&amp; grep -q "('users', 'nombre')" nexo/data/schema_guard.py &amp;&amp; ruff check nexo/ &amp;&amp; mypy nexo/</automated>
  </verify>
  <done>ORM column added, migration file present + idempotent, schema_guard aware. Linters happy.</done>
</task>

<task type="auto">
  <name>Task 2: Wire nombre through UserRepo + /ajustes/usuarios form + router</name>
  <read_first>
    - `nexo/data/repositories/nexo.py` — existing `UserRepo` methods (`create_user`, `update_user`).
    - `api/routers/usuarios.py` — POST create + POST edit endpoints.
    - `templates/ajustes_usuarios.html` full file — find the current `<form>` for create + edit modal.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Form inputs" + §"Label pattern (D-24)" + §"`(opcional)` marker (D-26)".
  </read_first>
  <action>
### Part A — Extend `UserRepo` (or equivalent) in `nexo/data/repositories/nexo.py`

For both `create_user` and `update_user`, add an optional `nombre: str | None = None` kwarg. Persist empty / whitespace-only strings as `NULL`:

```python
def create_user(self, *, email, password_hash, role, nombre=None, ...):
    ...
    user = NexoUser(
        email=email,
        password_hash=password_hash,
        role=role,
        nombre=(nombre.strip() if nombre and nombre.strip() else None),
        ...
    )
    ...

def update_user(self, user_id, *, nombre=None, ...):
    ...
    if nombre is not None:
        user.nombre = (nombre.strip() if nombre.strip() else None)
    ...
```

Preserve existing callers — they pass no `nombre`, so the default
`None` keeps behaviour identical.

### Part B — Extend `api/routers/usuarios.py`

Locate the POST create handler and the POST update handler (usually
`create_user_post` / `update_user_post`). Add:

```python
from fastapi import Form

async def create_user_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    nombre: str | None = Form(None),      # ← new
    departments: list[str] = Form([]),
    ...
):
    ...
    user = user_repo.create_user(
        email=email_norm,
        password_hash=hash_password(password),
        role=role,
        nombre=nombre,                    # ← pass through
        department_codes=departments,
        ...
    )
    ...
```

Same for update. The router preserves existing Phase 5 RBAC guards — do
NOT remove the `require_permission("usuarios:manage")` dependency.

### Part C — Update `templates/ajustes_usuarios.html`

Add a `Nombre (opcional)` input to both the create form AND the edit
form/modal. Place it BEFORE the email field for natural reading order.
Use the UI-SPEC §Label pattern markup:

```jinja
<label class="block text-sm font-semibold text-body mb-1" for="user-nombre">
  Nombre <span class="text-muted font-normal">(opcional)</span>
</label>
<input type="text"
       id="user-nombre"
       name="nombre"
       maxlength="120"
       value="{{ user.nombre if user and user.nombre else '' }}"
       class="input-inline"
       autocomplete="name">
<p class="mt-1 text-sm text-error" role="alert">{{ errors.nombre if errors else '' }}</p>
```

For the create form, `value` stays empty. For the edit modal, bind to
the loaded user's `nombre` if present.

Do NOT touch the rest of the form (email, password, role, departments,
must_change_password fields). Do NOT touch the Phase 5 `{% if can() %}`
wrappers around destructive row actions.
  </action>
  <acceptance_criteria>
    - `grep -cE "def create_user\b|def update_user\b" nexo/data/repositories/nexo.py` returns a value ≥ 2 and both signatures show `nombre`.
    - `grep -c "nombre: str | None = Form(None)" api/routers/usuarios.py` returns 2 (create + edit).
    - `grep -c "name=\"nombre\"" templates/ajustes_usuarios.html` returns 1 or more (form field present; 1 for inline form + optional 1 if there is a separate edit modal — at least 1).
    - `grep -c "Nombre <span class=\"text-muted font-normal\">(opcional)</span>" templates/ajustes_usuarios.html` returns 1 or more.
    - `ruff check api/ nexo/` exit 0, `mypy api/ nexo/` exit 0.
  </acceptance_criteria>
  <verify>
    <automated>grep -q "nombre: str | None = Form(None)" api/routers/usuarios.py &amp;&amp; grep -q "name=\"nombre\"" templates/ajustes_usuarios.html &amp;&amp; ruff check api/ nexo/ &amp;&amp; mypy api/ nexo/</automated>
  </verify>
  <done>Repo + router + template all round-trip `nombre`. Phase 5 gating intact.</done>
</task>

<task type="auto">
  <name>Task 3: Regression tests — migration idempotency, form round-trip, base.html display-name fallback</name>
  <read_first>
    - `tests/auth/` existing test files for NexoUser fixtures.
    - `tests/routers/` existing patterns for form-submission tests.
    - `tests/conftest.py` for DB session fixtures.
  </read_first>
  <action>
### Part A — `tests/auth/test_users_nombre_column.py`

```python
"""Regression for Phase 8 / Plan 08-03: nexo.users.nombre column.

Locks:
1. NexoUser model has `nombre` as optional String(120).
2. Schema guard lists the column as required.
3. Migration is idempotent (safe to run twice).
4. New users can be created with and without nombre.
5. Empty / whitespace nombre is stored as NULL.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import inspect, text

from nexo.data.models_nexo import NexoUser
from nexo.data.schema_guard import REQUIRED_COLUMNS  # or equivalent name


_MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "nexo"
    / "data"
    / "sql"
    / "nexo"
    / "migration_add_users_nombre.sql"
)


def test_migration_file_exists():
    assert _MIGRATION.exists()


def test_migration_is_idempotent_text():
    sql = _MIGRATION.read_text(encoding="utf-8")
    assert "ADD COLUMN IF NOT EXISTS nombre" in sql
    assert "UPDATE nexo.users" in sql
    assert "WHERE nombre IS NULL" in sql


def test_nexo_user_has_nombre_attribute():
    assert hasattr(NexoUser, "nombre")


def test_schema_guard_requires_nombre_column():
    assert ("users", "nombre") in REQUIRED_COLUMNS


def test_nombre_column_is_nullable_and_short(nexo_db_session):
    """Runtime check against the migrated schema."""
    engine = nexo_db_session.bind
    insp = inspect(engine)
    cols = {c["name"]: c for c in insp.get_columns("users", schema="nexo")}
    assert "nombre" in cols
    assert cols["nombre"]["nullable"] is True
    # VARCHAR(120) — length attribute is dialect-dependent; sanity check only
    # for the type name.
    assert "VARCHAR" in str(cols["nombre"]["type"]).upper()


def test_backfill_populates_existing_rows(nexo_db_session):
    """Backfill must have run — for every non-null email, nombre is populated."""
    rows = list(
        nexo_db_session.execute(
            text("SELECT email, nombre FROM nexo.users WHERE email IS NOT NULL")
        )
    )
    for email, nombre in rows:
        assert nombre is not None, f"{email} still has NULL nombre after migration"
        # First char uppercase (from UPPER(LEFT(...)))
        assert nombre[0] == nombre[0].upper()


def test_create_user_with_nombre(nexo_db_session, user_factory):
    u = user_factory(email="ada@example.com", nombre="Ada Lovelace")
    assert u.nombre == "Ada Lovelace"


def test_create_user_without_nombre(nexo_db_session, user_factory):
    u = user_factory(email="nobody@example.com", nombre=None)
    assert u.nombre is None


def test_empty_nombre_stored_as_null(nexo_db_session, user_factory):
    u = user_factory(email="blank@example.com", nombre="   ")
    assert u.nombre is None
```

If `user_factory` fixture doesn't exist, add it to `tests/conftest.py`
minimally: creates a NexoUser with the given kwargs, adds to session,
commits, returns the row. If that would contaminate Phase 5's fixtures,
create a local `tests/auth/conftest.py` with the factory scoped to this
file.

### Part B — `tests/routers/test_usuarios_nombre_form.py`

```python
"""Regression for Phase 8 / Plan 08-03: /ajustes/usuarios accepts nombre.

Locks:
1. The HTML form renders the nombre input with the exact UI-SPEC label.
2. POST /ajustes/usuarios with nombre=Ada persists correctly.
3. POST without nombre (empty form value) stores NULL.
4. Phase 5 `usuarios:manage` permission still guards the endpoint —
   a non-propietario POST gets 403 (no silent write).
"""

from __future__ import annotations

import pytest

# Import client + auth helpers from existing Phase 5 test patterns.


def test_form_renders_nombre_field(propietario_client):
    resp = propietario_client.get("/ajustes/usuarios")
    assert resp.status_code == 200
    body = resp.text
    assert 'name="nombre"' in body
    # UI-SPEC §Label pattern
    assert 'Nombre <span class="text-muted font-normal">(opcional)</span>' in body


def test_post_with_nombre_persists(propietario_client, nexo_db_session):
    resp = propietario_client.post(
        "/ajustes/usuarios",
        data={
            "email": "created@example.com",
            "password": "ValidPass1234",
            "role": "usuario",
            "nombre": "Ada",
            "departments": ["ingenieria"],
        },
        follow_redirects=False,
    )
    assert resp.status_code in (200, 302, 303)
    from nexo.data.models_nexo import NexoUser
    row = nexo_db_session.query(NexoUser).filter_by(email="created@example.com").one()
    assert row.nombre == "Ada"


def test_post_without_nombre_stores_null(propietario_client, nexo_db_session):
    resp = propietario_client.post(
        "/ajustes/usuarios",
        data={
            "email": "no-name@example.com",
            "password": "ValidPass1234",
            "role": "usuario",
            "nombre": "",   # empty
            "departments": ["ingenieria"],
        },
        follow_redirects=False,
    )
    assert resp.status_code in (200, 302, 303)
    from nexo.data.models_nexo import NexoUser
    row = nexo_db_session.query(NexoUser).filter_by(email="no-name@example.com").one()
    assert row.nombre is None


def test_non_propietario_cannot_post(directivo_client):
    resp = directivo_client.post(
        "/ajustes/usuarios",
        data={"email": "shouldfail@example.com", "password": "ValidPass1234",
              "role": "usuario", "nombre": "Evil"},
        follow_redirects=False,
    )
    # Phase 5: HTML 403 → redirect+cookie OR JSON 403
    assert resp.status_code in (302, 303, 403)
```

Reuse the `propietario_client` / `directivo_client` fixtures already
present in `tests/conftest.py` or `tests/auth/conftest.py` from Phase 5.
If they do not exist under those exact names, port the nearest
equivalent (`client_as_propietario`, `authenticated_client`, etc.) and
log the rename in the SUMMARY.
  </action>
  <acceptance_criteria>
    - `test -f tests/auth/test_users_nombre_column.py` returns 0.
    - `test -f tests/routers/test_usuarios_nombre_form.py` returns 0.
    - `pytest tests/auth/test_users_nombre_column.py tests/routers/test_usuarios_nombre_form.py -x -q` exits 0.
    - `pytest tests/ -x -q` (full suite) exits 0.
    - `ruff check tests/` exit 0.
    - `ruff format --check tests/auth/test_users_nombre_column.py tests/routers/test_usuarios_nombre_form.py` exit 0.
  </acceptance_criteria>
  <verify>
    <automated>ruff check tests/auth/test_users_nombre_column.py tests/routers/test_usuarios_nombre_form.py &amp;&amp; ruff format --check tests/auth/test_users_nombre_column.py tests/routers/test_usuarios_nombre_form.py &amp;&amp; pytest tests/auth/test_users_nombre_column.py tests/routers/test_usuarios_nombre_form.py -x -q &amp;&amp; pytest tests/ -x -q</automated>
  </verify>
  <done>Migration locked, form round-trips, Phase 5 RBAC untouched, full suite green.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Form → DB | User-supplied `nombre` crosses into persistence. Must be sanitised. |
| DB → browser (display) | `current_user.nombre` rendered in `base.html` and (eventually) landing — must be HTML-escaped. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-08-03-01 | Tampering | POST /ajustes/usuarios with malicious nombre (script tags, long strings) | mitigate | `VARCHAR(120)` hard limit at DB level; Jinja auto-escapes `{{ }}` so rendered values cannot break out. No use of `|safe` on `nombre`. |
| T-08-03-02 | Information Disclosure | `nombre` displayed to other users via audit log / listing | accept | `ajustes/usuarios` is propietario-only (Phase 5 `usuarios:manage` → `[]` = propietario); audit log already filtered. No new disclosure surface. |
| T-08-03-03 | Elevation of Privilege | Can a non-propietario set their own nombre via another endpoint? | mitigate | `nombre` is only mutable via `/ajustes/usuarios` (gated) in Mark-III. `/cambiar-password` + profile self-service are NOT opened here. |
| T-08-03-04 | Denial of Service | Oversized nombre attempting to blow up memory | mitigate | `maxlength="120"` HTML + VARCHAR(120) DB cap + Form-level FastAPI validation. |
</threat_model>

<verification>
Whole-phase sanity after this plan:

1. `pytest tests/ -x -q` exit 0.
2. `ruff check api/ nexo/` and `ruff format --check api/ nexo/` pass.
3. Manual:
   - `make dev`. Log in as propietario. The top-bar greeting circle now
     shows the initial of `propietario`'s `nombre` (from the backfill,
     e.g. `P` for `Propietario` derived from `propietario@…`).
   - Visit `/ajustes/usuarios`. The "Crear usuario" form has a
     `Nombre (opcional)` field. Creating `Ada <ada@x>` then logging in as
     Ada shows `A` in the avatar and `Ada` in the user menu dropdown.
4. Roll back test: comment the column addition temporarily; schema_guard
   should refuse to start with a clear Spanish error message.
</verification>

<success_criteria>
- `nexo.users.nombre` exists, is nullable, is VARCHAR(120).
- Backfill ran for every existing row (no NULLs for pre-existing users).
- ORM, schema_guard, repo, router, template all round-trip nombre.
- Phase 5 RBAC on `/ajustes/usuarios` intact.
- `pytest tests/ -x -q` green.
</success_criteria>

<output>
After completion, create `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-03-SUMMARY.md` with:

- Migration SQL committed.
- ORM + schema_guard + repo + router + template changes listed.
- Tests added: counts per file.
- Manual verification notes (is the dev user now showing a real name?).
- Handoff to 08-04 (landing): `current_user.nombre` is the canonical
  source for the greeting `{nombre}`; email-local-part is the fallback.
</output>