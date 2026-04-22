---
phase: 08-redise-o-ui-modo-claro-moderno
plan: 03
subsystem: users-nombre-column
tags: [db, migration, users, nombre, ajustes-usuarios, nexo, phase-8]
requires:
  - 08-02-chrome-topbar-drawer-toasts (base.html already reads current_user.nombre via getattr)
  - 05-ui-por-roles (usuarios:manage permission gating on /ajustes/usuarios)
provides:
  - nexo.users.nombre VARCHAR(120) NULL column + backfill
  - NexoUser.nombre ORM attribute
  - UserRow DTO nombre field (pipeline-ready for 08-04 landing)
  - /ajustes/usuarios form + list with Nombre (opcional) field
  - schema_guard REQUIRED_COLUMNS mechanism (generic for future column adds)
affects:
  - templates/base.html (happy path: display_name resolves to user.nombre)
  - api/deps.py (getattr fallback still present; no longer the hot path)
tech-stack:
  added:
    - "schema_guard.REQUIRED_COLUMNS + _COLUMN_MIGRATIONS (generic)"
  patterns:
    - "Idempotent SQL migration via ALTER TABLE ... ADD COLUMN IF NOT EXISTS"
    - "Backfill guard via WHERE nombre IS NULL (safe re-run)"
    - "Normalise empty/whitespace form input to NULL (database semantic)"
    - "Separate last_login_display (str) from tojson-safe dict (Rule 1 bugfix)"
key-files:
  created:
    - nexo/data/sql/nexo/migration_add_users_nombre.sql
    - tests/auth/test_users_nombre_column.py
    - tests/routers/test_usuarios_nombre_form.py
  modified:
    - nexo/data/models_nexo.py (NexoUser.nombre column)
    - nexo/data/schema_guard.py (REQUIRED_COLUMNS + column migration runner)
    - nexo/data/dto/nexo.py (UserRow.nombre field)
    - nexo/data/repositories/nexo.py (UserRepo get_by_email + list_all populate nombre)
    - api/routers/usuarios.py (POST /crear + POST /{id}/editar accept nombre; Rule 1 _serialize_user datetime bugfix)
    - templates/ajustes_usuarios.html (Nombre (opcional) field in form + list column)
decisions:
  - "Column nullable + no default: pre-existing rows backfilled with UPPER(LEFT(local-part, 1)) || SUBSTRING(local-part FROM 2) — e.g. e.eguskiza -> E.eguskiza. Fallback to email local-part preserved in templates for users created later without nombre."
  - "schema_guard extended with generic REQUIRED_COLUMNS mechanism instead of hardcoding users.nombre. Future column additions register a (table, column) tuple + _COLUMN_MIGRATIONS entry."
  - "Migration applies via connection.exec_driver_sql (raw psycopg2 path) because SQLAlchemy text() rejects multi-statement payloads. Wrapped in engine.begin() for transactional safety."
  - "Rule 1 bugfix scope: _serialize_user returned datetime; the pre-existing {{ u|tojson }} in openEdit()/openReset() Alpine handlers exploded for any logged-in user (bug latent since Plan 02-04). Fix: preformat last_login as string server-side."
  - "Alpine form.nombre initialized to user.nombre || '' so NULL renders as empty input; backend re-normalises whitespace to NULL on POST."
metrics:
  duration_minutes: 30
  tasks_completed: 3
  files_created: 3
  files_modified: 6
  tests_added: 14
  completed: 2026-04-22
---

# Phase 8 Plan 03: Users Nombre Column Migration Summary

One-liner: Adds `nexo.users.nombre VARCHAR(120) NULL` with idempotent migration + backfill + ORM/DTO/repo wiring + `Nombre (opcional)` field in `/ajustes/usuarios` form, making the Phase 8 top bar's display_name fallback the exception rather than the rule.

## What was built

### Database migration

- `nexo/data/sql/nexo/migration_add_users_nombre.sql` — idempotent SQL:
  - `ALTER TABLE nexo.users ADD COLUMN IF NOT EXISTS nombre VARCHAR(120)` (Postgres ≥ 9.6).
  - `UPDATE nexo.users SET nombre = UPPER(LEFT(local, 1)) || SUBSTRING(local FROM 2) WHERE nombre IS NULL` (guard clause keeps the backfill idempotent — second run does nothing).
- Applied in dev via `docker compose exec -T db psql -U oee -d oee_planta < …`. Verified: 4 rows backfilled on first run, 0 rows on second run (idempotent), notice on re-run (`column already exists, skipping`).

### ORM + schema_guard

- `NexoUser.nombre = Column(String(120), nullable=True)` declared after `email` so the attribute appears early in the mapper.
- `schema_guard.REQUIRED_COLUMNS: tuple[tuple[str, str], ...]` — a new generic mechanism that verifies specific columns after the table check. `('users', 'nombre')` is the first entry; future columns register here plus an entry in `_COLUMN_MIGRATIONS` mapping to their idempotent SQL file.
- `schema_guard.verify` now runs `_missing_columns()` after the existing table check. If a column is missing and `NEXO_AUTO_MIGRATE=true`, `_apply_column_migration()` runs the registered SQL via `engine.begin() + connection.exec_driver_sql()` (multi-statement safe). Otherwise, a Spanish `RuntimeError` includes the exact SQL path the operator must run.

### Repository / DTO wiring

- `UserRow.nombre: Optional[str] = None` added to the Pydantic DTO (`from_attributes=True` picks it up automatically from the ORM).
- `UserRepo.get_by_email` and `UserRepo.list_all` now populate `nombre` into the DTO. Pipeline-ready for Plan 08-04 (`/bienvenida` greeting) without additional plumbing.

### Router + form

- `api/routers/usuarios.py`:
  - `_normalize_nombre(str | None) -> str | None` — shared helper: `None` → `None`; whitespace-only → `None`; otherwise stripped value. Semantic: empty form value stores NULL so the template's email-local-part fallback can activate.
  - `POST /ajustes/usuarios/crear` signature adds `nombre: str | None = Form(None)`; passes the normalised value to `NexoUser(...)` constructor.
  - `POST /ajustes/usuarios/{id}/editar` adds the same parameter; updates `user.nombre = _normalize_nombre(nombre)` before commit.
  - `_serialize_user` refactored — returns `last_login_display: str` (preformatted `YYYY-MM-DD HH:MM` or `""`) instead of the raw `datetime`. See "Deviations".
- Phase 5 RBAC preserved: router-level `dependencies=[Depends(require_permission("usuarios:manage"))]` unchanged. Test `test_non_propietario_cannot_post_crear` verifies that a `directivo` + `ingenieria` user receives 302/303/403 and the DB row is NOT created (zero-trust).

### Template

- `templates/ajustes_usuarios.html`:
  - New form field placed **before** the email input (natural reading order: Nombre → Email → credentials):
    ```html
    <label class="block text-sm font-semibold text-body mb-1" for="user-nombre">
      Nombre <span class="text-muted font-normal">(opcional)</span>
    </label>
    <input type="text" id="user-nombre" name="nombre" x-model="form.nombre"
           maxlength="120" autocomplete="name" class="…">
    ```
    Exactly matches UI-SPEC §Label pattern (D-24) + `(opcional)` marker (D-26).
  - Table gains a new leftmost `Nombre` column; `(sin nombre)` italic renders when NULL (defensive — backfill makes this rare).
  - Alpine form state includes `nombre`. `openEdit(user)` binds `user.nombre || ''` so NULL renders as empty input; `openCreate()` initialises to `''`.

### Tests

- `tests/auth/test_users_nombre_column.py` (9 tests = 4 offline + 5 integration):
  - Migration file exists, contains idempotent markers.
  - `NexoUser.nombre` is `VARCHAR` and nullable.
  - `('users', 'nombre')` registered in `REQUIRED_COLUMNS`.
  - Runtime inspection of Postgres schema confirms column + nullability + type.
  - Backfill populated all pre-migration rows with uppercase-first-char names.
  - CRUD: create with nombre, without nombre, and at 120-char hard limit (T-08-03-04 DoS mitigation).
- `tests/routers/test_usuarios_nombre_form.py` (5 integration tests):
  - `GET /ajustes/usuarios` renders `name="nombre"` + UI-SPEC literal label.
  - `POST /crear` with `nombre="Ada Lovelace"` persists the value.
  - `POST /crear` with whitespace-only nombre persists NULL (fallback-safe).
  - `POST /{id}/editar` round-trip: `None → "Bob" → None`.
  - Phase 5 RBAC: non-propietario receives 302/303/403 and the target user is NOT created.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 — Bug] `_serialize_user` datetime leaked into `{{ u|tojson }}`**

- **Found during:** Task 3 (running `test_form_renders_nombre_field`).
- **Issue:** `_serialize_user` returned `last_login: datetime`. The template inlined `{{ u|tojson }}` inside the Alpine `@click='openEdit(…)'` handlers (see template lines 99, 103, 291). `TypeError: Object of type datetime is not JSON serializable` fired for **any** render that included a user with a non-NULL `last_login` — i.e. virtually any real user. Pre-existing bug since Plan 02-04; never triggered by prior tests because they didn't log in + GET the page in the same module with live users.
- **Fix:** Replaced `last_login: datetime` in the serialised dict with `last_login_display: str` (preformatted `YYYY-MM-DD HH:MM` or empty string). Template uses the preformatted field directly. Dict is now fully JSON-serialisable.
- **Files modified:** `api/routers/usuarios.py`, `templates/ajustes_usuarios.html`.
- **Commit:** `dc8ab87`.

**2. [Rule 2 — Completeness] Generic `REQUIRED_COLUMNS` mechanism instead of ad-hoc check**

- **Plan suggested:** Append `("users", "nombre")` to an existing `REQUIRED_COLUMNS` list — but that list didn't exist in `schema_guard.py` (plan assumption was wrong; only `CRITICAL_TABLES` existed).
- **Fix:** Added a new `REQUIRED_COLUMNS` constant plus `_COLUMN_MIGRATIONS` mapping `(table, column)` → migration SQL path. Future column additions in Mark-III follow the same pattern without touching the verify function.
- **Files modified:** `nexo/data/schema_guard.py`.
- **Commit:** `f0c643f`.

**3. [Rule 2 — Completeness] `_serialize_user` also returns `nombre` for Alpine `openEdit()`**

- **Plan:** Template edit modal binds to user data via `openEdit({{ u|tojson }})`. For the edit flow to show the current `nombre` in the input, the serialised dict must include it.
- **Fix:** Added `"nombre": u.nombre` to `_serialize_user`; Alpine `openEdit()` now reads `user.nombre || ''` and rebinds the input.
- **Files modified:** `api/routers/usuarios.py`, `templates/ajustes_usuarios.html` (Alpine state).
- **Commit:** `a867f77`.

### Auth gates

None. The plan was fully autonomous; no auth prompts.

## Manual verification notes

- `docker compose exec -T db psql -U oee -d oee_planta -c "SELECT email, nombre FROM nexo.users"` →
  ```
  e.eguskiza@ecsmobility.com | E.eguskiza
  debug2@debug.local         | Debug2
  d3@debug3.local            | D3
  debug-gerencia@test.local  | Debug-gerencia
  ```
- Migration re-run: `ALTER TABLE / UPDATE 0` (idempotent).
- `docker compose exec -T web python -c "from nexo.data.models_nexo import NexoUser; print(NexoUser.__table__.columns['nombre'].type, NexoUser.__table__.columns['nombre'].nullable)"` → `VARCHAR(120) True`.
- Web container boots cleanly: `schema_guard OK — 11 tablas nexo.* + 1 columnas criticas`.

## Test results

- 14/14 new tests pass (both files).
- Full suite: 495 pass, 17 skip, 3 pre-existing failures in `tests/routers/test_thresholds_crud.py::test_recalibrate_*` (already documented in `deferred-items.md` from Plan 08-01).

## Handoff to 08-04 (bienvenida landing)

- `current_user.nombre` is now a populated ORM attribute on every authenticated request (no more `getattr` fallback on the happy path — `base.html` line 66's `getattr(current_user, 'nombre', None) or email.split('@')[0]|capitalize` now resolves to the first branch for backfilled users).
- `UserRow.nombre` is available in the DTO pipeline if 08-04 uses the repo instead of the ORM directly.
- 08-04 landing `{nombre}` greeting should use `{{ current_user.nombre or current_user.email.split('@')[0]|capitalize }}` — same fallback idiom so newly-created users without a nombre still get a friendly display.

## Threat Flags

None. The new surface is strictly additive to an already-RBAC-gated propietario-only endpoint. Auto-escape + VARCHAR(120) + HTML `maxlength="120"` mitigate T-08-03-01 / T-08-03-04 per the plan's threat model.

## Self-Check: PASSED

Files exist:
- FOUND: nexo/data/sql/nexo/migration_add_users_nombre.sql
- FOUND: tests/auth/test_users_nombre_column.py
- FOUND: tests/routers/test_usuarios_nombre_form.py
- FOUND: nexo/data/models_nexo.py (modified)
- FOUND: nexo/data/schema_guard.py (modified)
- FOUND: nexo/data/dto/nexo.py (modified)
- FOUND: nexo/data/repositories/nexo.py (modified)
- FOUND: api/routers/usuarios.py (modified)
- FOUND: templates/ajustes_usuarios.html (modified)

Commits:
- FOUND: f0c643f (feat(08-03): add nexo.users.nombre column + schema_guard column check)
- FOUND: a867f77 (feat(08-03): wire nombre through UserRow DTO, UserRepo, usuarios router + form)
- FOUND: dc8ab87 (test(08-03): regression tests for nombre column + tojson datetime bugfix)
