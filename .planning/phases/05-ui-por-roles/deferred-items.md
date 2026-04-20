# Phase 05 — Deferred Items

Issues found during plan execution that are out-of-scope for the current
plan but worth tracking.

---

## Plan 05-01 — discovered during regression sweep (Task 4)

### DEF-05-01-A: `tests/routers/test_thresholds_crud.py` recalibrate tests leak DB rows

**Status:** Pre-existing from Phase 4 closure (baseline `f726275`).

**Tests failing:**

- `tests/routers/test_thresholds_crud.py::test_recalibrate_preview_and_confirm_persists`
  — asserts `preview["sample_size"] == 15`, gets `18`.
- `tests/routers/test_thresholds_crud.py::test_recalibrate_filters_outliers_under_500ms`
  — same class of failure.
- `tests/routers/test_thresholds_crud.py::test_recalibrate_insufficient_data_returns_400`
  — observed during Plan 05-04 regression sweep (2026-04-20). Sample-size
  contamination likely causes recalibrate to find ≥10 samples where the
  test expected <10 (so it returns 200 instead of 400). Same root-cause
  class as the other two; reverting to `f0d984c~3` reproduces.

**Root cause (suspected):** `nexo.query_log` rows from prior test runs or
from co-running tests in the same suite are not purged before the
recalibrate preview runs, so `compute_factor` sees more samples than the
test seeded. The test fixture likely needs to truncate / delete query_log
rows for the specific `user_id + endpoint` tuple before the test body
runs.

**Reproduction:**

```bash
git checkout f726275 -- nexo/services/auth.py api/deps.py  # baseline
NEXO_PG_HOST=localhost NEXO_PG_PORT=5433 NEXO_PG_USER=oee \
NEXO_PG_PASSWORD=oee NEXO_PG_DB=oee_planta \
NEXO_SECRET_KEY=testsecretkeytestsecretkeytestsecretkey \
pytest tests/routers/test_thresholds_crud.py::test_recalibrate_preview_and_confirm_persists \
       tests/routers/test_thresholds_crud.py::test_recalibrate_filters_outliers_under_500ms
# → 2 failed (same error)
```

**Not in scope for Plan 05-01:** the 05-01 refactor (extract `can()`,
Jinja global) does not touch `nexo/query_log`, `compute_factor`, or the
`/api/thresholds/*/recalibrate` endpoint. Reverting auth.py/deps.py to
the baseline reproduces the failure.

**Recommended resolution plan:** open a small fix plan (Phase 4
follow-up or Phase 6 test-hardening) to make the recalibrate test
fixture purge `nexo.query_log` rows for the synthetic user **before**
seeding, rather than relying on teardown-only.

**Re-observed during Plan 05-05 (2026-04-20):** Regression sweep after
button gating + HTML GET hardening re-reproduced the exact same failure
on `test_recalibrate_insufficient_data_returns_400` (200 instead of
400). Scope-boundary: Plan 05-05 touches only `templates/*.html`,
`api/routers/pages.py` (route deps), and new test files — no overlap
with `nexo/query_log` or `/api/thresholds/*/recalibrate`. Confirmed
pre-existing; tracked here, not fixed by Plan 05-05.

---

## Phase 5.1 polish candidates

Findings from `05-REVIEW.md` (2026-04-20 standard review) that are
MEDIUM or LOW severity. Out of scope for the Phase 5 close (HI-01 was
the only HIGH; fixed in commit `f821e46`). Ordered by severity.

### DEF-05-REVIEW-MD-01: `recursos.html` drawer edits not gated for read-only users

**Severity:** MEDIUM
**File:** `templates/recursos.html:154, 179, 201-214, 222`

The per-card "Delete máquina" and the toolbar's "Detectar / Add /
Guardar / Delete" are wrapped in `{% if can(current_user,
"recursos:edit") %}`, but once a read-only user opens the drawer the
following controls remain mutating:

- Line 154: `toggleActive()` bound to the "activo" toggle.
- Line 179: `input[type=number]` for `centro_trabajo` with
  `@input=...; dirty = true`.
- Lines 201-214: ciclo rows (`input[referencia]`,
  `input[tiempo_ciclo]`, `removeCiclo` button) — editable despite
  "Guardar" being hidden.
- Line 222: `addCiclo()` "+ Añadir referencia" button.

**Impact:** UX confusion (editable fields that never persist, "Sin
guardar" badge that can't be cleared). Not a privilege bug today
(backend gate still blocks save) but the gap widens the first time
someone adds a keyboard shortcut or autosave.

**Fix sketch (per review):** wrap each mutating element with `{% if
can(current_user, "recursos:edit") %}` to match the rest of the file,
or pass the permission as an Alpine prop and gate handlers in JS
(`x-data="recursosPage({{ can(current_user, 'recursos:edit')|tojson
}})"`). Prefer the Jinja wrap (consistent with the existing pattern in
this file); also add `:disabled` to reference/tiempo_ciclo inputs for
visual feedback.

### DEF-05-REVIEW-MD-02: `base.html` separator relies on branch-order invariant for empty `icon_paths`

**Severity:** MEDIUM
**File:** `templates/base.html:54, 61-73`

The separator tuple `("_sep1", "", "", "", None)` sets `icon_paths=""`;
the loop's `{% for d in icon_paths.split('||') %}` sits INSIDE the
`key.startswith('_sep')`-NOT branch, so today the split never runs.
Invariant is implicit: rename the separator prefix or add a non-nav
entry with `icon_paths=None` and the template either renders nothing or
throws `AttributeError: 'NoneType' object has no attribute 'split'`.

**Fix sketch:** either (a) set separator `icon_paths=None` and guard
the split (`{% if icon_paths %}{% for d in ... %}{% endfor %}{% endif
%}`) or (b) move `nav_items` to a typed Python structure in
`api/deps.py` (TypedDict/dataclass), enforcing schema at edit time
rather than render time. Option (b) is the more durable fix.

### DEF-05-REVIEW-MD-03: 403 handler parses `exc.detail` as string prefix — brittle contract

**Severity:** MEDIUM
**File:** `api/main.py:248-252`

```python
raw = str(exc.detail or "")
perm = raw.replace("Permiso requerido: ", "") if raw.startswith(
    "Permiso requerido: "
) else ""
friendly = _friendly_permission_label(perm) if perm else "esta seccion"
```

The literal prefix `"Permiso requerido: "` (produced by
`nexo/services/auth.py:287`) is the only channel between
`require_permission` and the 403 handler. Any rewording (translation,
punctuation) silently degrades the toast label to `"esta seccion"` with
zero test failure.

**Fix sketch:** subclass `HTTPException` with a typed `permission` attr
(`class PermissionDenied(HTTPException): ...`), update
`require_permission` to raise it, and switch the 403 handler to
`perm = getattr(exc, "permission", "")`. Keeps string-detail backward
compatible (for any test/client parsing `detail` raw) but adds a typed
channel. Add a unit test asserting the structured extraction.

### DEF-05-REVIEW-MD-04: `ajustes_conexion.html` comments reference stale line numbers

**Severity:** MEDIUM
**File:** `templates/ajustes_conexion.html:14, 110, 121`

```jinja
{# ── Conexion SQL Server (extraído verbatim de ajustes.html:74-168) ── #}
...
{# ── Info sistema (extraído de ajustes.html:170-177) ── #}
...
/* Extraído verbatim de ajustes.html:180-207 (ajustesPage()).
```

Plan 05-04 moved the blocks here and reduced `ajustes.html` to a 6-card
hub, so the referenced line ranges no longer contain that code. Future
readers following the trail will be confused.

**Fix sketch:** drop the line numbers entirely (the file owns the code
now) or swap for a prose-only reference: `{# Extracted from
ajustes.html during Plan 05-04 (pre-refactor 'Conexion SQL Server' card
block) #}`.

### DEF-05-REVIEW-LO-01: `ajustes_conexion.html` calls `showToast` with wrong arity

**Severity:** LOW (tech-debt sweep)
**File:** `templates/ajustes_conexion.html:138`

```javascript
showToast('Configuracion guardada'); await this.testConnection();
```

`base.html:188` defines `window.showToast = function(type, title,
msg)`. One-arg call maps to `type='Configuracion guardada'`,
`title=undefined`, `msg=undefined` → toast renders with "undefined"
title and the info fallback icon/style. Bug predates Phase 5 (verbatim
copy from pre-refactor `ajustes.html`).

**Fix sketch:** `showToast('info', 'Guardado', 'Configuracion de
conexion guardada');`. Also sweep `historial.html`, `recursos.html`,
`ciclos_calc.html`, `bbdd.html`, `plantillas.html` — all use 1-arg or
2-arg `showToast` with the same symptom. Worth a dedicated small plan.

### DEF-05-REVIEW-LO-02: `PERMISSION_MAP` docstring references stale plan IDs

**Severity:** LOW (doc drift)
**File:** `nexo/services/auth.py:229-231`

```python
"aprobaciones:manage": [],  # Plan 04-03 (QUERY-06) — propietario-only
"limites:manage":     [],   # Plan 04-04 (QUERY-02) — propietario-only
"rendimiento:read":   [],   # Plan 04-04 (D-11) — propietario-only
```

`QUERY-06`, `QUERY-02`, `D-11` are Phase 4 internal plan identifiers,
not stable refs. Readers can't locate them.

**Fix sketch:** drop the identifiers and keep only the plan number, or
replace with a doc reference (`# See docs/AUTH_MODEL.md §Apendice
PERMISSION_MAP`, already referenced at `auth.py:198`).

### DEF-05-REVIEW-LO-03: `_PERMISSION_LABELS` uses unaccented Spanish — convention undocumented

**Severity:** LOW (docs)
**File:** `api/main.py:195-217`

Friendly labels drop Spanish accents (`"la configuracion"`,
`"auditoria"`). Consistent with the rest of the codebase
(`historial.html:35`, `recursos.html:74`, etc.) but the convention is
nowhere in docs — a future contributor may reach for the accented form
and create inconsistency.

**Fix sketch:** add a short "UI text conventions" section to
`docs/BRANDING.md`: "All user-facing strings in toasts, labels, and
placeholders use unaccented Spanish for compatibility with legacy Jinja
templates and to avoid double-encoding issues in HTMX partial
responses." Non-blocking.
