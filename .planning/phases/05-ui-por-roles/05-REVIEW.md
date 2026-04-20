---
phase: 05-ui-por-roles
type: review
depth: standard
status: issues_found
reviewed: 2026-04-20
critical: 0
high: 1
medium: 4
low: 3
---

# Phase 5 (ui-por-roles) — Code Review

**Reviewed:** 2026-04-20
**Depth:** standard
**Files reviewed:** 10 source files across 5 plans (auth.py, deps.py, flash.py, main.py, pages.py, base.html, ajustes.html, ajustes_conexion.html, pipeline.html, historial.html, recursos.html, ciclos_calc.html)
**Status:** issues_found

## Summary

Overall the RBAC refactor is clean and well-designed. The `can()` extraction is the single source of truth, `require_permission` trampolines correctly, 401/403 contract preserved, middleware ordering (Auth → Flash → Audit → QueryTiming) is correct, flash cookie policy (HttpOnly, SameSite=Lax, Secure in prod-only) is sound, exception handler delegates correctly for non-403 status codes, and the Jinja global is registered at import time per Pitfall 5.

The **one HIGH-severity finding** is a flash-cookie race condition in `FlashMiddleware`: when a request carries an incoming `nexo_flash` cookie AND the handler emits a new one (e.g. 403 → redirect path while a prior flash is still in-flight), the middleware's blind `delete_cookie` after `call_next` erases the newly-set cookie. UX impact only (no security), 60s TTL window, but worth fixing with a "only delete if the response didn't re-set the cookie" guard.

Medium findings concentrate on **UI consistency** (ciclos inputs editable for read-only users, inline `removeCiclo` / `addCiclo` / `toggleActive` not gated while "Guardar" is) and one minor potential Jinja template crash if `icon_paths` were ever `None` alongside a non-permissioned separator.

No CRITICAL issues: `can(None, ...)` correctly returns False without raising, no XSS (flash passed through `|tojson`), cookie is HttpOnly, eager-load of `user.departments` in AuthMiddleware prevents `DetachedInstanceError`, no hardcoded `role == "propietario"` checks outside `can()`.

---

## Summary table

| # | Sev | File | Issue |
|---|-----|---|-----|
| HI-01 | HIGH | `nexo/middleware/flash.py:32-34` | Blind `delete_cookie` after `call_next` clobbers a just-set flash on chained-403 path |
| MD-01 | MEDIUM | `templates/recursos.html:179, 201-214, 222` | Ciclo inputs, `addCiclo`, `removeCiclo`, `toggleActive`, CT edit are NOT wrapped in `can(current_user, "recursos:edit")` — read-only users see broken/ineffective edits |
| MD-02 | MEDIUM | `templates/base.html:54, 61-62` | Separator entry `("_sep1", "", "", "", None)` has `icon_paths=""`, not `None`; relies on the fact that the separator branch short-circuits before `.split('||')`. Defensive only — today it works, but an accidental rename of `_sep` prefix would crash |
| MD-03 | MEDIUM | `api/main.py:229-263` | `StarletteHTTPException` handler parses `exc.detail` as a string starting with `"Permiso requerido: "` — fragile contract coupling between `require_permission` detail text and the 403 handler |
| MD-04 | MEDIUM | `templates/ajustes_conexion.html:14` | Comment says "extraído verbatim de ajustes.html:74-168" but the new file now owns that code. The reference-by-line comment will rot the first time someone edits this file |
| LO-01 | LOW | `templates/ajustes_conexion.html:138` | `showToast('Configuracion guardada')` calls the 3-arg signature with 1 arg (title becomes the literal string, msg is empty, type defaults to `info` fallback). Behavioural but looks unintentional. Predates Phase 5 (verbatim copy) |
| LO-02 | LOW | `nexo/services/auth.py:200-232` | `PERMISSION_MAP` docstring on line 229 mentions `aprobaciones:manage` as `Plan 04-03 (QUERY-06)` and `limites:manage` as `Plan 04-04 (QUERY-02)` — references stale plan IDs (QUERY-\* is Phase 4 naming); non-blocking doc drift |
| LO-03 | LOW | `api/main.py:195-217` | `_PERMISSION_LABELS` includes Spanish-accent-free variants ("configuracion", "auditoria"). Consistent with rest of codebase but worth documenting the convention somewhere (e.g. `BRANDING.md`) |

---

## HIGH findings

### HI-01 — `FlashMiddleware.delete_cookie` after `call_next` races with a just-set cookie

**File:** `nexo/middleware/flash.py:26-34`

```python
async def dispatch(self, request: Request, call_next):
    flash = request.cookies.get(_FLASH_COOKIE)
    request.state.flash = flash

    response = await call_next(request)

    if flash is not None:
        response.delete_cookie(_FLASH_COOKIE, path="/")
    return response
```

**Issue:** Consider the following chained-403 sequence:

1. Request A hits `/pipeline` → 403 → `RedirectResponse("/", set_cookie("nexo_flash", "msg A"))`.
2. Browser redirects to `/` carrying `nexo_flash=msg A`. Middleware reads it (flash != None), stores in `request.state.flash`, passes on. `/` renders, shows toast A. Response: middleware runs `delete_cookie` → cookie cleared. Correct.
3. But suppose the user clicks a second forbidden link BEFORE (2) completes, or the `/` redirect is intercepted and the browser issues another forbidden request (e.g. HTMX fetch in the dashboard) WHILE still carrying `nexo_flash=msg A`:
   - Request B: `GET /recursos` with cookie `nexo_flash=msg A`.
   - `FlashMiddleware` reads `flash = "msg A"` → `request.state.flash = "msg A"`.
   - `call_next` → handler raises 403.
   - Exception handler (`api/main.py:253-262`) produces `RedirectResponse("/", status_code=302)` with `set_cookie("nexo_flash", "msg B", ...)`.
   - Response bubbles back; `FlashMiddleware` sees `flash is not None` → calls `response.delete_cookie(_FLASH_COOKIE, path="/")`.
   - Response now has TWO `Set-Cookie` headers for `nexo_flash`: one setting `msg B; Max-Age=60`, then one expiring the cookie (`Max-Age=0`). Per RFC 6265 §4.1.2, when a UA receives multiple Set-Cookie for the same name in one response, the last one wins. Result: **message B is lost; user sees no toast**.

Not a security bug (HttpOnly stays intact) but a real UX regression under chained-403 scenarios, and the symptom is silent.

**Fix:** Only delete the cookie if the response did not already re-emit one. Check response headers for an existing `set-cookie: nexo_flash=` entry, or bypass deletion when the handler has populated it. Minimal patch:

```python
async def dispatch(self, request: Request, call_next):
    flash = request.cookies.get(_FLASH_COOKIE)
    request.state.flash = flash

    response = await call_next(request)

    if flash is not None:
        # Don't clobber a newly-set flash cookie (chained-403 scenario).
        already_set = any(
            cookie.split(b"=", 1)[0].strip() == _FLASH_COOKIE.encode()
            for cookie in response.headers.getlist("set-cookie")
            # defensive: Starlette uses MutableHeaders; filter by name
            if b"=" in cookie
        )
        # Fallback: getlist returns strings, not bytes — use str API:
        already_set = any(
            h.lower().startswith(f"{_FLASH_COOKIE}=")
            for h in response.headers.getlist("set-cookie")
        )
        if not already_set:
            response.delete_cookie(_FLASH_COOKIE, path="/")
    return response
```

Or the simpler contract-based fix: have `http_exception_handler_403` set `request.state.flash = None` before returning (a sentinel that tells FlashMiddleware "don't touch the outgoing cookie"). Add a test in `tests/middleware/test_flash.py` covering the chained-403 case.

---

## MEDIUM findings

### MD-01 — `templates/recursos.html` partial edit-gating: inputs and in-drawer actions remain usable for read-only users

**File:** `templates/recursos.html`

- Line 154 (`toggleActive()` — bind to the activo toggle): not wrapped.
- Line 179 (`input[type=number]` for `centro_trabajo` with `@input=...; dirty = true`): not wrapped.
- Line 201-214 (ciclo rows inside the drawer): `input[referencia]`, `input[tiempo_ciclo]` are editable, `removeCiclo` button not wrapped.
- Line 222 (`addCiclo()` — "+ Anadir referencia"): not wrapped.

**Issue:** The per-card Delete máquina (line 164) and the toolbar's Detectar / Add / Guardar / Delete section ARE gated via `{% if can(current_user, "recursos:edit") %}`, but once a read-only user opens the drawer they can mutate the Alpine state freely — CT number, activo flag, ciclo referencias, add new ciclo rows, delete existing ones. They'll never be able to persist (Guardar is hidden), but:

1. The UX is visually broken: they see editable fields and a "+ Anadir referencia" button that does nothing useful.
2. `dirty = true` will show the "Sin guardar" badge (line 12) with no way to save or discard — confusing.
3. If at some later point someone adds a keyboard shortcut or auto-save, the gap widens into a real privilege bug.

**Fix:** Wrap the drawer's mutating controls. Either:

- Wrap each mutating element with `{% if can(current_user, "recursos:edit") %}` (consistent with the rest of the file), OR
- Wrap the entire Alpine `@input` / `@click` handlers so they no-op when the user lacks the permission (server-injected flag: `x-data="recursosPage({{ can(current_user, 'recursos:edit')|tojson }})"`).

Prefer the first (follows existing pattern in this file) for the toggle, CT input, addCiclo button, and removeCiclo button. The reference/tiempo_ciclo inputs should also be `:disabled` when gated to make read-only mode visually obvious.

### MD-02 — `base.html` nav_items separator entry passes empty `icon_paths`, works only by branch order

**File:** `templates/base.html:54, 61-73`

```jinja
{% set nav_items = [
  ...
  ("_sep1",     "",           "",            "", None),
  ...
] %}
{% for key, href, label, icon_paths, permission in nav_items %}
  {% if permission is none or can(current_user, permission) %}
    {% if key.startswith('_sep') %}
      <div class="my-2 mx-3 h-px bg-brand-700"></div>
    {% else %}
      ...
      {% for d in icon_paths.split('||') %}
```

**Issue:** The separator tuple has `icon_paths=""`. The loop body does `icon_paths.split('||')` INSIDE the `_sep`-NOT branch, so today it's safe. But the invariant is implicit: if someone later renames the separator prefix (e.g. to `_divider`) or passes another non-nav entry with `icon_paths=None`, the template will either silently skip rendering or throw `AttributeError: 'NoneType' object has no attribute 'split'`.

**Fix:** Either (a) make the separator's icon_paths explicitly `None` and guard the split (`{% if icon_paths %}{% for d in icon_paths.split('||') %}...{% endfor %}{% endif %}`), or (b) extract the nav_items list to a Python-side constant in `api/deps.py` with proper typing (a `TypedDict` or dataclass), so the schema is enforced at edit time rather than at render time.

Low-risk today; defensive-only. Promoting to MEDIUM because the list is the main source of RBAC-driven sidebar rendering — a silent breakage would be hard to diagnose.

### MD-03 — `http_exception_handler_403` parses `exc.detail` as a string prefix — brittle inter-module contract

**File:** `api/main.py:248-252`

```python
raw = str(exc.detail or "")
perm = raw.replace("Permiso requerido: ", "") if raw.startswith(
    "Permiso requerido: "
) else ""
friendly = _friendly_permission_label(perm) if perm else "esta seccion"
```

**Issue:** The 403 handler relies on the literal prefix string `"Permiso requerido: "` produced by `nexo/services/auth.py:287`. Any refactor of the detail format (e.g. translation, adding a quote, adding a period) silently breaks the user-friendly toast label → falls back to `"esta seccion"` without test failure.

**Fix:** Attach structured metadata to the `HTTPException`. FastAPI supports extra attributes on `HTTPException` via subclassing:

```python
class PermissionDenied(HTTPException):
    def __init__(self, permission: str):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permiso requerido: {permission}",
        )
        self.permission = permission  # structured field for handlers
```

Then in the 403 handler: `perm = getattr(exc, "permission", "")`. Keeps backward compatibility with string detail but adds a typed channel. Add a unit test: `test_403_handler_extracts_permission_from_structured_exception`.

### MD-04 — `ajustes_conexion.html` contains line-number references that will bit-rot

**File:** `templates/ajustes_conexion.html:14, 110, 121`

```jinja
{# ── Conexion SQL Server (extraído verbatim de ajustes.html:74-168) ── #}
...
{# ── Info sistema (extraído de ajustes.html:170-177) ── #}
...
/* Extraído verbatim de ajustes.html:180-207 (ajustesPage()).
```

**Issue:** These comments reference source line ranges in `ajustes.html` that no longer contain that content (Plan 05-04 moved the block here and reduced `ajustes.html` to a hub with 6 cards). The comments are now pointing at code that doesn't exist at those lines. Any future reader who follows the reference will be confused.

**Fix:** Replace line numbers with a git SHA or a prose description of origin: e.g. `{# Extracted from ajustes.html in Plan 05-04 (pre-refactor: the "Conexion SQL Server" card block) #}`. Or drop the reference entirely — the new file owns this code now.

---

## LOW findings

### LO-01 — `ajustes_conexion.html` calls `showToast` with wrong arity

**File:** `templates/ajustes_conexion.html:138`

```javascript
showToast('Configuracion guardada'); await this.testConnection();
```

**Issue:** `base.html:188` defines `window.showToast = function(type, title, msg)`. Calling with one positional argument maps to `type='Configuracion guardada'`, `title=undefined`, `msg=undefined`. Inside `showToast`: `styles[type]` is undefined → falls back to `styles.info`; `icons[type]` is undefined → falls back to `icons.info`; `title` is undefined → the toast shows "undefined" as the title. The Phase 5 plan notes this is a verbatim copy from the pre-existing `ajustes.html`, so the bug predates this phase — but since Phase 5 created this file fresh, it's in scope to fix.

**Fix:** `showToast('info', 'Guardado', 'Configuracion de conexion guardada');`

Worth a sweep across the other templates using 1-arg / 2-arg `showToast` (historial.html, recursos.html, ciclos_calc.html, bbdd.html, plantillas.html — see Grep output, all call `showToast('text')` or `showToast('text', 'error')`). Not a Phase 5 issue but a tech-debt item worth logging for a future plan.

### LO-02 — `PERMISSION_MAP` docstring references stale plan IDs

**File:** `nexo/services/auth.py:229-231`

```python
"aprobaciones:manage": [],  # Plan 04-03 (QUERY-06) — propietario-only
"limites:manage":     [],   # Plan 04-04 (QUERY-02) — propietario-only
"rendimiento:read":   [],   # Plan 04-04 (D-11) — propietario-only
```

**Issue:** `QUERY-06`, `QUERY-02`, and `D-11` are Phase 4 internal identifiers, not stable references. Readers of this file won't know where to look.

**Fix:** Either drop the identifiers and keep only the plan number, or replace with a stable doc reference: `# See docs/AUTH_MODEL.md §Apendice PERMISSION_MAP`. Already referenced at line 198 — can just remove the per-entry comments.

### LO-03 — `_PERMISSION_LABELS` uses unaccented Spanish (`configuracion`, `auditoria`)

**File:** `api/main.py:195-217`

**Issue:** The friendly labels drop Spanish accents (`"la configuracion"` vs `"la configuración"`). This is consistent with the rest of the codebase's UI strings (see `historial.html:35` "Sin ejecuciones", `recursos.html:74` "Seccion", etc.) but the convention isn't documented anywhere. When a future contributor adds a label they may reach for the accented form and create inconsistency.

**Fix:** Add a short note to `docs/BRANDING.md` under a new section "UI text conventions": "All user-facing strings in toasts, labels, and placeholders use unaccented Spanish for compatibility with legacy Jinja templates and to avoid double-encoding issues in HTMX partial responses." Non-blocking.

---

## Confirmation block (what was checked per category)

**Security:**

- `can(None, perm)` → False, no raise. [`auth.py:250-251`] ✓
- Flash cookie: `HttpOnly=True`, `SameSite=lax`, `secure=not settings.debug`, `max_age=60`, `path="/"`. [`main.py:254-262`] ✓
- Flash body emitted via `|tojson` in `base.html:242` → no XSS. ✓
- No hardcoded `role == "propietario"` outside `can()` [verified via Grep in templates/ and api/]. ✓
- `require_permission` 401 (no user) vs 403 (no perm) distinction preserved. [`auth.py:277-289`] ✓
- `StarletteHTTPException` handler delegates non-403 to `fastapi.exception_handlers.http_exception_handler` [`main.py:241-242`] ✓ — 401/404/422/429 contract preserved.
- `_wants_json` correctly identifies JSON callers (`/api/*`, `Accept: application/json`, `HX-Request: true`) before deciding to redirect vs JSON-respond. [`main.py:136-146`] ✓

**Correctness:**

- `can` registered in `templates.env.globals` at import time (not lifespan, not factory). [`deps.py:26-32`] ✓
- `current_user` NOT registered as Jinja global; only injected per-request via `render()`. [`deps.py:52-54`] ✓
- FlashMiddleware reads cookie BEFORE `call_next`, deletes AFTER. [`flash.py:27-34`] ✓ (with HI-01 caveat).
- 403 redirect sets cookie on the `RedirectResponse` (not the original request). [`main.py:253-262`] ✓
- `require_permission` trampolines to `can` — no duplicate intersection logic. [`auth.py:284`] ✓
- `user.departments` eager-loaded in `AuthMiddleware.dispatch` via `_ = list(user.departments)` [middleware/auth.py:163] ✓ — prevents DetachedInstanceError in template `can()` calls.

**Coverage:**

- 8 HTML GETs in `pages.py` guarded (pipeline, recursos, historial, ciclos-calc, operarios, datos, bbdd, capacidad, ajustes, ajustes/conexion) → 10 total. ✓
- `ajustes.html` hub: 6 cards, each `{% if can(current_user, "<perm>") %}`-gated, SMTP card fully removed. ✓
- 11 buttons gated (pipeline:run × 1, informes:delete × 2, recursos:edit × 6, ciclos:edit × 1 — per plan scope). Count matches. See MD-01 for additional ungated mutating controls.
- `_PERMISSION_LABELS` covers all 10 HTML-guarded perms plus extras. ✓
- `/ajustes/conexion` route guarded with `conexion:config` (empty list = propietario-only). ✓

**Quality:**

- No Python `print()` statements introduced (all logging via `logger`). ✓
- No `TODO`/`FIXME` introduced by Phase 5.
- Type annotations on all new function signatures (`can`, `_check`, `_wants_json`, `_friendly_permission_label`, `dispatch`). ✓
- `from __future__ import annotations` at the top of all modified Python files. ✓
- No circular imports introduced (verified import chain: `deps.py` → `auth.py`; `main.py` → `flash.py` → stdlib; `pages.py` → `auth.py`, `deps.py`). ✓
- `render()` name preserved; no `render_html` rename happened despite prompt text (matches plan 05-01 Pitfall 7). Note to orchestrator: the review prompt's scope text says "`render_html` passes flash_message", but the actual function is named `render` — this is the plan's intended outcome.

---

_Reviewed: 2026-04-20_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
