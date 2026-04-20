---
phase: 05-ui-por-roles
type: patterns
created: 2026-04-20
new_files: 3
modified_files: 10
---

# Phase 5: UI por roles — Pattern Map

**Mapped:** 2026-04-20
**Files analyzed:** 13 (3 new + 10 modified)
**Analogs found:** 13/13 (100% coverage — no greenfield patterns)
**Research references:** 05-CONTEXT.md D-01..D-09; 05-RESEARCH.md §Pattern 1..5, §Pitfall 1..7, §Catalogue

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| **NEW** `nexo/middleware/flash.py` | middleware | request-response | `api/middleware/audit.py` + `nexo/middleware/query_timing.py` | exact (BaseHTTPMiddleware, read-request-modify-response) |
| **NEW** `templates/ajustes_conexion.html` | template (sub-page) | request-response | `templates/ajustes_limites.html` | exact (hub sub-page, propietario-only, `ajustesPage()` Alpine component already extractable from ajustes.html:180-207) |
| **NEW** toast block in `templates/base.html` (§D-08) | template partial | event-driven | `templates/base.html:160-233` (`showToast` system) | exact — REUSE existing `window.showToast`, no new component |
| **MOD** `nexo/services/auth.py` | service / RBAC core | pure function | `nexo/services/auth.py:235-271` (`require_permission` itself) | self — factor `can()` OUT of `require_permission` |
| **MOD** `api/main.py` (HTTPException handler + FlashMiddleware wiring) | app factory | exception handling | `api/main.py:126-167` (`global_exception_handler` + `_wants_json`) | exact — add sibling handler, reuse heuristic |
| **MOD** `api/deps.py` (Jinja globals + flash ctx) | deps / templates | module-init | `api/deps.py:17-22` (existing `templates.env.globals.update(...)`) | exact — append `can=_can` to the dict; extend `render()` ctx dict |
| **MOD** `templates/base.html:48-79` (nav_items refactor) | template | conditional render | `templates/base.html:48-79` itself | self — change 5th tuple slot from `visible_to` to `permission`; swap filter |
| **MOD** `templates/base.html:82-94` (solicitudes badge) | template | conditional render | same file (role check already there) | self — refactor `role == 'propietario'` → `can(...,"aprobaciones:manage")` |
| **MOD** `templates/ajustes.html` (hub refactor) | template | conditional render | `templates/ajustes.html:34-71` (5 cards with `role == 'propietario'`) | self — remove SMTP (never existed), remove conexion inline block 74-168, add conexion card, refactor 5 `role ==` to `can(...)` |
| **MOD** `templates/pipeline.html:185` (wrap Ejecutar) | template | conditional render | none yet in repo — first `{% if can() %}` wrap | new pattern (Phase 5 seeds it) |
| **MOD** `templates/historial.html:107,112` (wrap Generar + Borrar) | template | conditional render | same pattern as pipeline.html:185 wrap | repeat |
| **MOD** `templates/recursos.html` (6 button wraps) | template | conditional render | same pattern | repeat |
| **MOD** `templates/ciclos_calc.html:208` (wrap Guardar) | template | conditional render | same pattern | repeat |
| **MOD** `api/routers/pages.py` (add `require_permission` to HTML GETs) | router | request-response | `api/routers/pages.py:106-111` (`ajustes_page` with `dependencies=[...]`) | exact — copy the pattern for `/bbdd`, `/pipeline`, etc. |

---

## Pattern Assignments

### `nexo/middleware/flash.py` (NEW, middleware, request-response)

**Analog:** `api/middleware/audit.py:88-151` (preferred — shows the read-state-then-mutate-response pattern) + `nexo/middleware/query_timing.py:83-197` (shows the `BaseHTTPMiddleware` + `call_next` async pattern).

**Imports pattern** (copy from `audit.py:20-30` and `query_timing.py:45-58`, trimmed since flash has no DB):
```python
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
```
No DB, no logging necessary — flash is stateless cookie hopping. Zero new imports beyond Starlette.

**Core middleware pattern** (structural template from `audit.py:103-151`; simplified because no body-parse, no DB write):
```python
class FlashMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        flash = request.cookies.get(_FLASH_COOKIE)
        request.state.flash = flash                 # None si no hay cookie

        response = await call_next(request)

        if flash is not None:
            response.delete_cookie(_FLASH_COOKIE, path="/")
        return response
```

**Why this analog:** `AuditMiddleware.dispatch` is the canonical "read state pre-call, mutate post-response" pattern in this codebase (line 103-151). `QueryTimingMiddleware.dispatch` (line 83-197) shows the same shape. Both use `BaseHTTPMiddleware` from `starlette.middleware.base`. Flash is strictly simpler — no body parsing, no DB persistence, no error recovery — but the signature and flow are identical.

**Anti-pattern to avoid:** Do **NOT** fold this into `AuthMiddleware` (api/middleware/auth.py). Research §Pattern 4 explicitly rejects that: flash is orthogonal to auth (anonymous users can also carry a flash — e.g., post-login redirect with message). Keep single-responsibility. Do NOT write to DB. Do NOT add `logger` — flash failure is silent-and-harmless.

---

### `templates/ajustes_conexion.html` (NEW, template, request-response)

**Analog:** `templates/ajustes_limites.html:1-10` (frontmatter + `{% extends "base.html" %}` + `{% block page_title %}Ajustes · Límites{% endblock %}`) — the hub sub-page naming convention post Plan 04-04.

**Template scaffold** (copy from `ajustes_limites.html:1-10`):
```jinja
{% extends "base.html" %}

{% block title %}Conexión SQL Server — {{ app_name }}{% endblock %}
{% block page_title %}Ajustes · Conexión{% endblock %}

{% block content %}
<div x-data="ajustesConexionPage()" x-init="loadConfig()" class="max-w-2xl mx-auto space-y-6">
  {# CUT of ajustes.html:74-168 — card "Conexión SQL Server" + info sistema #}
</div>

<script>
  /* Extracted verbatim from ajustes.html:180-207 ajustesPage() —
     rename to ajustesConexionPage() to avoid collision if both coexist
     during the refactor wave. */
</script>
{% endblock %}
```

**Body source:** cut `templates/ajustes.html:74-168` (the entire "Conexion SQL Server" card + "Info sistema" footer links) and paste into `{% block content %}`. The Alpine component (`ajustes.html:180-207`) moves with it, renamed.

**Why this analog:** `ajustes_limites.html` is the most recent hub sub-page (Plan 04-04), same propietario-only gating, same `max-w-*xl mx-auto space-y-*` container, same `{% extends "base.html" %}` pattern. `ajustes_usuarios.html` / `ajustes_auditoria.html` / `ajustes_solicitudes.html` / `ajustes_rendimiento.html` are all interchangeable analogs — all 5 sub-pages follow the same shape.

**Anti-pattern to avoid:** Do NOT duplicate the Alpine component into a separate JS file — research §Open Q4 confirms the backend endpoints (`/api/conexion/config`, `/api/conexion/status`) are unchanged. Pure template split. Do NOT create an `ajustes_smtp.html` — D-04 explicitly forbids it.

---

### Toast render block in `templates/base.html` (NEW but inline, template partial, event-driven)

**Analog:** `templates/base.html:182-223` (the existing `window.showToast` definition) + `templates/base.html:160-180` (the Alpine `toastSystem()` host container).

**Pattern to follow** (insert near end of `<body>`, after the existing `toastSystem` script block at line 233, before `</body>` line 236):
```jinja
{% if flash_message %}
<script>
  document.addEventListener('DOMContentLoaded', function () {
    if (typeof showToast === 'function') {
      showToast('info', 'Aviso', {{ flash_message|tojson }});
    }
  });
</script>
{% endif %}
```

**Why this analog:** `window.showToast` (base.html:188-223) already does:
- 5s auto-dismiss (line 214).
- Type-based styling — `info` uses `bg-brand-700 text-white` (line 195).
- Queue via `window._toasts` (line 212) so concurrent toasts don't collide.
- XSS-safe message rendering (the Alpine template uses `x-text`, line 176).

Reusing it means **zero new JS and zero new CSS**. The `|tojson` Jinja filter prevents injection from the flash message string.

**Anti-pattern to avoid:** Do NOT build a new toast CSS/JS/Alpine component. Do NOT call `showToast` synchronously at the top of the body — `showToast` is defined later in the file; must wait for `DOMContentLoaded`. Do NOT inline the message as text (XSS risk) — always `|tojson`.

---

### `nexo/services/auth.py` (MODIFIED, service, pure function)

**Analog:** `nexo/services/auth.py:235-271` (`require_permission` itself — the function being refactored).

**Region to touch:**
- **INSERT** a new `can()` helper at **line 234** (right after the PERMISSION_MAP dict closes, right before `def require_permission`). The function is a pure extraction of the intersection logic currently living in lines 259-267.
- **REFACTOR** the body of `require_permission._check` (lines 252-268) to trampoline over `can()`. The outer factory (`def require_permission(permission: str):` at line 235) and its docstring (236-250) stay verbatim. Only lines 259-267 change: replace the inline propietario-bypass + intersection with `if not can(user, permission): raise HTTPException(...)`.

**Pattern to add** (new `can()` before PERMISSION_MAP is wrong — research §Pattern 1 Example A says "justo antes de PERMISSION_MAP" but the cleaner place is right AFTER PERMISSION_MAP at line 234, since `can()` reads `PERMISSION_MAP`):
```python
def can(user: NexoUser | None, permission: str) -> bool:
    """True si ``user`` tiene ``permission``. Puro, sin side effects.

    Reglas (idénticas a ``require_permission`` — esta función ES la fuente
    de verdad; ``require_permission`` trampolina sobre ella):

    - ``user is None`` → False.
    - ``user.role == 'propietario'`` → True (bypass).
    - Intersecta ``{d.code for d in user.departments}`` con
      ``PERMISSION_MAP[permission]``. True si hay intersección.

    Safe desde templates (Jinja global, D-03) y tests — sin async,
    sin HTTPException.
    """
    if user is None:
        return False
    if user.role == "propietario":
        return True
    allowed = PERMISSION_MAP.get(permission, [])
    user_depts = {d.code for d in user.departments}
    return bool(user_depts.intersection(allowed))
```

**`require_permission` refactor** (replace lines 259-267 with a single `can()` call):
```python
async def _check(request: Request) -> NexoUser:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    if not can(user, permission):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permiso requerido: {permission}",
        )
    return user
```

**Why this analog:** The function being refactored IS the closest analog — it contains the exact algorithm to extract. Research §Pattern 1 confirms factoring (not try/except trampoline) is correct because `require_permission` is async + raises `HTTPException`, neither of which Jinja templates can consume directly.

**Anti-pattern to avoid:** Do **NOT** write a second `has_permission(user, perm)` helper. There must be exactly one intersection-and-bypass algorithm. Do NOT change `require_permission`'s public signature (factory returning async `_check`) — all the `Depends(require_permission("x"))` call sites across 14+ routers depend on it. Do NOT move `PERMISSION_MAP` — it's the seed of truth and the import surface of `nexo.services.auth`.

---

### `api/main.py` (MODIFIED, app factory, exception handling + middleware wiring)

**Analog:** `api/main.py:126-167` (existing `global_exception_handler` + `_wants_json` helper).

**Region to touch:**
- **INSERT NEW** `@app.exception_handler(StarletteHTTPException)` at approximately **line 168** (immediately after `global_exception_handler` closes at line 167, before the rate_limit block at line 170). This is a SIBLING handler — NOT a replacement.
- **INSERT** new imports at the top (near lines 11-27):
  ```python
  from starlette.exceptions import HTTPException as StarletteHTTPException
  from fastapi.exception_handlers import http_exception_handler as _default_http_handler
  from fastapi.responses import RedirectResponse  # already: JSONResponse, HTMLResponse
  from nexo.middleware.flash import FlashMiddleware
  ```
- **INSERT NEW LINE** in the middleware stack around **line 202** (between `add_middleware(AuditMiddleware)` and `add_middleware(AuthMiddleware)`):
  ```python
  app.add_middleware(FlashMiddleware)        # entre Audit y Auth
  ```
  Final stack (lines 200-203):
  ```python
  app.add_middleware(QueryTimingMiddleware)  # innermost
  app.add_middleware(AuditMiddleware)
  app.add_middleware(FlashMiddleware)        # NEW
  app.add_middleware(AuthMiddleware)         # outermost
  ```

**Pattern to add — HTTPException handler** (follows shape of `global_exception_handler` at 139-167):
```python
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Negocia Accept para 403: HTML → redirect + flash; resto → default."""
    if exc.status_code != 403:
        return await _default_http_handler(request, exc)
    if _wants_json(request):                            # REUSE existing helper
        return await _default_http_handler(request, exc)

    perm = str(exc.detail).replace("Permiso requerido: ", "")
    friendly = _friendly_permission_label(perm)
    response = RedirectResponse("/", status_code=302)
    response.set_cookie(
        "nexo_flash",
        f"No tienes permiso para acceder a {friendly}",
        max_age=60,
        httponly=True,
        secure=not settings.debug,     # Pitfall 2: dev HTTP → False
        samesite="lax",
        path="/",
    )
    return response
```

**Why this analog:** `global_exception_handler` (lines 139-167) already uses `_wants_json` for Accept-negotiation. The new 403 handler uses the SAME heuristic for consistency. The imports at lines 11-27 and the `@app.exception_handler(Exception)` decorator pattern at line 139 are the exact precedent.

**Key finding (research §Pattern 3):** FastAPI's `@app.exception_handler(Exception)` does NOT catch `HTTPException` — they're separate handler registries. Adding a `StarletteHTTPException` handler does NOT regress the existing 500 handler. They coexist.

**Anti-pattern to avoid:**
- Do NOT register against `fastapi.HTTPException` — use `starlette.exceptions.HTTPException` (the parent class). FastAPI's is a subclass.
- Do NOT re-implement `_wants_json` — reuse the one at line 126.
- Do NOT place FlashMiddleware outermost — it must run AFTER AuthMiddleware for `request.state.user` to be available (though for Pitfall 3 it doesn't strictly need user, it's more consistent with the LIFO comment block at lines 181-199).

---

### `api/deps.py` (MODIFIED, deps / templates, module-init)

**Analog:** `api/deps.py:17-22` (existing `templates.env.globals.update(app_name=..., company_name=..., logo_path=..., ecs_logo_path=...)`).

**Region to touch:**
- **ADD** import at top (around line 11): `from nexo.services.auth import can as _can`.
- **APPEND** to the existing `templates.env.globals.update(...)` call at lines 17-22:
  ```python
  templates.env.globals.update(
      app_name=settings.app_name,
      company_name=settings.company_name,
      logo_path=settings.nexo_logo_path,
      ecs_logo_path=settings.nexo_ecs_logo_path,
      can=_can,                        # NEW — D-03 / D-09
  )
  ```
- **EXTEND** `render()` context dict at lines 42-45 to include `flash_message`:
  ```python
  ctx: dict[str, Any] = {
      "request": request,
      "current_user": getattr(request.state, "user", None),
      "flash_message": getattr(request.state, "flash", None),   # NEW
  }
  ```

**Why this analog:** `api/deps.py:17-22` is the canonical precedent for Jinja globals in this codebase — the brand helpers (app_name, company_name, logo_path, ecs_logo_path) are registered exactly the same way. `can` follows the same shape: a pure callable needed everywhere without threading it through every `render()` call.

**Anti-pattern to avoid:**
- Do **NOT** register `current_user` as a Jinja global (Pitfall 3). `current_user` is request-scoped; `env.globals` is process-scoped. Mixing them races between concurrent requests.
- Do **NOT** rename `render()` to `render_html` (Pitfall 7) — 7 routers import `render`.
- Do **NOT** set `can` in the lifespan (Pitfall 5) — must run at import time so the Environment has it before any render.

---

### `templates/base.html:48-79` (MODIFIED, template, conditional render — nav_items refactor)

**Analog:** same file, same region — self-refactor. Preserve the `nav_items` 5-tuple unpack structure (research §Pattern 5).

**Region to touch (D-01):**
- **Lines 48-60:** replace the 5th column of each tuple. Currently `visible_to` is `"*"` or `"propietario"`. After: `permission` is a permission string (e.g., `"pipeline:read"`) or `None` for "always visible".
- **Line 61-62:** change unpack + filter. Current: `{% for key, href, label, icon_paths, visible_to in nav_items %}{% if visible_to == "*" or (current_user and current_user.role == visible_to) %}`. After: `{% for key, href, label, icon_paths, permission in nav_items %}{% if permission is none or can(current_user, permission) %}`.

**Pattern to follow** (exact permission strings from `nexo/services/auth.py:200-232`):
```jinja
{% set nav_items = [
  ("dashboard",    "/",            "Centro Mando",    "M3.75 3v...",  None),
  ("datos",        "/datos",       "Datos",           "M4 16v...",    "datos:read"),
  ("pipeline",     "/pipeline",    "Analisis",        "M4 4v...",     "pipeline:read"),
  ("historial",    "/historial",   "Historial",       "M12 8v...",    "historial:read"),
  ("capacidad",    "/capacidad",   "Capacidad",       "M9 19V...",    "capacidad:read"),
  ("_sep1",        "",             "",                "",             None),
  ("recursos",     "/recursos",    "Recursos",        "M20 7l...",    "recursos:read"),
  ("ciclos_calc",  "/ciclos-calc", "Calcular Ciclos", "M9 7h...",     "ciclos:read"),
  ("operarios",    "/operarios",   "Operarios",       "M17 20h...",   "operarios:read"),
  ("bbdd",         "/bbdd",        "BBDD",            "M4 7v...",     "bbdd:read"),
  ("ajustes",      "/ajustes",     "Ajustes",         "M10.325...",   "ajustes:manage"),
] %}
{% for key, href, label, icon_paths, permission in nav_items %}
  {% if permission is none or can(current_user, permission) %}
    {# ... render item exactly as lines 63-77 ... #}
  {% endif %}
{% endfor %}
```

**Pitfall avoided:** tuple arity stays at 5 — adding a 6th column breaks the unpack. Replace in place.

**Anti-pattern to avoid:** Do NOT read `current_user.role` anywhere in nav logic — always go through `can()`. Do NOT maintain a second list of "propietario-only items" (D-05 says `/ajustes` → `"ajustes:manage"` which has `[]` in PERMISSION_MAP → propietario via bypass; that's the coherent path).

---

### `templates/base.html:82-94` (MODIFIED, template, solicitudes badge refactor)

**Analog:** same block — self-refactor from `role == 'propietario'` to `can()`.

**Region to touch:**
- **Line 82:** `{% if current_user and current_user.role == 'propietario' %}` → `{% if can(current_user, "aprobaciones:manage") %}`.

**Why:** `PERMISSION_MAP["aprobaciones:manage"] == []` (line 229 of auth.py) means propietario-only today; the `can()` call is functionally equivalent. If Mark-IV ever opens it to a department, the template is already correct. Research §Pattern 5 last paragraph.

**Anti-pattern to avoid:** Do NOT leave mixed gates (some `role == 'propietario'`, some `can(...)`); Phase 5 commits to `can()` everywhere.

---

### `templates/ajustes.html` (MODIFIED, template, hub refactor)

**Analog:** same file — surgery. Target regions per D-04/D-05/D-06:

**Region to touch:**
- **Line 34 + line 71:** remove `{% if current_user and current_user.role == 'propietario' %}` wrapping and replace with PER-CARD `{% if can(current_user, "<perm>") %}` gating. Permissions: `usuarios:manage`, `auditoria:read`, `aprobaciones:manage`, `limites:manage`, `rendimiento:read` — all have `[]` in PERMISSION_MAP so behavior is equivalent. Refactor for DRY + future-readiness (research §Open Q5).
- **Lines 74-168:** DELETE entire "Conexion SQL Server" card block — it moves to `ajustes_conexion.html`. Replace with a new card link to `/ajustes/conexion` following the pattern of existing cards at lines 10-21 (usuarios card) — same `<a href=... class="group bg-white rounded-2xl ..."` shell.
- **Lines 170-177:** optionally keep or move "Info sistema" to `ajustes_conexion.html` (cleaner: belongs with the BBDD/docs context).
- **Lines 180-207:** DELETE the `ajustesPage()` Alpine component — it moves to `ajustes_conexion.html`.
- **SMTP**: was NEVER in this file. No removal needed. D-04 says do NOT create an SMTP card.

**Pattern for the new "Conexión" card** (clone from lines 10-21):
```jinja
{% if can(current_user, "conexion:config") %}
<a href="/ajustes/conexion"
   class="group bg-white rounded-2xl border border-surface-200 shadow-sm p-5 hover:border-brand-400 hover:shadow-md transition-all">
  <div class="flex items-start justify-between">
    <div>
      <h3 class="font-bold text-gray-800 group-hover:text-brand-700 transition-colors">Conexión SQL Server</h3>
      <p class="text-xs text-gray-500 mt-1">Credenciales, puerto, driver ODBC. Probar conexión en vivo.</p>
    </div>
    <svg class="w-5 h-5 text-gray-300 group-hover:text-brand-500 ..."> ... </svg>
  </div>
</a>
{% endif %}
```

**Anti-pattern to avoid:** Do NOT leave the SMTP dream card commented or stubbed. Do NOT cut the Alpine component into a separate JS file — it moves verbatim with the template body. Do NOT leave an orphaned `{% if current_user and current_user.role == 'propietario' %}` wrapping the whole hub — refactor to per-card `can()`.

---

### `templates/pipeline.html:185` (MODIFIED, wrap "Ejecutar")

**Analog:** none yet in repo — this plan seeds the `{% if can() %}` wrap convention.

**Region to touch:**
- **Line 184-185:** the "Atras" button at 184 stays unwrapped (always visible). The "Ejecutar" button at 185 gets wrapped.

**Pattern to follow** (from research §Pattern 5 Example C):
```jinja
      <button @click="step = source === 'db' ? 3 : 2" class="btn-ghost btn-sm">Atras</button>
      {% if can(current_user, "pipeline:run") %}
      <button @click="ejecutar()" class="btn-success !px-5 !py-3" :disabled="selectedModules.length === 0">
        <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
          <path d="M8 5v14l11-7z"/>
        </svg>
      </button>
      {% endif %}
```

**Why:** `pipeline:run` = `["ingenieria", "produccion"]` (auth.py:202). A rrhh user hitting `/pipeline` (after Pitfall 4 fix) won't even see the button.

**Anti-pattern to avoid:** Do NOT wrap the confirmation modal buttons (pipeline.html:349 `confirmRun`, pipeline.html:371 `requestApproval`) — research §Catalogue last paragraph: those are already gated by the backend via `require_permission` + preflight. Wrapping the entry button at 185 gates the entire flow.

---

### `templates/historial.html:107,112` (MODIFIED, wrap "Generar PDFs" + "Borrar ejecución")

**Analog:** same `{% if can() %}` wrap as pipeline.html:185.

**Region to touch:**
- **Line 107:** "Generar PDFs" (regenerar) button — permission `informes:delete` (research §Catalogue #3; `api/routers/historial.py:167` also uses this).
- **Line 112:** "Borrar ejecución" button (trash icon) — permission `informes:delete` (§Catalogue #2).

**Pattern to follow:**
```jinja
{% if can(current_user, "informes:delete") %}
<button @click="regenerar()" class="btn-primary btn-sm" :disabled="regenerando">
  ...
  Generar PDFs
</button>
<button @click="borrar()" class="w-8 h-8 rounded-lg flex items-center justify-center text-gray-300 hover:text-red-500 hover:bg-red-50 transition-colors" title="Borrar ejecucion">
  ...
</button>
{% endif %}
```

Can share the same `{% if %}` since both have the same permission.

**Anti-pattern to avoid:** Do NOT use `informes:read` for "Generar PDFs" — regeneration destroys existing files. `informes:delete` is correct per §Catalogue #3.

---

### `templates/recursos.html` (MODIFIED, wrap 6 actions)

**Analog:** same `{% if can() %}` wrap pattern. Permission: `recursos:edit` (auth.py:205 = `["ingenieria"]`).

**Region to touch (§Catalogue #4-9):**
- **Line 18:** "Detectar" button.
- **Line 35:** "Add" dropdown trigger (revealing lines 41 + 45).
- **Lines 41 + 45:** "Añadir máquina" / "Añadir sección" dropdown items — inside the dropdown, can rely on the parent wrap at line 34 (`<div class="relative" x-data="{ addOpen: false }">`) being wrapped.
- **Line 52:** "Guardar todo" button.
- **Line 76:** "Eliminar sección" (X icon on section header).
- **Line 157:** "Eliminar máquina" (trash icon in drawer).

**Pattern to follow** (wrap each standalone button, or wrap the entire toolbar row if buttons are contiguous):
```jinja
{% if can(current_user, "recursos:edit") %}
<button @click="detectar()" ... >Detectar</button>
<div class="relative" x-data="{ addOpen: false }" @click.away="addOpen = false">
  <button @click="addOpen = !addOpen" ... >Añadir</button>
  <div x-show="addOpen" ...>
    <button @click="showAddMachine = true; ...">Añadir máquina</button>
    <button @click="showAddSection = true; ...">Añadir sección</button>
  </div>
</div>
<button @click="guardarTodo()" ... >Guardar todo</button>
{% endif %}
```

**Anti-pattern to avoid:** Do NOT wrap "organizing" toggle (line 25), `openMachine` (line 92), or `moveMachine` (line 120) — those are UI/navigation, not destructive. Do NOT create one big `{% if %}` wrapping unrelated sections of the template — keep wraps tight around the sensitive button clusters.

---

### `templates/ciclos_calc.html:208` (MODIFIED, wrap "Guardar")

**Analog:** same pattern. Permission: `ciclos:edit` (auth.py:207 = `["ingenieria"]`).

**Region to touch:**
- **Line 207-215:** wrap the `<div class="shrink-0" @click.stop>` container (or just the `<button @click="openSaveDialog(r)">` at 208).

**Pattern:**
```jinja
{% if can(current_user, "ciclos:edit") %}
  <div class="shrink-0" @click.stop>
    <button @click="openSaveDialog(r)" ... >
      <span x-text="r._applied ? 'Actualizado' : ...">...</span>
    </button>
  </div>
{% endif %}
```

**Anti-pattern to avoid:** Do NOT wrap `calcular()` (line 38) or `exportCSV()` (line 135) — those are read-only / export-only. Do NOT wrap the modal buttons (lines 366-367) — once the modal opens, the backend persists and enforces.

---

### `api/routers/pages.py` (MODIFIED, router, add `require_permission` to HTML GETs)

**Analog:** `api/routers/pages.py:106-111` (`ajustes_page` with `dependencies=[Depends(require_permission("ajustes:manage"))]`).

**Region to touch (Pitfall 4 hardening — 8 routes need dependency):**
- **Line 29** (`pipeline_page`) → add `dependencies=[Depends(require_permission("pipeline:read"))]`.
- **Line 46** (`recursos_page`) → `"recursos:read"`.
- **Line 58** (`historial_page`) → `"historial:read"`.
- **Line 69** (`ciclos_calc_page`) → `"ciclos:read"`.
- **Line 80** (`operarios_page`) → `"operarios:read"`.
- **Line 85** (`datos_page`) → `"datos:read"`.
- **Line 96** (`bbdd_page`) → `"bbdd:read"`.
- **Line 101** (`capacidad_page`) → `"capacidad:read"`.
- **NEW route** `ajustes_conexion_page` — follow `ajustes_page` pattern exactly (line 106-111), using permission `"conexion:config"`:

**Pattern to follow** (clone from lines 106-111):
```python
@router.get(
    "/pipeline",
    dependencies=[Depends(require_permission("pipeline:read"))],
)
def pipeline_page(request: Request, db: Session = Depends(get_db)):
    extra = _common_extra("pipeline")
    ...
```

**NEW route:**
```python
@router.get(
    "/ajustes/conexion",
    dependencies=[Depends(require_permission("conexion:config"))],
)
def ajustes_conexion_page(request: Request):
    return render("ajustes_conexion.html", request, _common_extra("ajustes_conexion"))
```

**Why this analog:** `ajustes_page` at line 106-111 is the only existing HTML GET with `require_permission` — it's the exact shape for the rest. The decorator syntax is the idiomatic way in this codebase.

**Anti-pattern to avoid:** Do NOT gate the `/` (index) handler at line 24 — the dashboard is always-visible per `nav_items` (permission `None`). Do NOT use `conexion:read` for `/ajustes/conexion` — research §Open Q4 confirms the form PUTs to `/api/conexion/config` which requires `conexion:config`; HTML gate must match backend gate.

---

## Shared Patterns

### Permission check (`can()`)
**Source:** `nexo/services/auth.py` (NEW function post-refactor, approx line 234).
**Apply to:** every template wrap, the `require_permission` dependency, and anywhere in Python that needs a boolean permission check.
**Signature:** `can(user: NexoUser | None, permission: str) -> bool`
**Contract:** pure, synchronous, no side effects, safe from templates and tests.
**Precedent:** the body at `nexo/services/auth.py:259-267` (currently inline in `_check`).

### Accept-header negotiation (`_wants_json`)
**Source:** `api/main.py:126-136` AND `api/middleware/auth.py:70-80` (duplicated pattern).
**Apply to:** the new 403 handler in `api/main.py` — REUSE the main.py one, do not re-import the middleware one.
**Pattern:**
```python
def _wants_json(request: Request) -> bool:
    if request.url.path.startswith("/api/"): return True
    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept: return True
    if request.headers.get("hx-request") == "true": return True
    return False
```
**Anti-duplication flag:** the helper exists TWICE (main.py + middleware/auth.py). Phase 5 does NOT triplicate it; it consumes main.py:126 from within main.py.

### Jinja globals pattern
**Source:** `api/deps.py:17-22`.
**Apply to:** registering `can` as a template-global callable.
**Pattern:** append to the existing `templates.env.globals.update(...)` call. No new `update()` call; keep one.

### BaseHTTPMiddleware + `dispatch`
**Source:** `api/middleware/audit.py:88-151` (preferred analog), `api/middleware/auth.py:95-170`, `nexo/middleware/query_timing.py:83-197`.
**Apply to:** `nexo/middleware/flash.py`.
**Pattern:**
```python
class MyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # (1) read from request.state / request.cookies
        # (2) response = await call_next(request)
        # (3) mutate response (set_cookie / delete_cookie / headers)
        # (4) return response
```
All three existing middlewares use `from starlette.middleware.base import BaseHTTPMiddleware` + `from starlette.requests import Request`.

### `render()` helper (api/deps.py:25-50)
**Source:** `api/deps.py:25-50`.
**Apply to:** every HTML GET in `pages.py` and the new `ajustes_conexion_page`. Do NOT rename to `render_html` (Pitfall 7).
**Extension in Phase 5:** add `"flash_message": getattr(request.state, "flash", None)` to the context dict at line 42.

### Hub sub-page template shell
**Source:** `templates/ajustes_limites.html:1-10` (most recent).
**Apply to:** `templates/ajustes_conexion.html`.
**Pattern:** `{% extends "base.html" %}` + `{% block title %}`, `{% block page_title %}`, `{% block content %}` + `max-w-*xl mx-auto space-y-*` container.

### Toast system (reuse, not rebuild)
**Source:** `templates/base.html:160-233` (`showToast` function + `toastSystem` Alpine host).
**Apply to:** flash rendering (NEW block near end of base.html).
**Anti-duplication flag:** do NOT introduce a second toast component. The existing one supports 5s auto-dismiss, queue, 6 styles including `info`.

### Propietario-only permission via empty list
**Source:** `nexo/services/auth.py:221, 226-231` (empty lists in PERMISSION_MAP).
**Apply to:** any new permission that should be propietario-only (e.g., the research example `conexion:config` already uses this at line 221).
**Pattern:** set `PERMISSION_MAP["new:perm"] = []`. No code change; the `can()` bypass at role == 'propietario' handles it.

---

## Reuse Register

Helpers/functions/patterns the planner MUST wire-up rather than reinvent:

| # | Reuse | Location | Used by |
|---|-------|----------|---------|
| 1 | `can(user, permission)` — the NEW function Phase 5 adds | `nexo/services/auth.py:~234` (post-refactor) | `require_permission`; every template `{% if can(...) %}`; tests |
| 2 | `PERMISSION_MAP` dict | `nexo/services/auth.py:200-232` | `can()` reads it; no other mutations |
| 3 | `require_permission(perm)` factory | `nexo/services/auth.py:235-271` | Added to `pages.py` routes (Pitfall 4); already used by 14+ routers |
| 4 | `_wants_json(request)` heuristic | `api/main.py:126-136` | New 403 handler in main.py consumes this local helper |
| 5 | `_wants_json(request)` duplicate | `api/middleware/auth.py:70-80` | DO NOT add a 3rd copy; this already exists |
| 6 | `templates.env.globals.update(...)` dict | `api/deps.py:17-22` | Append `can=_can` here |
| 7 | `render(template, request, extra, *, status_code=200)` helper | `api/deps.py:25-50` | New `ajustes_conexion_page` route; unchanged signature |
| 8 | `render()` context dict | `api/deps.py:42-45` | Add `flash_message` key |
| 9 | `BaseHTTPMiddleware` + async `dispatch` pattern | `api/middleware/audit.py:88-151` (clearest), `auth.py:95-170`, `query_timing.py:83-197` | `nexo/middleware/flash.py` copies the shape |
| 10 | `window.showToast(type, title, msg)` | `templates/base.html:188-223` | Flash toast render block at end of base.html |
| 11 | `toastSystem()` Alpine host | `templates/base.html:160-180` + `225-232` | Already present; no changes needed |
| 12 | `RedirectResponse("/", status_code=302)` + `.set_cookie(...)` | `fastapi.responses.RedirectResponse` (pattern in `api/middleware/auth.py:92`) | New 403 HTML handler in main.py |
| 13 | `@app.exception_handler(Exception)` registration pattern | `api/main.py:139` | Sibling handler `@app.exception_handler(StarletteHTTPException)` at approx line 168 |
| 14 | `fastapi.exception_handlers.http_exception_handler` default | `fastapi.exception_handlers` | Delegation target for non-403 and JSON-wanting 403 |
| 15 | Hub sub-page template shell (extends + blocks) | `templates/ajustes_limites.html:1-10` | `templates/ajustes_conexion.html` |
| 16 | Hub card markup (a.group.bg-white.rounded-2xl) | `templates/ajustes.html:10-21` | New "Conexión" card + refactored 5 existing cards |
| 17 | Alpine `ajustesPage()` component — cfg/testConnection/guardar | `templates/ajustes.html:180-207` | Moves verbatim to `ajustes_conexion.html` (renamed `ajustesConexionPage()`) |
| 18 | `ajustes_page` route with `dependencies=[Depends(require_permission(...))]` | `api/routers/pages.py:106-111` | Template for all 8 new/modified routes in `pages.py` |
| 19 | `settings.debug` flag for dev vs prod | `api/config.settings.debug` | `secure=not settings.debug` in flash cookie |
| 20 | `|tojson` Jinja filter for XSS-safe interpolation | built-in Jinja2 | `{{ flash_message|tojson }}` in toast render |

---

## No Analog Found

Zero files in this phase lack an analog. Everything composes over existing Phase 1-4 primitives.

**Gap flag:** None. The research explicitly notes (§Key insight) "Phase 5 compone sobre Phase 2 (PERMISSION_MAP + require_permission) y Phase 4 (render helper + toast system). No inventa infraestructura nueva."

---

## Metadata

**Analog search scope:**
- `nexo/services/` (auth service + helpers)
- `api/middleware/` (audit.py, auth.py)
- `nexo/middleware/` (query_timing.py)
- `api/routers/` (pages.py, limites.py, conexion.py — 23 routers total)
- `templates/` (base.html, ajustes*.html, pipeline.html, historial.html, recursos.html, ciclos_calc.html)
- `api/main.py` (exception handler + middleware stack)
- `api/deps.py` (Jinja templates + render helper)

**Files scanned:** ~15 direct reads + 6 targeted greps for button catalogue.

**Pattern extraction date:** 2026-04-20

**Downstream consumer:** `gsd-planner` for Phase 5 plans. Each plan should reference the exact file:line analog above when authoring the plan's "Actions" section.
