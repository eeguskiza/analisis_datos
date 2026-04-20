---
phase: 05-ui-por-roles
type: research
mode: implementation
created: 2026-04-20
references: [CONTEXT.md:D-01..D-09, REQUIREMENTS.md:UIROL-01..05, CLAUDE.md]
---

# Phase 5: UI por roles — Research

**Researched:** 2026-04-20
**Domain:** Retrofit de permission-aware rendering sobre FastAPI + Jinja2 + HTMX + Alpine
**Confidence:** HIGH

## Executive Summary

- **`can(user, perm) -> bool`** se debe extraer como helper público en `nexo/services/auth.py` y `require_permission` refactoriza para trampolinar sobre él. Factoring (opción a) es el shape correcto: simple, sin async, sin HTTPException, sin código muerto. Evita duplicación de la intersección y mantiene una única fuente de verdad.
- **`templates.env.globals`** ya se usa en `api/deps.py:17-22` para registrar `app_name`, `company_name`, `logo_path`, `ecs_logo_path` — precedente exacto. La registración de `can` y `current_user`-fallback sigue el mismo patrón; es idempotente y se ejecuta a import-time (antes del lifespan), por lo que cualquier template rendered (incluso los de tests con TestClient) lo verá.
- **El `global_exception_handler` de NAMING-07 NO captura `HTTPException`**. `@app.exception_handler(Exception)` es ignorado por FastAPI cuando la excepción es `HTTPException` (tiene su propio handler por defecto que devuelve JSON). Para D-07 se añade un handler SEPARADO `@app.exception_handler(StarletteHTTPException)` que negocia Accept + redirige en 403, y delega a `http_exception_handler` default para el resto. Sin regresión del handler 500 existente.
- **Flash cookie con 302**: `RedirectResponse("/", status_code=302)` admite `.set_cookie(..., httponly=True, samesite="lax")`. Las cookies sobreviven al redirect (spec HTTP). Al siguiente request, middleware lee cookie + llama `response.delete_cookie("nexo_flash")` DESPUÉS de `call_next` para expirar.
- **Catálogo de acciones sensibles**: 11 botones identificados en 6 templates. Lista exhaustiva en §Catalogue abajo. UIROL-04 cita 3 de ellos como lower bound; hay 8 más que deberían envolverse con `{% if can(...) %}` para ser coherentes.

**Primary recommendation:** Landing order — `can()` helper primero (commit 1), globals registration (commit 2), sidebar refactor (commit 3), flash middleware + HTTPException handler (commit 4), ajustes split (commit 5), buttons wrap (commit 6). Cada commit es verificable independientemente.

## Standard Stack (confirmado — locked por Phase 1 + Phase 2)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| FastAPI | 0.135+ | Web framework | Ya en uso (api/main.py) |
| Starlette | ~0.48 (tracked via FastAPI) | Middleware / exceptions | Phase 2 middleware pattern (api/middleware/auth.py, audit.py) |
| Jinja2 | via `fastapi.templating.Jinja2Templates` | Templates | Ya en uso (api/deps.py:12) |
| HTMX | 2.0.4 (CDN) | Polling + fragments | Ya en uso (base.html:8) |
| Alpine.js | 3.14.8 (CDN) | Reactive UI | Ya en uso (base.html:10) |
| Tailwind | via CDN | Styling | Ya en uso (base.html:7) |
| itsdangerous | ya en requirements | — | Session cookies (no se toca en P5) |

**Zero new dependencies.** Todo lo que necesita Phase 5 ya está en el `requirements.txt` actual.

**Version verification:** No se valida nada contra el registry — no hay paquete nuevo que instalar. Todos los imports vienen de `fastapi`, `starlette`, `jinja2` (ya instalados en Phases 1-4 con versiones pineadas).

## Architecture Patterns

### System Data Flow (request → response con permisos)

```
┌─────────┐
│ Browser │
└────┬────┘
     │ GET /bbdd (HTML)
     ▼
┌──────────────────────┐
│ AuthMiddleware       │ → si no sesión: 401 JSON o 302 /login
│ (outermost)          │
└────┬─────────────────┘
     │ request.state.user poblado (eager-loaded depts)
     ▼
┌──────────────────────┐
│ NEW: FlashMiddleware │ → lee cookie nexo_flash, pobla
│ (o extensión Auth)   │   request.state.flash, marca para delete
└────┬─────────────────┘
     │ request.state.flash = "mensaje" | None
     ▼
┌──────────────────────┐
│ AuditMiddleware      │ → log append-only (Phase 2)
└────┬─────────────────┘
     │
     ▼
┌──────────────────────┐
│ QueryTiming (Phase 4)│
└────┬─────────────────┘
     │
     ▼
┌──────────────────────┐
│ Route handler        │
│ (p.ej. /ajustes →    │
│ require_permission)  │ → si 403: raise HTTPException(403)
└────┬─────────────────┘
     │
     ▼                         ┌─────────────────────────────────┐
Si HTTPException(403):         │ NEW: http_exception_handler     │
     │                         │ - Accept text/html o path HTML: │
     └──────────────────────►  │   302 / + Set-Cookie nexo_flash │
                               │ - Accept json o /api/*:         │
                               │   JSON {detail} 403 (original)  │
                               └─────────────────────────────────┘
```

### Pattern 1: `can()` — factoring helper

**What:** Extraer la lógica de intersección de `require_permission` en un helper público puro.

**Target code** (en `nexo/services/auth.py`, justo ANTES de `PERMISSION_MAP`):

```python
def can(user: NexoUser | None, permission: str) -> bool:
    """True si ``user`` tiene ``permission``. Puro, sin side effects.

    Reglas:
    - ``user is None`` → False (no autenticado nunca pasa).
    - ``user.role == 'propietario'`` → True (bypass global).
    - Intersecta ``{d.code for d in user.departments}`` con
      ``PERMISSION_MAP[permission]``. True si hay intersección.

    Safe para llamar desde templates (D-03, D-09) sin preocuparse por
    raise, await, o dependencias de FastAPI.
    """
    if user is None:
        return False
    if user.role == "propietario":
        return True
    allowed = PERMISSION_MAP.get(permission, [])
    user_depts = {d.code for d in user.departments}
    return bool(user_depts.intersection(allowed))


def require_permission(permission: str):
    """Factory que devuelve un Dependency de FastAPI. Reutiliza ``can``."""
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

    _check.__name__ = f"require_permission__{permission.replace(':', '_')}"
    return _check
```

**When to use:** Siempre que necesites un `bool` de "este user tiene este permiso?" fuera del dependency system. Templates, tests, helpers — todos consumen `can()`. `require_permission` solo existe en los decoradores de FastAPI.

**Why factoring (not trampoline try/except):**
- `require_permission` es `async`; llamarla desde Jinja (sync) requiere `asyncio.run` — anti-pattern.
- `require_permission` raise `HTTPException`; atraparla en template es ilegible.
- Factoring es trivial: 5 líneas se mueven, firma de `require_permission` no cambia.

**Tests existentes que no deben romperse:** grep por `test_require_permission` muestra que los tests comprueban 401 (no user), 403 (wrong dept), y 200 (propietario bypass). El refactor preserva los 3 paths exactamente.

### Pattern 2: Jinja globals registration

**What:** Añadir `can` (y `current_user` como fallback opcional) a `templates.env.globals`.

**Target code** (extensión de `api/deps.py:17-22`):

```python
from nexo.services.auth import can as _can

templates = Jinja2Templates(directory=str(settings.project_root / "templates"))

templates.env.globals.update(
    app_name=settings.app_name,
    company_name=settings.company_name,
    logo_path=settings.nexo_logo_path,
    ecs_logo_path=settings.nexo_ecs_logo_path,
    # Phase 5 (D-03, D-09) — permission helper accesible en cualquier template
    can=_can,
)
```

**When to use:** Registrar globals al import-time de `api/deps.py`. Esto ocurre antes del lifespan, antes de cualquier request. Los tests que instancian `TestClient(app)` también heredan los globals — no hay riesgo de double-registration porque `api.deps` es un módulo, Python importa una sola vez.

**Why NOT a middleware o lifespan:** Los globals son estado estático del `Environment` Jinja, no del request. El `Jinja2Templates` se construye una vez y vive toda la app; es el lugar natural para pegar helpers puros.

**Template usage:**
```jinja
{% if can(current_user, "pipeline:run") %}
  <button id="btn-ejecutar" @click="ejecutar()">Ejecutar pipeline</button>
{% endif %}
```

**Nota sobre `current_user` global**: NO se registra como global. Sigue viniendo por contexto vía `render()` (api/deps.py:44). Razón: `current_user` es específico del request — mezclarlo con `env.globals` es confuso y perdería el valor distinto entre requests concurrentes (Jinja `Environment` es compartido). La alternativa limpia es que los templates que NO pasen por `render()` reciban `None` por contexto; el guard `can(user, perm)` ya maneja `user is None` devolviendo False.

**Precedent:** `api/deps.py:17-22` ya registra 4 globals; Phase 4 añadió Chart.js via `<script>` no via Jinja (no aplica). No hay precedente para filtros custom aún — Phase 5 es el primero en añadir una función global.

### Pattern 3: HTTPException(403) handler con Accept negotiation

**What:** Handler separado del `global_exception_handler` existente. Captura `StarletteHTTPException` (clase padre de `fastapi.HTTPException`).

**Target code** (en `api/main.py`, tras el `@app.exception_handler(Exception)` existente):

```python
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exception_handlers import http_exception_handler as default_http_handler


@app.exception_handler(StarletteHTTPException)
async def forbidden_html_handler(request: Request, exc: StarletteHTTPException):
    """Negocia Accept para 403: HTML → redirect + flash; JSON/API → default.

    Para cualquier otro status code (404, 422, etc.), delega al handler
    default de FastAPI que preserva el contract JSON existente.
    """
    if exc.status_code != 403:
        return await default_http_handler(request, exc)

    # Clientes API/JSON mantienen el contract actual: 403 JSON con detail.
    if _wants_json(request):
        return await default_http_handler(request, exc)

    # Usuarios HTML: redirect al dashboard con toast informativo.
    # Mensaje leído desde exc.detail ("Permiso requerido: bbdd:read") y
    # traducido a user-friendly via PERMISSION_LABELS (ver abajo).
    perm = str(exc.detail).replace("Permiso requerido: ", "")
    friendly = _friendly_permission_label(perm)
    response = RedirectResponse("/", status_code=302)
    response.set_cookie(
        "nexo_flash",
        f"No tienes permiso para acceder a {friendly}",
        max_age=60,           # vive 60s máx por si el user deja pendiente
        httponly=True,
        secure=not settings.debug,
        samesite="lax",
        path="/",
    )
    return response
```

**Key finding from FastAPI docs:** `@app.exception_handler(Exception)` NO atrapa `HTTPException`. FastAPI tiene dos handler registries separados:
1. Exception handlers específicos (p.ej. `StarletteHTTPException`, `RequestValidationError`) — se disparan ANTES.
2. El catch-all `Exception` handler — solo atrapa lo que no matchea arriba.

Esto significa que añadir el handler de 403 NO regresiona el de 500 (NAMING-07). Son handlers separados.

**Why `StarletteHTTPException` not `fastapi.HTTPException`:** La docs de FastAPI recomienda registrar sobre la clase padre (Starlette's). `fastapi.HTTPException` hereda de ella, así que un handler registrado sobre la padre atrapa ambas. Si se registra solo sobre `fastapi.HTTPException`, no atrapa excepciones internas de Starlette (p.ej. 404 de rutas). Importar de `starlette.exceptions`.

**Why reusing `_wants_json` from main.py:76-89:** Ya existe, mismo heurístico que usa `global_exception_handler` y `AuthMiddleware._wants_json` (api/middleware/auth.py:70-80). Consistencia garantizada.

### Pattern 4: Flash cookie middleware

**What:** Middleware que lee `nexo_flash` en request-in y borra en response-out.

**Target code** (nuevo archivo `api/middleware/flash.py`):

```python
"""FlashMiddleware — lee cookie `nexo_flash`, la expone y la expira.

Fuente del mensaje: ``forbidden_html_handler`` (api/main.py) lo escribe
tras 403 HTML; posibles futuras fuentes son acciones con redirect (ver
D-08 Deferred ideas).

Cookie policy (D-08): HttpOnly, Secure en prod, SameSite=Lax, max_age 60s.
Read-and-clear: se lee en el próximo request y se borra en la response.
"""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from api.config import settings


_FLASH_COOKIE = "nexo_flash"


class FlashMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        flash = request.cookies.get(_FLASH_COOKIE)
        request.state.flash = flash  # None si no existe

        response = await call_next(request)

        if flash is not None:
            # Borrar la cookie: Set-Cookie con max_age=0 y mismo path.
            # NOTA: Starlette.delete_cookie() también funciona pero envía
            # expires=epoch+0 que algunos clientes viejos ignoran. Preferir
            # el set explícito para control total.
            response.delete_cookie(_FLASH_COOKIE, path="/")

        return response
```

**Wiring in `api/main.py`** (orden LIFO: FlashMiddleware debe correr DESPUÉS de AuthMiddleware — es decir, registrarse ANTES — para que `request.state.flash` esté disponible en cualquier handler; pero también antes de Audit para que no aparezca en audit como campo desconocido):

```python
app.add_middleware(QueryTimingMiddleware)  # innermost
app.add_middleware(AuditMiddleware)
app.add_middleware(FlashMiddleware)        # NEW — entre Audit y Auth
app.add_middleware(AuthMiddleware)         # outermost
```

**Why separate middleware (not extension of AuthMiddleware):**
- Flash no depende de sesión. Un usuario anónimo que intente ver `/bbdd` puede recibir flash después de login (edge case raro pero posible). Si va dentro de Auth, se salta en rutas públicas.
- Separación de concerns: Auth decide si dejar pasar, Flash comunica resultado de acciones previas. Single-responsibility.
- Tests independientes: se puede testear `FlashMiddleware` sin levantar el stack de auth.

**Template integration** (extensión de `render()` en `api/deps.py:42-45`):

```python
ctx: dict[str, Any] = {
    "request": request,
    "current_user": getattr(request.state, "user", None),
    "flash_message": getattr(request.state, "flash", None),  # NEW
}
```

**Toast rendering in `base.html`** (al final del `<body>`, antes del cierre):

```jinja
{% if flash_message %}
  <script>
    // Dispara el sistema de toast ya existente (base.html:188-223).
    // Usa type='info' para el estilo azul del brand, 5s auto-dismiss.
    document.addEventListener('DOMContentLoaded', () => {
      if (typeof showToast === 'function') {
        showToast('info', 'Aviso', {{ flash_message|tojson }});
      }
    });
  </script>
{% endif %}
```

**Why reuse existing `showToast`:** Ya está en `base.html:188-223` con 5s auto-dismiss, animación, cola. Phase 5 no reinventa — solo añade una nueva fuente (server-rendered flash). El `tojson` filter escapa comillas y caracteres de control; previene XSS en el mensaje.

### Pattern 5: Sidebar refactor (de `visible_to` string a `can()`)

**What:** Reemplazar la tupla de 5 campos `(key, href, label, icons, visible_to)` por una tupla de 5 campos `(key, href, label, icons, permission)`. El filtro en línea 62 cambia de `visible_to == "*" or current_user.role == visible_to` a `permission is None or can(current_user, permission)`.

**Target diff** (en `templates/base.html`):

```jinja
{% set nav_items = [
  ("dashboard",    "/",            "Centro Mando",    "M3.75 3v...", None),
  ("datos",        "/datos",       "Datos",           "M4 16v...",   "datos:read"),
  ("pipeline",     "/pipeline",    "Analisis",        "M4 4v...",    "pipeline:read"),
  ("historial",    "/historial",   "Historial",       "M12 8v...",   "historial:read"),
  ("capacidad",    "/capacidad",   "Capacidad",       "M9 19V...",   "capacidad:read"),
  ("_sep1",        "",             "",                "",            None),
  ("recursos",     "/recursos",    "Recursos",        "M20 7l...",   "recursos:read"),
  ("ciclos_calc",  "/ciclos-calc", "Calcular Ciclos", "M9 7h...",    "ciclos:read"),
  ("operarios",    "/operarios",   "Operarios",       "M17 20h...",  "operarios:read"),
  ("bbdd",         "/bbdd",        "BBDD",            "M4 7v...",    "bbdd:read"),
  ("ajustes",      "/ajustes",     "Ajustes",         "M10.325...",  "ajustes:manage"),
] %}
{% for key, href, label, icon_paths, permission in nav_items %}
  {% if permission is none or can(current_user, permission) %}
    {# ... render item ... #}
  {% endif %}
{% endfor %}
```

**Why `None` for "always visible"**: más claro que el string `"*"`. El dashboard (`/`) siempre se muestra — es la home post-login. El separador `_sep1` también. Cualquier otro item mapea a un permiso del `PERMISSION_MAP`.

**Pitfall avoided:** La clave del `nav_items` tuple es 5-elemento, no 6. Cambiar de 5 a 6 (añadiendo `permission` sin quitar `visible_to`) rompe el unpack. Reemplazar in-place.

**Solicitudes badge** (base.html:82-94): sigue siendo `{% if current_user and current_user.role == 'propietario' %}`. Razón: `aprobaciones:manage` tiene lista vacía en PERMISSION_MAP → equivalente a `propietario-only`. Refactorizar para consistencia: cambiar a `{% if can(current_user, "aprobaciones:manage") %}`.

## Implementation Order

Secuencia sugerida — cada paso es independientemente verificable:

### Wave 1: Helper core (1 commit)
1. **`can()` helper** en `nexo/services/auth.py` + refactor de `require_permission` para usarlo.
   - Verificación: existing tests (`tests/services/test_auth.py`, `tests/middleware/test_rbac.py` si existen) siguen pasando sin cambios.
   - Si no hay tests de `require_permission` — añadir 3: propietario bypass, dept match, dept no-match.

### Wave 2: Template integration (1 commit)
2. **Register `can` as Jinja global** en `api/deps.py:17-22`.
   - Verificación: `grep -n "can=" api/deps.py` matches.
   - Smoke: render cualquier template existente en test → no errores.

### Wave 3: Sidebar refactor (1 commit)
3. **`base.html` nav_items** — cambiar tuple de 5 a 5 con permission; reemplazar filtro.
   - Verificación: login como propietario → ve todos los items (11); login como usuario/ingeniería → ve {dashboard, datos, pipeline, historial, capacidad, _sep, recursos, ciclos_calc, bbdd} = 9 items sin ajustes; login como usuario/rrhh → {dashboard, pipeline (read), historial, _sep, operarios} = 5 items.
4. **Solicitudes badge refactor** (base.html:82-94) de role-check a `can("aprobaciones:manage")`.

### Wave 4: Forbidden UX (2 commits)
5. **HTTPException handler** para 403 HTML redirect + flash cookie.
6. **FlashMiddleware** + integration in `render()` + toast rendering en `base.html`.
   - Verificación: `curl -H "Accept: text/html" /ajustes` como non-propietario → 302 to `/` + `Set-Cookie: nexo_flash=...`. Siguiente request muestra toast.

### Wave 5: Ajustes split (1 commit)
7. **`ajustes_conexion.html`** nuevo + refactor `ajustes.html` hub + router `pages.py`.
   - Cut del bloque `ajustes.html:74-168` (Conexión SQL Server + info sistema) → `ajustes_conexion.html`.
   - Hub queda: 5 cards (usuarios, auditoría, solicitudes, límites, rendimiento) + 1 nueva card de conexión. Sin SMTP (D-04). Sin sección de conexión inline.
   - Añadir ruta `/ajustes/conexion` en `api/routers/pages.py` con `require_permission("conexion:config")` (propietario-only por `[]` en PERMISSION_MAP).

### Wave 6: Sensitive buttons wrap (1 commit)
8. **`{% if can() %}` wrap** alrededor de los 11 botones catalogados (§Catalogue).
   - Single commit porque cada botón es atómico y no depende de los otros.

### Wave 7: Verification (no commit)
9. **Manual E2E per UIROL-05**: login con 3 roles distintos, capturar sidebar + páginas accesibles.

## Code Examples

### Example A — `can()` helper signature

```python
# nexo/services/auth.py — justo antes de PERMISSION_MAP

def can(user: NexoUser | None, permission: str) -> bool:
    if user is None:
        return False
    if user.role == "propietario":
        return True
    allowed = PERMISSION_MAP.get(permission, [])
    user_depts = {d.code for d in user.departments}
    return bool(user_depts.intersection(allowed))
```

### Example B — Sidebar item template snippet (after refactor)

```jinja
{# templates/base.html:48-79 reemplazado #}
{% set nav_items = [
    ("dashboard",   "/",            "Centro Mando",    "M3.75 ...",  None),
    ("pipeline",    "/pipeline",    "Analisis",        "M4 4v...",   "pipeline:read"),
    ("bbdd",        "/bbdd",        "BBDD",            "M4 7v...",   "bbdd:read"),
    ("ajustes",     "/ajustes",     "Ajustes",         "M10.325 ...","ajustes:manage"),
] %}
{% for key, href, label, icon_paths, permission in nav_items %}
  {% if permission is none or can(current_user, permission) %}
    {% if key.startswith('_sep') %}
      <div class="my-2 mx-3 h-px bg-brand-700"></div>
    {% else %}
      {% set active = page == key %}
      <a href="{{ href }}" class="...">
        {# ... icon + label ... #}
      </a>
    {% endif %}
  {% endif %}
{% endfor %}
```

### Example C — Button-level `{% if can() %}` wrap

```jinja
{# templates/pipeline.html:185 (Ejecutar button) #}
{% if can(current_user, "pipeline:run") %}
  <button @click="ejecutar()" class="btn-success !px-5 !py-3"
          :disabled="selectedModules.length === 0">
    <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
      <path d="M8 5v14l11-7z"/>
    </svg>
  </button>
{% endif %}
```

### Example D — Exception handler with Accept negotiation

```python
# api/main.py — añadir después del global_exception_handler existente

from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exception_handlers import http_exception_handler as _default_http_handler

_PERMISSION_LABELS = {
    "bbdd:read": "la explorador de BBDD",
    "pipeline:run": "ejecutar el pipeline",
    "ajustes:manage": "la configuración",
    "usuarios:manage": "la gestión de usuarios",
    "auditoria:read": "el log de auditoría",
    "operarios:read": "los datos de operarios",
    "recursos:edit": "la edición de recursos",
    # ... resto en apéndice completo
}


def _friendly_permission_label(perm: str) -> str:
    return _PERMISSION_LABELS.get(perm, perm)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code != 403:
        return await _default_http_handler(request, exc)
    if _wants_json(request):
        return await _default_http_handler(request, exc)

    perm = str(exc.detail).replace("Permiso requerido: ", "")
    friendly = _friendly_permission_label(perm)
    response = RedirectResponse("/", status_code=302)
    response.set_cookie(
        "nexo_flash",
        f"No tienes permiso para acceder a {friendly}",
        max_age=60,
        httponly=True,
        secure=not settings.debug,
        samesite="lax",
        path="/",
    )
    return response
```

### Example E — FlashMiddleware

```python
# api/middleware/flash.py (nuevo)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_FLASH_COOKIE = "nexo_flash"


class FlashMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        flash = request.cookies.get(_FLASH_COOKIE)
        request.state.flash = flash
        response = await call_next(request)
        if flash is not None:
            response.delete_cookie(_FLASH_COOKIE, path="/")
        return response
```

### Example F — Toast rendering in base.html

```jinja
{# templates/base.html — antes del </body>, después del script de toastSystem #}
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

## Catalogue of Sensitive Actions

Enumeración exhaustiva de botones/acciones que deben envolverse en `{% if can(...) %}`. Lista completa — UIROL-04 cita solo 3, aquí hay 11.

| # | Template | Line | Action | Permission | Notes |
|---|----------|------|--------|------------|-------|
| 1 | `templates/pipeline.html` | 185 | "Ejecutar" (SVG play) | `pipeline:run` | ✅ UIROL-04 explícito |
| 2 | `templates/historial.html` | 112 | "Borrar ejecución" (SVG trash) | `informes:delete` | ✅ UIROL-04 explícito. El router es `DELETE /api/historial/{ejec_id}` con `Depends(require_permission("informes:delete"))` en `api/routers/historial.py:188`. |
| 3 | `templates/historial.html` | 107 | "Generar PDFs" (regenerar) | `informes:delete` | Regenera PDFs — destruye existentes. Mismo perm que borrar. Alternativa: `informes:read` si se considera no-destructivo. Sugerencia: `informes:delete` (explícito en `api/routers/historial.py:167`). |
| 4 | `templates/recursos.html` | 18 | "Detectar" máquinas | `recursos:edit` | Dispara detección → escribe sugerencias. ✅ UIROL-04 "Sincronizar recursos" mapea aquí. |
| 5 | `templates/recursos.html` | 52 | "Guardar todo" | `recursos:edit` | PUT masivo de la lista de recursos. |
| 6 | `templates/recursos.html` | 76 | "Eliminar sección" (X icon) | `recursos:edit` | Destructivo. |
| 7 | `templates/recursos.html` | 157 | "Eliminar máquina" (trash) | `recursos:edit` | Destructivo. |
| 8 | `templates/recursos.html` | 41 | "Añadir máquina" (dropdown) | `recursos:edit` | Abre modal de creación. |
| 9 | `templates/recursos.html` | 45 | "Añadir sección" (dropdown) | `recursos:edit` | Abre modal de creación. |
| 10 | `templates/ciclos_calc.html` | 208 | "Openar guardar diálogo" (save) | `ciclos:edit` | Persiste ciclos calculados. |
| 11 | `templates/ajustes.html` | 34-71 | 5 cards propietario-only | `ajustes:manage` | Ya gatean con role == 'propietario'; refactorizar a `can()` para DRY. |

**Botones NO sensibles** (no necesitan wrap): filtros de UI (`setRange`, `sortBy`, `toggleExpand`), navegación local (`step = X`, `activeTab = ...`), modales informativos (`cancel`, `closeModal`), toggles de show/hide (`sidebarOpen`, `showPass`).

**Botones con approval flow** (gatados por router, no por rol directo): pipeline.html:349 `confirmRun()`, pipeline.html:371 `requestApproval()`, bbdd.html:325 `confirmRun()`, bbdd.html:347 `requestApproval()`. Éstos NO necesitan wrap: el backend ya usa `require_permission` + preflight + approvals (Phase 4). El wrap en pipeline.html:185 ya gate la ENTRADA al flujo; una vez el modal se abre, confirmar ejecución NO requiere re-verificación de UI porque el handler hace el check.

**Pagination of `/api/recursos`**: el router `api/routers/recursos.py` tiene `_edit = [Depends(require_permission("recursos:edit"))]` (línea 29) aplicado a PUT/POST/DELETE. Los botones visibles en recursos.html disparan estas llamadas — envolviendo con `{% if can(current_user, "recursos:edit") %}` evita que un usuario de producción (que tiene `recursos:read` pero no `:edit`) vea botones que fallarían 403.

## Don't Hand-Roll

Trampas explícitas de duplicación a evitar:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Check "user puede X?" en Python | Nueva función `has_permission(user, perm)` que copia el algoritmo | `can(user, perm)` del §Pattern 1 | Una sola fuente de verdad. |
| Check permiso en JS (frontend) | `window.__USER__.permissions = [...]` + `x-show="canPipeline"` | Renderizar server-side con `{% if can() %}` (zero-trust DOM, D-02) | Backend + frontend se desincronizan; devtools bypass. |
| Toast post-403 custom | Ventana modal nueva + manejo de timers manual | Reuse `showToast('info', title, msg)` de `base.html:188` | 5s auto-dismiss, cola, styling Tailwind ya resueltos. |
| Session tracking para flash | `request.session["flash"] = ...` con SessionMiddleware | Cookie simple `nexo_flash` (D-08) | SessionMiddleware no está instalado; cookie cumple el job. |
| Página 403 dedicada | Nuevo `templates/forbidden.html` + route handler | Redirect + flash (D-07) | Friction extra; UX peor; sidebar ya filtra. |
| Re-validación de 403 en templates de `/api/*` | Cambiar contract JSON para incluir redirect URL | NO TOCAR `/api/*` — solo handler HTML cambia (D-07) | HTMX y clients esperan JSON; romper contract = regresión P4. |
| Modelo de roles en DB más fino | Crear tabla `nexo.feature_flags` o `nexo.ui_visibility` | Reutilizar `PERMISSION_MAP` existente (§CONTEXT D-01) | Segunda fuente de verdad = desincronización garantizada. |
| Polling `/api/me/permissions` para refrescar | Fetch en cada page load | Server-rendered con `can()` (D-02, D-09) | +1 round-trip, botones parpadean; overkill para Mark-III (permisos no cambian en sesión). |

**Key insight:** Phase 5 compone sobre Phase 2 (PERMISSION_MAP + require_permission) y Phase 4 (render helper + toast system). No inventa infraestructura nueva — orquesta primitivas existentes.

## Common Pitfalls

### Pitfall 1: `@app.exception_handler(Exception)` NO atrapa `HTTPException`

**What goes wrong:** Plan asume que "el global_exception_handler existente ya atrapa todo, solo hay que añadir un case para 403". FALSO.

**Why it happens:** FastAPI maneja `HTTPException` con un handler default (retorna JSON con `detail`). El `Exception` handler es catch-all de lo que NO es HTTPException. Registrar uno no desactiva el otro; los handlers son específicos por tipo.

**How to avoid:** Añadir un nuevo `@app.exception_handler(StarletteHTTPException)` (clase padre, no `fastapi.HTTPException`). Para status != 403, delegar a `fastapi.exception_handlers.http_exception_handler` (el default). Para 403 HTML, redirect + flash. Para 403 JSON, delegar a default.

**Warning signs:** Si tras el merge se ejecuta `curl /ajustes -H "Accept: text/html"` y se recibe `{"detail": "Permiso requerido: ajustes:manage"}` (JSON) en vez de 302, el handler no está capturando — probablemente se registró sobre `fastapi.HTTPException` en vez de `StarletteHTTPException`, o hay orden de registro incorrecto.

### Pitfall 2: Cookie `nexo_flash` se pierde si Secure=True en dev HTTP

**What goes wrong:** En dev (HTTP local, `NEXO_DEBUG=true`), si `Secure=True` el browser descarta la cookie. Usuario hace login, recibe redirect 403 → `/` con cookie → cookie no llega → no toast.

**Why it happens:** Browsers modernos (Chrome 100+, Firefox 100+) aplican estrictamente `Secure=True`: solo enviadas sobre HTTPS.

**How to avoid:** Usar `secure=not settings.debug`. En prod (Caddy + TLS) queda `True`; en dev queda `False`.

**Warning signs:** Flash funciona en prod pero no en dev. Verificar con devtools → Application → Cookies → `nexo_flash` presente o ausente tras el redirect.

### Pitfall 3: Registrar `current_user` como Jinja global rompe concurrencia

**What goes wrong:** `templates.env.globals["current_user"] = lambda: request.state.user` captura un closure sobre `request` que no existe al registro-time. O peor, una instancia fija → todos los requests concurrentes ven el mismo user.

**Why it happens:** `env.globals` es compartido entre requests; Jinja NO pone current request en el global namespace.

**How to avoid:** Dejar `current_user` venir por contexto (`render()` en deps.py lo hace ya). Solo `can` va a globals — es función pura que recibe `user` como parámetro explícito. Los templates hacen `{% if can(current_user, "x") %}` donde `current_user` viene del contexto del request.

**Warning signs:** Logs de errores con `RuntimeError: Working outside of request context` o, peor, auditoría cruzada donde user A ve datos de user B.

### Pitfall 4: HTML GET de páginas sensibles (e.g. `/bbdd`) no tiene `require_permission`

**What goes wrong:** Un usuario de RRHH teclea `/bbdd` en el browser. Sidebar ya lo oculta (tras D-01), pero la URL sigue accesible. Llega a la página, render exitoso. Los XHR/fetch subsecuentes fallan 403 JSON — UX confusa (página cargada pero todo roto).

**Why it happens:** `api/routers/pages.py:96-98` `bbdd_page` NO tiene `Depends(require_permission("bbdd:read"))`. Solo `/ajustes` sí lo tiene (línea 108).

**How to avoid:** Añadir `dependencies=[Depends(require_permission("{modulo}:read"))]` a los GET de `/bbdd`, `/pipeline`, `/historial`, `/recursos`, `/ciclos-calc`, `/operarios`, `/datos`, `/capacidad`. Esto dispara el flujo D-07 (redirect + flash) cuando un user sin permiso teclea la URL.

**Warning signs:** Manual test UIROL-05 revela que un usuario de RRHH puede tipear `/pipeline` y ver la página, aunque vacía. Este pitfall NO está en las decisiones de CONTEXT.md pero es un gap coherente con D-01 (sidebar filtra) y D-07 (403 → flash) — sin esto, D-07 nunca se dispara en navegación humana.

**Recommendation:** Incluir este hardening como task extra en el plan. Coste: 7 líneas (una por página). Beneficio: UIROL-05 smoke pasa sin hand-waving.

### Pitfall 5: Jinja global `can` se registra DESPUÉS del primer render

**What goes wrong:** Si `templates.env.globals["can"] = can` se llama desde el lifespan (antes de yield), los templates renderizados ANTES del startup (p.ej. error handlers que se disparan durante import) ven `can` undefined.

**Why it happens:** Import-order issues. `api/deps.py` importa a tiempo de import de `api/main.py`, `templates` se instancia a import-time. El lifespan corre DESPUÉS.

**How to avoid:** Registrar `can` en `api/deps.py` a import-time (al lado de los otros globals, líneas 17-22). No lifespan, no factory.

**Warning signs:** `UndefinedError: 'can' is undefined` en los primeros requests post-deploy.

### Pitfall 6: Ajustes split — `/ajustes` hub queda accesible si alguien teclea la URL

**What goes wrong:** Sidebar oculta `/ajustes` para non-propietario (D-01). Pero si tecleas la URL, `require_permission("ajustes:manage")` tira 403 → Pitfall 4 fix lo convierte en redirect + flash ✅. OK entonces.

**However:** Si se olvida añadir `require_permission("conexion:config")` al nuevo `/ajustes/conexion` (D-06), la página queda públicamente accesible para cualquier usuario autenticado. El router DEBE tener el dependency.

**How to avoid:** En `pages.py`, nuevo handler:
```python
@router.get(
    "/ajustes/conexion",
    dependencies=[Depends(require_permission("conexion:config"))],
)
def ajustes_conexion_page(request: Request):
    return render("ajustes_conexion.html", request, _common_extra("ajustes_conexion"))
```

**Warning signs:** UIROL-05 smoke como usuario de ingeniería → `/ajustes/conexion` no debería cargar. Si carga → falta el dependency.

### Pitfall 7: `render()` en `api/deps.py` NO renombrar a `render_html`

**What goes wrong:** El downstream_consumer prompt menciona `render_html`. Sin embargo el código real usa `render()` (api/deps.py:25). Si el planner renombra, rompe todos los imports existentes en `pages.py`, `auth.py`, `usuarios.py`, `auditoria.py`, `approvals.py`, `limites.py`, `rendimiento.py`.

**How to avoid:** Mantener el nombre `render`. Solo extender con `flash_message` (línea 42 modificada). NO tocar la firma.

**Warning signs:** Tests integración que importan `from api.deps import render_html` fallan.

## Open Questions

1. **Friendly labels for permissions** — ¿Dónde vive el dict `_PERMISSION_LABELS`? Opciones:
   - (a) En `api/main.py` junto al handler (explícito, local, 20 líneas).
   - (b) En `nexo/services/auth.py` junto a `PERMISSION_MAP` (acoplado a la fuente de verdad).
   - (c) En `docs/AUTH_MODEL.md` con función de lookup que lee un YAML.
   - **Recommendation:** Opción (b). El label es metadata del permiso, vive con el permiso. El handler importa `PERMISSION_LABELS`. Cambiar labels no requiere tocar el handler.

2. **HTML GET endpoints sin `require_permission`** (Pitfall 4) — ¿Se aborda en Phase 5 o se difiere? La CONTEXT no lo menciona explícitamente.
   - **Recommendation:** Incluir en Phase 5 como "hardening de D-07" — sin esto, D-07 no se dispara en URL typing. Coste bajo (7 líneas), beneficio alto (UX coherente).

3. **Directivo vs usuario dentro del mismo departamento** — PERMISSION_MAP no distingue. La CONTEXT explícitamente marca esto como out-of-scope Mark-III. No es una pregunta para Phase 5.

4. **`/ajustes/conexion` — qué API backend consume?** — El formulario del template actual (ajustes.html:180-206) hace `PUT /api/conexion/config` y `GET /api/conexion/status`. Esos endpoints ya existen con `require_permission("conexion:config")` y `conexion:read` respectivamente (`api/routers/conexion.py`). El split de template NO requiere nuevos endpoints — solo mueve el markup + Alpine componente. Confirmar en el plan que no se duplica lógica de conexión.

5. **¿Se añade `can(current_user, ...)` al hub de ajustes para las 5 cards?** — Hoy gate con `role == 'propietario'` (ajustes.html:34). Refactorizar a `can(current_user, "usuarios:manage")` etc. da consistencia. Tradeoff: 5 permissions distintas (`usuarios:manage`, `auditoria:read`, `aprobaciones:manage`, `limites:manage`, `rendimiento:read`) — todas lista vacía en PERMISSION_MAP hoy. Funcionalmente equivalente a `propietario`. Recomendación: refactorizar para DRY; si Mark-IV habilita alguna card a directivo, el template ya está listo.

## Metadata

**Confidence breakdown:**
- `can()` extraction pattern: HIGH — código visible en auth.py:235-271, refactor es mecánico.
- Jinja globals registration: HIGH — precedente exacto en api/deps.py:17-22.
- Accept-header handler: HIGH — verificado contra docs oficiales FastAPI + discusión GitHub #11741.
- Flash cookie middleware: HIGH — patrón documentado en Starlette responses + middleware docs.
- Button catalogue: HIGH — enumeración manual de 6 templates con line numbers exactos.
- Permission coverage of all sensitive actions: MEDIUM — los 11 botones identificados son los evidentes; una auditoría adicional con `grep -r "fetch(" templates/` podría encontrar más.

**Research date:** 2026-04-20
**Valid until:** 2026-05-20 (30 días — stack estable, FastAPI no pivota).

**Sources:**

- [FastAPI Handling Errors — Starlette HTTPException](https://fastapi.tiangolo.com/tutorial/handling-errors/) — HIGH confidence on handler registration pattern.
- [Starlette Responses — set_cookie signature](https://starlette.dev/responses/) — HIGH confidence on cookie policy.
- [Starlette Middleware — BaseHTTPMiddleware dispatch](https://starlette.dev/middleware/) — HIGH confidence on response modification after call_next.
- [Jinja API — env.globals vs env.filters](https://jinja.palletsprojects.com/en/stable/api/) — HIGH confidence on register-function-as-global pattern.
- `nexo/services/auth.py:200-271` — fuente de PERMISSION_MAP + require_permission.
- `api/deps.py:17-22` — precedente de Jinja globals registration.
- `api/main.py:126-167` — global_exception_handler existente (NAMING-07).
- `api/middleware/auth.py:70-80` — `_wants_json` heurístico reutilizable.
- `templates/base.html:48-95` — nav_items target de refactor D-01.
- `templates/ajustes.html:74-168` — bloque Conexión SQL Server target de cut D-06.
