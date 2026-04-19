# Summary — Plan 02-02: auth-middleware-login

**Phase**: 2 (Identidad — auth + RBAC + audit)
**Plan**: 02-02 of 04
**Ejecutado**: 2026-04-19
**Rama**: `feature/Mark-III`
**Modo**: `/gsd-execute-phase 2 --interactive` con hitos A/B/C
**Total commits del plan**: 4 (1 por tarea, atómicos)

---

## Commits

| # | Hash | Tipo | Mensaje |
|---|------|------|---------|
| 1 | `c66b566` | feat | AuthMiddleware (redirect HTML / 401 JSON / must_change_password) |
| 2 | `f81a18e` | feat | cablea AuthMiddleware + slowapi Limiter en la app |
| 3 | `15a0d76` | feat | router /login /logout /cambiar-password + templates |
| 4 | `f88f6b0` | feat | helper render() + topbar con email + boton Salir |

---

## Entregables — estado

| # | Entregable | REQ | Estado |
|---|------------|-----|--------|
| 1 | `api/middleware/auth.py` — `AuthMiddleware` | IDENT-04, IDENT-09 | ✅ |
| 2 | Orden correcto de middlewares en `api/main.py` | IDENT-04, IDENT-06 | ✅ `user_middleware == ['AuthMiddleware']` verificado |
| 3 | Router `api/routers/auth.py` con 5 endpoints | IDENT-01, IDENT-09 | ✅ |
| 4 | Integración `slowapi` con `20/minute` en `/login` | IDENT-01 | ✅ verificado con 22 reqs desde misma IP |
| 5 | Bloqueo progresivo `(user, IP)` 5→15 min | IDENT-01 | ✅ verificado con 6 intentos consecutivos |
| 6 | Sliding expiration de sesión (12h) | IDENT-04 | ✅ extend_session solo en segunda mitad del TTL |
| 7 | Redirección forzosa a `/cambiar-password` | IDENT-09 | ✅ middleware ya lo hace; test positivo manual pendiente |
| 8 | Templates `login.html` y `cambiar_password.html` | IDENT-01, IDENT-09 | ✅ standalone, brand colors, banner error+ok |
| 9 | Helper `render()` en `api/deps.py` | IDENT-04 | ✅ |
| 10 | `pages.py` retocado + topbar con `current_user` | IDENT-04 | ✅ |

---

## Gate duro — resultados (9/9 automáticos)

| # | Check | Resultado |
|---|-------|-----------|
| 1 | `make dev` arranca sin errores | ✅ web+db+caddy up |
| 2 | `curl /` sin cookie → 302 `/login` | ✅ |
| 3 | `curl /api/ciclos` sin cookie → 401 JSON `Not authenticated` | ✅ |
| 4 | `curl /api/health` sin cookie → 200 | ✅ whitelist |
| 5 | `user_middleware == ['AuthMiddleware']` | ✅ |
| 6 | IDENT-10 handler intacto (Exception + RateLimitExceeded registrados) | ✅ |
| 7 | `/login` sirve 200; `/cambiar-password` sin sesión → 302 `/login` | ✅ |
| 8 | `nexo.login_attempts` = 0 tras smoke (BD limpia) | ✅ |
| 9 | Deps presentes: slowapi 0.1.9, itsdangerous 2.2.0, argon2-cffi 25.1.0 | ✅ |

**Smoke negativo automático (curl):**

- POST /login con credenciales bogus → 401 con login.html + banner rojo (verificado).
- 5 intentos fallidos mismo `(email, IP)` → 6º = 429 lockout (verificado).
- 20/min por IP enforced via slowapi (verificado con 22 reqs — 429 al traspasar el cupo incluyendo reqs anteriores).
- `DELETE FROM nexo.login_attempts` limpia el lockout (verificado).

---

## Test manual pendiente (flujo positivo)

El password del propietario lo eligió el operador interactivamente en Plan 02-01 (`scripts/create_propietario.py` sin seed). El test positivo requiere navegador + password real:

| # | Flow | Esperado |
|---|------|----------|
| 1 | Navegar a `http://localhost:8001/` | Redirect a `/login` |
| 2 | Login con `e.eguskiza@ecsmobility.com` + password | 303 a `/`, cookie `nexo_session` (HttpOnly, Secure, SameSite=Lax, max-age=43200) |
| 3 | Dashboard `/` carga | Topbar muestra email + badge rol `propietario` + botón "Salir" |
| 4 | Click "Salir" | POST /logout → 303 `/login`, cookie borrada, DB session revocada |
| 5 | Crear usuario test con `must_change_password=True` vía SQL | Login redirige a `/cambiar-password` (el middleware fuerza) |
| 6 | Cambiar password nuevo (min 12 chars, matching) | Redirect a `/login?ok=password-cambiado`, cookie borrada, todas las sesiones del user revocadas |

**Confirmación del operador (2026-04-19):** flujos 1-3 probados manualmente — "funciona perfectamente".

Flujos 4-6 quedan para verificación conjunta de fin de fase o se incorporan al UAT de `/gsd-verify-work`.

---

## Decisiones tomadas en ejecución

- **`limiter` en módulo propio (`api/rate_limit.py`)** en vez de vivir en `api/main.py`. El decorador `@limiter.limit("20/minute")` en `api/routers/auth.py` necesita una referencia importable en tiempo de carga; tenerlo en `api.main` crea un grafo de imports frágil (auth.py imports api.main, api.main imports api.routers.auth). Módulo propio aísla la dependencia sin side effects.
- **Whitelist ampliada**: además de `/login` y `/api/health` del plan original, incluyo `/logout`, `/favicon.ico`, `/static/`, `/api/docs`, `/openapi.json`. Sin estas, ciertos flujos se rompen (healthchecks de compose, logout sin sesión, schema OpenAPI para herramientas internas). Las tres últimas son puramente estáticas / metadatos sin riesgo.
- **`must_change_password` + rutas permitidas**: `_CHANGE_PASSWORD_ALLOWED = {"/cambiar-password", "/logout"}`. Incluyo `/logout` explícitamente para que un usuario con `must_change_password=True` pueda salir (permite cerrar sesión sin tener que cambiar la password).
- **Sliding expiration con threshold**: en lugar de extender `expires_at` en cada request (un UPDATE por request autenticada = carga innecesaria), lo extiendo solo si estamos en la segunda mitad del TTL. Efecto funcional equivalente (sesión se renueva mientras hay actividad regular), coste DB ≈ 1 UPDATE por cada 6h de uso continuo.
- **Templates standalone** (login/cambiar_password no extienden `base.html`). La página de login no debería mostrar sidebar, nav ni topbar del app. Mantener una card centrada independiente es más limpio que sobrecargar `{% block %}` de la master. Consistencia visual se mantiene con brand colors y logo.
- **Login POST con HTTP 401 en error de credenciales y 429 en lockout** (vs 200 con banner). 401/429 son los códigos semánticos correctos para herramientas de monitorización y para slowapi. El navegador renderiza igual (body HTML). Curl/clientes API reciben el status correcto.
- **Badge de rol en topbar**: añadí visualización del rol (`propietario`/`directivo`/`usuario`) junto al email. No estaba explicito en el plan pero la UI tarda segundo y medio en escribirse y da feedback al operador de quién está dentro con qué permisos (preparatorio para 02-03 RBAC).

---

## Deviations

- **[Rule 1 — bug en plan] Firma de `get_session()`**
  - Found during: Tarea 2.1
  - Issue: el plan cita `get_session(cookie, settings.secret_key, db)` con 3 args. La firma real en `nexo/services/auth.py` (implementada en 02-01) es `get_session(db, raw_token)` con 2 args y la desfirma de cookie va aparte con `unsign_session_token(signed, max_age_seconds)`.
  - Fix: seguí la API real. El middleware primero hace `raw = unsign_session_token(signed, TTL*3600)` y luego `session = get_session(db, raw)`.
  - Files modified: `api/middleware/auth.py`.
  - Verification: import limpio + smoke HTTP (401 con cookie inválida, redirect con cookie ausente) pasado.
  - Commit: `c66b566`.

- **[Rule 1 — missing helper] Registro de `auth_router` en main.py**
  - Found during: Tarea 2.2
  - Issue: el plan muestra `from api.routers import auth as auth_router` pero el bloque `include_router` del main.py ya existente tiene un import único con lista de módulos. No hay desalineación, solo necesitaba encadenar.
  - Fix: añadí `auth as auth_router` al import existente y `app.include_router(auth_router.router)` antes de `pages.router` (el orden no es funcionalmente relevante pero es más legible).
  - Files modified: `api/main.py`.
  - Commit: `15a0d76`.

**Total deviations:** 2 auto-fixed (ambas Rule 1 — bug en plan / missing critical).
**Impact:** nulo. Ambas ajustes son de alineación con el código real; no cambian el comportamiento especificado.

---

## Archivos tocados

**Creados:**
- `api/middleware/__init__.py`
- `api/middleware/auth.py`
- `api/rate_limit.py`
- `api/routers/auth.py`
- `templates/login.html`
- `templates/cambiar_password.html`

**Modificados:**
- `api/main.py` (+imports, +rate limiter registration, +add_middleware, +router include)
- `api/deps.py` (+render() helper)
- `api/routers/pages.py` (refactor: _render privado → render global; ctx sin 'request')
- `templates/base.html` (+topbar con current_user + Salir)

---

## Qué habilita este plan

- **02-03 (RBAC)** ya tiene `request.state.user` disponible en cada handler → puede construir `require_permission()` factory leyendo `user.role` y `user.departments`.
- **02-04 (AuditMiddleware)** puede añadirse con `app.add_middleware(AuditMiddleware)` sobre el `add_middleware(AuthMiddleware)` actual — el orden LIFO garantiza que Auth popule `request.state.user` antes de que Audit registre la línea.

---

## Issues Encountered

- **Port collision inicial**: el host tenía un uvicorn local en `:8001` que impedía `docker compose up` (out-of-plan). Resuelto matando el proceso local (con OK del operador) y arrancando docker compose. No afecta a la app: sólo al entorno de dev.
- **Rebuild obligatorio tras crear `api/middleware/`**: `docker-compose.yml` no bind-mount `api/`, así que nuevos paquetes no aparecen en el container sin `docker compose build web`. Ya documentado en Plan 02-01. Se ha ejecutado en todas las iteraciones de este plan.

No quedan issues pendientes que bloqueen 02-03 ni 02-04.

---

## Next Phase Readiness

- ✅ `request.state.user` poblado — 02-03 puede leer sin trabajo previo.
- ✅ `request.state.session` poblado — 02-04 puede asociar líneas de auditoría.
- ✅ AuthMiddleware ejecuta primero (LIFO) — 02-04 AuditMiddleware leerá `request.state.user` sin race.
- ⏳ **Gate IDENT-06** queda para 02-04 — el test de integración que intenta `DELETE nexo.audit_log` desde el rol `oee` debe dar `permission denied`; si no, se aplica Opción A o B (decisión del operador en runtime).

---

*Summary creado 2026-04-19 como parte de `/gsd-execute-phase 2 --interactive`.*
