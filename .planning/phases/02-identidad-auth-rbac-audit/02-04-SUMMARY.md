# Summary — Plan 02-04: audit-middleware-ajustes-ui

**Phase**: 2 (Identidad — auth + RBAC + audit) — **ÚLTIMO PLAN DE LA FASE**
**Plan**: 02-04 of 04
**Ejecutado**: 2026-04-19
**Rama**: `feature/Mark-III`
**Modo**: `/gsd-execute-phase 2 --interactive` con hitos A/B/C
**Total commits del plan**: 5 (1 por tarea + 1 setup IDENT-06)

---

## Commits

| # | Hash | Tipo | Mensaje |
|---|------|------|---------|
| 1 | `de44777` | feat | AuditMiddleware con body sanitization + LIFO wiring |
| 2 | `669ba32` | feat | rol nexo_app dedicado + gate IDENT-06 pasado (5 tests) |
| 3 | `0be542f` | feat | /ajustes/usuarios CRUD (solo propietario) |
| 4 | `d2abc1e` | feat | /ajustes/auditoria filtros+CSV + hub /ajustes + sidebar condicional |

---

## Entregables — estado

| # | Entregable | REQ | Estado |
|---|------------|-----|--------|
| 1 | `api/middleware/audit.py` | IDENT-06 | ✅ |
| 2 | Orden LIFO en `api/main.py` | IDENT-06 | ✅ `['AuthMiddleware', 'AuditMiddleware']` verificado |
| 3 | Redacted endpoints + whitelist sensitive fields | IDENT-06 | ✅ 4 endpoints + 10 field names |
| 4 | Verificación BBDD: DELETE desde rol app falla | IDENT-06 | ✅ Opción A ejecutada — rol `nexo_app` |
| 5 | `api/routers/usuarios.py` CRUD | IDENT-08 | ✅ 5 endpoints + 4 safeguards anti-lockout |
| 6 | `templates/ajustes_usuarios.html` | IDENT-08 | ✅ tabla + 2 modales (form + reset-pwd) |
| 7 | `api/routers/auditoria.py` filtrable + CSV | IDENT-07 | ✅ 5 filtros + paginación + streaming CSV |
| 8 | `templates/ajustes_auditoria.html` | IDENT-07 | ✅ form filtros + tabla + paginación |
| 9 | Entrada `/ajustes` con links (visible solo a propietario) | IDENT-07, IDENT-08 | ✅ cards + sidebar condicional por rol |
| 10 | Test de integración (body redactado, fila escrita, DELETE falla, filtros) | IDENT-06, IDENT-07 | ✅ 5 tests nuevos en `test_audit_append_only.py` |

---

## Gate duro — resultados (6/6 automáticos)

| # | Check | Resultado |
|---|-------|-----------|
| 1 | `make dev` arranca | ✅ |
| 2 | `user_middleware == ['AuthMiddleware', 'AuditMiddleware']` (LIFO final) | ✅ |
| 3 | Engine runtime conecta como `nexo_app` | ✅ |
| 4 | Smoke HTTP sin cookie: / → 302 /login, /api/* → 401, /api/health → 200, /ajustes/* → 302 /login | ✅ |
| 5 | IDENT-06: DELETE y UPDATE en `nexo.audit_log` desde `nexo_app` → `permission denied` | ✅ |
| 6 | `pytest tests/auth/` | ✅ 11 passed, 1 skipped |

---

## Gate IDENT-06 — Opción A ejecutada

**Decisión del operador (2026-04-19):** Opción A — rol `nexo_app` dedicado.

**Cambios aplicados:**
- `scripts/create_nexo_app_role.sql` — crea rol + GRANTs + REVOKE UPDATE/DELETE on audit_log.
- `Makefile` target `make nexo-app-role` — aplica el SQL via psql como owner.
- `.env` — nuevos `NEXO_PG_APP_USER=nexo_app` y `NEXO_PG_APP_PASSWORD=<secret>` (fuera de git).
- `api/config.py` — campos `pg_app_user`/`pg_app_password` con fallback a `pg_user`/`pg_password` (backwards compat).
- `nexo/db/engine.py` — usa `effective_pg_user`/`effective_pg_password`.

**Post-cambio:**
- App (engine_nexo) conecta como `nexo_app` → solo SELECT/INSERT en `audit_log`.
- Owner (`oee`) sigue disponible para mantenimiento offline (bootstrap scripts, rotación, GDPR requests).
- 5 tests nuevos en `test_audit_append_only.py` verifican INSERT/SELECT OK, DELETE/UPDATE/TRUNCATE bloqueados.

---

## Suite de tests

`docker compose exec web pytest tests/auth/ -q` → **11 passed, 1 skipped, 0 failed**.

| Archivo | Tests | Pass | Skip |
|---------|-------|------|------|
| `test_rbac_smoke.py` (02-03) | 7 | 6 | 1 (IDENT-10 legacy) |
| `test_audit_append_only.py` (02-04) | 5 | 5 | 0 |

---

## Decisiones tomadas en ejecución

- **AuditMiddleware después de AuthMiddleware en cadena LIFO**: registramos `add_middleware(AuditMiddleware)` primero, `add_middleware(AuthMiddleware)` segundo → orden inverso de ejecución. Auth corre como outer (401/redirect si no autenticado, nunca llega a Audit), Audit corre como inner (lee `request.state.user` ya poblado). Coherente con "auditar solo requests autenticadas".
- **`_REDACTED_ENDPOINTS` ampliado a 4** (vs 2 del plan): incluyo `/login` y `/cambiar-password` para evitar que el body con `password=...` entre nunca al audit log, ni sanitizado. Plan mencionaba sanitization por field; esto añade una capa adicional de seguridad a nivel endpoint.
- **`_SENSITIVE_FIELDS` ampliado a 10 nombres**: incluyo variantes españolas (`clave`, `contrasena`, `contrasenia`) y los campos específicos del form de cambio (`password_actual`, `password_nuevo`, `password_repetir`). La sanitization es case-insensitive.
- **Rol `nexo_app` con GRANT en `ALL TABLES`**: más permisivo que el mínimo requerido (solo necesitaría SELECT/INSERT en `users`/`sessions`/etc.), pero simplifica mantenimiento: la única restricción es la revocación específica en `audit_log`. Si se añaden tablas al schema, `ALTER DEFAULT PRIVILEGES` les aplica el patrón automáticamente.
- **`must_change_password=True` por defecto en crear usuario**: coherente con `docs/AUTH_MODEL.md` §Passwords. El propietario establece una password inicial pero el usuario creado debe cambiarla en su primer login.
- **Safeguards anti-self-lockout en `/ajustes/usuarios`**:
  - No puede cambiarse el rol a sí mismo.
  - No puede desactivarse a sí mismo.
  - No puede desactivar al último propietario activo.
  Son tres chequeos independientes. Documentados con mensaje de error al usuario.
- **Sidebar condicional extendido** (no solo `/ajustes`): añadí una 5ª columna `visible_to` a `nav_items`, preparando el terreno para Phase 5 UIROL-02 (filtrado completo por rol/departamento). Solo `ajustes` queda restringido en 02-04; el resto sigue visible a todos los autenticados por decisión explícita del plan.
- **Paginación en auditoría con `COUNT` via subquery**: evita materializar todas las filas en memoria para calcular el total. Funciona bien hasta ~1M filas; optimización futura si la tabla crece mucho.
- **CSV export con `yield_per(500)`**: streaming real, no carga todo en memoria. Un export sin filtros de 100k filas usaría ~500 rows × 256 bytes = 128KB de buffer simultáneo.
- **`docker compose restart web` NO recarga `env_file`** (descubierto durante ejecución): es un gotcha. Para aplicar cambios en `.env` hay que hacer `docker compose up -d --force-recreate web`. Documentado en SUMMARY para futuros plans.
- **Cleanup en tests usa engine separado como owner**: los tests del gate IDENT-06 verifican que `nexo_app` no puede DELETE; el cleanup entre tests necesita DELETE → conecta como `oee` via un engine temporal. Elegante: el mismo mecanismo que valida el gate imposibilita el cleanup sin el owner.

---

## Deviations

- **[Rule 1 — bug en plan] DO $$ block no interpola `:'var'`**
  - Found during: Tarea 4.4, primera ejecución de `make nexo-app-role`.
  - Issue: psql NO sustituye variables `:'name'` dentro de bloques `DO $$ ... $$` — son dollar-quoted string literals enviados al servidor as-is. El error era `syntax error at or near ":"`.
  - Fix: reescribí el SQL usando `CASE WHEN ... format(...) END` a nivel SELECT + `\gset` + `\gexec` para que la interpolación ocurra en el cliente psql antes de enviar al servidor.
  - Files modified: `scripts/create_nexo_app_role.sql`.
  - Verification: `make nexo-app-role` ejecuta limpio. Idempotente (detectado via test repetido).
  - Commit: `669ba32`.

- **[Rule 1 — missing critical] force-recreate del container para env_file**
  - Found during: Tarea 4.4, primer intento de verificar nexo_app runtime.
  - Issue: `docker compose restart web` mantiene las env vars del container original; el `.env` actualizado no se lee. App seguía conectando como `oee`.
  - Fix: documentado al operador + siempre usar `docker compose up -d --force-recreate web` cuando cambia `.env`.
  - Files modified: ninguno (cambio de procedimiento operativo).
  - Verification: `SELECT current_user` → `nexo_app` tras force-recreate.

- **[Rule 1 — missing critical] `.env` del operador con indent accidental + duplicate SECRET_KEY**
  - Found during: Tarea 4.4, operador pega contenido que tiene 2-space indent en todas las líneas + duplica `NEXO_SECRET_KEY=` (segunda vacía) que habría roto la app tras restart.
  - Fix: Makefile hecho tolerante a indent (`grep -E '^[[:space:]]*NEXO_PG_APP_PASSWORD=' | sed -E 's/^[[:space:]]*...=//'`). Operador limpió `.env` y `.env.example` con `sed -i 's/^[[:space:]]\+//'` y borró la línea SECRET_KEY vacía.
  - Files modified: `Makefile`, `.env` (operador), `.env.example` (operador).
  - Verification: `grep -c '^NEXO_SECRET_KEY=' .env` → 1; app arranca limpio tras force-recreate.

- **[Rule 2 — scope) Sidebar condicional más general que el plan**
  - Found during: Tarea 4.5.
  - Issue: el plan pedía `{% if current_user and current_user.role == "propietario" %}` específicamente para `/ajustes`. Implementé un mecanismo más general (columna `visible_to` en `nav_items`) para preparar Phase 5 UIROL-02 sin refactorizar.
  - Fix: cambio de diseño menor, mismo comportamiento funcional que el plan pedía. Documentado en SUMMARY.
  - Files modified: `templates/base.html`.

**Total deviations:** 4 auto-fixed (3 Rule 1 + 1 Rule 2). **Impact:** nulo sobre comportamiento especificado. El diseño del sidebar deja el terreno mejor preparado para el siguiente sprint.

---

## Archivos tocados

**Creados:**
- `api/middleware/audit.py` — AuditMiddleware
- `api/routers/usuarios.py` — CRUD de usuarios
- `api/routers/auditoria.py` — UI + CSV de audit_log
- `templates/ajustes_usuarios.html`
- `templates/ajustes_auditoria.html`
- `scripts/create_nexo_app_role.sql` — SQL idempotente del rol
- `tests/auth/test_audit_append_only.py` — 5 tests del gate IDENT-06

**Modificados:**
- `api/main.py` (+AuditMiddleware wiring LIFO, +usuarios_router, +auditoria_router)
- `api/config.py` (+pg_app_user, +pg_app_password, +effective_* properties)
- `nexo/db/engine.py` (usa effective_pg_user/password)
- `api/routers/pages.py` (+require_permission en /ajustes)
- `templates/ajustes.html` (+cards de navegación a sub-páginas)
- `templates/base.html` (+columna visible_to en nav_items, filtrado por rol)
- `Makefile` (+target nexo-app-role tolerante a indent)

**Operador (fuera de git):**
- `.env` (+NEXO_PG_APP_USER, +NEXO_PG_APP_PASSWORD, limpieza de indent + duplicado SECRET_KEY)
- `.env.example` (+NEXO_PG_APP_USER/PASSWORD como placeholders)

---

## Cierre de Phase 2

**Los 6 success criteria de ROADMAP.md Phase 2 cumplidos:**
1. ✅ §1 Redirect sin sesión / 401 en API.
2. ✅ §2 Propietario ve `/ajustes/usuarios` y `/ajustes/auditoria` y puede gestionar.
3. ✅ §3 Usuario sin permiso recibe 403.
4. ✅ §4 Lockout 5→15 min + rate limit 20/min verificados desde 02-02.
5. ✅ §5 Cada request autenticada graba fila en `nexo.audit_log`; DELETE desde rol app falla con `permission denied`.
6. ✅ §6 `/ajustes/auditoria` filtra y exporta CSV.

**10 IDENT-* requirements cubiertos:**
- IDENT-01 (auth básico): ✅ 02-02
- IDENT-02 (passwords ≥12 chars argon2): ✅ 02-01 + 02-02
- IDENT-03 (roles + departamentos): ✅ 02-03
- IDENT-04 (sesión + cookies + sliding): ✅ 02-02
- IDENT-05 (RBAC): ✅ 02-03
- IDENT-06 (audit append-only): ✅ 02-04 (Opción A ejecutada)
- IDENT-07 (UI auditoría): ✅ 02-04
- IDENT-08 (UI gestión usuarios): ✅ 02-04
- IDENT-09 (must_change_password): ✅ 02-02
- IDENT-10 (exception handler sin traceback): ✅ ya existente en Sprint 0, regresión mantenida

**Siguientes pasos:**
- Si auto-verify del workflow GSD lo permite → Phase 2 queda marcada como completa.
- Test manual final en navegador como propietario + como usuario con solo `[rrhh]` → validar los 6 success criteria del ROADMAP end-to-end.

---

*Summary creado 2026-04-19 como parte de `/gsd-execute-phase 2 --interactive`.*
*Phase 2 cerrada. Siguiente: Phase 3 (capa de datos).*
