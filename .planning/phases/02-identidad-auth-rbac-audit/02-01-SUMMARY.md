# Summary — Plan 02-01: Schema + engine + bootstrap

**Phase**: 2 (Identidad — auth + RBAC + audit)
**Plan**: 02-01 of 04
**Ejecutado**: 2026-04-18
**Rama**: `feature/Mark-III`
**Modo**: `/gsd-execute-phase 2 --interactive` checkpoint por hito
**Total commits del plan**: 5 (1 plan-commit + 4 out-of-plan por entorno del operador)

---

## Commits

| # | Hash | Tipo | Mensaje |
|---|------|------|---------|
| 1 | `8b9ab68` | feat | schema Postgres + engine_nexo + bootstrap (core del plan) |
| 2 | `04550dc` | chore | compose db port 5432→host 5433 (colisión en host) |
| 3 | `6aaed06` | chore | compose web port 8000→host 8001 (colisión en host) |
| 4 | `cfb7329` | fix | env_file: .env inyectado al container web |
| 5 | `66523fd` | build | Dockerfile copia nexo/ + scripts/; Makefile targets nexo-* |
| 6 | `4e7d375` | fix | psql targets leen POSTGRES_USER del container db |

---

## Gate duro — resultados

- `make dev` regresión Phase 1: OK.
- `make up` arranca db + web sin errores en logs (`Base de datos inicializada OK`).
- `make nexo-init`: 8 tablas creadas, seeds aplicados (3 roles + 5 depts + 50 permisos), GRANT SELECT/INSERT sobre `audit_log`.
- `make nexo-owner`: `e.eguskiza@ecsmobility.com` creado como `propietario` con `must_change_password=false`.
- `make nexo-verify`: las 8 tablas listadas, 3 roles, 5 departments, 1 user visible.
- `make nexo-smoke`: hash argon2id `$argon2id$v=19$m=65536,t=3,p=4$...`, `verify(ok)=True`, `verify(bad)=False`.

---

## Decisiones tomadas en ejecución

- **Puerto host del db**: 5432→5433 vía `NEXO_PG_HOST_PORT`. Host del operador ya tenía un Postgres nativo escuchando en 5432.
- **Puerto host del web**: 8000→8001 vía `NEXO_WEB_HOST_PORT`. Host del operador tenía otro proceso en 8000. Caddy (LAN HTTPS) sigue llegando al web por la red interna, sin cambio.
- **`.env` injection**: añadido `env_file: .env` al servicio web del compose. Sin ello el container no recibía `NEXO_SECRET_KEY`, aunque `.env` del host lo tuviera.
- **Password del propietario**: `must_change_password=False` intencional para evitar loop redirect (endpoint `/cambiar-password` aún no existe hasta Plan 02-02). El operador deberá cambiarla manualmente tras el primer login post-02-02.

---

## Estado del "Owner Issue" (IDENT-06)

Todas las tablas del schema `nexo` tienen `Owner=oee`, que es el mismo rol que usa la app. El GRANT SELECT, INSERT aplicado por `init_nexo_schema.py` **no** impide UPDATE/DELETE al rol `oee` porque owner siempre tiene todos los privilegios en Postgres.

**Esperado**: el research §Pattern 8 ya anticipó este caso. El gate definitivo IDENT-06 en Plan 02-04 detectará (via test de integración que intenta `DELETE` y espera `permission denied`) y presentará las dos opciones al operador:

- **Opción A**: crear rol `nexo_app` separado (no owner), reasignar `GRANT USAGE ON SCHEMA`, `GRANT SELECT ... ON ALL TABLES`, `GRANT INSERT ON audit_log`.
- **Opción B**: aceptar y diferir a Mark-IV. Aceptable en LAN con trust interno.

---

## Archivos tocados

- **Creados**: `nexo/__init__.py`, `nexo/db/__init__.py`, `nexo/db/engine.py`, `nexo/db/models.py`, `nexo/services/__init__.py`, `nexo/services/auth.py`, `scripts/init_nexo_schema.py`, `scripts/create_propietario.py`.
- **Modificados**: `requirements.txt` (+argon2-cffi, itsdangerous, slowapi), `api/config.py` (+secret_key, session_cookie_name, session_ttl_hours, pg_host, pg_port), `docker-compose.yml` (puertos configurables + env_file), `Dockerfile` (COPY nexo/ + scripts/), `Makefile` (targets nexo-*), `.env.example` (bloque Auth + PG host/port).

---

## Hallazgos propuestos para plans posteriores

- **02-02**: tras cablear `/cambiar-password`, el operador debería cambiar la password del propietario bootstrap (actualmente creada con el password del tiempo de bootstrap).
- **02-04**: gate IDENT-06 debe correr antes de que el audit middleware se considere cerrado. Si fallo → decidir Opción A o B.
- **Phase 3 (Sprint 2)**: unificación de `data/ecs-logo.png` con `static/img/brand/ecs/logo.png`. Sin prisa.
- **Mark-IV**: introducir rol `nexo_app` dedicado para cerrar append-only de forma robusta.

---

## Siguiente

Plan 02-02 — `auth-middleware-login`. AuthMiddleware + `/login` + `/logout` + `/cambiar-password` + cableado de slowapi y del lockout (los servicios ya existen, sólo hay que montarlos).

---

*SUMMARY creado 2026-04-18.*
