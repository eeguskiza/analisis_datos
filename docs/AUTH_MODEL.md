# Nexo · Auth Model

Modelo de identidad, roles y departamentos acordado en la sesión de arranque
de Sprint 0 (2026-04-18). **Se implementa en Sprint 1 (Phase 2)**, no en
Sprint 0. Este documento sirve como contrato para el diseño y como
referencia al escribir código de auth.

Este modelo sustituye al bosquejo `admin / analyst / viewer` de la versión
inicial de `docs/MARK_III_PLAN.md`. Si hay contradicciones entre este doc y
el plan, **gana este doc**.

---

## Principio básico

Cada usuario tiene:

- **Exactamente 1 rol de nivel**: `propietario`, `directivo` o `usuario`.
- **N departamentos**: subconjunto de `{rrhh, comercial, ingenieria, produccion, gerencia}`.

El permiso efectivo sobre un módulo/acción se deriva de:
```
permiso_efectivo(user, modulo, accion) =
    role(user) ∈ roles_permitidos(modulo, accion)
    AND (role(user) == 'propietario' OR departamentos_del_modulo(modulo) ∩ departments(user) ≠ ∅)
```

Es decir: el rol decide el **nivel** de acceso; los departamentos decidan el
**alcance** de los datos visibles. El propietario es la única excepción: ve
todo, global, sin filtro de departamento.

---

## Roles

### `propietario`

- Acceso total a todos los módulos y acciones.
- Ignora cualquier filtro de departamento.
- Único rol que puede:
  - Gestionar usuarios (crear, editar, desactivar, asignar rol y departamentos).
  - Ver el audit log completo sin filtros.
  - Editar umbrales de preflight/postflight (Sprint 3).
  - Aprobar solicitudes de queries rojas (Sprint 3).
- Pensado para **1-2 personas** en ECS (IT lead + operador principal).

### `directivo`

- Acceso total a los módulos de **sus departamentos**.
- Puede leer, ejecutar y editar dentro de sus departamentos; NO puede gestionar usuarios ni ver audit global.
- Ve audit **de sus departamentos** — diferido a Mark-IV, en Mark-III el audit sólo es accesible a `propietario`.
- Pensado para jefes de área, responsables de planta.

### `usuario`

- Operaciones básicas (lectura + ejecución) dentro de sus departamentos.
- No puede:
  - Gestionar usuarios.
  - Borrar ejecuciones ni datos.
  - Acceder a `/bbdd` (explorador SQL) ni a `/ajustes/*`.
- Pensado para el resto de devs, analistas, operarios.

---

## Departamentos

| Código | Nombre | Módulos esperados (Mark-III) |
|--------|--------|------------------------------|
| `rrhh` | Recursos Humanos | operarios (fichajes), informes |
| `comercial` | Comercial | historial, capacidad |
| `ingenieria` | Ingeniería | pipeline, recursos, ciclos, bbdd (lectura), historial |
| `produccion` | Producción | centro_mando (LUK4), pipeline, historial, capacidad |
| `gerencia` | Gerencia | lectura global de informes y centro_mando |

El mapeo módulo → departamentos se materializa en código en la tabla
`nexo.module_departments` (Sprint 1) o en un `dict` centralizado en
`nexo/services/auth.py` — decisión de implementación en Sprint 1.

Un usuario puede pertenecer a **varios** departamentos. Ejemplo: un jefe
de ingeniería con responsabilidad sobre producción tendría
`role=directivo, departments=[ingenieria, produccion]`.

---

## Passwords

- **Algoritmo**: `argon2id` (vía `argon2-cffi`). Parámetros recomendados en Sprint 1 (OWASP 2024): `time_cost=3`, `memory_cost=65536`, `parallelism=4`.
- **Longitud mínima**: 12 caracteres.
- **Reglas adicionales de complejidad**: no se imponen (evita UX fricción innecesaria en LAN interna; la longitud cubre el riesgo principal).
- **Cambio obligatorio al primer login**: sí. Flag `must_change_password` en `nexo.users`. Tras el primer login, el usuario es redirigido a `/cambiar-password` hasta resetear el flag.
- **Expiración forzada**: no. Nunca obliga a cambiar password por tiempo. Evita tickets de soporte y rotaciones triviales.
- **Recuperación**: en Mark-III, propietario resetea password del usuario desde `/ajustes/usuarios`. Sin flujo "olvidé mi contraseña" por email (SMTP está roto hasta decisión de infraestructura).

---

## Bloqueo progresivo

- **Regla única**: **5 intentos fallidos consecutivos → 15 minutos lock** sobre la tupla `(user, IP)`.
- Contador por `(user, IP)` en tabla `nexo.login_attempts` con TTL de 15 min tras último fallo o tras login exitoso.
- El lock es por tupla: un atacante que prueba 10 000 usuarios desde una IP saturará la IP; un atacante distribuido que cambia IP seguirá activo — aceptable en LAN-only.
- No hay escalado progresivo (3→30s / 5→5min / 10→manual del MARK_III_PLAN.md v0 — descartado por complejidad sin beneficio claro).

---

## Sesión

- Cookie HttpOnly + Secure + SameSite=Lax.
- Tiempo de vida: 12 horas por defecto; renovable por actividad (sliding expiration).
- Decisión de implementación (Sprint 1): cookie firmada con `itsdangerous` o JWT — decisión en Sprint 1.

---

## Lo que **NO** entra en Mark-III

- **2FA** (TOTP, WebAuthn) — Mark-IV.
- **LDAP / Active Directory** — Mark-IV. Usuarios locales en Postgres durante Mark-III.
- **Audit filtrado por departamento visible a directivo** — Mark-IV. En Mark-III sólo propietario accede al audit.
- **Flujo "olvidé mi contraseña" por email** — depende de SMTP configurado (Out of Scope Mark-III).
- **Rotación automática de credenciales SQL Server usadas por la app** — Mark-IV.

---

## Tablas Postgres relacionadas (schema `nexo`)

Especificación detallada en el PLAN.md de Phase 2. Resumen del modelo:

| Tabla | Propósito |
|-------|-----------|
| `nexo.users` | usuarios + hash + rol + departamentos + flags |
| `nexo.roles` | catálogo de roles (`propietario`, `directivo`, `usuario`) |
| `nexo.departments` | catálogo de departamentos |
| `nexo.user_departments` | N:M entre users y departments |
| `nexo.permissions` | catálogo de permisos (`modulo:accion`) con role+departamento |
| `nexo.login_attempts` | contadores para bloqueo progresivo |
| `nexo.sessions` | sesiones activas (si no se usa JWT stateless) |
| `nexo.audit_log` | append-only; middleware escribe cada request autenticada |

---

*Última revisión: 2026-04-19 (apéndice PERMISSION_MAP tras Plan 02-03).*

---

## Apéndice: PERMISSION_MAP (Phase 2 / Sprint 1)

**Fuente de verdad:** `nexo/services/auth.py::PERMISSION_MAP`.

Esta tabla es una instantánea al cierre de Plan 02-03 para trazabilidad.
Cualquier cambio se hace en código; este apéndice se regenera en milestones
posteriores (p. ej. Phase 5 cuando se añada filtrado de sidebar por rol).

**Regla invariante:** el rol `propietario` **NO aparece** en el mapa.
Tiene bypass hardcodeado en `require_permission()` — ignora departamentos y
ve todos los módulos.

**Convención:** valor `[]` (lista vacía) = "solo propietario".

| Permiso | Departamentos autorizados | Routers / endpoints que lo aplican |
|---------|---------------------------|------------------------------------|
| `pipeline:read`      | ingenieria, produccion, gerencia | router `api/routers/pipeline.py` |
| `pipeline:run`       | ingenieria, produccion           | `POST /api/pipeline/run` (estricto) |
| `recursos:read`      | ingenieria, produccion           | router `api/routers/recursos.py` |
| `recursos:edit`      | ingenieria                       | `PUT`/`POST`/auto-detectar en recursos |
| `ciclos:read`        | ingenieria                       | router `api/routers/ciclos.py` |
| `ciclos:edit`        | ingenieria                       | `PUT`/`POST`/`DELETE`/sync-csv en ciclos |
| `centro_mando:read`  | produccion, ingenieria, gerencia | router `api/routers/centro_mando.py` (prefix `/api/dashboard`) |
| `luk4:read`          | produccion, ingenieria, gerencia | router `api/routers/luk4.py` |
| `capacidad:read`     | comercial, ingenieria, produccion, gerencia | router `api/routers/capacidad.py` |
| `historial:read`     | ingenieria, produccion, comercial, gerencia, rrhh | router `api/routers/historial.py` |
| `informes:read`      | ingenieria, produccion, comercial, gerencia, rrhh | router `api/routers/informes.py` |
| `informes:delete`    | ingenieria                       | `DELETE /api/informes/{date_str}` (estricto) |
| `datos:read`         | ingenieria, produccion           | router `api/routers/datos.py` |
| `operarios:read`     | rrhh                             | router `api/routers/operarios.py` |
| `operarios:export`   | rrhh                             | reservado — sin endpoint asignado en Mark-III |
| `bbdd:read`          | ingenieria                       | router `api/routers/bbdd.py` (explorador SQL completo) |
| `conexion:read`      | ingenieria                       | router `api/routers/conexion.py` |
| `conexion:config`    | **[]** (solo propietario)        | `PUT /api/conexion/config` (estricto, toca credenciales) |
| `email:send`         | rrhh, ingenieria, gerencia       | router `api/routers/email.py` |
| `plantillas:read`    | ingenieria                       | router `api/routers/plantillas.py` |
| `plantillas:edit`    | ingenieria                       | `PUT`/`POST`/`DELETE` en plantillas |
| `ajustes:manage`     | **[]** (solo propietario)        | reservado — Plan 02-04 lo cableará a `/ajustes/usuarios` |
| `auditoria:read`     | **[]** (solo propietario)        | reservado — Plan 02-04 lo cableará a `/ajustes/auditoria` |
| `usuarios:manage`    | **[]** (solo propietario)        | reservado — Plan 02-04 lo cableará a CRUD de users |

**Total:** 24 permisos. **Routers retrofit:** 14 (+ `api/routers/health.py` whitelisted en `AuthMiddleware`). **Ocurrencias de `require_permission` en `api/routers/`:** 34 (verificado post-plan 02-03).

### Anti-patrones documentados

- **NO** añadir `propietario` como valor en ninguna lista del mapa. El bypass es por rol, antes del lookup. Fuente: research §Anti-Patterns.
- **NO** consultar `nexo.permissions` en runtime. Esa tabla es catálogo seed para Phase 5; el dict en código manda. Fuente: research §Stack Decision 2.
- **NO** añadir decorator `@require_permission(...)` a nivel de función — usar `dependencies=[Depends(require_permission("..."))]` en el `APIRouter()` o en el endpoint. La factory devuelve un callable async que FastAPI inyecta como dependency.

### Pendientes declarados

- **Filtrado de sidebar por rol** (UI): la navegación lateral sigue mostrando todos los items para todos los roles; los que no tengan permiso verán errores/vacío en los paneles. Resolución en Phase 5 UIROL-02.
- **Homepage por rol**: un `usuario` solo-rrhh entra en `/` y ve LUK4 vacío. Resolución en Phase 5 UIROL-02.
- **Handler global de 403 en JS**: las fetches del front no muestran toast específico al recibir 403 — falla silenciosamente o muestra "Error de red". Phase 5 UIROL-02 lo resuelve como parte del rediseño por rol.
- **Endpoint de export en `operarios`**: el permiso `operarios:export` está declarado pero no aplicado (no hay endpoint de export todavía). Se cableará cuando el endpoint aterrice.
- **Historial mutaciones** (`DELETE /api/historial/{id}`, `POST /api/historial/{id}/regenerar`): quedan bajo `historial:read` por falta de `historial:edit`/`:delete` en el mapa. Iteración futura si se quiere tightening.
- **Tests `@pytest.mark.integration` non-blocking en CI**: el job `test` de GitHub Actions tiene `continue-on-error: true` hasta Phase 7.

*Apéndice generado 2026-04-19 como parte de `/gsd-execute-phase 2 --interactive` → Plan 02-03.*
