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

*Última revisión: 2026-04-18 (sesión de arranque Sprint 0).*
