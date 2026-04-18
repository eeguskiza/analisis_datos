# Security Audit — historial git

**Fecha**: 2026-04-18
**Rama auditada**: `feature/Mark-III` (incluye todo el historial de `main`, `feature/Mark-II` y anteriores).
**Scope**: localización de credenciales potencialmente expuestas en el historial.
**Método**: `git log --all --full-history -p -- <patrones_sensibles>` + regex `password|secret|token|api_key` con valores redactados antes de cargarlos a contexto.

**Regla de operación**: este documento enumera hallazgos. **No se ejecuta
`git filter-repo` ni `git push --force`**. Cada item se marca
`REVIEW PENDING` hasta que el operador (Erik) decida si el valor
commiteado era placeholder o credencial real; sólo tras esa revisión se
decide rotación y/o reescritura de historial.

---

## Resumen ejecutivo

| Nivel | Recuento | Acción recomendada |
|-------|----------|---------------------|
| 🔴 **Probablemente real** | 1 commit (`b0e80b9`) | Revisar valor; si real, rotar credencial MES. Considerar filter-repo. |
| 🟠 **A verificar** | 2 commits (`d1b4339`, `f22d689`) | Revisar valores de `POSTGRES_PASSWORD` en `docker-compose.yml`. |
| 🟡 **Presumiblemente placeholder** | 3 commits (`3007dc5`, `03f3992`, `d1b4339` .env.example) | Revisar que sean placeholders (`tu_password`, `changeme`, etc.). |
| 🟢 **No credenciales** | 5 archivos `:Zone.Identifier` | Ruido NTFS/WSL. Se borran en commit 2 del Sprint 0. |

**Total de hallazgos con patrón sensible**: 12 líneas (7 `password` + 5 `PASSWORD`), repartidas en **5 commits históricos** y **3 archivos actualmente trackeados** (`.env.example`, `docker-compose.yml`, `data/db_config.example.json`).

---

## 🔴 Hallazgos críticos — probable credencial real

### H1 — `data/db_config.json` @ commit `b0e80b9` (2026-03-16)

- **Commit**: `b0e80b9` — mensaje: *"Cosas"*
- **Archivo**: `data/db_config.json`
- **Tipo**: clave `"password": "..."` en JSON de configuración MES (IZARO / SQL Server `dbizaro`).
- **Contexto**: este fichero se trackeó un tiempo antes de moverse a `.gitignore`. El campo `password` es la **credencial del usuario de lectura de MES** (muy probablemente `sa` o equivalente de SQL Server `192.168.0.4:1433`). Coincide con el driver ODBC que hoy se sigue usando.
- **Severidad**: **crítica** si el valor es real — esa password da acceso de lectura a toda `dbizaro` y `ecs_mobility` (misma instancia).
- **Estado**: REVIEW PENDING.
- **Preguntas para el operador**:
  1. ¿El valor commiteado en `b0e80b9` es la password real actual?
  2. Si sí, ¿queremos rotar (fuera de Sprint 0, decisión del operador) o mantener y reescribir historial con `filter-repo`?
  3. ¿El repo tiene forks públicos o clones en equipos de terceros? Si sí, la rotación es obligatoria porque el historial ya se replicó.

**Comando de inspección (manual, valor no cargado en contexto)**:
```
git show b0e80b9 -- data/db_config.json
```

---

## 🟠 Hallazgos a verificar — `POSTGRES_PASSWORD` en docker-compose

### H2 — `docker-compose.yml` @ commits `f22d689` y `d1b4339`

- **Commit `f22d689` (2026-03-31)** — mensaje: *"Reestructurar app: FastAPI + Docker + PostgreSQL + nueva interfaz"*
  - Añade `POSTGRES_PASSWORD` en `docker-compose.yml` (primera vez).
- **Commit `d1b4339` (2026-04-10)** — mensaje: *"Credenciales Seguras"*
  - Reemplaza el valor de `POSTGRES_PASSWORD` (line delete + line add) y añade `OEE_DATABASE_URL` con credencial incrustada.
- **Archivo actual** (`docker-compose.yml:7`): `POSTGRES_PASSWORD: ${OEE_PG_PASSWORD:-oee}` — interpolado desde env var con default `oee`. **El default `oee` es un valor trivial conocido**. Si `OEE_PG_PASSWORD` no está definida en el servidor LAN, se usa `oee` tanto para crear la BD como para conectarse — no es secreto pero sí expone una BD Postgres local con credencial predecible.
- **Preguntas para el operador**:
  1. ¿El valor de `POSTGRES_PASSWORD` introducido en `f22d689` era un placeholder (`oee`, `changeme`) o una credencial real?
  2. ¿El `OEE_DATABASE_URL` añadido en `d1b4339` contenía credenciales embebidas? Revisar el diff.
- **Severidad**: media. Postgres local es la casa futura de `nexo.*` en Mark-III (users, audit); hoy no se usa en runtime. Aun así, el default `oee` queda pendiente de reforzar en Sprint 5 (deploy LAN) o antes si se decide.
- **Estado**: REVIEW PENDING.

**Comando de inspección**:
```
git show f22d689 -- docker-compose.yml
git show d1b4339 -- docker-compose.yml
```

---

## 🟡 Hallazgos presumiblemente placeholder — revisar brevemente

### H3 — `.env.example` @ commits `d1b4339` y `03f3992`

- **Commit `d1b4339` (2026-04-10)** — añade `OEE_DB_PASSWORD`, `OEE_PG_PASSWORD`.
- **Commit `03f3992` (2026-03-11)** — añade `DB_PASSWORD` (prefijo viejo).
- **Estado actual de `.env.example`** (líneas 11, 19): contiene `OEE_DB_PASSWORD=...` y `OEE_PG_PASSWORD=...`. Los valores no se pueden leer desde Claude Code por la regla de permisos (`.env*` está en denyList); el operador debe verificar manualmente.
- **Expectativa**: un `.env.example` correcto contiene placeholders (`tu_password`, `changeme`, `<your-password-here>`), **nunca** credenciales reales.
- **Acción**: Erik revisa `.env.example` y confirma que los valores son placeholders. Si encuentra credenciales reales, sustituirlos por placeholders y añadir el archivo al informe como crítico.
- **Estado**: REVIEW PENDING.

### H4 — `data/db_config.example.json` @ commit `3007dc5` (2026-03-11)

- **Commit**: `3007dc5` — mensaje: *"Simplificar interfaz: credenciales solo en fichero, añadir test de conexión"*.
- **Archivo**: `data/db_config.example.json` — trackeado actualmente.
- **Patrón**: `"password": ...` + texto instructivo *"Rellena los campos server, user y password con tus datos reales."* — el texto instructivo sugiere que el fichero **es un ejemplo**, no producción.
- **Expectativa**: el valor de `"password"` en este JSON es placeholder (`""` vacío o `"YOUR_PASSWORD_HERE"`).
- **Acción**: Erik revisa `data/db_config.example.json` y confirma.
- **Estado**: REVIEW PENDING.

---

## 🟢 Residuos no-credencial — `:Zone.Identifier` (NTFS/WSL)

Archivos metadata de Windows NTFS que se trackearon por error al copiar ficheros desde Windows a WSL. Contienen típicamente la URL de origen (MarkOfTheWeb); no son credenciales per se pero son **ruido, revelan filesystem origen y deben borrarse**.

- `.env:Zone.Identifier` — commit histórico de cuando `.env` se trackeó brevemente o por accidente. (`.env` en sí ya está en `.gitignore` actualmente.)
- `data/db_config.json:Zone.Identifier` — metadato de `db_config.json`.
- `Pabellon2.pdf:Zone.Identifier`
- `Pabellon3.pdf:Zone.Identifier`
- `Pabellon4.pdf:Zone.Identifier`

**Acción**: **se borran del tracking en Sprint 0 commit 2 y 3** (esta misma fase). `.gitignore` añade patrón `*:Zone.Identifier` para prevenir recurrencia.

---

## Archivos no trackeados hoy pero presentes en historial

| Archivo | Estado actual | En historial |
|---------|---------------|--------------|
| `data/db_config.json` | ✓ en `.gitignore` | ⚠️ commit `b0e80b9` (H1) |
| `.env:Zone.Identifier` | 🗑️ pendiente borrado (Sprint 0 commit 2) | ⚠️ historial |
| `data/db_config.json:Zone.Identifier` | 🗑️ pendiente borrado | ⚠️ historial |

---

## Qué NO hace este audit

- **NO ejecuta `git filter-repo`**. La reescritura de historial es una decisión del operador (Erik), no automatizable.
- **NO ejecuta `git push --force`**. Nunca.
- **NO carga valores literales de credenciales a contexto del agente**. Todos los hallazgos se procesaron con valores redactados.
- **NO rota credenciales**. La rotación SQL Server está explícitamente diferida fuera de Sprint 0 (`docs/OPEN_QUESTIONS.md` decisión 2).

---

## Decisiones pendientes del operador

Para cada hallazgo 🔴 y 🟠:

1. **Verificar manualmente el valor** con `git show <hash> -- <archivo>`.
2. **Si es placeholder**: marcar el hallazgo como RESOLVED — CLEAN en este documento.
3. **Si es credencial real**:
   a. **Rotar la credencial** en el sistema correspondiente (SQL Server MES, Postgres local, SMTP, etc.).
   b. Decidir si además quieres **reescribir historial**:
      - **Pros**: la credencial desaparece del log público si se pushea de nuevo.
      - **Cons**: rompe hashes, clones existentes tienen que re-clonar, CI/hooks externos pueden referenciar hashes antiguos.
      - **Herramienta recomendada**: `git filter-repo --invert-paths --path data/db_config.json` (o por patrones), revisar resultado en clon aislado, push con `--force-with-lease` tras coordinar con cualquier clon externo.

---

## Estado global del audit

| Hallazgo | Severidad | Estado |
|----------|-----------|--------|
| H1: `data/db_config.json` @ `b0e80b9` | 🔴→🟢 Crítica → RESOLVED | CONFIRMED real por operador; password **rotada** post-audit el 2026-04-18. Valor en historial ya no es credencial válida. `filter-repo` sigue diferido como limpieza cosmética. |
| H2: `docker-compose.yml` @ `f22d689`, `d1b4339` | 🟠→🟢 Media → RESOLVED | Valor era el mismo password de SA (confirmado por contexto). Rotación de H1 cubre también este hallazgo. |
| H3: `.env.example` @ `d1b4339`, `03f3992` | 🟡 | Actualizado en commit 6 (pin env + NEXO_*). `.env.example` actual tiene placeholders (`tu_password`). Queda sólo como registro histórico. |
| H4: `data/db_config.example.json` @ `3007dc5` | 🟡 | Archivo de ejemplo con texto "rellena con datos reales" — placeholder, sin credencial. Sin acción. |
| `:Zone.Identifier` (5 archivos) | 🟢 No credenciales | Borrados en Sprint 0 commit 2 (`dec03d1`). Patrón `*:Zone.Identifier` añadido a `.gitignore` en commit 3 (`bfe35e7`). |

---

## Operator Review — 2026-04-18 post-Sprint-0

- **Repo hosting**: GitHub privado dentro de org privada ECS-Mobility. Audiencia limitada a miembros de la org; no público.
- **H1 confirmado real por operador** al inspeccionar `git show b0e80b9 -- data/db_config.json`: era la password activa del usuario `sa` de SQL Server `192.168.0.4:1433`.
- **Rotación ejecutada** por el operador tras ver el audit. La password que aparece en el historial git ya no abre la BD.
- **Nota adicional**: durante la sesión de ejecución de Sprint 0 el valor de la password fue pegado en el chat con Claude Code para confirmar el hallazgo, por lo que viajó también por la API de Anthropic. La rotación post-audit cubre también esa exposición.
- **`git filter-repo` no ejecutado**. Al quedar la credencial muerta, reescribir historial es cosmético. Decisión diferida; si en Mark-IV se decide limpiar, hacerlo entonces coordinando con los clones existentes (2 máquinas locales de Erik).

---

*Audit generado el 2026-04-18 como Sprint 0 commit 1 (Gate 1). Resoluciones anotadas en commit 13 del mismo sprint tras ejecución.*
