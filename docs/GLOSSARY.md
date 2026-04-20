# Glosario — Nexo (Mark-III)

Términos de dominio recurrentes en el código, los docs y las
conversaciones de planning. Primer destinatario: dev o IA nueva en el
repo que se encuentra un término y no sabe qué quiere decir aquí.

---

## Producto

- **Nexo** — plataforma interna de ECS Mobility. Sucesora de "OEE Planta / analisis_datos". Mismo stack, mismo repo, mismo equipo; Mark-III es rebrand + refactor estructural, no reescritura. Corre en LAN de ECS, sin exposición a internet.
- **ECS Mobility** — empresa propietaria de Nexo. Aparece como marca corporativa (footer, docs, emails). Distinto de "Nexo": Nexo es el producto; ECS Mobility es la empresa.
- **Mark-III / Mark-IV / Mark-V** — milestones de Nexo. Mark-III es el alcance actual (auth + datos + queries pesadas + UI roles + deploy LAN + DevEx). Ver `docs/MARK_III_PLAN.md`.

## Bases de datos

- **MES** (Manufacturing Execution System) — sistema externo de IZARO que gestiona la planta. Nexo lee de él en modo read-only. En código vive como BD `dbizaro` en SQL Server (`192.168.0.4:1433`). Variables de entorno: `NEXO_MES_*` (antiguo: `OEE_IZARO_DB` + `OEE_DB_*`).
- **APP** — BD propia de Nexo en SQL Server: `ecs_mobility`. Contiene los schemas `cfg`, `oee`, `luk4`. Variables de entorno: `NEXO_APP_*` (antiguo: `OEE_DB_*`).
- **dbizaro** — nombre físico de la BD MES en SQL Server.
- **ecs_mobility** — nombre físico de la BD APP en SQL Server.
- **cfg.*** — schema en `ecs_mobility` con tablas de configuración: `cfg.recursos`, `cfg.ciclos`, `cfg.contactos`. Datos de negocio compartidos con Power BI y el túnel IoT; se quedan en SQL Server (no migran a Postgres en Mark-III).
- **oee.*** — schema en `ecs_mobility` con datos de producción: `oee.ejecuciones`, `oee.datos`, `oee.metricas`, `oee.referencias`, `oee.incidencias`, `oee.informes`.
- **luk4.*** — schema en `ecs_mobility` para el panel del pabellón 5: `luk4.estado`, `luk4.tiempos_ciclo`, `luk4.alarmas`, `luk4.plano_zonas`.
- **Postgres / `nexo.***` — casa nueva para `users`, `roles`, `permissions`, `audit_log`, `query_log`, `login_attempts`, `sessions`. Runtime en contenedor `db`, schema `nexo`. Variables de entorno: `NEXO_PG_*`.
- **IZARO** — sistema de información de planta del que se extraen los datos. Tablas relevantes: `fmesdtc`, `fmesddf`, `fmesinc`, `fprolof`, `fmesrec`, `fmesope`, `fmesmic`, `zmeshva`.

## Arquitectura

- **Módulo** — unidad funcional del producto visible en la UI (sidebar): `pipeline`, `historial`, `bbdd`, `capacidad`, `operarios`, `recursos`, `ciclos`, `luk4`, `ajustes`, `centro_mando`, `datos`.
- **Pipeline** — flujo que extrae datos de MES, los persiste en `oee.datos`, y genera PDFs por módulo (disponibilidad, rendimiento, calidad, oee_secciones) vía matplotlib.
- **Módulos OEE** — subcarpetas de `OEE/` que calculan cada componente del OEE: `disponibilidad`, `rendimiento`, `calidad`, `oee_secciones`. No se renombran a `modules/` en Mark-III (diferido a Sprint 2).
- **engine_mes / engine_app / engine_nexo** — SQLAlchemy engines separados que se introducen en Sprint 2 (capa de datos). Hoy el código usa un único engine SQLAlchemy más pyodbc suelto.

## Auth & permisos

Ver `docs/AUTH_MODEL.md` para detalle.

- **Rol** — nivel de acceso del usuario. En Nexo hay tres: `propietario`, `directivo`, `usuario`. Un usuario tiene exactamente **uno**.
- **Departamento** — área funcional de la empresa. En Nexo: `rrhh`, `comercial`, `ingenieria`, `produccion`, `gerencia`. Un usuario tiene **N** departamentos.
- **propietario** — rol global. Acceso total ignorando departamento. Único que gestiona usuarios y ve el audit log completo.
- **directivo** — rol de área. Acceso total a los módulos de sus departamentos.
- **usuario** — rol básico. Operaciones de lectura y ejecución dentro de sus departamentos.
- **audit_log** — tabla Postgres `nexo.audit_log` append-only que registra cada request autenticada. Nunca se borra filas (REVOKE UPDATE, DELETE al rol app). Sprint 1.
- **login_attempts** — tabla `nexo.login_attempts` para el bloqueo progresivo (5 intentos → 15 min lock `(user, IP)`). Sprint 1.

## Performance / ops

- **Preflight** — estimación **antes de ejecutar** una operación cara (pipeline OEE, query `/bbdd`, capacidad de 180 días, export CSV masivo). Devuelve `Estimation(ms, level, reason)` con `level ∈ {green, amber, red}`. Sprint 3.
- **Postflight** — medición **después de ejecutar** que graba `actual_ms` en `nexo.query_log` y alerta si excede `warn_ms × 1.5`. Sprint 3.
- **query_log** — tabla `nexo.query_log` con una fila por ejecución medida (endpoint, user, params, estimated_ms, actual_ms, rows, status). Sprint 3.
- **query_thresholds** — tabla `nexo.query_thresholds` con umbrales editables por endpoint (`warn_ms`, `block_ms`). Sprint 3.
- **estimated_ms** — coste estimado (en ms) por el preflight antes de ejecutar. Se calcula con heurística `n_recursos × n_días × factor_ms` para pipeline, o `factor_ms × n_días` para capacidad/operarios. Se persiste en `nexo.query_log.estimated_ms`.
- **actual_ms** — coste real (en ms) medido por el middleware `query_timing` al terminar la request. Se persiste en `nexo.query_log.actual_ms`. Phase 4 / Plan 04-02.
- **slow_query** — ejecución donde `actual_ms > warn_ms × 1.5`. Se marca con `status='slow'` en `query_log` y dispara `log.warning` (D-17). Propietario las ve filtrando en `/ajustes/rendimiento`.
- **divergence_pct** — `(avg_actual - avg_estimated) / avg_estimated × 100` por endpoint en una ventana temporal. >50% = factor mal calibrado (recalcular vía `/ajustes/limites`). 0 si `avg_estimated` es nulo.
- **thresholds_cache** — cache in-memory de `nexo.query_thresholds` (módulo `nexo/services/thresholds_cache.py`). D-19: `LISTEN/NOTIFY` propaga edits en <1s cross-worker; safety-net refresca si `loaded_at > 5 min` si el listener cae. Phase 4 / Plans 04-01 + 04-04.
- **factor_auto_refresh** — cron que el 1er Monday de cada mes (03:10 UTC) recalcula `factor_ms` por endpoint si `factor_updated_at > NEXO_AUTO_REFRESH_STALE_DAYS` (default 60d). Doble safety-net sobre el botón manual "Recalcular" (D-04 / D-20).
- **query_log_cleanup** — job Monday 03:00 UTC que borra filas de `nexo.query_log` con `ts < now() - NEXO_QUERY_LOG_RETENTION_DAYS` (default 90d; 0 = forever). Graba audit_log con `path='__cleanup_query_log__'` (D-10).
- **SSE** (Server-Sent Events) — mecanismo de streaming que usa `/api/pipeline/run` para empujar progreso al navegador mientras corre el pipeline.

## Infraestructura

- **LAN-only** — Nexo no se expone a internet. Sólo accesible desde la red interna de ECS (`192.168.*.*`). Decisión de producto cerrada.
- **Caddy** — reverse proxy HTTPS delante de la app. Hoy con cert autofirmado (`tls internal`); Sprint 5 decide si pasa a Let's Encrypt DNS-01 o se queda interno.
- **Compose profile `mcp`** — el servicio `mcp` está aparcado en este profile. `make up` / `make dev` no lo arrancan; `docker compose --profile mcp up -d mcp` sí.
- **MCP** (Model Context Protocol) — servidor stdio que expone 15 tools read-only sobre la API de Nexo a Claude Code durante el desarrollo. ID en Mark-III: `nexo-mcp` (antes: `oee-planta`).

## Gestión del proyecto

- **GSD** (Get Shit Done) — framework de planning que usamos para organizar Mark-III. Comandos relevantes: `/gsd-progress`, `/gsd-plan-phase N`, `/gsd-execute-phase N`. Ver `CLAUDE.md`.
- **Phase** — unidad de trabajo de GSD. Phase N de Mark-III ≈ Sprint (N-1) de `docs/MARK_III_PLAN.md`.
- **Plan** — fichero `NN-MM-PLAN.md` dentro de `.planning/phases/NN-slug/`. Lista de tareas y commits de una phase.
- **Gate** — punto duro de interrupción dentro de un PLAN. Gate 1 del Sprint 0 = audit del historial git antes de cualquier modificación de código.
- **Strangler-fig** — estrategia de refactor de monolito: añades la nueva arquitectura pieza a pieza en el mismo repo, manteniendo la vieja funcionando, hasta que puedas eliminar la vieja. Lo opuesto a "big bang rewrite".
- **Proyección derivada** — los archivos `.planning/PROJECT.md`, `REQUIREMENTS.md` y `ROADMAP.md` se derivan de `docs/MARK_III_PLAN.md`, `docs/OPEN_QUESTIONS.md` y `docs/CURRENT_STATE_AUDIT.md`. No se editan a mano. Ver `CLAUDE.md` sección "Regeneración de `.planning/`".

---

*Glosario creado 2026-04-18 como Sprint 0 commit 12. Se actualiza cuando aparece terminología nueva.*
