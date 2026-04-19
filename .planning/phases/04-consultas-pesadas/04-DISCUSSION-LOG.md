# Phase 4: Consultas pesadas — Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisiones capturadas en CONTEXT.md — este log preserva las alternativas consideradas.

**Date:** 2026-04-19
**Phase:** 04-consultas-pesadas
**Areas discussed:** Umbrales semilla + factores, UX toast amber/red, query_log retención + PII, Approval request semantics, Postflight divergence alerts, Matplotlib concurrency, Hot-reload thresholds, Preflight learning schedule

---

## Umbrales semilla + factores

### Pipeline OEE — warn/block

| Option | Description | Selected |
|--------|-------------|----------|
| warn 60s / block 300s | Pipeline rápido (hardware potente) | |
| warn 120s / block 600s | Defecto medio-conservador | ✓ (via delegación) |
| warn 300s / block 1800s | Hardware modesto | |
| Delego | Planner elige prudente | ✓ (elegido) |

**User's choice:** "Lo decides tú con más contexto" → default warn 120s / block 600s.
**Notes:** Editables desde /ajustes/límites sin tocar código. Sin baseline real preferimos prudencia.

### bbdd/query — warn/block

| Option | Description | Selected |
|--------|-------------|----------|
| warn 1s / block 10s | Estricto | |
| warn 3s / block 30s | Medio | ✓ (via delegación) |
| warn 5s / block 60s | Permisivo | |
| Delego — medio | Start 3s/30s, ajusta vía UI | ✓ (elegido) |

**User's choice:** "Delego — planner elige prudente" → warn 3s / block 30s.

### Capacidad/operarios — trigger de preflight

| Option | Description | Selected |
|--------|-------------|----------|
| 90 días (QUERY-07) | Fiel al requirement | ✓ (via delegación) |
| 30 días | Más agresivo | |
| Siempre (agnóstico rango) | Preflight en todas | |
| Delego — 90d | Alineado con QUERY-07 | ✓ (elegido) |

**User's choice:** "Delego — planner usa 90d (alineado con QUERY-07)".

### factor_por_modulo inicial

| Option | Description | Selected |
|--------|-------------|----------|
| Seed 500ms | Conservador-bajo | |
| Seed 2000ms | Medio | ✓ (via delegación) |
| Seed 5000ms | Alto (hardware modesto) | |
| Delego + reseteable | Seed 2000ms + botón "Recalcular desde últimos 30 runs" en UI | ✓ (elegido) |

**User's choice:** "Delego + añadir título 'reseteable'" → seed 2000ms + UI expone botón de recálculo manual.
**Notes:** Self-tuning no automático; propietario dispara.

---

## UX toast amber/red

### Comportamiento AMBER

| Option | Description | Selected |
|--------|-------------|----------|
| Modal bloqueante con [Continuar]+[Cancelar] | Per-request, sin sticky | ✓ (explícito) |
| Toast no-bloqueante con countdown 5s | Más fluido | |
| Modal + "No volver a preguntar hoy" | Checkbox sticky 8h | |
| Delego — modal bloqueante simple | Planner default | |

**User's choice:** "Modal bloqueante con [Continuar] + [Cancelar]" (elección explícita, no delegada).
**Notes:** El usuario prefirió control explícito sobre fluidez. Rechazó sticky por observabilidad.

### Comportamiento RED

| Option | Description | Selected |
|--------|-------------|----------|
| Modal rojo con [Solicitar aprobación] | Crea request, confirma, sin redirect | ✓ (via delegación) |
| 409 Conflict + redirect | A /ajustes/solicitudes?pending=<id> | |
| Modal con descarga diferida | Programar para la noche (Mark-IV) | |
| Delego — modal con [Solicitar aprobación] | Opción 1 | ✓ (elegido) |

**User's choice:** "Delego — modal con [Solicitar aprobación]" → opción 1 (sin redirect).

### Formato texto modal

| Option | Description | Selected |
|--------|-------------|----------|
| Duración + contexto mínimo | Simple, reasegura | |
| Duración + breakdown cálculo | Útil debug operativo | ✓ (via delegación) |
| Duración + historial reciente | Muestra varianza real | |
| Delego — texto con breakdown | Opción 2 | ✓ (elegido) |

**User's choice:** "Delego — texto con breakdown" → jerarquía: duración destacada + breakdown pequeño + "La UI seguirá respondiendo" (reassurance).

### Persistencia (sticky)

| Option | Description | Selected |
|--------|-------------|----------|
| Per-request siempre | Máx observabilidad | ✓ (via delegación) |
| No durante 8h (sessionStorage) | Una jornada laboral | |
| No durante 30 min (cookie) | Compromiso | |
| Delego — per-request siempre | Más simple y observable | ✓ (elegido) |

**User's choice:** "Delego — per-request siempre". Sin sticky.

---

## query_log retención + PII

### Contenido de `params_json` (bbdd)

| Option | Description | Selected |
|--------|-------------|----------|
| SQL completo + params | Debug-friendly, PII en BD | ✓ (via delegación) |
| SQL hash + metadata | Sin PII, debug difícil | |
| SQL truncado 500 chars | Compromiso medio | |
| Delego — SQL completo + retención corta | Opción 1 + purge 30-90d | ✓ (elegido) |

**User's choice:** "Delego — SQL completo, retención corta" → SQL completo + protección via retención.

### Retención query_log

| Option | Description | Selected |
|--------|-------------|----------|
| 30 días | Mínimo funcional | |
| 90 días (trimestre) | Margen para tendencias | ✓ (via delegación) |
| Forever (sin purga) | PII persiste indefinido | |
| Delego — 90 días con purga automática | Cron Monday 03:00; env var override | ✓ (elegido) |

**User's choice:** "Delego — 90 días con purga automática" → cron semanal + env var NEXO_QUERY_LOG_RETENTION_DAYS.

### Página dedicada `/ajustes/rendimiento`

| Option | Description | Selected |
|--------|-------------|----------|
| Página `/ajustes/rendimiento` dedicada | Filtros + gráfica, propietario | ✓ (explícito) |
| Sin página — query vía /bbdd | Mínimo trabajo | |
| Vista embebida en `/ajustes/límites` | Contextual, últimas 20 filas | |
| Delego — sin página (defer Mark-IV) | Scope tight | |

**User's choice:** "Página `/ajustes/rendimiento` dedicada" (elección explícita, no delegada).
**Notes:** Primera preferencia explícita — observabilidad primera clase aunque requiera trabajo extra.

### Tipo de gráfica

| Option | Description | Selected |
|--------|-------------|----------|
| Línea estimated vs actual timeseries | Muestra tendencia | ✓ (via delegación) |
| Scatter estimated vs actual | Útil calibrar factor | |
| Tabla simple con sparkline | Sin chart lib externa | |
| Delego — línea + tabla | Chart.js CDN + filtros dropdown | ✓ (elegido) |

**User's choice:** "Delego — línea + tabla" → layout con tabla resumen + gráfica línea.

---

## Approval request semantics

### Notificación al propietario

| Option | Description | Selected |
|--------|-------------|----------|
| Badge numérico sidebar | HTMX polling 30s | ✓ (via delegación) |
| Banner top global + badge | Imposible perderse | |
| Email (Mark-IV) | Diferido | |
| Delego — badge sidebar + auto-refresh | Patrón Nexo existente | ✓ (elegido) |

**User's choice:** "Delego — badge sidebar + auto-refresh en /ajustes/solicitudes".
**Notes:** Consistente con patrón existente ('Conexión MES' badge).

### TTL solicitud pendiente

| Option | Description | Selected |
|--------|-------------|----------|
| 24 horas | Fuerza revisión diaria | |
| 7 días | Margen vacaciones | ✓ (via delegación) |
| Infinite (manual purge) | Sin expiración | |
| Delego — 7 días con cron | Expired visibles histórico 30d | ✓ (elegido) |

**User's choice:** "Delego — 7 días con cron de limpieza".

### Semántica force=true

| Option | Description | Selected |
|--------|-------------|----------|
| force=true sin approval_id | Trust, falsificable | |
| force=true + approval_id verificado | Single-use, seguro | ✓ (via delegación) |
| force=true + token firmado JWT | Elegante pero complejo | |
| Delego — approval_id single-use | Opción 2 + link query_log.approval_id | ✓ (elegido) |

**User's choice:** "Delego — force=true + approval_id single-use".

### Cancelación por usuario

| Option | Description | Selected |
|--------|-------------|----------|
| Usuario puede cancelar (UX amable) | Pantalla "Mis solicitudes" | ✓ (explícito) |
| Sólo propietario puede rechazar | Simple, menos UI | |
| Usuario cancela + ve histórico | Máx UX | |
| Delego — sólo propietario decide | Scope tight | |

**User's choice:** "Usuario puede cancelar (UX amable)" (elección explícita, no delegada).
**Notes:** Segunda preferencia explícita — reducir ruido para propietario y dar control al operador.

---

## Postflight divergence alerts

### Canal de alerta (>50% divergence)

| Option | Description | Selected |
|--------|-------------|----------|
| Sólo log.WARNING | Docker logs | |
| log + flag query_log.status='slow' | Pull vía /ajustes/rendimiento | ✓ (via delegación) |
| log + flag + badge sidebar | Observabilidad activa | |
| Delego — log + flag en query_log | Opción 2 | ✓ (elegido) |

**User's choice:** "Delego — log + flag en query_log". Sin push notification (no spam).

---

## Matplotlib concurrency

### Limitación de pipelines paralelos

| Option | Description | Selected |
|--------|-------------|----------|
| Sin límite | Confiamos en LAN | |
| Semáforo global(3) | Protege memoria | |
| Semáforo(3) + timeout 15 min | Kill switch hangs | ✓ (via delegación) |
| Delego — semáforo(3) + timeout 15 min | Env vars configurables | ✓ (elegido) |

**User's choice:** "Delego — semaforo(3) + timeout 15 min" → env vars NEXO_PIPELINE_MAX_CONCURRENT y NEXO_PIPELINE_TIMEOUT_SEC.

---

## Hot-reload thresholds

### Propagación de edición en /ajustes/límites

| Option | Description | Selected |
|--------|-------------|----------|
| Query Postgres por request | Simple, +1 query | |
| Cache in-memory TTL 60s | Menos queries, delay | |
| Cache + invalidate on edit | LISTEN/NOTIFY instantáneo | ✓ (explícito) |
| Delego — cache TTL 60s | Balance | |

**User's choice:** "Cache + invalidate on edit" (elección explícita, no delegada).
**Notes:** Tercera preferencia explícita — robustez sobre simplicidad. LISTEN/NOTIFY + fallback de 5min por si falla.

---

## Preflight learning schedule

### Cuándo recalcular factor_por_modulo

| Option | Description | Selected |
|--------|-------------|----------|
| Automático con cada run | Exponencial suave | |
| Cron semanal | Estable, lento | |
| Botón manual en /ajustes/límites | Control total | |
| Delego — botón manual + cron mensual fallback | Doble safety net | ✓ (elegido) |

**User's choice:** "Delego — botón manual + cron mensual como fallback" → operador controla + cron evita olvido.

---

## Claude's Discretion

Áreas donde el usuario delegó y el planner decide:
- Umbrales semilla (D-01/D-02/D-03) — defaults medio-conservadores editables.
- Factor inicial (D-04) — seed 2000ms con UI reseteable.
- Formato modal (D-07) — jerarquía visual con breakdown.
- Sticky modal (D-08) — per-request.
- Retención query_log (D-10) — 90 días + env var.
- Gráfica tipo (D-12) — Chart.js línea + tabla.
- Notificación approvals (D-13) — badge sidebar + auto-refresh.
- TTL approvals (D-14) — 7 días + cron.
- Semántica force=true (D-15) — approval_id single-use.
- Postflight alerts (D-17) — log + status='slow', sin push.
- Matplotlib concurrency (D-18) — semáforo(3) + timeout 15 min.
- Preflight learning (D-20) — manual + cron mensual fallback.
- Schema detallado de `query_log`, `query_thresholds`, `query_approvals` — tipos e índices.
- Orden de rollout dentro del plan.
- Integración de tests con CI (mocks vs docker compose up db).

## Decisiones explícitas (LOCKED por preferencia usuario, no delegadas)

1. **D-05** Modal AMBER bloqueante con [Continuar] + [Cancelar]. Rechazó toast no-bloqueante y sticky.
2. **D-11** Página dedicada `/ajustes/rendimiento`. Rechazó "sin página, consulta vía /bbdd".
3. **D-16** Usuario puede cancelar solicitud propia. Rechazó "sólo propietario decide".
4. **D-19** Cache + invalidate on edit (LISTEN/NOTIFY). Rechazó cache TTL 60s.

## Deferred Ideas

- Email notification approvals (Mark-IV, depende SMTP).
- Banner global top (reconsiderar si badge insuficiente).
- Websockets para rendimiento en vivo (Mark-IV).
- Rate limiting fino (Mark-IV).
- Sustituir matplotlib (gate: >20% timeouts en 30d prod).
- Paginación /ajustes/rendimiento (Mark-IV si volumen crece).
- Filtros por departamento en /ajustes/solicitudes (Phase 5).
- Alertas Slack/Telegram (Mark-IV).
- Forecast ML sobre query_log (Mark-V).
- Aprobación delegada (Mark-IV).
- Exportar CSV de query_log (nice-to-have).

---

*Discussion completed: 2026-04-19. 4 explícitas + 16 delegadas + 0 pendientes.*
