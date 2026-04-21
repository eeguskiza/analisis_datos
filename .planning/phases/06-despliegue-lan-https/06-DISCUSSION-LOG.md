# Phase 6: Despliegue LAN HTTPS — Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-21
**Phase:** 06-despliegue-lan-https
**Areas discussed:** Cert HTTPS, Backup pgdata + recuperación, Firewall + acceso SSH, Compose prod + resource limits, Healthchecks/deploy/SMTP

---

## Cert HTTPS

### Q1: Hostname interno para Nexo

| Option | Description | Selected |
|--------|-------------|----------|
| nexo.ecsmobility.local | TLD .local estándar redes internas | ✓ |
| nexo.ecsmobility.lan | Variante .lan, menos estándar | |
| nexo.ecsmobility.internal | RFC 8375 reserva .internal | |
| Otro | Usuario decide | |

**Notes:** Usuario quiere hostname fijo y bookmarkable que no cambie.

### Q2: Estrategia certificado HTTPS (segunda iteración tras descartar LE)

| Option | Description | Selected |
|--------|-------------|----------|
| Let's Encrypt DNS-01 | Cert válido sin warnings | rechazado en Q intermedia |
| Cert firmado por CA corporativa ECS | IT emite cert, sin friction usuario | |
| tls internal + distribuir root CA | Caddy CA autofirmada + 1 instalación por equipo | ✓ |
| Aceptar warnings (sin distribuir root) | Browser warning permanente | |

**Notes:** Primera respuesta del usuario eligió LE DNS-01, pero Q complementaria reveló "no tenemos ese dominio para esto" → LE descartado por incompatibilidad con dominio público no controlado.

### Q3: DNS resolution para hostname interno

| Option | Description | Selected |
|--------|-------------|----------|
| hosts-file por usuario + nota router como mejora futura | Cada equipo añade 1 línea, ship sin depender de IT | ✓ |
| Esperar DNS interno | Phase 6 bloqueada hasta IT/router | |
| Solo IP estática sin hostname | Bookmarks por IP, cert internal nunca válido | |

**Notes:** Usuario explicó: "estamos desarrollando esto en mi pc pero la idea es lanzar este contenedor en una maquina con ubuntu y un ip estatica". Sin infra DNS conocida — hosts-file es la solución pragmática.

### Q4: IP estática del Ubuntu

| Option | Description | Selected |
|--------|-------------|----------|
| IP literal (la doy ahora) | Aparece directamente en docs | |
| Placeholder `<IP_NEXO>` | Operador reemplaza antes de deploy | ✓ |

**Notes:** IP aún no asignada — máquina física por preparar.

---

## Backup pgdata + recuperación

### Q1: Estrategia de backup Postgres

| Option | Description | Selected |
|--------|-------------|----------|
| pg_dump diario via cron + retención 7d | Backup lógico portable, restaurable a cualquier PG16+ | ✓ |
| Snapshot del volumen Docker (rsync) | Bit-exact pero requiere downtime | |
| Sin backup (solo SSD) | NO recomendado para Nexo (audit_log sensible) | |

### Q2: Ubicación de backups

| Option | Description | Selected |
|--------|-------------|----------|
| Mismo Ubuntu, /var/backups/nexo/ | Sin dependencia externa, ship-able | ✓ |
| NAS / equipo externo via scp/rsync | Protege de muerte SSD pero requiere infra | |
| Ambos (local + sync externo) | Máxima resiliencia, requiere infra externa | |

### Q3: RPO aceptable

| Option | Description | Selected |
|--------|-------------|----------|
| Hasta 24h (1 backup nocturno) | Aceptable para uso laboral | ✓ |
| Hasta 1h (backups cada hora) | Más presión disco | |
| Prácticamente cero (WAL archiving) | Sobre-engineering LAN | |

### Q4: RTO objetivo

| Option | Description | Selected |
|--------|-------------|----------|
| 1-2 horas con runbook claro | Checklist seguible por admin con SSH | ✓ |
| Mismo día sin presión | Sin SLA | |
| Best effort | Sin compromiso | |

### Q5: Owner del runbook

| Option | Description | Selected |
|--------|-------------|----------|
| e.eguskiza único responsable | Bus factor 1 | |
| e.eguskiza + 1 persona respaldo | Bus factor 2, runbook accesible para admin con Linux básico | ✓ |
| IT / cualquier admin SSH | Runbook 100% paso a paso, requiere IT acepte ownership | |

---

## Firewall + acceso SSH

### Q1: SSH (puerto 22)

| Option | Description | Selected |
|--------|-------------|----------|
| Toda la LAN (subred del Ubuntu) | Pragmático, LAN ya controlada por perimetral | ✓ |
| Solo IPs admin específicas (whitelist) | Más seguro, más fricción mantenimiento | |
| SSH cerrado, solo consola física | Máximo aislamiento | |

### Q2: Puerto 80

| Option | Description | Selected |
|--------|-------------|----------|
| Redirect automático a 443 (Caddy default) | UX más suave | ✓ |
| Deny puerto 80 totalmente | "Connection refused" para http:// | |

### Q3: Otros puertos

| Option | Description | Selected |
|--------|-------------|----------|
| Solo 22/80/443, deny all else | Estricto | ✓ |
| Sí, casos específicos | Lista a aportar | |

---

## Compose prod + resource limits

### Q1: Estructura compose prod

| Option | Description | Selected |
|--------|-------------|----------|
| docker-compose.prod.yml override | Mantiene dev intacto, override solo lo que cambia | ✓ |
| profiles: [prod] en mismo archivo | Un solo archivo, más ruido visual | |

### Q2: Resource limits por contenedor

| Option | Description | Selected |
|--------|-------------|----------|
| Sí, límites suaves (web 4G/2cpus, db 2G, caddy 256M) | OOM-killer mata pipeline, no server | ✓ |
| No, sin límites | Confianza 100% en preflight Phase 4 | |
| Ajustar después en producción | TODO con `docker stats` baseline | |

---

## Healthchecks + deploy + SMTP

### Q1: Healthcheck endpoint web

| Option | Description | Selected |
|--------|-------------|----------|
| /api/health | Endpoint ya existe | ✓ |
| TCP check puerto 8000 | Solo verifica uvicorn escuchando | |

### Q2: Alcance scripts/deploy.sh

| Option | Description | Selected |
|--------|-------------|----------|
| Solo lo básico (pull/build/up) | Mínimo y predecible | |
| Básico + smoke test post-deploy | curl /api/health al final | |
| Básico + backup pre-deploy + smoke | Máxima paranoia | ✓ |

### Q3: SMTP en .env.prod.example

| Option | Description | Selected |
|--------|-------------|----------|
| Vars comentadas con # TODO Mark-IV | Lector sabe que está pendiente | ✓ |
| Omitir totalmente | Falla con KeyError sin contexto | |

### Q4: Cerrar?

| Option | Description | Selected |
|--------|-------------|----------|
| Cerrar y escribir CONTEXT.md | Listo para planner | ✓ |
| Revisar más | Más matización antes de cerrar | |

---

## Claude's Discretion

- Estructura interna de DEPLOY_LAN.md (orden secciones, nivel de detalle)
- Convención naming `.sql.gz` rotados
- Si pre-deploy backup vive en `/var/backups/nexo/predeploy/` separado o en mismo dir con tag

## Deferred Ideas

- Migrar a DNS interno (router/AD/dnsmasq) cuando IT lo permita
- Sync de backups a NAS o equipo externo
- Cert firmado por CA corporativa ECS
- WAL archiving / replicación Postgres (RPO < 1h)
- Smoke test extendido (login + query + PDF)
- Monitoring / alertas (Prometheus, uptime-kuma)
