# RUNBOOK.md — Procedimientos de incidencia para Nexo

> Audiencia: operador (admin IT) respondiendo a incidencia en produccion.
> Complementa [DEPLOY_LAN.md](DEPLOY_LAN.md) (instalacion) con respuesta runtime.
> Ver [ARCHITECTURE.md](ARCHITECTURE.md) para contexto del sistema.

Ultima revision: 2026-04-22 (Sprint 6 / Phase 7).

Cada escenario tiene: **Sintomas** -> **Diagnostico** -> **Remedio** -> **Prevencion**.
Comandos asumen que estas en el servidor productivo (`ssh <IP_NEXO>`) en
`/opt/nexo` salvo indicacion contraria.

Placeholders usados en este doc (reemplazar con el valor real antes de ejecutar):

- `<IP_NEXO>` — IP del servidor productivo en la LAN ECS Mobility.
- `<POSTGRES_USER>` / `<POSTGRES_DB>` — credenciales de la BD Nexo (ver `/opt/nexo/.env`).
- `<email>` — email del usuario afectado.
- `<email-propietario>` — email del usuario con rol propietario.
- `<ip>` — IP desde la que el usuario intentaba loguear.
- `<latest>` — nombre del fichero de backup mas reciente en `/var/backups/nexo/`.

---

## Escenario 1: MES caido (SQL Server dbizaro inaccesible)

### Sintomas

- `/bbdd` devuelve 503 o timeout > 30s.
- `/pipeline` con rango que consulta MES falla con mensaje "no se puede conectar a MES".
- Logs del container web muestran `pyodbc.OperationalError` con `engine_mes`:

  ```bash
  docker compose logs web | grep -iE "engine_mes|pyodbc"
  ```

- `/api/health` responde **200** (no 503) pero con `services.mes.ok=false`. Este
  es el disenyo intencional: la app sigue viva aunque MES este caido, Caddy no
  marca el backend como down.

### Diagnostico

1. Confirma que MES es el problema (no todo el SQL Server):

   ```bash
   docker compose exec web python -c "
   from nexo.data.engines import engine_mes
   from sqlalchemy import text
   try:
       engine_mes.connect().execute(text('SELECT 1'))
       print('MES: OK')
   except Exception as e:
       print(f'MES: FAIL - {e}')
   "
   ```

2. Test de red al host SQL Server:

   ```bash
   MES_HOST=$(grep '^NEXO_MES_SERVER=' .env | cut -d= -f2)
   MES_PORT=$(grep '^NEXO_MES_PORT=' .env | cut -d= -f2)
   docker compose exec web nc -zv "$MES_HOST" "$MES_PORT"
   ```

   - `open` -> red OK, problema es auth o BBDD.
   - `refused` / `timeout` -> red caida o firewall.

3. Verifica que `engine_app` SI funciona (para aislar el problema a MES solo):

   ```bash
   docker compose exec web python -c "
   from nexo.data.engines import engine_app
   from sqlalchemy import text
   engine_app.connect().execute(text('SELECT 1'))
   print('engine_app: OK')
   "
   ```

### Remedio

**Caso A — Red caida al SQL Server:**

- Escalar a IT: cable, switch, firewall, ruta estatica.
- Mientras tanto: la app sigue viva, solo las pantallas que consultan MES
  (`/bbdd`, `/pipeline`, `/capacidad`) daran error claro al usuario.

**Caso B — Credenciales SQL Server cambiadas:**

- Rotacion reciente: actualizar `NEXO_MES_PASSWORD` en `/opt/nexo/.env`.
- Reiniciar stack para recargar env:

  ```bash
  cd /opt/nexo
  make prod-down && make prod-up
  make prod-health
  ```

**Caso C — SQL Server MES caido (servicio dbizaro abajo):**

- Escalar al DBA del equipo MES (dbizaro). Fuera del alcance de Nexo.

### Prevencion

- Healthcheck `/api/health` expone `services.mes.ok` — integrar en monitoreo
  externo (cron LAN que llame cada 5 min, alerta si `ok=false` durante > 15 min
  consecutivos).
- Las credenciales MES viven en `.env` (no en la imagen Docker) — al rotar, un
  solo reinicio las recoge.
- La app esta disenyada para degradarse graciosamente: MES caido no tumba Nexo
  completo, solo las pantallas que lo consumen.

---

## Escenario 2: Postgres no arranca

### Sintomas

- `docker compose ps` muestra el servicio `db` en `unhealthy` o `exited`.
- El servicio `web` queda en `dependency failed` o reintentando.
- La UI es inaccesible: Caddy devuelve 502 Bad Gateway desde `:443`.
- Logs de `db` con errores de arranque (ver Diagnostico).

### Diagnostico

1. Ver el estado de los containers y los logs de `db`:

   ```bash
   docker compose ps
   docker compose logs db | tail -100
   ```

2. Comprobar espacio en disco de Docker:

   ```bash
   df -h /var/lib/docker
   ```

3. Causas tipicas (por orden de probabilidad):
   - `pgdata` volumen corrupto tras apagon o kill -9 brutal.
   - Espacio disco insuficiente en `/var/lib/docker` (>90% lleno).
   - Cambio de version Postgres incompatible (ej. 15 -> 16 sin `pg_upgrade`).
   - Permisos del volumen `pgdata` incorrectos tras restauracion manual.

### Remedio

**Caso A — Espacio disco:**

- Liberar espacio borrando imagenes no usadas:

  ```bash
  docker system prune -a
  ```

- **WARNING (Landmine 6 Phase 6):** NO usar `docker volume rm pgdata` ni
  `docker compose down -v` — ambos **borran la BD completa** de Nexo. El
  volumen `pgdata` solo se borra manualmente tras backup verificado.

**Caso B — Corrupcion del volumen `pgdata` (irrecuperable):**

- Restaurar desde el backup mas reciente en `/var/backups/nexo/`:

  ```bash
  ls -lt /var/backups/nexo/nexo_*.sql.gz | head -5
  # tomar el mas reciente -> <latest>

  # Detener web para que no escriba durante la restauracion
  docker compose stop web

  # Restaurar
  gunzip < /var/backups/nexo/<latest>.sql.gz | \
    docker compose exec -T db psql -U <POSTGRES_USER> -d <POSTGRES_DB>

  # Levantar web
  docker compose start web
  ```

- El script `scripts/backup_nightly.sh` (Phase 6) escribe diario en
  `/var/backups/nexo/` con retencion 7d. Si ningun backup de los ultimos 7d es
  valido, escalar: la perdida de datos es real.

**Caso C — Version incompatible:**

- Downgrade de imagen en `docker-compose.yml` (ej. `postgres:15-alpine`) si el
  volumen fue creado con 15 y alguien subio a 16 sin migrar. Alternativa:
  `pg_dump` desde contenedor 15 + `pg_restore` en contenedor 16 nuevo.

### Prevencion

- `scripts/backup_nightly.sh` en cron diario (Phase 6) — verificar con:

  ```bash
  crontab -l | grep backup_nightly
  ls -lt /var/backups/nexo/ | head -5
  ```

- Monitorizar `df -h /var/lib/docker` con alerta a >80%.
- **NO usar `docker compose down -v`** en el servidor productivo — borra los
  volumenes nombrados `pgdata` y `caddy_data` (Landmine 6 Phase 6). Usar
  `make prod-down` que no pasa `-v`.
- Backup del volumen `pgdata` incluido implicitamente via `pg_dump` diario; el
  volumen fisico (capa docker) no se replica, solo el dump logico.

---

## Escenario 3: Certificado Caddy expira / warning en browsers

### Sintomas

- El browser muestra `ERR_CERT_AUTHORITY_INVALID` o
  `NET::ERR_CERT_DATE_INVALID` al visitar `https://nexo.ecsmobility.local`.
- Los usuarios reportan banner amarillo "no seguro" en Chrome/Edge.
- `curl https://nexo.ecsmobility.local` falla con error de verificacion.

### Contexto

Phase 6 usa Caddy con `tls internal` (CA interna de Caddy). Rotacion de
certificados (verificado en [DEPLOY_LAN.md](DEPLOY_LAN.md)):

- **Root CA** auto-regenera cada **10 anos**.
- **Intermediate** auto-regenera cada **7 dias**.

La CA root se distribuye manualmente a los clientes LAN la primera vez (ver
seccion hosts-file + root CA en DEPLOY_LAN.md).

### Diagnostico

1. Listar certificados actuales en Caddy:

   ```bash
   docker compose exec caddy caddy list-certificates
   ```

2. Inspeccionar el cert servido en `:443`:

   ```bash
   openssl s_client -connect nexo.ecsmobility.local:443 -showcerts < /dev/null 2>&1 \
     | grep -E "issuer|subject|Not After"
   ```

3. Si el `issuer` no es la Caddy Local Authority esperada, la CA root cambio
   (re-instalacion limpia que borro `caddy_data`).

4. Si `Not After` esta en el pasado, la renovacion automatica fallo.

### Remedio

**Caso A — Intermediate caducado pero Caddy vivo:**

- Forzar renovacion reiniciando Caddy:

  ```bash
  docker compose restart caddy
  ```

- Verificar logs post-restart:

  ```bash
  docker compose logs caddy | tail -50 | grep -iE "cert|tls"
  ```

**Caso B — Root CA cambiada (tras re-instalacion limpia que borro `caddy_data`):**

- Re-extraer la root CA del volumen `caddy_data`:

  ```bash
  docker compose exec caddy cat /data/caddy/pki/authorities/local/root.crt \
    > /tmp/nexo-root-ca.crt
  ```

- Redistribuir `/tmp/nexo-root-ca.crt` a cada cliente LAN (instalar en el
  trust store de Windows/macOS/Linux segun [DEPLOY_LAN.md](DEPLOY_LAN.md)
  seccion hosts-file + root CA).

**Caso C — Auto-renovacion rota (raro):**

- Verificar logs completos:

  ```bash
  docker compose logs caddy | grep -iE "cert|renew|error"
  ```

- Si persiste, considerar migrar a `cert-manager` o `certbot` con DNS-01. Esto
  es un cambio arquitectural — ver con el dev lead antes de ejecutar.

### Prevencion

- **Root CA vive en el volumen nombrado `caddy_data`** — NO borrar con
  `docker compose down -v` (Landmine 6 Phase 6).
- Backup de `caddy_data` junto con `pgdata` (aunque se pueda regenerar, requiere
  redistribuir la CA a todos los clientes — tiempo muerto operativo).
- Monitorizar la fecha `Not After` del intermediate con un cron semanal:

  ```bash
  openssl s_client -connect nexo.ecsmobility.local:443 -showcerts < /dev/null 2>&1 \
    | openssl x509 -noout -dates
  ```

- Documentar en CLAUDE.md o en un `docs/CADDY_CA_REBUILD.md` el procedimiento
  completo de re-distribucion de root CA (pendiente para Mark-IV).

---

## Escenario 4: Pipeline atascado (HALLAZGO CRITICO: semaforo in-process)

### Sintomas

- `/api/pipeline/run` devuelve 504 Timeout tras 15 min.
- Un usuario lanza un pipeline y otros 3 esperan indefinidamente.
- Los logs muestran timeouts de `asyncio.wait_for`:

  ```bash
  docker compose logs web | grep -iE "asyncio.*timeout|pipeline"
  ```

### Contexto (HALLAZGO CRITICO)

`nexo/services/pipeline_lock.py` usa `asyncio.Semaphore(3)` **in-process**.
**NO es un lock en BD**, **NO existe un helper `list_locks()`** ni endpoint
administrativo para consultar el estado del semaforo. El semaforo es opaco
desde fuera del proceso. Si el pipeline se cuelga (matplotlib en bucle
infinito, pyodbc en deadlock), el slot sigue ocupado hasta que el thread muera
o el proceso web se reinicie.

### Diagnostico

1. Ver slots libres (indicador INDIRECTO accediendo a atributo privado, fragil):

   ```bash
   docker compose exec web python -c "
   from nexo.services.pipeline_lock import pipeline_semaphore
   print('value:', pipeline_semaphore._value)
   "
   ```

   - `value: 3` = todos los slots libres.
   - `value: 0` = los 3 slots ocupados (atasco probable).

   **ADVERTENCIA**: `._value` es atributo privado de `asyncio.Semaphore`. Solo
   para diagnostico; no escribir codigo que dependa de el.

2. Ver threads vivos en el proceso web:

   ```bash
   docker compose exec web python -c "
   import threading
   for t in threading.enumerate():
       print(t.name, t.is_alive())
   "
   ```

3. Ver el ultimo pipeline ejecutado en `nexo.query_log`:

   ```bash
   docker compose exec -T db psql -U <POSTGRES_USER> -d <POSTGRES_DB> -c \
     "SELECT created_at, endpoint, actual_ms, warn_ms FROM nexo.query_log \
      WHERE endpoint='/api/pipeline/run' \
      ORDER BY created_at DESC LIMIT 5;"
   ```

### Remedio

**Remedio nuclear (rapido, mata sesiones activas):**

```bash
docker compose restart web
```

- Mata threads colgados, reinicia event loop, libera el semaforo.
- Duracion ~30s (lifespan re-inicializa schedulers + cache de thresholds).
- Los usuarios pierden sesion si no tienen cookie persistente.
- Es la via recomendada cuando hay usuarios bloqueados esperando.

**Remedio no-nuclear (lento, menos disruptivo):**

- Esperar el timeout duro de 15 min del propio pipeline. El thread de matplotlib
  puede seguir 30-60s tras el timeout (Pitfall 1 documentado en el docstring de
  `pipeline_lock.py`). El semaforo se libera inmediatamente al expirar el
  `asyncio.wait_for`.
- Solo viable si no hay usuarios activos que queden bloqueados.

### Prevencion

- Phase 4 preflight bloquea pipelines muy caros ANTES de tomar slot en el
  semaforo. Verificar que preflight esta activo en `api/routers/pipeline.py`.
- Monitorear `nexo.query_log` con alerta si `actual_ms > warn_ms * 3` — indica
  pipeline anormalmente lento que puede acabar atascandose.
- Mark-IV: migrar a pool de workers persistentes con supervisor (RQ, Celery o
  similar) con endpoint admin `/api/admin/pipelines` que liste locks activos.

---

## Escenario 5: Lockout del unico propietario (HALLAZGO CRITICO: no hay `unlock_user`)

### Sintomas

- El unico usuario con rol `propietario` intenta loguear y responde:
  `"Usuario y contrasena incorrectos o cuenta bloqueada. Intentalo en 15 min."`
- Otros usuarios (`directivo`, `usuario`) no pueden gestionarlo: la UI de
  gestion de usuarios solo esta disponible para `propietario` (ver
  [AUTH_MODEL.md](AUTH_MODEL.md)).
- Situacion soft-brick: nadie puede desbloquear por UI.

### Contexto (HALLAZGO CRITICO)

**NO existe una funcion `unlock_user()`** ni helper CLI ni endpoint admin. La
fila del lockout esta en `nexo.login_attempts` y hay que borrarla a mano.
`nexo/services/auth.py` tiene `clear_attempts(db, email, ip)` pero se llama
solo desde el flujo de login exitoso — no hay endpoint admin que la invoque.

Si **el unico** `propietario` se lockea, no hay otro usuario que pueda crear
o desbloquear cuentas por UI. La recuperacion requiere acceso root al host.

### Diagnostico

Confirmar que el usuario esta lockeado consultando `nexo.login_attempts`:

```bash
docker compose exec -T db psql -U <POSTGRES_USER> -d <POSTGRES_DB> -c \
  "SELECT email, ip, failed_at FROM nexo.login_attempts \
   WHERE email='<email>' AND failed_at > now() - interval '15 minutes';"
```

Si hay 5 o mas filas en los ultimos 15 min, la tupla `(email, ip)` esta
bloqueada (politica: 5 intentos fallidos -> 15 min lock, ver [AUTH_MODEL.md](AUTH_MODEL.md)).

### Remedio (DELETE manual — EMERGENCY ONLY)

**Variante A — SQL directo (preferida):**

```bash
docker compose exec -T db psql -U <POSTGRES_USER> -d <POSTGRES_DB> -c \
  "DELETE FROM nexo.login_attempts WHERE email='<email-propietario>';"
```

**Variante B — Python via `clear_attempts` (si psql no disponible):**

```bash
docker compose exec web python -c "
from nexo.data.engines import SessionLocalNexo
from nexo.services.auth import clear_attempts
db = SessionLocalNexo()
clear_attempts(db, '<email>', '<ip>')
"
```

**ATENCION:** `clear_attempts(db, email, ip)` requiere tupla `(email, ip)`. Si
la IP desde la que el usuario se lockeo es desconocida, usar la Variante A con
SQL directo que borra por email sin filtrar por IP.

**WARNING de seguridad:** esta operacion es EMERGENCY ONLY. Requiere acceso
root al host Ubuntu y `docker compose exec` a la BD. El DELETE no se audita en
`nexo.audit_log` (es recovery manual fuera del flujo normal). Documentar en
el log operativo externo que se ejecuto.

### Prevencion (HARD RULE)

1. **SIEMPRE crear >=2 propietarios** (usuarios con rol `propietario`) —
   uno principal operativo, uno break-glass con password muy fuerte (32 chars
   random) guardado en sobre sellado en caja fuerte IT. Esta regla de tener
   al menos 2 propietarios activos en todo momento es lo que evita el
   soft-brick.
2. Documentar en CLAUDE.md el procedimiento break-glass: quien tiene la copia
   del sobre, bajo que condiciones se abre, obligacion de rotar la password y
   resellar tras cada apertura.
3. Mark-IV: anyadir `make unlock-user EMAIL=<x>` wrappeando el DELETE, y
   endpoint `/api/admin/unlock` con token de emergencia en `.env` para que
   otro `propietario` pueda desbloquear por UI sin acceso SSH.

---

## Referencias

- [ARCHITECTURE.md](ARCHITECTURE.md) — mapa tecnico del sistema.
- [DEPLOY_LAN.md](DEPLOY_LAN.md) — instalacion y deploy LAN.
- [AUTH_MODEL.md](AUTH_MODEL.md) — modelo de roles, lockout, departamentos.
- [RELEASE.md](RELEASE.md) — checklist de corte de release.
- [CLAUDE.md](../CLAUDE.md) — decisiones cerradas y reglas del juego.
