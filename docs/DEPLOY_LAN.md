# DEPLOY_LAN — Runbook de despliegue Nexo en LAN ECS

Ultima actualizacion: 2026-04-21
Operador principal: e.eguskiza@ecsmobility.com
Operador de respaldo: `<ADMIN_BACKUP_NAME>` (a designar tras el primer despliegue real)

---

## 1. Vista general

Nexo corre en un servidor Ubuntu Server 24.04 de la LAN ECS, accesible desde
cualquier equipo interno en `https://nexo.ecsmobility.local`. No esta expuesto
a internet (decision cerrada Mark-III, ver `CLAUDE.md` §"Que NO hacer"). Este
runbook asume que el admin de respaldo tiene conocimientos basicos de Linux
(SSH, sudo, editor de texto) pero NO del proyecto Nexo.

Tiempo de recuperacion objetivo (RTO): 1-2 horas desde un Ubuntu limpio hasta
tener Nexo accesible en LAN con backup restaurado (D-11).

Bus factor objetivo: 2 (D-12). Este documento es la unica fuente de verdad
operativa para el admin de respaldo. Si encuentras un paso que no entiendes,
preguntale al operador principal antes de improvisar.

**Modelo de amenazas (resumido):**

- LAN interna ECS = confianza moderada. No exposicion a internet.
- El auth de Phase 2 (cookies + argon2id + RBAC) controla el acceso a la app.
- Los backups contienen hashes de contrasena + audit_log: permisos 600 obligatorios.
- TLS emitido por Caddy Local CA (`tls internal`). El root CA se distribuye
  una sola vez por equipo LAN usando un canal confiable (scp, USB, chat
  interno — nunca email publico).

**Contrato inmutable durante Mark-III:**

- Hostname: `nexo.ecsmobility.local` (D-01).
- Estrategia TLS: `tls internal` con root CA propia (D-04; Let's Encrypt
  DNS-01 y CA corporativa ECS descartadas).
- Backups: pg_dump + gzip diario en `/var/backups/nexo/`, retencion 7 dias (D-07/D-08/D-09).
- Firewall: ufw con 22 (subred LAN), 80, 443, deny resto (D-14..D-17).
- Resolucion DNS: hosts-file por equipo LAN (D-02). Migracion a DNS interno = mejora futura.

---

## 2. Prerrequisitos

Antes de empezar, confirmar con IT:

- IP estatica LAN asignada al servidor: `<IP_NEXO>` (rellenar valor real
  antes de editar ningun archivo del deploy).
- Subred LAN en notacion CIDR: `<SUBNET_LAN>` (ej. `192.168.1.0/24`).
- Acceso SSH con `sudo` desde tu equipo al servidor.
- Puerto 22 ya abierto en los switches/firewalls corporativos (si los hay).
- Credenciales de los SQL Server corporativos (`ecs_mobility` y `dbizaro`);
  si no las tienes a mano, pide al operador principal que te comparta
  `.env.prod` por canal seguro.
- Un segundo equipo LAN (laptop admin) para validar DEPLOY-07 (peer LAN con
  hosts-file + root CA instalados).

Hardware objetivo (referencia; la CI no exige valores exactos):

- CPU: Intel i5 7a gen o superior.
- RAM: 16 GB.
- Disco: SSD >=500 GB.
- Red: interfaz ethernet con IP estatica (DHCP reservation en el router vale).

---

## 3. Instalar Docker CE + compose plugin en Ubuntu 24.04

El servidor Ubuntu llega limpio. Instalar Docker CE desde el repositorio
oficial de Docker (no usar `apt install docker.io` del repo de Ubuntu; esta
desactualizado y complica el soporte). Fuente: `docs.docker.com/engine/install/ubuntu/`.

```bash
# 1. Prerequisites
sudo apt update
sudo apt install -y ca-certificates curl git

# 2. Docker apt repo
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu noble stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update

# 3. Install Docker Engine + compose plugin + buildx
sudo apt install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin

# 4. Group membership (usar el usuario real del operador, p.ej. "nexo-ops")
sudo usermod -aG docker "${USER}"
# El usuario debe hacer `logout` + `login` (o `newgrp docker`) para aplicar

# 5. Enable + start
sudo systemctl enable --now docker

# 6. Verificar
docker --version
docker compose version        # debe ser v2.24+ (requisito para `!reset`)
sudo systemctl status docker --no-pager
docker run --rm hello-world
```

Si `docker compose version` devuelve algo menor que `v2.24`, detente aqui y
avisa al operador principal. El override prod (`docker-compose.prod.yml`)
usa el tag YAML `!reset`, introducido en Compose v2.24 (Dic 2023).

---

## 4. Clonar el repo y configurar .env

El directorio canonico en el servidor es `/opt/nexo`. Usalo para que los
cron jobs, logs del sistema y rutas documentadas en este runbook sean
consistentes.

```bash
sudo mkdir -p /opt && sudo chown "${USER}:${USER}" /opt
cd /opt
git clone <URL_REPO_NEXO> nexo
cd /opt/nexo
git checkout main    # o la rama de release estable que indique el operador

cp .env.prod.example .env
```

Editar `/opt/nexo/.env` con los valores reales. Campos obligatorios que
DEBEN rellenarse (todos los `<CHANGEME-*>`):

- `NEXO_SECRET_KEY`: generar con
  `python3 -c "import secrets; print(secrets.token_urlsafe(48))"`
- `NEXO_PG_USER`, `NEXO_PG_PASSWORD`, `NEXO_PG_DB`: credenciales Postgres de
  Nexo (Postgres 16 del container `db`).
- `NEXO_PG_APP_PASSWORD`: rol aplicacion separado (recomendado en prod,
  distinto al owner).
- `NEXO_APP_SERVER` / `NEXO_APP_USER` / `NEXO_APP_PASSWORD`: SQL Server
  `ecs_mobility` (pedir a IT).
- `NEXO_MES_SERVER` / `NEXO_MES_USER` / `NEXO_MES_PASSWORD`: SQL Server
  `dbizaro` (MES, solo lectura).

Dejar comentado el bloque SMTP (`# TODO Mark-IV`): SMTP esta Out of Scope
Mark-III. Descomentarlo con valores ficticios hace que la app intente
conectar a un servidor inexistente.

**Critico:** NO commitear `/opt/nexo/.env` al repo. El `.gitignore` ya lo
ignora, pero verificar siempre antes de cada push:

```bash
cd /opt/nexo && git status   # .env NO debe aparecer en untracked
```

---

## 5. Configurar el firewall (ufw)

Hacerlo ANTES de arrancar el stack, para no dejar ventanas abiertas a la
LAN. Fuente: seccion Topic 5 de `.planning/phases/06-despliegue-lan-https/06-RESEARCH.md`.

Secuencia segura (primero las reglas, al final el `enable` — si no, te
quedas fuera del SSH):

```bash
# 1. Instalar (ya viene en Ubuntu Server 24.04; reinstalar por si acaso)
sudo apt update && sudo apt install -y ufw

# 2. PRIMERO la regla SSH — CRITICO: antes de `ufw enable`
sudo ufw allow from <SUBNET_LAN> to any port 22 proto tcp comment 'SSH LAN'

# 3. HTTP y HTTPS (abiertos a toda la LAN; auth Phase 2 controla acceso)
sudo ufw allow 80/tcp  comment 'HTTP redirect to HTTPS'
sudo ufw allow 443/tcp comment 'HTTPS Nexo'

# 4. Defaults
sudo ufw default deny incoming
sudo ufw default allow outgoing

# 5. AHORA si, enable (prompt "Proceed with operation (y|n)?" — contestar y)
sudo ufw enable

# 6. Verificar
sudo ufw status verbose
sudo ufw status numbered
```

Salida esperada en `ufw status numbered` (los numeros pueden variar):

```
[ 1] 22/tcp        ALLOW IN    <SUBNET_LAN>               # SSH LAN
[ 2] 80/tcp        ALLOW IN    Anywhere                   # HTTP redirect to HTTPS
[ 3] 443/tcp       ALLOW IN    Anywhere                   # HTTPS Nexo
Default: deny (incoming), allow (outgoing), disabled (routed)
```

---

## 6. Arrancar el stack Nexo

Primera vez (desde `/opt/nexo`):

```bash
make prod-up
```

Equivalente sin Makefile (por si el make no esta disponible en el servidor):

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Esperar ~30 segundos para que los healthchecks marquen los 3 containers como
`healthy`. Verificar estado:

```bash
make prod-status
# Esperado: 3 containers (db, web, caddy) en estado running y healthy.
```

La primera vez, Caddy genera la Local CA (`/data/caddy/pki/authorities/local/`)
y emite el cert del hostname `nexo.ecsmobility.local`. Siguiente paso:
distribuir el root CA a los equipos LAN.

---

## 7. Extraer y distribuir la root CA de Caddy

Sin esto, los browsers de los equipos LAN mostraran un warning TLS. El
certificado esta dentro del container Caddy. Se extrae una sola vez al host
y luego se distribuye por SO.

Ruta dentro del container (inmutable desde Caddy 2.0):

```
/data/caddy/pki/authorities/local/root.crt
```

### 7.1 Extraer al host

```bash
cd /opt/nexo
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec caddy \
    cat /data/caddy/pki/authorities/local/root.crt > nexo-ca.crt
```

Distribuir `nexo-ca.crt` a cada equipo LAN usando un canal confiable (scp,
USB, chat interno corporativo). NO usar email publico: un atacante con
acceso al correo podria interceptar el certificado y montar un MITM.

### 7.2 Instalacion en Windows 10/11

Requiere admin. Via GUI:

1. Doble-click `nexo-ca.crt`.
2. "Install Certificate" -> "Local Machine".
3. "Place all certificates in the following store" ->
   "Trusted Root Certification Authorities".

Via PowerShell como administrador:

```powershell
certutil.exe -addstore -f "ROOT" C:\path\to\nexo-ca.crt
```

### 7.3 Instalacion en Ubuntu / Debian

Tambien aplica al propio servidor Ubuntu si se va a hacer `curl` local a
HTTPS sin `-k`:

```bash
sudo cp nexo-ca.crt /usr/local/share/ca-certificates/nexo-ca.crt
sudo update-ca-certificates
# Salida esperada: "1 added, 0 removed; done."
```

### 7.4 Instalacion en macOS

```bash
sudo security add-trusted-cert -d -r trustRoot \
    -k /Library/Keychains/System.keychain nexo-ca.crt
```

### 7.5 Instalacion en Firefox (trust store propio)

Firefox usa su propio trust store, separado del SO:

```
Menu -> Configuracion -> Privacidad y seguridad -> Certificados -> Ver certificados...
-> Autoridades -> Importar... -> seleccionar nexo-ca.crt
-> marcar "Confiar en esta CA para identificar sitios web" -> OK
```

### 7.6 Verificacion end-to-end

Desde un equipo LAN con CA + hosts-file ya configurados:

```bash
openssl s_client -connect nexo.ecsmobility.local:443 -servername nexo.ecsmobility.local </dev/null 2>&1 | \
    grep -E "(subject|issuer|Verify return code)"
# Esperado:
#   subject=CN=nexo.ecsmobility.local
#   issuer=CN=Caddy Local Authority - ECC Intermediate
#   Verify return code: 0 (ok)
```

---

## 8. Configurar hosts-file en cada equipo LAN

Mientras no haya DNS interno (mejora futura), cada equipo LAN que quiera
acceder a Nexo necesita una entrada en su hosts-file.

Entrada canonica (reemplazar `<IP_NEXO>` con la IP real):

```
<IP_NEXO>   nexo.ecsmobility.local
```

### 8.1 Windows 10/11

Ruta: `C:\Windows\System32\drivers\etc\hosts`. Requiere admin:

1. Boton Inicio -> escribir "Bloc de notas".
2. Click derecho -> "Ejecutar como administrador".
3. Archivo -> Abrir -> `C:\Windows\System32\drivers\etc\hosts` (si no lo ves,
   cambiar filtro a "Todos los archivos").
4. Anadir al final:

   ```
   <IP_NEXO>   nexo.ecsmobility.local
   ```

5. Guardar (Ctrl+S). Cerrar.

Alternativa PowerShell como administrador:

```powershell
Add-Content -Path "$env:SystemRoot\System32\drivers\etc\hosts" -Value "`n<IP_NEXO>`tnexo.ecsmobility.local"
```

### 8.2 Linux

```bash
echo "<IP_NEXO>   nexo.ecsmobility.local" | sudo tee -a /etc/hosts
```

### 8.3 macOS

```bash
echo "<IP_NEXO>   nexo.ecsmobility.local" | sudo tee -a /etc/hosts
sudo dscacheutil -flushcache
sudo killall -HUP mDNSResponder
```

### 8.4 Verificacion cross-OS

```bash
# Todos los SO:
ping -c 1 nexo.ecsmobility.local
# Esperado: "PING nexo.ecsmobility.local (<IP_NEXO>) ..."

# Nota: `nslookup` consulta DNS, no el hosts-file, y fallara con NXDOMAIN.
# El test correcto es `ping`.
```

---

## 9. Avisos criticos (Landmines)

Estas son las trampas conocidas que pueden destrozar un deploy si no se
entienden. Leer entero antes de tocar nada.

### 9.1 Docker bypassa ufw

[Contexto verificado en `docs.docker.com/engine/network/packet-filtering-firewalls/`.]
Los `ports:` publicados por Docker van directamente a la chain `DOCKER-USER`
de iptables, SIN pasar por ufw.

Consecuencias:

- `ufw deny 5432/tcp` NO cierra Postgres si el compose publica ese puerto.
  La regla simplemente no se aplica a containers.
- `ufw allow 443/tcp` es redundante para el trafico hacia el container de
  Caddy: Docker ya abre el puerto. Lo dejamos solo como defensa en profundidad
  por si manana alguien levanta un nginx nativo en el host.

**Defensa real del 5432:** `docker-compose.prod.yml` tiene

```yaml
db:
  ports: !reset []
```

Docker directamente no publica el puerto al host. Sin publicacion, no hay
regla iptables de DNAT hacia el container y la red externa no puede llegar.

Verificacion desde el propio servidor:

```bash
sudo ss -tlnp | grep :5432
# Esperado: sin output (puerto no escucha en el host)
```

Verificacion desde un peer LAN:

```bash
nmap -Pn -p 22,80,443,5432 <IP_NEXO>
# Esperado: 22, 80, 443 open; 5432 closed o filtered
```

Moraleja: si ves `ufw status` sin el puerto 443 y piensas que Caddy esta
mal, estas mirando el sitio equivocado. `ufw status` no muestra reglas
Docker. La verdad esta en `iptables -L DOCKER-USER -n -v` y en `docker ps`.

### 9.2 "docker compose down -v" borra pgdata

>  AVISO (HIGH): JAMAS usar `docker compose down -v` en prod.

El flag `-v` borra los volumes definidos en el compose, incluido `pgdata`.
Pierdes toda la BD de produccion en un segundo (hashes de usuarios,
audit_log, approvals, cache de thresholds, etc.).

- Correcto en prod: `docker compose down` (sin `-v`) o `make prod-down`.
- El target `make clean` del Makefile SI usa `-v` y esta pensado SOLO para
  dev. Jamas ejecutar `make clean` en el servidor de produccion.

### 9.3 No hacer `git commit` en el servidor

`scripts/deploy.sh` usa `git pull --ff-only`. Si alguien hace un `git commit`
local en el servidor (por ejemplo un hotfix editado in situ), el siguiente
deploy fallara por divergencia con `main` remoto.

Flujo correcto: cualquier cambio se mergea a `main` en GitHub y luego se
ejecuta `bash scripts/deploy.sh` en el servidor. Si ya ocurrio un commit
local por error, `git stash` -> `git pull` -> aplicar los cambios via PR en
GitHub.

### 9.4 No copiar `caddy/Caddyfile` dev como `Caddyfile.prod`

El Caddyfile dev tiene `auto_https disable_redirects`. Copiado tal cual a
prod rompe el redirect 80->443 (los usuarios que escriban
`http://nexo.ecsmobility.local` reciben `connection refused`). El prod se
escribe from-scratch y deliberadamente omite ese flag.

### 9.5 La root CA expira en ~10 anos

Caddy emite un root CA de 10 anos (3600d). Anotar la fecha de expiracion.
Cuando se acerque, Caddy generara uno nuevo y habra que RE-DISTRIBUIR
`nexo-ca.crt` a todos los equipos LAN. Comando para consultar fecha:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec caddy \
    openssl x509 -in /data/caddy/pki/authorities/local/root.crt -noout -enddate
```

Planificar repeticion del paso 7 ~2035.

### 9.6 `/api/health` llama a SQL Server MES

Un MES caido NO tira el container (el endpoint devuelve 200 con `ok: false`).
Es correcto por diseno: Nexo sigue funcionando en modo degradado (login,
auditoria, Postgres). NO cambiar el healthcheck a `curl -fs ... && jq -e .ok`:
eso pondria el web en bucle de reinicio cuando MES este caido.

### 9.7 No rotar credenciales SQL Server durante un deploy ordinario

Las credenciales SQL Server se rotan en una ventana de mantenimiento
planificada (coordinada con IT). Un deploy rutinario no debe tocar
credenciales; si hace falta rotarlas, se hace en un procedimiento aparte.

---

## 10. Configurar el cron de backup nightly

Instalar el cron entry que ejecuta `scripts/backup_nightly.sh` diariamente
a las 03:00 UTC (D-07):

```bash
sudo tee /etc/cron.d/nexo-backup > /dev/null <<'EOF'
# /etc/cron.d/nexo-backup — Nexo pg_dump nightly (D-07: 03:00 UTC).
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

0 3 * * * root /opt/nexo/scripts/backup_nightly.sh >> /var/log/nexo-backup.log 2>&1
EOF

sudo touch /var/log/nexo-backup.log
sudo chmod 0640 /var/log/nexo-backup.log
sudo systemctl restart cron
```

Verificar que cron reconoce el job:

```bash
sudo grep nexo-backup /var/log/syslog | tail
# Tras la primera ejecucion real a las 03:00 UTC:
ls -lh /var/backups/nexo/
```

Ejecucion manual inmediata (recomendado antes del primer cron real, para
validar que la mecanica funciona):

```bash
sudo /opt/nexo/scripts/backup_nightly.sh
ls -lh /var/backups/nexo/
# Debe aparecer nexo-YYYYMMDD-HHMM.sql.gz con permisos 600.
```

Ubicacion `/var/backups/nexo/`, retencion 7 dias (D-08/D-09). El sync a un
NAS externo esta como mejora futura (ver seccion 15).

---

## 11. Validacion post-deploy (8 checks DEPLOY-*)

La forma rapida: ejecutar el script scripteado de smoke desde el propio
servidor:

```bash
bash /opt/nexo/tests/infra/deploy_smoke.sh
```

El script imprime una linea por check con `[DEPLOY-XX] OK|FAIL: mensaje` y
sale con exit code = numero de fallos. Si todos pasan, exit 0.

O manualmente, uno por uno:

| DEPLOY-* | Comando | Esperado |
|----------|---------|----------|
| DEPLOY-01 | `curl -fsI -k https://nexo.ecsmobility.local/api/health` | `HTTP/2 200` |
| DEPLOY-01 | `openssl s_client ... \| grep issuer` | `issuer=CN=Caddy Local Authority...` |
| DEPLOY-02 | `sudo ss -tlnp \| grep :5432` | sin output |
| DEPLOY-02 | `docker compose -f docker-compose.yml -f docker-compose.prod.yml exec db bash -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1;"'` | `1 row` |
| DEPLOY-03 | `docker inspect $(... ps -q web) --format '{{.State.Health.Status}}'` | `healthy` |
| DEPLOY-03 | `docker inspect $(... ps -q caddy) --format '{{.State.Health.Status}}'` | `healthy` |
| DEPLOY-03 | `docker inspect $(... ps -q web) --format '{{.HostConfig.RestartPolicy.Name}}'` | `unless-stopped` |
| DEPLOY-04 | `bash scripts/deploy.sh; bash scripts/deploy.sh` (x2) | ambas `DONE`, exit 0 |
| DEPLOY-06 | `sudo ufw status numbered` | solo 22 LAN, 80, 443; deny default |

Desde un peer LAN con hosts-file + root CA instalados (DEPLOY-07):

```bash
# Cert aceptado sin warning
curl -sI https://nexo.ecsmobility.local/api/health
# Esperado: HTTP/2 200, sin warning TLS

# Firewall desde fuera
nmap -Pn -p 22,80,443,5432 <IP_NEXO>
# Esperado: 22, 80, 443 open; 5432 closed
```

---

## 12. Operaciones rutinarias

| Operacion | Comando |
|-----------|---------|
| Deploy (git pull + build + up + smoke) | `cd /opt/nexo && bash scripts/deploy.sh` (o `make deploy`) |
| Backup manual | `bash scripts/backup_nightly.sh` (o `make backup`) |
| Ver logs prod | `make prod-logs` |
| Estado containers | `make prod-status` |
| Health check local | `make prod-health` |
| Parar stack (pgdata persiste) | `make prod-down` |
| Abrir psql interactivo | `docker compose -f docker-compose.yml -f docker-compose.prod.yml exec -it db bash -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'` |

Nota sobre `/api/health`: si la BD MES del cliente esta caida,
`make prod-health` puede devolver JSON con `services.mes.ok = false`
mientras `services.web.ok = true` y `services.db.ok = true`. Esto es
normal: Nexo sigue operativo en modo degradado (login, auditoria, preflight
pipeline estan vivos). El healthcheck Docker sigue marcando `healthy` porque
el endpoint responde 200 — es exactamente lo que queremos (ver Landmine 9.6).

---

## 13. Restaurar desde backup

Procedimiento para restaurar un `.sql.gz` concreto sobre la BD Nexo en
produccion. Usar con cuidado: sobreescribe el schema `nexo` entero.

```bash
cd /opt/nexo
BACKUP=/var/backups/nexo/nexo-20260421-0300.sql.gz   # ajustar al archivo real

COMPOSE="docker compose -f docker-compose.yml -f docker-compose.prod.yml"

# 1. Drop + recreate schema nexo (evita conflictos con tablas existentes)
$COMPOSE exec -T db bash -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "DROP SCHEMA IF EXISTS nexo CASCADE; CREATE SCHEMA nexo;"'

# 2. Restore (zcat descomprime al vuelo)
zcat "${BACKUP}" | $COMPOSE exec -T db bash -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'

# 3. Verificar tablas restauradas
$COMPOSE exec -T db bash -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\dt nexo.*"'
```

Si el restore es a raiz de un incidente, tras verificar que las tablas
`nexo.*` existen, ejecutar el smoke:

```bash
bash /opt/nexo/tests/infra/deploy_smoke.sh
```

y hacer login manual en `https://nexo.ecsmobility.local` para confirmar
que la auth y el audit_log siguen operativos.

---

## 14. Recuperacion desde cero (RTO 1-2h)

Si el servidor original muere y hay que levantar Nexo en otro Ubuntu:

1. Provisionar Ubuntu 24.04, asignar IP LAN (idealmente la misma
   `<IP_NEXO>` para no tener que reeditar hosts-file en los clientes),
   SSH + sudo.
2. Seguir secciones 3-6 (Docker install, clone, `.env`, `prod-up`).
3. Copiar el `.sql.gz` mas reciente de `/var/backups/nexo/` al nuevo
   servidor (scp desde el backup externo o desde NAS si ya esta configurado).
4. Seguir seccion 13 para restaurar.
5. Seguir seccion 5 (ufw) en el nuevo servidor.
6. Seguir seccion 10 (cron nightly) en el nuevo servidor.
7. Si la IP cambia: todos los equipos LAN deben actualizar su hosts-file
   (seccion 8).
8. **La root CA del nuevo Caddy SERA DISTINTA** a la del servidor viejo.
   Hay que re-extraer `nexo-ca.crt` (seccion 7.1) y RE-DISTRIBUIR a todos
   los equipos LAN. Esto es lo que mas tiempo consume del RTO.

Tiempo total realista en un Ubuntu limpio: 1h instalacion + 30 min restore
+ 30 min re-distribucion CA = 2h.

---

## 15. Mejoras futuras (deferred)

Estas no son bloqueantes para Mark-III pero quedan anotadas para Mark-IV o
posterior:

- **DNS interno** (router / Active Directory / dnsmasq): elimina el paso de
  hosts-file por usuario (seccion 8). Requiere coordinacion con IT.
- **Sync de backups a NAS** con cron `rsync`: actualmente los `.sql.gz`
  viven solo en `/var/backups/nexo/` del propio servidor. Si el SSD muere,
  se pierde todo. Cuando haya NAS corporativo disponible, anadir un segundo
  cron que haga `rsync` al destino externo.
- **Cert firmado por CA corporativa ECS**: si IT confirma que mantiene una
  CA interna ya distribuida en los equipos del personal, migrar de
  `tls internal` a esa CA elimina la friccion de distribuir la root CA
  Caddy (seccion 7).
- **WAL archiving / replicacion Postgres**: para llegar a RPO < 1h (hoy
  RPO = 24h por el cron nightly). Sobre-engineering para Mark-III pero una
  opcion si Nexo se vuelve operacionalmente critico.
- **Smoke test extendido**: el smoke actual cubre `/api/health` + puertos.
  Una version mas robusta haria login + ejecutar 1 query + generar 1 PDF.
  Mark-IV cuando sobre tiempo.
- **Monitoring / alertas**: Prometheus + Grafana o uptime-kuma. Out of
  Scope Mark-III. Phase 7 (DevEx) puede incluirlo si sobra alcance.

---

## 16. Glosario y referencias

### Glosario minimo

- **Nexo**: nombre comercial de la plataforma interna de ECS Mobility que
  corre en este servidor. Ver `docs/GLOSSARY.md` para terminos de dominio.
- **MES**: Manufacturing Execution System. Aqui se refiere al SQL Server
  `dbizaro` (Izaro MES), solo lectura.
- **APP**: SQL Server `ecs_mobility` (configuracion de recursos, ciclos,
  contactos). Lectura y escritura.
- **Schema nexo**: tablas propias de Nexo en Postgres (users, roles,
  permissions, audit_log, query_log, login_attempts, approvals).
- **`tls internal`**: directiva Caddy que hace que Caddy genere su propia
  CA local y emita certs sin Let's Encrypt. Sin exposicion a internet.
- **Landmine**: trampa conocida del sistema. Ver seccion 9.

### Referencias cruzadas

- `CLAUDE.md` §"Convenciones de naming" y §"Que NO hacer" — politicas que
  este runbook respeta.
- `.env.prod.example` — template literal de las vars NEXO_* que este
  runbook instruye rellenar en seccion 4.
- `caddy/Caddyfile.prod` — configuracion Caddy de produccion referenciada
  en la seccion 6.
- `docker-compose.prod.yml` — override prod consumido por `scripts/deploy.sh`
  y por `make prod-up` / `make deploy`.
- `scripts/deploy.sh` — script idempotente del paso 12 (operaciones).
- `scripts/backup_nightly.sh` — script invocado por el cron de la seccion 10.
- `tests/infra/deploy_smoke.sh` — checklist scriptable de la seccion 11.
- `.planning/phases/06-despliegue-lan-https/06-RESEARCH.md` — investigacion
  completa (10 landmines, topics de cert/ufw/hosts/Docker install).

### Contactos

- Operador principal: e.eguskiza@ecsmobility.com
- Operador de respaldo: `<ADMIN_BACKUP_NAME>` (a designar por el operador
  principal tras el primer despliegue real).

---

## Apendice A: Comandos de un vistazo

```bash
# Arranque inicial
cd /opt/nexo && make prod-up

# Deploy de nueva version
cd /opt/nexo && bash scripts/deploy.sh

# Estado
make prod-status

# Backup manual
bash scripts/backup_nightly.sh

# Restore desde el backup mas reciente
BACKUP=$(ls -t /var/backups/nexo/nexo-*.sql.gz | head -1)
zcat "${BACKUP}" | \
    docker compose -f docker-compose.yml -f docker-compose.prod.yml exec -T db \
    bash -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'

# psql interactivo
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec -it db \
    bash -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'

# Validacion smoke 8 checks
bash /opt/nexo/tests/infra/deploy_smoke.sh
```

---

## Apendice B: Secuencia mental para un deploy rutinario

Cuando ya esta todo instalado y llega un release nuevo:

1. `ssh <IP_NEXO>` con el usuario operador.
2. `cd /opt/nexo`.
3. `git status` — debe estar clean (ningun commit local).
4. `make deploy` (o `bash scripts/deploy.sh`) — el script hace pull,
   pre-deploy backup, build, up -d, smoke.
5. Si falla, el script imprime el hint de rollback manual. Seguirlo.
6. Si OK, el script imprime `DONE`. Ejecutar `bash tests/infra/deploy_smoke.sh`
   para checklist completo.
7. Abrir `https://nexo.ecsmobility.local` desde tu laptop (con CA y
   hosts-file ya configurados) y hacer login manual de sanity.
