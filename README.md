# Nexo — Plataforma interna de ECS Mobility

Plataforma interna de ECS Mobility (sucesora de "OEE Planta" / `analisis_datos`). En Mark-III centraliza OEE (Overall Equipment Effectiveness); en milestones posteriores añade calidad, trazabilidad y otros módulos de planta. Extrae datos de produccion directamente de la BD MES (IZARO/dbizaro), los almacena en base de datos local, y genera informes PDF bajo demanda.

> **Rename del servicio MCP**: el ID pasa de `oee-planta` a `nexo-mcp` y el contenedor se renombra a `nexo-mcp`. Si tienes `.claude.json` apuntando al ID antiguo, actualizalo (ver `mcp/README.md`).

## Arquitectura

```
IZARO (SQL Server)  ──extraer──>  BD local (SQLite/PostgreSQL)  ──generar──>  PDFs bajo demanda
     fmesdtc                         datos_produccion
     fmesddf                         ejecuciones
     fmesinc                         ciclos, recursos
     fprolof
```

- **Extraccion**: conecta a IZARO, extrae datos de produccion y los guarda en la BD local
- **Almacenamiento**: los datos quedan en `datos_produccion`, no se acumulan PDFs
- **Informes**: se regeneran al momento desde los datos guardados
- **Turnos**: T1 (06-14), T2 (14-22), T3 (22-06). Dia productivo = 06:00 a 06:00

## Primer despliegue

```bash
# 1. Clonar el repo
git clone <url> && cd analisis_datos

# 2. Copiar la config de ejemplo
cp .env.example .env

# 3. Construir y arrancar
make build
make up

# 4. Abrir http://localhost:8000 (o https://<tu-ip> desde otro equipo)
#    - Ir a Ajustes > configurar la conexion al SQL Server (IP, usuario, password)
#    - Ir a Recursos > verificar que los centros de trabajo son correctos
#    - Ir a Pipeline > seleccionar fechas y ejecutar
```

## Centros de trabajo

| Recurso | CT   | Seccion     |
|---------|------|-------------|
| luk1    | 1001 | LINEAS      |
| luk2    | 1002 | LINEAS      |
| luk3    | 1003 | LINEAS      |
| luk6    | 4001 | LINEAS      |
| vw1     | 3001 | LINEAS      |
| t48     | 48   | TALLADORAS  |

Estos codigos se configuran en `data/db_config.json` o desde la UI en Ajustes/Recursos.

## Uso diario

```bash
make up          # Arranca los servicios (si estaban parados)
make down        # Para los servicios
make dev         # Desarrollo local sin Docker (SQLite)
```

## Acceso desde otros equipos (LAN)

Al hacer `make up`, la consola muestra tu IP local. Cualquier equipo en la misma red puede acceder:

```
https://<tu-ip>          # HTTPS con certificado auto-firmado
http://<tu-ip>           # HTTP sin cifrar
```

Usa `make ip` para ver tu IP.

## Despues de un pull

```bash
make rebuild     # Reconstruye y arranca
```

## Comandos

| Comando | Descripcion |
|---------|-------------|
| `make up` | Arranca todos los servicios |
| `make down` | Para todos los servicios |
| `make build` | Reconstruye las imagenes Docker |
| `make rebuild` | Reconstruye sin cache y arranca |
| `make restart` | Reinicia todos los servicios |
| `make logs` | Logs en tiempo real (todos) |
| `make logs-web` | Logs solo de la web |
| `make logs-db` | Logs solo de PostgreSQL |
| `make status` | Estado de los contenedores |
| `make db-shell` | Shell psql en la base de datos |
| `make ip` | Muestra tu IP + enlace LAN |
| `make dev` | Servidor local sin Docker |
| `make health` | Health check de los servicios |
| `make clean` | Elimina todo (contenedores, BD, caches) |

## Estructura

```
api/            Backend FastAPI (routers, servicios, modelos)
OEE/            Modulos de calculo OEE (disponibilidad, rendimiento, calidad, oee_secciones)
  db/           Conector IZARO (SQL Server via pyodbc)
templates/      Interfaz web (Jinja2 + Tailwind + Alpine.js)
static/         CSS y JS
data/           Configuracion (ciclos.csv, db_config.json)
caddy/          Reverse proxy HTTPS para acceso LAN
```

## Requisitos

- Docker y Docker Compose
- (Para desarrollo local sin Docker: Python 3.11+, pyodbc, pandas)

## Despliegue en Raspberry Pi

Funciona igual. `make build && make up`. Docker detecta la arquitectura (ARM64) automaticamente.
