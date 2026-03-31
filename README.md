# OEE Planta — Informes de produccion

Herramienta interna para generar informes OEE (Overall Equipment Effectiveness) de la planta. Extrae datos de produccion directamente de la BD MES (IZARO/dbizaro) o desde ficheros Excel, calcula metricas de disponibilidad, rendimiento y calidad, y genera informes PDF por maquina y seccion.

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

Al primer arranque la app importa automaticamente `data/ciclos.csv` a la base de datos local.

## Uso diario

```bash
make up          # Arranca los servicios (si estaban parados)
make down        # Para los servicios
```

Si no has hecho `make down`, los servicios siguen corriendo. Solo abre el navegador.

## Acceso desde otros equipos (LAN)

Al hacer `make up`, la consola muestra tu IP local. Cualquier equipo en la misma red puede acceder:

```
https://<tu-ip>          # HTTPS con certificado auto-firmado
http://<tu-ip>           # HTTP sin cifrar
```

El navegador avisara del certificado auto-firmado — es normal, dale a "Continuar". Usa `make ip` para ver tu IP.

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
| `make logs-mcp` | Logs solo del MCP server |
| `make status` | Estado de los contenedores |
| `make db-shell` | Shell psql en la base de datos |
| `make ip` | Muestra tu IP + enlace LAN |
| `make dev` | Servidor local sin Docker |
| `make install` | Instala dependencias Python |
| `make health` | Health check de los servicios |
| `make clean` | Elimina todo (contenedores, BD, caches) |
| `make help` | Muestra todos los comandos |

## Estructura

```
api/            Backend FastAPI (routers, servicios, modelos)
OEE/            Modulos de calculo OEE (disponibilidad, rendimiento, calidad, oee_secciones)
templates/      Interfaz web (Jinja2 + Tailwind + HTMX + Alpine.js)
static/         CSS y JS
data/           Datos de entrada (ciclos, config BD, CSVs)
informes/       PDFs generados (por fecha/seccion/maquina)
caddy/          Reverse proxy HTTPS para acceso LAN
mcp/            MCP server para integracion con Claude
```

## Requisitos

- Docker y Docker Compose
- (Para desarrollo local sin Docker: Python 3.11+, pip)

## Despliegue en Raspberry Pi

Funciona igual. Clona el repo, `make build && make up`. Docker detecta la arquitectura (ARM64) automaticamente. El Dockerfile soporta tanto Intel (amd64) como ARM (arm64/Apple Silicon/RPi).
