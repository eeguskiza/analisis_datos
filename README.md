# OEE Planta — Informes de produccion

Herramienta interna para generar informes OEE (Overall Equipment Effectiveness) de la planta. Extrae datos de produccion directamente de la BD MES (IZARO/dbizaro) o desde ficheros Excel, calcula metricas de disponibilidad, rendimiento y calidad, y genera informes PDF por maquina y seccion.

## Primer despliegue

```bash
# 1. Clonar el repo
git clone <url> && cd analisis_datos

# 2. Copiar la config de ejemplo
cp .env.example .env

# 3. Construir y arrancar (PostgreSQL + Web)
make build
make up

# 4. Abrir http://localhost:8000
#    - Ir a Ajustes > configurar la conexion al SQL Server (IP, usuario, password)
#    - Ir a Recursos > verificar que los centros de trabajo son correctos
#    - Ir a Pipeline > seleccionar fechas y ejecutar
```

Al primer arranque la app importa automaticamente los datos de `data/ciclos.csv` a la base de datos local.

## Uso diario

```bash
make up          # Arranca los servicios (si estaban parados)
                 # Abrir http://localhost:8000
                 # Cuando termines:
make down        # Para los servicios
```

Si has hecho `make up` y no has hecho `make down`, los servicios siguen corriendo. Solo necesitas abrir el navegador.

## Despues de un pull (actualizacion de codigo)

```bash
make rebuild     # Reconstruye la imagen y arranca
```

## Comandos

| Comando | Descripcion |
|---------|-------------|
| `make up` | Arranca todos los servicios (db + web) |
| `make down` | Para todos los servicios |
| `make build` | Reconstruye las imagenes Docker |
| `make rebuild` | Reconstruye sin cache y arranca |
| `make restart` | Reinicia todos los servicios |
| `make logs` | Logs en tiempo real (todos) |
| `make logs-web` | Logs solo de la web |
| `make logs-db` | Logs solo de PostgreSQL |
| `make status` | Estado de los contenedores |
| `make db-shell` | Shell psql en la base de datos |
| `make dev` | Servidor local sin Docker (desarrollo) |
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
```

## Requisitos

- Docker y Docker Compose
- (Para desarrollo local sin Docker: Python 3.11+, pip)
