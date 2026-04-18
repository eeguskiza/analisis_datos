# Technology Stack

**Analysis Date:** 2026-04-18

## Languages

**Primary:**
- Python 3.11 - Core data analysis and API backend
- JavaScript/HTML/CSS - Web frontend (templates, static assets)
- SQL - Database queries (SQL Server, PostgreSQL)

**Secondary:**
- Bash - Installation and deployment scripts
- JSON - Configuration and data serialization

## Runtime

**Environment:**
- Docker containers (Debian Bookworm base) for containerized deployment
- Native Linux/WSL support via `install_odbc.sh` for local development

**Package Manager:**
- pip (Python package management)
- Lockfile: Not detected (uses requirements.txt with pinned versions)

## Frameworks

**Core:**
- FastAPI 0.109.0+ - Web API framework with async support
- Uvicorn 0.27.0+ - ASGI application server

**Data Processing:**
- pandas 1.4.0+ - Data manipulation and CSV processing
- matplotlib 3.5.0+ - Chart generation for reports (backend: Agg in Docker)

**Database:**
- SQLAlchemy 2.0.0 - ORM for PostgreSQL tables
- pyodbc 5.0.0 - ODBC bridge to SQL Server (dbizaro MES)
- psycopg2-binary 2.9.0 - PostgreSQL adapter

**Templating & Forms:**
- Jinja2 3.1.0 - Server-side HTML templating
- python-multipart 0.0.6 - Form data parsing
- pydantic-settings 2.0.0 - Environment-based configuration

**MCP:**
- mcp 1.0.0+ - Model Context Protocol server (requires Python 3.11+)
- httpx 0.27.0+ - Async HTTP client for API calls

## Key Dependencies

**Critical:**
- pyodbc 5.0.0 - Bridges FastAPI to SQL Server via ODBC Driver 18 (primary production database source)
- SQLAlchemy 2.0.0 - Manages PostgreSQL schema (local reference tables: ciclos, recursos, contactos, ejecuciones, informes, metricas)
- psycopg2-binary 2.9.0 - Connects to PostgreSQL in docker-compose

**Infrastructure:**
- uvicorn[standard] 0.27.0 - Production-grade ASGI server with SSL/reload support

## Configuration

**Environment:**
- Pydantic BaseSettings reads from `.env` file with `OEE_` prefix (see `api/config.py`)
- Key variables (required):
  - `OEE_DB_SERVER` - SQL Server hostname (default: "192.168.0.4")
  - `OEE_DB_PORT` - SQL Server port (default: 1433)
  - `OEE_DB_USER` - SQL Server username
  - `OEE_DB_PASSWORD` - SQL Server password
  - `OEE_DB_NAME` - SQL Server database name (default: "ecs_mobility")
  - `OEE_IZARO_DB` - MES database name (default: "dbizaro")
  - `OEE_PG_USER` - PostgreSQL username (default: "oee")
  - `OEE_PG_PASSWORD` - PostgreSQL password (default: "oee")
  - `OEE_PG_DB` - PostgreSQL database name (default: "oee_planta")
  - `OEE_DATA_DIR` - Local data directory (default: `./data`)
  - `OEE_INFORMES_DIR` - Reports directory (default: `./informes`)

**Build:**
- `Dockerfile` - Multi-stage build, installs ODBC Driver 18 for SQL Server + matplotlib fonts
- `docker-compose.yml` - Orchestrates web (FastAPI), db (PostgreSQL 16), caddy (reverse proxy), mcp (model context protocol server)
- MCP Dockerfile at `mcp/Dockerfile` - Separate container for Claude Code integration

## Platform Requirements

**Development:**
- Linux/WSL with `install_odbc.sh` to install Microsoft ODBC Driver 18 and unixodbc-dev
- Python 3.11 (local or via Docker)
- PostgreSQL client libraries (psycopg2-binary) for local pg connections

**Production:**
- Docker Compose v1.29+ (for service orchestration)
- Caddy 2-alpine reverse proxy (handles HTTP/HTTPS on ports 80/443 with self-signed certificates)
- PostgreSQL 16-alpine database container
- Network isolation: web, db, caddy, mcp services on Docker internal network
- Volumes for persistence: `pgdata`, `caddy_data`, `caddy_config`, mounted `./data`, `./informes`

---

*Stack analysis: 2026-04-18*
