# External Integrations

**Analysis Date:** 2026-04-18

## APIs & External Services

**MCP Server (Model Context Protocol):**
- MCP 1.0.0+ server at `mcp/server.py` - Read-only query interface for Claude Code
  - SDK/Client: httpx 0.27.0+
  - Exposes tools: `bizaro_list_databases`, `bizaro_query`, `bizaro_preview`, `bizaro_columns` (read-only SELECT only)
  - Configuration: `OEE_API_URL` env var (docker: `http://web:8000`, local: `http://127.0.0.1:8000`)

**Email (SMTP):**
- Office 365 SMTP (smtp.office365.com port 587) - Report distribution
  - SDK/Client: Python standard library `smtplib`
  - Auth: Stored in `OEE.db.connector.load_config()` reading from `data/db_config.json` (SMTP section)
  - Usage: `api/services/email.py` enviar_informes() - sends PDFs to `Contacto` list (test: `test_email.py`)
  - Configuration loaded via: OEE/db/connector.py load_config() → db_config.json

## Data Storage

**Databases:**

**SQL Server (ecs_mobility):**
- Connection: `OEE_DB_SERVER` (192.168.0.4), `OEE_DB_PORT` (1433), `OEE_DB_NAME` (ecs_mobility)
- Client: pyodbc 5.0.0 with ODBC Driver 18 for SQL Server
- Purpose: Primary production database (ECS Mobility app data, read-only from OEE perspective)
- Connection string format: `DRIVER={ODBC Driver 18 for SQL Server};SERVER=<host>,<port>;DATABASE=<db>;UID=<user>;PWD=<password>;TrustServerCertificate=yes;Encrypt=yes;`
- Implementation: `api/database.py` _mssql_creator() function

**SQL Server (dbizaro) - MES:**
- Connection: Same server as ecs_mobility, database name: `OEE_IZARO_DB` (default: "dbizaro")
- Client: pyodbc via OEE/db/connector.py
- Purpose: Manufacturing Execution System (MES) data - production orders, machines, cycle times
- Schema: admuser (tables: fmesdtc, routes, phases, machine specs)
- Accessed by: `api/services/db.py` wrapper around OEE/db/connector functions
- Configuration: `api/config.py` settings.izaro_db

**PostgreSQL 16 (oee_planta):**
- Connection: docker-compose service `db`, internal hostname: `db`, port 5432
- Client: psycopg2-binary 2.9.0 via SQLAlchemy 2.0.0 ORM
- Purpose: Local reference tables and OEE calculation results (ciclos, recursos, contactos, ejecuciones, informes, metricas)
- Schemas: cfg (config tables), oee (execution and metrics tables)
- URL: `postgresql://<user>:<password>@db:5432/<dbname>`
- Models: `api/database.py` models (Ciclo, Recurso, Ejecucion, InformeMeta, DatosProduccion, Contacto, MetricaOEE)

**File Storage:**
- Local filesystem only
- Paths: `/app/data` (contains recursos/, report_templates/, db_config.json) and `/app/informes` (generated PDFs)
- Mounted volumes in docker-compose for persistence: `./data:/app/data`, `./informes:/app/informes`

**Caching:**
- None detected (no Redis, memcached, or caching library imports)

## Authentication & Identity

**Auth Provider:**
- Custom (SQL Server stored credentials + static configuration file)
  - No OAuth, API keys, or JWT token validation detected
  - Approach: Environment variables (OEE_DB_USER, OEE_DB_PASSWORD) + JSON config file (db_config.json)
  - SMTP credentials stored in `data/db_config.json` (server, port, email, password)
  - No frontend authentication - public web interface, protected by network isolation (Caddy reverse proxy)

## Monitoring & Observability

**Error Tracking:**
- None detected (no Sentry, DataDog, or similar integration)

**Logs:**
- Python logging module (api/main.py) with level INFO
- Logger: "oee" root logger
- Output: stdout (will appear in Docker container logs or Uvicorn console)
- Error details: Global exception handler in FastAPI catches all exceptions and logs traceback

## CI/CD & Deployment

**Hosting:**
- Docker Compose (local orchestration) or docker-compose.yml deployment model
- Services: web (FastAPI 8000), db (PostgreSQL 5432), caddy (reverse proxy 80/443), mcp (stdio server)
- No cloud provider detected (AWS, Azure, GCP)

**CI Pipeline:**
- None detected (no GitHub Actions, GitLab CI, Jenkins configs)

## Environment Configuration

**Required env vars:**
```
OEE_DB_SERVER          (SQL Server hostname, e.g., 192.168.0.4)
OEE_DB_PORT            (SQL Server port, default: 1433)
OEE_DB_USER            (SQL Server username)
OEE_DB_PASSWORD        (SQL Server password)
OEE_DB_NAME            (SQL Server database, default: ecs_mobility)
OEE_IZARO_DB           (MES database, default: dbizaro)
OEE_PG_USER            (PostgreSQL user, default: oee)
OEE_PG_PASSWORD        (PostgreSQL password, default: oee)
OEE_PG_DB              (PostgreSQL database, default: oee_planta)
OEE_DATA_DIR           (data directory, default: ./data)
OEE_INFORMES_DIR       (reports directory, default: ./informes)
OEE_API_URL            (for MCP server, e.g., http://web:8000 in Docker, http://127.0.0.1:8000 locally)
```

**Secrets location:**
- `.env` file (git-ignored) - environment variables at runtime
- `data/db_config.json` - persistent JSON file with server, db, user, password, SMTP config (read by OEE/db/connector.py)
- Docker environment variables in `docker-compose.yml` with defaults, overridable via .env file (compose reads .env prefix `OEE_`)

## Webhooks & Callbacks

**Incoming:**
- None detected (no webhook endpoints exposed by API routers)

**Outgoing:**
- Email notifications via SMTP (one-way: OEE system → user mailbox)
  - Triggered by: `/api/email/enviar` endpoint
  - Destination: SMTP server (smtp.office365.com) for distribution

## ODBC Driver Installation

**Critical Infrastructure:**
- `install_odbc.sh` - Local development setup script
  - Installs: Microsoft ODBC Driver 18 for SQL Server, unixodbc-dev
  - Runs: `apt update` → `apt install msodbcsql18`
  - GPG key setup: Microsoft repository key for package verification
  - Must be run before connecting to SQL Server from native Python (non-Docker)

- `Dockerfile` - Automated driver installation for containerized deployments
  - Detects architecture (amd64 vs arm64 for Apple Silicon)
  - Adds Microsoft repository, installs msodbcsql18 during image build
  - Includes DejaVu fonts for matplotlib rendering in Docker

---

*Integration audit: 2026-04-18*
