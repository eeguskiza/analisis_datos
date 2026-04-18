# Architecture

**Analysis Date:** 2026-04-18

## Pattern Overview

**Overall:** Layered + Domain-Driven architecture with separation between HTTP API layer and data processing domain logic.

**Key Characteristics:**
- FastAPI REST API serves HTTP endpoints and static web UI
- Domain logic in `OEE/` module processes production metrics (disponibilidad, rendimiento, calidad)
- Data extraction pipeline bridges IZARO MES database and internal business logic
- SQLAlchemy ORM for configuration/audit database (ecs_mobility)
- Direct SQL Server queries for live production data (dbizaro MES)
- Report generation (PDF) as async orchestration service

## Layers

**HTTP API Layer:**
- Purpose: Handle HTTP requests, serve web UI, orchestrate services
- Location: `api/main.py` (FastAPI app factory), `api/routers/` (endpoint groupings)
- Contains: FastAPI routes, request/response serialization, template rendering
- Depends on: Services layer, OEE domain, configuration
- Used by: Web browsers, MCP server, external tools

**Services Layer:**
- Purpose: Orchestrate cross-cutting operations (database access, pipeline execution, reporting)
- Location: `api/services/`
- Contains: `db.py` (data extraction wrapper), `pipeline.py` (report generation orchestrator), `informes.py` (report navigation), `metrics.py` (metric computation), `email.py` (notifications)
- Depends on: Database connectors, OEE modules, configuration
- Used by: Routers, pipeline orchestration, MCP server

**Domain Logic (OEE):**
- Purpose: Calculate production metrics (OEE components), parse raw production data
- Location: `OEE/` with subdirectories for each metric: `disponibilidad/`, `rendimiento/`, `calidad/`, `oee_secciones/`
- Contains: Pure metric calculation, PDF report generation, data parsing logic
- Depends on: CSV data, matplotlib for visualization, utility functions
- Used by: Pipeline service, tests

**Data Access Layer:**
- Purpose: Connect to two distinct database sources
- Location: `OEE/db/connector.py` (IZARO MES connector), `api/database.py` (ecs_mobility ORM)
- Contains: SQL Server connection pooling, ODBC driver detection, ORM models, schema migration
- Depends on: pyodbc, SQLAlchemy
- Used by: Services layer, direct queries

**Configuration:**
- Purpose: Centralize all configuration from environment variables
- Location: `api/config.py`
- Contains: Pydantic settings class (`Settings`), database credentials, path resolution
- Used by: All layers

## Data Flow

**Pipeline Execution (Core Workflow):**

1. User submits pipeline request → `POST /api/pipeline/run` (router: `api/routers/pipeline.py`)
2. Router delegates to `api.services.pipeline.run_pipeline()`
3. Pipeline service:
   - Creates `Ejecucion` audit record in ecs_mobility
   - Calls `db_service.extract_data()` to pull data from IZARO (dbizaro) via `OEE/db/connector.py`
   - Extracts data as list of dicts with fields: `recurso, seccion, fecha, h_ini, h_fin, tiempo, proceso, incidencia, cantidad, malas, recuperadas, referencia`
   - Saves extraction to `DatosProduccion` table in ecs_mobility
   - Writes cycles from DB to temporary `ciclos.csv`
   - For each enabled module (disponibilidad, rendimiento, calidad, oee_secciones):
     - Calls module's `generar_informes_*()` function from `OEE/[module]/main.py`
     - Module reads CSV data from `data/recursos/[SECCION]/`, processes, generates PDF
   - Yields SSE messages with progress
   - Returns list of PDF file paths to client

**Live Machine Status:**

1. User accesses dashboard → `GET /` → renders `luk4.html`
2. Frontend calls `GET /api/dashboard/recursos`
3. Router queries fmesmic table directly in dbizaro (live machine counters)
4. Returns real-time piece counts, last event time, active/inactive status

**Report Serving:**

1. User requests PDF → `GET /api/informes/pdf/{filepath:path}`
2. Service `api/services/informes.get_pdf_path()` resolves path from `informes/` directory structure
3. Returns file with inline disposition (no forced download)

**State Management:**
- **Execution audit:** Stored in `Ejecucion` table (ecs_mobility) with status transitions (running → completed/error)
- **Configuration:** Stored in `Recurso`, `Ciclo` tables (ecs_mobility) and via `api/config.py` environment
- **Live data:** Cached in-memory with 15-minute TTL in `centro_mando` router (CACHE_TTL = 900s)
- **Generated reports:** Persisted in filesystem under `informes/[date]/[section]/[machine]/`

## Key Abstractions

**Metrics Dataclass:**
- Purpose: Represent aggregated machine performance for a time period (day/shift/month)
- Examples: `OEE/disponibilidad/main.py:DisponibilidadMetrics`, `OEE/oee_secciones/main.py:MachineSectionMetrics`
- Pattern: Dataclass with typed fields for availability %, performance %, quality %, OEE %, time components (hours)

**CSV Pipeline:**
- Purpose: Bridge IZARO extraction and OEE metric calculation
- Flow: IZARO data → `OEE/db/connector.datos_a_csvs()` → `data/recursos/[SECTION]/` → OEE modules read and process
- Format: Headers like `recurso, seccion, fecha, h_ini, h_fin, tiempo, proceso, incidencia, cantidad, malas, recuperadas, referencia`

**Report Templates:**
- Purpose: Render PDF reports with standardized layout
- Location: `data/report_templates/` (user-provided templates), matplotlib figures generated in-memory
- Format: PDF via matplotlib PdfPages backend, multi-page layout per machine

**Configuration Resolution Chain:**
1. Environment variables (OEE_ prefix) take precedence
2. `.env` file loads defaults
3. `data/db_config.json` stores user customizations (IZARO connection, resource mapping)
4. Code defaults in `api/config.Settings`

## Entry Points

**HTTP Server:**
- Location: `server.py` (WSGI entrypoint)
- Triggers: `python server.py` or `uvicorn api.main:app`
- Responsibilities: Start FastAPI app with database initialization, register all routers, serve static files

**Batch Scripts:**
- Location: `scripts/create_oee_db.py`, `scripts/extract_2025.py`, `scripts/migrate_schemas.py`
- Triggers: Manual invocation via `python scripts/*.py`
- Responsibilities: One-time setup (create databases, migrate schemas, extract historical data)

**MCP Server:**
- Location: `mcp/server.py`
- Triggers: `python -m mcp.server` (protocol: stdio)
- Responsibilities: Expose read-only inspection tools to Claude Code (database explorer, health check, live status queries)

**Web UI:**
- Entry point: `GET /` → renders `templates/luk4.html`
- Routes: `/pipeline`, `/historial`, `/recursos`, `/bbdd`, `/ciclos-calc`, `/operarios`, `/datos`, `/capacidad`
- Template engine: Jinja2

## Error Handling

**Strategy:** Comprehensive error logging with user-friendly HTML responses

**Patterns:**
- Global exception handler in `api/main.py` catches all unhandled exceptions, logs full traceback, returns styled HTML error page
- Services layer wraps external calls (database, file I/O) with try-except, yields error messages to SSE
- Router-level validation via Pydantic models catches schema mismatches before logic runs
- Specific HTTPException raises for 404 (resource not found), 400 (bad request)

## Cross-Cutting Concerns

**Logging:** 
- Framework: Python stdlib `logging` module
- Configuration: Set to INFO level in `api/main.py`
- Usage: Loggers named by module (e.g., `logger = logging.getLogger("oee")`)
- Production: Logs go to stdout for docker container capture

**Validation:** 
- User input: Pydantic models in `api/models.py` (PipelineRequest, CicloRow, etc.)
- Database schema: SQLAlchemy declarative models with constraints (UniqueConstraint for ciclos)
- Data extraction: CSV parsing with type coercion (parse_float, parse_datetime in OEE modules)

**Authentication:**
- Current: None (assumes internal network / trusted environment)
- API endpoints are public-accessible within network

**Async/Concurrency:**
- FastAPI async routes support parallel request handling
- Pipeline execution is synchronous (single-threaded) to avoid race conditions on report generation
- Report streams via Server-Sent Events (SSE) for long-running operations

---

*Architecture analysis: 2026-04-18*
