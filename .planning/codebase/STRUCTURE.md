# Codebase Structure

**Analysis Date:** 2026-04-18

## Directory Layout

```
analisis_datos/
├── server.py                 # WSGI entrypoint (uvicorn runner)
├── api/                      # HTTP API layer (FastAPI)
│   ├── main.py              # FastAPI app factory, lifespan, exception handlers
│   ├── config.py            # Pydantic Settings (centralized configuration)
│   ├── database.py          # SQLAlchemy ORM models + engine (ecs_mobility)
│   ├── models.py            # Pydantic request/response schemas
│   ├── deps.py              # Shared dependencies (templates)
│   ├── routers/             # HTTP endpoint groupings
│   │   ├── pages.py         # HTML page rendering (Jinja2 templates)
│   │   ├── centro_mando.py  # Live machine status dashboard
│   │   ├── pipeline.py      # Report generation orchestration
│   │   ├── informes.py      # Report listing, serving, deletion
│   │   ├── ciclos.py        # Machine cycle time configuration
│   │   ├── recursos.py      # Machine/resource CRUD
│   │   ├── conexion.py      # Database connectivity checks
│   │   ├── historial.py     # Historical data browsing
│   │   ├── bbdd.py          # Database explorer (tables, columns, preview)
│   │   ├── datos.py         # Raw production data export
│   │   ├── email.py         # Email notifications
│   │   ├── operarios.py     # Operator tracking
│   │   ├── luk4.py          # LUK4 machine-specific dashboard
│   │   ├── capacidad.py     # Capacity planning
│   │   └── [others].py      # Additional domain-specific routers
│   └── services/            # Business logic layer
│       ├── db.py            # Wrapper around OEE.db.connector (data extraction)
│       ├── pipeline.py      # Report generation orchestration
│       ├── informes.py      # Report file management
│       ├── metrics.py       # Metric calculations (helper)
│       ├── email.py         # Email service
│       └── turnos.py        # Shift management
├── OEE/                      # Domain logic: OEE metric calculation
│   ├── __init__.py
│   ├── db/
│   │   ├── connector.py     # IZARO MES connector (direct SQL Server access)
│   │   └── __init__.py
│   ├── disponibilidad/      # Availability metric module
│   │   ├── main.py          # Calculate + report generation
│   │   └── __init__.py
│   ├── rendimiento/         # Performance metric module
│   │   ├── main.py
│   │   └── __init__.py
│   ├── calidad/             # Quality metric module
│   │   ├── main.py
│   │   └── __init__.py
│   ├── oee_secciones/       # Composite OEE metric (all 3 components)
│   │   ├── main.py          # ~1624 lines: metric calculation, PDF report
│   │   └── __init__.py
│   └── utils/
│       ├── data_files.py    # CSV location helpers (data/recursos/[SECTION]/)
│       ├── cycles_registry.py # Cycle time registry
│       ├── excel_import.py  # Import Excel data + resource mapping
│       └── __init__.py
├── mcp/                      # MCP server for Claude Code integration
│   ├── server.py            # MCP tools: DB explorer, health check, live status
│   └── __init__.py
├── scripts/                  # One-time batch operations
│   ├── create_oee_db.py     # Initialize local databases
│   ├── extract_2025.py      # Extract historical 2025 data
│   ├── migrate_schemas.py   # Schema migrations
│   └── [others].py
├── templates/               # Jinja2 HTML templates
│   ├── luk4.html           # LUK4 dashboard (index page)
│   ├── pipeline.html       # Report generation interface
│   ├── historial.html      # Historical report browser
│   ├── recursos.html       # Resource configuration
│   ├── ciclos.html         # Cycle time management
│   ├── bbdd.html           # Database explorer
│   ├── operarios.html      # Operator management
│   ├── datos.html          # Data export interface
│   ├── capacidad.html      # Capacity planning
│   ├── base.html           # Base layout (if used)
│   └── [others].html
├── static/                  # Static assets
│   ├── css/                 # Stylesheets
│   ├── js/                  # JavaScript (frontend logic, API calls)
│   └── img/                 # Images (logo, icons)
├── data/                    # Working data directory
│   ├── ciclos.csv          # Cycle time lookup table (maquina, referencia, tiempo_ciclo)
│   ├── db_config.json      # IZARO database connection config
│   ├── db_config.example.json # Template
│   ├── ecs-logo.png        # Company logo
│   ├── oee.db              # Local SQLite cache (optional)
│   ├── report_templates/   # User-provided PDF report templates
│   ├── recursos/           # Temporary CSV data per section
│   │   ├── LINEAS/         # CSV files extracted from IZARO for LINEAS section
│   │   ├── TALLADORAS/     # CSV files for TALLADORAS section
│   │   └── [OTHER SECTIONS]/
│   ├── export_2025_*.csv   # Cached historical extracts
│   └── [other data files]
├── informes/               # Generated PDF reports (organized by date)
│   └── 2026-04-16/         # Date-organized directory
│       ├── LINEAS/         # Section directory
│       │   ├── luk1/       # Machine directory
│       │   │   ├── disponibilidad_luk1_2026-04-16.pdf
│       │   │   ├── rendimiento_luk1_2026-04-16.pdf
│       │   │   ├── calidad_luk1_2026-04-16.pdf
│       │   │   └── oee_secciones_luk1_2026-04-16.pdf
│       │   ├── luk2/
│       │   └── [other machines]
│       └── TALLADORAS/
├── tests/                  # Test suite
│   ├── conftest.py         # pytest fixtures and configuration
│   ├── test_oee_calc.py    # Tests for OEE metric calculation functions
│   ├── test_oee_helpers.py # Tests for utility functions
│   └── __init__.py
├── docs/                   # Documentation
│   └── [markdown files]
├── caddy/                  # Reverse proxy configuration
│   ├── Caddyfile           # Caddy server config (HTTPS, routing)
│   └── [cert files]
├── .planning/              # Generated planning documents
│   └── codebase/
│       ├── ARCHITECTURE.md # Architecture pattern and layers
│       ├── STRUCTURE.md    # This file
│       ├── CONVENTIONS.md  # Coding conventions
│       ├── TESTING.md      # Test organization and patterns
│       ├── STACK.md        # Technology stack
│       ├── INTEGRATIONS.md # External service integrations
│       └── CONCERNS.md     # Known issues and technical debt
├── Dockerfile             # Container image definition
├── docker-compose.yml     # Multi-container orchestration
├── Makefile              # Development task automation
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (secrets, not committed)
├── .env.example          # Template for .env
├── .gitignore            # Git ignore rules
├── README.md             # Project overview
└── ERRORES.md            # Known issues log
```

## Directory Purposes

**api/:**
- Purpose: HTTP request handling, routing, serialization, dependency injection
- Contains: FastAPI application, route definitions, business service orchestration
- Key files: `main.py` (app factory), `routers/` (endpoint groups), `services/` (logic)

**OEE/:**
- Purpose: Production metric domain logic (calculations, report generation)
- Contains: Metric calculation functions, PDF report generation, data parsing
- Key files: `oee_secciones/main.py` (core metric engine, ~1600 lines), `db/connector.py` (data extraction)

**mcp/:**
- Purpose: MCP protocol server for LLM integration and database inspection
- Contains: Tool definitions, API queries, database explorer
- Key files: `server.py` (tool implementations)

**scripts/:**
- Purpose: One-time batch operations, migrations, data imports
- Contains: Schema creation, data extraction, initialization routines
- No shared imports—each script is standalone

**templates/:**
- Purpose: Server-side rendered HTML (Jinja2)
- Contains: Page layouts, forms, dashboard views
- Rendering: Happens in `api/routers/pages.py` via dependency-injected `templates` object

**static/:**
- Purpose: Browser-loaded assets (CSS, JavaScript, images)
- Contains: Frontend logic, styling, media
- Mount point: `/static/` in FastAPI

**data/:**
- Purpose: Working directory for configuration, lookups, and temporary processing
- Contains: CSV files (cycles, exports), database connection config, logo
- Subdirectories:
  - `recursos/`: Temporary CSV staging area (one directory per section: LINEAS, TALLADORAS, etc.)
  - `report_templates/`: User-provided PDF templates

**informes/:**
- Purpose: Archive of generated PDF reports
- Structure: `informes/[YYYY-MM-DD]/[SECTION]/[MACHINE]/[MODULE]_[MACHINE]_[DATE].pdf`
- Lifecycle: Created by pipeline, served by informes router, deletable by date

**tests/:**
- Purpose: Unit and integration test suite
- Framework: pytest
- Test discovery: `test_*.py` files in directory

## Key File Locations

**Entry Points:**
- `server.py`: Main application launcher (entrypoint for `python server.py`)
- `api/main.py`: FastAPI app instance and lifespan management
- `mcp/server.py`: MCP protocol server for LLM access

**Configuration:**
- `api/config.py`: Pydantic `Settings` class (single source of truth for all config)
- `.env`: Environment variables (database credentials, host, port, debug flag)
- `data/db_config.json`: IZARO MES connection parameters (user-configurable)

**Core Logic:**
- `OEE/oee_secciones/main.py`: Metric calculation engine (~1624 lines)
- `OEE/db/connector.py`: IZARO data extraction (SQL Server direct access)
- `api/services/pipeline.py`: Pipeline orchestrator (extraction → metric calc → report gen)

**Database:**
- `api/database.py`: SQLAlchemy ORM for ecs_mobility (Ciclo, Recurso, Ejecucion, DatosProduccion, InformeMeta models)
- `OEE/db/connector.py`: Direct SQL queries to dbizaro (IZARO MES production data)

**Testing:**
- `tests/conftest.py`: pytest configuration and shared fixtures
- `tests/test_oee_calc.py`: Unit tests for metric calculation (pure functions)
- `tests/test_oee_helpers.py`: Tests for utility functions

## Naming Conventions

**Files:**
- Python modules: `snake_case.py` (e.g., `oee_secciones.py`, `db_config.json`)
- Routers: Feature-named, singular (e.g., `resources.py` → controls `/recursos` endpoint)
- Templates: Descriptive with `.html` extension (e.g., `luk4.html`, `pipeline.html`)
- Database tables: Plural snake_case (e.g., `ciclos`, `recursos`, `ejecuciones`)

**Directories:**
- API layers: Lowercase with underscores (`routers/`, `services/`)
- Domain modules: UPPERCASE for section names (LINEAS, TALLADORAS) in `data/recursos/`
- Report structure: Date format ISO (2026-04-18), section names, machine names (e.g., `luk1`)

**Functions:**
- Module functions: `snake_case` (e.g., `generar_informes_disponibilidad`, `calcular_oee`)
- Class methods: `snake_case` with leading `_` for private (e.g., `_sync_ciclos_to_csv`)

**Classes:**
- Dataclasses: `PascalCase` (e.g., `DisponibilidadMetrics`, `MachineSectionMetrics`)
- Pydantic models: `PascalCase` (e.g., `PipelineRequest`, `ConnectionStatus`)
- FastAPI routers: Plural lowercase (e.g., `router = APIRouter(prefix="/ciclos")`)

**Variables:**
- Constants: `UPPER_CASE` (e.g., `CACHE_TTL`, `MIN_PIEZAS_OEE`)
- Module-level: `snake_case`, private with leading `_` (e.g., `_MODULE_MAP`, `_cache`)

## Where to Add New Code

**New Feature (e.g., new production metric):**
- Primary code: `OEE/[new_module]/main.py` (create new directory under `OEE/`)
  - Implement `generar_informes_[new_module](data_dir, report_dir, **kwargs)` function
  - Return list of PDF file paths
  - Use matplotlib for PDF generation (Agg backend)
- Router: `api/routers/[feature].py` (new router file)
  - Register in `api/main.py` via `app.include_router([new_router])`
- Template (if UI needed): `templates/[feature].html`
- Tests: `tests/test_[new_module].py`
- Service (if orchestration needed): `api/services/[feature].py`

**New Endpoint:**
- Router: Add to appropriate file in `api/routers/` or create new router
- Service: If complex logic, add helper to `api/services/` and import in router
- Model: Add Pydantic schema to `api/models.py` for request/response types
- Registration: Import router in `api/main.py` and call `app.include_router()`

**New Component/Module:**
- Pure calculation: `OEE/utils/` (if generic), or `OEE/[module]/` (if specific to metric)
- Business logic: `api/services/` (if service-level)
- Data model: `api/database.py` (if persistent), or `api/models.py` (if transient)

**Utilities:**
- Shared helpers: `OEE/utils/` (for domain logic), `api/services/` (for API layer)
- Example: Data parsing → `OEE/utils/excel_import.py`, configuration → `api/config.py`

**Tests:**
- Unit tests: `tests/test_[module].py` with `@pytest.mark.unit` decorator
- Integration tests: `tests/test_[module].py` with `@pytest.mark.integration` decorator
- Run with: `pytest --cov=src --cov-report=term-missing` for coverage

## Special Directories

**data/recursos/:**
- Purpose: Temporary CSV staging during pipeline execution
- Generated: Yes (dynamically during `OEE/db/connector.py:datos_a_csvs()`)
- Committed: No (git-ignored, regenerated each pipeline run)
- Structure: One subdirectory per section (LINEAS, TALLADORAS) with CSV files per resource

**informes/:**
- Purpose: Persistent PDF report archive
- Generated: Yes (by OEE module functions during pipeline)
- Committed: Yes (organized by date for historical reference, but can be large)
- Cleanup: Manual via `DELETE /api/informes/{date_str}` endpoint

**.planning/codebase/:**
- Purpose: Architecture and structure documentation (this directory)
- Generated: Yes (by gsd-map-codebase command)
- Committed: Yes (reference for development planning)
- Contents: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, STACK.md, INTEGRATIONS.md, CONCERNS.md

---

*Structure analysis: 2026-04-18*
