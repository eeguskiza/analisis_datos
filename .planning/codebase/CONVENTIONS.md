# Coding Conventions

**Analysis Date:** 2026-04-18

## Naming Patterns

**Files:**
- Module/function files use lowercase with underscores: `pipeline.py`, `email.py`, `oee_secciones`
- Package directories use lowercase: `api`, `OEE`, `services`, `routers`, `scripts`
- Test files follow pytest convention: `test_*.py` or `*_test.py`
- Constants in UPPERCASE: `MIN_PIEZAS_OEE`, `SHIFT_LABELS`, `REF_ROWS_PER_PAGE`, `ORDER_PRIORITY`

**Functions:**
- All functions use snake_case: `calcular_solapamiento()`, `determinar_turno()`, `normalizar_proceso()`, `parse_float()`
- Internal helper functions prefixed with underscore: `_make_raw()`, `_get_smtp_config()`, `_sync_ciclos_to_csv()`
- Query/data access functions often descriptive: `discover_resources()`, `extract_data()`, `get_config()`

**Variables:**
- snake_case for all local and module variables
- Database session variables: `db` or `session`
- Dictionary keys use Spanish descriptive names: `"mensaje"`, `"recurso"`, `"seccion"`, `"disponibilidad_pct"`
- Boolean variables descriptive: `ok`, `activo`
- Timestamp variables: `created_at`, `updated_at`

**Types:**
- PEP 484 style type hints on all function signatures
- Use `from __future__ import annotations` in all modules for forward references
- Optional types written as `Optional[T]` or `T | None`
- Dict types explicit: `Dict[str, float]`, `dict[str, int]`
- Classes inherit from appropriate base: `BaseModel` (Pydantic), `Base` (SQLAlchemy), `DeclarativeBase`

## Code Style

**Formatting:**
- No explicit formatter configured (no .flake8, pyproject.toml, or setup.cfg)
- de facto standard: 4-space indentation, following Python conventions
- Line length appears to be 120+ characters (based on observed code)
- Blank lines: 2 between top-level definitions, 1 between methods

**Linting:**
- No linter configuration detected (no ruff, black, pylint config)
- Conventions enforced by code review: imports organized, docstrings present

**Imports Organization:**
Order observed in codebases:
1. `from __future__ import annotations` (always first if present)
2. Standard library: `import os`, `import csv`, `from pathlib import Path`, `from datetime import datetime`, `from typing import...`
3. Third-party: `from fastapi import FastAPI`, `from sqlalchemy import...`, `from pydantic import...`
4. Local/project: `from api.config import settings`, `from OEE.oee_secciones.main import...`

**Path Aliases:**
- Project uses absolute imports, no relative imports (except `from __future__`)
- Modules imported with full path: `from api.config import settings`, `from OEE.db.connector import...`
- No `.` or `..` relative imports

## Error Handling

**Patterns:**
- Try/except blocks used for external operations (DB, SMTP, file I/O, data parsing)
- Function `parse_float()` returns default (0.0) on parse failure instead of raising
- Function `parse_datetime()` returns None on parse failure instead of raising
- Exceptions logged with context: `msg = f"ERROR en {label}: {exc}"` before yielding to caller
- Database operations wrapped: try/except with _log + yield pattern in `api/services/pipeline.py`

Example from `api/services/pipeline.py`:
```python
try:
    data_rows = db_service.extract_data(fecha_inicio, fecha_fin, recursos=recursos)
    if not data_rows:
        msg = "ERROR: Sin datos para el periodo/recursos indicados."
        _log(msg)
        yield msg
        status = "error"
        return
except Exception as exc:
    msg = f"ERROR extraccion: {exc}"
    _log(msg)
    yield msg
    status = "error"
    return
```

**Error Returns:**
- API functions return tuples: `(ok: bool, message: str, ...)`
- Example: `check_db_health() -> tuple[bool, str]` returns `(True, "OK")` or `(False, error_msg)`
- SMTP functions return string status: `"OK"` or `f"ERROR: {exc}"`
- Database operations commit after modify, no rollback pattern observed

## Logging

**Framework:** Python standard `logging` module

**Patterns:**
- Module-level logger: `logger = logging.getLogger("oee")` in `api/main.py`
- Configured at app start: `logging.basicConfig(level=logging.INFO)`
- Logged on startup: `logger.info("Base de datos inicializada OK")`, `logger.error(f"Error inicializando BD: {exc}")`
- Error tracebacks logged: `logger.error(traceback.format_exc())`
- Pipeline uses local list accumulation: `log_lines: list[str] = []` with `_log(msg)` helper function
- Pipeline yields logs to caller (SSE pattern): `yield msg` after logging

No `print()` statements observed in library code (only in main.py for uncaught exceptions).

## Comments

**When to Comment:**
- Comments used for section headers with `# ── Title ──` separator style
- Inline comments explain non-obvious logic: `# ciclo real = ideal`, `# malas = totales - (malas - recuperadas) = 100 - 15 = 85`
- Comments used on dataclass fields to explain meaning: `# T. Bruto = Producción + Preparación`, `# Avería + Mant. Preventivo + Limpieza`

**Docstrings:**
- Module docstrings present: `"""Pydantic models para requests y responses de la API."""`
- Function docstrings use triple quotes, typically single-line: `"""Devuelve (ok, mensaje, server, database)."""`
- Multi-line docstrings for complex functions:
  ```python
  def run_pipeline(...) -> Generator[str, None, None]:
      """
      Genera mensajes de log paso a paso (para SSE).
      Extrae datos → guarda en BD → genera CSVs temporales → OEE → PDFs temporales.
      """
  ```
- Docstrings are Spanish (language of codebase)
- Type hints in signature, not in docstring

## Function Design

**Size:** 
- Functions range 5-50 lines typically
- Pipeline orchestration functions can be longer (100+ lines with clear sections marked by comments)
- Helper functions extracted from large sections: `_sync_ciclos_to_csv()`, `_parse_pdf_metadata()`, `_save_datos_to_db()`

**Parameters:**
- Type-hinted parameters mandatory
- Positional arguments for required values
- Keyword-only arguments rare (no `*` separator observed)
- Default values used: `modulos: list[str] | None = None`, `source: str = "db"`

**Return Values:**
- Explicit return types: `-> dict`, `-> tuple[bool, str]`, `-> Generator[str, None, None]`
- Generators used for streaming logs: `yield msg` pattern
- Tuple unpacking common: `ok, msg, server, database = check_connection()`
- None used for optional returns: `-> Optional[datetime]`, `-> Path | None`

## Module Design

**Exports:**
- Services explicitly import only needed functions: `from api.services import db as db_service`
- No `__all__` pattern observed
- Router modules follow pattern:
  ```python
  router = APIRouter(tags=["pages"])
  
  @router.get("/path")
  def endpoint(...): ...
  ```

**Barrel Files:**
- Package `__init__.py` files either empty or minimal (not used for re-exports)
- Imports use full module paths

**Dataclasses:**
- Heavy use of `@dataclass` for data structures: `MachineSectionMetrics` in `OEE/oee_secciones/main.py`
- Immutability via `@dataclass(frozen=True)` not observed; mutable dataclasses standard
- Field defaults via `field(default_factory=dict)` for collections

**Pydantic Models:**
- All API request/response models inherit from `BaseModel` in `api/models.py`
- Field defaults: `seccion: str = "GENERAL"`, `activo: bool = True`
- Optional fields: `Optional[list[str]] = None`

---

*Convention analysis: 2026-04-18*
