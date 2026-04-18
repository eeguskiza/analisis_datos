# Testing Patterns

**Analysis Date:** 2026-04-18

## Test Framework

**Runner:**
- pytest (imported in test files but no pytest.ini or pyproject.toml configuration found)
- Tests discovered via pytest naming convention: `tests/test_*.py` and `conftest.py`

**Assertion Library:**
- Standard pytest assertions: `assert m["disponibilidad_pct"] == 100.0`, `assert m["rendimiento_pct"] == 100.0`
- Comparison assertions: `assert m["disponibilidad_pct"] < 100.0`, `assert m["disponibilidad_pct"] > 0.0`
- Floating-point tolerance: `assert abs(m["oee_pct"] - expected_oee) < 0.01`

**Run Commands:**
```bash
pytest                 # Run all tests in tests/ directory
pytest -v             # Verbose output
pytest --tb=short     # Short traceback format
pytest tests/test_oee_calc.py  # Run specific test file
pytest tests/test_oee_calc.py::TestOEEPerfecto::test_100_pct  # Run specific test class/method
```

No coverage configuration found in Makefile or config files. Manual coverage runs possible:
```bash
pytest --cov=OEE --cov-report=term-missing
```

## Test File Organization

**Location:**
- Tests co-located in separate `tests/` directory at project root (`/home/eeguskiza/analisis_datos/tests/`)
- Not co-located with source code (not `api/tests/`, `OEE/tests/`)

**Naming:**
- Test files: `test_oee_calc.py`, `test_oee_helpers.py`
- Test functions: `test_100_pct()`, `test_indisponibilidad_reduce_disponibilidad()`
- Test classes: `TestOEEPerfecto`, `TestDisponibilidad`, `TestRendimiento` (group related tests)

**Structure:**
```
tests/
├── __init__.py              # Empty
├── conftest.py              # Shared fixtures
├── test_oee_calc.py         # Core OEE calculation tests
└── test_oee_helpers.py      # Helper function tests
```

## Test Structure

**Suite Organization:**
Tests use class-based organization (pytest-style) with shared setup:

```python
class TestOEEPerfecto:
    """Caso ideal: todo funciona sin perdidas."""

    def test_100_pct(self):
        raw = _make_raw(
            horas_produccion=8.0,
            horas_preparacion=0.0,
            horas_indisponibilidad=0.0,
            horas_paros=0.0,
            tiempo_ideal=8.0,  # ciclo real = ideal
            piezas_totales=100.0,
            piezas_malas=0.0,
            piezas_recuperadas=0.0,
        )
        m = convertir_raw_a_metricas(raw)
        assert m["disponibilidad_pct"] == 100.0
        assert m["rendimiento_pct"] == 100.0
        assert m["calidad_pct"] == 100.0
        assert m["oee_pct"] == 100.0
```

**Patterns:**
- **Setup**: Tests create fixtures inline (no pytest.fixture decorators). Helper `_make_raw()` creates test data with defaults:
  ```python
  def _make_raw(**overrides):
      """Helper: crea raw metricas con defaults sensatos."""
      raw = crear_raw_metricas()
      raw.update(overrides)
      return raw
  ```
- **Teardown**: None observed (pure functions tested, no state to clean)
- **Assertion**: Direct value checks with descriptive assertion messages. Example:
  ```python
  assert m["disponibilidad_pct"] == 100.0  # No message, relies on test name for clarity
  ```

## Mocking

**Framework:** None detected (all tests are unit tests of pure functions)

**Patterns:**
- No mock imports (`from unittest.mock`, `pytest-mock`)
- No HTTP mocking, database mocking, or external service mocking
- Tests depend only on `OEE.oee_secciones.main` functions and local helpers

**What to Mock:**
- Not applicable — no external dependencies in test scope

**What NOT to Mock:**
- Core OEE calculation functions tested directly (pure functions with no side effects)

## Fixtures and Factories

**Test Data:**
Helper factory `_make_raw()` creates base dictionary with sensible defaults:

```python
def _make_raw(**overrides):
    """Helper: crea raw metricas con defaults sensatos."""
    raw = crear_raw_metricas()  # Creates dict with all zeros
    raw.update(overrides)       # Apply test-specific values
    return raw
```

Usage:
```python
raw = _make_raw(
    horas_produccion=8.0,
    horas_preparacion=0.0,
    # ... rest of overrides
)
m = convertir_raw_a_metricas(raw)
```

**Location:**
- Helpers defined in same test file (not in conftest.py)
- `conftest.py` contains only module-level setup (sys.path manipulation)

**Shared Fixtures:**
From `tests/conftest.py`:
```python
"""Fixtures compartidos para tests OEE."""
import sys
from pathlib import Path

# Asegurar que el proyecto esta en el path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

This ensures the project root is in Python path so imports work.

## Coverage

**Requirements:** Not enforced (no coverage target configured in Makefile, pytest.ini, or CI)

**Actual Coverage:** Estimated 40-50% based on test file count vs codebase size
- Tests cover: OEE calculation core (`OEE.oee_secciones.main`) and helper functions
- Not covered: API routes, database operations, services, email, pipeline orchestration
- Not covered: Database models, configuration

**View Coverage:**
```bash
pytest --cov=OEE --cov-report=html
pytest --cov=OEE --cov-report=term-missing
```

Coverage commands not integrated into Makefile (no `make test-coverage` target).

## Test Types

**Unit Tests:**
- **Scope:** Pure functions in `OEE.oee_secciones.main` module
- **Approach:** Input → Assert Output (no side effects, no external dependencies)
- **Examples:** 
  - `TestOEEPerfecto::test_100_pct` — calculates OEE with perfect metrics
  - `TestDisponibilidad::test_indisponibilidad_reduce_disponibilidad` — availability loss
  - `TestCasosLimite::test_todo_cero` — edge case with all zero inputs
  - `TestCasosLimite::test_pocas_piezas_anula_oee` — validation of minimum piece count

**Integration Tests:**
- None found in test suite
- Would need: Database setup, API endpoints, external service mocks
- Gap: No tests for `api/routers/`, `api/services/`, `api/database.py`, or `api/main.py`

**E2E Tests:**
- None found in test suite
- Gap: No end-to-end tests for pipeline execution or critical user workflows
- Manual testing required for: data extraction → report generation → PDF output

## Common Patterns

**Async Testing:**
Not applicable — codebase uses sync FastAPI with `uvicorn`, no async tests needed.

**Error Testing:**
Tests verify edge cases and validation:

```python
def test_sin_ciclo_ideal(self):
    raw = _make_raw(
        horas_produccion=8.0,
        tiempo_ideal=0.0,  # no hay ciclo configurado
        piezas_totales=100.0,
    )
    m = convertir_raw_a_metricas(raw)
    assert m["rendimiento_pct"] == 0.0  # Division by zero handled gracefully
```

```python
def test_pocas_piezas_anula_oee(self):
    raw = _make_raw(
        horas_produccion=1.0,
        tiempo_ideal=1.0,
        piezas_totales=float(MIN_PIEZAS_OEE - 1),
    )
    m = convertir_raw_a_metricas(raw)
    # Con pocas piezas el OEE se anula
    assert m["oee_pct"] == 0.0
```

## Test Dependencies

**Imports in Test Files:**

From `tests/test_oee_calc.py`:
```python
import pytest
from OEE.oee_secciones.main import (
    convertir_raw_a_metricas,
    crear_raw_metricas,
    clamp_pct,
    MIN_PIEZAS_OEE,
)
```

From `tests/test_oee_helpers.py`:
```python
import pytest
from datetime import datetime, time
from OEE.oee_secciones.main import (
    calcular_solapamiento,
    determinar_turno,
    normalizar_proceso,
    clasificar_incidencia,
    dividir_en_turnos,
    parse_time_value,
)
```

**No External Dependencies:**
- No database connections in tests
- No file I/O
- No API calls
- No email/SMTP
- Only pure Python + pytest

## Testing Best Practices

**Currently Applied:**
1. Pure function testing (functions with no side effects)
2. Descriptive test names (`test_indisponibilidad_reduce_disponibilidad`)
3. Class-based test grouping by concern (TestDisponibilidad, TestRendimiento, TestCalidad)
4. Helper factories for test data (`_make_raw()`)
5. Edge case coverage (TestCasosLimite class)
6. Floating-point tolerance for numeric comparisons

**Gaps:**
1. No integration tests for API endpoints
2. No database operation tests
3. No pipeline orchestration tests
4. No mocking for external services (SMTP, database)
5. No CI/CD test automation (no test target in Makefile)
6. No coverage requirement enforcement
7. No pytest fixtures (inline test data only)
8. No parametrized tests (no `@pytest.mark.parametrize`)

---

*Testing analysis: 2026-04-18*
