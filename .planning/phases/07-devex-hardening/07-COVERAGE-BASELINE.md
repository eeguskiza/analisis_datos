# Coverage Baseline (Phase 7)

Medido: 2026-04-22 (antes de activar cov-fail-under)

**Baseline TOTAL**: 60%

**Decision locked** (conforme user_confirmations 2026-04-21):

- BASELINE medido = 60% → **fail_under=60** en `pyproject.toml` +
  `--cov-fail-under=60` en CI (Plan 07-02).
- No hay que diferir mejora de cobertura: el baseline ya satisface el objetivo
  DEVEX-02 (cobertura mínima 60% en `api/` y `nexo/`). El gate se activa
  exactamente en el piso medido.

## Comando reproducible

```bash
docker compose exec -T web pytest tests/ -q --cov=api --cov=nexo --cov-report=term
```

## Módulos con menor cobertura (informativo)

Top 10 de menor cobertura extraído de la ejecución del 2026-04-22 sobre
`api/` + `nexo/` (stmts reales, miss, %):

| Módulo | Stmts | Miss | Cover |
|--------|-------|------|-------|
| `api/routers/plantillas.py` | 61 | 61 | 0% |
| `api/services/metrics.py` | 205 | 187 | 9% |
| `api/routers/luk4.py` | 177 | 144 | 19% |
| `api/services/informes.py` | 45 | 35 | 22% |
| `api/services/email.py` | 53 | 41 | 23% |
| `api/routers/usuarios.py` | 108 | 80 | 26% |
| `api/services/turnos.py` | 43 | 31 | 28% |
| `nexo/services/factor_auto_refresh.py` | 43 | 31 | 28% |
| `api/routers/operarios.py` | 97 | 69 | 29% |
| `api/routers/recursos.py` | 99 | 70 | 29% |

Nota: el módulo `api/routers/plantillas.py` está a 0% porque el plan de
Mark-III lo marca como experimental (ver PROJECT.md §Out of Scope); no se
invierte en subir su cobertura hasta Mark-IV.

## Notas de ejecución

- Suite ejecutada contra contenedor `analisis_datos-web-1` (imagen
  `analisis_datos-web:latest`, Python 3.11).
- `pytest-cov==6.0.0` + `httpx==0.28.1` instalados ad-hoc para medir; se
  pinearan formalmente en `requirements-dev.txt` en Task 2 (ya están, sólo
  se bumpea ruff y se añaden mypy + pre-commit).
- Tests totales: 271 pasados / 17 skipped / 64 fallidos (todos los fallidos
  son tests de infra que dependen de artefactos Phase 6 presentes sólo en
  el árbol principal; no afectan la medida de cobertura sobre `api/` +
  `nexo/`).
