---
phase: 7
slug: devex-hardening
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-21
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (ya operativo) + pre-commit + GitHub Actions |
| **Config file** | `pyproject.toml` (nuevo o extendido), `.pre-commit-config.yaml` (nuevo) |
| **Quick run command** | `pytest -x -q tests/ --cov=api --cov=nexo --cov-fail-under=<baseline>` |
| **Full suite command** | `make test && make lint` |
| **Estimated runtime** | ~60-90 segundos (test suite actual) + ~10s pre-commit hooks |

Phase 7 es tooling-heavy: validación por **presencia de config + ejecución idempotente** de herramientas. Tests de app existentes (222 tests Phase 5 + 73 tests Phase 6) sirven como smoke de que los hooks no rompieron nada.

---

## Sampling Rate

- **After every task commit:** `pre-commit run --files <files changed>` (si hook instalado) + `pytest -x tests/infra/` si son tests infra
- **After every plan wave:** `make test && make lint` full suite
- **Before `/gsd-verify-work`:** Full suite green + `pre-commit run --all-files` passing + CI green on pushed branch
- **Max feedback latency:** 90 segundos

---

## Per-Task Verification Map

> Plan/task IDs se asignan durante planning — skeleton inicial:

| Task ID | Plan | Wave | Requirement | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------------|-----------|-------------------|-------------|--------|
| 07-00-TBD | 01 | 0 | DEVEX-02 baseline | Medir coverage actual api/+nexo/ | tooling | `pytest --cov=api --cov=nexo --cov-report=term tests/ \| tail -5` | ❌ W0 | ⬜ pending |
| 07-01-TBD | 01 | 1 | DEVEX-01 | .pre-commit-config.yaml valido + ruff/mypy config en pyproject | config | `pre-commit validate-config && pre-commit run --all-files --hook-stage manual` | ❌ W0 | ⬜ pending |
| 07-01-TBD | 01 | 1 | DEVEX-01 backfill | black/ruff --fix aplicado en commit aislado antes de activar hooks | git | `git log --oneline \| grep -q 'chore(07): backfill'` + `ruff check api/ nexo/` exit 0 | ❌ W0 | ⬜ pending |
| 07-02-TBD | 02 | 2 | DEVEX-02 | CI matrix 3.11+3.12 + coverage gate + smoke | config | `grep -q 'python-version.*3.11.*3.12' .github/workflows/ci.yml && grep -q 'cov-fail-under' .github/workflows/ci.yml` | ❌ W0 | ⬜ pending |
| 07-02-TBD | 02 | 2 | DEVEX-03 | Makefile test/lint/format/migrate targets | make | `make -n test && make -n lint && make -n format && make -n migrate` | ❌ W0 | ⬜ pending |
| 07-03-TBD | 03 | 3 | DEVEX-04 | ARCHITECTURE.md exists, Mermaid con 3 engines | docs | `test -f docs/ARCHITECTURE.md && grep -q 'engine_mes' docs/ARCHITECTURE.md && grep -q 'engine_app' docs/ARCHITECTURE.md && grep -q 'engine_nexo' docs/ARCHITECTURE.md && grep -q '\`\`\`mermaid' docs/ARCHITECTURE.md` | ❌ W0 | ⬜ pending |
| 07-03-TBD | 03 | 3 | DEVEX-05 | RUNBOOK.md 5 escenarios exactos | docs | `test -f docs/RUNBOOK.md && [ $(grep -cE '^## Escenario [1-5]:' docs/RUNBOOK.md) -eq 5 ]` | ❌ W0 | ⬜ pending |
| 07-03-TBD | 03 | 3 | DEVEX-06 | RELEASE.md con checklist semver | docs | `test -f docs/RELEASE.md && grep -qE 'v[0-9]+\.[0-9]+\.[0-9]+' docs/RELEASE.md && grep -q 'scripts/deploy.sh' docs/RELEASE.md` | ❌ W0 | ⬜ pending |
| 07-04-TBD | 04 | 4 | DEVEX-07 | CLAUDE.md Última revisión actualizada post-Mark-III | docs | `grep -q 'Última revisión: 2026-0' CLAUDE.md && grep -q 'make prod-' CLAUDE.md && grep -q 'pre-commit' CLAUDE.md` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Medir coverage baseline actual: `pytest --cov=api --cov=nexo --cov-report=term tests/ 2>&1 | tail -30` — registrar resultado en PLAN.md para decidir `--cov-fail-under=N`
- [ ] `pip install pre-commit ruff mypy pytest-cov` — deps dev añadidas a `requirements-dev.txt`
- [ ] Verificar `tests/infra/` sigue pasando (73 tests Phase 6) — no regresión por nuevas deps
- [ ] `tests/infra/test_pre_commit_config.py` — valida `.pre-commit-config.yaml` existe, hooks correctos, scope api/+nexo/
- [ ] `tests/infra/test_pyproject_config.py` — valida `[tool.ruff]` + `[tool.mypy]` + `[tool.coverage]` blocks presentes con config mínima
- [ ] `tests/infra/test_ci_matrix.py` — valida `.github/workflows/ci.yml` tiene matriz 3.11+3.12 y cov-fail-under

Dependencias: `pre-commit`, `ruff`, `mypy`, `pytest-cov` (añadir a `requirements-dev.txt`).

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Hook bloquea commit sucio | DEVEX-01 | Requiere commit real (no test) | `echo "x = 1  " > /tmp/test.py && cp /tmp/test.py api/_test.py && git add api/_test.py && git commit -m "test"` — debe fallar con ruff format; `rm api/_test.py` después |
| CI matriz efectivamente ejecuta ambos pythons | DEVEX-02 | Requiere GH Actions real | Push trigger → verificar en GitHub Actions UI que ambos jobs aparecen y pasan |
| Runbook es seguible por segundo admin | DEVEX-05 | Requiere persona diferente | Admin secundario sigue `docs/RUNBOOK.md` escenario 3 (cert expira) — debe restaurar sin contexto del autor |
| RELEASE.md produce tag válido | DEVEX-06 | Requiere release real | Seguir checklist → tag v0.9.0-rc1 (pre-release) → verificar `git tag -l 'v*'` lista y GH Release renderiza |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter (pending — after planner wires tasks)

**Approval:** pending
