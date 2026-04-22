---
phase: 07-devex-hardening
reviewed: 2026-04-22T00:00:00Z
depth: standard
files_reviewed: 16
files_reviewed_list:
  - pyproject.toml
  - .pre-commit-config.yaml
  - requirements-dev.txt
  - .github/workflows/ci.yml
  - Makefile
  - CLAUDE.md
  - CHANGELOG.md
  - docs/ARCHITECTURE.md
  - docs/RUNBOOK.md
  - docs/RELEASE.md
  - tests/infra/test_ci_matrix.py
  - tests/infra/test_claude_md.py
  - tests/infra/test_devex_docs.py
  - tests/infra/test_makefile_devex.py
  - tests/infra/test_pre_commit_config.py
  - tests/infra/test_pyproject_config.py
findings:
  critical: 0
  warning: 4
  info: 9
  total: 13
status: issues_found
---

# Phase 7: Code Review Report

**Reviewed:** 2026-04-22
**Depth:** standard
**Files Reviewed:** 16
**Status:** issues_found

## Summary

Phase 7 (DevEx hardening) is in very good shape overall. The CI matrix,
coverage gate consistency (60 in `pyproject.toml` + `ci.yml` + `Makefile`),
pre-commit scope (`^(api|nexo)/`), and the "no `--profile mcp`" smoke
invariant are all correctly wired. The quartet of new docs (ARCHITECTURE,
RUNBOOK, RELEASE, CHANGELOG) is accurate and self-consistent, and CLAUDE.md
has been bumped with the new sections expected by `tests/infra/test_claude_md.py`.

No critical or security-blocking issues were found. The findings below are
correctness / maintainability concerns, concentrated in two areas:

1. **Test suite weaker than the invariant it claims to protect** — three
   regression tests gate `--cov-fail-under` at `>= 50` while the actual
   baseline (and the value documented in CLAUDE.md "Qué NO hacer") is 60.
   A silent 60 → 50 regression would pass those tests.
2. **Ruff per-file-ignores glob pattern does not recurse** — `"tests/*"`
   matches direct children only (ruff globs don't cross `/` without `**`),
   so nested test files under `tests/infra/`, `tests/auth/`, `tests/data/`
   etc. do NOT actually inherit the `B` / `SIM` relaxations the comment
   advertises. Latent: only bites when a new nested test triggers one of
   those rules.

Other findings are smaller: a race-safety concern in the smoke job's
`.env` seeding (trailing-newline dependency), over-broad `except ImportError:`
coverage exclusion, a few tests with weak string-matching assertions, an
unused `types-requests` mypy dep, and a process-substitution reliance in
`RELEASE.md` step 3 that breaks under POSIX `sh`.

Style-only backfill on `api/` and `nexo/` was explicitly declared out of
scope and not inspected.

---

## Warnings

### WR-01: Ruff `per-file-ignores` pattern `tests/*` does not match nested test files

**File:** `pyproject.toml:37`
**Issue:** Ruff glob patterns do not cross `/` unless `**` is used. The
rule `"tests/*" = ["B", "SIM"]` matches `tests/test_foo.py` but NOT
`tests/infra/test_foo.py`, `tests/auth/test_bar.py`, `tests/data/...`,
etc. All the new Phase-7 tests live in `tests/infra/` — none of them are
actually covered by the documented relaxation. This is latent: current
tests don't trigger `B`/`SIM`, so nothing fails today, but the moment a
new nested test uses e.g. `assert x == []` (B015 territory) or the classic
`try/except/pass` the lint job will block it while the author reasonably
believes tests are relaxed.
**Fix:**
```toml
[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["B", "SIM"]        # recursive
"nexo/data/models_*.py" = ["F401"]
"api/routers/historial.py" = ["F841"]
"api/services/metrics.py" = ["F841"]
```

### WR-02: Coverage-gate regression tests accept `>= 50`, allowing a silent drop from 60

**File:** `tests/infra/test_ci_matrix.py:88-90`,
`tests/infra/test_makefile_devex.py:138-140`,
`tests/infra/test_pyproject_config.py:49-51`
**Issue:** Three tests assert `value >= 50` with a comment `"baseline Phase 7 = 60"`.
CLAUDE.md "Qué NO hacer" explicitly says _"No bajar la cobertura por debajo
del gate configurado… CI bloquea PRs que regresionen cobertura."_ The tests,
however, would happily green-light a PR that lowers the gate to 50 in any
of the three places — exactly the regression they claim to prevent. They
also don't enforce the **cross-file consistency invariant** called out in
the Makefile header comment (`pyproject.toml` == `ci.yml` == `Makefile`).
**Fix:** Tighten to exact equality and add a cross-file consistency check:
```python
# In each test:
assert value == 60, f"--cov-fail-under={value}; Phase 7 baseline locked at 60."

# New test (either file, or a shared one):
def test_coverage_gate_consistent_across_files() -> None:
    pp = _load_pyproject()["tool"]["coverage"]["report"]["fail_under"]
    ci = re.search(r"--cov-fail-under=(\d+)", CI_PATH.read_text()).group(1)
    mk = re.search(r"--cov-fail-under=(\d+)", MAKEFILE.read_text()).group(1)
    assert pp == int(ci) == int(mk), (
        f"Coverage gate divergence: pyproject={pp}, ci={ci}, makefile={mk}"
    )
```

### WR-03: `make test-docker` will fail — pytest is not installed in the runtime `web` image

**File:** `Makefile:177-178`
**Issue:** `test-docker` runs `docker compose exec -T web pytest -q …`.
`requirements-dev.txt` (which contains `pytest`, `pytest-cov`, `httpx`,
`pytest-asyncio`) is, by design, not installed in the runtime image
(see comment at the top of `requirements-dev.txt`: _"No se instalan en el
contenedor runtime, solo en el entorno local del dev y en el CI"_). The
target advertised in `help` will therefore always fail with
`pytest: command not found` or `exec: "pytest": executable file not found`.
Either drop the target, or install dev deps in the container, or pivot to
`python -m pytest` with a `pip install -r requirements-dev.txt` first.
**Fix:** Two options:
```makefile
# Option A: drop test-docker entirely (simplest; `make test` on host is enough).

# Option B: install dev deps on the fly inside the container.
test-docker: ## Corre pytest dentro del container web (instala dev deps on-the-fly)
	docker compose exec -T web sh -c 'pip install -q -r requirements-dev.txt && \
	  pytest -q --cov=api --cov=nexo --cov-fail-under=60'
```
Also consider adding a `.PHONY` guard test that exercises `test-docker` in
CI with a non-default image if you want it to stay live.

### WR-04: Smoke job's `.env` seeding depends on `.env.example` ending in a newline

**File:** `.github/workflows/ci.yml:121-123`
**Issue:**
```bash
cp .env.example .env
echo "NEXO_SECRET_KEY=ci-smoke-$(openssl rand -hex 16)" >> .env
```
If `.env.example` lacks a trailing newline, the appended `NEXO_SECRET_KEY=…`
line gets concatenated with the last line of the file (e.g.
`NEXO_DEBUG=falseNEXO_SECRET_KEY=ci-smoke-...`), silently breaking whatever
var happens to be last in the example file. Docker-compose env-file parsing
then either ignores the malformed last entry or assigns it to the wrong key.
This is easy to miss because it depends on a file the smoke job doesn't
lint. Defensive fix is one line and cost-free:
**Fix:**
```yaml
- name: Seed .env from example
  run: |
    cp .env.example .env
    # Guarantee newline terminator before appending (handles missing EOL).
    [ -n "$(tail -c1 .env)" ] && echo >> .env
    echo "NEXO_SECRET_KEY=ci-smoke-$(openssl rand -hex 16)" >> .env
```

---

## Info

### IN-01: `exclude_lines` drops all `except ImportError:` branches from coverage silently

**File:** `pyproject.toml:101-107`
**Issue:** `"except ImportError:"` in `exclude_lines` excludes every
line that textually matches that pattern from coverage measurement. The
typical intent is "skip optional-import fallbacks", but this is a blunt
instrument: a handler that does real work (logging, raising a wrapped
error, falling back to a different impl) becomes invisible to coverage.
The Phase-7 baseline of 60% was measured with this exclusion in place, so
removing it now would re-measure lower; but going forward, prefer explicit
`# pragma: no cover` on the specific lines you want to exclude rather
than a global pattern.
**Fix:** Either keep the current exclusion and document it in
`07-COVERAGE-BASELINE.md` as "intentional, re-evaluate in Mark-IV", or
narrow it:
```toml
exclude_lines = [
    "pragma: no cover",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    # Drop the blanket "except ImportError:" — use pragma: no cover on
    # the specific optional-import site instead.
]
```

### IN-02: Mypy `additional_dependencies` includes `types-requests` but `requests` is not a runtime dep

**File:** `.pre-commit-config.yaml:50`
**Issue:** The pre-commit mypy hook pulls in `types-requests`, but the
project uses `httpx` (see `requirements-dev.txt:27` and stack table in
`docs/ARCHITECTURE.md`). Dead dependency; slightly slows hook install and
will produce a warning from mypy (`warn_unused_ignores`-style noise) if a
stub starts conflicting.
**Fix:**
```yaml
additional_dependencies:
  - "fastapi==0.135.3"
  - "pydantic-settings==2.13.1"
  - "sqlalchemy==2.0.49"
  # drop types-requests — project uses httpx, not requests
```

### IN-03: `_job_steps_as_text` flattens all `run:` blocks, weakening step-boundary assertions

**File:** `tests/infra/test_ci_matrix.py:34-41`, `tests/infra/test_ci_matrix.py:113-116`
**Issue:** `_job_steps_as_text(job)` concatenates every `run:` body with
`\n` separators. `test_smoke_job_exists` then greps
`r"docker compose up -d.*--build.*db.*web"` with `re.DOTALL`, which means
this would also match if `docker compose up -d` appeared in step N and
`--build db web` appeared in step N+2. Unlikely in practice, but the test
claims to enforce a specific command in a specific step. Consider asserting
against a single step's `run` block instead of the concatenated blob.
**Fix:**
```python
def _find_step(job: dict, keyword: str) -> str:
    for step in job.get("steps", []) or []:
        run = step.get("run") or ""
        if keyword in run:
            return run
    return ""

def test_smoke_job_exists() -> None:
    data = _load()
    smoke = data["jobs"]["smoke"]
    start_step = _find_step(smoke, "docker compose up")
    assert re.search(r"docker compose up -d\s+--build\s+db\s+web", start_step)
    # still on the blob: negative assertion is safer
    assert "--profile mcp" not in _job_steps_as_text(smoke)
```

### IN-04: No test asserts the "no `--profile mcp`" invariant for the smoke job

**File:** `tests/infra/test_ci_matrix.py:103-121`
**Issue:** The review brief explicitly calls out the `--profile mcp`
exclusion invariant. `test_smoke_job_exists` enforces it for Makefile
targets (`test_up_target_does_not_start_mcp`, `test_dev_target_does_not_start_mcp`)
but there is no negative assertion on the CI smoke job. A future
contributor adding `--profile mcp` to smoke would pass CI.
**Fix:**
```python
def test_smoke_job_does_not_start_mcp() -> None:
    data = _load()
    blob = _job_steps_as_text(data["jobs"]["smoke"])
    assert "--profile mcp" not in blob, (
        "Smoke job must NOT include --profile mcp (invariant Phase 1)."
    )
    # Also assert that the compose up command does not list `mcp` as a service.
    up_match = re.search(r"docker compose up -d\s+--build\s+([^\n]+)", blob)
    assert up_match is not None
    assert "mcp" not in up_match.group(1).split(), (
        f"Smoke job must not enumerate `mcp` service; got: {up_match.group(1)}"
    )
```

### IN-05: `RELEASE.md` step 3 relies on bash process substitution — will break under `sh`

**File:** `docs/RELEASE.md:82-86`
**Issue:**
```bash
gh release create v1.0.0 \
  --title "Nexo v1.0.0 - Mark-III" \
  --notes-file <(sed -n '/## \[1\.0\.0\]/,/## \[/p' CHANGELOG.md | head -n -1)
```
The `<(...)` construct is a bash/zsh feature; it fails under POSIX `sh`
or `dash` (common on Debian/Ubuntu `/bin/sh`). If the release runner is a
macOS or Linux shell that isn't bash, this errors with
`syntax error near unexpected token`. Also, `head -n -1` is GNU-coreutils
only; BSD/macOS `head` doesn't accept negative N. Both gotchas hit the
same line.
**Fix:** Use a temp file, keeps the command portable:
```bash
sed -n '/## \[1\.0\.0\]/,/## \[/p' CHANGELOG.md | sed '$d' > /tmp/notes.md
gh release create v1.0.0 \
  --title "Nexo v1.0.0 - Mark-III" \
  --notes-file /tmp/notes.md
rm /tmp/notes.md
```
Alternatively, document that the command must run under `bash` on GNU
coreutils (add `#!/usr/bin/env bash` to a `scripts/release.sh` wrapper).

### IN-06: `test_architecture_md_exists_and_has_mermaid` is a pure substring check

**File:** `tests/infra/test_devex_docs.py:33-39`
**Issue:** `assert "mermaid" in content` matches anywhere in the doc —
including in a prose sentence like "the diagram uses mermaid" without an
actual fenced block. A stronger assertion verifies the fenced block:
**Fix:**
```python
def test_architecture_md_exists_and_has_mermaid():
    assert ARCH.exists()
    content = ARCH.read_text(encoding="utf-8")
    # Require a fenced mermaid block, not just the word.
    assert re.search(r"^```mermaid\b", content, re.MULTILINE), (
        "ARCHITECTURE.md must contain a fenced ```mermaid block."
    )
```

### IN-07: `caddy/` has no YAML — the `check-yaml` exclude is dead config

**File:** `.pre-commit-config.yaml:22-23`
**Issue:** `caddy/` contains `Caddyfile` and `Caddyfile.prod` only (no
`.yaml`/`.yml`). The `exclude: ^caddy/` on `check-yaml` is inert. Harmless,
but misleading to readers who assume Caddyfiles are being explicitly
skipped because they _are_ YAML (they aren't). Consider dropping the
exclude or commenting why it's defensive.
**Fix:** Drop the line, or annotate:
```yaml
- id: check-yaml
  # Defensive exclude: if someone adds docker-compose-style YAML under
  # caddy/ it should be hand-validated (Caddyfile is not YAML).
  exclude: ^caddy/
```

### IN-08: Per-file F841 blanket-ignore hides future regressions in two modules

**File:** `pyproject.toml:41-42`
**Issue:** `"api/routers/historial.py" = ["F841"]` and
`"api/services/metrics.py" = ["F841"]` blanket-ignore "local variable
assigned but never used" for the entire file. The inline comment says
"2 F841 residuales" and defers to "el owner del modulo". The risk: any
future F841 in those files (including genuine dead-assignment bugs) is
invisible to ruff. Prefer `# noqa: F841` on the specific lines or a
narrower `lint.extend-per-file-ignores` scoped to specific line ranges
(not supported by ruff — so the `# noqa` approach is the right one).
**Fix:** Add `# noqa: F841` to the specific offending lines and remove
the per-file entries:
```python
# In historial.py / metrics.py, at the two residual sites:
x = some_call()  # noqa: F841  # TODO(owner): revisar bloque WIP (Phase 7 07-01)
```
Then drop both entries from `[tool.ruff.lint.per-file-ignores]`.

### IN-09: `test_runbook_md_lockout_scenario_has_delete_remedy` regex has a redundant branch

**File:** `tests/infra/test_devex_docs.py:126`
**Issue:** `r"(>= ?2|>=2|dos|al menos 2) propietarios"` — `>= ?2` already
matches both `>=2` and `>= 2` (the `?` makes the space optional), so the
standalone `>=2` alternative is redundant. Purely cosmetic; no behavior
change. The current assertion will still pass against `RUNBOOK.md` as
written (`"SIEMPRE crear >=2 propietarios"`).
**Fix:**
```python
assert re.search(r"(>=\s?2|dos|al menos 2) propietarios", content)
```

---

_Reviewed: 2026-04-22_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
