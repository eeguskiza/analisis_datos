---
phase: 08-redise-o-ui-modo-claro-moderno
plan: 10
type: execute
wave: 7
depends_on: [08-06, 08-07, 08-08, 08-09]
files_modified:
  - .github/workflows/ci.yml
  - .pa11yci.json
  - scripts/seed_pa11y_user.sql
  - scripts/pa11y_login.js
  - tests/infra/test_pa11y_config.py
autonomous: true
gap_closure: false
requirements: [UIREDO-08]
tags: [ci, pa11y, accessibility, wcag-aa, phase-8]
user_setup: []

must_haves:
  truths:
    - "`.pa11yci.json` scans 8-10 URLs at WCAG AA level: /login (public), /bienvenida, /, /pipeline, /historial, /bbdd, /capacidad, /ajustes, /ajustes/usuarios, /cambiar-password."
    - "A seed SQL file creates a fixture user `pa11y@nexo.local` with an argon2id hash (fixed, checked into source — this is CI-only, no production exposure)."
    - "A JS `actions` file performs login via form submission in the pa11y-ci context so every URL post-login is accessible."
    - "The existing `smoke` job in `.github/workflows/ci.yml` is extended with a pa11y-ci step that runs AFTER the healthcheck (compose already up) and BEFORE tear-down."
    - "The step installs pa11y-ci globally (`npm install -g pa11y-ci@3.1.0`) and runs `pa11y-ci` against the compose-served URL."
    - "The step fails the CI job on any WCAG 2.1 AA contrast or interaction error (not just warnings)."
    - "A tests/infra regression test asserts the `.pa11yci.json` config is valid JSON, lists all the required URLs, sets WCAG2AA standard, and references the login actions."
  artifacts:
    - path: ".pa11yci.json"
      provides: "pa11y-ci config — URLs, standard, actions"
      contains: "WCAG2AA"
    - path: "scripts/seed_pa11y_user.sql"
      provides: "CI-only fixture user seed"
      contains: "pa11y@nexo.local"
    - path: "scripts/pa11y_login.js"
      provides: "JS actions to log in before scanning authenticated URLs"
      contains: "navigate to"
    - path: ".github/workflows/ci.yml"
      provides: "smoke job extended with pa11y-ci step"
      contains: "pa11y-ci"
    - path: "tests/infra/test_pa11y_config.py"
      provides: "Regression — config file exists, lists required URLs, targets WCAG2AA"
      contains: "WCAG2AA"
  key_links:
    - from: ".github/workflows/ci.yml"
      to: ".pa11yci.json"
      via: "pa11y-ci --config .pa11yci.json"
      pattern: "pa11y-ci"
    - from: ".pa11yci.json"
      to: "scripts/pa11y_login.js"
      via: "actions field on authenticated URLs"
      pattern: "pa11y_login"
---

<objective>
Extend the Phase 7 `smoke` job with `pa11y-ci` to enforce WCAG 2.1 AA
on every redesigned screen. Provide a CI-only seed user + login
actions so authenticated URLs are actually reached.

Purpose: UIREDO-08 / D-21. Without this gate, future contributors can
regress accessibility silently.

Output: pa11y-ci config + login actions + CI job extension + seed SQL
+ regression test for the config.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/08-redise-o-ui-modo-claro-moderno/08-CONTEXT.md
@.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md
@.github/workflows/ci.yml
@nexo/services/auth.py
@docs/AUTH_MODEL.md
@CLAUDE.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Seed SQL + Login JS actions</name>
  <read_first>
    - `nexo/services/auth.py` — argon2id hash format.
    - `docs/AUTH_MODEL.md` — password policy (min 12 chars).
    - `nexo/data/sql/nexo/` — existing seed file patterns.
  </read_first>
  <action>
### Part A — `scripts/seed_pa11y_user.sql`

Create a CI-only seed for a fixture user. Fixed argon2id hash of the
password `Pa11yNexoCiTest2026!` computed once via `argon2` Python lib.

```sql
-- scripts/seed_pa11y_user.sql — CI-only fixture for pa11y-ci auth flow.
-- Password: Pa11yNexoCiTest2026!  (hash below is argon2id RFC_9106_LOW_MEMORY).
-- This user does NOT exist in production. Seed only runs in CI during the
-- smoke job.
--
-- The hash is generated with:
--   from nexo.services.auth import hash_password
--   print(hash_password("Pa11yNexoCiTest2026!"))
-- Replace the placeholder below with an actual hash before committing.

INSERT INTO nexo.users (email, password_hash, role, active, must_change_password, nombre)
VALUES (
  'pa11y@nexo.local',
  '$argon2id$v=19$m=65536,t=3,p=4$REPLACE_WITH_REAL_SALT$REPLACE_WITH_REAL_HASH',
  'propietario',  -- propietario bypasses per-dept gating → can visit every URL
  TRUE,
  FALSE,
  'Pa11y Ci'
)
ON CONFLICT (email) DO NOTHING;
```

**Hash generation note** — the plan executor must run this once in a
Python shell before committing the file:

```bash
docker compose exec -T web python -c \
  "from nexo.services.auth import hash_password; print(hash_password('Pa11yNexoCiTest2026!'))"
```

…and paste the resulting PHC string into the SQL. If the executor
cannot run the command, leave the placeholder and document in the
SUMMARY that a follow-up commit must replace the hash before the CI
job can execute successfully. This is a NON-BLOCKING footgun — the
plan is still valid; only the CI step won't pass until the hash is
real.

### Part B — `scripts/pa11y_login.js`

```js
// scripts/pa11y_login.js — pa11y-ci actions that log in before scanning
// authenticated URLs. Used as the `actions` value in .pa11yci.json.

module.exports = [
  "navigate to http://localhost:8001/login",
  "wait for element #login-email to be visible",
  "set field #login-email to pa11y@nexo.local",
  "set field #login-password to Pa11yNexoCiTest2026!",
  "click element form[action='/login'] button[type='submit']",
  "wait for path to not be /login",
];
```
  </action>
  <acceptance_criteria>
    - `test -f scripts/seed_pa11y_user.sql` returns 0.
    - `test -f scripts/pa11y_login.js` returns 0.
    - `grep -c "pa11y@nexo.local" scripts/seed_pa11y_user.sql` returns 1 or more.
    - `grep -c "pa11y@nexo.local" scripts/pa11y_login.js` returns 1 or more.
    - `node -e "const a=require('./scripts/pa11y_login.js'); if(!Array.isArray(a)||a.length<4)process.exit(1);"` exit 0.
  </acceptance_criteria>
  <verify>
    <automated>test -f scripts/seed_pa11y_user.sql &amp;&amp; test -f scripts/pa11y_login.js &amp;&amp; node -e "const a=require('./scripts/pa11y_login.js'); if(!Array.isArray(a)||a.length&lt;4)process.exit(1);"</automated>
  </verify>
  <done>Seed + login actions shipped.</done>
</task>

<task type="auto">
  <name>Task 2: .pa11yci.json — config with 10 URLs, WCAG2AA, login actions on protected routes</name>
  <read_first>
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-UI-SPEC.md` §"Accessibility" for contrast pairs + standard.
    - `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-RESEARCH.md` §pa11y-ci configuration notes.
    - [pa11y-ci README](https://github.com/pa11y/pa11y-ci) — config schema.
  </read_first>
  <action>
Create `.pa11yci.json`:

```json
{
  "defaults": {
    "standard": "WCAG2AA",
    "timeout": 30000,
    "wait": 500,
    "chromeLaunchConfig": {
      "args": ["--no-sandbox", "--disable-setuid-sandbox"]
    },
    "ignore": [
      "warning"
    ]
  },
  "urls": [
    "http://localhost:8001/login",
    {
      "url": "http://localhost:8001/bienvenida",
      "actions": "./scripts/pa11y_login.js"
    },
    {
      "url": "http://localhost:8001/",
      "actions": "./scripts/pa11y_login.js"
    },
    {
      "url": "http://localhost:8001/pipeline",
      "actions": "./scripts/pa11y_login.js"
    },
    {
      "url": "http://localhost:8001/historial",
      "actions": "./scripts/pa11y_login.js"
    },
    {
      "url": "http://localhost:8001/bbdd",
      "actions": "./scripts/pa11y_login.js"
    },
    {
      "url": "http://localhost:8001/capacidad",
      "actions": "./scripts/pa11y_login.js"
    },
    {
      "url": "http://localhost:8001/ajustes",
      "actions": "./scripts/pa11y_login.js"
    },
    {
      "url": "http://localhost:8001/ajustes/usuarios",
      "actions": "./scripts/pa11y_login.js"
    },
    {
      "url": "http://localhost:8001/cambiar-password",
      "actions": "./scripts/pa11y_login.js"
    }
  ]
}
```

Note on `actions` — pa11y-ci loads it as a JS module per URL when the
value is a path ending in `.js`. The login runs per URL (re-login each
time) — slower but deterministic. An optimization (shared session)
is Mark-IV work.
  </action>
  <acceptance_criteria>
    - `test -f .pa11yci.json` returns 0.
    - `python -c "import json; cfg=json.load(open('.pa11yci.json')); assert cfg['defaults']['standard']=='WCAG2AA'; assert len(cfg['urls'])>=10"` exit 0.
    - `grep -c "/bienvenida" .pa11yci.json` returns 1.
    - `grep -c "/ajustes/usuarios" .pa11yci.json` returns 1.
  </acceptance_criteria>
  <verify>
    <automated>test -f .pa11yci.json &amp;&amp; python -c "import json; cfg=json.load(open('.pa11yci.json')); assert cfg['defaults']['standard']=='WCAG2AA'; assert len(cfg['urls'])&gt;=10"</automated>
  </verify>
  <done>Config ships with 10 URLs at WCAG2AA.</done>
</task>

<task type="auto">
  <name>Task 3: Extend .github/workflows/ci.yml smoke job + regression test for config</name>
  <read_first>
    - `.github/workflows/ci.yml` current `smoke` job (lines 114-150).
  </read_first>
  <action>
### Part A — Extend `.github/workflows/ci.yml`

Modify the existing `smoke` job. After the "Wait for web healthy" step
and before "Tear down", add:

```yaml
      - name: Seed pa11y-ci fixture user
        run: |
          docker compose exec -T db psql -U oee -d oee_planta \
            < scripts/seed_pa11y_user.sql

      - name: Install pa11y-ci
        run: npm install -g pa11y-ci@3.1.0

      - name: Run pa11y-ci (WCAG2AA)
        run: pa11y-ci --config .pa11yci.json
```

Keep the "Tear down" step as the final step (with `if: always()`).

### Part B — `tests/infra/test_pa11y_config.py`

```python
"""Regression for Phase 8 / Plan 08-10: pa11y-ci config.

Locks:
1. .pa11yci.json is valid JSON.
2. Uses WCAG2AA standard.
3. Scans >= 10 URLs including /login, /bienvenida, /, /ajustes.
4. Authenticated URLs reference scripts/pa11y_login.js as actions.
5. CI workflow has the pa11y step after healthcheck.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_CFG = _ROOT / ".pa11yci.json"
_CI = _ROOT / ".github" / "workflows" / "ci.yml"
_LOGIN_JS = _ROOT / "scripts" / "pa11y_login.js"
_SEED_SQL = _ROOT / "scripts" / "seed_pa11y_user.sql"


def test_pa11yci_json_is_valid_json():
    data = json.loads(_CFG.read_text(encoding="utf-8"))
    assert "defaults" in data
    assert "urls" in data


def test_standard_is_wcag2aa():
    data = json.loads(_CFG.read_text(encoding="utf-8"))
    assert data["defaults"]["standard"] == "WCAG2AA"


def test_scans_required_urls():
    data = json.loads(_CFG.read_text(encoding="utf-8"))

    def _url_of(entry):
        return entry["url"] if isinstance(entry, dict) else entry

    urls = {_url_of(e) for e in data["urls"]}
    required_fragments = [
        "/login", "/bienvenida", "/pipeline",
        "/historial", "/bbdd", "/capacidad", "/ajustes", "/cambiar-password",
    ]
    for frag in required_fragments:
        assert any(frag in u for u in urls), f"pa11y config missing scan for {frag}"

    assert len(urls) >= 10


def test_authenticated_urls_reference_login_actions():
    data = json.loads(_CFG.read_text(encoding="utf-8"))
    for entry in data["urls"]:
        if isinstance(entry, dict) and "/login" not in entry["url"]:
            assert "actions" in entry
            assert "pa11y_login" in entry["actions"]


def test_login_actions_file_exists():
    assert _LOGIN_JS.exists()


def test_seed_sql_file_exists():
    assert _SEED_SQL.exists()
    assert "pa11y@nexo.local" in _SEED_SQL.read_text(encoding="utf-8")


def test_ci_workflow_has_pa11y_step():
    src = _CI.read_text(encoding="utf-8")
    assert "pa11y-ci" in src
    assert "seed_pa11y_user.sql" in src
```
  </action>
  <acceptance_criteria>
    - `grep -c "pa11y-ci" .github/workflows/ci.yml` returns 2 or more.
    - `grep -c "seed_pa11y_user.sql" .github/workflows/ci.yml` returns 1 or more.
    - `pytest tests/infra/test_pa11y_config.py -x -q` exit 0.
    - `pytest tests/ -x -q` exit 0.
    - `ruff check tests/infra/test_pa11y_config.py` exit 0.
    - `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"` exit 0.
  </acceptance_criteria>
  <verify>
    <automated>python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" &amp;&amp; ruff check tests/infra/test_pa11y_config.py &amp;&amp; ruff format --check tests/infra/test_pa11y_config.py &amp;&amp; pytest tests/infra/test_pa11y_config.py -x -q &amp;&amp; pytest tests/ -x -q</automated>
  </verify>
  <done>CI wired; regression test locks invariants.</done>
</task>

</tasks>

<threat_model>
## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-08-10-01 | Information Disclosure | Seed user hash + password checked into repo | accept | CI-only fixture; hash is a public argon2id string. Production auth is independent. Password string `Pa11yNexoCiTest2026!` is documented in this plan, not elsewhere. |
| T-08-10-02 | Tampering | CI workflow mutation bypasses pa11y step | mitigate | Regression test `test_ci_workflow_has_pa11y_step`. |
| T-08-10-03 | DoS | pa11y scan times out, CI hangs | mitigate | `timeout: 30000` per URL in defaults. Job-level timeout is GitHub's 6h default. |
</threat_model>

<verification>
1. Local: `docker compose up -d web db` + `docker compose exec -T db psql -U oee -d oee_planta < scripts/seed_pa11y_user.sql` + `npm install -g pa11y-ci` + `pa11y-ci --config .pa11yci.json`. Expected: 0 errors.
2. CI: push the branch; observe `smoke` job running the new step and passing (or failing with actionable error output).
3. If pa11y reports a violation, fix the template; do NOT silence the rule.
</verification>

<success_criteria>
- `.pa11yci.json` scans 10 URLs at WCAG2AA.
- Seed user + login actions exist.
- CI smoke job runs pa11y-ci.
- Config regression test passes.
</success_criteria>

<output>
Create `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-10-SUMMARY.md` noting: hash committed or placeholder, initial pa11y run results, any remediation follow-up.
</output>
