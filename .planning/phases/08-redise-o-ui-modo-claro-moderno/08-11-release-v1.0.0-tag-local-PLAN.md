---
phase: 08-redise-o-ui-modo-claro-moderno
plan: 11
type: execute
wave: 8
depends_on: [08-10]
files_modified:
  - CHANGELOG.md
  - .planning/STATE.md
autonomous: true
gap_closure: false
requirements: [UIREDO-01, UIREDO-02, UIREDO-03, UIREDO-04, UIREDO-05, UIREDO-06, UIREDO-07, UIREDO-08]
tags: [release, tag, v1.0.0, changelog, phase-8, mark-iii-close]
user_setup: []

must_haves:
  truths:
    - "`CHANGELOG.md` moves `[Unreleased]` → `[1.0.0] - 2026-04-22` with a new `[Unreleased]` placeholder block at the top."
    - "The `[1.0.0]` section lists the Phase 8 deliverables: tokens + tailwind config, chrome rewrite (top bar + drawer + toast + print.css), /bienvenida landing, users.nombre migration, centro-mando/luk4 re-skin, auth screens, data screens, config screens, ajustes suite, pa11y-ci CI extension."
    - "A local annotated git tag `v1.0.0` is created from HEAD with a message summarising Mark-III closure."
    - "No `git push` is executed — the operator pushes manually on return."
    - "No deploy is executed — `docs/DEPLOY_LAN.md` is the next step for the operator."
    - "STATE.md is updated with a session-continuity note pointing the operator at the next manual steps (`git push --tags origin feature/Mark-III`, then `docs/DEPLOY_LAN.md`)."
    - "The tag is listed by `git tag --list v1.0.0` and has an annotated commit message (not a lightweight tag)."
  artifacts:
    - path: "CHANGELOG.md"
      provides: "Versioned changelog with 1.0.0 entry and new Unreleased block"
      contains: "## [1.0.0] - 2026-04-22"
    - path: ".planning/STATE.md"
      provides: "Session-continuity note for the operator"
      contains: "v1.0.0 tag created locally — push manually"
  key_links:
    - from: "CHANGELOG.md"
      to: "docs/RELEASE.md"
      via: "release checklist — CHANGELOG bump is step 1"
      pattern: "1.0.0"
---

<objective>
Close Mark-III locally. Cut CHANGELOG.md from `[Unreleased]` to
`[1.0.0]`, create an annotated tag `v1.0.0`, and leave the remaining
publish / deploy steps for the operator to run manually when they
return. Do NOT run `git push`, do NOT run `docs/DEPLOY_LAN.md` deploy
— those are explicit human-gated steps per locked decision #4.

Output: updated CHANGELOG.md + local tag + STATE.md note.
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
@CHANGELOG.md
@docs/RELEASE.md
@docs/DEPLOY_LAN.md
@CLAUDE.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Bump CHANGELOG.md to [1.0.0] - 2026-04-22 + re-open [Unreleased]</name>
  <read_first>
    - `CHANGELOG.md` in full — understand the Keep a Changelog structure currently in place.
    - `docs/RELEASE.md` — release checklist per Phase 7.
    - The Phase 8 plan summaries (`08-01-SUMMARY.md` through `08-10-SUMMARY.md`) if present — source of the Phase 8 bullet list.
  </read_first>
  <action>
Rewrite the top of `CHANGELOG.md` to move the current `[Unreleased]`
content into `[1.0.0] - 2026-04-22` and create a new empty
`[Unreleased]` block above it. Append Phase 8 bullet points to the
1.0.0 section.

Structure (new top of file):

```markdown
# Changelog

All notable changes to Nexo documented here.

El formato sigue [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
y el proyecto adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

## [1.0.0] - 2026-04-22

**Cierre de Mark-III.** Primera release productiva de Nexo como plataforma
interna de ECS Mobility. Sucesora del monolito "OEE Planta" manteniendo el
mismo stack; cambio de nombre y refactor estructural, no reescritura.

### Added

- **Phase 1 (Sprint 0) — Naming + Higiene + CI:** [...preservar el texto
  existente del placeholder 1.0.0...]
- **Phase 2 (Sprint 1) — Identidad:** [...]
- **Phase 3 (Sprint 2) — Capa de datos:** [...]
- **Phase 4 (Sprint 3) — Consultas pesadas:** [...]
- **Phase 5 (Sprint 4) — UI por roles:** [...]
- **Phase 6 (Sprint 5) — Despliegue LAN HTTPS:** [...]
- **Phase 7 (Sprint 6) — DevEx hardening:** [...]
- **Phase 8 (Sprint 7) — Rediseño UI modo claro moderno:**
  - Sistema de tokens de diseño en `static/css/tokens.css` (dos capas:
    raw + semántica) y config Tailwind extraída a
    `static/js/tailwind.config.js`. Consumido via
    `rgb(var(--color-xxx) / <alpha-value>)`.
  - Chrome global rewrite: top bar 56px + drawer oculto tras hamburger
    (persiste estado en `localStorage`), toast top-right con API 3-arg
    `showToast(type, title, msg)`, animaciones ≤200ms con
    `prefers-reduced-motion` honrado.
  - Nuevo screen `/bienvenida` tras login: saludo por franja horaria
    (Buenos días / tardes / noches), día de la semana + fecha en
    castellano, reloj HH:MM:SS en tiempo real, CTA primario `Ir a
    Centro de Mando`.
  - Migración `nexo.users.nombre VARCHAR(120) NULL` + backfill desde
    email local-part. Form `/ajustes/usuarios` acepta `nombre`
    opcional.
  - Plugins Alpine Focus + Persist cargados antes del core para
    focus-trap + `$persist` en el drawer.
  - Print stylesheet `static/css/print.css` para `Ctrl+P` limpio.
  - Extracción de primer frame `static/img/gif-corona.png` para
    fallback `prefers-reduced-motion` del loading hero.
  - Fix de firma duplicada `showToast` (Pitfall 3): una sola definición
    3-arg en `static/js/app.js`; 12+ callers migrados.
  - Slugs GFM alineados con `docs/RUNBOOK.md` en los 5 enlaces de
    error-state.
  - Centro de Mando (`/`, `luk4.html`) re-skinned manteniendo
    interacción plano + máquinas LOCKED (D-16).
  - Auth screens (`login`, `cambiar_password`, `mis_solicitudes`)
    refactoradas con tokens + autocomplete correcto + status pills
    semánticos.
  - Data screens (`pipeline`, `historial`, `bbdd`, `datos`,
    `informes`) refactoradas preservando preflight modal Alpine +
    RBAC (`informes:delete`).
  - Config screens (`capacidad`, `operarios`, `recursos`, `ciclos`,
    `ciclos_calc`, `plantillas`) refactoradas preservando RBAC
    (`recursos:edit`, `ciclos:edit`, `operarios:export`).
  - Ajustes suite: hub con 6-card grid + 6 sub-páginas
    (conexión/usuarios/auditoría/límites/rendimiento/solicitudes) con
    breadcrumbs `Ajustes / <Sub-page>`. Chart.js series colors
    tokenizados (`--color-primary`, `--color-info`, `--color-warn`,
    `--color-success`).
  - Nueva sección "Configuración" en el drawer con 7 items flat
    (Ajustes, Conexión, Usuarios, Auditoría, Límites, Rendimiento,
    Solicitudes), sección entera gateada por `can()` defensivo.
  - Shortcut `[` (corchete izquierdo) togglea el drawer; guarda
    contra intercepción al escribir en inputs/textareas/contentEditable.
  - CI ampliado con `pa11y-ci` 10-URL WCAG 2.1 AA dentro del job
    `smoke` existente (D-21), con seed fixture `pa11y@nexo.local` y
    login actions JS.
  - Skills de `sketch-findings-*/SKILL.md` documentan la variante
    elegida por Claude en modo autónomo (D-12 override documentado)
    para cada pantalla — reversible por `git revert`.

### Changed

- Post-login redirect pasa de `/` a `/bienvenida` (excepto
  `must_change_password` → `/cambiar-password`).
- Label `Analisis` → `Análisis` (acento correcto) en el drawer.
- `.btn-primary`, `.card`, `.data-table`, `.input-inline`,
  `.spinner`, `.log-console` mantienen nombres de clase pero sus
  estilos se re-declaran sobre tokens semánticos.

### Fixed

- Firma conflictiva `showToast` (2-arg en `app.js` vs 3-arg en
  `base.html`) — 3-arg ahora canónico; el flash middleware y el
  modal de preflight mantienen su contrato.
- Slugs GFM de los enlaces a `docs/RUNBOOK.md` ahora coinciden con
  los headings reales (antes usaban una versión abreviada que
  devolvía 404 al anchor).
```

If the existing `[1.0.0] - 2026-MM-DD` placeholder in the current file
already contains the Phases 1-7 bullet blocks, reuse them verbatim —
do not regenerate. Only the Phase 8 Added/Changed/Fixed entries are
new content.
  </action>
  <acceptance_criteria>
    - `grep -c "## \[1.0.0\] - 2026-04-22" CHANGELOG.md` returns 1.
    - `grep -c "^## \[Unreleased\]" CHANGELOG.md` returns 1 (new empty block at top).
    - `grep -c "## \[1.0.0\] - 2026-MM-DD" CHANGELOG.md` returns 0 (old placeholder gone).
    - `grep -c "Phase 8 (Sprint 7)" CHANGELOG.md` returns 1.
    - `grep -c "pa11y-ci" CHANGELOG.md` returns 1 or more.
    - `grep -c "bienvenida" CHANGELOG.md` returns 1 or more.
  </acceptance_criteria>
  <verify>
    <automated>grep -q "## \[1.0.0\] - 2026-04-22" CHANGELOG.md &amp;&amp; grep -q "^## \[Unreleased\]" CHANGELOG.md &amp;&amp; ! grep -q "## \[1.0.0\] - 2026-MM-DD" CHANGELOG.md</automated>
  </verify>
  <done>CHANGELOG.md cut.</done>
</task>

<task type="auto">
  <name>Task 2: Commit CHANGELOG + create annotated tag v1.0.0 locally (NO push)</name>
  <read_first>
    - `docs/RELEASE.md` — read the release checklist; honor its step order.
    - Current git state: `git status` should be clean except for the CHANGELOG change and the STATE.md update from Task 3.
  </read_first>
  <action>
Execute the following git commands. **Do NOT run `git push`.**

```bash
# 1) Stage + commit CHANGELOG (this is a separate commit from the tag point)
git add CHANGELOG.md
git commit -m "release: prepare v1.0.0 — Mark-III closure

- Cut CHANGELOG.md [Unreleased] to [1.0.0] - 2026-04-22.
- Record Phase 8 (UI redesign + /bienvenida + pa11y-ci + users.nombre).
- Mark-III complete: 8 phases, 74 requirements delivered.

Next steps (manual):
  git push --tags origin feature/Mark-III
  See docs/DEPLOY_LAN.md for LAN deploy.
"

# 2) Create annotated tag pointing at HEAD
git tag -a v1.0.0 -m "Nexo v1.0.0 — Mark-III closure (2026-04-22)

Primera release productiva de Nexo. Sucesor del monolito
'OEE Planta' de ECS Mobility, manteniendo stack (FastAPI +
Jinja2 + Alpine + Tailwind CDN + Postgres 16 + SQL Server
dbizaro read-only + Caddy internal TLS).

Phases entregadas:
1. Naming + Higiene + CI (Phase 1 / Sprint 0)
2. Identidad — auth + RBAC + audit (Phase 2 / Sprint 1)
3. Capa de datos (Phase 3 / Sprint 2)
4. Consultas pesadas — preflight + postflight (Phase 4 / Sprint 3)
5. UI por roles (Phase 5 / Sprint 4)
6. Despliegue LAN HTTPS (Phase 6 / Sprint 5)
7. DevEx hardening (Phase 7 / Sprint 6)
8. Rediseño UI modo claro moderno (Phase 8 / Sprint 7)

Requirements cerrados: 74 / 74.

Deploy productivo: docs/DEPLOY_LAN.md.
Release checklist: docs/RELEASE.md.
"

# 3) Verify the tag is annotated (shows the message)
git show --no-patch v1.0.0
```

If the last commit message author email or GPG signing differs from
the session default, that's fine — the tag is local and can be re-cut
by the operator after review. Do NOT set `-c commit.gpgsign=false` or
pass `--no-verify` — if the pre-commit hook fails, fix the underlying
issue and create a new commit.

**Absolute invariant: NO `git push`, NO `git push --tags`, NO
`git push --force`. The tag stays local until the operator reviews
and pushes manually.**
  </action>
  <acceptance_criteria>
    - `git tag --list v1.0.0` returns `v1.0.0` (exactly).
    - `git cat-file -t v1.0.0` returns `tag` (annotated, not lightweight).
    - `git log --oneline -1` shows the release commit.
    - `git rev-list --no-walk v1.0.0` equals `git rev-parse HEAD`.
    - `git status --porcelain` output (after Task 3 commit) is empty.
    - `grep -c "v1.0.0" $(git rev-parse --git-dir)/refs/tags/v1.0.0 2>/dev/null || git tag --list v1.0.0` returns 1 (the tag exists).
    - The remote `origin/feature/Mark-III` should still point to the same commit as before this plan (the plan's commits are AHEAD of the remote). `git rev-parse HEAD` !=  `git rev-parse origin/feature/Mark-III`.
  </acceptance_criteria>
  <verify>
    <automated>git tag --list v1.0.0 | grep -q "^v1.0.0$" &amp;&amp; test "$(git cat-file -t v1.0.0)" = "tag" &amp;&amp; test "$(git rev-list --no-walk v1.0.0)" = "$(git rev-parse HEAD)"</automated>
  </verify>
  <done>Local annotated tag v1.0.0 created. No push executed.</done>
</task>

<task type="auto">
  <name>Task 3: Update .planning/STATE.md with next-steps guidance for the operator</name>
  <read_first>
    - `.planning/STATE.md` in full.
    - `docs/DEPLOY_LAN.md` — the runbook the next step references.
    - `docs/RELEASE.md` — the checklist.
  </read_first>
  <action>
Update `.planning/STATE.md` to reflect Phase 8 closure and leave a
clear "next manual steps" section at the top or under Session
Continuity. Minimal patch:

- Update `milestone_name`, `status`, `stopped_at`, `last_updated`,
  `last_activity`, `progress.completed_phases`, `progress.percent` so
  they reflect 8/8 phases and 100% of Phase 8 plans complete.
- In the "Current Position" section, set `Phase: 8`, `Plan: 11
  Complete`, `Status: Mark-III shipped — awaiting manual push + deploy`.
- Append to "Accumulated Context > Decisions" a short paragraph:
  `Plan 08-11 decisions: CHANGELOG cut 2026-04-22; v1.0.0 annotated
  tag creado en local; git push + deploy deferred hasta revisión
  manual del operador.`
- Under `Session Continuity`, replace the `Stopped at:` value with:
  `Mark-III complete. v1.0.0 tag created locally — push manually.`
- Add a new `Next manual steps` block at the BOTTOM of the file:

  ```markdown
  ## Next Manual Steps (operator)

  Mark-III has shipped locally. To finalise:

  1. **Review Claude's auto-selected UI variants** (sketch skills):
     ```bash
     for f in .claude/skills/sketch-findings-*/SKILL.md; do
       echo "=== $f ==="
       head -20 "$f"
     done
     ```
     If any variant is off, `git log --oneline --grep=08-0` finds
     that screen's commit; `git revert <sha>` rewinds it. Then
     re-plan that screen specifically with `/gsd-plan-phase 8 --screen=<name>`.

  2. **Push to origin**:
     ```bash
     git push origin feature/Mark-III
     git push --tags origin feature/Mark-III
     ```

  3. **Deploy to LAN**:
     Follow `docs/DEPLOY_LAN.md` section by section.
     Smoke check: `bash tests/infra/deploy_smoke.sh` (11 checks).

  4. **Close Mark-III milestone**:
     - Move ROADMAP.md Phase 8 checkbox to `[x]`.
     - Create a Mark-IV milestone file in `.planning/` when ready.
  ```
  </action>
  <acceptance_criteria>
    - `grep -c "## Next Manual Steps (operator)" .planning/STATE.md` returns 1.
    - `grep -c "v1.0.0" .planning/STATE.md` returns 1 or more.
    - `grep -c "git push --tags origin feature/Mark-III" .planning/STATE.md` returns 1.
    - `grep -c "docs/DEPLOY_LAN.md" .planning/STATE.md` returns 1 or more.
    - `grep -c "Mark-III complete" .planning/STATE.md` returns 1 or more.
    - After this task the commit including STATE.md goes in a SEPARATE commit (or amended into the release commit). Recommended: a separate commit like `docs(state): record Mark-III closure and v1.0.0 local tag`. Whichever is chosen, the plan summary documents it.
  </acceptance_criteria>
  <verify>
    <automated>grep -q "## Next Manual Steps (operator)" .planning/STATE.md &amp;&amp; grep -q "v1.0.0" .planning/STATE.md &amp;&amp; grep -q "git push --tags origin feature/Mark-III" .planning/STATE.md</automated>
  </verify>
  <done>STATE.md captures the closure + next-step handoff.</done>
</task>

</tasks>

<threat_model>
## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-08-11-01 | Repudiation | Tag was pushed by accident | mitigate | Acceptance criterion asserts `HEAD != origin/feature/Mark-III` post-plan; plan text forbids `git push` in three separate places. |
| T-08-11-02 | Tampering | CHANGELOG narrative does not reflect reality | mitigate | The bullets list exactly what Phase 8 plans 01-10 shipped; each can be cross-checked against the SUMMARY files. |
| T-08-11-03 | Information Disclosure | Tag message mentions internal components | accept | Tags remain local until operator reviews. |
</threat_model>

<verification>
1. `pytest tests/ -x -q` green.
2. `git tag --list` shows `v1.0.0`.
3. `git cat-file -t v1.0.0` → `tag` (annotated).
4. `git log --oneline -5` shows the release commit + STATE update
   commit ahead of any previous commit.
5. `git ls-remote --tags origin | grep v1.0.0` → empty (NOT pushed).
6. Manual: the operator, returning, runs `cat .planning/STATE.md` and
   finds clear instructions.
</verification>

<success_criteria>
- CHANGELOG.md bumped to 1.0.0.
- Annotated tag `v1.0.0` exists locally.
- No `git push` was executed.
- STATE.md carries the next-steps handoff.
- Full test suite green at release commit.
</success_criteria>

<output>
Create `.planning/phases/08-redise-o-ui-modo-claro-moderno/08-11-SUMMARY.md` with:

- CHANGELOG diff summary.
- Tag sha + annotated flag.
- STATE.md changes.
- Reminder: `git push --tags origin feature/Mark-III` + `docs/DEPLOY_LAN.md` are deferred to the operator.
- Mark-III complete.
</output>
