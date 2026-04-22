# RELEASE.md — Checklist de release versionado

> Audiencia: quien corta una release de Nexo (tipicamente el dev lead o
> responsable de la milestone). Se ejecuta en un dia planificado, no en
> respuesta a incidencia.
> Ver [RUNBOOK.md](RUNBOOK.md) para recuperacion si el release rompe prod.

Ultima revision: 2026-04-22 (Sprint 6 / Phase 7).

---

## Versionado

Nexo usa **Semantic Versioning** (https://semver.org), formato `vMAJOR.MINOR.PATCH`:

- **MAJOR** (`vX.0.0`) — milestones estructurales (Mark-III, Mark-IV).
- **MINOR** (`v1.X.0`) — features no-breaking dentro de una milestone.
- **PATCH** (`v1.0.X`) — fixes, hotfixes, parches de seguridad.

Mapa de versiones:

- **v1.0.0** — cierre Mark-III (Sprint 6 / Phase 7 verificado).
- **v1.0.x** — fixes sobre Mark-III (ej. `v1.0.1`, `v1.0.2`).
- **v2.0.0** — cierre Mark-IV (fecha TBD).

Calver (year-based) descartado: las milestones Mark-III/Mark-IV ya dan contexto
temporal implicito en el nombre de la rama.

**Tags son inmutables.** Si un release falla en produccion, se corta
`v1.0.1` con el fix. NUNCA reescribir un tag publicado
(`git push --force` sobre un tag rompe clones de otros devs y despliegues
automaticos). Misma regla que CLAUDE.md seccion "Politica de commits".

---

## Checklist pre-release

Marcar todos antes de iniciar el corte. Si algun punto falla, NO cortar release
— cerrar el blocker primero.

- [ ] Todos los plans de la phase de cierre marcados como completados en
      `.planning/ROADMAP.md` (seccion Progress con checkmark verde).
- [ ] CI verde en `feature/Mark-III`: todos los jobs pasaron en el ultimo
      push (lint matrix 3.11 + 3.12, test matrix 3.11 + 3.12, smoke, build).
- [ ] `pytest --cov=api --cov=nexo --cov-fail-under=60` exit 0 en local.
- [ ] `pre-commit run --all-files` exit 0 (sin warnings de ruff/mypy).
- [ ] `.planning/STATE.md` seccion "Blockers/Concerns" sin items criticos
      abiertos.
- [ ] `CHANGELOG.md` actualizado con seccion nueva `## [1.0.0] - 2026-MM-DD`
      (reemplazar `MM-DD` por la fecha real del corte).
- [ ] Docs de la milestone actualizados: `docs/ARCHITECTURE.md`,
      `docs/RUNBOOK.md`, `docs/CLAUDE.md` (si aplica), cross-links validados.

---

## Checklist de release (ejecucion)

Ejecutar en orden. Cada paso debe completarse antes del siguiente.

1. **Merge `feature/Mark-III` -> `main`** (desde tu maquina local):

   ```bash
   git checkout main
   git pull origin main
   git merge --no-ff feature/Mark-III -m "release: Mark-III v1.0.0"
   git push origin main
   ```

2. **Tag en `main`** (tag firmado, mensaje descriptivo):

   ```bash
   git tag -a v1.0.0 -m "Nexo Mark-III - LAN deployment + auth + RBAC + preflight + devex"
   git push origin v1.0.0
   ```

   Si el repo tiene branch protection y requieres PR para merge, sustituir el
   paso 1 por un PR `feature/Mark-III -> main` con squash/merge politico del
   equipo; el tag se sigue creando sobre el commit de merge en `main`.

3. **GitHub Release** desde el tag (usa el CHANGELOG como notas):

   ```bash
   gh release create v1.0.0 \
     --title "Nexo v1.0.0 - Mark-III" \
     --notes-file <(sed -n '/## \[1\.0\.0\]/,/## \[/p' CHANGELOG.md | head -n -1)
   ```

4. **Deploy en el servidor productivo** (ejecuta `scripts/deploy.sh` Phase 6):

   ```bash
   ssh <IP_NEXO> 'cd /opt/nexo && git pull && make deploy'
   ```

   `scripts/deploy.sh` hace: pre-backup atomic a `/var/backups/nexo/predeploy/`
   -> `git pull` -> `docker compose build` -> `docker compose up -d` -> smoke
   HTTPS local. Ver [DEPLOY_LAN.md](DEPLOY_LAN.md) para detalles.

5. **Smoke externo post-deploy** desde el servidor:

   ```bash
   ssh <IP_NEXO> 'cd /opt/nexo && bash tests/infra/deploy_smoke.sh'
   ```

   Aceptacion: exit 0 (0 fallos sobre los 11 checks que ejecuta
   `tests/infra/deploy_smoke.sh` — Phase 6 / DEPLOY-07).

6. **Verificacion manual** desde equipo LAN (no-automatizable):

   - `https://nexo.ecsmobility.local` -> pantalla de login carga sin warnings
     del browser (root CA de Caddy instalada correctamente).
   - Login con usuario `propietario` -> `/` (Centro de Mando) carga sin errores
     en consola.
   - `/api/health` -> `{"status":"ok","services":{"db":{"ok":true},...}}` con
     todos los servicios `ok=true` (salvo MES si esta en ventana conocida).

7. **Anuncio interno** (email / chat ECS Mobility):

   - Link a la GitHub Release creada.
   - Highlights extraidos del CHANGELOG.md (3-5 bullets de la seccion
     `### Added` de la version).
   - Instrucciones para usuarios finales si hay cambios de UX (ej. nuevos
     permisos requeridos, cambios en el sidebar, nueva pantalla).
   - Link al runbook `docs/RUNBOOK.md` si hay procedimientos operativos nuevos.

---

## Rollback

Si el release rompe produccion y `RUNBOOK.md` no cubre el escenario:

```bash
ssh <IP_NEXO>
cd /opt/nexo
git log --oneline -5
git checkout <tag-anterior>  # ej. v0.9.0
make prod-down && make prod-up
```

Restaurar BD desde `/var/backups/nexo/predeploy/<latest>.sql.gz` si el release
incluyo migracion destructiva:

```bash
gunzip < /var/backups/nexo/predeploy/<latest>.sql.gz | \
  docker compose exec -T db psql -U <POSTGRES_USER> -d <POSTGRES_DB>
```

Tras estabilizar, abrir issue de hotfix -> `v1.0.1` con el fix minimo que
arregla el breakage. NO reintentar el release v1.0.0 — los tags son inmutables.

---

## Referencias

- [CHANGELOG.md](../CHANGELOG.md) — historial completo Mark-III.
- [DEPLOY_LAN.md](DEPLOY_LAN.md) — deploy LAN detallado.
- [RUNBOOK.md](RUNBOOK.md) — escenarios de incidencia runtime.
- [ARCHITECTURE.md](ARCHITECTURE.md) — mapa tecnico del sistema.
- [scripts/deploy.sh](../scripts/deploy.sh) — ejecutor del deploy Phase 6.
- [tests/infra/deploy_smoke.sh](../tests/infra/deploy_smoke.sh) — smoke 11 checks.
