# Data Migration Notes

Decisiones de migración de datos tomadas durante Mark-III. Cada entrada
documenta: qué había, qué se movió/borró, dónde está el backup (si aplica),
y el hash git de referencia para recuperación.

---

## 2026-04-18 — `data/oee.db` (SQLite residual)

**Contexto**: `data/oee.db` es un SQLite de 172 KB que era la BD principal
de una versión anterior de la app, antes de migrar a SQL Server
(`ecs_mobility` + schemas `cfg/oee/luk4`). El código actual **no lo lee**:
`api/database.py` conecta a SQL Server vía `mssql+pyodbc://`, no a
`sqlite:///data/oee.db`. Había quedado trackeado por inercia.

**Inspección** (Sprint 0 commit 5, via `python3 -c "import sqlite3; ..."`):

| Tabla | Rows | Contenido |
|-------|------|-----------|
| `ciclos` | 58 | Ciclos teóricos por máquina+referencia (`maquina`, `referencia`, `tiempo_ciclo`, `updated_at`). |
| `contactos` | 0 | Vacía. |
| `datos_produccion` | 1 433 | Datos de producción extraídos de IZARO (`recurso`, `seccion`, `fecha`, `h_ini`, `h_fin`, etc.). |
| `ejecuciones` | 10 | Audit trail de ejecuciones de pipeline (`fecha_inicio`, `fecha_fin`, `source`, `status`, `modulos`). |
| `informes_meta` | 37 | Metadatos de PDFs generados (`fecha`, `seccion`, `maquina`, `modulo`, `pdf_path`). |
| `recursos` | 7 | Catálogo de centros de trabajo (`centro_trabajo`, `nombre`, `seccion`, `activo`). |

**Decisión**: **preservar backup offline + borrar del tracking**.

Razón: aunque el tamaño (172 KB) queda por debajo del umbral arbitrario
de "1 MB" del PLAN, las **1 433 filas de `datos_produccion`** y las
**10 `ejecuciones`** son datos históricos reales que representan trabajo
de extracción ya hecho. Tirarlos sin backup sería pérdida irreversible
incluso si hoy no se consumen.

**Acciones ejecutadas**:

1. `cp data/oee.db data/backups/oee_db_snapshot.sqlite` — copia offline.
2. `data/backups/` añadido a `.gitignore` (entrada nueva bajo "Datos variables"). El backup NO se trackea.
3. `git rm data/oee.db` — removido del tracking.
4. Este documento creado.

**Hash git de referencia para recuperación**: último commit que tocó
`data/oee.db` en historial es **`c611b78`** ("Best"). Para recuperar:

```bash
# Recuperar blob desde historial (si se perdiera el backup offline):
git show c611b78:data/oee.db > data/oee.db_recovered
```

El blob sigue en el historial git hasta que se decida reescribir (ver
`docs/SECURITY_AUDIT.md` para la política de reescritura de historial).

**Qué no se hizo**:

- No se ejecutó script de import de los datos a `ecs_mobility`. Si en el
  futuro se necesitan esas 1 433 filas de `datos_produccion` en SQL Server,
  hay que escribir un import ad-hoc — no está en scope Mark-III.
- No se verificó solape entre `data/oee.db` y lo que hay hoy en
  `ecs_mobility.oee.datos`. Asumimos que lo que está en SQL Server es más
  reciente y cubre.

---

*Última actualización: 2026-04-18 (Sprint 0 commit 5).*
