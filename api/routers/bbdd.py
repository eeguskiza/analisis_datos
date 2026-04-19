"""Explorador de bases de datos del SQL Server.

Plan 03-02 Task 3.6: elimina pyodbc y el par ``_get_conn_string`` /
``_connect`` / ``_connect_db``. Todas las ops de metadata corren
sobre ``engine_mes`` (SQLAlchemy + DSN ``DATABASE=dbizaro``). El
handler POST /query delega SQL validada a
``MesRepository.consulta_readonly`` (D-05: el whitelist queda en el
router). El whitelist se expone como funcion module-level
``_validate_sql`` para poder ejercerlo desde un contract test
(tests/data/test_bbdd_whitelist.py).

Limitacion Mark-III: el explorer queda restringido a ``dbizaro``
porque ``engine_mes`` tiene el catalog baked en el DSN (DATA-09).
Mark-IV podra construir engines ad-hoc por catalog si se necesita
navegar otras BDs del servidor.
"""
from __future__ import annotations

import re

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy import text

from api.deps import EngineMes
from nexo.data.repositories.mes import MesRepository
from nexo.data.sql.loader import load_sql
from nexo.services.auth import require_permission

router = APIRouter(
    prefix="/bbdd",
    tags=["bbdd"],
    dependencies=[Depends(require_permission("bbdd:read"))],
)


# ── Whitelist anti-DDL (D-05) ────────────────────────────────────────────
# Extraido a funcion module-level para poder testearlo desde
# tests/data/test_bbdd_whitelist.py (contract test del Threat T-03-02-01).

_FORBIDDEN_TOKENS = [
    "INSERT", "UPDATE", "DELETE", "MERGE", "TRUNCATE", "DROP",
    "ALTER", "CREATE", "GRANT", "REVOKE", "EXEC", "EXECUTE",
    "SP_", "XP_", "BACKUP", "RESTORE", "SHUTDOWN", "BULK",
    "OPENROWSET", "OPENQUERY",
]


def _validate_sql(sql: str) -> None:
    """Valida que ``sql`` sea SELECT/WITH puro y sin multi-statement.

    Raises HTTPException(400) si rechaza; None si acepta. Preservado
    literalmente del handler pre-refactor (palabra por palabra — los
    casos de test cubren los 9 DDL/DML variantes + multi-statement).
    """
    if not sql or not sql.strip():
        raise HTTPException(400, "Falta el campo 'sql'")

    # 1) Debe empezar por SELECT o WITH (CTE)
    sql_stripped = sql.lstrip(" \t\r\n;(")
    first_word = sql_stripped.split(None, 1)[0].upper() if sql_stripped else ""
    if first_word not in {"SELECT", "WITH"}:
        raise HTTPException(400, "Solo se permiten SELECT/WITH")

    # 2) Rechazar palabras reservadas peligrosas (busqueda por palabra
    #    entera, insensible a mayus/minus).
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sql.upper())
    for bad in _FORBIDDEN_TOKENS:
        if bad in tokens:
            raise HTTPException(400, f"Palabra prohibida: {bad}")
        # SP_ / XP_ son prefijos, comprobar cualquier token que empiece asi
        if bad.endswith("_"):
            for t in tokens:
                if t.startswith(bad):
                    raise HTTPException(400, f"Prefijo prohibido: {bad}")

    # 3) Rechazar punto y coma internos (multi-statement). Permitimos
    #    un ';' final, pero no en medio.
    stripped = sql.rstrip().rstrip(";").strip()
    if ";" in stripped:
        raise HTTPException(400, "No se permiten multiples sentencias")


def _get_default_db() -> str:
    """Mark-III: el explorer queda fijado a dbizaro (engine_mes DSN)."""
    from api.config import settings
    return settings.mes_db  # "dbizaro"


def _guard_mark3_db_scope(database: str) -> None:
    """Rechaza peticiones que piden navegar una BD distinta de dbizaro.

    Mark-III limitation: ``engine_mes`` tiene el catalog baked en el
    DSN; navegar otra BD requeriria un engine ad-hoc. Devolvemos 400
    claro en vez de ejecutar queries que silenciosamente caen en dbizaro.
    """
    default = _get_default_db()
    if database and database != default:
        raise HTTPException(
            400,
            f"Mark-III: explorer limitado a '{default}'. "
            f"Otras BDs ({database}) se habilitaran en Mark-IV con engines ad-hoc.",
        )


@router.get("/databases")
def list_databases(engine_mes: EngineMes):
    """Lista las bases de datos disponibles en el servidor."""
    try:
        with engine_mes.connect() as conn:
            rows = conn.execute(text(load_sql("mes/bbdd_list_databases"))).fetchall()
        return {
            "default": _get_default_db(),
            "databases": [
                {"name": r[0], "state": r[1], "size_mb": float(r[2] or 0)}
                for r in rows
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Error conectando: {exc}")


@router.get("/tables")
def list_tables(engine_mes: EngineMes, database: str = Query(...)):
    """Lista tablas de una base de datos con conteo de filas."""
    _guard_mark3_db_scope(database)
    try:
        with engine_mes.connect() as conn:
            rows = conn.execute(text(load_sql("mes/bbdd_list_tables"))).fetchall()
        return {
            "database": database,
            "tables": [
                {"schema": r[0], "table": r[1], "rows": int(r[2] or 0)}
                for r in rows
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Error: {exc}")


@router.get("/columns")
def list_columns(
    engine_mes: EngineMes,
    database: str = Query(...),
    schema: str = Query("dbo"),
    table: str = Query(...),
):
    """Lista columnas de una tabla con tipos, nullability y PKs."""
    _guard_mark3_db_scope(database)
    try:
        with engine_mes.connect() as conn:
            rows = conn.execute(
                text(load_sql("mes/bbdd_list_columns")),
                {"schema": schema, "table": table},
            ).fetchall()
        return {
            "database": database,
            "schema": schema,
            "table": table,
            "columns": [
                {
                    "name": r[0],
                    "type": r[1] + (f"({r[2]})" if r[2] else ""),
                    "nullable": r[3] == "YES",
                    "pk": bool(r[4]),
                }
                for r in rows
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Error: {exc}")


@router.get("/schema")
def full_schema(
    engine_mes: EngineMes,
    database: str = Query(...),
    light: bool = Query(False),
):
    """
    Esquema de la BBDD: tablas, columnas y relaciones FK.

    Con light=true solo devuelve nombres de tablas + filas + FKs (para BBDDs grandes).
    Con light=false devuelve tambien columnas completas.
    """
    _guard_mark3_db_scope(database)
    try:
        with engine_mes.connect() as conn:
            # Siempre: tablas con conteo de filas
            table_rows = conn.execute(text("""
                SELECT s.name, t.name, p.rows
                FROM sys.tables t
                JOIN sys.schemas s ON t.schema_id = s.schema_id
                JOIN sys.partitions p ON t.object_id = p.object_id AND p.index_id IN (0, 1)
                ORDER BY s.name, t.name
            """)).fetchall()

            tables = {}
            for r in table_rows:
                key = f"{r[0]}.{r[1]}"
                tables[key] = {"schema": r[0], "name": r[1], "rows": int(r[2] or 0), "columns": []}

            # Columnas solo en modo completo
            if not light:
                col_rows = conn.execute(text("""
                    SELECT
                        c.TABLE_SCHEMA, c.TABLE_NAME, c.COLUMN_NAME, c.DATA_TYPE,
                        CASE WHEN pk.COLUMN_NAME IS NOT NULL THEN 1 ELSE 0 END AS is_pk
                    FROM INFORMATION_SCHEMA.COLUMNS c
                    LEFT JOIN (
                        SELECT ku.TABLE_SCHEMA, ku.TABLE_NAME, ku.COLUMN_NAME
                        FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                        JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE ku ON tc.CONSTRAINT_NAME = ku.CONSTRAINT_NAME
                        WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
                    ) pk ON c.TABLE_SCHEMA = pk.TABLE_SCHEMA
                         AND c.TABLE_NAME = pk.TABLE_NAME
                         AND c.COLUMN_NAME = pk.COLUMN_NAME
                    WHERE c.TABLE_SCHEMA NOT IN ('sys', 'INFORMATION_SCHEMA')
                    ORDER BY c.TABLE_SCHEMA, c.TABLE_NAME, c.ORDINAL_POSITION
                """)).fetchall()
                for r in col_rows:
                    key = f"{r[0]}.{r[1]}"
                    if key in tables:
                        tables[key]["columns"].append({"name": r[2], "type": r[3], "pk": bool(r[4])})

            # Foreign keys (siempre)
            fk_rows = conn.execute(text("""
                SELECT
                    OBJECT_SCHEMA_NAME(fk.parent_object_id),
                    OBJECT_NAME(fk.parent_object_id),
                    cp.name,
                    OBJECT_SCHEMA_NAME(fk.referenced_object_id),
                    OBJECT_NAME(fk.referenced_object_id),
                    cr.name
                FROM sys.foreign_keys fk
                JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
                JOIN sys.columns cp ON fkc.parent_object_id = cp.object_id AND fkc.parent_column_id = cp.column_id
                JOIN sys.columns cr ON fkc.referenced_object_id = cr.object_id AND fkc.referenced_column_id = cr.column_id
            """)).fetchall()

        relationships = [
            {"from": f"{r[0]}.{r[1]}", "from_col": r[2],
             "to": f"{r[3]}.{r[4]}", "to_col": r[5]}
            for r in fk_rows
        ]

        return {
            "database": database,
            "light": light,
            "tables": list(tables.values()),
            "relationships": relationships,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Error: {exc}")


@router.get("/buscar-rutas")
def buscar_tablas_rutas(
    engine_mes: EngineMes,
    database: str = Query("dbizaro"),
):
    """
    Descubre candidatas a tabla de rutas en la BBDD dada.

    Busca en el esquema admuser tablas cuyo nombre contenga patrones tipo
    ruta/fase/operacion/camino, y puntua cada candidata segun:
      - si tiene columna que referencia a fprolof (referencia)
      - si tiene columna que referencia a fmesrec (centro de trabajo)
      - si tiene columna de secuencia/orden (para identificar ultima fase)

    Devuelve lista ranqueada con columnas y una muestra de 5 filas.
    """
    if not re.match(r'^[\w]+$', database):
        raise HTTPException(400, "Nombre de BBDD invalido")
    _guard_mark3_db_scope(database)

    try:
        with engine_mes.connect() as conn:
            # 1) Candidatas por nombre
            patterns = ["%rut%", "%fas%", "%ope%", "%ruf%", "%cam%", "%prc%", "%prr%", "%prf%"]
            # Named params :p0, :p1... para evitar abuse con IN expanding
            params = {f"p{i}": p for i, p in enumerate(patterns)}
            placeholders = " OR ".join([f"TABLE_NAME LIKE :p{i}" for i in range(len(patterns))])
            cand_sql = text(f"""
                SELECT TABLE_SCHEMA, TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = 'admuser'
                  AND TABLE_TYPE = 'BASE TABLE'
                  AND ({placeholders})
                ORDER BY TABLE_NAME
            """)
            candidatas = [(r[0], r[1]) for r in conn.execute(cand_sql, params).fetchall()]

            resultados = []
            for schema, table in candidatas:
                # Columnas
                col_rows = conn.execute(
                    text("""
                        SELECT COLUMN_NAME, DATA_TYPE
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
                        ORDER BY ORDINAL_POSITION
                    """),
                    {"schema": schema, "table": table},
                ).fetchall()
                cols = [(r[0], r[1]) for r in col_rows]
                col_names_lower = [c[0].lower() for c in cols]

                # Conteo de filas (identificadores sanitizados arriba via regex ^[\w]+$)
                try:
                    n_rows = int(
                        conn.execute(
                            text(f"SELECT COUNT(*) FROM [{schema}].[{table}]")
                        ).scalar() or 0
                    )
                except Exception:
                    n_rows = 0

                # Score heuristico
                score = 0
                hints = []
                name_lower = table.lower()
                if "rut" in name_lower: score += 3; hints.append("nombre-ruta")
                if "fas" in name_lower: score += 2; hints.append("nombre-fase")
                if "ope" in name_lower: score += 1; hints.append("nombre-operacion")
                if any("lo030" in c or "ref" in c or "art" in c for c in col_names_lower):
                    score += 2; hints.append("tiene-ref")
                if any(c in col_names_lower for c in ("re010", "ct", "seccion", "centro")):
                    score += 2; hints.append("tiene-ct")
                if any("orden" in c or "secuencia" in c or "nfase" in c or "fase" in c for c in col_names_lower):
                    score += 2; hints.append("tiene-secuencia")

                # Muestra
                try:
                    result = conn.execute(text(f"SELECT TOP 5 * FROM [{schema}].[{table}]"))
                    sample_cols = list(result.keys())
                    sample_rows = [
                        [str(v) if v is not None else None for v in row]
                        for row in result.fetchall()
                    ]
                except Exception:
                    sample_cols, sample_rows = [], []

                resultados.append({
                    "schema": schema,
                    "table": table,
                    "rows": n_rows,
                    "score": score,
                    "hints": hints,
                    "columns": [{"name": c[0], "type": c[1]} for c in cols],
                    "sample": {"columns": sample_cols, "rows": sample_rows},
                })

        resultados.sort(key=lambda x: (-x["score"], x["table"]))
        return {"database": database, "candidatas": resultados}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Error: {exc}")


@router.get("/explorar-rutas-detalle")
def explorar_rutas_detalle(
    engine_mes: EngineMes,
    database: str = Query("dbizaro"),
):
    """
    Devuelve muestras grandes de fprorut, fprorut_edm y fproope para
    entender como se enlaza una fase de ruta con una operacion (CT/tiempo).
    """
    if not re.match(r'^[\w]+$', database):
        raise HTTPException(400, "Nombre de BBDD invalido")
    _guard_mark3_db_scope(database)

    try:
        with engine_mes.connect() as conn:
            out = {}

            # 1) Resumen de fprorut: cuantas referencias distintas, fases por ref
            r = conn.execute(text("""
                SELECT COUNT(DISTINCT ru000) AS n_refs,
                       COUNT(*) AS n_filas,
                       MIN(ru020) AS fase_min, MAX(ru020) AS fase_max,
                       MIN(ru030) AS ru030_min, MAX(ru030) AS ru030_max
                FROM admuser.fprorut
            """)).fetchone()
            out["fprorut_resumen"] = {
                "n_referencias": int(r[0] or 0),
                "n_filas": int(r[1] or 0),
                "fase_min": float(r[2]) if r[2] is not None else None,
                "fase_max": float(r[3]) if r[3] is not None else None,
                "ru030_min": float(r[4]) if r[4] is not None else None,
                "ru030_max": float(r[5]) if r[5] is not None else None,
            }

            # 2) 30 filas de fprorut variadas (distintas referencias)
            rows = conn.execute(text("""
                SELECT TOP 30 ru999, ru000, ru010, ru020, ru030
                FROM admuser.fprorut
                WHERE ru000 IN (
                    SELECT TOP 6 ru000 FROM admuser.fprorut
                    GROUP BY ru000
                    ORDER BY COUNT(*) DESC
                )
                ORDER BY ru000, ru020
            """)).fetchall()
            out["fprorut_muestra"] = {
                "columns": ["ru999", "ru000", "ru010", "ru020", "ru030"],
                "rows": [[str(v) if v is not None else None for v in r] for r in rows],
            }

            # 3) fprorut_edm: 30 filas variadas
            rows = conn.execute(text("""
                SELECT TOP 30 ru999, ru000, ru010, ru020, ru030, ru040, ru050,
                              ru060, ru070, ru080, ru090, ru100, ru110, ru120
                FROM admuser.fprorut_edm
                WHERE ru000 IN (
                    SELECT TOP 6 ru000 FROM admuser.fprorut_edm
                    GROUP BY ru000
                    ORDER BY COUNT(*) DESC
                )
                ORDER BY ru000, ru020
            """)).fetchall()
            out["fprorut_edm_muestra"] = {
                "columns": ["ru999", "ru000", "ru010", "ru020", "ru030", "ru040",
                            "ru050", "ru060", "ru070", "ru080", "ru090", "ru100",
                            "ru110", "ru120"],
                "rows": [[str(v) if v is not None else None for v in r] for r in rows],
            }

            # 4) Todas las operaciones de fproope (solo 87)
            rows = conn.execute(text("""
                SELECT op999, op000, op010, op020, op030, op040, op050, op055,
                       op060, op090, op100, op110, op120
                FROM admuser.fproope
                ORDER BY op000
            """)).fetchall()
            out["fproope_todas"] = {
                "columns": ["op999", "op000", "op010", "op020", "op030", "op040",
                            "op050", "op055", "op060", "op090", "op100", "op110", "op120"],
                "rows": [[str(v) if v is not None else None for v in r] for r in rows],
            }

            # 5) Intento de JOIN — probar si fprorut.ru030 enlaza con fproope.op000
            rows = conn.execute(text("""
                SELECT TOP 30
                    r.ru000 AS referencia,
                    r.ru020 AS fase,
                    r.ru030 AS ru030,
                    o.op000 AS op_id,
                    o.op010 AS operacion,
                    o.op030 AS ct,
                    o.op040 AS tiempo
                FROM admuser.fprorut r
                LEFT JOIN admuser.fproope o ON o.op000 = r.ru030
                WHERE r.ru000 IN (
                    SELECT TOP 5 ru000 FROM admuser.fprorut
                    GROUP BY ru000
                    ORDER BY COUNT(*) DESC
                )
                ORDER BY r.ru000, r.ru020
            """)).fetchall()
            out["join_prueba_ru030_op000"] = {
                "columns": ["referencia", "fase", "ru030", "op_id", "operacion", "ct", "tiempo"],
                "rows": [[str(v) if v is not None else None for v in r] for r in rows],
            }

            # 6) Verificar distintos valores de ru030 (por si no es siempre 0)
            rows = conn.execute(text("""
                SELECT TOP 20 ru030, COUNT(*) AS n
                FROM admuser.fprorut
                GROUP BY ru030
                ORDER BY COUNT(*) DESC
            """)).fetchall()
            out["fprorut_distribucion_ru030"] = {
                "columns": ["ru030", "n"],
                "rows": [[str(v) if v is not None else None for v in r] for r in rows],
            }

        return out
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Error: {exc}")


@router.post("/query")
def query_readonly(
    engine_mes: EngineMes,
    payload: dict = Body(...),
):
    """
    Ejecuta una consulta SELECT de solo lectura en la BBDD indicada.

    Body:
        {
          "database": "dbizaro",
          "sql": "SELECT TOP 10 * FROM admuser.fprorut",
          "max_rows": 500  // opcional, por defecto 500, tope 5000
        }

    Valida que la sentencia sea SELECT/WITH (CTE) pura. Rechaza cualquier
    palabra reservada de DML/DDL para garantizar solo consulta.
    """
    database = payload.get("database") or _get_default_db()
    sql = (payload.get("sql") or "").strip()
    max_rows = int(payload.get("max_rows") or 500)
    max_rows = max(1, min(max_rows, 5000))

    if not re.match(r"^[\w]+$", database):
        raise HTTPException(400, "Nombre de BBDD invalido")
    _guard_mark3_db_scope(database)

    # Whitelist (D-05) ANTES de construir el repo.
    _validate_sql(sql)

    try:
        repo = MesRepository(engine=engine_mes)
        result = repo.consulta_readonly(sql, database)

        # Stringify values y aplicar cap max_rows (MesRepository no lo hace).
        all_rows = result["rows"]
        rows_capped = all_rows[:max_rows]
        rows_str = [
            [str(v) if v is not None else None for v in row]
            for row in rows_capped
        ]
        truncated = len(all_rows) > max_rows
        return {
            "database": database,
            "columns": result["columns"],
            "rows": rows_str,
            "n_rows": len(rows_str),
            "truncated": truncated,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Error: {exc}")


@router.get("/preview")
def preview_data(
    engine_mes: EngineMes,
    database: str = Query(...),
    schema: str = Query("dbo"),
    table: str = Query(...),
    limit: int = Query(50, ge=1, le=500),
):
    """Preview de las primeras filas de una tabla."""
    # Sanitize identifiers to prevent SQL injection
    if not re.match(r'^[\w]+$', database) or not re.match(r'^[\w]+$', schema) or not re.match(r'^[\w]+$', table):
        raise HTTPException(400, "Nombre invalido")
    _guard_mark3_db_scope(database)

    try:
        with engine_mes.connect() as conn:
            result = conn.execute(text(f"SELECT TOP {limit} * FROM [{schema}].[{table}]"))
            columns = list(result.keys())
            rows = []
            for row in result.fetchall():
                rows.append([
                    str(v) if v is not None else None
                    for v in row
                ])
        return {"columns": columns, "rows": rows, "total": len(rows)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Error: {exc}")
