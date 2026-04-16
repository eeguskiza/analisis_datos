"""Explorador de bases de datos del SQL Server."""
from __future__ import annotations

import re

from fastapi import APIRouter, Body, HTTPException, Query

router = APIRouter(prefix="/bbdd", tags=["bbdd"])


def _get_conn_string(database: str | None = None):
    """Construye connection string usando config (con credenciales de .env)."""
    from api.services.db import get_config
    from OEE.db.connector import detectar_driver
    cfg = get_config()
    original_db = cfg.get("database", "")
    db = database or original_db

    driver = (cfg.get("driver") or "").strip() or detectar_driver()
    server = cfg["server"]
    port = cfg.get("port") or "1433"
    user = (cfg.get("user") or "").strip()
    password = cfg.get("password") or ""

    return (
        f"DRIVER={{{driver}}};"
        f"SERVER={server},{port};"
        f"DATABASE={db};"
        f"UID={user};"
        f"PWD={password};"
        "Encrypt=yes;TrustServerCertificate=yes;"
    ), original_db


def _connect():
    """Conexion a master para listar databases."""
    import pyodbc
    cs, original_db = _get_conn_string("master")
    return pyodbc.connect(cs, timeout=15), original_db


def _connect_db(database: str):
    """Conexion a una base de datos concreta."""
    import pyodbc
    cs, _ = _get_conn_string(database)
    return pyodbc.connect(cs, timeout=15)


@router.get("/databases")
def list_databases():
    """Lista las bases de datos disponibles en el servidor."""
    try:
        conn, default_db = _connect()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT d.name, d.state_desc,
                   CAST(COALESCE(SUM(f.size) * 8.0 / 1024, 0) AS DECIMAL(10,1)) AS size_mb
            FROM sys.databases d
            LEFT JOIN sys.master_files f ON d.database_id = f.database_id
            WHERE d.name NOT IN ('master', 'tempdb', 'model', 'msdb')
            GROUP BY d.name, d.state_desc
            ORDER BY d.name
        """)
        rows = cursor.fetchall()
        conn.close()
        return {
            "default": default_db,
            "databases": [
                {"name": r[0], "state": r[1], "size_mb": float(r[2] or 0)}
                for r in rows
            ],
        }
    except Exception as exc:
        raise HTTPException(502, f"Error conectando: {exc}")


@router.get("/tables")
def list_tables(database: str = Query(...)):
    """Lista tablas de una base de datos con conteo de filas."""
    try:
        conn = _connect_db(database)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                s.name AS schema_name,
                t.name AS table_name,
                p.rows AS row_count
            FROM sys.tables t
            JOIN sys.schemas s ON t.schema_id = s.schema_id
            JOIN sys.partitions p ON t.object_id = p.object_id AND p.index_id IN (0, 1)
            ORDER BY s.name, t.name
        """)
        rows = cursor.fetchall()
        conn.close()
        return {
            "database": database,
            "tables": [
                {"schema": r[0], "table": r[1], "rows": int(r[2] or 0)}
                for r in rows
            ],
        }
    except Exception as exc:
        raise HTTPException(502, f"Error: {exc}")


@router.get("/columns")
def list_columns(database: str = Query(...), schema: str = Query("dbo"), table: str = Query(...)):
    """Lista columnas de una tabla con tipos, nullability y PKs."""
    try:
        conn = _connect_db(database)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                c.COLUMN_NAME,
                c.DATA_TYPE,
                c.CHARACTER_MAXIMUM_LENGTH,
                c.IS_NULLABLE,
                CASE WHEN pk.COLUMN_NAME IS NOT NULL THEN 1 ELSE 0 END AS is_pk
            FROM INFORMATION_SCHEMA.COLUMNS c
            LEFT JOIN (
                SELECT ku.TABLE_SCHEMA, ku.TABLE_NAME, ku.COLUMN_NAME
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE ku
                    ON tc.CONSTRAINT_NAME = ku.CONSTRAINT_NAME
                WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
            ) pk ON c.TABLE_SCHEMA = pk.TABLE_SCHEMA
                 AND c.TABLE_NAME = pk.TABLE_NAME
                 AND c.COLUMN_NAME = pk.COLUMN_NAME
            WHERE c.TABLE_SCHEMA = ?
              AND c.TABLE_NAME = ?
            ORDER BY c.ORDINAL_POSITION
        """, (schema, table))
        rows = cursor.fetchall()
        conn.close()
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
    except Exception as exc:
        raise HTTPException(502, f"Error: {exc}")


@router.get("/schema")
def full_schema(database: str = Query(...), light: bool = Query(False)):
    """
    Esquema de la BBDD: tablas, columnas y relaciones FK.

    Con light=true solo devuelve nombres de tablas + filas + FKs (para BBDDs grandes).
    Con light=false devuelve tambien columnas completas.
    """
    try:
        conn = _connect_db(database)
        cursor = conn.cursor()

        # Siempre: tablas con conteo de filas
        cursor.execute("""
            SELECT s.name, t.name, p.rows
            FROM sys.tables t
            JOIN sys.schemas s ON t.schema_id = s.schema_id
            JOIN sys.partitions p ON t.object_id = p.object_id AND p.index_id IN (0, 1)
            ORDER BY s.name, t.name
        """)
        table_rows = cursor.fetchall()

        tables = {}
        for r in table_rows:
            key = f"{r[0]}.{r[1]}"
            tables[key] = {"schema": r[0], "name": r[1], "rows": int(r[2] or 0), "columns": []}

        # Columnas solo en modo completo
        if not light:
            cursor.execute("""
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
            """)
            for r in cursor.fetchall():
                key = f"{r[0]}.{r[1]}"
                if key in tables:
                    tables[key]["columns"].append({"name": r[2], "type": r[3], "pk": bool(r[4])})

        # Foreign keys (siempre)
        cursor.execute("""
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
        """)
        fk_rows = cursor.fetchall()
        conn.close()

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
    except Exception as exc:
        raise HTTPException(502, f"Error: {exc}")


@router.get("/buscar-rutas")
def buscar_tablas_rutas(database: str = Query("dbizaro")):
    """
    Descubre candidatas a tabla de rutas en la BBDD dada.

    Busca en el esquema admuser tablas cuyo nombre contenga patrones tipo
    ruta/fase/operacion/camino, y puntua cada candidata segun:
      - si tiene columna que referencia a fprolof (referencia)
      - si tiene columna que referencia a fmesrec (centro de trabajo)
      - si tiene columna de secuencia/orden (para identificar ultima fase)

    Devuelve lista ranqueada con columnas y una muestra de 5 filas.
    """
    import re
    import pyodbc
    if not re.match(r'^[\w]+$', database):
        raise HTTPException(400, "Nombre de BBDD invalido")

    try:
        conn = _connect_db(database)
        cursor = conn.cursor()

        # 1) Candidatas por nombre
        patterns = ["%rut%", "%fas%", "%ope%", "%ruf%", "%cam%", "%prc%", "%prr%", "%prf%"]
        placeholders = " OR ".join(["TABLE_NAME LIKE ?"] * len(patterns))
        cursor.execute(f"""
            SELECT TABLE_SCHEMA, TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = 'admuser'
              AND TABLE_TYPE = 'BASE TABLE'
              AND ({placeholders})
            ORDER BY TABLE_NAME
        """, patterns)
        candidatas = [(r[0], r[1]) for r in cursor.fetchall()]

        resultados = []
        for schema, table in candidatas:
            # Columnas
            cursor.execute("""
                SELECT COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                ORDER BY ORDINAL_POSITION
            """, (schema, table))
            cols = [(r[0], r[1]) for r in cursor.fetchall()]
            col_names_lower = [c[0].lower() for c in cols]

            # Conteo de filas
            try:
                cursor.execute(f"SELECT COUNT(*) FROM [{schema}].[{table}]")
                n_rows = int(cursor.fetchone()[0] or 0)
            except Exception:
                n_rows = 0

            # Score heuristico
            score = 0
            hints = []
            name_lower = table.lower()
            if "rut" in name_lower: score += 3; hints.append("nombre-ruta")
            if "fas" in name_lower: score += 2; hints.append("nombre-fase")
            if "ope" in name_lower: score += 1; hints.append("nombre-operacion")
            # Referencia-like (lo030, ref, articulo)
            if any("lo030" in c or "ref" in c or "art" in c for c in col_names_lower):
                score += 2; hints.append("tiene-ref")
            # CT/seccion-like
            if any(c in col_names_lower for c in ("re010", "ct", "seccion", "centro")):
                score += 2; hints.append("tiene-ct")
            # Numero de orden / secuencia
            if any("orden" in c or "secuencia" in c or "nfase" in c or "fase" in c for c in col_names_lower):
                score += 2; hints.append("tiene-secuencia")

            # Muestra
            try:
                cursor.execute(f"SELECT TOP 5 * FROM [{schema}].[{table}]")
                sample_cols = [d[0] for d in cursor.description]
                sample_rows = [
                    [str(v) if v is not None else None for v in row]
                    for row in cursor.fetchall()
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

        conn.close()
        resultados.sort(key=lambda x: (-x["score"], x["table"]))
        return {"database": database, "candidatas": resultados}
    except Exception as exc:
        raise HTTPException(502, f"Error: {exc}")


@router.get("/explorar-rutas-detalle")
def explorar_rutas_detalle(database: str = Query("dbizaro")):
    """
    Devuelve muestras grandes de fprorut, fprorut_edm y fproope para
    entender como se enlaza una fase de ruta con una operacion (CT/tiempo).
    """
    import re
    if not re.match(r'^[\w]+$', database):
        raise HTTPException(400, "Nombre de BBDD invalido")

    try:
        conn = _connect_db(database)
        cursor = conn.cursor()
        out = {}

        # 1) Resumen de fprorut: cuantas referencias distintas, fases por ref
        cursor.execute("""
            SELECT COUNT(DISTINCT ru000) AS n_refs,
                   COUNT(*) AS n_filas,
                   MIN(ru020) AS fase_min, MAX(ru020) AS fase_max,
                   MIN(ru030) AS ru030_min, MAX(ru030) AS ru030_max
            FROM admuser.fprorut
        """)
        r = cursor.fetchone()
        out["fprorut_resumen"] = {
            "n_referencias": int(r[0] or 0),
            "n_filas": int(r[1] or 0),
            "fase_min": float(r[2]) if r[2] is not None else None,
            "fase_max": float(r[3]) if r[3] is not None else None,
            "ru030_min": float(r[4]) if r[4] is not None else None,
            "ru030_max": float(r[5]) if r[5] is not None else None,
        }

        # 2) 30 filas de fprorut variadas (distintas referencias)
        cursor.execute("""
            SELECT TOP 30 ru999, ru000, ru010, ru020, ru030
            FROM admuser.fprorut
            WHERE ru000 IN (
                SELECT TOP 6 ru000 FROM admuser.fprorut
                GROUP BY ru000
                ORDER BY COUNT(*) DESC
            )
            ORDER BY ru000, ru020
        """)
        rows = cursor.fetchall()
        out["fprorut_muestra"] = {
            "columns": ["ru999", "ru000", "ru010", "ru020", "ru030"],
            "rows": [[str(v) if v is not None else None for v in r] for r in rows],
        }

        # 3) fprorut_edm: 30 filas variadas
        cursor.execute("""
            SELECT TOP 30 ru999, ru000, ru010, ru020, ru030, ru040, ru050,
                          ru060, ru070, ru080, ru090, ru100, ru110, ru120
            FROM admuser.fprorut_edm
            WHERE ru000 IN (
                SELECT TOP 6 ru000 FROM admuser.fprorut_edm
                GROUP BY ru000
                ORDER BY COUNT(*) DESC
            )
            ORDER BY ru000, ru020
        """)
        rows = cursor.fetchall()
        out["fprorut_edm_muestra"] = {
            "columns": ["ru999", "ru000", "ru010", "ru020", "ru030", "ru040",
                        "ru050", "ru060", "ru070", "ru080", "ru090", "ru100",
                        "ru110", "ru120"],
            "rows": [[str(v) if v is not None else None for v in r] for r in rows],
        }

        # 4) Todas las operaciones de fproope (solo 87)
        cursor.execute("""
            SELECT op999, op000, op010, op020, op030, op040, op050, op055,
                   op060, op090, op100, op110, op120
            FROM admuser.fproope
            ORDER BY op000
        """)
        rows = cursor.fetchall()
        out["fproope_todas"] = {
            "columns": ["op999", "op000", "op010", "op020", "op030", "op040",
                        "op050", "op055", "op060", "op090", "op100", "op110", "op120"],
            "rows": [[str(v) if v is not None else None for v in r] for r in rows],
        }

        # 5) Intento de JOIN — probar si fprorut.ru030 enlaza con fproope.op000
        cursor.execute("""
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
        """)
        rows = cursor.fetchall()
        out["join_prueba_ru030_op000"] = {
            "columns": ["referencia", "fase", "ru030", "op_id", "operacion", "ct", "tiempo"],
            "rows": [[str(v) if v is not None else None for v in r] for r in rows],
        }

        # 6) Verificar distintos valores de ru030 (por si no es siempre 0)
        cursor.execute("""
            SELECT TOP 20 ru030, COUNT(*) AS n
            FROM admuser.fprorut
            GROUP BY ru030
            ORDER BY COUNT(*) DESC
        """)
        rows = cursor.fetchall()
        out["fprorut_distribucion_ru030"] = {
            "columns": ["ru030", "n"],
            "rows": [[str(v) if v is not None else None for v in r] for r in rows],
        }

        conn.close()
        return out
    except Exception as exc:
        raise HTTPException(502, f"Error: {exc}")


@router.post("/query")
def query_readonly(
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
    database = payload.get("database") or "dbizaro"
    sql = (payload.get("sql") or "").strip()
    max_rows = int(payload.get("max_rows") or 500)
    max_rows = max(1, min(max_rows, 5000))

    if not re.match(r"^[\w]+$", database):
        raise HTTPException(400, "Nombre de BBDD invalido")
    if not sql:
        raise HTTPException(400, "Falta el campo 'sql'")

    # Validar SELECT-only
    # 1) Debe empezar por SELECT o WITH (CTE)
    sql_stripped = sql.lstrip(" \t\r\n;(")
    first_word = sql_stripped.split(None, 1)[0].upper() if sql_stripped else ""
    if first_word not in {"SELECT", "WITH"}:
        raise HTTPException(400, "Solo se permiten SELECT/WITH")

    # 2) Rechazar palabras reservadas peligrosas (busqueda por palabra entera,
    #    insensible a mayus/minus)
    forbidden = [
        "INSERT", "UPDATE", "DELETE", "MERGE", "TRUNCATE", "DROP",
        "ALTER", "CREATE", "GRANT", "REVOKE", "EXEC", "EXECUTE",
        "SP_", "XP_", "BACKUP", "RESTORE", "SHUTDOWN", "BULK",
        "OPENROWSET", "OPENQUERY",
    ]
    # Dividir en tokens tipo palabra para evitar falsos positivos
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sql.upper())
    for bad in forbidden:
        if bad in tokens:
            raise HTTPException(400, f"Palabra prohibida: {bad}")
        # SP_ / XP_ son prefijos, comprobar cualquier token que empiece asi
        if bad.endswith("_"):
            for t in tokens:
                if t.startswith(bad):
                    raise HTTPException(400, f"Prefijo prohibido: {bad}")

    # 3) Rechazar punto y coma internos (multi-statement)
    #    Permitimos un ';' final, pero no en medio
    stripped = sql.rstrip().rstrip(";").strip()
    if ";" in stripped:
        raise HTTPException(400, "No se permiten multiples sentencias")

    try:
        conn = _connect_db(database)
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = []
        for row in cursor.fetchmany(max_rows):
            rows.append([
                str(v) if v is not None else None
                for v in row
            ])
        truncated = len(rows) >= max_rows
        conn.close()
        return {
            "database": database,
            "columns": columns,
            "rows": rows,
            "n_rows": len(rows),
            "truncated": truncated,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Error: {exc}")


@router.get("/preview")
def preview_data(
    database: str = Query(...),
    schema: str = Query("dbo"),
    table: str = Query(...),
    limit: int = Query(50, ge=1, le=500),
):
    """Preview de las primeras filas de una tabla."""
    # Sanitize identifiers to prevent SQL injection
    import re
    if not re.match(r'^[\w]+$', database) or not re.match(r'^[\w]+$', schema) or not re.match(r'^[\w]+$', table):
        raise HTTPException(400, "Nombre invalido")

    try:
        conn = _connect_db(database)
        cursor = conn.cursor()
        cursor.execute(f"SELECT TOP {limit} * FROM [{schema}].[{table}]")
        columns = [desc[0] for desc in cursor.description]
        rows = []
        for row in cursor.fetchall():
            rows.append([
                str(v) if v is not None else None
                for v in row
            ])
        conn.close()
        return {"columns": columns, "rows": rows, "total": len(rows)}
    except Exception as exc:
        raise HTTPException(502, f"Error: {exc}")
