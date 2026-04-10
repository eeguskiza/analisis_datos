"""Explorador de bases de datos del SQL Server."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/bbdd", tags=["bbdd"])


def _get_conn_string(database: str | None = None):
    """Construye connection string usando la config guardada."""
    from OEE.db.connector import load_config, detectar_driver
    cfg = load_config()
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
