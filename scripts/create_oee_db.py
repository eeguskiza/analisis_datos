"""Crea la base de datos ecs_mobility con esquemas cfg/oee/luk4."""
import sys
from pathlib import Path

import pyodbc

# Cargar configuracion del .env (misma que usa la app)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.config import settings

def _conn_str(db: str) -> str:
    return (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={settings.db_server},{settings.db_port};"
        f"DATABASE={db};"
        f"UID={settings.db_user};"
        f"PWD={settings.db_password};"
        "TrustServerCertificate=yes;"
        "Encrypt=yes;"
    )

SCHEMAS = ["cfg", "oee", "luk4"]

TABLES = [
    # ── cfg ──────────────────────────────────────────────────────────
    ("cfg.ciclos", """
        CREATE TABLE cfg.ciclos (
            id              INT IDENTITY(1,1) PRIMARY KEY,
            maquina         NVARCHAR(50) NOT NULL,
            referencia      NVARCHAR(50) NOT NULL,
            tiempo_ciclo    FLOAT DEFAULT 0,
            updated_at      DATETIME2 DEFAULT GETDATE(),
            CONSTRAINT uq_ciclos UNIQUE (maquina, referencia)
        )
    """),
    ("cfg.recursos", """
        CREATE TABLE cfg.recursos (
            id              INT IDENTITY(1,1) PRIMARY KEY,
            centro_trabajo  INT NOT NULL,
            nombre          NVARCHAR(50) NOT NULL UNIQUE,
            seccion         NVARCHAR(50) DEFAULT 'GENERAL',
            activo          BIT DEFAULT 1
        )
    """),
    ("cfg.contactos", """
        CREATE TABLE cfg.contactos (
            id              INT IDENTITY(1,1) PRIMARY KEY,
            nombre          NVARCHAR(100) NOT NULL,
            email           NVARCHAR(200) NOT NULL UNIQUE
        )
    """),

    # ── oee (ejecuciones primero por FKs) ────────────────────────────
    ("oee.ejecuciones", """
        CREATE TABLE oee.ejecuciones (
            id              INT IDENTITY(1,1) PRIMARY KEY,
            fecha_inicio    DATE NOT NULL,
            fecha_fin       DATE NOT NULL,
            source          NVARCHAR(20) DEFAULT 'db',
            status          NVARCHAR(20) DEFAULT 'running',
            modulos         NVARCHAR(500),
            log             NVARCHAR(MAX),
            n_pdfs          INT DEFAULT 0,
            created_at      DATETIME2 DEFAULT GETDATE()
        )
    """),
    ("oee.datos", """
        CREATE TABLE oee.datos (
            id              INT IDENTITY(1,1) PRIMARY KEY,
            ejecucion_id    INT NOT NULL REFERENCES oee.ejecuciones(id),
            recurso         NVARCHAR(50),
            seccion         NVARCHAR(50),
            fecha           DATE,
            h_ini           NVARCHAR(10),
            h_fin           NVARCHAR(10),
            tiempo          FLOAT DEFAULT 0,
            proceso         NVARCHAR(30),
            incidencia      NVARCHAR(200),
            cantidad        FLOAT DEFAULT 0,
            malas           FLOAT DEFAULT 0,
            recuperadas     FLOAT DEFAULT 0,
            referencia      NVARCHAR(50)
        )
    """),
    ("oee.metricas", """
        CREATE TABLE oee.metricas (
            id                      INT IDENTITY(1,1) PRIMARY KEY,
            ejecucion_id            INT NOT NULL REFERENCES oee.ejecuciones(id),
            seccion                 NVARCHAR(50),
            recurso                 NVARCHAR(50),
            fecha                   DATE,
            turno                   NVARCHAR(5),
            horas_brutas            FLOAT DEFAULT 0,
            horas_disponible        FLOAT DEFAULT 0,
            horas_operativo         FLOAT DEFAULT 0,
            horas_preparacion       FLOAT DEFAULT 0,
            horas_indisponibilidad  FLOAT DEFAULT 0,
            horas_paros             FLOAT DEFAULT 0,
            tiempo_ideal            FLOAT DEFAULT 0,
            perdidas_rend           FLOAT DEFAULT 0,
            piezas_totales          INT DEFAULT 0,
            piezas_malas            INT DEFAULT 0,
            piezas_recuperadas      INT DEFAULT 0,
            buenas_finales          INT DEFAULT 0,
            disponibilidad_pct      FLOAT DEFAULT 0,
            rendimiento_pct         FLOAT DEFAULT 0,
            calidad_pct             FLOAT DEFAULT 0,
            oee_pct                 FLOAT DEFAULT 0
        )
    """),
    ("oee.referencias", """
        CREATE TABLE oee.referencias (
            id              INT IDENTITY(1,1) PRIMARY KEY,
            ejecucion_id    INT NOT NULL REFERENCES oee.ejecuciones(id),
            recurso         NVARCHAR(50),
            referencia      NVARCHAR(50),
            ciclo_ideal     FLOAT,
            ciclo_real      FLOAT,
            cantidad        INT DEFAULT 0,
            horas           FLOAT DEFAULT 0
        )
    """),
    ("oee.incidencias", """
        CREATE TABLE oee.incidencias (
            id              INT IDENTITY(1,1) PRIMARY KEY,
            ejecucion_id    INT NOT NULL REFERENCES oee.ejecuciones(id),
            recurso         NVARCHAR(50),
            nombre          NVARCHAR(200),
            tipo            NVARCHAR(20),
            horas           FLOAT DEFAULT 0
        )
    """),
    ("oee.informes", """
        CREATE TABLE oee.informes (
            id              INT IDENTITY(1,1) PRIMARY KEY,
            ejecucion_id    INT,
            fecha           NVARCHAR(10) NOT NULL,
            seccion         NVARCHAR(50) NOT NULL,
            maquina         NVARCHAR(50) DEFAULT '',
            modulo          NVARCHAR(50) DEFAULT '',
            pdf_path        NVARCHAR(500) NOT NULL,
            created_at      DATETIME2 DEFAULT GETDATE()
        )
    """),

    # ── luk4 ─────────────────────────────────────────────────────────
    ("luk4.estado", """
        CREATE TABLE luk4.estado (
            idcelula_estado INT IDENTITY(1,1) PRIMARY KEY,
            timestamp       DATETIME2,
            estado_global   INT DEFAULT 0,
            codigo_error    INT DEFAULT 0,
            porcentaje_manual FLOAT DEFAULT 0,
            porcentaje_auto   FLOAT DEFAULT 0,
            porcentaje_error  FLOAT DEFAULT 0
        )
    """),
    ("luk4.tiempos_ciclo", """
        CREATE TABLE luk4.tiempos_ciclo (
            idtiempos_ciclo INT IDENTITY(1,1) PRIMARY KEY,
            timestamp       DATETIME2,
            tiempo_ciclo_total    FLOAT,
            tiempo_ciclo_temple   FLOAT,
            tiempo_ciclo_revenido FLOAT,
            tiempo_ciclo_torno    FLOAT,
            contador_piezas_buenas  INT,
            contador_piezas_malas   INT,
            contador_piezas_totales INT
        )
    """),
    ("luk4.alarmas", """
        CREATE TABLE luk4.alarmas (
            codigo      INT PRIMARY KEY,
            componente  NVARCHAR(100),
            mensaje     NVARCHAR(500)
        )
    """),
    ("luk4.plano_zonas", """
        CREATE TABLE luk4.plano_zonas (
            id          NVARCHAR(50) NOT NULL,
            pabellon    NVARCHAR(10) NOT NULL CONSTRAINT DF_plano_zonas_pabellon DEFAULT 'p5',
            label       NVARCHAR(100),
            left_pct    FLOAT,
            top_pct     FLOAT,
            width_pct   FLOAT,
            height_pct  FLOAT,
            source      NVARCHAR(20) DEFAULT 'none',
            CONSTRAINT PK_plano_zonas PRIMARY KEY (pabellon, id)
        )
    """),
]

INDEXES = [
    "CREATE INDEX ix_datos_ejec ON oee.datos(ejecucion_id)",
    "CREATE INDEX ix_datos_recurso ON oee.datos(recurso, fecha)",
    "CREATE INDEX ix_metricas_ejec ON oee.metricas(ejecucion_id)",
    "CREATE INDEX ix_metricas_recurso ON oee.metricas(recurso, fecha)",
    "CREATE INDEX ix_metricas_seccion ON oee.metricas(seccion)",
    "CREATE INDEX ix_refs_ejec ON oee.referencias(ejecucion_id)",
    "CREATE INDEX ix_inc_ejec ON oee.incidencias(ejecucion_id)",
]

# Sinonimos en dbo para tunel IoT (escriben con los nombres viejos)
SYNONYMS = [
    ("luk4_estado", "luk4.estado"),
    ("luk4_tiempos_ciclo", "luk4.tiempos_ciclo"),
    ("alarmas_luk4", "luk4.alarmas"),
    ("plano_zonas", "luk4.plano_zonas"),
]


def main():
    # Crear BD
    conn = pyodbc.connect(_conn_str("master"), timeout=10, autocommit=True)
    cursor = conn.cursor()
    cursor.execute(
        "IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'ecs_mobility') "
        "CREATE DATABASE ecs_mobility"
    )
    print("Base de datos ecs_mobility: OK")
    conn.close()

    # Crear esquemas + tablas
    conn = pyodbc.connect(_conn_str(settings.db_name), timeout=10, autocommit=True)
    cursor = conn.cursor()

    for schema in SCHEMAS:
        cursor.execute(
            f"IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = '{schema}') "
            f"EXEC('CREATE SCHEMA {schema}')"
        )
        print(f"  Esquema {schema}: OK")

    for full_name, ddl in TABLES:
        cursor.execute(
            f"IF OBJECT_ID('{full_name}', 'U') IS NULL " + ddl
        )
        print(f"  Tabla {full_name}: OK")

    for idx_sql in INDEXES:
        idx_name = idx_sql.split("INDEX ")[1].split(" ON")[0]
        cursor.execute(
            f"IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = '{idx_name}') "
            + idx_sql
        )
    print("  Indices: OK")

    for syn_name, target in SYNONYMS:
        cursor.execute(
            f"IF NOT EXISTS (SELECT 1 FROM sys.synonyms WHERE name = '{syn_name}' "
            f"AND schema_id = SCHEMA_ID('dbo')) "
            f"CREATE SYNONYM dbo.{syn_name} FOR {target}"
        )
    print("  Sinonimos IoT: OK")

    # Verificar
    print("\n=== TABLAS EN ecs_mobility ===")
    cursor.execute("""
        SELECT s.name + '.' + t.name
        FROM sys.tables t JOIN sys.schemas s ON t.schema_id = s.schema_id
        WHERE s.name IN ('cfg', 'oee', 'luk4', 'calidad')
        ORDER BY s.name, t.name
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}")
    conn.close()

    print("\nTodo creado correctamente.")


if __name__ == "__main__":
    main()
