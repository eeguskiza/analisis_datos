"""Crea la base de datos oee_ecs y sus tablas en SQL Server."""
import pyodbc

CONN_STR = (
    "DRIVER={{ODBC Driver 18 for SQL Server}};"
    "SERVER=192.168.0.4,1433;"
    "DATABASE={db};"
    "UID=sa;"
    "PWD=AdmS1552+;"
    "TrustServerCertificate=yes;"
    "Encrypt=yes;"
)

TABLES = [
    ("ejecuciones", """
        CREATE TABLE ejecuciones (
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
    ("datos_produccion", """
        CREATE TABLE datos_produccion (
            id              INT IDENTITY(1,1) PRIMARY KEY,
            ejecucion_id    INT NOT NULL REFERENCES ejecuciones(id),
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
    ("metricas_oee", """
        CREATE TABLE metricas_oee (
            id                      INT IDENTITY(1,1) PRIMARY KEY,
            ejecucion_id            INT NOT NULL REFERENCES ejecuciones(id),
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
    ("ciclos", """
        CREATE TABLE ciclos (
            id              INT IDENTITY(1,1) PRIMARY KEY,
            maquina         NVARCHAR(50) NOT NULL,
            referencia      NVARCHAR(50) NOT NULL,
            tiempo_ciclo    FLOAT DEFAULT 0,
            CONSTRAINT uq_ciclos UNIQUE (maquina, referencia)
        )
    """),
    ("referencias_stats", """
        CREATE TABLE referencias_stats (
            id              INT IDENTITY(1,1) PRIMARY KEY,
            ejecucion_id    INT NOT NULL REFERENCES ejecuciones(id),
            recurso         NVARCHAR(50),
            referencia      NVARCHAR(50),
            ciclo_ideal     FLOAT,
            ciclo_real      FLOAT,
            cantidad        INT DEFAULT 0,
            horas           FLOAT DEFAULT 0
        )
    """),
    ("incidencias_resumen", """
        CREATE TABLE incidencias_resumen (
            id              INT IDENTITY(1,1) PRIMARY KEY,
            ejecucion_id    INT NOT NULL REFERENCES ejecuciones(id),
            recurso         NVARCHAR(50),
            nombre          NVARCHAR(200),
            tipo            NVARCHAR(20),
            horas           FLOAT DEFAULT 0
        )
    """),
]

INDEXES = [
    "CREATE INDEX ix_datos_ejec ON datos_produccion(ejecucion_id)",
    "CREATE INDEX ix_datos_recurso ON datos_produccion(recurso, fecha)",
    "CREATE INDEX ix_metricas_ejec ON metricas_oee(ejecucion_id)",
    "CREATE INDEX ix_metricas_recurso ON metricas_oee(recurso, fecha)",
    "CREATE INDEX ix_metricas_seccion ON metricas_oee(seccion)",
    "CREATE INDEX ix_refs_ejec ON referencias_stats(ejecucion_id)",
    "CREATE INDEX ix_inc_ejec ON incidencias_resumen(ejecucion_id)",
]


def main():
    # Crear BD
    conn = pyodbc.connect(CONN_STR.format(db="master"), timeout=10, autocommit=True)
    cursor = conn.cursor()
    cursor.execute(
        "IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'oee_ecs') "
        "CREATE DATABASE oee_ecs"
    )
    print("Base de datos oee_ecs: OK")
    conn.close()

    # Crear tablas
    conn = pyodbc.connect(CONN_STR.format(db="oee_ecs"), timeout=10, autocommit=True)
    cursor = conn.cursor()

    for name, ddl in TABLES:
        cursor.execute(
            f"IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = '{name}') "
            + ddl
        )
        print(f"  Tabla {name}: OK")

    for idx_sql in INDEXES:
        idx_name = idx_sql.split("INDEX ")[1].split(" ON")[0]
        cursor.execute(
            f"IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = '{idx_name}') "
            + idx_sql
        )
    print("  Indices: OK")

    # Verificar
    print("\n=== TABLAS EN oee_ecs ===")
    cursor.execute("SELECT name FROM sys.tables ORDER BY name")
    for row in cursor.fetchall():
        print(f"  {row[0]}")
    conn.close()

    # Listar BDs
    conn = pyodbc.connect(CONN_STR.format(db="master"), timeout=10)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sys.databases ORDER BY name")
    print("\n=== TODAS LAS BBDD ===")
    for row in cursor.fetchall():
        print(f"  {row[0]}")
    conn.close()

    print("\nTodo creado correctamente.")


if __name__ == "__main__":
    main()
