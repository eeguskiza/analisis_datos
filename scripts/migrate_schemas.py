"""Migra tablas de dbo a esquemas cfg/oee/luk4 en ecs_mobility.

Idempotente: se puede ejecutar multiples veces sin riesgo.
Crea sinonimos en dbo para las tablas luk4 (compatibilidad tunel IoT).
"""
import sys
from pathlib import Path

import pyodbc

# Cargar configuracion del .env (misma que usa la app)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.config import settings

CONN_STR = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    f"SERVER={settings.db_server},{settings.db_port};"
    f"DATABASE={settings.db_name};"
    f"UID={settings.db_user};"
    f"PWD={settings.db_password};"
    "TrustServerCertificate=yes;"
    "Encrypt=yes;"
)

# Esquemas a crear
SCHEMAS = ["cfg", "oee", "luk4"]

# Tablas a mover: (esquema_destino, nombre_actual_en_dbo, nombre_final)
# El orden importa: ejecuciones primero porque tiene FKs apuntando a ella
TRANSFERS = [
    # cfg (sin renombrar)
    ("cfg", "ciclos", "ciclos"),
    ("cfg", "recursos", "recursos"),
    ("cfg", "contactos", "contactos"),
    # oee (ejecuciones primero por FKs)
    ("oee", "ejecuciones", "ejecuciones"),
    ("oee", "datos_produccion", "datos"),
    ("oee", "metricas_oee", "metricas"),
    ("oee", "referencias_stats", "referencias"),
    ("oee", "incidencias_resumen", "incidencias"),
    ("oee", "informes_meta", "informes"),
    # luk4
    ("luk4", "luk4_estado", "estado"),
    ("luk4", "luk4_tiempos_ciclo", "tiempos_ciclo"),
    ("luk4", "alarmas_luk4", "alarmas"),
    ("luk4", "plano_zonas", "plano_zonas"),
]

# Sinonimos en dbo para compatibilidad con tunel IoT
SYNONYMS = [
    ("luk4_estado", "luk4.estado"),
    ("luk4_tiempos_ciclo", "luk4.tiempos_ciclo"),
    ("alarmas_luk4", "luk4.alarmas"),
    ("plano_zonas", "luk4.plano_zonas"),
]


def main():
    conn = pyodbc.connect(CONN_STR, timeout=10, autocommit=True)
    cursor = conn.cursor()

    # 1) Crear esquemas
    for schema in SCHEMAS:
        cursor.execute(
            f"IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = '{schema}') "
            f"EXEC('CREATE SCHEMA {schema}')"
        )
        print(f"  Esquema {schema}: OK")

    # 2) Mover tablas de dbo al esquema destino
    for schema, old_name, new_name in TRANSFERS:
        # Comprobar si la tabla existe en dbo
        cursor.execute(f"SELECT OBJECT_ID('dbo.{old_name}', 'U')")
        if cursor.fetchone()[0] is None:
            # Ya no esta en dbo (ya migrada o no existe)
            cursor.execute(f"SELECT OBJECT_ID('{schema}.{new_name}', 'U')")
            if cursor.fetchone()[0] is not None:
                print(f"  {schema}.{new_name}: ya existe, skip")
            else:
                print(f"  {old_name}: no existe en dbo ni en {schema}, skip")
            continue

        # Mover al nuevo esquema
        cursor.execute(f"ALTER SCHEMA {schema} TRANSFER dbo.{old_name}")
        print(f"  dbo.{old_name} -> {schema}.{old_name}")

        # Renombrar si hace falta
        if old_name != new_name:
            cursor.execute(f"EXEC sp_rename '{schema}.{old_name}', '{new_name}', 'OBJECT'")
            print(f"  {schema}.{old_name} -> {schema}.{new_name}")

    # 3) Crear sinonimos en dbo para tunel IoT
    for syn_name, target in SYNONYMS:
        cursor.execute(
            f"IF NOT EXISTS (SELECT 1 FROM sys.synonyms WHERE name = '{syn_name}' "
            f"AND schema_id = SCHEMA_ID('dbo')) "
            f"CREATE SYNONYM dbo.{syn_name} FOR {target}"
        )
        print(f"  Sinonimo dbo.{syn_name} -> {target}: OK")

    conn.close()
    print("\nMigracion completada.")


if __name__ == "__main__":
    main()
