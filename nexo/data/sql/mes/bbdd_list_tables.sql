-- Lista tablas (schema + nombre + filas estimadas) de la BBDD actual.
-- Se ejecuta en el contexto del engine (engine_mes → dbizaro). En
-- Mark-III el explorer /bbdd queda limitado a dbizaro (Mark-IV puede
-- crear engines ad-hoc para otros catalogs). Extraido de
-- api/routers/bbdd.py pre-refactor (plan 03-02 Task 3.6).
SELECT
    s.name AS schema_name,
    t.name AS table_name,
    p.rows AS row_count
FROM sys.tables t
JOIN sys.schemas s ON t.schema_id = s.schema_id
JOIN sys.partitions p ON t.object_id = p.object_id AND p.index_id IN (0, 1)
ORDER BY s.name, t.name
