-- Lista columnas de una tabla (schema/tabla parametrizados via :schema/:table)
-- con tipos, nullability y flag de PK. 2-part names implicitas: las
-- vistas INFORMATION_SCHEMA son locales al catalog. Extraido de
-- api/routers/bbdd.py pre-refactor (plan 03-02 Task 3.6).
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
WHERE c.TABLE_SCHEMA = :schema
  AND c.TABLE_NAME = :table
ORDER BY c.ORDINAL_POSITION
