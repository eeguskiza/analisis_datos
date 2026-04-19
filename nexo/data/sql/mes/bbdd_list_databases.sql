-- Lista las bases de datos del servidor SQL Server (excluyendo sistema).
-- Extraido de api/routers/bbdd.py pre-refactor (plan 03-02 Task 3.6).
-- Requiere VIEW DATABASE STATE sobre el usuario del DSN (engine_mes)
-- o fallback a nombre unicamente; el JOIN con sys.master_files puede
-- devolver 0 filas de size si no hay permiso. Devuelve name/state/size_mb.
SELECT d.name, d.state_desc,
       CAST(COALESCE(SUM(f.size) * 8.0 / 1024, 0) AS DECIMAL(10,1)) AS size_mb
FROM sys.databases d
LEFT JOIN sys.master_files f ON d.database_id = f.database_id
WHERE d.name NOT IN ('master', 'tempdb', 'model', 'msdb')
GROUP BY d.name, d.state_desc
ORDER BY d.name
