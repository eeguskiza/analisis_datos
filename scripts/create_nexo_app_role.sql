-- Creacion del rol 'nexo_app' (Plan 02-04 — gate IDENT-06).
--
-- Proposito: separar el rol que usa la app en runtime del owner del
-- schema para que UPDATE/DELETE sobre nexo.audit_log falle con
-- "permission denied" y la tabla de auditoria sea realmente append-only.
--
-- Se ejecuta desde el owner del schema (oee). La password del nuevo rol
-- la fija el operador — este script la lee de la variable psql
-- ``:'nexo_app_password'``. Ejecucion tipica:
--
--    docker compose exec -T db psql -U oee -d oee_planta \
--        -v nexo_app_password="<password generado>" \
--        -f /app/scripts/create_nexo_app_role.sql
--
-- El script es idempotente: si el rol ya existe, solo actualiza la
-- password (para rotaciones) y reaplica los GRANT/REVOKE.

\set ON_ERROR_STOP on

-- 1. Crear el rol (o actualizar password si ya existe).
--    psql NO interpola ``:'var'`` dentro de bloques ``DO $$ ... $$`` (son
--    string literales dollar-quoted para el servidor). Por eso construimos
--    la DDL via CASE + format() y la ejecutamos con ``\gexec`` — asi la
--    interpolacion ocurre a nivel de cliente psql antes de enviarla.

SELECT CASE
    WHEN EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nexo_app')
        THEN format('ALTER ROLE nexo_app PASSWORD %L', :'nexo_app_password')
    ELSE format('CREATE ROLE nexo_app LOGIN PASSWORD %L', :'nexo_app_password')
END AS ddl \gset

\echo '→' :'ddl'
SELECT :'ddl' \gexec

BEGIN;

-- 2. Acceso al schema.
GRANT USAGE ON SCHEMA nexo TO nexo_app;

-- 3. CRUD basico en todas las tablas del schema (de momento).
GRANT SELECT, INSERT, UPDATE, DELETE
    ON ALL TABLES IN SCHEMA nexo
    TO nexo_app;

-- 4. Sequences para PKs autoincrementales.
GRANT USAGE, SELECT, UPDATE
    ON ALL SEQUENCES IN SCHEMA nexo
    TO nexo_app;

-- 5. Default privileges para futuras tablas (p.ej. si un plan posterior
--    anade una nueva, hereda los mismos privilegios sin tener que
--    re-correr este script).
ALTER DEFAULT PRIVILEGES IN SCHEMA nexo
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO nexo_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA nexo
    GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO nexo_app;

-- 6. ── CORE DEL GATE IDENT-06 ───────────────────────────────────────
--    Revocar UPDATE/DELETE especificamente en nexo.audit_log. Asi la
--    app puede INSERT nuevas filas (lo necesita el AuditMiddleware)
--    pero NO puede borrarlas ni modificarlas desde runtime, haciendo
--    la tabla append-only a nivel de BD.
--
--    Nota: el owner (oee) sigue pudiendo UPDATE/DELETE porque el
--    schema/tabla le pertenece — los privilegios owner son implicitos
--    y no se revocan via GRANT/REVOKE. Esto es aceptable: tareas de
--    mantenimiento (rotacion de logs, GDPR request, etc.) se hacen
--    conectando explicitamente como owner, no desde la app.

REVOKE UPDATE, DELETE ON nexo.audit_log FROM nexo_app;

COMMIT;

-- 7. Verificacion visible en la salida del script.
\echo '────────────────────────────────────────────────────────────'
\echo 'Privilegios finales de nexo_app sobre nexo.audit_log:'
SELECT grantee, privilege_type
FROM information_schema.table_privileges
WHERE grantee = 'nexo_app' AND table_schema = 'nexo' AND table_name = 'audit_log'
ORDER BY privilege_type;
\echo '(esperado: solo INSERT y SELECT; si aparecen UPDATE/DELETE, el REVOKE fallo)'
\echo '────────────────────────────────────────────────────────────'
