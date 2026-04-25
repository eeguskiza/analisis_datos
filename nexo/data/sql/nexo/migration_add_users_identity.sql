-- Reestructura de identidad de usuarios.
--
-- Datos funcionales:
--   username  -> identificador corto para login/display
--   name      -> nombre
--   surname   -> apellidos
--   email     -> correo (columna existente)
--   password_hash -> password cifrada Argon2 (columna existente)
--
-- Idempotente y compatible con usuarios existentes: username se backfillea
-- desde el local-part del correo; name se backfillea desde nombre legacy.

ALTER TABLE nexo.users
  ADD COLUMN IF NOT EXISTS username VARCHAR(80),
  ADD COLUMN IF NOT EXISTS name VARCHAR(80),
  ADD COLUMN IF NOT EXISTS surname VARCHAR(120);

UPDATE nexo.users
SET username = lower(regexp_replace(split_part(email, '@', 1), '[^a-zA-Z0-9._-]+', '_', 'g'))
WHERE username IS NULL
  AND email IS NOT NULL;

WITH ranked AS (
  SELECT
    id,
    username,
    row_number() OVER (PARTITION BY username ORDER BY id) AS rn
  FROM nexo.users
  WHERE username IS NOT NULL
)
UPDATE nexo.users u
SET username = concat(r.username, '-', u.id)
FROM ranked r
WHERE u.id = r.id
  AND r.rn > 1;

UPDATE nexo.users
SET name = NULLIF(split_part(nombre, ' ', 1), '')
WHERE name IS NULL
  AND nombre IS NOT NULL;

UPDATE nexo.users
SET surname = NULLIF(trim(substr(nombre, length(split_part(nombre, ' ', 1)) + 1)), '')
WHERE surname IS NULL
  AND nombre IS NOT NULL
  AND position(' ' in nombre) > 0;

UPDATE nexo.users
SET nombre = NULLIF(trim(concat_ws(' ', name, surname)), '')
WHERE nombre IS NULL
  AND (name IS NOT NULL OR surname IS NOT NULL);

CREATE UNIQUE INDEX IF NOT EXISTS ux_nexo_users_username
  ON nexo.users (username)
  WHERE username IS NOT NULL;
