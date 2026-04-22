-- Phase 8 / Plan 08-03 (UIREDO-02) — add nombre column to nexo.users.
--
-- Idempotent: safe to run multiple times. Usa ``ADD COLUMN IF NOT EXISTS``
-- (Postgres >= 9.6) y un backfill guardado con ``WHERE nombre IS NULL``
-- para no re-escribir rows ya backfilleadas.
--
-- Backfill: mapea el local-part del email con la primera letra en
-- mayuscula. Ej: ``e.eguskiza@ecsmobility.com`` -> ``E.eguskiza``.
-- El operario puede sobrescribirlo despues desde ``/ajustes/usuarios``.

ALTER TABLE nexo.users
  ADD COLUMN IF NOT EXISTS nombre VARCHAR(120);

-- Backfill solo donde nombre es NULL (idempotencia post-ejecucion).
UPDATE nexo.users
   SET nombre = (
           UPPER(LEFT(split_part(email, '@', 1), 1))
           || SUBSTRING(split_part(email, '@', 1) FROM 2)
       )
 WHERE nombre IS NULL;
