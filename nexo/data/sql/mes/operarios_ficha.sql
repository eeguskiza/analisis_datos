-- Datos basicos de un operario en admuser.fmesope.
-- Usado por /operarios/{codigo} endpoint para la cabecera de la ficha.
-- El resto de agregaciones (centros, referencias, evolucion) sigue
-- inline en el router por ahora (tres queries distintas con el mismo
-- :codigo + ventana de fechas).
SELECT RTRIM(op020) AS nombre,
       CAST(op060 AS INT) AS activo,
       op055 AS dni
FROM admuser.fmesope
WHERE op010 = :codigo AND op000 = 'ALGA'
