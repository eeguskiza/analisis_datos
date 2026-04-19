-- NOTA T3: el filtro de turno 3 cruza medianoche (T3 empieza a las 22:00
-- del dia N y termina a las 06:00 del dia N+1). La logica real esta en
-- OEE/db/connector.py:extraer_datos — este archivo es un placeholder
-- para DATA-05 (un .sql por metodo repo) mientras D-04 mantiene la query
-- canonica en el connector.
--
-- NO ejecutar este archivo directamente; MesRepository.extraer_datos_produccion
-- delega en OEE.db.connector.extraer_datos (wrapper delgado, D-04).
SELECT 'placeholder — ver OEE/db/connector.extraer_datos' AS note;
