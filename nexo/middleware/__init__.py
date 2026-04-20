"""nexo.middleware package — middlewares custom de Nexo (Phase 4+).

En Phase 2/3 los middlewares viven en ``api/middleware/``. Desde Phase 4
se introduce este paquete paralelo para middlewares cuya lógica se
apoya en ``nexo/services/*`` puro (observabilidad, preflight/postflight).

Actualmente expone:

- :class:`nexo.middleware.query_timing.QueryTimingMiddleware`: postflight
  que escribe una fila en ``nexo.query_log`` por request a los 4
  endpoints con preflight (pipeline/run, bbdd/query, capacidad, operarios).
"""
