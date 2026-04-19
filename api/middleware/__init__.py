"""Middlewares de Nexo.

Orden de ejecucion (LIFO sobre add_middleware):
- AuthMiddleware (registrado ultimo → ejecuta primero)
- AuditMiddleware (registrado primero → ejecuta ultimo; llega en Plan 02-04)

AuthMiddleware puebla ``request.state.user`` antes de que AuditMiddleware
registre la linea de auditoria (research §Pitfall 1).
"""
