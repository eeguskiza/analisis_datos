"""FastAPI application factory."""
from __future__ import annotations

import asyncio
import logging
import threading
import traceback
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exception_handlers import (
    http_exception_handler as _default_http_handler,
)
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.config import settings
from api.database import init_db
from api.middleware.audit import AuditMiddleware
from api.middleware.auth import AuthMiddleware
from api.rate_limit import limiter
from nexo.data import schema_guard
from nexo.data.engines import engine_nexo
from nexo.logging_config import configure_logging
from nexo.middleware.flash import FlashMiddleware
from nexo.middleware.query_timing import QueryTimingMiddleware
from nexo.services import thresholds_cache
from nexo.services.cleanup_scheduler import cleanup_loop

# Configurar logging ANTES de cualquier getLogger para que los handlers
# esten en su sitio cuando los modulos importados empiecen a emitir.
configure_logging(level=logging.DEBUG if settings.debug else logging.INFO)
logger = logging.getLogger("nexo.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa la BBDD al arrancar.

    1. ``schema_guard.verify`` ANTES de ``init_db``: si falla aquí,
       abortamos claro — el ``RuntimeError`` sube hasta uvicorn y la app
       NO arranca. D-06 / D-07 del plan 03-01.
    2. ``init_db()`` conserva su try/except legacy — el bootstrap de
       ``ecs_mobility`` es best-effort históricamente (los schemas ya
       existen en SQL Server, sólo carga CSVs idempotentes).
    """
    # 1. Schema guard (Phase 3, plan 03-01) — aborta arranque si faltan
    #    tablas en nexo.* y NEXO_AUTO_MIGRATE != true.
    schema_guard.verify(engine_nexo)

    # 2. init_db legacy (bootstrap ecs_mobility) — mantener best-effort.
    try:
        init_db()
        logger.info("Base de datos inicializada OK")
    except Exception as exc:
        logger.error(f"Error inicializando BD: {exc}")
        logger.error(traceback.format_exc())

    # 3. Plan 04-04: hidratar cache de thresholds antes de cualquier
    #    request. Los routers de preflight (pipeline/bbdd/capacidad/
    #    operarios) hacen thresholds_cache.get(...) desde el primer hit;
    #    sin full_reload inicial caeria al safety-net 5min en cada worker.
    try:
        thresholds_cache.full_reload()
    except Exception:
        # No tumbamos el arranque: el safety-net de get() hara full_reload
        # on-demand en la primera lectura.
        logger.exception(
            "thresholds_cache full_reload inicial fallido (se reintentara on-demand)",
        )

    # 4. Plan 04-04: listener LISTEN/NOTIFY (D-19 completo). Thread
    #    dedicado wrappeado en asyncio.to_thread; graceful shutdown via
    #    stop_event.set() + task.cancel() en el finally.
    listener_stop_event = threading.Event()
    listener_task = asyncio.create_task(
        thresholds_cache.start_listener(listener_stop_event),
    )
    logger.info("thresholds_cache listener task started")

    # 5. Plan 04-03: scheduler asyncio para cleanup jobs.
    #    Plan 04-03 registra approvals_cleanup (Mon 03:05).
    #    Plan 04-04 añade query_log_cleanup (Mon 03:00) y
    #    factor_auto_refresh (1er Mon del mes 03:10) al mismo loop.
    cleanup_task = asyncio.create_task(cleanup_loop())
    logger.info("cleanup_scheduler task started")

    try:
        yield
    finally:
        # Orden de shutdown (Plan 04-04):
        # 1. stop_event.set() termina el while interno del worker LISTEN.
        # 2. listener_task.cancel() termina el wrapper to_thread.
        # 3. cleanup_task.cancel() termina el scheduler.
        listener_stop_event.set()
        listener_task.cancel()
        try:
            await listener_task
        except (asyncio.CancelledError, Exception):
            pass
        logger.info("thresholds_cache listener task cancelled")

        cleanup_task.cancel()
        try:
            await cleanup_task
        except (asyncio.CancelledError, Exception):
            pass
        logger.info("cleanup_scheduler task cancelled")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    description=f"Plataforma interna de {settings.company_name} — Nexo (Mark-III).",
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url=None,
    lifespan=lifespan,
)


# ── Error handler — loguea server-side, devuelve solo un UUID al cliente ─────
# NO incluye traceback ni detalles de la excepcion en el body. Fuga de
# informacion evitada (Sprint 0 commit 8 / NAMING-07).

def _wants_json(request: Request) -> bool:
    """Devuelve True si la request espera JSON (endpoints /api/*, Accept
    header con application/json, o htmx request). El resto recibe HTML."""
    if request.url.path.startswith("/api/"):
        return True
    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return True
    if request.headers.get("hx-request") == "true":
        return True
    return False


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_id = str(uuid.uuid4())
    logger.exception(
        "Unhandled exception %s at %s %s", error_id, request.method, request.url.path
    )

    if _wants_json(request):
        return JSONResponse(
            status_code=500,
            content={"error_id": error_id, "message": "Internal error"},
        )

    return HTMLResponse(
        status_code=500,
        content=(
            "<!DOCTYPE html>"
            "<html><head><title>Error</title>"
            "<style>body{font-family:system-ui,sans-serif;padding:3em;"
            "background:#0f2236;color:#e0e0e0;text-align:center}"
            "code{background:#1a3a5c;padding:.2em .5em;border-radius:4px;"
            "font-size:12px}a{color:#5995ff}</style></head>"
            "<body><h1>Error interno</h1>"
            "<p>Algo ha fallado procesando tu peticion. "
            "Contacta con el equipo tecnico incluyendo este identificador.</p>"
            f"<p>Error ID: <code>{error_id}</code></p>"
            '<p><a href="/">Volver al dashboard</a></p></body></html>'
        ),
    )


# ── Handler 403 HTML (D-07) — Accept-aware redirect + flash ──────────────────
# Sibling del ``global_exception_handler`` de NAMING-07. El handler
# ``@app.exception_handler(Exception)`` de arriba NO captura
# ``HTTPException`` (FastAPI tiene handler default separado). Registrar
# sobre ``StarletteHTTPException`` (padre de ``fastapi.HTTPException``)
# para capturar ambas ramas (Pitfall 1 de 05-RESEARCH).
#
# Comportamiento:
#   - status != 403 → delegar al default de FastAPI (no regresiona 401/404/422).
#   - status == 403 + cliente JSON (/api/*, Accept JSON, HX-Request) →
#     delegar (contract JSON estable).
#   - status == 403 + HTML → RedirectResponse("/", 302) + cookie `nexo_flash`
#     con mensaje user-friendly (leída y borrada por FlashMiddleware en el
#     siguiente request).

_PERMISSION_LABELS: dict[str, str] = {
    "bbdd:read":           "el explorador de BBDD",
    "pipeline:read":       "analisis",
    "pipeline:run":        "ejecutar el pipeline",
    "historial:read":      "el historial",
    "capacidad:read":      "capacidad",
    "recursos:read":       "recursos",
    "recursos:edit":       "editar recursos",
    "ciclos:read":         "calculo de ciclos",
    "ciclos:edit":         "editar ciclos",
    "operarios:read":      "operarios",
    "datos:read":          "datos",
    "ajustes:manage":      "la configuracion",
    "auditoria:read":      "el log de auditoria",
    "usuarios:manage":     "la gestion de usuarios",
    "aprobaciones:manage": "solicitudes de aprobacion",
    "limites:manage":      "limites de queries",
    "rendimiento:read":    "metricas de rendimiento",
    "conexion:config":     "la configuracion de conexion",
    "conexion:read":       "la conexion",
    "informes:read":       "informes",
    "informes:delete":     "borrar informes",
}


def _friendly_permission_label(perm: str) -> str:
    """Traduce ``modulo:accion`` a texto user-friendly para el toast.

    Fallback: si el permiso no está en ``_PERMISSION_LABELS``, devuelve la
    cadena raw — mejor mostrar "pipeline:run" que un placeholder vacío.
    """
    return _PERMISSION_LABELS.get(perm, perm)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler_403(
    request: Request, exc: StarletteHTTPException
):
    """Negocia Accept para 403: HTML → redirect + flash; resto → default.

    Para cualquier otro status code (401, 404, 422) delega al handler
    default de FastAPI — preserva el contract JSON existente sin
    regresion. El handler ``@app.exception_handler(Exception)`` de 500
    (NAMING-07) queda intacto: actua sobre ``Exception``, no sobre
    ``HTTPException``.
    """
    if exc.status_code != 403:
        return await _default_http_handler(request, exc)

    if _wants_json(request):
        return await _default_http_handler(request, exc)

    # HTML path: redirect + flash toast.
    raw = str(exc.detail or "")
    perm = raw.replace("Permiso requerido: ", "") if raw.startswith(
        "Permiso requerido: "
    ) else ""
    friendly = _friendly_permission_label(perm) if perm else "esta seccion"
    response = RedirectResponse("/", status_code=302)
    response.set_cookie(
        "nexo_flash",
        f"No tienes permiso para acceder a {friendly}",
        max_age=60,
        httponly=True,
        secure=not settings.debug,   # Pitfall 2: dev HTTP local
        samesite="lax",
        path="/",
    )
    return response


# ── Rate limit global (slowapi) ──────────────────────────────────────────────
# El ``limiter`` es una instancia compartida (definida en ``api.rate_limit``)
# que los routers importan para decorar endpoints con ``@limiter.limit(...)``.
# La referencia en ``app.state.limiter`` la usa slowapi internamente para
# construir el ``429`` handler.

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Middlewares ──────────────────────────────────────────────────────────────
# ORDEN LIFO (research §Pitfall 1):
#
# Starlette ejecuta los middlewares en orden INVERSO al registro. El ultimo
# ``add_middleware`` es el primero en procesar la request (outer); el primero
# en registrarse es el mas cercano al handler (inner).
#
# Cadena deseada tras Phase 4 (outer → inner):
#   Request → AuthMiddleware → AuditMiddleware → QueryTimingMiddleware → handler
#                                                                       ↓
#   Response ← Auth ← Audit ← QueryTiming ← (handler)
#
# AuthMiddleware va primero (outermost): si la sesion es invalida, retorna
# 401/redirect sin llegar a Audit. AuditMiddleware va despues: lee
# request.state.user ya poblado por Auth. QueryTimingMiddleware (Phase 4)
# va innermost: mide actual_ms tan cerca del handler como sea posible, y
# lee request.state.estimated_ms que el router pobló antes de ejecutar.
#
# Registro correspondiente (primero registrado = innermost):

app.add_middleware(QueryTimingMiddleware)  # innermost — ultima capa antes del handler
app.add_middleware(AuditMiddleware)        # inner — segundo en ejecutar
app.add_middleware(FlashMiddleware)        # Plan 05-03 — entre Audit y Auth (W-02)
app.add_middleware(AuthMiddleware)         # outer — primero en ejecutar


# ── Static files ──────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(settings.project_root / "static")), name="static")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(settings.project_root / "static" / "img" / "brand" / "nexo" / "logo-tab-navegador.png")

# ── Routers ───────────────────────────────────────────────────────────────────

from api.routers import auth as auth_router, auditoria as auditoria_router, pages, conexion, recursos, pipeline, informes, ciclos, health, historial, email, operarios, centro_mando, bbdd, datos, luk4, capacidad, usuarios as usuarios_router, approvals as approvals_router, limites as limites_router, rendimiento as rendimiento_router  # noqa: E402

app.include_router(health.router, prefix="/api")
app.include_router(centro_mando.router, prefix="/api")
app.include_router(auth_router.router)  # /login, /logout, /cambiar-password
app.include_router(usuarios_router.router)  # /ajustes/usuarios/*
app.include_router(auditoria_router.router)  # /ajustes/auditoria + /export
app.include_router(pages.router)
app.include_router(conexion.router, prefix="/api")
app.include_router(recursos.router, prefix="/api")
app.include_router(pipeline.router, prefix="/api")
app.include_router(informes.router, prefix="/api")
app.include_router(ciclos.router, prefix="/api")
app.include_router(historial.router, prefix="/api")
app.include_router(email.router, prefix="/api")
app.include_router(operarios.router, prefix="/api")
app.include_router(bbdd.router, prefix="/api")
app.include_router(datos.router, prefix="/api")
app.include_router(luk4.router, prefix="/api")
app.include_router(capacidad.router, prefix="/api")
app.include_router(approvals_router.router)  # Plan 04-03: /api/approvals/*, /ajustes/solicitudes, /mis-solicitudes
app.include_router(limites_router.router)  # Plan 04-04: /ajustes/limites + /api/thresholds/*
app.include_router(rendimiento_router.router)  # Plan 04-04: /ajustes/rendimiento + /api/rendimiento/*
