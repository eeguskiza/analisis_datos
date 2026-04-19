"""FastAPI application factory."""
from __future__ import annotations

import logging
import traceback
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.config import settings
from api.database import init_db
from api.middleware.auth import AuthMiddleware
from api.rate_limit import limiter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oee")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa la BBDD al arrancar."""
    try:
        init_db()
        logger.info("Base de datos inicializada OK")
    except Exception as exc:
        logger.error(f"Error inicializando BD: {exc}")
        logger.error(traceback.format_exc())
    yield


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


# ── Rate limit global (slowapi) ──────────────────────────────────────────────
# El ``limiter`` es una instancia compartida (definida en ``api.rate_limit``)
# que los routers importan para decorar endpoints con ``@limiter.limit(...)``.
# La referencia en ``app.state.limiter`` la usa slowapi internamente para
# construir el ``429`` handler.

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Middlewares ──────────────────────────────────────────────────────────────
# ORDEN LIFO — recordatorio para Plan 02-04 (AuditMiddleware):
#
#     app.add_middleware(AuditMiddleware)   # se anade en 02-04 (registra ultimo)
#     app.add_middleware(AuthMiddleware)    # ← registrado aqui (ejecuta primero)
#
# Starlette ejecuta en orden inverso al de registro: el ultimo ``add_middleware``
# es el primero en procesar la request. AuthMiddleware DEBE ejecutar antes que
# AuditMiddleware para poblar ``request.state.user`` (research §Pitfall 1).

app.add_middleware(AuthMiddleware)


# ── Static files ──────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(settings.project_root / "static")), name="static")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(settings.project_root / "static" / "img" / "brand" / "nexo" / "logo.png")

# ── Routers ───────────────────────────────────────────────────────────────────

from api.routers import pages, conexion, recursos, pipeline, informes, ciclos, health, historial, email, operarios, centro_mando, bbdd, datos, luk4, capacidad  # noqa: E402

app.include_router(health.router, prefix="/api")
app.include_router(centro_mando.router, prefix="/api")
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
