"""FastAPI application factory."""
from __future__ import annotations

import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from api.config import settings
from api.database import init_db

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
    title="ECS Mobility — Centro de Mando",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url=None,
    lifespan=lifespan,
)


# ── Error handler — muestra el error real en vez de "Internal Server Error" ──

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error(f"Error en {request.url}: {exc}\n{tb}")
    return HTMLResponse(
        content=f"""<!DOCTYPE html>
<html><head><title>Error</title>
<style>body{{font-family:monospace;padding:2em;background:#1a1a2e;color:#e0e0e0}}
pre{{background:#16213e;padding:1em;border-radius:8px;overflow-x:auto;font-size:13px}}
h1{{color:#e94560}}</style></head>
<body><h1>Error 500</h1><p>{exc}</p><pre>{tb}</pre>
<p><a href="/" style="color:#0f3460">Volver al dashboard</a></p></body></html>""",
        status_code=500,
    )


# ── Static files ──────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(settings.project_root / "static")), name="static")

# ── Routers ───────────────────────────────────────────────────────────────────

from api.routers import pages, conexion, recursos, pipeline, informes, ciclos, health, historial, email, operarios, centro_mando, bbdd, datos, luk4  # noqa: E402

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
