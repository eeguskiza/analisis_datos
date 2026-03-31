"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.config import settings
from api.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa la BBDD al arrancar."""
    init_db()
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="OEE Planta - Informes",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url=None,
    lifespan=lifespan,
)

# ── Static files ──────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(settings.project_root / "static")), name="static")

# ── Routers ───────────────────────────────────────────────────────────────────

from api.routers import pages, conexion, recursos, pipeline, informes, ciclos, health  # noqa: E402

app.include_router(health.router, prefix="/api")
app.include_router(pages.router)
app.include_router(conexion.router, prefix="/api")
app.include_router(recursos.router, prefix="/api")
app.include_router(pipeline.router, prefix="/api")
app.include_router(informes.router, prefix="/api")
app.include_router(ciclos.router, prefix="/api")
