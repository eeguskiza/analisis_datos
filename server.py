#!/usr/bin/env python3
"""Punto de entrada para arrancar la aplicación OEE."""
import uvicorn

from api.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
