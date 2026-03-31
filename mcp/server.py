"""
MCP Server para OEE Planta.

Expone herramientas que actuan como proxy al backend FastAPI,
permitiendo a Claude consultar datos, lanzar informes y gestionar la config.
"""
from __future__ import annotations

import json
import os

import httpx
from mcp.server import Server
from mcp.server.stdio import run_server
from mcp.types import TextContent, Tool

API_BASE = os.environ.get("OEE_API_URL", "http://web:8000")

app = Server("oee-planta")
client = httpx.Client(base_url=API_BASE, timeout=120)


# ── Tools ─────────────────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_health",
            description="Estado de todos los servicios de la plataforma OEE (web, db local, bd MES)",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_reports",
            description="Lista todos los informes PDF generados, organizados por fecha/seccion/maquina",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_executions",
            description="Lista las ultimas ejecuciones del pipeline con su estado y numero de PDFs",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Numero maximo de resultados", "default": 20},
                },
            },
        ),
        Tool(
            name="get_execution_detail",
            description="Obtiene el detalle de una ejecucion: log completo y lista de PDFs generados",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "ID de la ejecucion"},
                },
                "required": ["id"],
            },
        ),
        Tool(
            name="generate_report",
            description="Ejecuta el pipeline OEE: extrae datos y genera informes PDF. Usa source='csv_only' si no hay conexion a BD MES.",
            inputSchema={
                "type": "object",
                "properties": {
                    "fecha_inicio": {"type": "string", "description": "Fecha inicio YYYY-MM-DD"},
                    "fecha_fin": {"type": "string", "description": "Fecha fin YYYY-MM-DD"},
                    "source": {"type": "string", "enum": ["db", "excel", "csv_only"], "default": "db"},
                    "modulos": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["disponibilidad", "rendimiento", "calidad", "oee_secciones"]},
                        "description": "Modulos a ejecutar (default: todos)",
                    },
                },
                "required": ["fecha_inicio", "fecha_fin"],
            },
        ),
        Tool(
            name="get_ciclos",
            description="Obtiene los tiempos de ciclo ideales (piezas/hora) agrupados por seccion y maquina",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="update_ciclo",
            description="Actualiza el tiempo de ciclo de una maquina/referencia",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "ID del ciclo a actualizar"},
                    "maquina": {"type": "string"},
                    "referencia": {"type": "string"},
                    "tiempo_ciclo": {"type": "number", "description": "Piezas por hora"},
                },
                "required": ["id", "maquina", "referencia", "tiempo_ciclo"],
            },
        ),
        Tool(
            name="get_connection_status",
            description="Comprueba si la conexion al SQL Server MES (dbizaro) esta activa",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        result = _dispatch(name, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]
    except Exception as exc:
        return [TextContent(type="text", text=f"Error: {exc}")]


def _dispatch(name: str, args: dict) -> dict:
    if name == "get_health":
        return client.get("/api/health").json()

    elif name == "list_reports":
        return client.get("/api/informes").json()

    elif name == "list_executions":
        limit = args.get("limit", 20)
        return client.get(f"/api/historial?limit={limit}").json()

    elif name == "get_execution_detail":
        return client.get(f"/api/historial/{args['id']}").json()

    elif name == "generate_report":
        body = {
            "fecha_inicio": args["fecha_inicio"],
            "fecha_fin": args["fecha_fin"],
            "source": args.get("source", "db"),
            "modulos": args.get("modulos", ["disponibilidad", "rendimiento", "calidad", "oee_secciones"]),
        }
        # The pipeline uses SSE, so we collect all events
        lines = []
        with client.stream("POST", "/api/pipeline/run", json=body) as resp:
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    lines.append(line[6:])
        return {"log": lines}

    elif name == "get_ciclos":
        return client.get("/api/ciclos").json()

    elif name == "update_ciclo":
        body = {"maquina": args["maquina"], "referencia": args["referencia"], "tiempo_ciclo": args["tiempo_ciclo"]}
        return client.put(f"/api/ciclos/row/{args['id']}", json=body).json()

    elif name == "get_connection_status":
        return client.get("/api/conexion/status").json()

    else:
        return {"error": f"Tool desconocido: {name}"}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_server(app))
