"""
MCP Server para Nexo.

Expone herramientas de consulta READ-ONLY sobre la API FastAPI
(que a su vez habla con dbizaro/SQL Server y la BD local).

Diseñado para que Claude Code pueda inspeccionar datos reales
durante el desarrollo sin modificar nada.
"""
from __future__ import annotations

import json
import os

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# API base: en Docker apunta a "web:8000"; en local a 127.0.0.1:8000.
# NEXO_API_URL es el nombre canonico; OEE_API_URL se mantiene como compat durante Mark-III.
API_BASE = os.environ.get("NEXO_API_URL") or os.environ.get("OEE_API_URL", "http://127.0.0.1:8000")

app = Server("nexo-mcp")
client = httpx.Client(base_url=API_BASE, timeout=120)


# ── Tools ─────────────────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        # ── Plataforma / estado ────────────────────────────────────────
        Tool(
            name="get_health",
            description="Estado de todos los servicios (web, db local, dbizaro MES)",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_connection_status",
            description="Comprueba conexion al SQL Server MES (dbizaro)",
            inputSchema={"type": "object", "properties": {}},
        ),

        # ── Explorador de BBDD (Bizaro / otros) ─────────────────────────
        Tool(
            name="bizaro_list_databases",
            description="Lista las BBDD disponibles en el SQL Server",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="bizaro_list_tables",
            description="Lista tablas (schema + nombre + nº filas) de una BBDD",
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {"type": "string", "default": "dbizaro"},
                },
            },
        ),
        Tool(
            name="bizaro_columns",
            description="Lista columnas (nombre, tipo, nullable, pk) de una tabla",
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {"type": "string", "default": "dbizaro"},
                    "schema": {"type": "string", "default": "admuser"},
                    "table": {"type": "string"},
                },
                "required": ["table"],
            },
        ),
        Tool(
            name="bizaro_preview",
            description="Preview de las primeras filas de una tabla",
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {"type": "string", "default": "dbizaro"},
                    "schema": {"type": "string", "default": "admuser"},
                    "table": {"type": "string"},
                    "limit": {"type": "integer", "default": 50, "maximum": 500},
                },
                "required": ["table"],
            },
        ),
        Tool(
            name="bizaro_query",
            description=(
                "Ejecuta una consulta SELECT read-only sobre una BBDD. "
                "Solo se permiten SELECT/WITH. No DML/DDL. "
                "Util para explorar datos y debuggear."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {"type": "string", "default": "dbizaro"},
                    "sql": {
                        "type": "string",
                        "description": "Sentencia SELECT (o WITH ... SELECT). Una unica sentencia.",
                    },
                    "max_rows": {"type": "integer", "default": 500, "maximum": 5000},
                },
                "required": ["sql"],
            },
        ),
        Tool(
            name="bizaro_buscar_rutas",
            description="Autodescubre tablas candidatas a almacenar rutas/fases",
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {"type": "string", "default": "dbizaro"},
                },
            },
        ),

        # ── Servicios de negocio ───────────────────────────────────────
        Tool(
            name="get_capacidad",
            description=(
                "Capacidad teorica vs fabricado real por referencia en un "
                "rango de fechas. Cuenta solo piezas de la ultima fase de "
                "la ruta (evita doble contabilidad)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "fecha_inicio": {"type": "string", "description": "YYYY-MM-DD"},
                    "fecha_fin": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["fecha_inicio", "fecha_fin"],
            },
        ),
        Tool(
            name="get_ciclos",
            description="Tiempos de ciclo por seccion/maquina (solo lectura)",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_recursos",
            description="Lista de recursos (maquinas) configurados",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_reports",
            description="Lista informes PDF generados",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_executions",
            description="Ultimas ejecuciones del pipeline",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 20},
                },
            },
        ),
        Tool(
            name="get_execution_detail",
            description="Detalle de una ejecucion: log y PDFs",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                },
                "required": ["id"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        result = _dispatch(name, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False, default=str))]
    except httpx.HTTPStatusError as exc:
        body = ""
        try:
            body = exc.response.text[:500]
        except Exception:
            pass
        return [TextContent(type="text", text=f"HTTP {exc.response.status_code}: {body}")]
    except Exception as exc:
        return [TextContent(type="text", text=f"Error: {exc}")]


def _get(path: str, **params) -> dict:
    r = client.get(path, params={k: v for k, v in params.items() if v is not None})
    r.raise_for_status()
    return r.json()


def _post(path: str, body: dict) -> dict:
    r = client.post(path, json=body)
    r.raise_for_status()
    return r.json()


def _dispatch(name: str, args: dict) -> dict:
    # ── Estado ──────────────────────────────────────────────────────────
    if name == "get_health":
        return _get("/api/health")
    if name == "get_connection_status":
        return _get("/api/conexion/status")

    # ── Explorador BBDD ─────────────────────────────────────────────────
    if name == "bizaro_list_databases":
        return _get("/api/bbdd/databases")
    if name == "bizaro_list_tables":
        return _get("/api/bbdd/tables", database=args.get("database", "dbizaro"))
    if name == "bizaro_columns":
        return _get(
            "/api/bbdd/columns",
            database=args.get("database", "dbizaro"),
            schema=args.get("schema", "admuser"),
            table=args["table"],
        )
    if name == "bizaro_preview":
        return _get(
            "/api/bbdd/preview",
            database=args.get("database", "dbizaro"),
            schema=args.get("schema", "admuser"),
            table=args["table"],
            limit=args.get("limit", 50),
        )
    if name == "bizaro_query":
        return _post("/api/bbdd/query", {
            "database": args.get("database", "dbizaro"),
            "sql": args["sql"],
            "max_rows": args.get("max_rows", 500),
        })
    if name == "bizaro_buscar_rutas":
        return _get("/api/bbdd/buscar-rutas", database=args.get("database", "dbizaro"))

    # ── Negocio ────────────────────────────────────────────────────────
    if name == "get_capacidad":
        return _get(
            "/api/capacidad",
            fecha_inicio=args["fecha_inicio"],
            fecha_fin=args["fecha_fin"],
        )
    if name == "get_ciclos":
        return _get("/api/ciclos")
    if name == "get_recursos":
        return _get("/api/recursos")
    if name == "list_reports":
        return _get("/api/informes")
    if name == "list_executions":
        return _get("/api/historial", limit=args.get("limit", 20))
    if name == "get_execution_detail":
        return _get(f"/api/historial/{args['id']}")

    return {"error": f"Tool desconocido: {name}"}


# ── Entry point ───────────────────────────────────────────────────────────────

async def _main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(_main())
