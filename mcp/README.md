# MCP Server — OEE Planta

Servidor MCP (Model Context Protocol) que expone la API OEE a Claude Code
como herramientas de **consulta read-only**.

## Tools disponibles

### Plataforma
- `get_health` — estado de servicios
- `get_connection_status` — conexion a dbizaro

### Explorador de BBDD (dbizaro u otras)
- `bizaro_list_databases`
- `bizaro_list_tables`
- `bizaro_columns` — columnas de una tabla
- `bizaro_preview` — primeras filas
- `bizaro_query` — **SELECT libre read-only** (valida que no haya DML/DDL)
- `bizaro_buscar_rutas` — autodescubre tablas candidatas a rutas/fases

### Servicios de negocio
- `get_capacidad` — fabricado vs capacidad teorica por referencia
- `get_ciclos` — tiempos de ciclo
- `get_recursos` — maquinas configuradas
- `list_reports`, `list_executions`, `get_execution_detail`

## Requisitos

- Python 3.11+
- API FastAPI corriendo (`make dev` o `make up`)
- `pip install -r requirements.txt`

## Registrar en Claude Code (desarrollo local)

Asumiendo que `make dev` escucha en `127.0.0.1:8000`:

```bash
claude mcp add oee-planta \
  --env OEE_API_URL=http://127.0.0.1:8000 \
  -- python3 /home/eeguskiza/analisis_datos/mcp/server.py
```

O editando `~/.claude.json` (seccion `mcpServers`):

```json
{
  "mcpServers": {
    "oee-planta": {
      "command": "python3",
      "args": ["/home/eeguskiza/analisis_datos/mcp/server.py"],
      "env": { "OEE_API_URL": "http://127.0.0.1:8000" }
    }
  }
}
```

Luego reinicia Claude Code y verifica con `/mcp`.

## Docker

El `docker-compose.yml` ya arranca el MCP. Para usarlo desde Claude Code
en el host via `docker exec`:

```bash
claude mcp add oee-planta -- docker exec -i analisis_datos-mcp-1 python server.py
```

## Seguridad

- `bizaro_query` SOLO acepta `SELECT` / `WITH`. Cualquier token de DML/DDL
  (INSERT, UPDATE, DELETE, DROP, ALTER, EXEC, ...) provoca error 400.
- Una unica sentencia por llamada (se rechaza `;` intermedio).
- Limite de 5000 filas por consulta.
