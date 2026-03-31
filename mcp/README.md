# MCP Server — OEE Planta

Servidor MCP (Model Context Protocol) para exponer herramientas de consulta y generacion de informes OEE a agentes Claude.

## Estado: Pendiente de implementacion

## Herramientas previstas

| Tool | Descripcion |
|------|-------------|
| `generate_oee_report` | Ejecuta el pipeline completo (POST /api/pipeline/run) |
| `list_reports` | Lista informes generados (GET /api/informes) |
| `get_connection_status` | Estado de conexion a BD (GET /api/conexion/status) |
| `get_ciclos` | Tiempos de ciclo actuales (GET /api/ciclos) |
| `update_ciclos` | Actualiza tiempos de ciclo (PUT /api/ciclos) |

## Recursos MCP

| Resource | Descripcion |
|----------|-------------|
| `oee://informes/{date}` | PDFs de una fecha |
| `oee://ciclos` | Contenido de ciclos.csv |
| `oee://config` | Configuracion actual (sin credenciales) |

## Ejecucion (futuro)

```bash
docker compose up mcp
```
