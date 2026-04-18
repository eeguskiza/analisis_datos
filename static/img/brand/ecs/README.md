# Logo de ECS Mobility

Logo corporativo oficial de ECS Mobility SL. **No modificar.**

Uso: footer, documentación oficial, emails, cualquier lugar donde
aparezca la marca corporativa.

## Variantes disponibles

| Archivo | Uso sugerido |
|---------|--------------|
| `logo.png` | Principal (alias de `horizontal-positivo.png`). Fondos claros. |
| `horizontal-negativo.png` | Horizontal sobre fondos oscuros. |
| `vertical-positivo.png` | Vertical sobre fondos claros. Composiciones verticales (tarjetas, cabeceras de email). |
| `vertical-negativo.png` | Vertical sobre fondos oscuros. |

El archivo principal `logo.png` es el que cargan los templates por
defecto. Las otras variantes se consumen bajo demanda
(`/static/img/brand/ecs/vertical-positivo.png`, etc.).

Nota: el archivo `data/ecs-logo.png` sigue existiendo y lo usa el
módulo OEE para estampar el logo en los PDFs generados por matplotlib
(`api/config.py` → `logo_filename = "ecs-logo.png"`). La unificación
con `static/img/brand/ecs/logo.png` es trabajo de Sprint 2 (cuando la
capa de datos y config se reorganice); no entra en Sprint 0.
