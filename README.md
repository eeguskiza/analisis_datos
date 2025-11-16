# Informes OEE de planta

Este repositorio genera informaciones analíticas (Disponibilidad, Rendimiento,
Calidad y OEE maestro) por recurso a partir de los CSV exportados del sistema de
producción.

## Estructura

- `data/`
  - `*.csv`: registros de producción por recurso (ej. `t48-enero-nov.csv`).
  - `ciclos.csv`: capacidades ideales (piezas/hora) por recurso y referencia.
- `OEE/disponibilidad`: informe de Disponibilidad (PDF por recurso).
- `OEE/rendimiento`: informe de Rendimiento (PDF por recurso).
- `OEE/calidad`: informe de Calidad (PDF por recurso).
- `OEE/oee`: resumen maestro con todos los recursos en un PDF único.
- `OEE/informes/`: los PDF generados se almacenan aquí, segmentados por tipo.

## Requisitos

- Python 3.10+
- Matplotlib (se usa para renderizar los PDF).

## Uso

1. Coloca los CSV de producción dentro de `data/` con el formato
   `recurso-*.csv`. Asegúrate de incluir `ciclos.csv` para los tiempos ideales.
2. Ejecuta:

   ```bash
   python -m OEE
   ```

   Esto generará:
   - Informes de Disponibilidad (`OEE/informes/disponibilidad/*.pdf`)
   - Informes de Rendimiento (`OEE/informes/rendimiento/*.pdf`)
   - Informes de Calidad (`OEE/informes/calidad/*.pdf`)
   - Resumen maestro de OEE (`OEE/informes/oee/resumen_oee.pdf`)

3. Opcionalmente puedes ejecutar un módulo concreto, por ejemplo solo OEE:

   ```bash
   python -m OEE --modulo oee
   ```

## Notas

- Todos los informes respetan el mismo estilo visual (logo, cabecera, textos en
  español).
- Los CSV deben incluir las columnas descritas en los módulos para que los
  cálculos sean consistentes.
