# Informes OEE de planta

Este repositorio genera informaciones analíticas (Disponibilidad, Rendimiento,
Calidad y OEE maestro) por recurso a partir de los CSV exportados del sistema de
producción.

## Estructura

- `data/`
  - `excels/`: aquí se dejan los Excel de entrada. El flujo los convierte a CSV
    y elimina los originales.
  - `csv/`: CSV planos generados desde Excel.
  - `recursos/<SECCIÓN>/`: CSV definitivos por recurso (ej. `LINEAS/luk1-*.csv`,
    `TALLADORAS/t48-*.csv`). El mapeo recurso→sección está en `OEE/utils/excel_import.py`.
  - `ciclos.csv`: capacidades ideales (piezas/hora) por recurso y referencia.
- `OEE/...`: código de generación por módulo.
- `informes/`: salida final. Cada ejecución crea una carpeta `YYYY-MM-DD` con
  subcarpetas por sección (`LINEAS/`, `TALLADORAS/`, etc.).
  - Dentro de cada sección: `*_oee_seccion.pdf` (maestro), páginas de detalle
    por puesto, referencias y los informes individuales de disponibilidad,
    rendimiento y calidad de cada máquina.
  - En la raíz del día: `resumen_oee.pdf` (si ejecutas el módulo `oee`).

## Requisitos

- Python 3.10+
- Matplotlib (se usa para renderizar los PDF).

## Uso

1. Coloca los Excel en `data/excels/` (se convertirán a CSV y se eliminarán) o,
   si ya son CSV, colócalos directamente en `data/recursos/<SECCIÓN>/`.
   Asegúrate de incluir `data/ciclos.csv` para los tiempos ideales (piezas/hora).
2. Ejecuta:

   ```bash
   python -m OEE
   ```

   Esto generará (por ejemplo en `informes/2025-11-17/`):
   - Carpeta `LINEAS/` con `LINEAS_oee_seccion.pdf` y subcarpetas `coroa/`,
     `luk1/`, etc., cada una con sus PDF individuales.
   - Carpeta `TALLADORAS/` con su maestro y las máquinas de esa sección.
   - `resumen_oee.pdf` en la raíz del día (si ejecutas el módulo `oee`).

3. Opcionalmente puedes ejecutar un módulo concreto, por ejemplo solo OEE:

   ```bash
   python -m OEE --modulo oee
   ```

4. Para el informe maestro por secciones puedes limitar las secciones con `--seccion` (se puede repetir):

   ```bash
   python -m OEE --modulo oee_secciones --seccion talladoras --seccion lineas
   ```

## Notas

- Al arrancar, el flujo convierte automáticamente los Excel de `data/excels/`,
  coloca los CSV en `data/csv/` y los distribuye en `data/recursos/<SECCIÓN>/`
  según el mapa `RESOURCE_SECTION_MAP`.
- Los CSV deben incluir las columnas descritas en los módulos para que los
  cálculos sean consistentes.
- Si falta una referencia en `ciclos.csv`, el informe de Rendimiento la añade
  con `tiempo_ciclo = 0` y muestra un aviso para que completes el valor real.
