# Registro de errores conocidos

Lista de bugs y anomalías observadas en la aplicación. Cada entrada indica si es
cosa nuestra (código) o del origen (BBDD / PLC / MES) y, si aplica, el fix.

---

## LUK4 — desacoplamiento entre "tiempo produciendo" y "piezas"

**Fecha detectado:** 2026-04-16
**Endpoint afectado:** `/api/luk4/turno-detail` (panel Pabellón 5)
**Origen:** BBDD origen / PLC — NO es culpa nuestra
**Estado:** documentado, sin fix en la app

### Síntoma

En el panel de LUK4 se puede ver un turno con `availability_pct` bajo ("poco
tiempo produciendo") y al mismo tiempo con más piezas acumuladas que otro
turno con `availability_pct` alto. Ejemplo: T2 con menos tiempo activo pero
más piezas que T1.

### Causa

Los dos indicadores vienen de **tablas distintas** que el PLC puebla de forma
independiente:

- `availability_pct` ← `luk4.estado` (muestreo de `estado_global` — 1 = producing)
- `piezas_buenas / piezas_malas` ← `luk4.tiempos_ciclo` (contadores acumulados)

Si el PLC registra `estado_global ≠ 1` mientras los contadores siguen subiendo
(o al revés), los dos indicadores divergen. El código del dashboard lee
correctamente ambas tablas; el desajuste viene del volcado del PLC a SQL
Server.

### Qué hacer

Revisar en origen el mapeo PLC → `luk4.estado` vs `luk4.tiempos_ciclo`:
- ¿Existen estados distintos de 1 en los que también se produce?
  (por ejemplo modo semi-automático, revisión en curso, etc.)
- ¿El muestreo de `estado` tiene la misma frecuencia que el de `tiempos_ciclo`?
- ¿Hay reseteos de contador que generen saltos negativos?

Mientras no se alinee en origen, el panel puede mostrar cifras que parecen
incoherentes sin que haya un bug en la app.

### Reproducción

```sql
-- Comparar densidad de estado=1 vs incrementos de contador en el mismo rango
SELECT
  (SELECT COUNT(*) FROM luk4.estado
    WHERE timestamp >= '2026-04-16 14:00' AND timestamp < '2026-04-16 22:00'
    AND estado_global = 1) AS muestras_producing_t2,
  (SELECT MAX(contador_piezas_totales) - MIN(contador_piezas_totales)
    FROM luk4.tiempos_ciclo
    WHERE timestamp >= '2026-04-16 14:00' AND timestamp < '2026-04-16 22:00'
  ) AS delta_totales_t2;
```

Si `delta_totales_t2` es alto pero `muestras_producing_t2` es bajo en
proporción al resto de turnos, es el síntoma.

---

## LUK4 — piezas de frontera entre turnos (ARREGLADO)

**Fecha detectado:** 2026-04-16
**Fecha fix:** 2026-04-16
**Endpoint afectado:** `/api/luk4/turno-detail`
**Fichero:** `api/routers/luk4.py`
**Origen:** nuestro (código)
**Estado:** FIX APLICADO

### Síntoma

En el cambio de turno, las piezas producidas entre la última lectura del
turno N y la primera lectura del turno N+1 podían caer fuera de ambos
conteos. Si el PLC deja un hueco de muestreo de varios minutos en la
frontera (típico cuando la máquina está parada en la transición), las
piezas de ese tramo se perdían — `pz_T1 + pz_T2 + pz_T3 ≠ total_diario`.

### Causa

El endpoint calculaba `first_turno` como "primera lectura DENTRO del turno"
en lugar de "última lectura ANTES del turno". Eso dejaba un hueco entre
`last_T1` (última lectura < 14:00) y `first_T2` (primera lectura ≥ 14:00)
que podía ser de varios minutos.

### Fix

Cambiado `first_turno` a la última lectura con `timestamp < t_start` (cierra
la cadena sin huecos). Fallback a la primera lectura dentro del turno cuando
no hay lectura anterior (primer turno histórico del sistema). Además se
cambió `ORDER BY idtiempos_ciclo` por `ORDER BY timestamp` para robustez
frente a inserciones fuera de orden.
