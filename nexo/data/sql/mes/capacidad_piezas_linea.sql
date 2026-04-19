-- Capacidad / piezas reales por referencia en CTs "Linea%" (fmesrec.re020).
-- Cuenta piezas UNA sola vez en el centro de trabajo final (linea de
-- montaje). Extraido verbatim de api/routers/capacidad.py pre-refactor
-- (plan 03-02 Task 3.3). 2-part names (DATA-09): engine_mes tiene
-- DATABASE=dbizaro en DSN.
SELECT
    RTRIM(COALESCE(lof.lo030, ''))          AS referencia,
    RTRIM(CAST(dtc.dt150 AS NVARCHAR(50)))  AS ct,
    RTRIM(rec.re020)                         AS ct_nombre,
    SUM(CASE WHEN dtc.dt110 = '0'
             THEN COALESCE(CAST(dtc.dt130 AS FLOAT), 0)
             ELSE 0 END)                    AS piezas,
    SUM(CASE WHEN dtc.dt110 IN ('0','1')
             THEN COALESCE(CAST(dtc.dt090 AS FLOAT), 0)
             ELSE 0 END)                    AS tiempo_min,
    MIN(dtc.dt060)                           AS fecha_min,
    MAX(dtc.dt060)                           AS fecha_max
FROM admuser.fmesdtc dtc
INNER JOIN admuser.fmesrec rec
    ON CAST(rec.re010 AS INT) = CAST(dtc.dt150 AS INT)
LEFT JOIN admuser.fprolof lof
    ON  RTRIM(lof.lo010) = RTRIM(dtc.dt020)
    AND lof.lo020 = dtc.dt030
WHERE CONVERT(DATE, dtc.dt060) BETWEEN :fecha_inicio AND :fecha_fin
  AND RTRIM(rec.re020) LIKE 'Linea%'
GROUP BY RTRIM(COALESCE(lof.lo030, '')),
         RTRIM(CAST(dtc.dt150 AS NVARCHAR(50))),
         RTRIM(rec.re020)
HAVING SUM(CASE WHEN dtc.dt110 = '0'
                THEN COALESCE(CAST(dtc.dt130 AS FLOAT), 0)
                ELSE 0 END) > 0
