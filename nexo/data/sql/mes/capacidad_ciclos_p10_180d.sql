-- Ciclos observados por (referencia, CT) en los ultimos 180 dias para
-- calcular el percentil 10 (ciclo teorico = mejor ritmo sostenido).
-- La query devuelve TODOS los ciclos de refs y cts presentes en la
-- ventana; el router filtra client-side los pares (ref, ct) relevantes.
-- Esto evita la composicion dinamica de OR que hacia el codigo viejo
-- (DATA-09: no mas string-interpolation) — dos bindparams expanding
-- (:refs, :cts) acotan server-side antes del filtro final.
--
-- 2-part names (engine_mes DATABASE=dbizaro).
SELECT
    RTRIM(COALESCE(lof.lo030, ''))          AS referencia,
    RTRIM(CAST(dtc.dt150 AS NVARCHAR(50)))  AS ct,
    CAST(dtc.dt090 AS FLOAT)                AS tiempo_min,
    CAST(dtc.dt130 AS FLOAT)                AS cantidad
FROM admuser.fmesdtc dtc
LEFT JOIN admuser.fprolof lof
    ON  RTRIM(lof.lo010) = RTRIM(dtc.dt020)
    AND lof.lo020 = dtc.dt030
WHERE dtc.dt060 >= DATEADD(DAY, -180, GETDATE())
  AND dtc.dt110 = '0'
  AND COALESCE(CAST(dtc.dt130 AS FLOAT), 0) >= 5
  AND COALESCE(CAST(dtc.dt090 AS FLOAT), 0) >= 2
  AND RTRIM(COALESCE(lof.lo030, '')) IN :refs
  AND RTRIM(CAST(dtc.dt150 AS NVARCHAR(50))) IN :cts
