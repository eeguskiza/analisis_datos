-- NOTA: engine_mes tiene DATABASE=dbizaro en connection string (plan 03-01),
-- por eso usamos 2-part names (admuser.fmesmic) en lugar de 3-part.
-- IN :codes es bindparam expanding (SQLAlchemy 2.0) — reemplaza la
-- string interpolation ct0,ct1,... del router viejo (DATA-09).
SELECT
    CAST(RTRIM(mi020) AS INT)    AS ct,
    COUNT(*)                     AS piezas_hoy,
    MAX(CAST(mi100 AS TIME))     AS ultimo_evento,
    (SELECT TOP 1 RTRIM(m2.mi060)
     FROM admuser.fmesmic m2
     WHERE RTRIM(m2.mi020) = RTRIM(m.mi020)
       AND CONVERT(DATE, m2.mi090) = CONVERT(DATE, GETDATE())
     ORDER BY m2.mi050 DESC)     AS referencia
FROM admuser.fmesmic m
WHERE CONVERT(DATE, mi090) = CONVERT(DATE, GETDATE())
  AND RTRIM(mi020) IN :codes
GROUP BY RTRIM(mi020)
