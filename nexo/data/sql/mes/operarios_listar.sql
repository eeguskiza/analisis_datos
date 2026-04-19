-- Lista operarios de IZARO (admuser.fmesope) con numero de registros del
-- ultimo mes y fecha de ultimo registro (JOIN a admuser.fmesdtc).
-- Extraido verbatim de api/routers/operarios.py:34-53 (plan 03-02 Task 3.4).
-- 2-part names (DATA-09): engine_mes DATABASE=dbizaro.
SELECT
    CAST(ope.op010 AS INT) AS codigo,
    RTRIM(ope.op020) AS nombre,
    CAST(ope.op060 AS INT) AS activo,
    COALESCE(act.n_mes, 0) AS n_registros_mes,
    act.ultimo
FROM admuser.fmesope ope
LEFT JOIN (
    SELECT CAST(dt140 AS INT) AS ope_cod,
        COUNT(*) AS n_mes,
        MAX(CONVERT(DATE, dt060)) AS ultimo
    FROM admuser.fmesdtc
    WHERE dt060 >= DATEADD(MONTH, -1, GETDATE())
        AND dt140 IS NOT NULL AND RTRIM(dt140) != ''
    GROUP BY CAST(dt140 AS INT)
) act ON act.ope_cod = CAST(ope.op010 AS INT)
WHERE ope.op000 = 'ALGA'
ORDER BY COALESCE(act.n_mes, 0) DESC, ope.op010
