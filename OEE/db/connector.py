"""
Conector a la base de datos IZARO MES (SQL Server / dbizaro).

Extrae datos de producción directamente desde las tablas admuser.fmesdtc y las
tablas relacionadas, generando los CSV que el pipeline OEE espera en
data/recursos/<SECCION>/.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False


CONFIG_FILE = Path(__file__).resolve().parents[2] / "data" / "db_config.json"

# ---------------------------------------------------------------------------
# Driver ODBC: preferencia de mayor a menor versión
# ---------------------------------------------------------------------------
_DRIVER_PREFERENCE = [
    "ODBC Driver 18 for SQL Server",
    "ODBC Driver 17 for SQL Server",
    "ODBC Driver 13 for SQL Server",
    "SQL Server Native Client 11.0",
    "SQL Server",
]


def detectar_driver() -> str:
    """Devuelve el mejor driver ODBC para SQL Server disponible en este equipo."""
    if not PYODBC_AVAILABLE:
        return _DRIVER_PREFERENCE[1]  # fallback sin error
    drivers_instalados = pyodbc.drivers()
    for d in _DRIVER_PREFERENCE:
        if d in drivers_instalados:
            return d
    # Si ninguno conocido, devolver el primero que contenga "SQL Server"
    for d in drivers_instalados:
        if "SQL Server" in d:
            return d
    return _DRIVER_PREFERENCE[1]


DEFAULT_CONFIG: dict = {
    "server": "",
    "port": "1433",
    "database": "dbizaro",
    "driver": "",        # vacío → se auto-detecta al conectar
    "encrypt": "",       # vacío -> auto según driver (Driver 18: yes)
    "trust_server_certificate": "",  # vacío -> auto según driver (Driver 18: yes)
    "user": "",
    "password": "",
    "uf_code": "",
    "referencia_campo": "",
    "recursos": [],
}

# Mapa recurso -> sección (igual que excel_import.py)
RESOURCE_SECTION_MAP: Dict[str, str] = {
    "luk1": "LINEAS",
    "luk2": "LINEAS",
    "luk3": "LINEAS",
    "luk6": "LINEAS",
    "coroa": "LINEAS",
    "vw1": "LINEAS",
    "omr": "LINEAS",
    "t48": "TALLADORAS",
}


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if CONFIG_FILE.exists():
        stored = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        return {**DEFAULT_CONFIG, **stored}
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Conexión
# ---------------------------------------------------------------------------

def _to_bool(value: object, default: bool = False) -> bool:
    """Normaliza valores bool leídos de JSON/UI."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "si", "sí", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    return default


def _build_connection_string(cfg: dict) -> str:
    driver = (cfg.get("driver") or "").strip() or detectar_driver()
    server = cfg["server"]
    port = cfg.get("port") or "1433"
    database = cfg["database"]
    user = (cfg.get("user") or "").strip()
    password = cfg.get("password") or ""
    is_driver18 = "ODBC Driver 18 for SQL Server" in driver
    encrypt = _to_bool(cfg.get("encrypt"), default=is_driver18)
    trust_server_cert = _to_bool(
        cfg.get("trust_server_certificate"),
        default=is_driver18,
    )

    security_part = (
        f"Encrypt={'yes' if encrypt else 'no'};"
        f"TrustServerCertificate={'yes' if trust_server_cert else 'no'};"
    )

    if user:
        return (
            f"DRIVER={{{driver}}};"
            f"SERVER={server},{port};"
            f"DATABASE={database};"
            f"{security_part}"
            f"UID={user};"
            f"PWD={password};"
        )
    return (
        f"DRIVER={{{driver}}};"
        f"SERVER={server},{port};"
        f"DATABASE={database};"
        f"{security_part}"
        f"Trusted_Connection=yes;"
    )


def test_conexion(cfg: dict) -> str:
    """Prueba la conexión y devuelve un mensaje de estado."""
    if not PYODBC_AVAILABLE:
        return "ERROR: pyodbc no está instalado. Ejecuta: pip install pyodbc"
    try:
        conn = pyodbc.connect(_build_connection_string(cfg), timeout=10)
        cursor = conn.cursor()
        cursor.execute("SELECT @@VERSION")
        version = (cursor.fetchone()[0] or "")[:80]
        conn.close()
        return f"OK — {version}"
    except Exception as exc:
        return f"ERROR: {exc}"


def explorar_columnas_fmesdtc(cfg: dict) -> List[str]:
    """
    Devuelve las columnas de admuser.fmesdtc para ayudar a identificar el
    campo que contiene la referencia de pieza.
    """
    if not PYODBC_AVAILABLE:
        return ["pyodbc no instalado"]
    try:
        conn = pyodbc.connect(_build_connection_string(cfg), timeout=10)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = 'admuser' AND TABLE_NAME = 'fmesdtc'
            ORDER BY ORDINAL_POSITION
            """
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            f"{r[0]}  ({r[1]}{f'({r[2]})' if r[2] else ''})"
            for r in rows
        ]
    except Exception as exc:
        return [f"ERROR: {exc}"]


def detectar_recursos(cfg: dict) -> List[dict]:
    """
    Detecta centros de trabajo disponibles en IZARO.

    Devuelve lista de dicts:
      {codigo, nombre_izaro, ultimo_registro, n_registros_mes}
    """
    if not PYODBC_AVAILABLE:
        raise RuntimeError("pyodbc no está instalado")
    conn = pyodbc.connect(_build_connection_string(cfg), timeout=15)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT
                CAST(rec.re010 AS INT) AS codigo,
                RTRIM(rec.re020) AS nombre_izaro,
                act.ultimo,
                act.n_mes
            FROM admuser.fmesrec AS rec
            LEFT JOIN (
                SELECT
                    CAST(dt150 AS INT) AS ct,
                    MAX(CONVERT(DATE, dt060)) AS ultimo,
                    SUM(CASE WHEN dt060 >= DATEADD(MONTH, -1, GETDATE()) THEN 1 ELSE 0 END) AS n_mes
                FROM admuser.fmesdtc
                GROUP BY CAST(dt150 AS INT)
            ) AS act ON act.ct = CAST(rec.re010 AS INT)
            ORDER BY act.n_mes DESC, rec.re010
        """)
        rows = cursor.fetchall()
    finally:
        conn.close()

    return [
        {
            "codigo": int(r[0]),
            "nombre_izaro": (r[1] or "").strip(),
            "ultimo_registro": r[2].isoformat() if r[2] else None,
            "n_registros_mes": int(r[3] or 0),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Consulta principal
# ---------------------------------------------------------------------------

_SQL_TEMPLATE = """\
SELECT
    dtc.dt060                                                              AS fecha,
    STUFF(RIGHT('0000' + CAST(dtc.dt080 AS VARCHAR(4)), 4), 3, 0, ':')   AS hora_inicio,
    STUFF(RIGHT('0000' + CAST(dtc.dt085 AS VARCHAR(4)), 4), 3, 0, ':')   AS hora_fin,
    COALESCE(CAST(dtc.dt090 AS FLOAT), 0)                                  AS tiempo,
    RTRIM(CAST(dtc.dt150 AS NVARCHAR(50)))                                AS centro_trabajo,
    CASE dtc.dt110
        WHEN '0' THEN 'Producción'
        WHEN '1' THEN 'Preparación'
        WHEN '2' THEN 'Incidencia'
        ELSE ''
    END                                                                    AS proceso,
    COALESCE(RTRIM(CAST(inc.in020 AS NVARCHAR(MAX))), '')                 AS incidencia,
    COALESCE(CAST(dtc.dt130 AS FLOAT), 0)                                  AS cantidad,
    COALESCE(def.malas, 0)                                                 AS malas,
    COALESCE(def.recuperadas, 0)                                           AS recuperadas,
    COALESCE(RTRIM(lof.lo030), '')                                         AS referencia
FROM admuser.fmesdtc AS dtc
LEFT JOIN (
    SELECT dd010, dd030, dd040, dd050,
           SUM(COALESCE(CAST(dd080 AS FLOAT), 0)) AS malas,
           SUM(COALESCE(CAST(dd090 AS FLOAT), 0)) AS recuperadas
    FROM admuser.fmesddf
    GROUP BY dd010, dd030, dd040, dd050
) AS def
    ON  RTRIM(def.dd010) = RTRIM(dtc.dt020)
    AND def.dd030 = dtc.dt030
    AND def.dd040 = dtc.dt040
    AND def.dd050 = dtc.dt050
LEFT JOIN admuser.fmesinc AS inc
    ON  RTRIM(inc.in010) = RTRIM(dtc.dt120)
    AND RTRIM(inc.in000) = RTRIM(dtc.dt000)
LEFT JOIN admuser.fprolof AS lof
    ON  RTRIM(lof.lo010) = RTRIM(dtc.dt020)
    AND lof.lo020 = dtc.dt030
WHERE (
        -- Registros con hora >= 06:00 del rango pedido (T1, T2, inicio T3)
        (CONVERT(DATE, dtc.dt060) BETWEEN ? AND ? AND dtc.dt080 >= 600)
        OR
        -- Registros de madrugada (<06:00) del día siguiente a fecha_fin
        -- (continuación del T3 del último día pedido)
        (CONVERT(DATE, dtc.dt060) = DATEADD(DAY, 1, CAST(? AS DATE)) AND dtc.dt080 < 600)
  )
  AND RTRIM(CAST(dtc.dt150 AS NVARCHAR(50))) IN ({ct_placeholders})
  {uf_filter}
ORDER BY dtc.dt150, dtc.dt060, dtc.dt080
"""


def extraer_datos(
    cfg: dict,
    fecha_inicio: date,
    fecha_fin: date,
) -> List[dict]:
    """
    Extrae datos de IZARO y devuelve lista de dicts con columnas normalizadas.

    Cada dict: {recurso, seccion, fecha, h_ini, h_fin, tiempo, proceso,
                incidencia, cantidad, malas, recuperadas, referencia}
    """
    if not PYODBC_AVAILABLE:
        raise RuntimeError("pyodbc no está instalado. Ejecuta: pip install pyodbc")

    recursos_activos = [r for r in cfg.get("recursos", []) if r.get("activo", True)]
    if not recursos_activos:
        raise ValueError("No hay recursos activos configurados.")

    ct_codes = [str(r["centro_trabajo"]).strip() for r in recursos_activos]
    ct_to_resource: Dict[str, str] = {
        str(r["centro_trabajo"]).strip(): r["nombre"] for r in recursos_activos
    }

    uf_code = (cfg.get("uf_code") or "").strip()
    uf_filter = "AND dtc.dt010 = ?" if uf_code else ""

    sql = _SQL_TEMPLATE.format(
        ct_placeholders=",".join(["?"] * len(ct_codes)),
        uf_filter=uf_filter,
    )

    fi = fecha_inicio.strftime("%Y-%m-%d")
    ff = fecha_fin.strftime("%Y-%m-%d")
    params: list = [fi, ff, ff, *ct_codes]
    if uf_code:
        params.append(uf_code)

    conn = pyodbc.connect(_build_connection_string(cfg), timeout=30)
    try:
        df = pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()

    if df.empty:
        return []

    rows: List[dict] = []
    for _, r in df.iterrows():
        ct = str(r["centro_trabajo"]).strip()
        nombre = ct_to_resource.get(ct)
        if not nombre:
            continue
        seccion = RESOURCE_SECTION_MAP.get(nombre.lower(), "GENERAL")
        fecha_val = r["fecha"]
        if hasattr(fecha_val, "date"):
            fecha_val = fecha_val.date()
        rows.append({
            "recurso": nombre,
            "seccion": seccion,
            "fecha": fecha_val,
            "h_ini": str(r["hora_inicio"] or ""),
            "h_fin": str(r["hora_fin"] or ""),
            "tiempo": float(r["tiempo"] or 0),
            "proceso": str(r["proceso"] or ""),
            "incidencia": str(r["incidencia"] or ""),
            "cantidad": float(r["cantidad"] or 0),
            "malas": float(r["malas"] or 0),
            "recuperadas": float(r["recuperadas"] or 0),
            "referencia": str(r["referencia"] or ""),
        })
    return rows


def datos_a_csvs(rows: List[dict], recursos_dir: Path) -> Dict[str, Path]:
    """
    Escribe datos (lista de dicts) como CSVs en recursos_dir/<SECCION>/.
    Devuelve {nombre_recurso: path_csv}.
    """
    import pandas as _pd

    CSV_COLUMNS = {
        "fecha": "Fecha", "h_ini": "H Ini", "h_fin": "F Fin",
        "tiempo": "Tiempo", "proceso": "Proceso", "incidencia": "Incidencia",
        "cantidad": "Cantidad", "malas": "Malas", "recuperadas": "Recu.",
        "referencia": "Refer.",
    }

    # Agrupar por recurso
    by_recurso: Dict[str, list] = {}
    for r in rows:
        by_recurso.setdefault(r["recurso"], []).append(r)

    generated: Dict[str, Path] = {}
    for nombre, data in by_recurso.items():
        seccion = data[0]["seccion"]
        out_dir = recursos_dir / seccion
        out_dir.mkdir(parents=True, exist_ok=True)

        # Limpiar CSVs anteriores de este recurso
        for old_csv in out_dir.glob(f"{nombre}-*.csv"):
            old_csv.unlink()

        df = _pd.DataFrame(data)
        df = df.rename(columns=CSV_COLUMNS)
        df = df[list(CSV_COLUMNS.values())]

        fechas = sorted(df["Fecha"].unique())
        if len(fechas) <= 1:
            tag = str(fechas[0]).replace("-", "") if fechas else "nodata"
        else:
            tag = f"{str(fechas[0]).replace('-', '')}_{str(fechas[-1]).replace('-', '')}"

        out_path = out_dir / f"{nombre}-{tag}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        generated[nombre] = out_path

    return generated


def calcular_ciclos_reales(
    cfg: dict,
    centro_trabajo: int,
    dias_atras: int = 30,
) -> List[dict]:
    """
    Calcula ciclos reales por referencia a partir de contadores de IZARO.

    Lee los contadores (zmeshva) y cruza con la referencia que se fabricaba
    en ese momento (fmesdtc/fprolof). Filtra ruido y calcula:
      - ciclo mediano en segundos
      - piezas/hora equivalente

    Devuelve lista de dicts:
      {referencia, ciclo_seg, piezas_hora, n_muestras, fecha_min, fecha_max}
    """
    if not PYODBC_AVAILABLE:
        raise RuntimeError("pyodbc no está instalado")

    conn = pyodbc.connect(_build_connection_string(cfg), timeout=30)
    cursor = conn.cursor()
    ct = str(centro_trabajo)

    try:
        # 1) Contadores: valor acumulado en cada instante
        cursor.execute("""
            SELECT
                CONVERT(DATE, z.hv080) AS fecha,
                z.hv090               AS hora,
                CAST(z.hv040 AS FLOAT) AS valor
            FROM admuser.zmeshva z
            WHERE z.hv010 LIKE ?
              AND z.hv030 LIKE '%Contador%'
              AND z.hv030 NOT LIKE '%Ultimo%'
              AND z.hv080 >= DATEADD(DAY, ?, GETDATE())
            ORDER BY z.hv080, z.hv090
        """, (f'%{ct}%', -dias_atras))
        contadores = cursor.fetchall()

        if not contadores:
            conn.close()
            return []

        # 2) Referencia activa por CT+fecha+hora (produccion en fmesdtc)
        cursor.execute("""
            SELECT
                CONVERT(DATE, dtc.dt060) AS fecha,
                dtc.dt080 AS h_ini,
                dtc.dt085 AS h_fin,
                COALESCE(RTRIM(lof.lo030), '') AS referencia
            FROM admuser.fmesdtc dtc
            LEFT JOIN admuser.fprolof lof
                ON RTRIM(lof.lo010) = RTRIM(dtc.dt020)
                AND lof.lo020 = dtc.dt030
            WHERE RTRIM(CAST(dtc.dt150 AS NVARCHAR(50))) = ?
              AND dtc.dt110 = '0'
              AND dtc.dt060 >= DATEADD(DAY, ?, GETDATE())
              AND COALESCE(RTRIM(lof.lo030), '') <> ''
            ORDER BY dtc.dt060, dtc.dt080
        """, (ct, -dias_atras))
        tramos = cursor.fetchall()
    finally:
        conn.close()

    # 3) Construir mapa fecha+hora → referencia
    ref_map = []
    for t in tramos:
        ref_map.append({
            "fecha": t[0],
            "h_ini": int(t[1] or 0),
            "h_fin": int(t[2] or 0),
            "referencia": t[3],
        })

    def _find_ref(fecha, hora_str):
        """Busca la referencia activa en un momento dado."""
        try:
            hhmm = int(hora_str.replace(":", "").replace(".", "")[:4])
        except (ValueError, TypeError, AttributeError):
            return ""
        for tr in ref_map:
            if tr["fecha"] == fecha and tr["h_ini"] <= hhmm <= (tr["h_fin"] or 2359):
                return tr["referencia"]
        return ""

    # 4) Calcular deltas entre contadores consecutivos
    import math
    import statistics
    from collections import defaultdict
    from datetime import datetime as _dt

    samples: dict[str, list[float]] = defaultdict(list)
    daily: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    prev_val = None
    prev_time = None
    prev_fecha = None

    for row in contadores:
        fecha, hora, valor = row[0], row[1], row[2]
        if valor is None or valor == 0:
            prev_val = None
            continue

        if prev_val is not None and prev_fecha == fecha:
            delta_piezas = valor - prev_val
            try:
                t1 = _dt.combine(fecha, _dt.strptime(str(prev_time), "%H:%M:%S").time()) if isinstance(prev_time, str) else _dt.combine(fecha, prev_time) if hasattr(prev_time, 'hour') else None
                t2 = _dt.combine(fecha, _dt.strptime(str(hora), "%H:%M:%S").time()) if isinstance(hora, str) else _dt.combine(fecha, hora) if hasattr(hora, 'hour') else None
                delta_seg = (t2 - t1).total_seconds() if t1 and t2 and t2 > t1 else None
            except Exception:
                delta_seg = None

            if delta_seg and delta_piezas > 0 and delta_seg > 0:
                ciclo = delta_seg / delta_piezas
                if 1 <= ciclo <= 600:
                    ref = _find_ref(fecha, str(hora))
                    if ref:
                        samples[ref].append(ciclo)
                        daily[ref][fecha.isoformat()].append(ciclo)

        prev_val = valor
        prev_time = hora
        prev_fecha = fecha

    # ── Funciones de estimacion robusta ──────────────────────────────

    def _iqr_filter(data: list[float]) -> list[float]:
        """Filtra outliers usando IQR (rango intercuartilico)."""
        if len(data) < 4:
            return data
        s = sorted(data)
        n = len(s)
        q1 = s[n // 4]
        q3 = s[3 * n // 4]
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        return [x for x in data if lo <= x <= hi]

    def _kde_mode(data: list[float]) -> float:
        """
        Estima la moda usando KDE (Kernel Density Estimation) ligero.

        Encuentra el valor mas probable de la distribucion, que es el
        ciclo en regimen estable (ignora arranques, paros, cambios).
        Usa un kernel gaussiano con ancho de banda de Silverman.
        """
        if len(data) < 5:
            return statistics.median(data)

        n = len(data)
        std = statistics.stdev(data)
        if std == 0:
            return data[0]

        # Ancho de banda de Silverman
        bw = 0.9 * min(std, (statistics.median(sorted(
            [abs(x - statistics.median(data)) for x in data]
        )) * 1.4826)) * n ** (-0.2)
        if bw <= 0:
            bw = std * 0.5

        # Evaluar densidad en una rejilla
        lo = min(data)
        hi = max(data)
        n_bins = min(200, max(50, n))
        step = (hi - lo) / n_bins if hi > lo else 1
        best_x, best_density = lo, 0

        for i in range(n_bins + 1):
            x = lo + i * step
            density = sum(
                math.exp(-0.5 * ((x - xi) / bw) ** 2) for xi in data
            ) / (n * bw)
            if density > best_density:
                best_density = density
                best_x = x

        return best_x

    def _confidence(data: list[float], filtered: list[float]) -> int:
        """
        Score de confianza 0-100.

        Factores:
        - n muestras (mas = mejor, satura en ~100)
        - CV bajo (coeficiente de variacion, menos dispersion = mejor)
        - Ratio de datos no-outlier (mas datos utiles = mejor)
        """
        n = len(filtered)
        if n < 3:
            return 0

        # Factor muestras: log scale, 10→50%, 50→80%, 200→95%
        f_n = min(1.0, math.log(n + 1) / math.log(200))

        # Factor CV: cv=0→100%, cv=0.3→50%, cv=1→10%
        mean = statistics.mean(filtered)
        cv = (statistics.stdev(filtered) / mean) if mean > 0 and n > 1 else 1.0
        f_cv = max(0, 1.0 - cv * 1.5)

        # Factor retención: qué % de datos sobrevive al filtro IQR
        f_ret = len(filtered) / len(data) if data else 0

        score = (f_n * 0.35 + f_cv * 0.45 + f_ret * 0.20) * 100
        return min(100, max(0, round(score)))

    def _histogram(data: list[float], n_bins: int = 20) -> list[dict]:
        """Genera histograma para visualización."""
        if len(data) < 3:
            return []
        lo, hi = min(data), max(data)
        if lo == hi:
            return [{"x": round(lo, 1), "y": len(data)}]
        step = (hi - lo) / n_bins
        bins = [0] * n_bins
        for v in data:
            idx = min(int((v - lo) / step), n_bins - 1)
            bins[idx] += 1
        return [
            {"x": round(lo + (i + 0.5) * step, 1), "y": bins[i]}
            for i in range(n_bins) if bins[i] > 0
        ]

    def _agg(ciclos_raw: list[float]) -> dict:
        """Agrega ciclos: IQR filter → KDE mode → confianza."""
        filtered = _iqr_filter(ciclos_raw)
        if not filtered:
            filtered = ciclos_raw

        modo = _kde_mode(filtered)
        mediana = statistics.median(filtered)
        ciclo = round(modo, 1)
        ph = round(3600 / ciclo, 1) if ciclo > 0 else 0
        conf = _confidence(ciclos_raw, filtered)

        return {
            "ciclo_seg": ciclo,
            "piezas_hora": ph,
            "n_muestras": len(filtered),
            "n_descartados": len(ciclos_raw) - len(filtered),
            "confianza": conf,
            "mediana": round(mediana, 1),
            "metodo": "kde_mode",
        }

    # ── Agregar por referencia ───────────────────────────────────────

    result = []
    for ref, ciclos_raw in sorted(samples.items()):
        if len(ciclos_raw) < 3:
            continue

        agg = _agg(ciclos_raw)
        fechas = sorted(daily[ref].keys())

        por_dia = []
        for f in fechas:
            day_raw = daily[ref][f]
            if len(day_raw) >= 2:
                d = _agg(day_raw)
                por_dia.append({
                    "fecha": f,
                    "ciclo_seg": d["ciclo_seg"],
                    "piezas_hora": d["piezas_hora"],
                    "n_muestras": d["n_muestras"],
                })

        histo = _histogram(_iqr_filter(ciclos_raw))

        result.append({
            "referencia": ref,
            "ciclo_seg": agg["ciclo_seg"],
            "piezas_hora": agg["piezas_hora"],
            "n_muestras": agg["n_muestras"],
            "n_descartados": agg["n_descartados"],
            "confianza": agg["confianza"],
            "mediana_seg": agg["mediana"],
            "fecha_min": fechas[0] if fechas else None,
            "fecha_max": fechas[-1] if fechas else None,
            "por_dia": por_dia,
            "histograma": histo,
        })

    result.sort(key=lambda x: -x["n_muestras"])
    return result


def extraer_y_guardar_csv(
    cfg: dict,
    fecha_inicio: date,
    fecha_fin: date,
    recursos_dir: Path,
) -> Dict[str, Path]:
    """Extrae datos de IZARO y los guarda como CSVs. Wrapper de compatibilidad."""
    rows = extraer_datos(cfg, fecha_inicio, fecha_fin)
    if not rows:
        return {}
    return datos_a_csvs(rows, recursos_dir)
