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
    SELECT dd010, dd020, dd030, dd040, dd050,
           SUM(COALESCE(CAST(dd080 AS FLOAT), 0)) AS malas,
           SUM(COALESCE(CAST(dd090 AS FLOAT), 0)) AS recuperadas
    FROM admuser.fmesddf
    GROUP BY dd010, dd020, dd030, dd040, dd050
) AS def
    ON  RTRIM(def.dd010) = RTRIM(dtc.dt020)
    AND def.dd020 = dtc.dt030
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
