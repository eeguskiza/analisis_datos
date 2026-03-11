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


# ---------------------------------------------------------------------------
# Consulta principal
# ---------------------------------------------------------------------------

_SQL_TEMPLATE = """\
SELECT
    dtc.dt060                                                              AS fecha,
    STUFF(RIGHT('0000' + CAST(dtc.dt080 AS VARCHAR(4)), 4), 3, 0, ':')   AS hora_inicio,
    STUFF(RIGHT('0000' + CAST(dtc.dt085 AS VARCHAR(4)), 4), 3, 0, ':')   AS hora_fin,
    COALESCE(TRY_CONVERT(FLOAT, dtc.dt090), 0)                            AS tiempo,
    RTRIM(CAST(dtc.dt150 AS NVARCHAR(50)))                                AS centro_trabajo,
    CASE dtc.dt110
        WHEN '0' THEN 'Producción'
        WHEN '1' THEN 'Preparación'
        WHEN '2' THEN 'Incidencia'
        ELSE ''
    END                                                                    AS proceso,
    COALESCE(RTRIM(CAST(inc.in020 AS NVARCHAR(MAX))), '')                 AS incidencia,
    COALESCE(TRY_CONVERT(FLOAT, dtc.dt130), 0)                            AS cantidad,
    COALESCE(def.malas, 0)                                                 AS malas,
    COALESCE(def.recuperadas, 0)                                           AS recuperadas
    {ref_select}
FROM admuser.fmesdtc AS dtc
LEFT JOIN (
    SELECT dd010, dd020, dd030, dd040, dd050,
           SUM(COALESCE(TRY_CONVERT(FLOAT, dd080), 0)) AS malas,
           SUM(COALESCE(TRY_CONVERT(FLOAT, dd090), 0)) AS recuperadas
    FROM admuser.fmesddf
    GROUP BY dd010, dd020, dd030, dd040, dd050
) AS def
    ON  RTRIM(CAST(def.dd010 AS NVARCHAR(100))) = RTRIM(CAST(dtc.dt010 AS NVARCHAR(100)))
    AND RTRIM(CAST(def.dd020 AS NVARCHAR(100))) = RTRIM(CAST(dtc.dt020 AS NVARCHAR(100)))
    AND RTRIM(CAST(def.dd030 AS NVARCHAR(100))) = RTRIM(CAST(dtc.dt030 AS NVARCHAR(100)))
    AND RTRIM(CAST(def.dd040 AS NVARCHAR(100))) = RTRIM(CAST(dtc.dt040 AS NVARCHAR(100)))
    AND RTRIM(CAST(def.dd050 AS NVARCHAR(100))) = RTRIM(CAST(dtc.dt050 AS NVARCHAR(100)))
LEFT JOIN admuser.fmesinc AS inc
    ON  RTRIM(inc.in010) = RTRIM(dtc.dt120)
    AND RTRIM(CAST(inc.in000 AS NVARCHAR(100))) = RTRIM(CAST(dtc.dt010 AS NVARCHAR(100)))
WHERE CONVERT(DATE, dtc.dt060) BETWEEN ? AND ?
  AND RTRIM(CAST(dtc.dt150 AS NVARCHAR(50))) IN ({ct_placeholders})
  {uf_filter}
ORDER BY dtc.dt150, dtc.dt060, dtc.dt080
"""


def extraer_y_guardar_csv(
    cfg: dict,
    fecha_inicio: date,
    fecha_fin: date,
    recursos_dir: Path,
) -> Dict[str, Path]:
    """
    Consulta la BD y genera un CSV por recurso activo en recursos_dir/<SECCION>/.

    Devuelve un dict {nombre_recurso: path_csv}.
    Lanza excepciones si hay error de conexión o la consulta falla.
    """
    if not PYODBC_AVAILABLE:
        raise RuntimeError(
            "pyodbc no está instalado. Ejecuta: pip install pyodbc"
        )

    recursos_activos = [r for r in cfg.get("recursos", []) if r.get("activo", True)]
    if not recursos_activos:
        raise ValueError("No hay recursos activos configurados.")

    ct_codes = [str(r["centro_trabajo"]).strip() for r in recursos_activos]
    ct_to_resource: Dict[str, str] = {
        str(r["centro_trabajo"]).strip(): r["nombre"] for r in recursos_activos
    }

    # Construir query
    ref_campo = (cfg.get("referencia_campo") or "").strip()
    ref_select = f", dtc.{ref_campo} AS referencia" if ref_campo else ", '' AS referencia"

    uf_code = (cfg.get("uf_code") or "").strip()
    uf_filter = "AND dtc.dt010 = ?" if uf_code else ""

    sql = _SQL_TEMPLATE.format(
        ref_select=ref_select,
        ct_placeholders=",".join(["?"] * len(ct_codes)),
        uf_filter=uf_filter,
    )

    params: list = [
        fecha_inicio.strftime("%Y-%m-%d"),
        fecha_fin.strftime("%Y-%m-%d"),
        *ct_codes,
    ]
    if uf_code:
        params.append(uf_code)

    conn = pyodbc.connect(_build_connection_string(cfg), timeout=30)
    try:
        df = pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()

    if df.empty:
        return {}

    # Renombrar columnas al formato que espera el pipeline
    df = df.rename(columns={
        "fecha":          "Fecha",
        "hora_inicio":    "H Ini",
        "hora_fin":       "F Fin",
        "tiempo":         "Tiempo",
        "proceso":        "Proceso",
        "incidencia":     "Incidencia",
        "cantidad":       "Cantidad",
        "malas":          "Malas",
        "recuperadas":    "Recu.",
        "referencia":     "Refer.",
        "centro_trabajo": "_ct",
    })

    fecha_str = (
        fecha_inicio.strftime("%Y%m%d")
        if fecha_inicio == fecha_fin
        else f"{fecha_inicio.strftime('%Y%m%d')}-{fecha_fin.strftime('%Y%m%d')}"
    )

    generated: Dict[str, Path] = {}

    for ct_code, resource_name in ct_to_resource.items():
        df_r = df[df["_ct"].astype(str).str.strip() == ct_code].drop(columns=["_ct"]).copy()
        if df_r.empty:
            continue

        section = RESOURCE_SECTION_MAP.get(resource_name.lower(), "GENERAL")
        out_dir = recursos_dir / section
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{resource_name}-{fecha_str}.csv"
        df_r.to_csv(out_path, index=False, encoding="utf-8-sig")
        generated[resource_name] = out_path

    return generated
