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

DEFAULT_CONFIG: dict = {
    "server": "",
    "port": "1433",
    "database": "dbizaro",
    "driver": "ODBC Driver 17 for SQL Server",
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

def _build_connection_string(cfg: dict) -> str:
    driver = cfg.get("driver") or "ODBC Driver 17 for SQL Server"
    server = cfg["server"]
    port = cfg.get("port") or "1433"
    database = cfg["database"]
    user = (cfg.get("user") or "").strip()
    password = cfg.get("password") or ""

    if user:
        return (
            f"DRIVER={{{driver}}};"
            f"SERVER={server},{port};"
            f"DATABASE={database};"
            f"UID={user};"
            f"PWD={password};"
        )
    return (
        f"DRIVER={{{driver}}};"
        f"SERVER={server},{port};"
        f"DATABASE={database};"
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
    dtc.dt090                                                              AS tiempo,
    dtc.dt150                                                              AS centro_trabajo,
    CASE dtc.dt110
        WHEN '0' THEN 'Producción'
        WHEN '1' THEN 'Preparación'
        WHEN '2' THEN 'Incidencia'
        ELSE ''
    END                                                                    AS proceso,
    COALESCE(RTRIM(CAST(inc.in020 AS NVARCHAR(MAX))), '')                 AS incidencia,
    COALESCE(dtc.dt130, 0)                                                 AS cantidad,
    COALESCE(def.malas, 0)                                                 AS malas,
    COALESCE(def.recuperadas, 0)                                           AS recuperadas
    {ref_select}
FROM admuser.fmesdtc AS dtc
LEFT JOIN (
    SELECT dd010, dd020, dd030, dd040, dd050,
           SUM(dd080) AS malas,
           SUM(dd090) AS recuperadas
    FROM admuser.fmesddf
    GROUP BY dd010, dd020, dd030, dd040, dd050
) AS def
    ON  def.dd010 = dtc.dt010
    AND def.dd020 = dtc.dt020
    AND def.dd030 = dtc.dt030
    AND def.dd040 = dtc.dt040
    AND def.dd050 = dtc.dt050
LEFT JOIN admuser.fmesinc AS inc
    ON  RTRIM(inc.in010) = RTRIM(dtc.dt120)
    AND inc.in000 = dtc.dt010
WHERE CONVERT(DATE, dtc.dt060) BETWEEN ? AND ?
  AND dtc.dt150 IN ({ct_placeholders})
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

    ct_codes = [int(r["centro_trabajo"]) for r in recursos_activos]
    ct_to_resource: Dict[int, str] = {
        int(r["centro_trabajo"]): r["nombre"] for r in recursos_activos
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
        df_r = df[df["_ct"] == ct_code].drop(columns=["_ct"]).copy()
        if df_r.empty:
            continue

        section = RESOURCE_SECTION_MAP.get(resource_name.lower(), "GENERAL")
        out_dir = recursos_dir / section
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{resource_name}-{fecha_str}.csv"
        df_r.to_csv(out_path, index=False, encoding="utf-8-sig")
        generated[resource_name] = out_path

    return generated
