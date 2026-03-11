"""
Interfaz web local para extraer datos del MES (dbizaro) y generar informes OEE.

Arrancar con:  python app.py
Abrir en:      http://127.0.0.1:5000

Configuración de conexión → editar data/db_config.json (ver data/db_config.example.json)
"""
from __future__ import annotations

import io
import traceback
from contextlib import redirect_stdout
from datetime import date, datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from OEE.db.connector import (
    explorar_columnas_fmesdtc,
    extraer_y_guardar_csv,
    load_config,
    save_config,
    test_conexion,
)
from OEE.disponibilidad.main import generar_informes_disponibilidad
from OEE.rendimiento.main import generar_informes_rendimiento
from OEE.calidad.main import generar_informes_calidad
from OEE.oee_secciones.main import generar_informes_oee_secciones

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RECURSOS_DIR = DATA_DIR / "recursos"
INFORMES_DIR = BASE_DIR / "informes"

app = Flask(__name__)
app.secret_key = "mes-oee-internal-tool"


# ---------------------------------------------------------------------------
# Rutas principales
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    cfg = load_config()
    today = date.today().isoformat()
    return render_template("index.html", cfg=cfg, today=today)


# ---------------------------------------------------------------------------
# Estado de conexión (lo llama la web nada más cargar la página)
# ---------------------------------------------------------------------------

@app.route("/status")
def status():
    cfg = load_config()
    msg = test_conexion(cfg)
    ok = msg.startswith("OK")
    return jsonify({
        "ok": ok,
        "mensaje": msg,
        "server": cfg.get("server", ""),
        "database": cfg.get("database", ""),
    })


# ---------------------------------------------------------------------------
# Explorar columnas (para identificar el campo Refer.)
# ---------------------------------------------------------------------------

@app.route("/config/explorar", methods=["POST"])
def explorar_bd():
    cfg = load_config()
    cols = explorar_columnas_fmesdtc(cfg)
    return jsonify({"columnas": cols})


# ---------------------------------------------------------------------------
# Guardar solo la lista de recursos (la web no toca credenciales)
# ---------------------------------------------------------------------------

@app.route("/config/recursos", methods=["POST"])
def guardar_recursos():
    data = request.get_json(force=True)
    cfg = load_config()
    cfg["recursos"] = data.get("recursos", cfg["recursos"])
    save_config(cfg)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Ejecución del pipeline
# ---------------------------------------------------------------------------

@app.route("/run", methods=["POST"])
def run():
    data = request.get_json(force=True)

    try:
        fecha_inicio = date.fromisoformat(data["fecha_inicio"])
        fecha_fin = date.fromisoformat(data["fecha_fin"])
    except (KeyError, ValueError) as exc:
        return jsonify({"ok": False, "log": [f"Fechas inválidas: {exc}"], "pdfs": []})

    cfg = load_config()
    log: list[str] = []

    try:
        # 1. Extraer datos de la BD
        log.append(f"Conectando a {cfg.get('server')}:{cfg.get('port')} …")
        generados = extraer_y_guardar_csv(cfg, fecha_inicio, fecha_fin, RECURSOS_DIR)

        if not generados:
            return jsonify({
                "ok": False,
                "log": log + ["Sin datos para el período/recursos indicados."],
                "pdfs": [],
            })

        for nombre, path in generados.items():
            log.append(f"✓ CSV generado: {nombre}  ({path.name})")

        # 2. Ejecutar pipeline de informes
        log.append("Generando informes OEE …")
        run_dir = INFORMES_DIR / datetime.now().strftime("%Y-%m-%d")
        run_dir.mkdir(parents=True, exist_ok=True)

        logo_path = DATA_DIR / "ecs-logo.png"
        logo: Path | None = logo_path if logo_path.exists() else None

        buf = io.StringIO()
        with redirect_stdout(buf):
            generar_informes_disponibilidad(
                data_dir=DATA_DIR, output_dir=run_dir, logo_path=logo
            )
            generar_informes_rendimiento(
                data_dir=DATA_DIR, output_dir=run_dir, logo_path=logo
            )
            generar_informes_calidad(
                data_dir=DATA_DIR, output_dir=run_dir, logo_path=logo
            )
            generar_informes_oee_secciones(
                data_dir=DATA_DIR, output_dir=run_dir, logo_path=logo
            )

        pipeline_out = buf.getvalue().strip()
        if pipeline_out:
            log.extend(pipeline_out.splitlines())

        # 3. Recopilar PDFs generados
        pdfs: list[str] = []
        if run_dir.exists():
            for pdf in sorted(run_dir.rglob("*.pdf")):
                pdfs.append(str(pdf.relative_to(BASE_DIR)))

        log.append(f"✓ Completado. {len(pdfs)} PDF(s) generados.")
        return jsonify({"ok": True, "log": log, "pdfs": pdfs})

    except Exception as exc:
        log.append(f"ERROR: {exc}")
        log.extend(traceback.format_exc().splitlines())
        return jsonify({"ok": False, "log": log, "pdfs": []})


# ---------------------------------------------------------------------------
# Servir PDFs generados
# ---------------------------------------------------------------------------

@app.route("/informe/<path:filepath>")
def servir_informe(filepath):
    full_path = BASE_DIR / filepath
    if full_path.exists() and full_path.suffix == ".pdf":
        return send_file(full_path, mimetype="application/pdf")
    return "Fichero no encontrado", 404


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Verificar conexión al arrancar
    cfg = load_config()
    if not cfg.get("server"):
        print("⚠  Sin configurar: edita data/db_config.json con los datos de conexión.")
        print("   (Copia data/db_config.example.json como punto de partida)")
    else:
        print(f"Probando conexión a {cfg['server']} …", end=" ", flush=True)
        resultado = test_conexion(cfg)
        print("OK" if resultado.startswith("OK") else resultado)

    print("Abriendo interfaz en http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)
