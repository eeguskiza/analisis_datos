"""Orquesta la extraccion de datos + generacion de informes OEE."""
from __future__ import annotations

import csv as csv_mod
import io
import re
from contextlib import redirect_stdout
from datetime import date, datetime
from pathlib import Path
from typing import Generator

from api.config import settings
from api.database import Ciclo, Ejecucion, InformeMeta, SessionLocal
from api.services import db as db_service

from OEE.disponibilidad.main import generar_informes_disponibilidad
from OEE.rendimiento.main import generar_informes_rendimiento
from OEE.calidad.main import generar_informes_calidad
from OEE.oee_secciones.main import generar_informes_oee_secciones
from OEE.utils.excel_import import procesar_excels


_MODULE_MAP = {
    "disponibilidad": ("Disponibilidad", generar_informes_disponibilidad),
    "rendimiento": ("Rendimiento", generar_informes_rendimiento),
    "calidad": ("Calidad", generar_informes_calidad),
    "oee_secciones": ("OEE Secciones", generar_informes_oee_secciones),
}


def _sync_ciclos_to_csv() -> None:
    """Exporta la tabla ciclos a ciclos.csv para que los modulos OEE lo lean."""
    with SessionLocal() as db:
        rows = db.query(Ciclo).order_by(Ciclo.maquina, Ciclo.referencia).all()
        if not rows:
            return
        path = settings.ciclos_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=["maquina", "referencia", "tiempo_ciclo"])
            writer.writeheader()
            for r in rows:
                writer.writerow({"maquina": r.maquina, "referencia": r.referencia, "tiempo_ciclo": r.tiempo_ciclo})


def _parse_pdf_metadata(pdf_path: str, fecha_str: str) -> dict:
    """Extrae seccion, maquina y modulo de la ruta de un PDF."""
    parts = pdf_path.replace("\\", "/").split("/")
    # Typical: informes/2026-03-31/LINEAS/luk1/luk1_disponibilidad.pdf
    seccion = ""
    maquina = ""
    modulo = ""
    if len(parts) >= 3:
        seccion = parts[1] if len(parts) > 2 else ""  # After date
    if len(parts) >= 4:
        maquina = parts[2] if len(parts) > 3 else ""
    filename = parts[-1].replace(".pdf", "")
    for mod_key in _MODULE_MAP:
        if mod_key in filename.lower():
            modulo = mod_key
            break
    if "oee_seccion" in filename.lower():
        modulo = "oee_secciones"
    return {"fecha": fecha_str, "seccion": seccion, "maquina": maquina, "modulo": modulo}


def run_pipeline(
    fecha_inicio: date,
    fecha_fin: date,
    modulos: list[str] | None = None,
    source: str = "db",
) -> Generator[str, None, None]:
    """
    Genera mensajes de log paso a paso (para SSE).
    Persiste la ejecucion y los informes generados en la BD local.
    """
    if modulos is None:
        modulos = list(_MODULE_MAP.keys())

    run_dir = settings.informes_dir / datetime.now().strftime("%Y-%m-%d")
    run_dir.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    status = "completed"

    # ── Crear ejecucion en BD ─────────────────────────────────────────────
    db = SessionLocal()
    ejec = Ejecucion(
        fecha_inicio=fecha_inicio.isoformat(),
        fecha_fin=fecha_fin.isoformat(),
        source=source,
        status="running",
        modulos=",".join(modulos),
    )
    db.add(ejec)
    db.commit()
    db.refresh(ejec)
    ejec_id = ejec.id

    def _log(msg: str):
        log_lines.append(msg)

    # ── 0. Sync ciclos BD → CSV ───────────────────────────────────────────
    try:
        _sync_ciclos_to_csv()
        _log("Ciclos sincronizados a CSV.")
        yield "Ciclos sincronizados."
    except Exception as exc:
        _log(f"WARN sync ciclos: {exc}")

    # ── 1. Obtener datos ──────────────────────────────────────────────────
    if source == "db":
        cfg = db_service.get_config()
        msg = f"Conectando a {cfg.get('server', '?')}:{cfg.get('port', '1433')} ..."
        _log(msg)
        yield msg
        try:
            generados = db_service.extract_csvs(fecha_inicio, fecha_fin)
            if not generados:
                msg = "ERROR: Sin datos para el periodo/recursos indicados."
                _log(msg)
                yield msg
                status = "error"
                _finalize(db, ejec_id, status, log_lines, 0)
                db.close()
                return
            for nombre, path in generados.items():
                msg = f"CSV generado: {nombre} ({path.name})"
                _log(msg)
                yield msg
        except Exception as exc:
            msg = f"ERROR extraccion: {exc}"
            _log(msg)
            yield msg
            status = "error"
            _finalize(db, ejec_id, status, log_lines, 0)
            db.close()
            return

    elif source == "excel":
        msg = "Procesando ficheros Excel ..."
        _log(msg)
        yield msg
        try:
            procesar_excels(settings.data_dir)
            _log("Excels procesados.")
            yield "Excels procesados."
        except Exception as exc:
            msg = f"ERROR procesando excels: {exc}"
            _log(msg)
            yield msg
            status = "error"
            _finalize(db, ejec_id, status, log_lines, 0)
            db.close()
            return
    else:
        msg = "Usando CSVs existentes en data/recursos/."
        _log(msg)
        yield msg

    # ── 2. Ejecutar modulos OEE ───────────────────────────────────────────
    logo = settings.logo_path

    for mod_key in modulos:
        entry = _MODULE_MAP.get(mod_key)
        if not entry:
            msg = f"Modulo desconocido: {mod_key}"
            _log(msg)
            yield msg
            continue

        label, func = entry
        msg = f"Generando {label} ..."
        _log(msg)
        yield msg

        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                func(data_dir=settings.data_dir, output_dir=run_dir, logo_path=logo)
            stdout_text = buf.getvalue().strip()
            if stdout_text:
                for line in stdout_text.splitlines():
                    _log(f"  {line}")
                    yield f"  {line}"
            msg = f"{label} completado."
            _log(msg)
            yield msg
        except Exception as exc:
            msg = f"ERROR en {label}: {exc}"
            _log(msg)
            yield msg
            status = "error"

    # ── 3. Recopilar PDFs ─────────────────────────────────────────────────
    pdfs: list[str] = []
    if run_dir.exists():
        for pdf in sorted(run_dir.rglob("*.pdf")):
            pdfs.append(str(pdf.relative_to(settings.informes_dir)))

    # Persistir informes_meta
    fecha_str = datetime.now().strftime("%Y-%m-%d")
    for pdf_rel in pdfs:
        meta = _parse_pdf_metadata(pdf_rel, fecha_str)
        db.add(InformeMeta(
            ejecucion_id=ejec_id,
            fecha=meta["fecha"],
            seccion=meta["seccion"],
            maquina=meta["maquina"],
            modulo=meta["modulo"],
            pdf_path=pdf_rel,
        ))

    _finalize(db, ejec_id, status, log_lines, len(pdfs))
    db.close()

    # SSE final: PDFs con ruta relativa a informes/
    pdfs_full = [f"informes/{p}" for p in pdfs]
    yield f"DONE:{len(pdfs)}:" + "|".join(pdfs_full)


def _finalize(db, ejec_id: int, status: str, log_lines: list[str], n_pdfs: int) -> None:
    """Actualiza la ejecucion en BD."""
    ejec = db.query(Ejecucion).get(ejec_id)
    if ejec:
        ejec.status = status
        ejec.log = "\n".join(log_lines)
        ejec.n_pdfs = n_pdfs
    db.commit()
