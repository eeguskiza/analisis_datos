"""Orquesta la extraccion de datos + generacion de informes OEE."""
from __future__ import annotations

import csv as csv_mod
import io
import shutil
import tempfile
from contextlib import redirect_stdout
from datetime import date, datetime
from pathlib import Path
from typing import Generator

from api.config import settings
from api.database import Ciclo, DatosProduccion, Ejecucion, InformeMeta, SessionLocal
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


def _sync_ciclos_to_csv(target_dir: Path | None = None) -> None:
    """Exporta la tabla ciclos a ciclos.csv para que los modulos OEE lo lean."""
    with SessionLocal() as db:
        rows = db.query(Ciclo).order_by(Ciclo.maquina, Ciclo.referencia).all()
        if not rows:
            return
        path = (target_dir / "ciclos.csv") if target_dir else settings.ciclos_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=["maquina", "referencia", "tiempo_ciclo"])
            writer.writeheader()
            for r in rows:
                writer.writerow({"maquina": r.maquina, "referencia": r.referencia, "tiempo_ciclo": r.tiempo_ciclo})


def _save_datos_to_db(db, ejec_id: int, rows: list[dict]) -> int:
    """Guarda datos extraídos en la tabla datos_produccion. Devuelve nº de filas."""
    for r in rows:
        db.add(DatosProduccion(
            ejecucion_id=ejec_id,
            recurso=r["recurso"],
            seccion=r["seccion"],
            fecha=r["fecha"],
            h_ini=r["h_ini"],
            h_fin=r["h_fin"],
            tiempo=r["tiempo"],
            proceso=r["proceso"],
            incidencia=r["incidencia"],
            cantidad=r["cantidad"],
            malas=r["malas"],
            recuperadas=r["recuperadas"],
            referencia=r["referencia"],
        ))
    db.commit()
    return len(rows)


def _parse_pdf_metadata(pdf_path: str, fecha_str: str) -> dict:
    """Extrae seccion, maquina y modulo de la ruta de un PDF."""
    parts = pdf_path.replace("\\", "/").split("/")
    seccion = parts[0] if len(parts) >= 2 else ""
    maquina = parts[1] if len(parts) >= 3 else ""
    filename = parts[-1].replace(".pdf", "")
    modulo = ""
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
    recursos: list[str] | None = None,
) -> Generator[str, None, None]:
    """
    Genera mensajes de log paso a paso (para SSE).
    Extrae datos → guarda en BD → genera CSVs temporales → OEE → PDFs temporales.
    """
    if modulos is None:
        modulos = list(_MODULE_MAP.keys())

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

    # Directorio temporal para CSVs + PDFs
    tmp_root = Path(tempfile.mkdtemp(prefix="oee_"))
    tmp_data = tmp_root / "data"
    tmp_data.mkdir()
    tmp_informes = tmp_root / "informes"
    tmp_informes.mkdir()

    try:
        # ── 0. Sync ciclos BD → CSV temporal ─────────────────────────────
        try:
            _sync_ciclos_to_csv(target_dir=tmp_data)
            _log("Ciclos sincronizados.")
            yield "Ciclos sincronizados."
        except Exception as exc:
            _log(f"WARN sync ciclos: {exc}")

        # ── 1. Obtener datos ─────────────────────────────────────────────
        data_rows: list[dict] = []

        if source == "db":
            cfg = db_service.get_config()
            msg = f"Conectando a {cfg.get('server', '?')}:{cfg.get('port', '1433')} ..."
            _log(msg)
            yield msg
            try:
                data_rows = db_service.extract_data(fecha_inicio, fecha_fin, recursos=recursos)
                if not data_rows:
                    msg = "ERROR: Sin datos para el periodo/recursos indicados."
                    _log(msg)
                    yield msg
                    status = "error"
                    _finalize(db, ejec_id, status, log_lines, 0)
                    db.close()
                    return

                # Guardar datos en BD local
                n = _save_datos_to_db(db, ejec_id, data_rows)
                msg = f"{n} registros guardados en BD."
                _log(msg)
                yield msg

                # Escribir CSVs temporales para módulos OEE
                generados = db_service.write_csvs(data_rows, tmp_data / "recursos")
                for nombre, path in generados.items():
                    msg = f"CSV: {nombre} ({path.name})"
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
                # Para Excel, usamos data_dir real (no temporal)
                tmp_data = settings.data_dir
            except Exception as exc:
                msg = f"ERROR procesando excels: {exc}"
                _log(msg)
                yield msg
                status = "error"
                _finalize(db, ejec_id, status, log_lines, 0)
                db.close()
                return
        else:
            msg = "Usando CSVs existentes."
            _log(msg)
            yield msg
            tmp_data = settings.data_dir

        # ── 2. Ejecutar modulos OEE ──────────────────────────────────────
        logo = settings.logo_path
        for mod_key in modulos:
            entry = _MODULE_MAP.get(mod_key)
            if not entry:
                _log(f"Modulo desconocido: {mod_key}")
                yield f"Modulo desconocido: {mod_key}"
                continue

            label, func = entry
            msg = f"Generando {label} ..."
            _log(msg)
            yield msg

            buf = io.StringIO()
            try:
                with redirect_stdout(buf):
                    func(data_dir=tmp_data, output_dir=tmp_informes, logo_path=logo)
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

        # ── 3. Mover PDFs a directorio final ─────────────────────────────
        run_dir = settings.informes_dir / datetime.now().strftime("%Y-%m-%d")
        pdfs: list[str] = []

        if tmp_informes.exists():
            for pdf in sorted(tmp_informes.rglob("*.pdf")):
                rel = pdf.relative_to(tmp_informes)
                dest = run_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(pdf), str(dest))
                # Ruta relativa a informes_dir (incluye fecha)
                pdfs.append(str(dest.relative_to(settings.informes_dir)))

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

        yield f"DONE:{len(pdfs)}:" + "|".join(pdfs)

    finally:
        # Limpiar directorio temporal
        shutil.rmtree(tmp_root, ignore_errors=True)


def generar_informes_desde_bd(
    ejecucion_id: int,
    modulos: list[str] | None = None,
) -> list[str]:
    """
    Regenera PDFs a partir de datos almacenados en BD.
    Devuelve lista de rutas relativas a informes_dir.
    """
    if modulos is None:
        modulos = list(_MODULE_MAP.keys())

    db = SessionLocal()
    try:
        ejec = db.query(Ejecucion).get(ejecucion_id)
        if not ejec:
            raise ValueError(f"Ejecución {ejecucion_id} no encontrada")

        datos = db.query(DatosProduccion).filter(
            DatosProduccion.ejecucion_id == ejecucion_id
        ).all()
        if not datos:
            raise ValueError("No hay datos para esta ejecución")

        rows = [{
            "recurso": d.recurso, "seccion": d.seccion, "fecha": d.fecha,
            "h_ini": d.h_ini, "h_fin": d.h_fin, "tiempo": d.tiempo,
            "proceso": d.proceso, "incidencia": d.incidencia,
            "cantidad": d.cantidad, "malas": d.malas,
            "recuperadas": d.recuperadas, "referencia": d.referencia,
        } for d in datos]
    finally:
        db.close()

    tmp_root = Path(tempfile.mkdtemp(prefix="oee_regen_"))
    tmp_data = tmp_root / "data"
    tmp_data.mkdir()
    tmp_informes = tmp_root / "informes"
    tmp_informes.mkdir()

    try:
        _sync_ciclos_to_csv(target_dir=tmp_data)
        db_service.write_csvs(rows, tmp_data / "recursos")

        logo = settings.logo_path
        for mod_key in modulos:
            entry = _MODULE_MAP.get(mod_key)
            if not entry:
                continue
            _, func = entry
            buf = io.StringIO()
            with redirect_stdout(buf):
                func(data_dir=tmp_data, output_dir=tmp_informes, logo_path=logo)

        run_dir = settings.informes_dir / datetime.now().strftime("%Y-%m-%d")
        pdfs: list[str] = []
        for pdf in sorted(tmp_informes.rglob("*.pdf")):
            rel = pdf.relative_to(tmp_informes)
            dest = run_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(pdf), str(dest))
            pdfs.append(str(rel))
        return pdfs
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _finalize(db, ejec_id: int, status: str, log_lines: list[str], n_pdfs: int) -> None:
    """Actualiza la ejecucion en BD."""
    ejec = db.query(Ejecucion).get(ejec_id)
    if ejec:
        ejec.status = status
        ejec.log = "\n".join(log_lines)
        ejec.n_pdfs = n_pdfs
    db.commit()
