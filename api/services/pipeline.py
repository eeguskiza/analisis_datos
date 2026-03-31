"""Orquesta la extracción de datos + generación de informes OEE."""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from datetime import date, datetime
from pathlib import Path
from typing import Generator

from api.config import settings
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


def run_pipeline(
    fecha_inicio: date,
    fecha_fin: date,
    modulos: list[str] | None = None,
    source: str = "db",
) -> Generator[str, None, None]:
    """
    Genera mensajes de log paso a paso (para SSE).

    source: "db" → extrae de BD, "excel" → procesa excels, "csv_only" → usa CSVs existentes.
    """
    if modulos is None:
        modulos = list(_MODULE_MAP.keys())

    run_dir = settings.informes_dir / datetime.now().strftime("%Y-%m-%d")
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Obtener datos ──────────────────────────────────────────────────
    if source == "db":
        cfg = db_service.get_config()
        yield f"Conectando a {cfg.get('server', '?')}:{cfg.get('port', '1433')} ..."
        try:
            generados = db_service.extract_csvs(fecha_inicio, fecha_fin)
            if not generados:
                yield "ERROR: Sin datos para el periodo/recursos indicados."
                return
            for nombre, path in generados.items():
                yield f"CSV generado: {nombre} ({path.name})"
        except Exception as exc:
            yield f"ERROR extraccion: {exc}"
            return

    elif source == "excel":
        yield "Procesando ficheros Excel ..."
        try:
            procesar_excels(settings.data_dir)
            yield "Excels procesados."
        except Exception as exc:
            yield f"ERROR procesando excels: {exc}"
            return
    else:
        yield "Usando CSVs existentes en data/recursos/."

    # ── 2. Ejecutar modulos OEE ───────────────────────────────────────────
    logo = settings.logo_path

    for mod_key in modulos:
        entry = _MODULE_MAP.get(mod_key)
        if not entry:
            yield f"Modulo desconocido: {mod_key}"
            continue

        label, func = entry
        yield f"Generando {label} ..."

        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                func(data_dir=settings.data_dir, output_dir=run_dir, logo_path=logo)
            stdout_text = buf.getvalue().strip()
            if stdout_text:
                for line in stdout_text.splitlines():
                    yield f"  {line}"
            yield f"{label} completado."
        except Exception as exc:
            yield f"ERROR en {label}: {exc}"

    # ── 3. Recopilar PDFs ─────────────────────────────────────────────────
    pdfs: list[str] = []
    if run_dir.exists():
        for pdf in sorted(run_dir.rglob("*.pdf")):
            pdfs.append(str(pdf.relative_to(settings.project_root)))

    yield f"DONE:{len(pdfs)}:" + "|".join(pdfs)
