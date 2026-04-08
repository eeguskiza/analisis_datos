"""Calcula metricas OEE a partir de datos almacenados en BD."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from api.database import Ciclo, DatosProduccion, Ejecucion

from OEE.oee_secciones.main import (
    SHIFT_LABELS,
    MIN_PIEZAS_OEE,
    clasificar_incidencia,
    normalizar_proceso,
    normalize_ref,
    clamp_pct,
    convertir_raw_a_metricas,
    crear_raw_metricas,
    dividir_en_turnos,
    calcular_solapamiento,
    parse_time_value,
    determinar_turno,
)


def _build_ciclos_lookup(db: Session) -> Dict[str, Dict[str, float]]:
    """Construye {maquina: {referencia: piezas_por_hora}} desde la tabla ciclos."""
    lookup: Dict[str, Dict[str, float]] = {}
    for c in db.query(Ciclo).all():
        maq = (c.maquina or "").strip().lower()
        ref = normalize_ref(c.referencia)
        if maq and ref and c.tiempo_ciclo and c.tiempo_ciclo > 0:
            lookup.setdefault(maq, {})[ref] = c.tiempo_ciclo
    return lookup


def _parse_record_datetimes(r) -> tuple[Optional[datetime], Optional[datetime], float]:
    """Convierte fecha + h_ini/h_fin de un registro BD a datetimes y horas."""
    fecha = r.fecha
    if isinstance(fecha, str):
        try:
            fecha = datetime.strptime(fecha, "%Y-%m-%d").date()
        except ValueError:
            return None, None, 0.0

    start_time = parse_time_value(r.h_ini)
    end_time = parse_time_value(r.h_fin)

    if not fecha or not start_time:
        return None, None, 0.0

    start_dt = datetime.combine(fecha, start_time)
    end_dt = None
    if end_time:
        end_dt = datetime.combine(fecha, end_time)
        if end_dt < start_dt:
            end_dt += timedelta(days=1)
        elif end_dt == start_dt:
            end_dt = None

    horas = (end_dt - start_dt).total_seconds() / 3600.0 if start_dt and end_dt else 0.0
    return start_dt, end_dt, horas


def _metrics_to_json(m: Dict[str, float]) -> dict:
    """Convierte un dict de metricas a formato JSON con redondeo."""
    return {
        "horas_brutas": round(m.get("horas_brutas", 0), 4),
        "horas_disponible": round(m.get("horas_disponible", 0), 4),
        "horas_operativo": round(m.get("horas_operativo", 0), 4),
        "horas_preparacion": round(m.get("horas_preparacion", 0), 4),
        "horas_indisponibilidad": round(m.get("horas_indisponibilidad", 0), 4),
        "horas_paros": round(m.get("horas_paros", 0), 4),
        "tiempo_ideal": round(m.get("tiempo_ideal", 0), 4),
        "perdidas_rend": round(m.get("perdidas_rend", 0), 4),
        "piezas_totales": round(m.get("piezas_totales", 0)),
        "piezas_malas": round(m.get("piezas_malas", 0)),
        "piezas_recuperadas": round(m.get("piezas_recuperadas", 0)),
        "buenas_finales": round(m.get("buenas_finales", 0)),
        "disponibilidad_pct": round(m.get("disponibilidad_pct", 0), 2),
        "rendimiento_pct": round(m.get("rendimiento_pct", 0), 2),
        "calidad_pct": round(m.get("calidad_pct", 0), 2),
        "oee_pct": round(m.get("oee_pct", 0), 2),
    }


def _sum_raw(raws: List[Dict[str, float]]) -> Dict[str, float]:
    """Suma una lista de raw metricas."""
    total = crear_raw_metricas()
    for r in raws:
        for k in total:
            total[k] += r.get(k, 0.0)
    return total


def _process_machine(recurso: str, rows, ciclos_lookup: Dict[str, Dict[str, float]]) -> dict:
    """Procesa todos los registros de una maquina y devuelve metricas."""
    ciclos_maq = ciclos_lookup.get(recurso.lower(), {})

    # Parse all records
    parsed = []
    for r in rows:
        start_dt, end_dt, horas = _parse_record_datetimes(r)
        proceso = normalizar_proceso(r.proceso)
        incidencia = (r.incidencia or "").strip()
        referencia = normalize_ref(r.referencia)
        parsed.append({
            "start_dt": start_dt, "end_dt": end_dt, "horas": horas,
            "tiempo": r.tiempo or 0.0,
            "proceso": proceso, "incidencia": incidencia,
            "cantidad": r.cantidad or 0.0, "malas": r.malas or 0.0,
            "recuperadas": r.recuperadas or 0.0, "referencia": referencia,
        })

    prod_prep = [p for p in parsed if p["proceso"] in ("produccion", "preparacion")]
    inc_disp = [p for p in parsed if p["proceso"] == "incidencias"
                and clasificar_incidencia(p["incidencia"]) == "disponibilidad"]
    inc_paros = [p for p in parsed if p["proceso"] == "incidencias"
                 and clasificar_incidencia(p["incidencia"]) == "paros"]

    shift_raw = {s: crear_raw_metricas() for s in SHIFT_LABELS}
    daily_raw: Dict[date, Dict[str, float]] = defaultdict(crear_raw_metricas)
    ref_stats: Dict[str, dict] = {}

    # Incidencias para resumen
    incidencias_resumen: Dict[str, float] = defaultdict(float)

    for inc in inc_disp + inc_paros:
        if inc["horas"] > 0:
            incidencias_resumen[inc["incidencia"] or "Sin descripcion"] += inc["horas"]

    for reg in prod_prep:
        if not reg["start_dt"] or not reg["end_dt"]:
            continue

        # Overlap con incidencias
        t_indisp = sum(
            calcular_solapamiento(reg["start_dt"], reg["end_dt"], i["start_dt"], i["end_dt"])
            for i in inc_disp if i["start_dt"] and i["end_dt"]
        )
        t_paros = sum(
            calcular_solapamiento(reg["start_dt"], reg["end_dt"], i["start_dt"], i["end_dt"])
            for i in inc_paros if i["start_dt"] and i["end_dt"]
        )

        ciclo = ciclos_maq.get(reg["referencia"])
        segmentos = dividir_en_turnos(reg["start_dt"], reg["horas"])
        if not segmentos:
            start_time = reg["start_dt"].time() if reg["start_dt"] else None
            turno_def = determinar_turno(start_time)
            day_def = reg["start_dt"].date() if reg["start_dt"] else datetime.today().date()
            segmentos = [(turno_def, day_def, reg["horas"])]

        total_seg_h = sum(h for _, _, h in segmentos) or 1.0
        f_indisp = t_indisp / reg["horas"] if reg["horas"] > 0 else 0.0
        f_paros = t_paros / reg["horas"] if reg["horas"] > 0 else 0.0

        for turno, day, h_bruto in segmentos:
            peso = h_bruto / total_seg_h if total_seg_h > 0 else 0.0
            pzas = reg["cantidad"] * peso
            malas = reg["malas"] * peso
            recu = reg["recuperadas"] * peso

            for bucket in (shift_raw[turno], daily_raw[day]):
                bucket["piezas_totales"] += pzas
                bucket["piezas_malas"] += malas
                bucket["piezas_recuperadas"] += recu
                bucket["horas_indisponibilidad"] += h_bruto * f_indisp
                bucket["horas_paros"] += h_bruto * f_paros

                if reg["proceso"] == "produccion":
                    bucket["horas_produccion"] += h_bruto
                    if ciclo and ciclo > 0 and pzas > 0:
                        bucket["tiempo_ideal"] += pzas / ciclo
                else:
                    bucket["horas_preparacion"] += h_bruto

            # Ref stats
            if reg["proceso"] == "produccion" and (pzas > 0 or h_bruto > 0):
                h_neto = reg["tiempo"] * peso
                entry = ref_stats.setdefault(reg["referencia"], {
                    "ciclo_ideal": ciclo if ciclo and ciclo > 0 else None,
                    "dias": {},
                })
                if ciclo and ciclo > 0:
                    entry["ciclo_ideal"] = ciclo
                d = entry["dias"].setdefault(day, {"piezas": 0.0, "horas_brutas": 0.0, "horas_netas": 0.0})
                d["piezas"] += pzas
                d["horas_brutas"] += h_bruto
                d["horas_netas"] += h_neto

    # Totales de la maquina = suma de todos los turnos
    total_raw = _sum_raw(list(shift_raw.values()))
    total_metrics = convertir_raw_a_metricas(total_raw)

    # Por turno
    turnos = {}
    for s in SHIFT_LABELS:
        turnos[s] = _metrics_to_json(convertir_raw_a_metricas(shift_raw[s]))

    # Por dia
    resumen_diario = []
    for day in sorted(daily_raw):
        dm = convertir_raw_a_metricas(daily_raw[day])
        entry = _metrics_to_json(dm)
        entry["fecha"] = day.isoformat()
        resumen_diario.append(entry)

    # Ref stats JSON
    ref_json = []
    for ref, data in sorted(ref_stats.items()):
        total_pzas = sum(d["piezas"] for d in data["dias"].values())
        total_h = sum(d["horas_brutas"] for d in data["dias"].values())
        # Ciclo real = piezas / (horas - paros solapados con esa ref)
        # Simplificacion: usar horas_netas si disponible, sino horas_brutas
        total_h_neto = sum(d["horas_netas"] for d in data["dias"].values())
        ciclo_real = total_pzas / total_h_neto if total_h_neto > 0 else (
            total_pzas / total_h if total_h > 0 else 0
        )
        ref_json.append({
            "referencia": ref,
            "ciclo_ideal": data["ciclo_ideal"],
            "ciclo_real": round(ciclo_real, 1),
            "cantidad": round(total_pzas),
            "horas": round(total_h, 2),
        })

    # Incidencias
    inc_json = []
    for nombre, horas in sorted(incidencias_resumen.items(), key=lambda x: -x[1]):
        inc_json.append({"nombre": nombre, "horas": round(horas, 3)})

    result = _metrics_to_json(total_metrics)
    result["nombre"] = recurso.upper()
    result["turnos"] = turnos
    result["resumen_diario"] = resumen_diario
    result["ref_stats"] = ref_json
    result["incidencias"] = inc_json
    return result


def calcular_metrics_ejecucion(db: Session, ejec_id: int) -> dict:
    """Calcula todas las metricas OEE para una ejecucion dada."""
    ejec = db.get(Ejecucion, ejec_id)
    if not ejec:
        return {"error": "Ejecucion no encontrada"}

    datos = (
        db.query(DatosProduccion)
        .filter(DatosProduccion.ejecucion_id == ejec_id)
        .order_by(DatosProduccion.recurso, DatosProduccion.fecha, DatosProduccion.h_ini)
        .all()
    )
    if not datos:
        return {"error": "Sin datos para esta ejecucion"}

    ciclos_lookup = _build_ciclos_lookup(db)

    # Agrupar por recurso
    by_recurso: Dict[str, list] = defaultdict(list)
    for d in datos:
        by_recurso[d.recurso].append(d)

    # Agrupar por seccion
    seccion_map: Dict[str, str] = {}
    for d in datos:
        seccion_map[d.recurso] = d.seccion or "GENERAL"

    by_seccion: Dict[str, List[str]] = defaultdict(list)
    for recurso in by_recurso:
        sec = seccion_map.get(recurso, "GENERAL")
        if recurso not in by_seccion[sec]:
            by_seccion[sec].append(recurso)

    # Procesar cada maquina
    maquinas_metrics: Dict[str, dict] = {}
    for recurso, rows in by_recurso.items():
        maquinas_metrics[recurso] = _process_machine(recurso, rows, ciclos_lookup)

    # Construir respuesta por seccion
    secciones = {}
    for seccion, recursos in sorted(by_seccion.items()):
        maquinas_list = [maquinas_metrics[r] for r in sorted(recursos)]

        # Totales de seccion
        keys = ["horas_brutas", "horas_disponible", "horas_operativo", "horas_preparacion",
                "horas_indisponibilidad", "horas_paros", "tiempo_ideal", "perdidas_rend",
                "piezas_totales", "piezas_malas", "piezas_recuperadas", "buenas_finales"]
        totales_raw = crear_raw_metricas()
        for m in maquinas_list:
            for k in ["horas_produccion", "horas_preparacion", "horas_indisponibilidad",
                       "horas_paros", "tiempo_ideal", "piezas_totales", "piezas_malas",
                       "piezas_recuperadas"]:
                totales_raw[k] += m.get(k, 0) if k in m else 0

        # Recalcular desde raw para consistencia
        # Sumar raw values directamente de maquinas turnos
        sec_shift_raw = {s: crear_raw_metricas() for s in SHIFT_LABELS}
        for maq in maquinas_list:
            for s in SHIFT_LABELS:
                t = maq.get("turnos", {}).get(s, {})
                # Necesitamos raw - recalcular desde los totales de turno
                for k in sec_shift_raw[s]:
                    # Mapear keys de metricas a raw
                    if k == "horas_produccion":
                        sec_shift_raw[s][k] += t.get("horas_brutas", 0) - t.get("horas_preparacion", 0)
                    elif k in t:
                        sec_shift_raw[s][k] += t.get(k, 0)

        totales_metrics = convertir_raw_a_metricas(_sum_raw(list(sec_shift_raw.values())))
        totales_turnos = {}
        for s in SHIFT_LABELS:
            totales_turnos[s] = _metrics_to_json(convertir_raw_a_metricas(sec_shift_raw[s]))

        # Resumen diario de seccion
        all_days = set()
        for m in maquinas_list:
            for entry in m.get("resumen_diario", []):
                all_days.add(entry["fecha"])
        resumen_diario = []
        for day_str in sorted(all_days):
            day_raw = crear_raw_metricas()
            for m in maquinas_list:
                for entry in m.get("resumen_diario", []):
                    if entry["fecha"] == day_str:
                        day_raw["horas_produccion"] += entry.get("horas_brutas", 0) - entry.get("horas_preparacion", 0)
                        day_raw["horas_preparacion"] += entry.get("horas_preparacion", 0)
                        day_raw["horas_indisponibilidad"] += entry.get("horas_indisponibilidad", 0)
                        day_raw["horas_paros"] += entry.get("horas_paros", 0)
                        day_raw["tiempo_ideal"] += entry.get("tiempo_ideal", 0)
                        day_raw["piezas_totales"] += entry.get("piezas_totales", 0)
                        day_raw["piezas_malas"] += entry.get("piezas_malas", 0)
                        day_raw["piezas_recuperadas"] += entry.get("piezas_recuperadas", 0)
            dm = convertir_raw_a_metricas(day_raw)
            entry = _metrics_to_json(dm)
            entry["fecha"] = day_str
            resumen_diario.append(entry)

        # Incidencias de seccion
        all_inc: Dict[str, float] = defaultdict(float)
        for m in maquinas_list:
            for inc in m.get("incidencias", []):
                all_inc[inc["nombre"]] += inc["horas"]
        inc_json = [{"nombre": n, "horas": round(h, 3)} for n, h in sorted(all_inc.items(), key=lambda x: -x[1])]

        secciones[seccion] = {
            "totales": _metrics_to_json(totales_metrics),
            "totales_turnos": totales_turnos,
            "maquinas": maquinas_list,
            "resumen_diario": resumen_diario,
            "incidencias": inc_json,
        }

    return {
        "ejec_id": ejec_id,
        "fecha_inicio": ejec.fecha_inicio,
        "fecha_fin": ejec.fecha_fin,
        "secciones": secciones,
    }
