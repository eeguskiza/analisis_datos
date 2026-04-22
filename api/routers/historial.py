"""Historial de ejecuciones del pipeline - via EjecucionRepo + MetricaRepo.

Plan 03-03 Task 4.3: las queries ORM inline se encapsulan en
``EjecucionRepo`` y ``MetricaRepo`` (``nexo.data.repositories.app``).
El count aggregate de ``DatosProduccion`` se mantiene inline en el
handler (query compuesta con group_by, mas claro como transport logic
local).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func

from api.deps import DbApp
from nexo.data.models_app import DatosProduccion, InformeMeta, MetricaOEE
from nexo.data.repositories.app import EjecucionRepo, MetricaRepo
from nexo.services.auth import require_permission

router = APIRouter(
    prefix="/historial",
    tags=["historial"],
    dependencies=[Depends(require_permission("historial:read"))],
)


@router.get("")
def listar(db: DbApp, limit: int = 50):
    """Ultimas ejecuciones."""
    rows = EjecucionRepo(db).list_recent_orm(limit=limit)

    # Contar registros de datos por ejecución (join count - se queda
    # en el router como transport logic compuesta)
    counts = dict(
        db.query(DatosProduccion.ejecucion_id, func.count(DatosProduccion.id))
        .group_by(DatosProduccion.ejecucion_id)
        .all()
    )
    return {
        "ejecuciones": [
            {
                "id": e.id,
                "fecha_inicio": e.fecha_inicio,
                "fecha_fin": e.fecha_fin,
                "source": e.source,
                "status": e.status,
                "modulos": e.modulos,
                "n_pdfs": e.n_pdfs,
                "n_registros": counts.get(e.id, 0),
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in rows
        ]
    }


@router.get("/tendencias")
def tendencias(
    db: DbApp,
    recurso: str = Query(...),
    fecha_inicio: str = Query(None),
    fecha_fin: str = Query(None),
):
    """Metricas OEE diarias de un recurso a lo largo del tiempo."""
    # Delegamos a MetricaRepo.tendencias_recurso (devuelve DTOs MetricaRow
    # ya filtrados por fecha no nula + turno nulo).
    dtos = MetricaRepo(db).tendencias_recurso(
        recurso=recurso,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
    )

    # Deduplicar: para cada fecha, quedarse con la ejecucion mas reciente.
    # MetricaRow no expone buenas_finales/horas_* todos - necesitamos el
    # ORM para los campos extra. Re-query compacto sobre MetricaOEE filtrando
    # por las mismas condiciones y recurso.
    from datetime import date as _date

    query = (
        db.query(MetricaOEE)
        .filter(
            MetricaOEE.recurso.ilike(recurso),
            MetricaOEE.fecha.isnot(None),
            MetricaOEE.turno.is_(None),
        )
        .order_by(MetricaOEE.fecha)
    )
    if fecha_inicio:
        query = query.filter(MetricaOEE.fecha >= _date.fromisoformat(fecha_inicio))
    if fecha_fin:
        query = query.filter(MetricaOEE.fecha <= _date.fromisoformat(fecha_fin))
    rows = query.all()

    by_date: dict = {}
    for r in rows:
        key = r.fecha.isoformat() if hasattr(r.fecha, "isoformat") else str(r.fecha)
        if key not in by_date or r.ejecucion_id > by_date[key].ejecucion_id:
            by_date[key] = r

    data = []
    for key in sorted(by_date):
        r = by_date[key]
        data.append(
            {
                "fecha": key,
                "disponibilidad_pct": round(r.disponibilidad_pct or 0, 2),
                "rendimiento_pct": round(r.rendimiento_pct or 0, 2),
                "calidad_pct": round(r.calidad_pct or 0, 2),
                "oee_pct": round(r.oee_pct or 0, 2),
                "piezas_totales": r.piezas_totales or 0,
                "piezas_malas": r.piezas_malas or 0,
                "buenas_finales": r.buenas_finales or 0,
                "horas_brutas": round(r.horas_brutas or 0, 2),
                "horas_disponible": round(r.horas_disponible or 0, 2),
                "horas_operativo": round(r.horas_operativo or 0, 2),
            }
        )

    return {"recurso": recurso, "dias": data}


@router.get("/{ejec_id}")
def detalle(ejec_id: int, db: DbApp):
    """Detalle de una ejecucion: log + PDFs."""
    ejec = EjecucionRepo(db).get_orm_by_id(ejec_id)
    if not ejec:
        raise HTTPException(404, "Ejecucion no encontrada")

    informes = (
        db.query(InformeMeta)
        .filter_by(ejecucion_id=ejec_id)
        .order_by(InformeMeta.seccion, InformeMeta.maquina)
        .all()
    )

    return {
        "id": ejec.id,
        "fecha_inicio": ejec.fecha_inicio,
        "fecha_fin": ejec.fecha_fin,
        "source": ejec.source,
        "status": ejec.status,
        "modulos": ejec.modulos,
        "n_pdfs": ejec.n_pdfs,
        "log": ejec.log or "",
        "created_at": ejec.created_at.isoformat() if ejec.created_at else None,
        "informes": [
            {
                "id": i.id,
                "seccion": i.seccion,
                "maquina": i.maquina,
                "modulo": i.modulo,
                "pdf_path": i.pdf_path,
            }
            for i in informes
        ],
    }


@router.get("/{ejec_id}/metrics")
def metrics(ejec_id: int, db: DbApp):
    """Calcula metricas OEE interactivas desde datos almacenados."""
    ejec = EjecucionRepo(db).get_orm_by_id(ejec_id)
    if not ejec:
        raise HTTPException(404, "Ejecucion no encontrada")

    from api.services.metrics import calcular_metrics_ejecucion

    result = calcular_metrics_ejecucion(db, ejec_id)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.post("/{ejec_id}/regenerar")
def regenerar(ejec_id: int, db: DbApp):
    """Regenera PDFs a partir de datos almacenados en BD."""
    ejec = EjecucionRepo(db).get_orm_by_id(ejec_id)
    if not ejec:
        raise HTTPException(404, "Ejecucion no encontrada")

    n_datos = db.query(DatosProduccion).filter_by(ejecucion_id=ejec_id).count()
    if n_datos == 0:
        raise HTTPException(400, "No hay datos guardados para esta ejecución")

    from api.services.pipeline import generar_informes_desde_bd

    modulos = ejec.modulos.split(",") if ejec.modulos else None
    try:
        pdfs = generar_informes_desde_bd(ejec_id, modulos)
    except Exception as exc:
        raise HTTPException(500, f"Error regenerando: {exc}")

    return {"ok": True, "n_pdfs": len(pdfs), "pdfs": pdfs}


@router.delete("/{ejec_id}")
def borrar(ejec_id: int, db: DbApp):
    """Borra una ejecucion, sus datos y sus informes_meta."""
    ejec = EjecucionRepo(db).get_orm_by_id(ejec_id)
    if not ejec:
        raise HTTPException(404, "Ejecucion no encontrada")
    db.query(DatosProduccion).filter_by(ejecucion_id=ejec_id).delete()
    db.query(InformeMeta).filter_by(ejecucion_id=ejec_id).delete()
    db.delete(ejec)
    db.commit()
    return {"ok": True}
