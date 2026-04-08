"""Historial de ejecuciones del pipeline."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from api.database import DatosProduccion, Ejecucion, InformeMeta, get_db

router = APIRouter(prefix="/historial", tags=["historial"])


@router.get("")
def listar(limit: int = 50, db: Session = Depends(get_db)):
    """Ultimas ejecuciones."""
    rows = (
        db.query(Ejecucion)
        .order_by(Ejecucion.created_at.desc())
        .limit(limit)
        .all()
    )
    # Contar registros de datos por ejecución
    counts = dict(
        db.query(DatosProduccion.ejecucion_id, func.count(DatosProduccion.id))
        .group_by(DatosProduccion.ejecucion_id)
        .all()
    )
    return {"ejecuciones": [
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
    ]}


@router.get("/{ejec_id}")
def detalle(ejec_id: int, db: Session = Depends(get_db)):
    """Detalle de una ejecucion: log + PDFs."""
    ejec = db.get(Ejecucion, ejec_id)
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
def metrics(ejec_id: int, db: Session = Depends(get_db)):
    """Calcula metricas OEE interactivas desde datos almacenados."""
    ejec = db.get(Ejecucion, ejec_id)
    if not ejec:
        raise HTTPException(404, "Ejecucion no encontrada")

    from api.services.metrics import calcular_metrics_ejecucion
    result = calcular_metrics_ejecucion(db, ejec_id)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.post("/{ejec_id}/regenerar")
def regenerar(ejec_id: int, db: Session = Depends(get_db)):
    """Regenera PDFs a partir de datos almacenados en BD."""
    ejec = db.get(Ejecucion, ejec_id)
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
def borrar(ejec_id: int, db: Session = Depends(get_db)):
    """Borra una ejecucion, sus datos y sus informes_meta."""
    ejec = db.get(Ejecucion, ejec_id)
    if not ejec:
        raise HTTPException(404, "Ejecucion no encontrada")
    db.query(DatosProduccion).filter_by(ejecucion_id=ejec_id).delete()
    db.query(InformeMeta).filter_by(ejecucion_id=ejec_id).delete()
    db.delete(ejec)
    db.commit()
    return {"ok": True}
