"""Historial de ejecuciones del pipeline."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.database import Ejecucion, InformeMeta, get_db

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
    return {"ejecuciones": [
        {
            "id": e.id,
            "fecha_inicio": e.fecha_inicio,
            "fecha_fin": e.fecha_fin,
            "source": e.source,
            "status": e.status,
            "modulos": e.modulos,
            "n_pdfs": e.n_pdfs,
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }
        for e in rows
    ]}


@router.get("/{ejec_id}")
def detalle(ejec_id: int, db: Session = Depends(get_db)):
    """Detalle de una ejecucion: log + PDFs."""
    ejec = db.query(Ejecucion).get(ejec_id)
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


@router.delete("/{ejec_id}")
def borrar(ejec_id: int, db: Session = Depends(get_db)):
    """Borra una ejecucion y sus informes_meta (no borra PDFs del disco)."""
    ejec = db.query(Ejecucion).get(ejec_id)
    if not ejec:
        raise HTTPException(404, "Ejecucion no encontrada")
    db.query(InformeMeta).filter_by(ejecucion_id=ejec_id).delete()
    db.delete(ejec)
    db.commit()
    return {"ok": True}
