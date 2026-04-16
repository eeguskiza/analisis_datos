"""CRUD para ciclos (tiempos de ciclo ideales) — desde BBDD."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.database import Ciclo, Recurso, SECTION_MAP, get_db
from api.models import CicloRow, CiclosPayload
from api.services import db as mes_service

router = APIRouter(prefix="/ciclos", tags=["ciclos"])


@router.get("")
def listar(db: Session = Depends(get_db)):
    """Devuelve ciclos agrupados por seccion y maquina."""
    rows = db.query(Ciclo).order_by(Ciclo.maquina, Ciclo.referencia).all()

    # Flat list
    flat = [
        {"id": r.id, "maquina": r.maquina, "referencia": r.referencia, "tiempo_ciclo": r.tiempo_ciclo}
        for r in rows
    ]

    # Agrupado por seccion -> maquina
    grouped: dict[str, dict[str, list]] = {}
    for r in rows:
        sec = SECTION_MAP.get(r.maquina.lower(), "GENERAL")
        if sec not in grouped:
            grouped[sec] = {}
        if r.maquina not in grouped[sec]:
            grouped[sec][r.maquina] = []
        grouped[sec][r.maquina].append({
            "id": r.id, "referencia": r.referencia, "tiempo_ciclo": r.tiempo_ciclo,
        })

    return {"rows": flat, "grouped": grouped}


@router.put("")
def guardar(payload: CiclosPayload, db: Session = Depends(get_db)):
    """Reescribe todos los ciclos."""
    db.query(Ciclo).delete()
    for r in payload.rows:
        db.add(Ciclo(maquina=r.maquina, referencia=r.referencia, tiempo_ciclo=r.tiempo_ciclo))
    db.commit()
    return {"ok": True, "count": len(payload.rows)}


@router.post("/row")
def add_row(row: CicloRow, db: Session = Depends(get_db)):
    """Anade un ciclo."""
    exists = db.query(Ciclo).filter_by(maquina=row.maquina, referencia=row.referencia).first()
    if exists:
        raise HTTPException(409, "Ya existe esa combinacion maquina/referencia")
    db.add(Ciclo(maquina=row.maquina, referencia=row.referencia, tiempo_ciclo=row.tiempo_ciclo))
    db.commit()
    return {"ok": True}


@router.put("/row/{ciclo_id}")
def update_row(ciclo_id: int, row: CicloRow, db: Session = Depends(get_db)):
    """Actualiza un ciclo por ID."""
    ciclo = db.get(Ciclo, ciclo_id)
    if not ciclo:
        raise HTTPException(404, "Ciclo no encontrado")
    ciclo.maquina = row.maquina
    ciclo.referencia = row.referencia
    ciclo.tiempo_ciclo = row.tiempo_ciclo
    db.commit()
    return {"ok": True}


@router.delete("/row/{ciclo_id}")
def delete_row(ciclo_id: int, db: Session = Depends(get_db)):
    """Elimina un ciclo por ID."""
    ciclo = db.get(Ciclo, ciclo_id)
    if not ciclo:
        raise HTTPException(404, "Ciclo no encontrado")
    db.delete(ciclo)
    db.commit()
    return {"ok": True}


@router.post("/sync-csv")
def sync_to_csv(db: Session = Depends(get_db)):
    """Exporta la tabla ciclos a ciclos.csv (para que los modulos OEE lo lean)."""
    import csv as csv_mod
    from api.config import settings

    rows = db.query(Ciclo).order_by(Ciclo.maquina, Ciclo.referencia).all()
    path = settings.ciclos_path
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=["maquina", "referencia", "tiempo_ciclo"])
        writer.writeheader()
        for r in rows:
            writer.writerow({"maquina": r.maquina, "referencia": r.referencia, "tiempo_ciclo": r.tiempo_ciclo})

    return {"ok": True, "count": len(rows), "path": str(path)}


@router.get("/calcular/{nombre_recurso}")
def calcular_ciclos(
    nombre_recurso: str,
    dias: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """
    Calcula ciclos reales desde contadores de IZARO para un recurso.

    Analiza los contadores de piezas y cruza con la referencia en fabricacion.
    Devuelve por referencia: ciclo mediano (seg), piezas/hora, n muestras.
    """
    recurso = db.query(Recurso).filter_by(nombre=nombre_recurso).first()
    if not recurso:
        raise HTTPException(404, f"Recurso '{nombre_recurso}' no encontrado")

    try:
        result, fuente = mes_service.compute_real_cycles(recurso.centro_trabajo, dias)
    except Exception as exc:
        raise HTTPException(502, f"Error consultando IZARO: {exc}")

    return {
        "recurso": nombre_recurso,
        "centro_trabajo": recurso.centro_trabajo,
        "dias": dias,
        "fuente": fuente,
        "referencias": result,
    }
