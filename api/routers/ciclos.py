"""CRUD para ciclos (tiempos de ciclo ideales) - via CicloRepo (DATA-03).

Plan 03-03 Task 4.1: las queries ORM inline se encapsulan en
``CicloRepo`` (``nexo.data.repositories.app``). El router solo hace
transport logic: validacion de payload + grouping + return shape.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.database import SECTION_MAP
from api.deps import DbApp
from api.models import CicloRow, CiclosPayload
from api.services import db as mes_service
from nexo.data.repositories.app import CicloRepo, RecursoRepo
from nexo.services.auth import require_permission

router = APIRouter(
    prefix="/ciclos",
    tags=["ciclos"],
    dependencies=[Depends(require_permission("ciclos:read"))],
)

# Shortcut para endpoints mutables - inyecta el check adicional de :edit.
_edit = [Depends(require_permission("ciclos:edit"))]


@router.get("")
def listar(db: DbApp):
    """Devuelve ciclos agrupados por seccion y maquina."""
    rows = CicloRepo(db).list_all()

    # Flat list (DTOs -> dicts para JSON)
    flat = [
        {
            "id": r.id,
            "maquina": r.maquina,
            "referencia": r.referencia,
            "tiempo_ciclo": r.tiempo_ciclo,
        }
        for r in rows
    ]

    # Agrupado por seccion -> maquina (transport logic, se queda en router)
    grouped: dict[str, dict[str, list]] = {}
    for r in rows:
        sec = SECTION_MAP.get(r.maquina.lower(), "GENERAL")
        if sec not in grouped:
            grouped[sec] = {}
        if r.maquina not in grouped[sec]:
            grouped[sec][r.maquina] = []
        grouped[sec][r.maquina].append({
            "id": r.id,
            "referencia": r.referencia,
            "tiempo_ciclo": r.tiempo_ciclo,
        })

    return {"rows": flat, "grouped": grouped}


@router.put("", dependencies=_edit)
def guardar(payload: CiclosPayload, db: DbApp):
    """Reescribe todos los ciclos."""
    repo = CicloRepo(db)
    repo.replace_all([r.model_dump() for r in payload.rows])
    db.commit()
    return {"ok": True, "count": len(payload.rows)}


@router.post("/row", dependencies=_edit)
def add_row(row: CicloRow, db: DbApp):
    """Anade un ciclo."""
    repo = CicloRepo(db)
    if repo.exists(row.maquina, row.referencia):
        raise HTTPException(409, "Ya existe esa combinacion maquina/referencia")
    repo.add(
        maquina=row.maquina,
        referencia=row.referencia,
        tiempo_ciclo=row.tiempo_ciclo,
    )
    db.commit()
    return {"ok": True}


@router.put("/row/{ciclo_id}", dependencies=_edit)
def update_row(ciclo_id: int, row: CicloRow, db: DbApp):
    """Actualiza un ciclo por ID."""
    repo = CicloRepo(db)
    ciclo = repo.get_by_id(ciclo_id)
    if not ciclo:
        raise HTTPException(404, "Ciclo no encontrado")
    ciclo.maquina = row.maquina
    ciclo.referencia = row.referencia
    ciclo.tiempo_ciclo = row.tiempo_ciclo
    db.commit()
    return {"ok": True}


@router.delete("/row/{ciclo_id}", dependencies=_edit)
def delete_row(ciclo_id: int, db: DbApp):
    """Elimina un ciclo por ID."""
    repo = CicloRepo(db)
    ciclo = repo.get_by_id(ciclo_id)
    if not ciclo:
        raise HTTPException(404, "Ciclo no encontrado")
    repo.delete(ciclo)
    db.commit()
    return {"ok": True}


@router.post("/sync-csv", dependencies=_edit)
def sync_to_csv(db: DbApp):
    """Exporta la tabla ciclos a ciclos.csv (para que los modulos OEE lo lean)."""
    import csv as csv_mod
    from api.config import settings

    rows = CicloRepo(db).list_all_orm()
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
    db: DbApp,
    dias: int = Query(30, ge=1, le=365),
):
    """
    Calcula ciclos reales desde contadores de IZARO para un recurso.

    Analiza los contadores de piezas y cruza con la referencia en fabricacion.
    Devuelve por referencia: ciclo mediano (seg), piezas/hora, n muestras.
    """
    recurso = RecursoRepo(db).get_orm_by_nombre(nombre_recurso)
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


@router.get("/live/{nombre_recurso}")
def estado_live(
    nombre_recurso: str,
    db: DbApp,
    umbral: int = Query(600, ge=30, le=3600),
):
    """
    Estado en vivo de una maquina: si esta registrando contadores ahora mismo.

    Devuelve ultima lectura, segundos transcurridos y estado
    (activo/inactivo/sin_datos) usando `umbral` segundos como limite.
    """
    recurso = RecursoRepo(db).get_orm_by_nombre(nombre_recurso)
    if not recurso:
        raise HTTPException(404, f"Recurso '{nombre_recurso}' no encontrado")

    try:
        estado = mes_service.live_status(recurso.centro_trabajo, umbral)
    except Exception as exc:
        raise HTTPException(502, f"Error consultando IZARO: {exc}")

    return {
        "recurso": nombre_recurso,
        "centro_trabajo": recurso.centro_trabajo,
        **estado,
    }
