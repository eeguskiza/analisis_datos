"""CRUD de recursos / centros de trabajo — desde BBDD."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.database import Recurso, SECTION_MAP, get_db
from api.models import RecursosPayload
from api.models import Recurso as RecursoModel
from api.services import db as mes_service

router = APIRouter(prefix="/recursos", tags=["recursos"])

SECCIONES_DISPONIBLES = sorted(set(SECTION_MAP.values()) | {"GENERAL"})


@router.get("")
def listar(db: Session = Depends(get_db)):
    rows = db.query(Recurso).order_by(Recurso.seccion, Recurso.nombre).all()
    return {"recursos": [
        {"id": r.id, "centro_trabajo": r.centro_trabajo, "nombre": r.nombre,
         "seccion": r.seccion, "activo": r.activo}
        for r in rows
    ]}


@router.put("")
def guardar(payload: RecursosPayload, db: Session = Depends(get_db)):
    """Reescribe la lista de recursos."""
    db.query(Recurso).delete()
    for r in payload.recursos:
        seccion = SECTION_MAP.get(r.nombre.lower(), "GENERAL")
        db.add(Recurso(
            centro_trabajo=r.centro_trabajo,
            nombre=r.nombre,
            seccion=seccion,
            activo=r.activo,
        ))
    db.commit()

    # Sincronizar con db_config.json para que el conector MES lo use
    _sync_to_config(db)
    return {"ok": True}


@router.post("/row")
def add_row(recurso: RecursoModel, db: Session = Depends(get_db)):
    exists = db.query(Recurso).filter_by(nombre=recurso.nombre).first()
    if exists:
        raise HTTPException(409, "Ya existe ese recurso")
    seccion = SECTION_MAP.get(recurso.nombre.lower(), "GENERAL")
    db.add(Recurso(
        centro_trabajo=recurso.centro_trabajo,
        nombre=recurso.nombre,
        seccion=seccion,
        activo=recurso.activo,
    ))
    db.commit()
    _sync_to_config(db)
    return {"ok": True}


@router.get("/detectar")
def detectar(db: Session = Depends(get_db)):
    """
    Detecta centros de trabajo en IZARO.

    Devuelve para cada CT:
    - codigo: número del centro de trabajo en IZARO
    - nombre_izaro: nombre descriptivo en IZARO (ej: "Linea Luk 1")
    - ultimo_registro: fecha del último parte registrado
    - n_registros_mes: nº de registros en el último mes (0 = inactivo)
    - configurado: true si ya existe en nuestros recursos
    - nombre_local: nombre asignado localmente (si está configurado)
    - seccion_local: sección asignada (si está configurado)
    """
    try:
        maquinas = mes_service.discover_resources()
    except Exception as exc:
        raise HTTPException(502, f"Error conectando a IZARO: {exc}")

    # Marcar cuáles ya están configurados
    locales = {r.centro_trabajo: r for r in db.query(Recurso).all()}

    for m in maquinas:
        local = locales.get(m["codigo"])
        m["configurado"] = local is not None
        m["nombre_local"] = local.nombre if local else ""
        m["seccion_local"] = local.seccion if local else ""

    return {
        "maquinas": maquinas,
        "secciones": SECCIONES_DISPONIBLES,
    }


def _sync_to_config(db: Session) -> None:
    """Sincroniza recursos de la BBDD a db_config.json."""
    rows = db.query(Recurso).all()
    cfg = mes_service.get_config()
    cfg["recursos"] = [
        {"centro_trabajo": r.centro_trabajo, "nombre": r.nombre, "activo": r.activo}
        for r in rows
    ]
    mes_service.update_config(cfg)
