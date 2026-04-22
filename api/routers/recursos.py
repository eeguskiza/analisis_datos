"""CRUD de recursos / centros de trabajo - via RecursoRepo (DATA-03).

Plan 03-03 Task 4.2: las queries ORM inline se encapsulan en
``RecursoRepo`` (``nexo.data.repositories.app``). El router mantiene la
logica de transport (validacion de payload, auto-detect, _auto_name,
sync a config). ``_sync_to_config`` es service-layer helper y se queda
aqui (no es CRUD puro).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.database import SECTION_MAP
from api.deps import DbApp
from api.models import RecursosPayload
from api.models import Recurso as RecursoModel
from api.services import db as mes_service
from nexo.data.repositories.app import RecursoRepo
from nexo.services.auth import require_permission

router = APIRouter(
    prefix="/recursos",
    tags=["recursos"],
    dependencies=[Depends(require_permission("recursos:read"))],
)

_edit = [Depends(require_permission("recursos:edit"))]

SECCIONES_DISPONIBLES = sorted(set(SECTION_MAP.values()) | {"GENERAL"})


@router.get("")
def listar(db: DbApp):
    rows = RecursoRepo(db).list_all()
    return {
        "recursos": [
            {
                "id": r.id,
                "centro_trabajo": r.centro_trabajo,
                "nombre": r.nombre,
                "seccion": r.seccion,
                "activo": r.activo,
            }
            for r in rows
        ]
    }


@router.put("", dependencies=_edit)
def guardar(payload: RecursosPayload, db: DbApp):
    """Reescribe la lista de recursos."""
    repo = RecursoRepo(db)
    repo.replace_all([r.model_dump() for r in payload.recursos])
    db.commit()

    # Sincronizar con db_config.json para que el conector MES lo use
    _sync_to_config(db)
    return {"ok": True}


@router.post("/row", dependencies=_edit)
def add_row(recurso: RecursoModel, db: DbApp):
    repo = RecursoRepo(db)
    if repo.get_orm_by_nombre(recurso.nombre):
        raise HTTPException(409, "Ya existe ese recurso")
    repo.add(
        centro_trabajo=recurso.centro_trabajo,
        nombre=recurso.nombre,
        seccion=recurso.seccion,
        activo=recurso.activo,
    )
    db.commit()
    _sync_to_config(db)
    return {"ok": True}


@router.get("/detectar")
def detectar(db: DbApp):
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
    repo = RecursoRepo(db)
    locales = {r.centro_trabajo: r for r in repo.list_all_orm()}

    for m in maquinas:
        local = locales.get(m["codigo"])
        m["configurado"] = local is not None
        m["nombre_local"] = local.nombre if local else ""
        m["seccion_local"] = local.seccion if local else ""

    return {
        "maquinas": maquinas,
        "secciones": SECCIONES_DISPONIBLES,
    }


_NAME_RULES = [
    # (substring_in_izaro_name, section, name_format)
    ("soldadora", "SOLDADORAS", None),
    ("soldad", "SOLDADORAS", None),
    ("horno", "HORNOS", None),
    ("talladora", "TALLADORAS", None),
    ("tallado", "TALLADORAS", None),
    ("talla", "TALLADORAS", None),
    ("linea luk", "LINEAS", None),
    ("linea vw", "LINEAS", None),
    ("linea coroa", "LINEAS", None),
    ("luk", "LINEAS", None),
    ("vw", "LINEAS", None),
    ("coroa", "LINEAS", None),
    ("omr", "LINEAS", None),
    ("prensa", "PRENSAS", None),
    ("rectificad", "RECTIFICADORAS", None),
    ("embalaje", "EMBALAJE", None),
    ("robot", "ROBOTS", None),
    ("lavadora", "LAVADORAS", None),
    ("transport", "TRANSPORTE", None),
    ("almacen", "ALMACEN", None),
    ("centro mecan", "MECANIZADO", None),
    ("mecaniz", "MECANIZADO", None),
    ("torneado", "MECANIZADO", None),
    ("fresado", "MECANIZADO", None),
    ("equilibrad", "EQUILIBRADO", None),
    ("control", "CALIDAD", None),
    ("inspecc", "CALIDAD", None),
]


def _auto_name(codigo: int, nombre_izaro: str) -> tuple[str, str]:
    """Genera nombre legible + seccion a partir del nombre IZARO."""
    izaro_lower = nombre_izaro.lower().strip()

    # Buscar seccion por reglas
    seccion = "GENERAL"
    for substr, sec, _ in _NAME_RULES:
        if substr in izaro_lower:
            seccion = sec
            break

    # Ya conocidos en SECTION_MAP
    for known, known_sec in SECTION_MAP.items():
        if known in izaro_lower:
            seccion = known_sec
            break

    # Generar nombre legible: limpiar y compactar
    nombre = nombre_izaro.strip()
    # Quitar prefijos comunes tipo "Linea ", "Centro "
    for prefix in ["Linea ", "LINEA ", "Centro ", "CENTRO "]:
        if nombre.startswith(prefix):
            nombre = nombre[len(prefix) :]

    # Si tiene numeros, formatear como "Tipo Numero"
    nombre = nombre.strip()
    if not nombre:
        nombre = f"CT-{codigo}"

    # Normalizar: primera letra mayuscula, sin espacios multiples
    nombre = " ".join(nombre.split())

    # Hacer nombre unico y slug-friendly para uso interno
    slug = nombre.lower().replace(" ", "_").replace("-", "_")
    # Quitar chars raros
    slug = "".join(c for c in slug if c.isalnum() or c == "_")

    return slug, seccion


@router.post("/auto-detectar", dependencies=_edit)
def auto_detectar(db: DbApp):
    """
    Detecta TODAS las maquinas de IZARO y las anade automaticamente.

    Asigna nombres legibles y secciones segun el nombre en IZARO.
    Omite las que ya estan configuradas.
    """
    try:
        maquinas = mes_service.discover_resources()
    except Exception as exc:
        raise HTTPException(502, f"Error conectando a IZARO: {exc}")

    repo = RecursoRepo(db)
    existentes = {r.centro_trabajo for r in repo.list_all_orm()}
    añadidas = []

    for m in maquinas:
        codigo = m["codigo"]
        if codigo in existentes:
            continue

        nombre_izaro = m.get("nombre_izaro", "")
        nombre, seccion = _auto_name(codigo, nombre_izaro)

        # Evitar duplicados por nombre
        if repo.get_orm_by_nombre(nombre):
            nombre = f"{nombre}_{codigo}"

        repo.add(
            centro_trabajo=codigo,
            nombre=nombre,
            seccion=seccion,
            activo=True,
        )
        añadidas.append(
            {
                "codigo": codigo,
                "nombre_izaro": nombre_izaro,
                "nombre": nombre,
                "seccion": seccion,
            }
        )

    if añadidas:
        db.commit()
        _sync_to_config(db)

    return {"ok": True, "añadidas": len(añadidas), "maquinas": añadidas}


def _sync_to_config(db: Session) -> None:
    """Sincroniza recursos de la BBDD a db_config.json."""
    rows = RecursoRepo(db).list_all_orm()
    cfg = mes_service.get_config()
    cfg["recursos"] = [
        {"centro_trabajo": r.centro_trabajo, "nombre": r.nombre, "activo": r.activo}
        for r in rows
    ]
    mes_service.update_config(cfg)
