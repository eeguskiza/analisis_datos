"""CRUD de plantillas de informes (JSON en data/report_templates/)."""
from __future__ import annotations

import json
import shutil

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.config import settings
from nexo.services.auth import require_permission

router = APIRouter(
    prefix="/plantillas",
    tags=["plantillas"],
    dependencies=[Depends(require_permission("plantillas:read"))],
)

_edit = [Depends(require_permission("plantillas:edit"))]


def _templates_dir():
    d = settings.templates_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.get("")
def listar():
    """Lista plantillas disponibles."""
    d = _templates_dir()
    templates = []
    for f in sorted(d.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            templates.append({
                "name": f.stem,
                "filename": f.name,
                "description": data.get("description", ""),
            })
        except Exception:
            templates.append({"name": f.stem, "filename": f.name, "description": "Error leyendo"})
    return {"plantillas": templates}


@router.get("/{name}")
def leer(name: str):
    """Lee una plantilla por nombre (sin .json)."""
    path = _templates_dir() / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, "Plantilla no encontrada")
    return json.loads(path.read_text(encoding="utf-8"))


@router.put("/{name}", dependencies=_edit)
def guardar(name: str, payload: dict):
    """Guarda/actualiza una plantilla."""
    path = _templates_dir() / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"ok": True}


class NewTemplate(BaseModel):
    name: str
    description: str = ""


@router.post("", dependencies=_edit)
def crear(req: NewTemplate):
    """Crea nueva plantilla copiando default."""
    d = _templates_dir()
    dest = d / f"{req.name}.json"
    if dest.exists():
        raise HTTPException(409, "Ya existe una plantilla con ese nombre")
    default = d / "default.json"
    if not default.exists():
        raise HTTPException(404, "No existe plantilla default para copiar")
    data = json.loads(default.read_text(encoding="utf-8"))
    data["name"] = req.name
    data["description"] = req.description
    dest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "name": req.name}


@router.delete("/{name}", dependencies=_edit)
def borrar(name: str):
    """Borra una plantilla (no permite borrar default)."""
    if name == "default":
        raise HTTPException(400, "No se puede borrar la plantilla default")
    path = _templates_dir() / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, "Plantilla no encontrada")
    path.unlink()
    return {"ok": True}
