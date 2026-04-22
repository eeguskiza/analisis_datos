"""Endpoints para gestión de contactos y envío de informes por email."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.config import settings
from api.database import Contacto, get_db
from api.services.email import enviar_informes, test_smtp
from nexo.services.auth import require_permission

router = APIRouter(
    prefix="/email",
    tags=["email"],
    dependencies=[Depends(require_permission("email:send"))],
)


# ── Modelos ──────────────────────────────────────────────────────────────────


class ContactoIn(BaseModel):
    nombre: str
    email: str


class EnviarRequest(BaseModel):
    destinatarios: list[str]  # lista de emails
    pdfs: list[str]  # rutas relativas de PDFs (como las devuelve el pipeline)
    asunto: str = ""


# ── Contactos ────────────────────────────────────────────────────────────────


@router.get("/contactos")
def listar_contactos(db: Session = Depends(get_db)):
    rows = db.query(Contacto).order_by(Contacto.nombre).all()
    return {
        "contactos": [{"id": c.id, "nombre": c.nombre, "email": c.email} for c in rows]
    }


@router.post("/contactos")
def crear_contacto(payload: ContactoIn, db: Session = Depends(get_db)):
    existe = db.query(Contacto).filter_by(email=payload.email).first()
    if existe:
        raise HTTPException(409, "Ya existe un contacto con ese email")
    db.add(Contacto(nombre=payload.nombre, email=payload.email))
    db.commit()
    return {"ok": True}


@router.delete("/contactos/{contacto_id}")
def borrar_contacto(contacto_id: int, db: Session = Depends(get_db)):
    c = db.get(Contacto, contacto_id)
    if not c:
        raise HTTPException(404, "Contacto no encontrado")
    db.delete(c)
    db.commit()
    return {"ok": True}


# ── Test SMTP ────────────────────────────────────────────────────────────────


@router.get("/test")
def test_conexion_smtp():
    """Prueba que la conexión SMTP funciona."""
    resultado = test_smtp()
    return {"ok": resultado == "OK", "mensaje": resultado}


# ── Enviar ───────────────────────────────────────────────────────────────────


@router.post("/enviar")
def enviar(payload: EnviarRequest):
    """Envía PDFs por email a los destinatarios seleccionados."""
    if not payload.destinatarios:
        raise HTTPException(400, "No hay destinatarios")
    if not payload.pdfs:
        raise HTTPException(400, "No hay PDFs seleccionados")

    # Resolver rutas de PDFs

    pdf_paths = []
    for rel in payload.pdfs:
        full = settings.informes_dir / rel
        if full.exists():
            pdf_paths.append(full)

    if not pdf_paths:
        raise HTTPException(404, "No se encontraron los PDFs")

    asunto = payload.asunto or f"Informes OEE — {len(pdf_paths)} PDFs"
    cuerpo = (
        f"Se adjuntan {len(pdf_paths)} informes OEE generados.\n\n— Nexo (automatico)"
    )

    resultado = enviar_informes(payload.destinatarios, asunto, cuerpo, pdf_paths)

    if resultado != "OK":
        raise HTTPException(502, resultado)

    return {
        "ok": True,
        "enviados": len(pdf_paths),
        "destinatarios": payload.destinatarios,
    }
