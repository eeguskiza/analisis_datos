"""CRUD de usuarios - solo accesible al propietario.

Plan 03-03 Task 4.6: eliminado ``get_nexo_db`` local duplicado, ahora
usa ``DbNexo`` de ``api.deps``. Los queries de lectura se delegan a
``UserRepo`` / ``RoleRepo`` (``nexo.data.repositories.nexo``). Las
mutations (add/edit/delete) siguen con ORM inline porque es clarity
local - D-02 acepta ORM intra-services cuando es mas legible.

Permission: ``usuarios:manage`` (lista vacia en PERMISSION_MAP -> bypass
del propietario, todos los demas roles reciben 403).

Endpoints:
- ``GET  /ajustes/usuarios``              -> lista
- ``POST /ajustes/usuarios/crear``        -> crear nuevo usuario
- ``POST /ajustes/usuarios/{id}/editar``  -> cambia rol, departamentos, active
- ``POST /ajustes/usuarios/{id}/reset-password`` -> rehashes password +
  invalida sesiones
- ``POST /ajustes/usuarios/{id}/desactivar`` -> active=False + revoca sesiones

Todo via forms HTML + RedirectResponse 303. Sin JSON en este router -
la UI es pura Jinja2/Alpine.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.deps import DbNexo, render
from nexo.data.models_nexo import NexoDepartment, NexoUser
from nexo.data.repositories.nexo import RoleRepo
from nexo.services.auth import (
    hash_password,
    require_permission,
    revoke_all_sessions,
)

logger = logging.getLogger("nexo.usuarios")

router = APIRouter(
    prefix="/ajustes/usuarios",
    tags=["ajustes"],
    dependencies=[Depends(require_permission("usuarios:manage"))],
)


MIN_PASSWORD_LEN = 12
VALID_ROLES = frozenset({"propietario", "directivo", "usuario"})
VALID_DEPT_CODES = frozenset(
    {"rrhh", "comercial", "ingenieria", "produccion", "gerencia"}
)


def _serialize_user(u: NexoUser) -> dict:
    # Rule 1 bugfix (deviation Plan 08-03): el template llama ``u|tojson``
    # dentro de ``openEdit()`` / ``openReset()``; un ``datetime`` no es
    # JSON-serializable y el render explotaba con cualquier usuario que
    # hubiera logueado (``last_login != None``). Separamos el dict en
    # campos JSON-safe + un string ``last_login_display`` preformateado
    # para la celda de la tabla (``strftime`` ya no corre en Jinja).
    last_login_display: str = (
        u.last_login.strftime("%Y-%m-%d %H:%M") if u.last_login else ""
    )
    return {
        "id": u.id,
        "username": u.username,
        "name": u.name,
        "surname": u.surname,
        "email": u.email,
        # Plan 08-03 / UIREDO-02: nombre opcional; template muestra
        # "(sin nombre)" si es None. El openEdit() del Alpine reinyecta
        # este valor en el modal de edicion.
        "nombre": u.nombre,
        "role": u.role,
        "active": u.active,
        "must_change_password": u.must_change_password,
        "last_login_display": last_login_display,
        "departments": sorted(d.code for d in u.departments),
    }


def _normalize_nombre(nombre: str | None) -> str | None:
    """Normaliza el ``nombre`` del form: strip + empty -> None.

    El campo es opcional (D-26 / UI-SPEC §Label pattern). Vacio o solo
    whitespace se persiste como NULL para que el fallback del topbar
    (email local-part) pueda kicker si el operario no rellena nada.
    """
    if nombre is None:
        return None
    stripped = nombre.strip()
    return stripped if stripped else None


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _normalize_username(username: str | None) -> str:
    value = (username or "").strip().lower()
    if not value:
        return ""
    return re.sub(r"\s+", ".", value)


def _display_name(name: str | None, surname: str | None) -> str | None:
    return _normalize_text(" ".join(p for p in [name, surname] if p))


def _render_list(
    request: Request,
    db: Session,
    *,
    error: Optional[str] = None,
    ok: Optional[str] = None,
    open_create: bool = False,
    edit_id: Optional[int] = None,
) -> HTMLResponse:
    users = db.execute(select(NexoUser).order_by(NexoUser.email)).scalars().all()
    depts = RoleRepo(db).list_departments()
    return render(
        "ajustes_usuarios.html",
        request,
        {
            "page": "ajustes",
            "users": [_serialize_user(u) for u in users],
            "departments": [{"code": d.code, "name": d.name} for d in depts],
            "roles": sorted(VALID_ROLES),
            "error": error,
            "ok": ok,
            "open_create": open_create,
            "edit_id": edit_id,
        },
    )


@router.get("", response_class=HTMLResponse)
async def listar(
    request: Request,
    db: DbNexo,
    error: Optional[str] = None,
    ok: Optional[str] = None,
):
    return _render_list(request, db, error=error, ok=ok)


@router.post("/crear")
async def crear(
    request: Request,
    db: DbNexo,
    username: str = Form(...),
    name: str = Form(...),
    surname: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    password_repetir: str = Form(...),
    role: str = Form(...),
    nombre: str | None = Form(None),
    departments: list[str] = Form(default=[]),
):
    username_norm = _normalize_username(username)
    name_norm = _normalize_text(name)
    surname_norm = _normalize_text(surname)
    email_norm = email.strip().lower()
    nombre_norm = _display_name(name_norm, surname_norm) or _normalize_nombre(nombre)

    # Validaciones
    if not username_norm:
        return _render_list(request, db, error="Username obligatorio.", open_create=True)
    if not name_norm:
        return _render_list(request, db, error="Nombre obligatorio.", open_create=True)
    if not surname_norm:
        return _render_list(request, db, error="Apellidos obligatorios.", open_create=True)
    if not email_norm or "@" not in email_norm:
        return _render_list(request, db, error="Correo no valido.", open_create=True)
    if role not in VALID_ROLES:
        return _render_list(
            request, db, error=f"Rol no valido: {role}", open_create=True
        )
    if password != password_repetir:
        return _render_list(
            request, db, error="Las contrasenas no coinciden.", open_create=True
        )
    if len(password) < MIN_PASSWORD_LEN:
        return _render_list(
            request,
            db,
            error=f"Password minima {MIN_PASSWORD_LEN} caracteres.",
            open_create=True,
        )
    for code in departments:
        if code not in VALID_DEPT_CODES:
            return _render_list(
                request, db, error=f"Departamento invalido: {code}", open_create=True
            )

    if db.execute(
        select(NexoUser).where(NexoUser.username == username_norm)
    ).scalar_one_or_none():
        return _render_list(
            request,
            db,
            error=f"Ya existe un usuario con username {username_norm}.",
            open_create=True,
        )

    # Unicidad de email
    if db.execute(
        select(NexoUser).where(NexoUser.email == email_norm)
    ).scalar_one_or_none():
        return _render_list(
            request,
            db,
            error=f"Ya existe un usuario con email {email_norm}.",
            open_create=True,
        )

    # Cargar departamentos
    depts = []
    if departments:
        depts = (
            db.execute(
                select(NexoDepartment).where(NexoDepartment.code.in_(departments))
            )
            .scalars()
            .all()
        )

    new_user = NexoUser(
        username=username_norm,
        name=name_norm,
        surname=surname_norm,
        email=email_norm,
        nombre=nombre_norm,
        password_hash=hash_password(password),
        role=role,
        active=True,
        must_change_password=True,  # fuerza cambio en primer login
    )
    new_user.departments = list(depts)
    db.add(new_user)
    db.commit()
    logger.info(
        "usuario creado: %s (rol=%s, depts=%s, nombre=%s)",
        email_norm,
        role,
        sorted(departments),
        "<set>" if nombre_norm else "<empty>",
    )

    return RedirectResponse(
        f"/ajustes/usuarios?ok=usuario-creado:{email_norm}", status_code=303
    )


@router.post("/{user_id}/editar")
async def editar(
    user_id: int,
    request: Request,
    db: DbNexo,
    username: str = Form(...),
    name: str = Form(...),
    surname: str = Form(...),
    email: str = Form(...),
    role: str = Form(...),
    nombre: str | None = Form(None),
    departments: list[str] = Form(default=[]),
    active: str = Form(default="off"),
):
    user = db.get(NexoUser, user_id)
    if user is None:
        raise HTTPException(404, "Usuario no encontrado")

    username_norm = _normalize_username(username)
    name_norm = _normalize_text(name)
    surname_norm = _normalize_text(surname)
    email_norm = email.strip().lower()

    # Safeguard: el propietario no puede cambiarse a si mismo de rol ni
    # desactivarse (evita locking out del unico admin).
    current_user = request.state.user
    is_self = current_user.id == user.id

    if not username_norm:
        return _render_list(request, db, error="Username obligatorio.", edit_id=user_id)
    if not name_norm:
        return _render_list(request, db, error="Nombre obligatorio.", edit_id=user_id)
    if not surname_norm:
        return _render_list(request, db, error="Apellidos obligatorios.", edit_id=user_id)
    if not email_norm or "@" not in email_norm:
        return _render_list(request, db, error="Correo no valido.", edit_id=user_id)
    if role not in VALID_ROLES:
        return _render_list(
            request, db, error=f"Rol no valido: {role}", edit_id=user_id
        )
    for code in departments:
        if code not in VALID_DEPT_CODES:
            return _render_list(
                request, db, error=f"Departamento invalido: {code}", edit_id=user_id
            )

    new_active = active == "on"

    if is_self and role != user.role:
        return _render_list(
            request, db, error="No puedes cambiarte tu propio rol.", edit_id=user_id
        )
    if is_self and not new_active:
        return _render_list(
            request, db, error="No puedes desactivarte a ti mismo.", edit_id=user_id
        )

    # Cargar departamentos nuevos
    new_depts = []
    if departments:
        new_depts = (
            db.execute(
                select(NexoDepartment).where(NexoDepartment.code.in_(departments))
            )
            .scalars()
            .all()
        )

    username_dup = db.execute(
        select(NexoUser).where(
            NexoUser.username == username_norm,
            NexoUser.id != user.id,
        )
    ).scalar_one_or_none()
    if username_dup:
        return _render_list(
            request,
            db,
            error=f"Ya existe un usuario con username {username_norm}.",
            edit_id=user_id,
        )

    email_dup = db.execute(
        select(NexoUser).where(
            NexoUser.email == email_norm,
            NexoUser.id != user.id,
        )
    ).scalar_one_or_none()
    if email_dup:
        return _render_list(
            request,
            db,
            error=f"Ya existe un usuario con correo {email_norm}.",
            edit_id=user_id,
        )

    role_changed = user.role != role
    deactivated_now = user.active and not new_active

    user.username = username_norm
    user.name = name_norm
    user.surname = surname_norm
    user.email = email_norm
    user.role = role
    user.nombre = _display_name(name_norm, surname_norm) or _normalize_nombre(nombre)
    user.departments = list(new_depts)
    user.active = new_active
    db.commit()

    # Si rebaja de rol o desactiva -> revocar sesiones activas del target
    # para que la proxima request use los nuevos privilegios.
    if role_changed or deactivated_now:
        revoke_all_sessions(db, user.id)

    logger.info(
        "usuario editado: %s (id=%d, rol=%s, depts=%s, active=%s)",
        user.email,
        user.id,
        role,
        sorted(departments),
        new_active,
    )

    return RedirectResponse(
        f"/ajustes/usuarios?ok=usuario-editado:{user.email}", status_code=303
    )


@router.post("/{user_id}/reset-password")
async def reset_password(
    user_id: int,
    request: Request,
    db: DbNexo,
    password_nueva: str = Form(...),
):
    user = db.get(NexoUser, user_id)
    if user is None:
        raise HTTPException(404, "Usuario no encontrado")

    if len(password_nueva) < MIN_PASSWORD_LEN:
        return _render_list(
            request,
            db,
            error=f"Password minima {MIN_PASSWORD_LEN} caracteres.",
            edit_id=user_id,
        )

    user.password_hash = hash_password(password_nueva)
    user.must_change_password = True
    db.commit()
    revoke_all_sessions(db, user.id)

    logger.info("password reseteado: %s (id=%d)", user.email, user.id)

    return RedirectResponse(
        f"/ajustes/usuarios?ok=password-reseteado:{user.email}", status_code=303
    )


@router.post("/{user_id}/desactivar")
async def desactivar(
    user_id: int,
    request: Request,
    db: DbNexo,
):
    user = db.get(NexoUser, user_id)
    if user is None:
        raise HTTPException(404, "Usuario no encontrado")

    current_user = request.state.user
    if current_user.id == user.id:
        return _render_list(request, db, error="No puedes desactivarte a ti mismo.")

    # No permitir desactivar el ultimo propietario activo.
    if user.role == "propietario":
        other_owners = (
            db.execute(
                select(NexoUser).where(
                    NexoUser.role == "propietario",
                    NexoUser.active.is_(True),
                    NexoUser.id != user.id,
                )
            )
            .scalars()
            .all()
        )
        if not other_owners:
            return _render_list(
                request,
                db,
                error="No puedes desactivar al unico propietario activo del sistema.",
            )

    user.active = False
    db.commit()
    revoke_all_sessions(db, user.id)

    logger.info("usuario desactivado: %s (id=%d)", user.email, user.id)

    return RedirectResponse(
        f"/ajustes/usuarios?ok=usuario-desactivado:{user.email}", status_code=303
    )
