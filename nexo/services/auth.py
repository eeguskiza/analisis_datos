"""Servicios de auth — hash, sesión y lockout. Sin cablear al middleware todavía.

Ese cableado (middleware, endpoints ``/login``, ``/cambiar-password``) es
trabajo de Plan 02-02. Aquí queda la lógica reutilizable encapsulada.

Design notes:

- **Argon2id**: parámetros RFC 9106 LOW_MEMORY via ``argon2.profiles``
  (research §Pattern 1). ``PasswordHasher.check_needs_rehash()`` permite
  re-hash transparente si la política cambia.
- **Sesión**: cookie firmada con ``itsdangerous.URLSafeTimedSerializer`` +
  tabla ``nexo.sessions`` para revocación inmediata (research §Stack
  Decision 1). JWT explícitamente descartado.
- **Lockout**: 5 intentos fallidos en 15 min sobre la tupla ``(email, ip)``
  → lock de 15 min. Purga automática al login exitoso
  (research §Pattern 3, AUTH_MODEL.md §Bloqueo progresivo).
- **Token de sesión**: 32 bytes random URL-safe (``secrets.token_urlsafe``)
  serializados con ``itsdangerous`` usando ``settings.secret_key``.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from argon2.profiles import RFC_9106_LOW_MEMORY
from fastapi import HTTPException, Request, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from api.config import settings
from nexo.db.models import NexoLoginAttempt, NexoSession, NexoUser


# ── Constantes de la política ──────────────────────────────────────────────

LOCKOUT_WINDOW_MINUTES = 15
LOCKOUT_THRESHOLD = 5
SESSION_TOKEN_BYTES = 32


# ── Password hashing (Argon2id, RFC 9106 LOW_MEMORY) ───────────────────────

_ph = PasswordHasher(
    time_cost=RFC_9106_LOW_MEMORY.time_cost,
    memory_cost=RFC_9106_LOW_MEMORY.memory_cost,
    parallelism=RFC_9106_LOW_MEMORY.parallelism,
    hash_len=RFC_9106_LOW_MEMORY.hash_len,
    salt_len=RFC_9106_LOW_MEMORY.salt_len,
)


def hash_password(password: str) -> str:
    """Devuelve el hash Argon2id PHC string del password en claro."""
    return _ph.hash(password)


def verify_password(stored_hash: str, password: str) -> bool:
    """Compara ``password`` contra ``stored_hash``. Resistente a timing attacks."""
    try:
        _ph.verify(stored_hash, password)
        return True
    except VerifyMismatchError:
        return False


def needs_rehash(stored_hash: str) -> bool:
    """True si la política argon2 cambió y conviene re-hash en el próximo login."""
    return _ph.check_needs_rehash(stored_hash)


# ── Firma / verificación de tokens de sesión ───────────────────────────────

_serializer = URLSafeTimedSerializer(settings.secret_key, salt="nexo.session.v1")


def sign_session_token(raw_token: str) -> str:
    """Firma un token de sesión random para meterlo en cookie HttpOnly."""
    return _serializer.dumps(raw_token)


def unsign_session_token(signed: str, max_age_seconds: int) -> Optional[str]:
    """Devuelve el raw_token si la firma es válida y no ha expirado, sino None."""
    try:
        return _serializer.loads(signed, max_age=max_age_seconds)
    except (BadSignature, SignatureExpired):
        return None


# ── Sesiones (tabla ``nexo.sessions``) ─────────────────────────────────────


def create_session(db: Session, user_id: int) -> tuple[str, str]:
    """Crea una sesión en BD y devuelve ``(raw_token, signed_cookie_value)``."""
    raw = secrets.token_urlsafe(SESSION_TOKEN_BYTES)
    expires = datetime.now(timezone.utc) + timedelta(hours=settings.session_ttl_hours)
    row = NexoSession(user_id=user_id, token=raw, expires_at=expires)
    db.add(row)
    db.commit()
    signed = sign_session_token(raw)
    return raw, signed


def get_session(db: Session, raw_token: str) -> Optional[NexoSession]:
    """Devuelve la sesión activa (no expirada) asociada al ``raw_token``."""
    now = datetime.now(timezone.utc)
    stmt = select(NexoSession).where(
        NexoSession.token == raw_token,
        NexoSession.expires_at > now,
    )
    return db.execute(stmt).scalar_one_or_none()


def extend_session(db: Session, session: NexoSession) -> None:
    """Sliding expiration — renueva ``expires_at`` a TTL completo desde ahora."""
    session.expires_at = datetime.now(timezone.utc) + timedelta(
        hours=settings.session_ttl_hours
    )
    db.commit()


def revoke_session(db: Session, raw_token: str) -> None:
    """Borra la sesión. Logout efectivo — revocación inmediata."""
    db.execute(delete(NexoSession).where(NexoSession.token == raw_token))
    db.commit()


def revoke_all_sessions(db: Session, user_id: int) -> None:
    """Invalida todas las sesiones de un usuario (p.ej. cuando se desactiva)."""
    db.execute(delete(NexoSession).where(NexoSession.user_id == user_id))
    db.commit()


# ── Lockout progresivo (tabla ``nexo.login_attempts``) ─────────────────────


def _window_cutoff() -> datetime:
    return datetime.now(timezone.utc) - timedelta(minutes=LOCKOUT_WINDOW_MINUTES)


def check_lockout(db: Session, email: str, ip: str) -> bool:
    """True si ``(email, ip)`` está bloqueado (≥5 fallos en los últimos 15 min)."""
    cutoff = _window_cutoff()
    stmt = select(NexoLoginAttempt).where(
        NexoLoginAttempt.email == email,
        NexoLoginAttempt.ip == ip,
        NexoLoginAttempt.failed_at > cutoff,
    )
    count = len(db.execute(stmt).scalars().all())
    return count >= LOCKOUT_THRESHOLD


def record_failed_attempt(db: Session, email: str, ip: str) -> None:
    """Graba un intento fallido. Limpieza de antiguos es lazy (al consultar)."""
    row = NexoLoginAttempt(email=email, ip=ip)
    db.add(row)
    db.commit()


def clear_attempts(db: Session, email: str, ip: str) -> None:
    """Borra intentos fallidos de la tupla. Llamar tras login exitoso."""
    db.execute(
        delete(NexoLoginAttempt).where(
            NexoLoginAttempt.email == email,
            NexoLoginAttempt.ip == ip,
        )
    )
    db.commit()


# ── Helpers de usuario ─────────────────────────────────────────────────────


def get_user_by_email(db: Session, email: str) -> Optional[NexoUser]:
    """Helper publico. Delegado a ``UserRepo.get_by_email_orm`` (DATA-04).

    Firma externa estable: sigue devolviendo ``NexoUser`` ORM (no DTO)
    porque el middleware de auth necesita el modelo completo para
    session management + password hashing.
    """
    # Lazy import para evitar ciclo: UserRepo importa models_nexo, que
    # es el mismo metadata que este modulo ya consume.
    from nexo.data.repositories.nexo import UserRepo

    return UserRepo(db).get_by_email_orm(email)


def get_user_by_login(db: Session, login: str) -> Optional[NexoUser]:
    """Busca usuario activo por ``username`` o ``email``."""
    from nexo.data.repositories.nexo import UserRepo

    return UserRepo(db).get_by_login_orm(login)


# ── RBAC (Plan 02-03) ─────────────────────────────────────────────────────

# Fuente de verdad de los permisos en codigo. La tabla ``nexo.permissions``
# sigue siendo catalogo seed (Phase 5 la reconciliara) pero este dict es
# quien decide en runtime.
#
# Clave: ``"modulo:accion"``. Valor: lista de ``department_code`` que pueden
# usar el permiso. El rol ``propietario`` NUNCA aparece aqui — tiene bypass
# hardcodeado en ``require_permission`` (research §Anti-Patterns).
#
# Mapeo completo en ``docs/AUTH_MODEL.md`` §Apendice PERMISSION_MAP
# (regenerado al cerrar Phase 2 para trazabilidad).
PERMISSION_MAP: dict[str, list[str]] = {
    # ── Modulos de produccion / planta ──────────────────────────────────
    "pipeline:run": ["ingenieria", "produccion"],
    "pipeline:read": ["ingenieria", "produccion", "gerencia"],
    "recursos:read": ["ingenieria", "produccion"],
    "recursos:edit": ["ingenieria"],
    "ciclos:read": ["ingenieria"],
    "ciclos:edit": ["ingenieria"],
    "centro_mando:read": ["produccion", "ingenieria", "gerencia"],
    "luk4:read": ["produccion", "ingenieria", "gerencia"],
    "capacidad:read": ["comercial", "ingenieria", "produccion", "gerencia"],
    "historial:read": ["ingenieria", "produccion", "comercial", "gerencia", "rrhh"],
    "informes:read": ["ingenieria", "produccion", "comercial", "gerencia", "rrhh"],
    "informes:delete": ["ingenieria"],
    "datos:read": ["ingenieria", "produccion"],
    # ── RRHH ────────────────────────────────────────────────────────────
    "operarios:read": ["rrhh"],
    "operarios:export": ["rrhh"],
    # ── Administracion tecnica ─────────────────────────────────────────
    "bbdd:read": ["ingenieria"],
    "conexion:read": ["ingenieria"],
    "conexion:config": [],  # lista vacia → solo propietario (tocar credenciales)
    "email:send": ["rrhh", "ingenieria", "gerencia"],
    "plantillas:read": ["ingenieria"],
    "plantillas:edit": ["ingenieria"],
    # ── Ajustes — Mark-III solo propietario ─────────────────────────────
    "ajustes:manage": [],
    "auditoria:read": [],
    "usuarios:manage": [],
    "aprobaciones:manage": [],  # Plan 04-03 (QUERY-06) — propietario-only
    "limites:manage": [],  # Plan 04-04 (QUERY-02) — propietario-only
    "rendimiento:read": [],  # Plan 04-04 (D-11) — propietario-only
}


def can(user: NexoUser | None, permission: str) -> bool:
    """True si ``user`` tiene ``permission``. Puro, sin side effects.

    Reglas (idénticas a ``require_permission`` — esta función ES la fuente
    de verdad; ``require_permission`` trampoliniza sobre ella):

    - ``user is None`` → False.
    - ``user.role == 'propietario'`` → True (bypass global, no consulta
      ``PERMISSION_MAP``).
    - Intersecta ``{d.code for d in user.departments}`` con
      ``PERMISSION_MAP[permission]``. True si hay intersección.

    Safe desde templates (Jinja global, D-03 / D-09) y tests — sin async,
    sin HTTPException. Un permiso desconocido devuelve ``[]`` → False.
    """
    if user is None:
        return False
    if user.role == "propietario":
        return True
    allowed = PERMISSION_MAP.get(permission, [])
    user_depts = {d.code for d in user.departments}
    return bool(user_depts.intersection(allowed))


def require_permission(permission: str):
    """Factory que devuelve un Dependency de FastAPI.

    Trampoline sobre :func:`can` (Plan 05-01 / D-03): ``can`` es la fuente
    de verdad pura; este dependency añade las responsabilidades específicas
    de FastAPI (leer ``request.state.user`` y levantar HTTPException).

    Validacion en runtime:

    1. ``request.state.user`` poblado por ``AuthMiddleware`` (Plan 02-02).
       Si falta, 401 (defensivo — el middleware deberia haber cortado antes).
    2. Si ``can(user, permission)`` es False → 403 con detail explicando el
       permiso pedido. ``can`` maneja internamente el bypass de propietario
       y la intersección ``user.departments ∩ PERMISSION_MAP[permission]``.

    Nota: el middleware eager-loadea ``user.departments`` antes de cerrar
    la sesion ORM para evitar DetachedInstanceError al llegar aqui.
    """

    async def _check(request: Request) -> NexoUser:
        user = getattr(request.state, "user", None)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
            )
        if not can(user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permiso requerido: {permission}",
            )
        return user

    _check.__name__ = f"require_permission__{permission.replace(':', '_')}"
    return _check
