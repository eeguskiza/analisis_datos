#!/usr/bin/env python3
"""Bootstrap del primer usuario ``propietario`` de Nexo.

Interactivo: pide username, nombre, apellidos, email y password (doble
prompt, sin echo).
Idempotente: si ya existe un ``propietario``, sale con exit 0 sin crear
otro.

Uso::

    docker compose exec web python scripts/create_propietario.py

Notas
-----

- Password mínima 12 caracteres (decisión LOCKED en ``docs/AUTH_MODEL.md``).
- Se crea con ``must_change_password=False`` explícitamente:

  El propietario bootstrap **no** pasa por el flujo de cambio obligatorio
  en el primer login. Si fuera ``True``, el primer login redirigiría a
  ``/cambiar-password`` que aún no existe cuando corre este script
  (plan 02-02 la crea), produciendo un bucle redirect.

- Se crea sin departamentos: el rol ``propietario`` ignora departamentos
  por definición (acceso global).

- NO se envía email. SMTP está Out of Scope Mark-III.
"""
from __future__ import annotations

import getpass
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from nexo.db.engine import SessionLocalNexo  # noqa: E402
from nexo.db.models import NexoUser  # noqa: E402
from nexo.services.auth import hash_password  # noqa: E402


MIN_PASSWORD_LENGTH = 12
PROPIETARIO_ROLE = "propietario"


def _log(msg: str) -> None:
    print(f"[create_propietario] {msg}")


def _prompt_email() -> str:
    email = input("Email del propietario: ").strip().lower()
    if not email or "@" not in email:
        raise SystemExit("Email invalido — debe contener '@' y no estar vacio.")
    return email


def _prompt_username(email: str) -> str:
    default = email.split("@", 1)[0]
    raw = input(f"Username del propietario [{default}]: ").strip().lower()
    username = raw or default
    username = re.sub(r"\s+", ".", username)
    if not username:
        raise SystemExit("Username invalido — no puede estar vacio.")
    return username


def _prompt_text(label: str) -> str:
    value = input(f"{label}: ").strip()
    if not value:
        raise SystemExit(f"{label} no puede estar vacio.")
    return value


def _prompt_password() -> str:
    pw1 = getpass.getpass(f"Password (min {MIN_PASSWORD_LENGTH} chars): ")
    if len(pw1) < MIN_PASSWORD_LENGTH:
        raise SystemExit(
            f"Password demasiado corto ({len(pw1)} chars). Minimo {MIN_PASSWORD_LENGTH}."
        )
    pw2 = getpass.getpass("Password (repetir): ")
    if pw1 != pw2:
        raise SystemExit("Los dos passwords no coinciden. Reintenta.")
    return pw1


def main() -> int:
    db = SessionLocalNexo()
    try:
        # Idempotencia — si ya hay propietario, no creamos otro.
        existing = db.execute(
            select(NexoUser).where(NexoUser.role == PROPIETARIO_ROLE)
        ).scalars().first()
        if existing is not None:
            _log(f"Ya existe un propietario (email={existing.email}). Nada que hacer.")
            return 0

        email = _prompt_email()
        username = _prompt_username(email)
        name = _prompt_text("Nombre")
        surname = _prompt_text("Apellidos")
        # Comprueba unicidad de email (un usuario con otro rol podria tenerlo).
        dup = db.execute(
            select(NexoUser).where(NexoUser.email == email)
        ).scalars().first()
        if dup is not None:
            _log(
                f"Ya existe un usuario con email {email} (rol={dup.role}). "
                "Promociona a propietario manualmente via SQL o borra y reintenta."
            )
            return 1
        username_dup = db.execute(
            select(NexoUser).where(NexoUser.username == username)
        ).scalars().first()
        if username_dup is not None:
            _log(
                f"Ya existe un usuario con username {username} "
                f"(email={username_dup.email}). Elige otro username."
            )
            return 1

        password = _prompt_password()

        user = NexoUser(
            username=username,
            name=name,
            surname=surname,
            email=email,
            nombre=f"{name} {surname}",
            password_hash=hash_password(password),
            role=PROPIETARIO_ROLE,
            active=True,
            must_change_password=False,  # bootstrap: no redirige a /cambiar-password
        )
        db.add(user)
        db.commit()

        _log(f"Propietario creado: {email}")
        _log(
            "IMPORTANTE: este usuario no pasa por el flujo de cambio obligatorio "
            "de password. Recomendado cambiar la password manualmente tras el "
            "primer login desde /cambiar-password (plan 02-02)."
        )
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
