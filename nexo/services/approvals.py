"""Servicio de approvals (Plan 04-03 / QUERY-06).

Wrapper puro sobre ``ApprovalRepo`` que cierra el flujo de aprobación
asíncrona para queries pesadas (D-06 / D-13 / D-14 / D-15 / D-16).

Contratos:

- ``create_approval`` — crea fila ``pending`` en ``nexo.query_approvals``
  con ``params_json`` canonicalizado (sort_keys=True) para garantizar
  equality bit-a-bit entre creación y consumo (D-15).
- ``consume_approval`` — CAS single-use que marca la fila
  ``consumed``. Si algo falla (no existe, user equivocado, status
  incorrecto, ya consumido, params cambiaron) → ``HTTPException(403)``
  con diagnóstico específico (no silent-pass).
- ``approve`` / ``reject`` — pending → approved|rejected
  (propietario-only en el router).
- ``cancel`` — pending → cancelled con ownership check server-side
  (D-16).
- ``expire_stale`` — job semanal marca pending > TTL como ``expired``
  (D-14).

Este módulo **no** emite email/banner/websocket — D-13 explícitamente
descartó notificaciones push en Mark-III (SMTP Out of Scope).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from sqlalchemy.orm import Session

from nexo.data.dto.query import QueryApprovalRow
from nexo.data.models_nexo import NexoQueryApproval
from nexo.data.repositories.nexo import ApprovalRepo


# H-05 fix: listas que el contrato declara como "conjuntos no ordenados".
# Pydantic no normaliza list order; el frontend puede re-serializar con
# orden distinto (p.ej. <select multiple> en un orden y alfabético en
# otro). Canonicalizamos ordenando estos campos para que la equality
# ``params_json = :pj`` del CAS no genere falsos 403.
_CANONICAL_SET_FIELDS: frozenset[str] = frozenset({"recursos", "modulos"})


def _canonical_json(obj: dict) -> str:
    """Serializa dict a JSON con sort_keys=True + ensure_ascii=False.

    Garantía de equality bit-a-bit entre creación y consumo (D-15): los
    mismos params producen la misma cadena exacta aunque las keys se
    hayan insertado en orden distinto. Parte de la mitigación
    T-04-01-03 (params tampering) — el CAS en ``ApprovalRepo.consume``
    compara ``params_json = :pj`` textualmente.

    H-05 fix: además de ``sort_keys``, ordenamos los valores de los
    campos listados en ``_CANONICAL_SET_FIELDS`` (``recursos``,
    ``modulos``) para que listas lógicamente equivalentes con diferente
    orden produzcan el mismo canonical string. Otros listas (p.ej.
    ``columns`` que sí son ordenadas) se dejan intactas.
    """
    normalized: dict = {}
    for key, value in obj.items():
        if (
            key in _CANONICAL_SET_FIELDS
            and isinstance(value, list)
            and all(isinstance(x, (str, int, float, bool)) for x in value)
        ):
            # Solo ordenamos listas de primitivos comparables; si hay
            # dicts u otros objetos mezclados, skip ordering para no
            # levantar TypeError.
            try:
                normalized[key] = sorted(value)
            except TypeError:
                normalized[key] = value
        else:
            normalized[key] = value
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False)


def create_approval(
    db: Session,
    *,
    user_id: int,
    endpoint: str,
    params: dict,
    estimated_ms: int,
    ttl_days: int = 7,
) -> int:
    """Crea una fila ``pending`` en ``nexo.query_approvals`` y devuelve su id.

    Args:
        db: Session de Postgres ``nexo.*``.
        user_id: id del usuario que solicita.
        endpoint: identificador del endpoint (e.g. ``pipeline/run``,
            ``bbdd/query``, ``capacidad``, ``operarios``).
        params: dict serializable que describe la query exacta
            (fechas, recursos, sql, filters). Se canonicaliza antes de
            persistir para equality check en ``consume_approval``.
        estimated_ms: estimación del preflight para auditoría.
        ttl_days: días antes de marcar ``expired``. Default 7 (D-14).

    Returns:
        ``approval_id`` (int). El router devuelve este id al cliente.
    """
    params_json = _canonical_json(params)
    row = ApprovalRepo(db).create(
        user_id=user_id,
        endpoint=endpoint,
        params_json=params_json,
        estimated_ms=estimated_ms,
        ttl_days=ttl_days,
    )
    return row.id


def consume_approval(
    db: Session,
    *,
    approval_id: int,
    user_id: int,
    current_params: dict,
) -> QueryApprovalRow:
    """Consume (marca ``consumed``) una aprobación ``approved`` válida.

    CAS atómico (D-15) delegado a ``ApprovalRepo.consume``:
    ``UPDATE ... WHERE consumed_at IS NULL AND status='approved' AND
    user_id=:uid AND params_json=:pj RETURNING *``. Race-safe:
    segunda llamada con mismo approval_id devuelve None (0 rows
    affected).

    Diagnóstico detallado al fallar (research §Pattern 4 lines 920-935):

    1. Si la fila no existe → "Approval no existe".
    2. Si user mismatch → "Approval pertenece a otro usuario".
    3. Si status != 'approved' → "Approval está en estado {status}".
    4. Si consumed_at != NULL → "Approval ya fue consumido".
    5. Si params no coinciden → "Parámetros cambiaron respecto a la
       solicitud aprobada".

    Args:
        db: Session.
        approval_id: id de la fila a consumir.
        user_id: id del usuario que ejecuta (ownership check).
        current_params: dict con los params actuales del request
            (se canonicaliza y compara bit-a-bit con el
            ``params_json`` almacenado al crear).

    Returns:
        ``QueryApprovalRow`` (DTO frozen) con ``consumed_at`` poblado y
        ``status='consumed'``.

    Raises:
        HTTPException(403): con detail específico según el fallo.
    """
    current_json = _canonical_json(current_params)
    result = ApprovalRepo(db).consume(
        approval_id=approval_id,
        user_id=user_id,
        current_params_json=current_json,
    )
    if result is not None:
        return result

    # Diagnóstico — leer fila cruda para dar mensaje específico.
    row = db.get(NexoQueryApproval, approval_id)
    if row is None:
        raise HTTPException(status_code=403, detail="Approval no existe")
    if row.user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Approval pertenece a otro usuario",
        )
    if row.status != "approved":
        raise HTTPException(
            status_code=403,
            detail=f"Approval está en estado {row.status}",
        )
    if row.consumed_at is not None:
        raise HTTPException(
            status_code=403,
            detail="Approval ya fue consumido",
        )
    raise HTTPException(
        status_code=403,
        detail="Parámetros cambiaron respecto a la solicitud aprobada",
    )


def approve(db: Session, approval_id: int, approved_by: int) -> None:
    """pending → approved. Solo callable desde el router de propietario."""
    ApprovalRepo(db).approve(approval_id, approved_by)


def reject(db: Session, approval_id: int, decided_by: int) -> None:
    """pending → rejected. Solo callable desde el router de propietario."""
    ApprovalRepo(db).reject(approval_id, decided_by)


def cancel(db: Session, approval_id: int, user_id: int) -> bool:
    """pending → cancelled con ownership check (D-16).

    Devuelve ``False`` si el user no es el dueño o si status != pending.
    El router traduce False → ``HTTPException(403)``.
    """
    return ApprovalRepo(db).cancel(approval_id, user_id)


def expire_stale(db: Session, ttl_days: int) -> int:
    """pending → expired para filas con created_at < now - ttl_days.

    Llamado por el job semanal ``approvals_cleanup`` (Monday 03:05,
    D-14). Devuelve número de filas marcadas expired.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
    return ApprovalRepo(db).expire_stale(cutoff)


__all__ = [
    "create_approval",
    "consume_approval",
    "approve",
    "reject",
    "cancel",
    "expire_stale",
]
