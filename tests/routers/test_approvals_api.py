"""Wave 0 stub — approvals API router contracts (QUERY-06, D-13/D-16).

Implementacion de ``/api/approvals/*`` + ``/mis-solicitudes`` +
``/ajustes/solicitudes`` aterriza en Plan 04-03. Este archivo documenta
los contratos.

Planned tests:
  - test_create_approval_pending: POST /api/approvals con
    {endpoint, params, estimated_ms} → 201 + fila pending + id
    devuelto al cliente.
  - test_propietario_approve_flow: POST /api/approvals/<id>/approve
    como propietario → 200 + status='approved' + approved_at + audit_log.
  - test_non_propietario_approve_returns_403: usuario normal aprobando
    → 403 (require_permission propietario only).
  - test_user_cancel_own_pending: POST /api/approvals/<id>/cancel
    siendo el creador → 200 + status='cancelled' (D-16).
  - test_user_cannot_cancel_others: usuario A cancelando approval de
    usuario B → 403.
  - test_count_badge_returns_html_fragment: GET /api/approvals/count →
    fragmento HTMX "<span>(3)</span>" o string vacio si 0 (D-13).
  - test_mis_solicitudes_lists_only_own: GET /mis-solicitudes devuelve
    solo las filas del current_user.
"""
from __future__ import annotations

import pytest


pytest.skip(
    "Implemented in Plan 04-03 (approvals router + /mis-solicitudes UI).",
    allow_module_level=True,
)
