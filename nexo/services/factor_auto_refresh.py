"""Job ``factor_auto_refresh`` (Plan 04-04 / D-20 / QUERY-02).

Recalcula ``factor_ms`` de cada endpoint con preflight si
``factor_updated_at`` es stale (> ``NEXO_AUTO_REFRESH_STALE_DAYS``,
default 60). Doble safety-net sobre el boton manual "Recalcular factor"
(D-04) — si el propietario se olvida de recalibrar, el sistema lo hace
el 1er Monday del mes a las 03:10 UTC.

Algoritmo: reusa ``nexo.services.factor_learning.compute_factor`` —
mismo helper que usa el boton manual desde ``/ajustes/limites`` (DRY).

Auditoria: cada endpoint recalculado graba una fila en
``nexo.audit_log`` con ``path='__auto_refresh__'`` +
``details_json={endpoint, old_factor, new_factor, sample_size,
reason='stale'}``.

Emision de NOTIFY: tras cada UPDATE via ``ThresholdRepo.update``, se
llama ``thresholds_cache.notify_changed(endpoint)`` para que los
workers se actualicen en <1s (D-19).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone

from nexo.data.engines import SessionLocalNexo
from nexo.data.repositories.nexo import AuditRepo, ThresholdRepo
from nexo.services import thresholds_cache
from nexo.services.factor_learning import compute_factor

log = logging.getLogger("nexo.factor_auto_refresh")


def run_once() -> dict:
    """Ejecuta un ciclo de auto-refresh.

    Devuelve dict ``{endpoint: new_factor}`` con los endpoints
    actualizados. Endpoints con ``factor_updated_at`` reciente o sin
    muestras suficientes se omiten.
    """
    stale_days = int(
        os.environ.get("NEXO_AUTO_REFRESH_STALE_DAYS", "60")
    )
    cutoff_stale = datetime.now(timezone.utc) - timedelta(days=stale_days)
    updated: dict[str, float] = {}

    db = SessionLocalNexo()
    try:
        repo_t = ThresholdRepo(db)
        thresholds = repo_t.list_all()
        for t in thresholds:
            # Skip si factor_updated_at es reciente (< stale_days).
            if t.factor_updated_at is not None and t.factor_updated_at >= cutoff_stale:
                log.info(
                    "factor_auto_refresh: %s fresco (updated_at=%s) — skip",
                    t.endpoint, t.factor_updated_at.isoformat(),
                )
                continue

            factor_new, sample_size = compute_factor(db, t.endpoint, max_samples=30)
            if factor_new is None:
                log.info(
                    "factor_auto_refresh: %s sin suficientes datos (%d) — skip",
                    t.endpoint, sample_size,
                )
                continue

            old_factor = t.factor_ms
            repo_t.update(
                endpoint=t.endpoint,
                warn_ms=t.warn_ms,
                block_ms=t.block_ms,
                factor_ms=factor_new,
                updated_by=None,  # cron run, no user.
                factor_touched=True,  # marca factor_updated_at=now() (D-20).
            )

            # Audit log (best-effort).
            try:
                AuditRepo(db).append(
                    user_id=None,
                    ip="127.0.0.1",
                    method="UPDATE",
                    path="__auto_refresh__",
                    status=200,
                    details_json=json.dumps({
                        "endpoint": t.endpoint,
                        "old_factor": old_factor,
                        "new_factor": factor_new,
                        "sample_size": sample_size,
                        "reason": "stale",
                        "stale_days": stale_days,
                    }),
                )
                db.commit()
            except Exception:
                log.exception(
                    "factor_auto_refresh: audit append failed for %s",
                    t.endpoint,
                )
                db.rollback()

            # Propagacion cross-worker (D-19): NOTIFY usa conexion raw
            # propia (AUTOCOMMIT); best-effort para que un fallo de
            # NOTIFY no tumbe el job.
            try:
                thresholds_cache.notify_changed(t.endpoint)
            except Exception:
                log.exception(
                    "factor_auto_refresh: NOTIFY failed for %s",
                    t.endpoint,
                )

            updated[t.endpoint] = factor_new
            log.info(
                "factor_auto_refresh: %s factor %s -> %.1f (sample_size=%d)",
                t.endpoint, old_factor, factor_new, sample_size,
            )

        return updated
    finally:
        db.close()


__all__ = ["run_once"]
