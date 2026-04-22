"""Loader con ``lru_cache`` para archivos ``.sql`` versionados (D-01).

Uso::

    from nexo.data.sql.loader import load_sql
    from sqlalchemy import bindparam, text

    stmt = text(load_sql("mes/centro_mando_fmesmic")).bindparams(
        bindparam("codes", expanding=True),
    )

El ``name`` incluye el sub-directorio (engine) y la extensión es
opcional. ``mes/estado`` resuelve a ``mes/estado.sql``. El contenido
se cachea en memoria (``lru_cache(maxsize=128)``); como los ``.sql``
son estáticos no hay que invalidar entre requests.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

_PACKAGE = "nexo.data.sql"


@lru_cache(maxsize=128)
def load_sql(name: str) -> str:
    """Carga ``'mes/extraer_datos_produccion.sql'`` desde el paquete.

    Normaliza la extensión: ``'mes/estado'`` ⇒ ``'mes/estado.sql'``.
    Separa subdirectorios con ``/``. Cacheado con ``lru_cache`` porque
    los archivos son inmutables en runtime.
    """
    if not name.endswith(".sql"):
        name = f"{name}.sql"
    ref = files(_PACKAGE)
    for part in name.split("/"):
        ref = ref / part
    return ref.read_text(encoding="utf-8")
