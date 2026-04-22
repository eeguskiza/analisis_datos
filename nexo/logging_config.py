"""Logging centralizado para Nexo.

Sustituye el ``logging.basicConfig(level=logging.INFO)`` que tenia
``api/main.py``. Problemas que resuelve:

- Duplicacion: el root logger tenia un handler *y* los loggers de
  sqlalchemy / uvicorn propagaban al root, imprimiendo cada linea dos
  veces (una con formato propio, otra con ``INFO:logger.name:``).
- SQL flood: ``engine_nexo`` tenia ``echo=settings.debug``, por lo que
  ``NEXO_DEBUG=true`` activaba tambien el dump de cada SELECT/BEGIN/
  COMMIT/ROLLBACK. Ahora el echo se controla con ``NEXO_LOG_SQL=true``
  (desacoplado del debug de la app).
- Falta de jerarquia visual: sin colores era imposible distinguir un
  WARNING de un INFO rutinario en un terminal saturado.

Formato: ``HH:MM:SS LEVEL  logger.name              mensaje``.
ANSI colors: INFO verde, WARNING amarillo, ERROR rojo, DEBUG cyan,
CRITICAL magenta. No requiere ``colorama`` ni ``rich`` — los codigos
ANSI funcionan en terminales Unix y en Windows 10+ con VT activado.

Uso::

    from nexo.logging_config import configure_logging
    configure_logging()

Llamar UNA vez al arrancar el proceso (import-time en api/main.py).
Idempotente: llamarlo dos veces no duplica handlers.
"""

from __future__ import annotations

import logging
import os
import sys


# ── ANSI color codes ───────────────────────────────────────────────────────
_RESET = "\x1b[0m"
_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"

_LEVEL_COLORS = {
    logging.DEBUG: "\x1b[36m",  # cyan
    logging.INFO: "\x1b[32m",  # green
    logging.WARNING: "\x1b[33m",  # yellow
    logging.ERROR: "\x1b[31m",  # red
    logging.CRITICAL: "\x1b[35;1m",  # bold magenta
}


def _supports_color() -> bool:
    """True si el stream de salida tolera ANSI colors.

    TTYs normales si, pipes / redirecciones a fichero no. ``NO_COLOR``
    env var (convencion https://no-color.org) fuerza off.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("NEXO_FORCE_COLOR"):
        return True
    return sys.stderr.isatty()


class _ColorFormatter(logging.Formatter):
    """Formatter con columnas fijas y colores por nivel."""

    # Anchura de columnas (suficiente para loggers tipicos: "sqlalchemy.engine.Engine"
    # son 25 caracteres; "nexo.thresholds_cache" 21; truncamos a 24).
    _LOGGER_WIDTH = 24
    _LEVEL_WIDTH = 7

    def __init__(self, use_color: bool) -> None:
        super().__init__()
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%H:%M:%S")
        level = record.levelname.ljust(self._LEVEL_WIDTH)
        name = record.name
        if len(name) > self._LOGGER_WIDTH:
            # Conservar el sufijo (mas informativo que el prefijo).
            name = "…" + name[-(self._LOGGER_WIDTH - 1) :]
        else:
            name = name.ljust(self._LOGGER_WIDTH)
        msg = record.getMessage()
        if record.exc_info:
            msg = msg + "\n" + self.formatException(record.exc_info)

        if self._use_color:
            color = _LEVEL_COLORS.get(record.levelno, "")
            return (
                f"{_DIM}{ts}{_RESET} "
                f"{color}{level}{_RESET} "
                f"{_BOLD}{name}{_RESET}  "
                f"{msg}"
            )
        return f"{ts} {level} {name}  {msg}"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configura el root logger con un unico handler colorizado.

    - Elimina cualquier handler existente (tipicamente el de
      ``logging.basicConfig`` previo) para evitar duplicados.
    - Desactiva ``propagate`` en los loggers de uvicorn y sqlalchemy
      para que no emitan por duplicado via sus propios handlers.
    - Silencia ``sqlalchemy.engine`` a WARNING por defecto. Con
      ``NEXO_LOG_SQL=true`` el ``echo=True`` del engine emite por
      DEBUG y pasa a verse.

    Idempotente: llamarla dos veces deja el mismo estado que una.
    """
    root = logging.getLogger()

    # Purgar handlers previos (basicConfig, imports que metieron handler)
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_ColorFormatter(use_color=_supports_color()))
    root.addHandler(handler)
    root.setLevel(level)

    # ── SQLAlchemy: silenciar por defecto, escuchar solo si NEXO_LOG_SQL ──
    # El ``echo=True`` del engine fuerza DEBUG en este logger; si ponemos
    # WARNING, el echo queda suprimido. Cuando el operador activa
    # NEXO_LOG_SQL=true, api.config.settings.log_sql es True, engine se
    # crea con echo=True, y aqui dejamos el logger en INFO para que las
    # lineas del echo se vean.
    _try_log_sql = os.environ.get("NEXO_LOG_SQL", "").lower() in {"1", "true", "yes"}
    sql_level = logging.INFO if _try_log_sql else logging.WARNING
    for logger_name in ("sqlalchemy.engine", "sqlalchemy.pool", "sqlalchemy.dialects"):
        lg = logging.getLogger(logger_name)
        lg.setLevel(sql_level)
        lg.propagate = True  # que use nuestro handler unico

    # ── Uvicorn: dejar que propaguen al root (formato uniforme) ──────────
    # Uvicorn por defecto monta sus propios handlers coloreados. Al
    # quitarselos forzamos que sus logs pasen por nuestro formatter.
    for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(logger_name)
        for h in list(lg.handlers):
            lg.removeHandler(h)
        lg.propagate = True
        lg.setLevel(logging.INFO)

    # Silenciar watchfiles (uvicorn --reload imprime "1 change detected"
    # por cada edit; ruidoso en dev).
    logging.getLogger("watchfiles").setLevel(logging.WARNING)
