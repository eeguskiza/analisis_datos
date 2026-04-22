"""Motor SQLAlchemy, sesion y modelos ORM - conecta a SQL Server (ecs_mobility).

Plan 03-03 (DATA-03): los modelos ORM se migraron a
``nexo/data/models_app.py``. Este modulo mantiene engine + SessionLocal +
bootstrap (``init_db``, ``_import_*``) + ``get_db`` + ``check_db_health``
y RE-EXPORTA las clases ORM desde ``nexo.data.models_app`` para preservar
consumidores existentes (Landmine #9: ``api/services/pipeline.py`` tiene
~15 refs a ``MetricaOEE``, ``ReferenciaStats``, ``IncidenciaResumen``,
``Ejecucion``, ``InformeMeta``, ``DatosProduccion``).
"""

from __future__ import annotations

import csv

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from api.config import settings


# -- Engine -------------------------------------------------------------------


def _mssql_creator():
    """Crea conexion pyodbc directa (mas fiable que URL de SQLAlchemy)."""
    import pyodbc

    return pyodbc.connect(
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={settings.db_server},{settings.db_port};"
        f"DATABASE={settings.db_name};"
        f"UID={settings.db_user};"
        f"PWD={settings.db_password};"
        f"TrustServerCertificate=yes;"
        f"Encrypt=yes;",
        timeout=10,
    )


engine = create_engine(
    "mssql+pyodbc://",
    creator=_mssql_creator,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_recycle=3600,
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


# -- Modelos ORM - migrados a nexo/data/models_app.py en Plan 03-03 -----------
# Re-export para preservar consumidores existentes (Landmine #9 en
# api/services/pipeline.py - 15 refs a MetricaOEE, ReferenciaStats,
# IncidenciaResumen, Ejecucion, InformeMeta, DatosProduccion). Elimina
# cuando todos los consumers migren al path nuevo (Mark-IV).
from nexo.data.models_app import (  # noqa: E402, F401
    Base,
    Ciclo,
    Contacto,
    DatosProduccion,
    Ejecucion,
    IncidenciaResumen,
    InformeMeta,
    MetricaOEE,
    Recurso,
    ReferenciaStats,
    SECTION_MAP,
)


# -- Init ---------------------------------------------------------------------


def init_db() -> None:
    """Verifica conexion y carga datos iniciales si tablas vacias."""
    # Las tablas ya existen en SQL Server, no hacemos create_all
    with SessionLocal() as session:
        _import_ciclos_csv(session)
        _import_recursos_json(session)


def _import_ciclos_csv(session: Session) -> None:
    """Importa ciclos.csv si la tabla esta vacia."""
    if session.query(Ciclo).count() > 0:
        return

    csv_path = settings.ciclos_path
    if not csv_path.exists():
        return

    seen: set[tuple[str, str]] = set()
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            maq = (row.get("maquina") or "").strip()
            ref = (row.get("referencia") or "").strip()
            tc = float(row.get("tiempo_ciclo", 0) or 0)
            key = (maq, ref)
            if maq and ref and key not in seen:
                seen.add(key)
                session.add(Ciclo(maquina=maq, referencia=ref, tiempo_ciclo=tc))
    session.commit()


def _import_recursos_json(session: Session) -> None:
    """Importa recursos desde db_config.json + ciclos.csv si la tabla esta vacia."""
    if session.query(Recurso).count() > 0:
        return

    nombres_añadidos: set[str] = set()

    # 1) Desde db_config.json (tienen centro_trabajo real)
    try:
        from OEE.db.connector import load_config as _load_cfg

        cfg = _load_cfg()
    except Exception:
        cfg = {}
    for r in cfg.get("recursos", []):
        nombre = r.get("nombre", "").strip()
        if not nombre or nombre in nombres_añadidos:
            continue
        seccion = SECTION_MAP.get(nombre.lower(), "GENERAL")
        session.add(
            Recurso(
                centro_trabajo=int(r.get("centro_trabajo", 0)),
                nombre=nombre,
                seccion=seccion,
                activo=r.get("activo", True),
            )
        )
        nombres_añadidos.add(nombre)

    # 2) Maquinas que aparecen en ciclos.csv pero no en db_config
    csv_path = settings.ciclos_path
    if csv_path.exists():
        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                nombre = (row.get("maquina") or "").strip()
                if not nombre or nombre in nombres_añadidos:
                    continue
                seccion = SECTION_MAP.get(nombre.lower(), "GENERAL")
                session.add(
                    Recurso(
                        centro_trabajo=0,
                        nombre=nombre,
                        seccion=seccion,
                        activo=True,
                    )
                )
                nombres_añadidos.add(nombre)

    session.commit()


def get_db():
    """Dependency para FastAPI: yield session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_health() -> tuple[bool, str]:
    """Verifica conexion a la BBDD."""
    try:
        from sqlalchemy import text

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "OK"
    except Exception as exc:
        return False, str(exc)
