"""Motor SQLAlchemy, sesion y modelos ORM."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, ForeignKey, Integer, String, Text,
    UniqueConstraint, create_engine, inspect,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from api.config import settings


# ── Engine ────────────────────────────────────────────────────────────────────

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

# Habilitar FK constraints en SQLite (desactivados por defecto)
if "sqlite" in settings.database_url:
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class Base(DeclarativeBase):
    pass


# ── Modelos ───────────────────────────────────────────────────────────────────

class Ciclo(Base):
    __tablename__ = "ciclos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    maquina = Column(String(50), nullable=False)
    referencia = Column(String(100), nullable=False)
    tiempo_ciclo = Column(Float, nullable=False, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("maquina", "referencia", name="uq_maquina_ref"),
    )


class Recurso(Base):
    __tablename__ = "recursos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    centro_trabajo = Column(Integer, nullable=False)
    nombre = Column(String(50), nullable=False, unique=True)
    seccion = Column(String(50), nullable=False, default="GENERAL")
    activo = Column(Boolean, default=True)


class Ejecucion(Base):
    __tablename__ = "ejecuciones"

    id = Column(Integer, primary_key=True, autoincrement=True)
    fecha_inicio = Column(String(10), nullable=False)
    fecha_fin = Column(String(10), nullable=False)
    source = Column(String(20), nullable=False, default="db")
    status = Column(String(20), nullable=False, default="running")
    modulos = Column(Text, default="")
    log = Column(Text, default="")
    n_pdfs = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class InformeMeta(Base):
    __tablename__ = "informes_meta"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ejecucion_id = Column(Integer, nullable=True)
    fecha = Column(String(10), nullable=False)
    seccion = Column(String(50), nullable=False)
    maquina = Column(String(50), default="")
    modulo = Column(String(50), default="")
    pdf_path = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class DatosProduccion(Base):
    """Datos extraídos de IZARO, almacenados por extracción."""
    __tablename__ = "datos_produccion"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ejecucion_id = Column(Integer, ForeignKey("ejecuciones.id", ondelete="CASCADE"), nullable=False)
    recurso = Column(String(50), nullable=False)
    seccion = Column(String(50), nullable=False)
    fecha = Column(Date, nullable=False)
    h_ini = Column(String(10))
    h_fin = Column(String(10))
    tiempo = Column(Float, default=0)
    proceso = Column(String(30))
    incidencia = Column(String(200), default="")
    cantidad = Column(Float, default=0)
    malas = Column(Float, default=0)
    recuperadas = Column(Float, default=0)
    referencia = Column(String(50), default="")


class Contacto(Base):
    """Lista de contactos para envío de informes."""
    __tablename__ = "contactos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    nombre = Column(String(100), nullable=False)
    email = Column(String(200), nullable=False, unique=True)


# ── Mapa seccion ──────────────────────────────────────────────────────────────

SECTION_MAP = {
    "luk1": "LINEAS", "luk2": "LINEAS", "luk3": "LINEAS", "luk6": "LINEAS",
    "coroa": "LINEAS", "vw1": "LINEAS", "omr": "LINEAS", "t48": "TALLADORAS",
}


# ── Init / Migracion ─────────────────────────────────────────────────────────

def init_db() -> None:
    """Crea tablas si no existen e importa datos iniciales."""
    Base.metadata.create_all(engine)

    with SessionLocal() as session:
        _import_ciclos_csv(session)
        _import_recursos_json(session)


def _import_ciclos_csv(session: Session) -> None:
    """Importa ciclos.csv si la tabla esta vacia. Ignora duplicados."""
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
    """Importa recursos desde db_config.json si la tabla esta vacia."""
    if session.query(Recurso).count() > 0:
        return

    try:
        from OEE.db.connector import load_config
        cfg = load_config()
    except Exception:
        return
    for r in cfg.get("recursos", []):
        nombre = r.get("nombre", "").strip()
        if not nombre:
            continue
        seccion = SECTION_MAP.get(nombre.lower(), "GENERAL")
        session.add(Recurso(
            centro_trabajo=int(r.get("centro_trabajo", 0)),
            nombre=nombre,
            seccion=seccion,
            activo=r.get("activo", True),
        ))
    session.commit()


def get_db():
    """Dependency para FastAPI: yield session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_health() -> tuple[bool, str]:
    """Verifica conexion a la BBDD local."""
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "OK"
    except Exception as exc:
        return False, str(exc)
