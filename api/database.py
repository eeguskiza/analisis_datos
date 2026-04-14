"""Motor SQLAlchemy, sesion y modelos ORM — conecta a SQL Server (ecs_mobility)."""
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

def _mssql_creator():
    """Crea conexion pyodbc directa (más fiable que URL de SQLAlchemy)."""
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
        UniqueConstraint("maquina", "referencia", name="uq_ciclos"),
        {"schema": "cfg"},
    )


class Recurso(Base):
    __tablename__ = "recursos"
    __table_args__ = {"schema": "cfg"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    centro_trabajo = Column(Integer, nullable=False)
    nombre = Column(String(50), nullable=False, unique=True)
    seccion = Column(String(50), nullable=False, default="GENERAL")
    activo = Column(Boolean, default=True)


class Ejecucion(Base):
    __tablename__ = "ejecuciones"
    __table_args__ = {"schema": "oee"}

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
    __tablename__ = "informes"
    __table_args__ = {"schema": "oee"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    ejecucion_id = Column(Integer, nullable=True)
    fecha = Column(String(10), nullable=False)
    seccion = Column(String(50), nullable=False)
    maquina = Column(String(50), default="")
    modulo = Column(String(50), default="")
    pdf_path = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class DatosProduccion(Base):
    """Datos extraidos de IZARO, almacenados por extraccion."""
    __tablename__ = "datos"
    __table_args__ = {"schema": "oee"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    ejecucion_id = Column(Integer, ForeignKey("oee.ejecuciones.id"), nullable=False)
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
    """Lista de contactos para envio de informes."""
    __tablename__ = "contactos"
    __table_args__ = {"schema": "cfg"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    nombre = Column(String(100), nullable=False)
    email = Column(String(200), nullable=False, unique=True)


# Tablas adicionales para Power BI (solo escritura desde pipeline)

class MetricaOEE(Base):
    """Metricas OEE calculadas — tabla principal para Power BI."""
    __tablename__ = "metricas"
    __table_args__ = {"schema": "oee"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    ejecucion_id = Column(Integer, ForeignKey("oee.ejecuciones.id"), nullable=False)
    seccion = Column(String(50))
    recurso = Column(String(50))
    fecha = Column(Date)
    turno = Column(String(5))
    horas_brutas = Column(Float, default=0)
    horas_disponible = Column(Float, default=0)
    horas_operativo = Column(Float, default=0)
    horas_preparacion = Column(Float, default=0)
    horas_indisponibilidad = Column(Float, default=0)
    horas_paros = Column(Float, default=0)
    tiempo_ideal = Column(Float, default=0)
    perdidas_rend = Column(Float, default=0)
    piezas_totales = Column(Integer, default=0)
    piezas_malas = Column(Integer, default=0)
    piezas_recuperadas = Column(Integer, default=0)
    buenas_finales = Column(Integer, default=0)
    disponibilidad_pct = Column(Float, default=0)
    rendimiento_pct = Column(Float, default=0)
    calidad_pct = Column(Float, default=0)
    oee_pct = Column(Float, default=0)


class ReferenciaStats(Base):
    __tablename__ = "referencias"
    __table_args__ = {"schema": "oee"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    ejecucion_id = Column(Integer, ForeignKey("oee.ejecuciones.id"), nullable=False)
    recurso = Column(String(50))
    referencia = Column(String(50))
    ciclo_ideal = Column(Float)
    ciclo_real = Column(Float)
    cantidad = Column(Integer, default=0)
    horas = Column(Float, default=0)


class IncidenciaResumen(Base):
    __tablename__ = "incidencias"
    __table_args__ = {"schema": "oee"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    ejecucion_id = Column(Integer, ForeignKey("oee.ejecuciones.id"), nullable=False)
    recurso = Column(String(50))
    nombre = Column(String(200))
    tipo = Column(String(20))
    horas = Column(Float, default=0)


# ── Mapa seccion ──────────────────────────────────────────────────────────────

SECTION_MAP = {
    "luk1": "LINEAS", "luk2": "LINEAS", "luk3": "LINEAS", "luk6": "LINEAS",
    "coroa": "LINEAS", "vw1": "LINEAS", "omr": "LINEAS", "t48": "TALLADORAS",
}


# ── Init ──────────────────────────────────────────────────────────────────────

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
        session.add(Recurso(
            centro_trabajo=int(r.get("centro_trabajo", 0)),
            nombre=nombre,
            seccion=seccion,
            activo=r.get("activo", True),
        ))
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
                session.add(Recurso(
                    centro_trabajo=0,
                    nombre=nombre,
                    seccion=seccion,
                    activo=True,
                ))
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
