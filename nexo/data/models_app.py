"""Modelos ORM APP (schema ``ecs_mobility``) - Plan 03-03 (DATA-03).

Migrado desde ``api/database.py`` en Plan 03-03. ``api/database.py`` mantiene
re-export de estas clases para no romper consumidores Phase 2 (Landmine #9
del RESEARCH: ``api/services/pipeline.py`` tiene ~15 refs a ``MetricaOEE``,
``ReferenciaStats``, ``IncidenciaResumen``, ``Ejecucion``, ``InformeMeta``,
``DatosProduccion`` desde ``api.database``). La identidad del objeto se
preserva: ``api.database.Recurso is nexo.data.models_app.Recurso``.

Nota: el engine (``_mssql_creator``, ``SessionLocal``), la logica de
bootstrap (``init_db``, ``_import_*``), el dependency (``get_db``) y
``check_db_health`` NO se migran - siguen en ``api/database.py`` porque
son infraestructura, no modelos.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# -- Modelos ------------------------------------------------------------------


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
    """Metricas OEE calculadas - tabla principal para Power BI."""

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


# -- Mapa seccion -------------------------------------------------------------

SECTION_MAP = {
    "luk1": "LINEAS",
    "luk2": "LINEAS",
    "luk3": "LINEAS",
    "luk6": "LINEAS",
    "coroa": "LINEAS",
    "vw1": "LINEAS",
    "omr": "LINEAS",
    "t48": "TALLADORAS",
}


__all__ = [
    "Base",
    "Ciclo",
    "Recurso",
    "Ejecucion",
    "InformeMeta",
    "DatosProduccion",
    "Contacto",
    "MetricaOEE",
    "ReferenciaStats",
    "IncidenciaResumen",
    "SECTION_MAP",
]
