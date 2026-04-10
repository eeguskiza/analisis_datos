"""Tests de funciones helper del calculo OEE."""
import pytest
from datetime import datetime, time
from OEE.oee_secciones.main import (
    calcular_solapamiento,
    determinar_turno,
    normalizar_proceso,
    clasificar_incidencia,
    dividir_en_turnos,
    parse_time_value,
)


class TestSolapamiento:
    """calcular_solapamiento: horas de overlap entre dos rangos."""

    def test_overlap_total(self):
        a1 = datetime(2026, 1, 1, 8, 0)
        a2 = datetime(2026, 1, 1, 16, 0)
        b1 = datetime(2026, 1, 1, 8, 0)
        b2 = datetime(2026, 1, 1, 16, 0)
        assert calcular_solapamiento(a1, a2, b1, b2) == 8.0

    def test_sin_overlap(self):
        a1 = datetime(2026, 1, 1, 8, 0)
        a2 = datetime(2026, 1, 1, 12, 0)
        b1 = datetime(2026, 1, 1, 14, 0)
        b2 = datetime(2026, 1, 1, 16, 0)
        assert calcular_solapamiento(a1, a2, b1, b2) == 0.0

    def test_overlap_parcial(self):
        a1 = datetime(2026, 1, 1, 8, 0)
        a2 = datetime(2026, 1, 1, 14, 0)
        b1 = datetime(2026, 1, 1, 12, 0)
        b2 = datetime(2026, 1, 1, 16, 0)
        assert calcular_solapamiento(a1, a2, b1, b2) == 2.0

    def test_b_dentro_de_a(self):
        a1 = datetime(2026, 1, 1, 6, 0)
        a2 = datetime(2026, 1, 1, 18, 0)
        b1 = datetime(2026, 1, 1, 10, 0)
        b2 = datetime(2026, 1, 1, 12, 0)
        assert calcular_solapamiento(a1, a2, b1, b2) == 2.0


class TestDeterminarTurno:
    """determinar_turno: T1(06-14), T2(14-22), T3(22-06)."""

    def test_t1(self):
        assert determinar_turno(time(8, 0)) == "T1"
        assert determinar_turno(time(6, 0)) == "T1"
        assert determinar_turno(time(13, 59)) == "T1"

    def test_t2(self):
        assert determinar_turno(time(14, 0)) == "T2"
        assert determinar_turno(time(18, 0)) == "T2"
        assert determinar_turno(time(21, 59)) == "T2"

    def test_t3(self):
        assert determinar_turno(time(22, 0)) == "T3"
        assert determinar_turno(time(23, 59)) == "T3"
        assert determinar_turno(time(0, 0)) == "T3"
        assert determinar_turno(time(5, 59)) == "T3"


class TestNormalizarProceso:
    """normalizar_proceso: texto libre → produccion/preparacion/incidencias."""

    def test_produccion(self):
        assert normalizar_proceso("Producción") == "produccion"

    def test_preparacion(self):
        assert normalizar_proceso("Preparación") == "preparacion"

    def test_incidencia(self):
        assert normalizar_proceso("Incidencia") == "incidencias"

    def test_none_defaults_to_produccion(self):
        # Comportamiento intencionado: sin valor = produccion
        assert normalizar_proceso(None) == "produccion"
        assert normalizar_proceso("") == "produccion"


class TestParseTimeValue:
    """parse_time_value: convierte strings de hora a time."""

    def test_hhmm(self):
        t = parse_time_value("08:30")
        assert t == time(8, 30)

    def test_hhmmss(self):
        t = parse_time_value("14:30:00")
        assert t is not None
        assert t.hour == 14
        assert t.minute == 30

    def test_none(self):
        assert parse_time_value(None) is None
        assert parse_time_value("") is None
