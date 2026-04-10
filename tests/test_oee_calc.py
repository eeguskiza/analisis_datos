"""Tests del core de calculo OEE — convertir_raw_a_metricas.

Estas funciones son puras (dict in → dict out), no necesitan BD.
Si un refactor rompe el calculo, estos tests lo detectan.
"""
import pytest
from OEE.oee_secciones.main import (
    convertir_raw_a_metricas,
    crear_raw_metricas,
    clamp_pct,
    MIN_PIEZAS_OEE,
)


def _make_raw(**overrides):
    """Helper: crea raw metricas con defaults sensatos."""
    raw = crear_raw_metricas()
    raw.update(overrides)
    return raw


class TestOEEPerfecto:
    """Caso ideal: todo funciona sin perdidas."""

    def test_100_pct(self):
        raw = _make_raw(
            horas_produccion=8.0,
            horas_preparacion=0.0,
            horas_indisponibilidad=0.0,
            horas_paros=0.0,
            tiempo_ideal=8.0,  # ciclo real = ideal
            piezas_totales=100.0,
            piezas_malas=0.0,
            piezas_recuperadas=0.0,
        )
        m = convertir_raw_a_metricas(raw)
        assert m["disponibilidad_pct"] == 100.0
        assert m["rendimiento_pct"] == 100.0
        assert m["calidad_pct"] == 100.0
        assert m["oee_pct"] == 100.0


class TestDisponibilidad:
    """Perdidas de disponibilidad: indisponibilidad y paros."""

    def test_indisponibilidad_reduce_disponibilidad(self):
        raw = _make_raw(
            horas_produccion=8.0,
            horas_indisponibilidad=2.0,  # 2h de averia
            tiempo_ideal=6.0,
            piezas_totales=100.0,
        )
        m = convertir_raw_a_metricas(raw)
        assert m["disponibilidad_pct"] < 100.0
        assert m["disponibilidad_pct"] > 0.0

    def test_paros_reducen_disponibilidad(self):
        raw = _make_raw(
            horas_produccion=8.0,
            horas_paros=1.0,
            tiempo_ideal=7.0,
            piezas_totales=100.0,
        )
        m = convertir_raw_a_metricas(raw)
        assert m["disponibilidad_pct"] < 100.0

    def test_toda_indisponibilidad(self):
        raw = _make_raw(
            horas_produccion=8.0,
            horas_indisponibilidad=8.0,  # todo el tiempo parado
            piezas_totales=100.0,
        )
        m = convertir_raw_a_metricas(raw)
        assert m["disponibilidad_pct"] == 0.0


class TestRendimiento:
    """Perdidas de rendimiento: ciclo real > ideal."""

    def test_rendimiento_50_pct(self):
        # Ciclo ideal = 4h, pero tardamos 8h operativas
        raw = _make_raw(
            horas_produccion=8.0,
            tiempo_ideal=4.0,  # la mitad de rapido
            piezas_totales=100.0,
        )
        m = convertir_raw_a_metricas(raw)
        assert m["rendimiento_pct"] == 50.0

    def test_sin_ciclo_ideal(self):
        raw = _make_raw(
            horas_produccion=8.0,
            tiempo_ideal=0.0,  # no hay ciclo configurado
            piezas_totales=100.0,
        )
        m = convertir_raw_a_metricas(raw)
        assert m["rendimiento_pct"] == 0.0


class TestCalidad:
    """Perdidas de calidad: piezas malas."""

    def test_10pct_malas(self):
        raw = _make_raw(
            horas_produccion=8.0,
            tiempo_ideal=8.0,
            piezas_totales=100.0,
            piezas_malas=10.0,
        )
        m = convertir_raw_a_metricas(raw)
        assert m["calidad_pct"] == 90.0

    def test_recuperadas_mejoran_calidad(self):
        raw = _make_raw(
            horas_produccion=8.0,
            tiempo_ideal=8.0,
            piezas_totales=100.0,
            piezas_malas=10.0,
            piezas_recuperadas=5.0,  # 5 de 10 malas se recuperan
        )
        m = convertir_raw_a_metricas(raw)
        assert m["calidad_pct"] == 95.0

    def test_todas_malas(self):
        raw = _make_raw(
            horas_produccion=8.0,
            tiempo_ideal=8.0,
            piezas_totales=100.0,
            piezas_malas=100.0,
        )
        m = convertir_raw_a_metricas(raw)
        assert m["calidad_pct"] == 0.0


class TestOEECompuesto:
    """OEE = disponibilidad * rendimiento * calidad / 10000."""

    def test_oee_compuesto(self):
        # Disp=75%, Rend=80%, Cal=90% → OEE=54%
        raw = _make_raw(
            horas_produccion=8.0,
            horas_paros=2.0,  # reduce disponibilidad
            tiempo_ideal=4.8,  # 80% de 6h operativas
            piezas_totales=100.0,
            piezas_malas=10.0,
        )
        m = convertir_raw_a_metricas(raw)
        # OEE = disp * rend * cal / 10000
        expected_oee = m["disponibilidad_pct"] * m["rendimiento_pct"] * m["calidad_pct"] / 10000
        assert abs(m["oee_pct"] - expected_oee) < 0.01


class TestCasosLimite:
    """Edge cases: ceros, nulos, valores extremos."""

    def test_todo_cero(self):
        raw = crear_raw_metricas()
        m = convertir_raw_a_metricas(raw)
        assert m["disponibilidad_pct"] == 0.0
        assert m["rendimiento_pct"] == 0.0
        assert m["calidad_pct"] == 0.0
        assert m["oee_pct"] == 0.0

    def test_sin_produccion_solo_preparacion(self):
        raw = _make_raw(horas_preparacion=2.0)
        m = convertir_raw_a_metricas(raw)
        # No debe haber ZeroDivisionError
        assert m["oee_pct"] == 0.0

    def test_pocas_piezas_anula_oee(self):
        raw = _make_raw(
            horas_produccion=1.0,
            tiempo_ideal=1.0,
            piezas_totales=float(MIN_PIEZAS_OEE - 1),
        )
        m = convertir_raw_a_metricas(raw)
        # Con pocas piezas el OEE se anula
        assert m["oee_pct"] == 0.0

    def test_clamp_no_supera_100(self):
        assert clamp_pct(150.0) == 100.0
        assert clamp_pct(-10.0) == 0.0
        assert clamp_pct(50.0) == 50.0

    def test_horas_brutas_es_produccion_mas_preparacion(self):
        raw = _make_raw(horas_produccion=6.0, horas_preparacion=2.0, piezas_totales=100.0, tiempo_ideal=6.0)
        m = convertir_raw_a_metricas(raw)
        assert m["horas_brutas"] == 8.0

    def test_buenas_finales(self):
        raw = _make_raw(
            horas_produccion=8.0, tiempo_ideal=8.0,
            piezas_totales=100.0, piezas_malas=20.0, piezas_recuperadas=5.0,
        )
        m = convertir_raw_a_metricas(raw)
        # buenas = totales - (malas - recuperadas) = 100 - 15 = 85
        assert m["buenas_finales"] == 85.0
