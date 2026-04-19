"""DATA-02: MesRepository contrato + delegacion (Plan 03-02 Task 1, TDD RED)."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from nexo.data.repositories.mes import MesRepository


# ── Delegation tests — 4 wrappers over OEE.db.connector (D-04) ────────────

@patch("nexo.data.repositories.mes._legacy_extraer_datos")
@patch("nexo.data.repositories.mes._get_mes_config")
def test_extraer_datos_delega(mock_cfg, mock_extraer):
    mock_cfg.return_value = {"recursos": [{"nombre": "luk1"}]}
    mock_extraer.return_value = [{"recurso": "luk1", "fecha": "2026-04-01"}]

    repo = MesRepository(engine=MagicMock())
    rows = repo.extraer_datos_produccion(
        fecha_inicio=date(2026, 4, 1),
        fecha_fin=date(2026, 4, 1),
        recursos=["luk1"],
    )
    assert len(rows) == 1
    mock_extraer.assert_called_once()


@patch("nexo.data.repositories.mes._legacy_detectar_recursos")
@patch("nexo.data.repositories.mes._get_mes_config")
def test_detectar_recursos_delega(mock_cfg, mock_detectar):
    mock_cfg.return_value = {}
    mock_detectar.return_value = [{"codigo": 47}]
    repo = MesRepository(engine=MagicMock())
    assert repo.detectar_recursos() == [{"codigo": 47}]
    mock_detectar.assert_called_once()


@patch("nexo.data.repositories.mes._legacy_calcular_ciclos")
@patch("nexo.data.repositories.mes._get_mes_config")
def test_calcular_ciclos_delega(mock_cfg, mock_ciclos):
    mock_cfg.return_value = {}
    mock_ciclos.return_value = ([{"ref": "X", "ciclo_seg": 10.5}], "fmesdtc")
    repo = MesRepository(engine=MagicMock())
    rows, fuente = repo.calcular_ciclos_reales(centro_trabajo=47, dias_atras=30)
    assert rows[0]["ciclo_seg"] == 10.5
    assert fuente == "fmesdtc"
    mock_ciclos.assert_called_once_with({}, 47, 30)


@patch("nexo.data.repositories.mes._legacy_estado_maquina")
@patch("nexo.data.repositories.mes._get_mes_config")
def test_estado_maquina_delega(mock_cfg, mock_estado):
    mock_cfg.return_value = {}
    mock_estado.return_value = {"activa": True, "piezas_hoy": 5}
    repo = MesRepository(engine=MagicMock())
    assert repo.estado_maquina_live(47, 600) == {"activa": True, "piezas_hoy": 5}


# ── consulta_readonly shape test (D-05) ───────────────────────────────────


def test_consulta_readonly_shape():
    engine = MagicMock()
    cursor = engine.connect.return_value.__enter__.return_value
    cursor.execute.return_value.keys.return_value = ["c1", "c2"]
    cursor.execute.return_value.fetchall.return_value = [(1, "a"), (2, "b")]

    repo = MesRepository(engine=engine)
    out = repo.consulta_readonly("SELECT 1", "dbizaro")
    assert out["columns"] == ["c1", "c2"]
    assert out["rows"] == [[1, "a"], [2, "b"]]


# ── centro_mando_fmesmic — load_sql + bindparam expanding (DATA-09) ───────


def test_centro_mando_fmesmic_empty_returns_empty():
    repo = MesRepository(engine=MagicMock())
    assert repo.centro_mando_fmesmic([]) == []


@patch("nexo.data.repositories.mes.load_sql")
def test_centro_mando_fmesmic_uses_bindparam(mock_load_sql):
    mock_load_sql.return_value = "SELECT * FROM admuser.fmesmic WHERE ct IN :codes"
    engine = MagicMock()
    exec_ret = engine.connect.return_value.__enter__.return_value.execute.return_value

    fake_row = MagicMock()
    fake_row._mapping = {"ct": 47, "piezas_hoy": 10}
    exec_ret.fetchall.return_value = [fake_row]

    repo = MesRepository(engine=engine)
    rows = repo.centro_mando_fmesmic([47, 48])
    assert rows == [{"ct": 47, "piezas_hoy": 10}]
    mock_load_sql.assert_called_once_with("mes/centro_mando_fmesmic")


# ── Public method contract ────────────────────────────────────────────────


def test_mes_repository_public_api():
    """MesRepository expone los 6 metodos requeridos por el plan."""
    expected = {
        "extraer_datos_produccion",
        "detectar_recursos",
        "calcular_ciclos_reales",
        "estado_maquina_live",
        "consulta_readonly",
        "centro_mando_fmesmic",
    }
    actual = {m for m in dir(MesRepository) if not m.startswith("_")}
    missing = expected - actual
    assert not missing, f"Missing public methods: {missing}"


# ── DTOs frozen ──────────────────────────────────────────────────────────


def test_dtos_are_frozen():
    from nexo.data.dto.mes import (
        CapacidadRow,
        CicloRealRow,
        EstadoMaquinaRow,
        OperarioRow,
        ProduccionRow,
    )

    for model in (ProduccionRow, EstadoMaquinaRow, CicloRealRow, CapacidadRow, OperarioRow):
        assert model.model_config.get("frozen") is True, f"{model.__name__} no frozen"
        assert model.model_config.get("from_attributes") is True, (
            f"{model.__name__} sin from_attributes"
        )


# ── Scripts PDF existentes (Task 1 prerequisito de Task 2) ────────────────


def test_pdf_scripts_exist():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    assert (root / "scripts" / "gen_pdf_reference.py").is_file()
    assert (root / "scripts" / "pdf_regression_check.py").is_file()


def test_pdf_scripts_parse():
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    for fname in ("gen_pdf_reference.py", "pdf_regression_check.py"):
        source = (root / "scripts" / fname).read_text(encoding="utf-8")
        ast.parse(source)  # raises SyntaxError if bad
