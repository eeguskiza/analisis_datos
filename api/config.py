"""Configuracion centralizada de la aplicacion via variables de entorno."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    project_root: Path = _PROJECT_ROOT
    data_dir: Path = _PROJECT_ROOT / "data"
    informes_dir: Path = _PROJECT_ROOT / "informes"
    logo_filename: str = "ecs-logo.png"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    database_url: str = (
        "mssql+pyodbc://sa:AdmS1552%2B@192.168.0.4:1433/oee_ecs"
        "?driver=ODBC+Driver+18+for+SQL+Server"
        "&TrustServerCertificate=yes"
        "&Encrypt=yes"
    )
    # Fallback local: f"sqlite:///{_PROJECT_ROOT / 'data' / 'oee.db'}"

    @property
    def logo_path(self) -> Path | None:
        p = self.data_dir / self.logo_filename
        return p if p.exists() else None

    @property
    def recursos_dir(self) -> Path:
        return self.data_dir / "recursos"

    @property
    def ciclos_path(self) -> Path:
        return self.data_dir / "ciclos.csv"

    @property
    def templates_dir(self) -> Path:
        return self.data_dir / "report_templates"

    model_config = {"env_prefix": "OEE_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
