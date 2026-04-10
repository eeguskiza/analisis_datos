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

    # SQL Server — ecs_mobility (app)
    db_server: str = "192.168.0.4"
    db_port: int = 1433
    db_name: str = "ecs_mobility"
    db_user: str = ""
    db_password: str = ""

    # IZARO (MES)
    izaro_db: str = "dbizaro"

    # PostgreSQL (Docker)
    pg_user: str = "oee"
    pg_password: str = "oee"
    pg_db: str = "oee_planta"

    # database_url se construye dinamicamente o se sobreescribe via env
    database_url: str = ""

    @property
    def effective_database_url(self) -> str:
        """URL para SQLAlchemy. Si database_url esta definida, la usa. Si no, construye desde campos."""
        if self.database_url:
            return self.database_url
        pwd = self.db_password.replace("+", "%2B")
        return (
            f"mssql+pyodbc://{self.db_user}:{pwd}@{self.db_server}:{self.db_port}/{self.db_name}"
            "?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes&Encrypt=yes"
        )

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
