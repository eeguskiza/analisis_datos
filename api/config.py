"""Configuracion centralizada de Nexo via variables de entorno.

Durante Mark-III convive compat layer: cada campo acepta el nombre nuevo
`NEXO_*` y cae al legado `OEE_*` si el nuevo no esta definido. Esto
permite migrar el ``.env`` de disco sin romper instalaciones existentes.
La compat se elimina en Mark-IV.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    # ── Paths ─────────────────────────────────────────────────────────────
    project_root: Path = _PROJECT_ROOT
    data_dir: Path = Field(
        default=_PROJECT_ROOT / "data",
        validation_alias=AliasChoices("NEXO_DATA_DIR", "OEE_DATA_DIR"),
    )
    informes_dir: Path = Field(
        default=_PROJECT_ROOT / "informes",
        validation_alias=AliasChoices("NEXO_INFORMES_DIR", "OEE_INFORMES_DIR"),
    )

    # Nombre de fichero del logo que los modulos OEE estampan en los PDFs
    # (matplotlib lee desde data_dir / logo_filename). Sigue siendo
    # "ecs-logo.png" hasta que Sprint 2 unifique con static/img/brand/ecs/.
    logo_filename: str = "ecs-logo.png"

    # ── Servidor web ──────────────────────────────────────────────────────
    host: str = Field("0.0.0.0", validation_alias=AliasChoices("NEXO_HOST", "OEE_HOST"))
    port: int = Field(8000, validation_alias=AliasChoices("NEXO_PORT", "OEE_PORT"))
    debug: bool = Field(False, validation_alias=AliasChoices("NEXO_DEBUG", "OEE_DEBUG"))
    # SQL echo: desacoplado de debug. Con NEXO_DEBUG=true la app loguea rutas,
    # templates y errores verbose, pero no floodea con SELECT/BEGIN/COMMIT.
    # Para ver SQL en crudo: NEXO_LOG_SQL=true (solo dev, nunca en prod).
    log_sql: bool = Field(False, validation_alias=AliasChoices("NEXO_LOG_SQL"))

    # ── SQL Server — APP (ecs_mobility) ──────────────────────────────────
    # Conjunto "APP" = BD propia de Nexo con tablas cfg/oee/luk4.
    app_server: str = Field(
        "192.168.0.4",
        validation_alias=AliasChoices("NEXO_APP_SERVER", "OEE_DB_SERVER"),
    )
    app_port: int = Field(
        1433,
        validation_alias=AliasChoices("NEXO_APP_PORT", "OEE_DB_PORT"),
    )
    app_db: str = Field(
        "ecs_mobility",
        validation_alias=AliasChoices("NEXO_APP_DB", "OEE_DB_NAME"),
    )
    app_user: str = Field(
        "",
        validation_alias=AliasChoices("NEXO_APP_USER", "OEE_DB_USER"),
    )
    app_password: str = Field(
        "",
        validation_alias=AliasChoices("NEXO_APP_PASSWORD", "OEE_DB_PASSWORD"),
    )

    # ── SQL Server — MES (dbizaro, read-only) ─────────────────────────────
    # Conjunto "MES" = BD externa de IZARO. Hoy misma instancia que APP,
    # preparado para split en Mark-IV.
    mes_server: str = Field(
        "192.168.0.4",
        validation_alias=AliasChoices("NEXO_MES_SERVER", "OEE_DB_SERVER"),
    )
    mes_port: int = Field(
        1433,
        validation_alias=AliasChoices("NEXO_MES_PORT", "OEE_DB_PORT"),
    )
    mes_db: str = Field(
        "dbizaro",
        validation_alias=AliasChoices("NEXO_MES_DB", "OEE_IZARO_DB"),
    )
    mes_user: str = Field(
        "",
        validation_alias=AliasChoices("NEXO_MES_USER", "OEE_DB_USER"),
    )
    mes_password: str = Field(
        "",
        validation_alias=AliasChoices("NEXO_MES_PASSWORD", "OEE_DB_PASSWORD"),
    )

    # ── PostgreSQL (Docker) ───────────────────────────────────────────────
    pg_host: str = Field(
        "db",
        validation_alias=AliasChoices("NEXO_PG_HOST"),
    )
    pg_port: int = Field(
        5432,
        validation_alias=AliasChoices("NEXO_PG_PORT"),
    )
    pg_user: str = Field(
        "oee",
        validation_alias=AliasChoices("NEXO_PG_USER", "OEE_PG_USER"),
    )
    pg_password: str = Field(
        "oee",
        validation_alias=AliasChoices("NEXO_PG_PASSWORD", "OEE_PG_PASSWORD"),
    )
    pg_db: str = Field(
        "oee_planta",
        validation_alias=AliasChoices("NEXO_PG_DB", "OEE_PG_DB"),
    )

    # ── Postgres — rol 'app' dedicado (Plan 02-04, gate IDENT-06) ─────────
    # ``pg_app_user`` / ``pg_app_password`` son las credenciales que usa el
    # servidor web (engine_nexo) para conectar en runtime. ``nexo_app`` es
    # un rol con GRANTs limitados (SELECT/INSERT/UPDATE/DELETE en todo el
    # schema nexo EXCEPTO UPDATE/DELETE en nexo.audit_log — append-only).
    #
    # ``pg_user`` / ``pg_password`` siguen siendo las credenciales del
    # owner (``oee``), usadas por scripts de bootstrap (init_nexo_schema,
    # create_propietario) y por el container db como POSTGRES_USER.
    #
    # Si ``NEXO_PG_APP_USER`` no esta definido en .env, el engine usa
    # automaticamente ``pg_user``/``pg_password`` (backwards compat con
    # deploys anteriores al Plan 02-04).
    pg_app_user: str = Field(
        "",
        validation_alias=AliasChoices("NEXO_PG_APP_USER"),
    )
    pg_app_password: str = Field(
        "",
        validation_alias=AliasChoices("NEXO_PG_APP_PASSWORD"),
    )

    @property
    def effective_pg_user(self) -> str:
        return self.pg_app_user or self.pg_user

    @property
    def effective_pg_password(self) -> str:
        return self.pg_app_password or self.pg_password

    # ── Auth (Phase 2 — Sprint 1) ─────────────────────────────────────────
    # NEXO_SECRET_KEY NO tiene default: si falta en .env, FastAPI arranca
    # con ValidationError claro. Generar con:
    #     python -c "import secrets; print(secrets.token_urlsafe(48))"
    secret_key: str = Field(
        ...,
        validation_alias=AliasChoices("NEXO_SECRET_KEY"),
    )
    session_cookie_name: str = Field(
        "nexo_session",
        validation_alias=AliasChoices("NEXO_SESSION_COOKIE_NAME"),
    )
    session_ttl_hours: int = Field(
        12,
        validation_alias=AliasChoices("NEXO_SESSION_TTL_HOURS"),
    )
    session_cookie_secure: bool = Field(
        True,
        validation_alias=AliasChoices("NEXO_SESSION_COOKIE_SECURE"),
    )

    # ── Branding (Sprint 0 introduce los campos; Sprint 0 commit 7 los cablea en templates) ──
    app_name: str = Field("Nexo", validation_alias=AliasChoices("NEXO_APP_NAME"))
    company_name: str = Field(
        "ECS Mobility",
        validation_alias=AliasChoices("NEXO_COMPANY_NAME"),
    )
    nexo_logo_path: str = Field(
        "/static/img/brand/nexo/logo.png",
        validation_alias=AliasChoices("NEXO_LOGO_PATH"),
    )
    nexo_ecs_logo_path: str = Field(
        "/static/img/brand/ecs/logo.png",
        validation_alias=AliasChoices("NEXO_ECS_LOGO_PATH"),
    )

    # ── Override opcional para SQLAlchemy ──────────────────────────────────
    database_url: str = Field(
        "",
        validation_alias=AliasChoices("NEXO_DATABASE_URL", "OEE_DATABASE_URL"),
    )

    # ── Compat shims para codigo que aun lee los nombres viejos ───────────
    # Eliminar tras refactor de capa de datos (Sprint 2).
    @property
    def db_server(self) -> str:
        return self.app_server

    @property
    def db_port(self) -> int:
        return self.app_port

    @property
    def db_name(self) -> str:
        return self.app_db

    @property
    def db_user(self) -> str:
        return self.app_user

    @property
    def db_password(self) -> str:
        return self.app_password

    @property
    def izaro_db(self) -> str:
        return self.mes_db

    # ── Propiedades derivadas ─────────────────────────────────────────────
    @property
    def effective_database_url(self) -> str:
        """URL para SQLAlchemy. Si ``database_url`` esta definida, la usa.
        Si no, construye desde los campos APP (ecs_mobility)."""
        if self.database_url:
            return self.database_url
        pwd = self.app_password.replace("+", "%2B")
        return (
            f"mssql+pyodbc://{self.app_user}:{pwd}@{self.app_server}:{self.app_port}/{self.app_db}"
            "?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes&Encrypt=yes"
        )

    @property
    def logo_path(self) -> Path | None:
        """Ruta fisica al logo ECS que estampa matplotlib en los PDFs de
        los modulos OEE. Apunta a ``data/ecs-logo.png`` mientras
        ``data/`` sea la fuente canonica. Unificacion con
        ``static/img/brand/ecs/logo.png`` pendiente para Sprint 2."""
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

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
