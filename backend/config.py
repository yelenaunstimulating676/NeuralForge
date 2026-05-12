"""
Configurazione centralizzata di NeuralForge.

Legge variabili d'ambiente e .env tramite pydantic-settings.
Tutti i path relativi sono risolti rispetto alla cartella `backend/`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Settings caricate da env vars e .env. Prefisso NEURALFORGE_."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="NEURALFORGE_",
        case_sensitive=False,
        extra="ignore",
    )

    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = True

    # CORS
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/neuralforge.log"

    # Database
    database_url: str = "sqlite:///data/neuralforge.db"

    # Storage
    data_dir: str = "data"
    uploads_dir: str = "data/uploads"
    datasets_dir: str = "data/datasets"
    models_dir: str = "data/models"
    adapters_dir: str = "data/adapters"
    exports_dir: str = "data/exports"

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level deve essere uno di {allowed}")
        return v_upper

    def _resolve(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else (BACKEND_ROOT / p).resolve()

    @property
    def data_path(self) -> Path:
        return self._resolve(self.data_dir)

    @property
    def uploads_path(self) -> Path:
        return self._resolve(self.uploads_dir)

    @property
    def datasets_path(self) -> Path:
        return self._resolve(self.datasets_dir)

    @property
    def models_path(self) -> Path:
        return self._resolve(self.models_dir)

    @property
    def adapters_path(self) -> Path:
        return self._resolve(self.adapters_dir)

    @property
    def exports_path(self) -> Path:
        return self._resolve(self.exports_dir)

    @property
    def log_file_path(self) -> Path:
        return self._resolve(self.log_file)
    
    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def database_url_resolved(self) -> str:
        prefix = "sqlite:///"
        if self.database_url.startswith(prefix):
            rel = self.database_url[len(prefix):]
            abs_path = self._resolve(rel)
            return f"{prefix}{abs_path.as_posix()}"
        return self.database_url

    def ensure_directories(self) -> None:
        for path in (
            self.data_path,
            self.uploads_path,
            self.datasets_path,
            self.models_path,
            self.adapters_path,
            self.exports_path,
            self.log_file_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()