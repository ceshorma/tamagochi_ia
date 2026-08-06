"""Configuración global del pipeline, resuelta desde variables de entorno."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("SPRITE_DATA_DIR", "data")))
    provider: str = field(default_factory=lambda: os.environ.get("SPRITE_PROVIDER", "mock"))
    default_fps: int = int(os.environ.get("SPRITE_FPS", "10"))
    default_duration_s: float = float(os.environ.get("SPRITE_DURATION_S", "2.0"))
    # Claves de proveedores externos (opcionales; solo necesarias para providers reales)
    runway_api_key: str = field(default_factory=lambda: os.environ.get("RUNWAY_API_KEY", ""))
    luma_api_key: str = field(default_factory=lambda: os.environ.get("LUMA_API_KEY", ""))
    # Tamaño de trabajo del sprite (lado mayor del canvas normalizado)
    work_size: int = int(os.environ.get("SPRITE_WORK_SIZE", "256"))

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


def reset_settings_cache() -> None:
    """Para tests: fuerza a releer las variables de entorno."""
    get_settings.cache_clear()
