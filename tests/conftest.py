"""Fixtures compartidos: data dir temporal y criatura de muestra."""

from __future__ import annotations

from pathlib import Path

import pytest

from sprite_pipeline.config import reset_settings_cache
from sprite_pipeline.sample import save_sample_image


@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Aísla settings.data_dir en tmp_path para cada test."""
    d = tmp_path / "data"
    monkeypatch.setenv("SPRITE_DATA_DIR", str(d))
    reset_settings_cache()
    yield d
    reset_settings_cache()


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    """Imagen de la criatura de muestra en pose neutra."""
    return save_sample_image(tmp_path / "creature.png", size=200)
