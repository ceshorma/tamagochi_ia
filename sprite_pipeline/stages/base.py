"""Contrato base de las etapas del pipeline.

Una etapa transforma un FrameSet de un directorio de entrada a uno de salida:

    class MiEtapa(Stage):
        name = "mietapa"
        def process(self, fs, in_dir, out_dir, params) -> FrameSet: ...

Reglas para implementadores:

- ``process`` NO debe mutar los archivos de ``in_dir``; escribe los PNGs
  resultantes en ``out_dir`` con los nombres que declare en los ``Frame.file``
  del FrameSet devuelto (usar ``frame_filename(i)``).
- Las imágenes en memoria son arrays numpy RGBA uint8. Usar los helpers
  ``load_frame_rgba`` / ``save_frame_rgba`` para el I/O; si internamente se
  usa cv2 (BGR), convertir explícitamente.
- ``process`` devuelve el FrameSet nuevo SIN llamar a with_stage/save:
  ``run_stage`` se encarga del historial y de persistir el manifiesto.
- Una etapa debe ser idempotente: misma entrada + mismos params -> misma salida.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image

from sprite_pipeline.models import Frame, FrameSet, frame_filename  # noqa: F401


class Stage(ABC):
    name: str = "base"

    @abstractmethod
    def process(self, fs: FrameSet, in_dir: Path, out_dir: Path, params: dict) -> FrameSet:
        ...


STAGE_REGISTRY: dict[str, Stage] = {}


def register_stage(stage_cls: type[Stage]) -> type[Stage]:
    """Decorador: instancia y registra la etapa por su ``name``."""
    instance = stage_cls()
    STAGE_REGISTRY[instance.name] = instance
    return stage_cls


def get_stage(name: str) -> Stage:
    if name not in STAGE_REGISTRY:
        # Registro perezoso: el módulo de la etapa se importa bajo demanda.
        import importlib

        try:
            importlib.import_module(f"sprite_pipeline.stages.{name}")
        except ModuleNotFoundError:
            pass
    try:
        return STAGE_REGISTRY[name]
    except KeyError:
        raise KeyError(f"Etapa desconocida: {name!r}. Registradas: {sorted(STAGE_REGISTRY)}") from None


def run_stage(stage: Stage, in_dir: Path, out_dir: Path, params: dict | None = None) -> FrameSet:
    """Ejecuta una etapa: carga manifiesto, procesa, anota historial y guarda.

    La escritura es atómica respecto a ``out_dir``: se procesa hacia un dir
    temporal hermano (``<out_dir>.tmp``) y solo tras guardar el manifiesto se
    renombra a ``out_dir`` (reemplazando un ``out_dir`` previo si existiera).
    Así nunca queda un stage dir parcial sin ``frameset.json`` que envenene
    ``latest_stage_dir()``/``working_dir()``, y una etapa fallida se puede
    reintentar sin limpieza manual.
    """
    params = params or {}
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)  # restos de una corrida anterior fallida
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        fs = FrameSet.load(in_dir)
        result = stage.process(fs, in_dir, tmp_dir, params)
        result = result.with_stage(stage.name, params)
        result.reindex()
        result.save(tmp_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)
    return result


# ------------------------------------------------------------------ imagen I/O

def load_frame_rgba(directory: Path, frame: Frame) -> np.ndarray:
    """Carga el PNG de un cuadro como array RGBA uint8 (alto, ancho, 4)."""
    img = Image.open(Path(directory) / frame.file).convert("RGBA")
    return np.asarray(img, dtype=np.uint8).copy()


def save_frame_rgba(directory: Path, filename: str, rgba: np.ndarray) -> None:
    """Guarda un array RGBA uint8 como PNG."""
    if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"Se esperaba array RGBA uint8 (h,w,4); llegó {rgba.dtype} {rgba.shape}")
    Image.fromarray(rgba, mode="RGBA").save(Path(directory) / filename)
