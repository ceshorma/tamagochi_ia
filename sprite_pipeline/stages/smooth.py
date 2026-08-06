"""Etapa ``smooth``: estabilización temporal sin destruir detalle.

Qué hace (ver docs/API_SPEC.md):

- Normaliza el brillo global de cada cuadro hacia la mediana de la secuencia.
  El brillo de un cuadro es la media de luma (BT.601) de los píxeles del
  sujeto (alfa > 0). La ganancia por cuadro es ``mediana / brillo`` limitada
  a ±10 % (clip a [0.9, 1.1]) y se aplica solo a RGB (el alfa no se toca),
  con clip final a uint8.
- Suaviza el borde del canal alfa (feather leve, ``params["feather"]`` px de
  sigma, default 1.5): GaussianBlur del alfa aplicado SOLO en la banda del
  borde detectada con un gradiente morfológico, de modo que el interior
  profundo sigue en 255 y el exterior lejano en 0.
- Uniformiza ``duration_ms`` a la media redondeada de la secuencia, salvo que
  ``params["keep_timing"]`` sea verdadero.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from sprite_pipeline.models import Frame, FrameSet, frame_filename
from sprite_pipeline.stages.base import (
    Stage,
    load_frame_rgba,
    register_stage,
    save_frame_rgba,
)

# Pesos de luma BT.601 sobre RGB.
_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float64)

# Ganancia de brillo limitada a ±10 %.
GAIN_MIN = 0.9
GAIN_MAX = 1.1

_EPS = 1e-6


def _subject_brightness(rgba: np.ndarray) -> float | None:
    """Media de luma de los píxeles del sujeto (alfa > 0); ``None`` sin sujeto."""
    mask = rgba[..., 3] > 0
    if not mask.any():
        return None
    luma = rgba[..., :3].astype(np.float64) @ _LUMA_WEIGHTS
    return float(luma[mask].mean())


def _apply_gain(rgba: np.ndarray, gain: float) -> np.ndarray:
    """Aplica una ganancia a RGB (alfa intacto) con redondeo y clip a uint8."""
    out = rgba.copy()
    if not math.isclose(gain, 1.0, abs_tol=1e-9):
        rgb = out[..., :3].astype(np.float64) * gain
        out[..., :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return out


def _feather_alpha(rgba: np.ndarray, sigma: float) -> np.ndarray:
    """Feather del alfa solo en la banda del borde (gradiente morfológico).

    Muta y devuelve ``rgba`` (que ya es una copia de trabajo del llamador).
    Fuera de la banda nada cambia: el interior profundo queda en 255 y el
    exterior lejano en 0.
    """
    if sigma <= 0:
        return rgba
    alpha = rgba[..., 3]
    radius = max(1, int(math.ceil(sigma)))
    ksize = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    gradient = cv2.morphologyEx(alpha, cv2.MORPH_GRADIENT, kernel)
    band = gradient > 0  # banda del borde: transición 0 < alfa < 255 tras el gradiente
    if not band.any():
        return rgba
    blurred = cv2.GaussianBlur(alpha, (0, 0), sigmaX=float(sigma))
    feathered = alpha.copy()
    feathered[band] = blurred[band]
    rgba[..., 3] = feathered
    return rgba


@register_stage
class SmoothStage(Stage):
    """Estabilización temporal: brillo hacia la mediana + feather del alfa."""

    name = "smooth"

    def process(self, fs: FrameSet, in_dir: Path, out_dir: Path, params: dict) -> FrameSet:
        feather = float(params.get("feather", 1.5))
        keep_timing = bool(params.get("keep_timing", False))

        images = [load_frame_rgba(in_dir, frame) for frame in fs.frames]
        brightness = [_subject_brightness(img) for img in images]
        valid = [b for b in brightness if b is not None and b > _EPS]
        median = float(np.median(valid)) if valid else None

        if keep_timing or not fs.frames:
            durations = [frame.duration_ms for frame in fs.frames]
        else:
            mean_ms = sum(frame.duration_ms for frame in fs.frames) / len(fs.frames)
            durations = [max(1, round(mean_ms))] * len(fs.frames)

        new_frames: list[Frame] = []
        for i, (frame, img, bright) in enumerate(zip(fs.frames, images, brightness)):
            if median is not None and bright is not None and bright > _EPS:
                gain = float(np.clip(median / bright, GAIN_MIN, GAIN_MAX))
            else:
                gain = 1.0
            out = _apply_gain(img, gain)
            out = _feather_alpha(out, feather)
            filename = frame_filename(i)
            save_frame_rgba(out_dir, filename, out)
            scores = dict(frame.scores)
            scores["smooth_gain"] = round(gain, 4)
            new_frames.append(
                Frame(
                    index=i,
                    file=filename,
                    duration_ms=durations[i],
                    flags=list(frame.flags),
                    scores=scores,
                    transform=dict(frame.transform) if frame.transform else None,
                    source=frame.source,
                )
            )

        return FrameSet(
            stage=self.name,
            frames=new_frames,
            meta=dict(fs.meta),
            history=list(fs.history),
            schema_version=fs.schema_version,
        )
