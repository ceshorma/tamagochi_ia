"""Etapa ``align``: registra cada cuadro contra un ancla común.

Elimina la deriva de posición/escala entre cuadros de la secuencia. Requiere
que el alfa ya esté presente (la etapa ``background`` corre antes).

Por cuadro:

- bbox del alfa > 0;
- anchor ``params.get("anchor", "feet")``:
  - ``"feet"``: (centro-x del bbox, borde inferior del bbox);
  - ``"center"``: centroide del alfa (ponderado por alfa);
- target = mediana de los anchors de la secuencia;
- escala: alto del bbox / mediana de altos; si ``|ratio - 1|`` supera
  ``params.get("scale_tolerance", 0.05)`` se reescala el cuadro alrededor del
  anchor (factor aplicado = mediana / alto); luego se traslada ``dx, dy`` para
  llevar el anchor al target.

La transformación se aplica con ``cv2.warpAffine`` (INTER_LINEAR, borde
transparente ``(0,0,0,0)``) sobre los 4 canales RGBA de una vez, y se anota en
``Frame.transform = {"dx": ..., "dy": ..., "scale": ...}``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sprite_pipeline.models import Frame, FrameSet, frame_filename
from sprite_pipeline.stages.base import Stage, load_frame_rgba, register_stage, save_frame_rgba

DEFAULT_ANCHOR = "feet"
DEFAULT_SCALE_TOLERANCE = 0.05
_VALID_ANCHORS = ("feet", "center")


def _anchor_and_height(rgba: np.ndarray, anchor_mode: str) -> tuple[tuple[float, float], float]:
    """Anchor (x, y) y alto del bbox del alfa. Alto 0.0 => cuadro sin alfa."""
    alpha = rgba[:, :, 3]
    ys, xs = np.nonzero(alpha > 0)
    if ys.size == 0:
        h, w = alpha.shape
        return (w / 2.0, h / 2.0), 0.0
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    height = y1 - y0 + 1.0
    if anchor_mode == "center":
        weights = alpha.astype(np.float64)
        total = weights.sum()
        cx = float((weights.sum(axis=0) * np.arange(alpha.shape[1])).sum() / total)
        cy = float((weights.sum(axis=1) * np.arange(alpha.shape[0])).sum() / total)
        return (cx, cy), height
    # "feet": centro-x del bbox, borde inferior del bbox
    return ((x0 + x1) / 2.0, y1), height


def _warp(rgba: np.ndarray, anchor: tuple[float, float], scale: float, dx: float, dy: float) -> np.ndarray:
    """Reescala alrededor de ``anchor`` y traslada (dx, dy), en una sola afín."""
    h, w = rgba.shape[:2]
    # p' = scale * (p - anchor) + anchor + (dx, dy)
    m = np.array(
        [
            [scale, 0.0, anchor[0] * (1.0 - scale) + dx],
            [0.0, scale, anchor[1] * (1.0 - scale) + dy],
        ],
        dtype=np.float64,
    )
    return cv2.warpAffine(
        rgba,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


@register_stage
class AlignStage(Stage):
    name = "align"

    def process(self, fs: FrameSet, in_dir: Path, out_dir: Path, params: dict) -> FrameSet:
        anchor_mode = params.get("anchor", DEFAULT_ANCHOR)
        if anchor_mode not in _VALID_ANCHORS:
            raise ValueError(
                f"anchor desconocido: {anchor_mode!r}. Válidos: {list(_VALID_ANCHORS)}"
            )
        tolerance = float(params.get("scale_tolerance", DEFAULT_SCALE_TOLERANCE))

        images = [load_frame_rgba(in_dir, f) for f in fs.frames]
        infos = [_anchor_and_height(img, anchor_mode) for img in images]

        valid = [(anchor, height) for anchor, height in infos if height > 0]
        if valid:
            target_x = float(np.median([a[0] for a, _ in valid]))
            target_y = float(np.median([a[1] for a, _ in valid]))
            median_height = float(np.median([h for _, h in valid]))
        else:
            target_x = target_y = median_height = 0.0

        new_frames: list[Frame] = []
        for i, (frame, img, (anchor, height)) in enumerate(zip(fs.frames, images, infos)):
            if height <= 0 or median_height <= 0:
                dx = dy = 0.0
                scale = 1.0
                out = img
            else:
                ratio = height / median_height
                scale = 1.0 if abs(ratio - 1.0) <= tolerance else median_height / height
                dx = target_x - anchor[0]
                dy = target_y - anchor[1]
                if scale == 1.0 and abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    out = img  # identidad: copia sin resampleo
                else:
                    out = _warp(img, anchor, scale, dx, dy)

            filename = frame_filename(i)
            save_frame_rgba(out_dir, filename, out)
            new_frames.append(
                Frame(
                    index=i,
                    file=filename,
                    duration_ms=frame.duration_ms,
                    flags=list(frame.flags),
                    scores=dict(frame.scores),
                    transform={
                        "dx": round(float(dx), 3),
                        "dy": round(float(dy), 3),
                        "scale": round(float(scale), 4),
                    },
                    source=frame.source,
                )
            )

        meta = dict(fs.meta)
        if valid:
            meta["align"] = {
                "anchor": anchor_mode,
                "target": [round(target_x, 2), round(target_y, 2)],
                "median_height": round(median_height, 2),
            }
        return FrameSet(stage=fs.stage, frames=new_frames, meta=meta, history=list(fs.history))
