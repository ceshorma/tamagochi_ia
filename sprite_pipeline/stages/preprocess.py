"""Etapa ``preprocess``: normaliza encuadre y tamaño de la secuencia.

Para cada cuadro:

- Estima el color de fondo como la mediana RGB de un marco de ``BORDER_PX``
  píxeles pegado a los bordes.
- Máscara de sujeto: distancia euclídea RGB al fondo estimado mayor que
  ``params["bg_threshold"]`` (default 30) **o** alfa < 255 ya presente.
- Calcula el bounding box de la máscara (si queda vacía, el cuadro completo).

Todos los cuadros de la secuencia se recortan con el **mismo bbox-unión**
(unión de los bboxes de todos los cuadros) para no introducir jitter de
encuadre; luego se reescala manteniendo aspecto para caber en un canvas
cuadrado de lado ``work_size`` con margen relativo ``params["margin"]``
(default 0.08) y se centra.

IMPORTANTE: en esta etapa el fondo AÚN no se ha eliminado, así que el canvas
fuera del recorte se rellena con el color de fondo estimado (opaco, no
transparente) para no romper la etapa ``background``.

Anota en ``meta``: ``size = [work_size, work_size]`` y
``bg_color_estimate = [r, g, b]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from sprite_pipeline.config import get_settings
from sprite_pipeline.models import Frame, FrameSet, frame_filename
from sprite_pipeline.stages.base import Stage, load_frame_rgba, register_stage, save_frame_rgba

BORDER_PX = 4


def _estimate_bg_color(rgba: np.ndarray, border: int = BORDER_PX) -> np.ndarray:
    """Mediana RGB de un marco de ``border`` px pegado a los bordes del cuadro."""
    h, w = rgba.shape[:2]
    b = max(1, min(border, h, w))
    strips = [
        rgba[:b, :, :3].reshape(-1, 3),
        rgba[-b:, :, :3].reshape(-1, 3),
        rgba[:, :b, :3].reshape(-1, 3),
        rgba[:, -b:, :3].reshape(-1, 3),
    ]
    pixels = np.concatenate(strips, axis=0).astype(np.float64)
    return np.median(pixels, axis=0)


def _subject_mask(rgba: np.ndarray, bg_rgb: np.ndarray, threshold: float) -> np.ndarray:
    """Sujeto = lejos del color de fondo en RGB, o con alfa parcial ya presente."""
    rgb = rgba[..., :3].astype(np.float64)
    dist = np.sqrt(((rgb - np.asarray(bg_rgb, dtype=np.float64)) ** 2).sum(axis=2))
    return (dist > threshold) | (rgba[..., 3] < 255)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(y0, x0, y1, x1) inclusivo del área verdadera de la máscara, o None si vacía."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


@register_stage
class PreprocessStage(Stage):
    name = "preprocess"

    def process(self, fs: FrameSet, in_dir: Path, out_dir: Path, params: dict) -> FrameSet:
        work_size = int(params.get("work_size") or get_settings().work_size)
        margin = float(params.get("margin", 0.08))
        threshold = float(params.get("bg_threshold", 30))

        meta = dict(fs.meta)
        meta["size"] = [work_size, work_size]

        if not fs.frames:
            meta["bg_color_estimate"] = [0, 0, 0]
            return FrameSet(stage=fs.stage, frames=[], meta=meta,
                            history=list(fs.history), schema_version=fs.schema_version)

        images = [load_frame_rgba(in_dir, f) for f in fs.frames]
        bg_estimates = [_estimate_bg_color(img) for img in images]

        # bbox-unión de toda la secuencia: mismo encuadre para todos los cuadros.
        union: tuple[int, int, int, int] | None = None
        for img, bg in zip(images, bg_estimates):
            bbox = _mask_bbox(_subject_mask(img, bg, threshold))
            if bbox is None:  # máscara vacía -> cuadro completo
                bbox = (0, 0, img.shape[0] - 1, img.shape[1] - 1)
            if union is None:
                union = bbox
            else:
                union = (
                    min(union[0], bbox[0]),
                    min(union[1], bbox[1]),
                    max(union[2], bbox[2]),
                    max(union[3], bbox[3]),
                )
        assert union is not None
        uy0, ux0, uy1, ux1 = union

        # Lado interior disponible tras el margen relativo.
        inner = max(1, min(work_size, int(round(work_size * (1.0 - 2.0 * margin)))))

        new_frames: list[Frame] = []
        for i, (frame, img, bg) in enumerate(zip(fs.frames, images, bg_estimates)):
            h, w = img.shape[:2]
            y0, y1 = min(uy0, h - 1), min(uy1, h - 1)
            x0, x1 = min(ux0, w - 1), min(ux1, w - 1)
            crop = img[y0:y1 + 1, x0:x1 + 1]
            ch, cw = crop.shape[:2]

            scale = inner / max(ch, cw)
            nw = max(1, min(work_size, int(round(cw * scale))))
            nh = max(1, min(work_size, int(round(ch * scale))))
            resized = np.asarray(
                Image.fromarray(crop, mode="RGBA").resize((nw, nh), Image.LANCZOS),
                dtype=np.uint8,
            )

            # Canvas relleno con el fondo estimado (opaco): la etapa background
            # necesita el fondo intacto para segmentar.
            bg_u8 = np.clip(np.round(bg), 0, 255).astype(np.uint8)
            canvas = np.empty((work_size, work_size, 4), dtype=np.uint8)
            canvas[..., 0] = bg_u8[0]
            canvas[..., 1] = bg_u8[1]
            canvas[..., 2] = bg_u8[2]
            canvas[..., 3] = 255

            ox = (work_size - nw) // 2
            oy = (work_size - nh) // 2
            canvas[oy:oy + nh, ox:ox + nw] = resized

            fname = frame_filename(i)
            save_frame_rgba(out_dir, fname, canvas)
            new_frames.append(Frame(
                index=i,
                file=fname,
                duration_ms=frame.duration_ms,
                flags=list(frame.flags),
                scores=dict(frame.scores),
                transform=dict(frame.transform) if frame.transform else None,
                source=frame.source,
            ))

        overall_bg = np.median(np.stack(bg_estimates, axis=0), axis=0)
        meta["bg_color_estimate"] = [int(round(float(v))) for v in overall_bg]

        return FrameSet(stage=fs.stage, frames=new_frames, meta=meta,
                        history=list(fs.history), schema_version=fs.schema_version)
