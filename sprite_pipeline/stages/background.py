"""Etapa ``background``: segmenta al personaje y produce un canal alfa limpio.

Método por defecto ``"auto"``:

1. Color de fondo: ``meta["bg_color_estimate"]`` si existe (lo anota la etapa
   ``preprocess``) o mediana de los píxeles de los bordes del cuadro.
2. Distancia euclídea RGB de cada píxel al color de fondo; máscara inicial de
   fondo = ``distancia < tolerance``.
3. Flood fill desde los 4 bordes (componentes conexas) para quedarse SOLO con
   el fondo conectado a los bordes: los "agujeros" internos del sujeto (zonas
   de color parecido al fondo, p. ej. la panza clara) NO se vacían.
4. Sujeto = complemento; morfología close+open (kernel 3) y componente(s)
   principales para eliminar motas.
5. Alfa = máscara * 255 con el borde suavizado (GaussianBlur sigma≈1 aplicado
   solo en la banda del borde de la silueta).

Consistencia temporal: los píxeles cuya distancia cae en la banda ambigua
(``tolerance ± 10``) heredan la decisión (fondo/sujeto) del cuadro anterior,
lo que evita el parpadeo de silueta entre cuadros.

El RGB del resultado conserva el color original del sujeto.

``params["method"]`` queda reservado para métodos futuros ("rembg", "sam2");
pedir uno no disponible lanza ``ValueError``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sprite_pipeline.models import Frame, FrameSet, frame_filename
from sprite_pipeline.stages.base import Stage, load_frame_rgba, register_stage, save_frame_rgba

_KERNEL3 = np.ones((3, 3), np.uint8)
#: Semiancho de la banda ambigua alrededor de la tolerancia (prior temporal).
_PRIOR_BAND = 10.0
#: Área mínima de una componente del sujeto, relativa a la mayor, para conservarla.
_KEEP_RATIO = 0.02


def _estimate_bg_color(rgba: np.ndarray, border: int = 2) -> np.ndarray:
    """Mediana RGB de los píxeles de los 4 bordes del cuadro."""
    rgb = rgba[..., :3].astype(np.float32)
    strips = [
        rgb[:border].reshape(-1, 3),
        rgb[-border:].reshape(-1, 3),
        rgb[:, :border].reshape(-1, 3),
        rgb[:, -border:].reshape(-1, 3),
    ]
    return np.median(np.concatenate(strips, axis=0), axis=0)


def _border_connected(bg_mask: np.ndarray) -> np.ndarray:
    """Flood fill desde los bordes: fondo conectado a alguno de los 4 bordes.

    Equivale a cv2.floodFill sembrado en cada píxel de borde de la máscara de
    fondo, implementado con componentes conexas (conectividad 4 para que el
    fondo no se "cuele" en diagonal a través del contorno del sujeto).
    """
    _, labels = cv2.connectedComponents(bg_mask.astype(np.uint8), connectivity=4)
    border_labels = np.unique(
        np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    )
    border_labels = border_labels[border_labels != 0]  # 0 = píxeles que no son fondo
    if border_labels.size == 0:
        return np.zeros(bg_mask.shape, dtype=bool)
    return np.isin(labels, border_labels)


def _principal_components(mask: np.ndarray) -> np.ndarray:
    """Conserva la componente mayor y las que no sean despreciables frente a ella."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    keep = [i + 1 for i, area in enumerate(areas) if area >= _KEEP_RATIO * areas.max()]
    return np.isin(labels, keep)


def _feathered_alpha(mask: np.ndarray) -> np.ndarray:
    """Máscara binaria -> alfa uint8 con el borde suavizado.

    El GaussianBlur (sigma≈1) se aplica únicamente en la banda del borde de la
    silueta (dilatación - erosión); el interior queda en 255 y el fondo en 0.
    """
    m8 = mask.astype(np.uint8)
    m255 = m8 * 255
    blurred = cv2.GaussianBlur(m255, (0, 0), 1.0)
    band = cv2.dilate(m8, _KERNEL3) - cv2.erode(m8, _KERNEL3)
    return np.where(band > 0, blurred, m255).astype(np.uint8)


@register_stage
class BackgroundStage(Stage):
    """Elimina el fondo casi-uniforme y produce alfa limpio y estable."""

    name = "background"

    def process(self, fs: FrameSet, in_dir: Path, out_dir: Path, params: dict) -> FrameSet:
        method = params.get("method", "auto")
        if method != "auto":
            raise ValueError(
                f"Método de segmentación no disponible: {method!r}. "
                "Disponible: 'auto' ('rembg' y 'sam2' están reservados para el futuro)."
            )
        tolerance = float(params.get("tolerance", 35))

        meta = dict(fs.meta)
        meta_bg = meta.get("bg_color_estimate")

        prev_mask: np.ndarray | None = None  # silueta binaria del cuadro anterior
        new_frames: list[Frame] = []
        for i, frame in enumerate(fs.frames):
            rgba = load_frame_rgba(in_dir, frame)
            if meta_bg is not None:
                bg_color = np.asarray(meta_bg, dtype=np.float32)
            else:
                bg_color = _estimate_bg_color(rgba)

            dist = np.linalg.norm(rgba[..., :3].astype(np.float32) - bg_color, axis=2)
            bg = dist < tolerance

            # Prior temporal: en la banda ambigua se hereda la decisión previa.
            if prev_mask is not None and prev_mask.shape == bg.shape:
                ambiguous = np.abs(dist - tolerance) <= _PRIOR_BAND
                bg[ambiguous] = ~prev_mask[ambiguous]

            # Solo el fondo conectado a los bordes se vacía; los agujeros
            # internos del sujeto permanecen como sujeto.
            subject = ~_border_connected(bg)

            su8 = subject.astype(np.uint8)
            su8 = cv2.morphologyEx(su8, cv2.MORPH_CLOSE, _KERNEL3)
            su8 = cv2.morphologyEx(su8, cv2.MORPH_OPEN, _KERNEL3)
            subject = _principal_components(su8 > 0)
            prev_mask = subject

            alpha = _feathered_alpha(subject)
            out = rgba.copy()  # el RGB del sujeto se conserva tal cual
            out[..., 3] = np.minimum(rgba[..., 3], alpha)

            fname = frame_filename(i)
            save_frame_rgba(out_dir, fname, out)
            new_frames.append(
                Frame(
                    index=i,
                    file=fname,
                    duration_ms=frame.duration_ms,
                    flags=list(frame.flags),
                    scores=dict(frame.scores),
                    transform=dict(frame.transform) if frame.transform else None,
                    source=frame.source,
                )
            )
            if meta_bg is None and "bg_color_estimate" not in meta:
                meta["bg_color_estimate"] = [int(round(c)) for c in bg_color]

        return FrameSet(stage=self.name, frames=new_frames, meta=meta, history=list(fs.history))
