"""Etapa ``anomaly``: puntúa cada cuadro y marca los anómalos.

NO modifica píxeles: cada PNG se copia byte-idéntico a ``out_dir``
(``shutil.copy2``). Solo se anotan ``Frame.scores`` / ``Frame.flags`` y un
resumen en ``fs.meta["anomaly_summary"]``.

Scores por cuadro (todos en [0, 1]):

- ``identity``: similitud con una referencia. Si ``fs.meta["source_image"]``
  apunta a un archivo existente se usa esa imagen; si no, el cuadro cuyo
  histograma esté más cerca de la mediana de la secuencia.
  ``0.6 * sim_histograma_HSV + 0.4 * sim_area_silueta``, donde el histograma
  HSV se calcula solo sobre píxeles del sujeto (alfa > 0) y se compara con
  ``cv2.compareHist(HISTCMP_CORREL)`` reescalado de [-1, 1] a [0, 1].
- ``coherence``: media de similitud con los vecinos prev/next:
  ``1 - mean(|a - b| sobre la unión de alfas) / 255``.
- ``quality``: ``0.5 * identity + 0.5 * coherence`` recortado a [0, 1].

Cuadros con ``quality < params.get("threshold", 0.55)`` reciben los flags
``"anomaly"`` y ``"review"`` (sin duplicar). Nunca se eliminan cuadros.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sprite_pipeline.models import FrameSet
from sprite_pipeline.stages.base import Stage, load_frame_rgba, register_stage

DEFAULT_THRESHOLD = 0.55

_HIST_CHANNELS = [0, 1, 2]
_HIST_BINS = [16, 8, 8]
_HIST_RANGES = [0, 180, 0, 256, 0, 256]


def _hist_and_area(rgba: np.ndarray) -> tuple[np.ndarray, int]:
    """Histograma HSV normalizado del sujeto (alfa > 0) y área de la silueta."""
    alpha = rgba[:, :, 3]
    mask = np.where(alpha > 0, np.uint8(255), np.uint8(0))
    area = int(np.count_nonzero(mask))
    hsv = cv2.cvtColor(np.ascontiguousarray(rgba[:, :, :3]), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], _HIST_CHANNELS, mask, _HIST_BINS, _HIST_RANGES)
    total = float(hist.sum())
    if total > 0:
        hist = hist / total
    return hist.astype(np.float32), area


def _hist_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """Correlación de histogramas reescalada de [-1, 1] a [0, 1]."""
    corr = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
    if not np.isfinite(corr):
        corr = 1.0 if np.allclose(h1, h2) else -1.0
    return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


def _area_similarity(a: int, b: int) -> float:
    """``1 - |a - b| / max(a, b, 1)``, en [0, 1]."""
    return float(np.clip(1.0 - abs(a - b) / max(a, b, 1), 0.0, 1.0))


def _neighbor_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """``1 - mean(|a - b| sobre la unión de alfas) / 255``, en [0, 1]."""
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    union = (a[:, :, 3] > 0) | (b[:, :, 3] > 0)
    if not union.any():
        return 1.0  # ambos cuadros vacíos: idénticos
    diff = np.abs(a[union].astype(np.float32) - b[union].astype(np.float32))
    return float(np.clip(1.0 - float(diff.mean()) / 255.0, 0.0, 1.0))


def _pick_reference(
    fs: FrameSet, hists: list[np.ndarray], areas: list[int]
) -> tuple[np.ndarray, int]:
    """Referencia de identidad: imagen fuente si existe; si no, el cuadro cuyo
    histograma esté más cerca de la mediana de la secuencia."""
    src = fs.meta.get("source_image")
    if src:
        src_path = Path(src)
        if src_path.is_file():
            img = Image.open(src_path).convert("RGBA")
            rgba = np.asarray(img, dtype=np.uint8).copy()
            return _hist_and_area(rgba)
    median_hist = np.median(np.stack(hists, axis=0), axis=0).astype(np.float32)
    sims = [_hist_similarity(h, median_hist) for h in hists]
    ref_idx = int(np.argmax(sims))
    return hists[ref_idx], areas[ref_idx]


@register_stage
class AnomalyStage(Stage):
    """Detector de anomalías: solo puntúa y marca, jamás toca píxeles."""

    name = "anomaly"

    def process(self, fs: FrameSet, in_dir: Path, out_dir: Path, params: dict) -> FrameSet:
        in_dir, out_dir = Path(in_dir), Path(out_dir)
        threshold = float(params.get("threshold", DEFAULT_THRESHOLD))

        # Copia byte-idéntica de cada PNG (esta etapa no modifica píxeles).
        for frame in fs.frames:
            shutil.copy2(in_dir / frame.file, out_dir / frame.file)

        if not fs.frames:
            fs.meta["anomaly_summary"] = {"flagged": 0, "mean_quality": 0.0}
            return fs

        images = [load_frame_rgba(in_dir, f) for f in fs.frames]
        hists_areas = [_hist_and_area(img) for img in images]
        hists = [h for h, _ in hists_areas]
        areas = [a for _, a in hists_areas]
        ref_hist, ref_area = _pick_reference(fs, hists, areas)

        n = len(fs.frames)
        flagged = 0
        qualities: list[float] = []
        for i, frame in enumerate(fs.frames):
            identity = 0.6 * _hist_similarity(hists[i], ref_hist) + 0.4 * _area_similarity(
                areas[i], ref_area
            )
            identity = float(np.clip(identity, 0.0, 1.0))

            neighbor_sims: list[float] = []
            if i > 0:
                neighbor_sims.append(_neighbor_similarity(images[i], images[i - 1]))
            if i < n - 1:
                neighbor_sims.append(_neighbor_similarity(images[i], images[i + 1]))
            coherence = float(np.mean(neighbor_sims)) if neighbor_sims else 1.0
            coherence = float(np.clip(coherence, 0.0, 1.0))

            quality = float(np.clip(0.5 * identity + 0.5 * coherence, 0.0, 1.0))
            frame.scores["identity"] = identity
            frame.scores["coherence"] = coherence
            frame.scores["quality"] = quality
            qualities.append(quality)

            if quality < threshold:
                flagged += 1
                for flag in ("anomaly", "review"):
                    if flag not in frame.flags:
                        frame.flags.append(flag)

        fs.meta["anomaly_summary"] = {
            "flagged": flagged,
            "mean_quality": float(np.mean(qualities)),
        }
        return fs
