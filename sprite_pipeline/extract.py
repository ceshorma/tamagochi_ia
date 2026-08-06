"""Extracción de fotogramas: video crudo del proveedor -> FrameSet ``raw``.

Contrato (docs/API_SPEC.md):

- ``extract_frames(video_path, out_dir, fps=10, dedupe_threshold=0.995, meta=None) -> FrameSet``
- Lectura con ``cv2.VideoCapture``; fallback ``imageio.get_reader`` si cv2 no
  puede abrir/leer el archivo.
- Muestreo a ``fps`` efectivo: con ``fps_video`` del contenedor se toman los
  índices ``round(k * fps_video / fps)`` únicos hasta el final del video.
- fps efectivo y timing: con metadata de fps válida en el contenedor, el fps
  efectivo es ``min(fps, fps_video)`` — pedir más fps que los que tiene el
  video no puede inventar cuadros, así que el timing se deriva del muestreo
  real: ``duration_ms = round(1000 / fps_efectivo)``. Sin metadata válida de
  fps (0, negativa o no finita) se toman TODOS los cuadros del video y el
  timing usa el ``fps`` pedido como suposición (``duration_ms =
  round(1000/fps)``), porque no hay forma de saber el fps real.
- Dedupe de consecutivos: similitud ``1 - mean(|a - b|) / 255`` sobre RGB;
  si supera ``dedupe_threshold`` se descarta el segundo cuadro.
- Salida: PNGs RGBA (``frame_filename(i)``) + ``frameset.json`` con
  ``stage="raw"``, ``duration_ms=round(1000/fps_efectivo)`` y ``meta``
  fusionado con ``{"fps": fps_efectivo, "video": str(video_path)}``.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from sprite_pipeline.models import Frame, FrameSet, frame_filename
from sprite_pipeline.stages.base import save_frame_rgba

__all__ = ["extract_frames"]


# ------------------------------------------------------------------- lectura

def _to_rgba(arr: np.ndarray) -> np.ndarray:
    """Convierte un cuadro decodificado (gris/RGB/RGBA) a RGBA uint8."""
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] == 3:
        alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
        arr = np.concatenate([arr, alpha], axis=-1)
    elif arr.shape[2] != 4:
        raise ValueError(f"Cuadro con forma inesperada: {arr.shape}")
    return np.ascontiguousarray(arr)


def _read_with_cv2(video_path: Path) -> tuple[list[np.ndarray], float] | None:
    """Lee todos los cuadros como RGBA con cv2. ``None`` si cv2 no puede."""
    try:
        import cv2
    except ImportError:  # pragma: no cover - cv2 está en las dependencias
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    frames: list[np.ndarray] = []
    try:
        fps_video = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame.ndim == 3 and frame.shape[2] == 4:
                rgba = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
            elif frame.ndim == 3 and frame.shape[2] == 3:
                rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            else:
                rgba = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGBA)
            frames.append(_to_rgba(rgba))
    finally:
        cap.release()
    if not frames:
        return None
    return frames, fps_video


def _read_with_imageio(video_path: Path) -> tuple[list[np.ndarray], float]:
    """Fallback: lee todos los cuadros como RGBA con imageio (plugin ffmpeg)."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(video_path))
    try:
        try:
            fps_video = float(reader.get_meta_data().get("fps") or 0.0)
        except Exception:
            fps_video = 0.0
        frames = [_to_rgba(f) for f in reader]
    finally:
        reader.close()
    return frames, fps_video


# ------------------------------------------------------------------ muestreo

def _sample_indices(total: int, fps_video: float, fps: int) -> list[int]:
    """Índices ``round(k * fps_video / fps)`` únicos y menores que ``total``."""
    if not fps_video or fps_video <= 0 or not math.isfinite(fps_video):
        fps_video = float(fps)  # sin metadata de fps: tomar todos los cuadros
    step = fps_video / fps
    indices: list[int] = []
    seen: set[int] = set()
    k = 0
    while True:
        idx = round(k * step)
        if idx >= total:
            break
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
        k += 1
    return indices


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similitud simple entre cuadros: ``1 - mean(|a-b|)/255`` sobre RGB."""
    if a.shape != b.shape:
        return 0.0
    diff = np.abs(a[..., :3].astype(np.float32) - b[..., :3].astype(np.float32))
    return 1.0 - float(diff.mean()) / 255.0


# ------------------------------------------------------------------ API

def extract_frames(
    video_path: Path,
    out_dir: Path,
    fps: int = 10,
    dedupe_threshold: float = 0.995,
    meta: dict | None = None,
) -> FrameSet:
    """Extrae fotogramas de ``video_path`` a un FrameSet ``raw`` en ``out_dir``.

    El timing de los cuadros se deriva del fps EFECTIVO de muestreo:
    ``min(fps, fps_video)`` cuando el contenedor reporta un fps válido (no se
    pueden muestrear más cuadros por segundo de los que el video tiene), con
    ``duration_ms = round(1000 / fps_efectivo)``. Si el contenedor no reporta
    fps válido, se toman todos los cuadros y el timing usa el ``fps`` pedido
    como suposición documentada.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    if fps <= 0:
        raise ValueError(f"fps debe ser positivo; llegó {fps}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video no encontrado: {video_path}")

    loaded = _read_with_cv2(video_path)
    if loaded is None:
        loaded = _read_with_imageio(video_path)
    all_frames, fps_video = loaded
    if not all_frames:
        raise ValueError(f"El video no contiene cuadros legibles: {video_path}")

    # fps efectivo: con metadata válida no se puede muestrear por encima del
    # fps del video (pedir más solo toma todos los cuadros una vez); sin
    # metadata válida se toman todos los cuadros y el fps pedido es la única
    # suposición disponible para el timing.
    has_fps_meta = bool(fps_video) and fps_video > 0 and math.isfinite(fps_video)
    effective_fps = min(float(fps), float(fps_video)) if has_fps_meta else float(fps)

    sampled = [all_frames[i] for i in _sample_indices(len(all_frames), fps_video, fps)]

    kept: list[np.ndarray] = []
    for rgba in sampled:
        if kept and _similarity(kept[-1], rgba) > dedupe_threshold:
            continue  # casi idéntico al anterior conservado: se descarta
        kept.append(rgba)

    out_dir.mkdir(parents=True, exist_ok=True)
    duration_ms = round(1000 / effective_fps)
    frames: list[Frame] = []
    for i, rgba in enumerate(kept):
        filename = frame_filename(i)
        save_frame_rgba(out_dir, filename, rgba)
        frames.append(Frame(index=i, file=filename, duration_ms=duration_ms))

    merged_meta = dict(meta or {})
    meta_fps = int(effective_fps) if float(effective_fps).is_integer() else float(effective_fps)
    merged_meta.update({"fps": meta_fps, "video": str(video_path)})

    fs = FrameSet(
        stage="raw",
        frames=frames,
        meta=merged_meta,
        history=[
            {
                "stage": "raw",
                "params": {"fps": fps, "dedupe_threshold": dedupe_threshold},
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        ],
    )
    fs.save(out_dir)
    return fs
