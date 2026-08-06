"""Interpolación de cuadros intermedios por flujo óptico.

``interpolate_frames(a, b, t)`` sintetiza el cuadro intermedio entre dos
cuadros RGBA usando flujo óptico denso (Farneback) calculado sobre la
luminancia del RGB compuesto sobre negro. Se hace warp de ``a`` hacia
adelante ``t * flow`` y de ``b`` hacia atrás ``(1 - t) * flow`` con
``cv2.remap`` y se mezclan ambos warps ``(1-t)*warpA + t*warpB`` (los cuatro
canales RGBA en float, incluido el alfa).

Si el flujo produce valores no finitos o cualquier paso lanza una excepción,
se degrada a un crossfade simple entre ambos cuadros.
"""

from __future__ import annotations

import cv2
import numpy as np

# Parámetros de Farneback: pirámide de 4 niveles y ventana de 21 px para
# capturar desplazamientos grandes en sprites pequeños (96-256 px).
_FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=4,
    winsize=21,
    iterations=5,
    poly_n=7,
    poly_sigma=1.5,
    flags=0,
)


def _validate_rgba(img: np.ndarray, name: str) -> None:
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 4:
        raise ValueError(f"{name} debe ser un array RGBA uint8 (h,w,4); llegó "
                         f"{getattr(img, 'dtype', type(img))} {getattr(img, 'shape', None)}")


def _composite_gray(rgba: np.ndarray) -> np.ndarray:
    """Luminancia del RGB compuesto sobre fondo negro (premultiplicado por alfa)."""
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = (rgba[..., :3].astype(np.float32) * alpha).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _crossfade(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    out = (1.0 - t) * a.astype(np.float32) + t * b.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def interpolate_frames(a: np.ndarray, b: np.ndarray, t: float = 0.5) -> np.ndarray:
    """Cuadro intermedio RGBA entre ``a`` (t=0) y ``b`` (t=1).

    Fallback a crossfade si el flujo óptico falla o produce NaN/inf.
    """
    _validate_rgba(a, "a")
    _validate_rgba(b, "b")
    t = float(min(max(t, 0.0), 1.0))
    if b.shape != a.shape:
        h, w = a.shape[:2]
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)

    try:
        gray_a = _composite_gray(a)
        gray_b = _composite_gray(b)
        flow = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, **_FARNEBACK_PARAMS)
        if flow is None or not np.isfinite(flow).all():
            return _crossfade(a, b, t)

        h, w = gray_a.shape
        grid_x, grid_y = np.meshgrid(
            np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
        )
        fx, fy = flow[..., 0], flow[..., 1]
        a_f = a.astype(np.float32)
        b_f = b.astype(np.float32)
        # a avanza t*flow (muestreo inverso: grid - t*flow); b retrocede (1-t)*flow.
        warp_a = cv2.remap(
            a_f, grid_x - t * fx, grid_y - t * fy,
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        warp_b = cv2.remap(
            b_f, grid_x + (1.0 - t) * fx, grid_y + (1.0 - t) * fy,
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        out = (1.0 - t) * warp_a + t * warp_b
        if not np.isfinite(out).all():
            return _crossfade(a, b, t)
        return np.clip(out, 0, 255).astype(np.uint8)
    except (cv2.error, ValueError, RuntimeError):
        return _crossfade(a, b, t)
