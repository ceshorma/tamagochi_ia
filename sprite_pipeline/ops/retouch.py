"""Retoque de una región de un cuadro RGBA.

``retouch_region(img, mask, neighbors)`` regenera la zona blanca (255) de la
máscara:

- Con vecinos: el candidato es la media (float) de los cuadros vecinos y se
  pega sobre la región con blending de borde. El peso por píxel es la máscara
  difuminada con ``cv2.GaussianBlur``, forzada a 1 dentro de la región marcada
  (``np.maximum`` con la máscara binaria): la zona blanca se regenera POR
  COMPLETO y el feather solo suaviza hacia afuera.
- Sin vecinos: ``cv2.inpaint`` TELEA sobre el RGB (convertido a BGR para cv2)
  y el alfa se reconstruye a partir del blur de la máscara invertida (la
  región regenerada se desvanece suavemente, sin tocar el alfa exterior).
"""

from __future__ import annotations

import cv2
import numpy as np

_BLUR_SIGMA = 2.0
_INPAINT_RADIUS = 3


def _validate_rgba(img: np.ndarray, name: str) -> None:
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 4:
        raise ValueError(f"{name} debe ser un array RGBA uint8 (h,w,4); llegó "
                         f"{getattr(img, 'dtype', type(img))} {getattr(img, 'shape', None)}")


def _normalize_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """Máscara binaria uint8 (0/255) 2D del tamaño del cuadro."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    m = m.astype(np.uint8)
    h, w = shape_hw
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.where(m > 127, 255, 0).astype(np.uint8)


def retouch_region(img: np.ndarray, mask: np.ndarray, neighbors: list[np.ndarray]) -> np.ndarray:
    """Regenera la región marcada (255) de ``mask`` en ``img`` (RGBA uint8)."""
    _validate_rgba(img, "img")
    h, w = img.shape[:2]
    binary = _normalize_mask(mask, (h, w))
    if not binary.any():
        return img.copy()

    if neighbors:
        # Candidato: media de los vecinos en float (reescalados si difieren).
        acc = np.zeros((h, w, 4), dtype=np.float32)
        for n in neighbors:
            _validate_rgba(n, "neighbor")
            if n.shape != img.shape:
                n = cv2.resize(n, (w, h), interpolation=cv2.INTER_AREA)
            acc += n.astype(np.float32)
        candidate = acc / float(len(neighbors))

        # Peso de blending: máscara difuminada, forzada a 1 en TODA la región
        # marcada (en máscaras delgadas el blur bajaría el peso interior muy
        # por debajo de 1 y el defecto original se mezclaría de vuelta). El
        # feather queda así solo hacia afuera de la región.
        binary_f = binary.astype(np.float32) / 255.0
        weight = cv2.GaussianBlur(binary_f, (0, 0), _BLUR_SIGMA)
        weight = np.maximum(weight, binary_f)
        weight = np.clip(weight, 0.0, 1.0)[..., None]
        out = img.astype(np.float32) * (1.0 - weight) + candidate * weight
        return np.clip(out, 0, 255).astype(np.uint8)

    # Sin vecinos: inpainting clásico del color + alfa desde la máscara invertida.
    bgr = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(bgr, binary, _INPAINT_RADIUS, cv2.INPAINT_TELEA)
    rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

    inv_blur = cv2.GaussianBlur(255 - binary, (0, 0), _BLUR_SIGMA)
    alpha = np.minimum(img[..., 3], inv_blur).astype(np.uint8)
    return np.dstack([rgb, alpha]).astype(np.uint8)
