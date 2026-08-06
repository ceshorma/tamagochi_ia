"""Tests de calidad de ``retouch_region``: regeneración TOTAL dentro de la máscara.

Regresión del hallazgo "el blending nunca llega a peso 1 en máscaras delgadas":
con una máscara más angosta que ~4*sigma del blur, el peso interior quedaba
muy por debajo de 1 y el defecto original se mezclaba de vuelta. El fix fuerza
peso 1 en toda la región marcada (feather solo hacia afuera).
"""

from __future__ import annotations

import numpy as np

from sprite_pipeline.ops.retouch import retouch_region

SIZE = 64
CLEAN = (200, 0, 0, 255)
DEFECT = (255, 255, 255, 255)
# Trazo delgado de 4 px de alto (más angosto que 4*sigma del blur, sigma=2).
MASK_ROWS = slice(30, 34)
MASK_COLS = slice(8, 56)


def _solid(color: tuple[int, int, int, int]) -> np.ndarray:
    arr = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    arr[...] = color
    return arr


def _defective_image_and_mask() -> tuple[np.ndarray, np.ndarray]:
    img = _solid(CLEAN)
    img[MASK_ROWS, MASK_COLS] = DEFECT  # defecto fuerte cubierto por la máscara
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[MASK_ROWS, MASK_COLS] = 255
    return img, mask


def test_mascara_delgada_regenera_defecto_por_completo():
    """Máscara de 4 px sobre defecto fuerte con vecino limpio: el defecto
    desaparece por completo dentro de la máscara (error < 2/255 por canal).

    Sin el fix, el peso interior de una máscara de 4 px llega solo a ~0.67 y
    el centro conserva ~33% del defecto (error ~55/255)."""
    img, mask = _defective_image_and_mask()
    neighbor = _solid(CLEAN)

    out = retouch_region(img, mask, [neighbor])

    masked = out[mask == 255].astype(np.int16)
    expected = np.asarray(CLEAN, dtype=np.int16)
    err = np.abs(masked - expected)
    assert err.max() < 2, f"error máximo dentro de la máscara: {err.max()}/255"


def test_centro_de_la_mascara_igual_al_vecino():
    """El píxel central de la región marcada es exactamente el candidato."""
    img, mask = _defective_image_and_mask()
    neighbor = _solid(CLEAN)

    out = retouch_region(img, mask, [neighbor])

    center = out[31, SIZE // 2]
    assert np.abs(center.astype(np.int16) - np.asarray(CLEAN, np.int16)).max() < 2


def test_pixeles_lejos_de_la_mascara_intactos():
    """Lejos de la máscara (fuera del alcance del feather) nada cambia."""
    img, mask = _defective_image_and_mask()
    img[0, 0] = (10, 20, 30, 255)
    neighbor = _solid(CLEAN)

    out = retouch_region(img, mask, [neighbor])

    assert tuple(out[0, 0]) == (10, 20, 30, 255)
    assert np.array_equal(out[:10, :10], img[:10, :10])
