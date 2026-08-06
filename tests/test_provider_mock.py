"""Tests del proveedor mock: determinista, sin red, imágenes pequeñas."""

from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from sprite_pipeline.providers.base import (
    PROVIDER_REGISTRY,
    GenerationRequest,
    build_prompt,
    get_provider,
)
from sprite_pipeline.providers.mock import MockProvider


def _make_req(image_path: Path, **overrides) -> GenerationRequest:
    params = dict(action="idle", seed=7, duration_s=1.0, fps=8, size=(96, 96))
    params.update(overrides)
    return GenerationRequest(image_path=image_path, **params)


def _read_frames_imageio(path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(str(path))
    try:
        return [np.asarray(f, dtype=np.uint8) for f in reader]
    finally:
        reader.close()


def _count_frames_cv2(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"cv2 no pudo abrir {path}"
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    return count


def _neighbor_diff_scores(frames: list[np.ndarray]) -> np.ndarray:
    """Diferencia media absoluta de cada cuadro contra sus vecinos."""
    scores = []
    for i, f in enumerate(frames):
        neighbors = []
        if i > 0:
            neighbors.append(frames[i - 1])
        if i < len(frames) - 1:
            neighbors.append(frames[i + 1])
        diffs = [np.mean(np.abs(f.astype(np.int16) - n.astype(np.int16))) for n in neighbors]
        scores.append(float(np.mean(diffs)))
    return np.asarray(scores)


def test_registered_and_result_contract(sample_image: Path, tmp_path: Path):
    assert PROVIDER_REGISTRY["mock"] is MockProvider
    provider = get_provider("mock")
    assert isinstance(provider, MockProvider)

    req = _make_req(sample_image)
    result = provider.generate(req, tmp_path / "work")

    assert result.provider == "mock"
    assert result.prompt == build_prompt(req)
    assert result.video_path == tmp_path / "work" / "video.mp4"
    assert result.video_path.exists()
    assert result.video_path.stat().st_size > 0


def test_video_readable_and_frame_count(sample_image: Path, tmp_path: Path):
    req = _make_req(sample_image, duration_s=1.5, fps=8)  # round(1.5 * 8) = 12
    result = MockProvider().generate(req, tmp_path / "work")

    frames = _read_frames_imageio(result.video_path)
    assert len(frames) == 12
    for f in frames:
        assert f.shape == (96, 96, 3)
        assert f.dtype == np.uint8

    assert _count_frames_cv2(result.video_path) == 12


def test_determinism_same_seed(sample_image: Path, tmp_path: Path):
    req_a = _make_req(sample_image, seed=123)
    req_b = _make_req(sample_image, seed=123)
    res_a = MockProvider().generate(req_a, tmp_path / "a")
    res_b = MockProvider().generate(req_b, tmp_path / "b")

    frames_a = _read_frames_imageio(res_a.video_path)
    frames_b = _read_frames_imageio(res_b.video_path)
    assert len(frames_a) == len(frames_b) == 8
    for fa, fb in zip(frames_a, frames_b):
        assert np.array_equal(fa, fb)

    # Otra semilla debe producir cuadros distintos (ruido/deriva diferentes).
    res_c = MockProvider().generate(_make_req(sample_image, seed=124), tmp_path / "c")
    frames_c = _read_frames_imageio(res_c.video_path)
    assert any(not np.array_equal(fa, fc) for fa, fc in zip(frames_a, frames_c))


def test_generic_action_uses_given_image(sample_image: Path, tmp_path: Path):
    # Acción sin pose definida en sample.py -> camino genérico bob/squash via PIL.
    req = _make_req(sample_image, action="spin around happily", duration_s=1.0, fps=6)
    result = MockProvider().generate(req, tmp_path / "work")

    frames = _read_frames_imageio(result.video_path)
    assert len(frames) == 6
    # La animación se mueve: no todos los cuadros son idénticos.
    assert any(not np.array_equal(frames[0], f) for f in frames[1:])
    # Hay contenido compuesto sobre el fondo (no es solo el gradiente gris).
    center = frames[0][24:72, 24:72]
    assert center.std() > 5.0


def test_inject_anomaly_marks_one_intermediate_frame(sample_image: Path, tmp_path: Path):
    req = _make_req(
        sample_image,
        duration_s=1.5,
        fps=8,
        seed=42,
        prompt_extra="please inject_anomaly for testing",
    )
    result = MockProvider().generate(req, tmp_path / "work")

    frames = _read_frames_imageio(result.video_path)
    assert len(frames) == 12

    scores = _neighbor_diff_scores(frames)
    worst = int(np.argmax(scores))
    median = float(np.median(scores))

    # El cuadro corrupto es intermedio y su diferencia media contra vecinos
    # supera claramente la mediana de la secuencia.
    assert 0 < worst < len(frames) - 1
    assert scores[worst] > 3.0 * median

    # Sin la señal en prompt_extra no aparece un outlier tan extremo.
    clean = MockProvider().generate(
        _make_req(sample_image, duration_s=1.5, fps=8, seed=42), tmp_path / "clean"
    )
    clean_scores = _neighbor_diff_scores(_read_frames_imageio(clean.video_path))
    assert scores[worst] > 3.0 * float(np.max(clean_scores))
