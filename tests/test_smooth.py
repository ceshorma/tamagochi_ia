"""Tests de la etapa ``smooth`` sobre FrameSets sintéticos (sin red, sin GPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sprite_pipeline.models import Frame, FrameSet, frame_filename
from sprite_pipeline.stages.base import (
    STAGE_REGISTRY,
    get_stage,
    load_frame_rgba,
    run_stage,
    save_frame_rgba,
)
from sprite_pipeline.stages.smooth import SmoothStage

SIZE = 128
RADIUS = 40
COLOR = (120, 200, 130)  # sobrevive *1.25 sin saturar ningún canal
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)

DURATIONS = [100, 120, 80, 100, 110, 90]  # media exacta = 100
BRIGHT_INDEX = 2
BRIGHT_FACTOR = 1.25


def _circle_mask(size: int = SIZE, radius: int = RADIUS) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    c = size // 2
    return (xx - c) ** 2 + (yy - c) ** 2 <= radius**2


def _make_frameset(
    directory: Path,
    durations: list[int],
    bright_index: int | None = None,
) -> FrameSet:
    """Secuencia de blobs circulares idénticos; uno opcionalmente +25 % brillante.

    El RGB llena todo el lienzo (el alfa recorta el blob), así la luma del
    sujeto no depende de dónde caiga exactamente el borde tras el feather.
    """
    directory.mkdir(parents=True, exist_ok=True)
    mask = _circle_mask()
    frames: list[Frame] = []
    for i, dur in enumerate(durations):
        color = np.array(COLOR, dtype=np.float64)
        if i == bright_index:
            color = np.clip(color * BRIGHT_FACTOR, 0, 255)
        rgba = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
        rgba[..., :3] = color.astype(np.uint8)
        rgba[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
        save_frame_rgba(directory, frame_filename(i), rgba)
        frames.append(Frame(index=i, file=frame_filename(i), duration_ms=int(dur)))
    fs = FrameSet(stage="align", frames=frames, meta={})
    fs.save(directory)
    return fs


def _subject_luma(rgba: np.ndarray) -> float:
    mask = rgba[..., 3] > 0
    luma = rgba[..., :3].astype(np.float64) @ _LUMA
    return float(luma[mask].mean())


def _run(tmp_path: Path, params: dict | None = None) -> tuple[Path, Path, FrameSet, FrameSet]:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    fs_in = _make_frameset(in_dir, DURATIONS, bright_index=BRIGHT_INDEX)
    result = run_stage(SmoothStage(), in_dir, out_dir, params or {})
    return in_dir, out_dir, fs_in, result


def test_stage_registrada():
    assert "smooth" in STAGE_REGISTRY
    assert isinstance(get_stage("smooth"), SmoothStage)


def test_brillo_se_estabiliza_con_ganancia_limitada(tmp_path: Path):
    in_dir, out_dir, fs_in, result = _run(tmp_path)

    before = [_subject_luma(load_frame_rgba(in_dir, f)) for f in fs_in.frames]
    after = [_subject_luma(load_frame_rgba(out_dir, f)) for f in result.frames]

    # La desviación de brillo de la secuencia baja significativamente.
    assert float(np.std(after)) < 0.6 * float(np.std(before))

    # La ganancia está limitada a 0.9: el cuadro brillante baja un 10 %,
    # no cae de golpe hasta la mediana.
    ratio = after[BRIGHT_INDEX] / before[BRIGHT_INDEX]
    assert ratio == pytest.approx(0.9, abs=0.02)
    median_before = float(np.median(before))
    assert after[BRIGHT_INDEX] > 1.05 * median_before

    # Los cuadros normales quedan prácticamente intactos (ganancia ~1).
    for i in range(len(after)):
        if i == BRIGHT_INDEX:
            continue
        assert after[i] == pytest.approx(before[i], rel=0.02)

    assert result.stage == "smooth"


def test_alfa_exterior_interior_y_feather(tmp_path: Path):
    _, out_dir, _, result = _run(tmp_path)

    c = SIZE // 2
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    dist = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    deep_interior = dist <= RADIUS - 6
    far_exterior = dist >= RADIUS + 6

    for frame in result.frames:
        alpha = load_frame_rgba(out_dir, frame)[..., 3]
        # Interior profundo intacto en 255 y exterior lejano en 0.
        assert (alpha[deep_interior] == 255).all()
        assert (alpha[far_exterior] == 0).all()
        # El borde quedó suavizado: existen valores intermedios en la banda.
        assert ((alpha > 0) & (alpha < 255)).any()


def test_duraciones_uniformadas_a_la_media(tmp_path: Path):
    in_dir, _, _, result = _run(tmp_path)

    expected = round(sum(DURATIONS) / len(DURATIONS))  # 100
    assert [f.duration_ms for f in result.frames] == [expected] * len(DURATIONS)

    # La entrada no se muta: el manifiesto de in_dir conserva sus duraciones.
    reloaded = FrameSet.load(in_dir)
    assert [f.duration_ms for f in reloaded.frames] == DURATIONS
    assert reloaded.stage == "align"


def test_keep_timing_conserva_duraciones(tmp_path: Path):
    _, _, _, result = _run(tmp_path, {"keep_timing": True})
    assert [f.duration_ms for f in result.frames] == DURATIONS
