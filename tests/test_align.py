"""Tests de la etapa ``align``: secuencia sintética con jitter conocido."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from sprite_pipeline.models import Frame, FrameSet, frame_filename
from sprite_pipeline.stages.align import AlignStage
from sprite_pipeline.stages.base import load_frame_rgba, run_stage, save_frame_rgba

CANVAS = 128
CX = 64.0
FEET_Y = 104.0
BLOB_W = 40.0
BLOB_H = 50.0

# (jx, jy, s): jitter de posición (±6 px) y de escala (±8 %) por cuadro.
JITTER = [
    (0, 0, 1.0),
    (-6, 4, 0.92),
    (5, -6, 1.08),
    (3, 2, 0.92),
    (-4, -3, 1.08),
    (6, 6, 1.0),
]
MEDIAN_JX = 1.5  # mediana de los jx de arriba
MEDIAN_JY = 1.0  # mediana de los jy de arriba


def _blob(jx: float = 0.0, jy: float = 0.0, s: float = 1.0) -> np.ndarray:
    """Mismo blob RGBA anclado por los 'pies' (borde inferior) con jitter."""
    img = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bw, bh = BLOB_W * s, BLOB_H * s
    x0, x1 = CX + jx - bw / 2, CX + jx + bw / 2
    y1 = FEET_Y + jy
    y0 = y1 - bh
    d.ellipse([x0, y0, x1, y1], fill=(200, 80, 90, 255))
    # marca interna para que el blob no sea simétrico trivial
    d.ellipse([x0 + bw * 0.2, y0 + bh * 0.2, x0 + bw * 0.5, y0 + bh * 0.45], fill=(60, 60, 200, 255))
    return np.asarray(img, dtype=np.uint8).copy()


def _bbox_stats(rgba: np.ndarray) -> dict:
    alpha = rgba[:, :, 3]
    ys, xs = np.nonzero(alpha > 0)
    assert ys.size > 0, "cuadro sin alfa"
    return {
        "cx": (float(xs.min()) + float(xs.max())) / 2.0,
        "bottom": float(ys.max()),
        "height": float(ys.max() - ys.min() + 1),
    }


def _centroid(rgba: np.ndarray) -> tuple[float, float]:
    a = rgba[:, :, 3].astype(np.float64)
    total = a.sum()
    cx = float((a.sum(axis=0) * np.arange(a.shape[1])).sum() / total)
    cy = float((a.sum(axis=1) * np.arange(a.shape[0])).sum() / total)
    return cx, cy


def _make_frameset(base: Path, arrays: list[np.ndarray]) -> Path:
    in_dir = base / "in"
    in_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, arr in enumerate(arrays):
        fn = frame_filename(i)
        save_frame_rgba(in_dir, fn, arr)
        frames.append(Frame(index=i, file=fn, duration_ms=100))
    FrameSet(stage="background", frames=frames, meta={}).save(in_dir)
    return in_dir


def test_align_feet_removes_position_and_scale_jitter(tmp_path: Path) -> None:
    arrays = [_blob(jx, jy, s) for jx, jy, s in JITTER]
    in_dir = _make_frameset(tmp_path, arrays)
    out_dir = tmp_path / "out"
    input_bytes = (in_dir / frame_filename(1)).read_bytes()

    fs_out = run_stage(AlignStage(), in_dir, out_dir, {})

    assert fs_out.stage == "align"
    assert fs_out.history[-1]["stage"] == "align"
    assert len(fs_out.frames) == len(JITTER)

    outs = [load_frame_rgba(out_dir, f) for f in fs_out.frames]
    stats = [_bbox_stats(o) for o in outs]

    # La deriva de posición desaparece: centro-x y borde inferior casi fijos.
    assert np.std([s["cx"] for s in stats]) < 1.5
    assert np.std([s["bottom"] for s in stats]) < 1.5
    # La deriva de escala desaparece: altos de bbox casi constantes.
    heights = [s["height"] for s in stats]
    assert np.std(heights) < 1.5
    assert np.ptp(heights) <= 3.0

    # Transform anotado y coherente con el jitter inyectado.
    for frame, (jx, jy, s) in zip(fs_out.frames, JITTER):
        t = frame.transform
        assert t is not None and set(t) == {"dx", "dy", "scale"}
        assert abs(t["dx"] - (MEDIAN_JX - jx)) <= 1.5
        assert abs(t["dy"] - (MEDIAN_JY - jy)) <= 1.5
        if s == 1.0:
            assert t["scale"] == pytest.approx(1.0)  # dentro de la tolerancia: sin reescalado
        else:
            assert abs(t["scale"] - 1.0 / s) < 0.04

    # La entrada no se mutó.
    assert (in_dir / frame_filename(1)).read_bytes() == input_bytes
    assert FrameSet.load(in_dir).stage == "background"


def test_align_center_anchor_aligns_centroids(tmp_path: Path) -> None:
    arrays = [_blob(jx, jy, s) for jx, jy, s in JITTER]
    in_dir = _make_frameset(tmp_path, arrays)
    out_dir = tmp_path / "out"

    fs_out = run_stage(AlignStage(), in_dir, out_dir, {"anchor": "center"})

    outs = [load_frame_rgba(out_dir, f) for f in fs_out.frames]
    centroids = [_centroid(o) for o in outs]
    assert np.std([c[0] for c in centroids]) < 1.5
    assert np.std([c[1] for c in centroids]) < 1.5
    for frame in fs_out.frames:
        assert frame.transform is not None
        assert set(frame.transform) == {"dx", "dy", "scale"}


def test_align_scale_within_tolerance_does_not_rescale(tmp_path: Path) -> None:
    # Solo jitter de posición: ninguna escala debe aplicarse (scale == 1.0).
    jitter = [(0, 0), (-5, 3), (4, -4), (2, 5), (-6, -2)]
    arrays = [_blob(jx, jy, 1.0) for jx, jy in jitter]
    in_dir = _make_frameset(tmp_path, arrays)
    out_dir = tmp_path / "out"

    fs_out = run_stage(AlignStage(), in_dir, out_dir, {})

    for frame in fs_out.frames:
        assert frame.transform["scale"] == pytest.approx(1.0)
    stats = [_bbox_stats(load_frame_rgba(out_dir, f)) for f in fs_out.frames]
    assert np.std([s["cx"] for s in stats]) < 1.5
    assert np.std([s["bottom"] for s in stats]) < 1.5


def test_align_unknown_anchor_raises(tmp_path: Path) -> None:
    in_dir = _make_frameset(tmp_path, [_blob()])
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="anchor"):
        run_stage(AlignStage(), in_dir, out_dir, {"anchor": "nariz"})
