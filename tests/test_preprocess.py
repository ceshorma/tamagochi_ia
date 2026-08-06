"""Tests de la etapa ``preprocess`` con FrameSets sintéticos (círculo sobre gris)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from sprite_pipeline.models import Frame, FrameSet, frame_filename
from sprite_pipeline.stages.base import get_stage, load_frame_rgba, run_stage

BG = (200, 200, 200)
CIRCLE = (220, 60, 60)
RADIUS = 18


def make_frameset(
    directory: Path,
    centers: list[tuple[int, int]],
    size: tuple[int, int] = (120, 140),  # (ancho, alto)
    bg: tuple[int, int, int] = BG,
) -> FrameSet:
    """FrameSet sintético: círculo de color sobre fondo gris con offsets variables."""
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, (cx, cy) in enumerate(centers):
        img = Image.new("RGBA", size, (*bg, 255))
        d = ImageDraw.Draw(img)
        d.ellipse([cx - RADIUS, cy - RADIUS, cx + RADIUS, cy + RADIUS], fill=(*CIRCLE, 255))
        fname = frame_filename(i)
        img.save(directory / fname)
        frames.append(Frame(index=i, file=fname, duration_ms=100))
    fs = FrameSet(stage="raw", frames=frames, meta={"fps": 10})
    fs.save(directory)
    return fs


def make_uniform_frameset(directory: Path, n: int = 2, size: tuple[int, int] = (100, 120)) -> FrameSet:
    """FrameSet sin sujeto: cuadros uniformes del color de fondo."""
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(n):
        Image.new("RGBA", size, (*BG, 255)).save(directory / frame_filename(i))
        frames.append(Frame(index=i, file=frame_filename(i)))
    fs = FrameSet(stage="raw", frames=frames)
    fs.save(directory)
    return fs


def subject_mask(rgba: np.ndarray, bg: tuple[int, int, int] = BG, thr: float = 60.0) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.float64)
    dist = np.sqrt(((rgb - np.asarray(bg, dtype=np.float64)) ** 2).sum(axis=2))
    return dist > thr


CENTERS = [(40, 50), (60, 55), (50, 70)]


@pytest.fixture()
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "in", tmp_path / "out"


def test_canvas_size_meta_and_manifest(dirs):
    in_dir, out_dir = dirs
    make_frameset(in_dir, CENTERS)
    stage = get_stage("preprocess")
    result = run_stage(stage, in_dir, out_dir, {"work_size": 96})

    assert result.stage == "preprocess"
    assert len(result.frames) == 3
    assert result.meta["size"] == [96, 96]
    assert result.meta["fps"] == 10  # meta previa preservada
    bg_est = result.meta["bg_color_estimate"]
    assert len(bg_est) == 3
    assert all(abs(c - 200) <= 2 for c in bg_est)

    for f in result.frames:
        out = load_frame_rgba(out_dir, f)
        assert out.shape == (96, 96, 4)
        assert out.dtype == np.uint8

    # manifiesto persistido por run_stage
    reloaded = FrameSet.load(out_dir)
    assert reloaded.stage == "preprocess"
    assert reloaded.meta["size"] == [96, 96]
    assert reloaded.history[-1]["stage"] == "preprocess"


def test_input_dir_not_mutated(dirs):
    in_dir, out_dir = dirs
    make_frameset(in_dir, CENTERS)
    before = {p.name: p.read_bytes() for p in sorted(in_dir.iterdir())}
    run_stage(get_stage("preprocess"), in_dir, out_dir, {"work_size": 96})
    after = {p.name: p.read_bytes() for p in sorted(in_dir.iterdir())}
    assert before == after


def test_subject_contained_and_union_centered(dirs):
    in_dir, out_dir = dirs
    make_frameset(in_dir, CENTERS)
    work, margin = 96, 0.08
    result = run_stage(get_stage("preprocess"), in_dir, out_dir, {"work_size": work, "margin": margin})

    inner = round(work * (1 - 2 * margin))  # 81
    lo = (work - inner) // 2  # inicio de la zona interior

    union = np.zeros((work, work), dtype=bool)
    for f in result.frames:
        out = load_frame_rgba(out_dir, f)
        m = subject_mask(out)
        assert m.any(), "el sujeto debe estar presente en el cuadro de salida"
        # canvas fuera del recorte relleno con el fondo estimado, opaco
        assert np.all(out[..., 3] == 255)
        union |= m

    # sujeto contenido dentro de la zona interior (con tolerancia por resampleo)
    ys, xs = np.nonzero(union)
    tol = 2
    assert ys.min() >= lo - tol and xs.min() >= lo - tol
    assert ys.max() <= lo + inner - 1 + tol and xs.max() <= lo + inner - 1 + tol

    # bbox de la unión aproximadamente centrado en el canvas
    cy = (ys.min() + ys.max()) / 2
    cx = (xs.min() + xs.max()) / 2
    center = (work - 1) / 2
    assert abs(cy - center) <= 3
    assert abs(cx - center) <= 3


def test_shared_union_bbox_keeps_offsets_and_scale(dirs):
    """Mismo bbox-unión para todos: los offsets relativos sobreviven (sin
    re-centrado por cuadro) y la escala es idéntica en toda la secuencia."""
    in_dir, out_dir = dirs
    make_frameset(in_dir, CENTERS)
    result = run_stage(get_stage("preprocess"), in_dir, out_dir, {"work_size": 96})

    centroids, areas = [], []
    for f in result.frames:
        m = subject_mask(load_frame_rgba(out_dir, f))
        pts = np.argwhere(m)
        centroids.append(pts.mean(axis=0))
        areas.append(float(m.sum()))

    # misma escala -> mismo tamaño aparente del círculo
    assert max(areas) / min(areas) < 1.15

    # el offset entre cuadros NO desaparece: si se centrara cada cuadro por su
    # propio bbox, todos los centroides coincidirían.
    d02 = float(np.linalg.norm(centroids[0] - centroids[2]))
    assert d02 > 5.0


def test_empty_mask_uses_full_frame(dirs):
    in_dir, out_dir = dirs
    make_uniform_frameset(in_dir)
    result = run_stage(get_stage("preprocess"), in_dir, out_dir, {"work_size": 96})

    assert result.meta["bg_color_estimate"] == [200, 200, 200]
    for f in result.frames:
        out = load_frame_rgba(out_dir, f)
        assert out.shape == (96, 96, 4)
        assert np.all(out[..., 3] == 255)
        diff = np.abs(out[..., :3].astype(int) - 200)
        assert diff.max() <= 2  # todo el canvas queda del color de fondo


def test_partial_alpha_counts_as_subject(dirs):
    in_dir, out_dir = dirs
    in_dir.mkdir(parents=True)
    arr = np.empty((120, 120, 4), dtype=np.uint8)
    arr[..., :3] = 200
    arr[..., 3] = 255
    arr[40:70, 30:80, 3] = 120  # mismo color que el fondo, pero alfa parcial
    Image.fromarray(arr, "RGBA").save(in_dir / frame_filename(0))
    fs = FrameSet(stage="raw", frames=[Frame(index=0, file=frame_filename(0))])
    fs.save(in_dir)

    result = run_stage(get_stage("preprocess"), in_dir, out_dir, {"work_size": 96})
    out = load_frame_rgba(out_dir, result.frames[0])

    # la región de alfa parcial fue detectada como sujeto y llevada al centro
    center_alpha = out[48, 48, 3]
    assert center_alpha < 200
    # las esquinas (fuera del recorte) quedan opacas con el fondo estimado
    for y, x in [(0, 0), (0, 95), (95, 0), (95, 95)]:
        assert out[y, x, 3] == 255
        assert np.all(np.abs(out[y, x, :3].astype(int) - 200) <= 2)


def test_default_work_size_from_settings(dirs, data_dir, monkeypatch):
    from sprite_pipeline.config import get_settings

    # work_size se evalúa al importar config, así que se ajusta el singleton.
    monkeypatch.setattr(get_settings(), "work_size", 120)
    assert get_settings().work_size == 120

    in_dir, out_dir = dirs
    make_frameset(in_dir, CENTERS)
    result = run_stage(get_stage("preprocess"), in_dir, out_dir, {})

    assert result.meta["size"] == [120, 120]
    out = load_frame_rgba(out_dir, result.frames[0])
    assert out.shape == (120, 120, 4)


def test_deterministic(dirs, tmp_path):
    in_dir, out_dir = dirs
    make_frameset(in_dir, CENTERS)
    out_dir2 = tmp_path / "out2"
    run_stage(get_stage("preprocess"), in_dir, out_dir, {"work_size": 96})
    run_stage(get_stage("preprocess"), in_dir, out_dir2, {"work_size": 96})
    for i in range(3):
        a = (out_dir / frame_filename(i)).read_bytes()
        b = (out_dir2 / frame_filename(i)).read_bytes()
        assert a == b
