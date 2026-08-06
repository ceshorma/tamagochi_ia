"""Tests de la etapa ``background`` sobre secuencias sintéticas.

La secuencia se construye aquí mismo: la criatura de ``sample.draw_creature``
compuesta sobre un fondo gris con gradiente horizontal y ruido leve, con un
movimiento suave (bob/squash) entre cuadros. Se guarda la máscara verdadera
(alfa de la criatura antes de componer) para validar la segmentación.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from sprite_pipeline.models import Frame, FrameSet, frame_filename
from sprite_pipeline.sample import draw_creature
from sprite_pipeline.stages.base import get_stage, load_frame_rgba, run_stage, save_frame_rgba

SIZE = 128        # lado del cuadro
CREATURE = 88     # lado de la criatura compuesta
N_FRAMES = 6


def _make_sequence(
    directory: Path, n: int = N_FRAMES, with_meta: bool = True, seed: int = 7
) -> tuple[FrameSet, list[np.ndarray]]:
    """Escribe un FrameSet sintético y devuelve (frameset, máscaras verdaderas)."""
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    grad = np.linspace(190.0, 210.0, SIZE, dtype=np.float32)  # gradiente horizontal
    off = ((SIZE - CREATURE) // 2, (SIZE - CREATURE) // 2)

    frames: list[Frame] = []
    gt_masks: list[np.ndarray] = []
    for i in range(n):
        w = 2 * math.pi * i / n
        sprite = draw_creature(size=CREATURE, bob=0.4 * math.sin(w), squash=0.1 * math.sin(w))

        bg = np.zeros((SIZE, SIZE, 4), np.uint8)
        base = np.broadcast_to(grad[None, :, None], (SIZE, SIZE, 3))
        noise = rng.normal(0.0, 2.0, size=(SIZE, SIZE, 3))
        bg[..., :3] = np.clip(base + noise, 0, 255).astype(np.uint8)
        bg[..., 3] = 255

        canvas = Image.fromarray(bg, "RGBA")
        canvas.alpha_composite(sprite, dest=off)
        arr = np.asarray(canvas, dtype=np.uint8).copy()

        fname = frame_filename(i)
        save_frame_rgba(directory, fname, arr)
        frames.append(Frame(index=i, file=fname))

        gt = np.zeros((SIZE, SIZE), dtype=bool)
        gt[off[1] : off[1] + CREATURE, off[0] : off[0] + CREATURE] = (
            np.asarray(sprite)[..., 3] > 128
        )
        gt_masks.append(gt)

    meta: dict = {"size": [SIZE, SIZE]}
    if with_meta:
        meta["bg_color_estimate"] = [200, 200, 200]
    fs = FrameSet(stage="preprocess", frames=frames, meta=meta)
    fs.save(directory)
    return fs, gt_masks


def _run(tmp_path: Path, params: dict | None = None, with_meta: bool = True):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _, gt_masks = _make_sequence(in_dir, with_meta=with_meta)
    fs_out = run_stage(get_stage("background"), in_dir, out_dir, params or {})
    return fs_out, out_dir, gt_masks


def _inner_pixel(gt: np.ndarray) -> tuple[int, int]:
    """Un píxel bien adentro de la silueta verdadera (erosión + cercano al centroide)."""
    inner = cv2.erode(gt.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ys, xs = np.nonzero(inner)
    cy, cx = ys.mean(), xs.mean()
    k = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
    return int(ys[k]), int(xs[k])


def test_corners_transparent_center_opaque(tmp_path):
    fs_out, out_dir, gt_masks = _run(tmp_path)
    assert fs_out.stage == "background"
    assert len(fs_out.frames) == N_FRAMES
    for frame, gt in zip(fs_out.frames, gt_masks):
        alpha = load_frame_rgba(out_dir, frame)[..., 3]
        for y, x in [(0, 0), (0, SIZE - 1), (SIZE - 1, 0), (SIZE - 1, SIZE - 1)]:
            assert alpha[y, x] == 0
        y, x = _inner_pixel(gt)
        assert alpha[y, x] == 255


def test_area_close_to_expected_and_no_big_holes(tmp_path):
    fs_out, out_dir, gt_masks = _run(tmp_path)
    for frame, gt in zip(fs_out.frames, gt_masks):
        alpha = load_frame_rgba(out_dir, frame)[..., 3]
        mask = alpha > 128
        expected = int(gt.sum())
        assert abs(int(mask.sum()) - expected) <= 0.15 * expected

        # Sin agujeros grandes: componentes transparentes que no tocan el borde.
        num, _, stats, _ = cv2.connectedComponentsWithStats(
            (~mask).astype(np.uint8), connectivity=4
        )
        holes_area = 0
        for lbl in range(1, num):
            x, y, w, h, area = stats[lbl]
            touches_border = x == 0 or y == 0 or x + w == SIZE or y + h == SIZE
            if not touches_border:
                holes_area += int(area)
        assert holes_area <= 0.02 * expected


def test_preserves_subject_rgb(tmp_path):
    fs_out, out_dir, gt_masks = _run(tmp_path)
    in_dir = tmp_path / "in"
    fs_in = FrameSet.load(in_dir)
    for f_in, f_out, gt in zip(fs_in.frames, fs_out.frames, gt_masks):
        rgb_in = load_frame_rgba(in_dir, f_in)[..., :3]
        rgb_out = load_frame_rgba(out_dir, f_out)[..., :3]
        inner = cv2.erode(gt.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        assert np.array_equal(rgb_in[inner], rgb_out[inner])


def test_temporal_stability_iou(tmp_path):
    fs_out, out_dir, _ = _run(tmp_path)
    masks = [load_frame_rgba(out_dir, f)[..., 3] > 128 for f in fs_out.frames]
    for a, b in zip(masks, masks[1:]):
        iou = np.logical_and(a, b).sum() / np.logical_or(a, b).sum()
        assert iou > 0.9


def test_works_without_meta_bg_estimate(tmp_path):
    """Sin ``bg_color_estimate`` en meta: se estima con la mediana de los bordes."""
    fs_out, out_dir, gt_masks = _run(tmp_path, with_meta=False)
    for frame, gt in zip(fs_out.frames, gt_masks):
        alpha = load_frame_rgba(out_dir, frame)[..., 3]
        assert alpha[0, 0] == 0 and alpha[SIZE - 1, SIZE - 1] == 0
        expected = int(gt.sum())
        assert abs(int((alpha > 128).sum()) - expected) <= 0.15 * expected


def test_edge_band_decontaminated_no_bg_halo(tmp_path):
    """La banda semitransparente del borde no conserva el halo del fondo.

    Regresión: antes el feather regalaba alfa a un anillo FUERA de la silueta
    cuyo RGB era el color del fondo del video, horneando un fleco del color
    del fondo en el sprite. Ahora el feather es solo hacia adentro y el RGB de
    la banda 0<alfa<255 se des-mezcla del color de fondo estimado: todo píxel
    del borde queda más cerca del color del sujeto que del fondo original."""
    subject = np.array([30, 120, 40], dtype=np.float32)
    bg = np.array([220, 220, 230], dtype=np.float32)
    side = 60

    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir(parents=True)
    frames: list[Frame] = []
    tops: list[int] = []
    for i in range(3):
        arr = np.zeros((SIZE, SIZE, 4), np.uint8)
        arr[..., :3] = bg.astype(np.uint8)
        arr[..., 3] = 255
        y0 = 30 + i  # leve movimiento vertical entre cuadros
        arr[y0:y0 + side, 30:30 + side, :3] = subject.astype(np.uint8)
        fname = frame_filename(i)
        save_frame_rgba(in_dir, fname, arr)
        frames.append(Frame(index=i, file=fname))
        tops.append(y0)
    FrameSet(stage="preprocess", frames=frames, meta={"size": [SIZE, SIZE]}).save(in_dir)

    fs_out = run_stage(get_stage("background"), in_dir, out_dir)

    for frame, y0 in zip(fs_out.frames, tops):
        rgba = load_frame_rgba(out_dir, frame)
        alpha = rgba[..., 3]

        # Fuera de la silueta verdadera no se regala alfa (sin anillo exterior).
        outside = np.ones((SIZE, SIZE), dtype=bool)
        outside[y0:y0 + side, 30:30 + side] = False
        assert (alpha[outside] == 0).all()

        # La banda 0<alfa<255 existe y su RGB tiende al sujeto, no al fondo.
        semi = (alpha > 0) & (alpha < 255)
        assert semi.any()
        rgb = rgba[..., :3].astype(np.float32)
        dist_subject = np.linalg.norm(rgb[semi] - subject, axis=1)
        dist_bg = np.linalg.norm(rgb[semi] - bg, axis=1)
        assert (dist_subject < dist_bg).all(), (
            f"halo: {int((dist_subject >= dist_bg).sum())} píxeles del borde "
            "siguen más cerca del color del fondo original"
        )


def test_unavailable_method_raises_value_error(tmp_path):
    in_dir = tmp_path / "in"
    _make_sequence(in_dir, n=2)
    with pytest.raises(ValueError, match="sam2"):
        run_stage(get_stage("background"), in_dir, tmp_path / "out", {"method": "sam2"})
