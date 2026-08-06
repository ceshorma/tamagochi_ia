"""Tests de la etapa ``anomaly``: puntúa y marca sin tocar píxeles."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from sprite_pipeline.models import Frame, FrameSet, frame_filename
from sprite_pipeline.stages.anomaly import AnomalyStage
from sprite_pipeline.stages.base import get_stage, run_stage

SIZE = 128
N_FRAMES = 6
CORRUPT_INDEX = 3


def _blob_rgba(size: int = SIZE, offset: float = 0.0, pulse: float = 0.0) -> np.ndarray:
    """Blob tipo criatura, con variacion leve de posicion/tamano entre cuadros."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = size / 2 + offset
    cy = size / 2
    r = size * 0.30 * (1 + 0.03 * pulse)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(120, 200, 130, 255))
    d.ellipse([cx - r * 0.5, cy, cx + r * 0.5, cy + r * 0.7], fill=(215, 240, 205, 255))
    d.ellipse([cx - r * 0.45, cy - r * 0.45, cx - r * 0.15, cy - r * 0.15], fill=(40, 40, 45, 255))
    d.ellipse([cx + r * 0.15, cy - r * 0.45, cx + r * 0.45, cy - r * 0.15], fill=(40, 40, 45, 255))
    return np.asarray(img, dtype=np.uint8).copy()


def _clean_blob(i: int) -> np.ndarray:
    return _blob_rgba(offset=1.5 * math.sin(i), pulse=math.sin(0.7 * i))


def _noise_rgba(size: int = SIZE, seed: int = 99) -> np.ndarray:
    """Cuadro corrupto: ruido fuerte en todo el lienzo con alfa opaco."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size, size, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    return arr


def _build_frameset(
    in_dir: Path,
    n: int = N_FRAMES,
    corrupt_index: int | None = None,
    meta: dict | None = None,
    corrupt_flags: list[str] | None = None,
) -> FrameSet:
    in_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(n):
        arr = _noise_rgba() if i == corrupt_index else _clean_blob(i)
        fname = frame_filename(i)
        Image.fromarray(arr, "RGBA").save(in_dir / fname)
        flags = list(corrupt_flags) if (corrupt_flags and i == corrupt_index) else []
        frames.append(Frame(index=i, file=fname, duration_ms=100, flags=flags))
    fs = FrameSet(stage="align", frames=frames, meta=dict(meta or {}))
    fs.save(in_dir)
    return fs


def test_stage_registered():
    stage = get_stage("anomaly")
    assert isinstance(stage, AnomalyStage)
    assert stage.name == "anomaly"


def test_only_corrupt_frame_flagged_default_threshold(tmp_path: Path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _build_frameset(in_dir, corrupt_index=CORRUPT_INDEX)

    result = run_stage(AnomalyStage(), in_dir, out_dir)

    for f in result.frames:
        if f.index == CORRUPT_INDEX:
            assert "anomaly" in f.flags and "review" in f.flags
            assert f.scores["quality"] < 0.55
        else:
            assert "anomaly" not in f.flags and "review" not in f.flags
            assert f.scores["quality"] >= 0.55


def test_scores_present_and_in_range(tmp_path: Path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _build_frameset(in_dir, corrupt_index=CORRUPT_INDEX)

    result = run_stage(AnomalyStage(), in_dir, out_dir)

    assert result.stage == "anomaly"
    for f in result.frames:
        for key in ("identity", "coherence", "quality"):
            assert key in f.scores
            assert 0.0 <= f.scores[key] <= 1.0


def test_output_pixels_byte_identical(tmp_path: Path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    fs = _build_frameset(in_dir, corrupt_index=CORRUPT_INDEX)

    run_stage(AnomalyStage(), in_dir, out_dir)

    for f in fs.frames:
        assert (out_dir / f.file).exists()
        assert (out_dir / f.file).read_bytes() == (in_dir / f.file).read_bytes()


def test_summary_in_meta_and_persisted(tmp_path: Path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _build_frameset(in_dir, corrupt_index=CORRUPT_INDEX)

    result = run_stage(AnomalyStage(), in_dir, out_dir)

    summary = result.meta["anomaly_summary"]
    assert summary["flagged"] == 1
    expected_mean = float(np.mean([f.scores["quality"] for f in result.frames]))
    assert summary["mean_quality"] == pytest.approx(expected_mean)
    assert 0.0 < summary["mean_quality"] <= 1.0

    # El manifiesto guardado en out_dir conserva scores, flags y resumen.
    reloaded = FrameSet.load(out_dir)
    assert reloaded.meta["anomaly_summary"]["flagged"] == 1
    assert "anomaly" in reloaded.frames[CORRUPT_INDEX].flags
    assert reloaded.frames[0].scores["quality"] == pytest.approx(
        result.frames[0].scores["quality"]
    )


def test_flags_not_duplicated(tmp_path: Path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _build_frameset(in_dir, corrupt_index=CORRUPT_INDEX, corrupt_flags=["review"])

    result = run_stage(AnomalyStage(), in_dir, out_dir)

    flags = result.frames[CORRUPT_INDEX].flags
    assert flags.count("anomaly") == 1
    assert flags.count("review") == 1


def test_clean_sequence_has_no_flags(tmp_path: Path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    # source_image inexistente -> cae a la referencia mediana sin romperse.
    _build_frameset(in_dir, meta={"source_image": str(tmp_path / "missing.png")})

    result = run_stage(AnomalyStage(), in_dir, out_dir)

    assert result.meta["anomaly_summary"]["flagged"] == 0
    for f in result.frames:
        assert f.flags == []
        assert f.scores["quality"] >= 0.55


def test_source_image_reference_used_when_present(tmp_path: Path):
    ref_path = tmp_path / "ref.png"
    Image.fromarray(_blob_rgba(), "RGBA").save(ref_path)

    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _build_frameset(in_dir, corrupt_index=CORRUPT_INDEX, meta={"source_image": str(ref_path)})

    result = run_stage(AnomalyStage(), in_dir, out_dir)

    corrupt = result.frames[CORRUPT_INDEX]
    clean = [f for f in result.frames if f.index != CORRUPT_INDEX]
    assert "anomaly" in corrupt.flags
    for f in clean:
        assert "anomaly" not in f.flags
        assert f.scores["identity"] > corrupt.scores["identity"]


def test_custom_threshold_param(tmp_path: Path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _build_frameset(in_dir)  # secuencia limpia

    result = run_stage(AnomalyStage(), in_dir, out_dir, {"threshold": 1.01})

    # Con threshold imposible, todo cuadro queda marcado.
    assert result.meta["anomaly_summary"]["flagged"] == len(result.frames)
    for f in result.frames:
        assert "anomaly" in f.flags and "review" in f.flags
