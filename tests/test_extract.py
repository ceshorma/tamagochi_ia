"""Tests de ``extract_frames`` sobre un video sintético (determinista, sin red).

El video de prueba tiene 12 cuadros únicos de contenido muy distinto entre sí
más 3 duplicados consecutivos exactos (15 cuadros en total, 96x96 @ 10 fps).
mp4 es lossy: los duplicados decodificados quedan casi idénticos (similitud
> 0.995) y los cuadros únicos, al ser drásticamente distintos, quedan muy por
debajo del umbral.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PIL import Image

from sprite_pipeline.extract import extract_frames
from sprite_pipeline.models import FrameSet, frame_filename

SIZE = 96
FPS_VIDEO = 10
N_UNIQUE = 12
DUP_AFTER = (1, 4, 8)  # tras estos cuadros únicos se inserta un duplicado exacto
N_TOTAL = N_UNIQUE + len(DUP_AFTER)  # 15

_PALETTE = [
    (230, 40, 40),
    (40, 200, 60),
    (40, 70, 230),
    (230, 220, 40),
    (200, 40, 200),
    (40, 210, 210),
    (140, 70, 20),
    (20, 110, 60),
    (90, 40, 150),
    (210, 130, 40),
    (60, 60, 60),
    (200, 200, 200),
]


def _unique_frame(i: int) -> np.ndarray:
    """Cuadro RGB claramente distinto de los demás (fondo + bloque móvil)."""
    arr = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    arr[:, :] = _PALETTE[i % len(_PALETTE)]
    x = (i * 7) % (SIZE - 16)
    arr[10:30, x:x + 16] = 255 - arr[10:30, x:x + 16]
    return arr


def _video_frames() -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for i in range(N_UNIQUE):
        f = _unique_frame(i)
        frames.append(f)
        if i in DUP_AFTER:
            frames.append(f.copy())  # duplicado consecutivo exacto
    return frames


@pytest.fixture(scope="module")
def video_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("video") / "synthetic.mp4"
    imageio.mimwrite(str(path), _video_frames(), fps=FPS_VIDEO, macro_block_size=1, quality=8)
    return path


def test_no_dedupe_extracts_all_frames_as_rgba_pngs(video_path: Path, tmp_path: Path):
    out = tmp_path / "raw"
    fs = extract_frames(video_path, out, fps=FPS_VIDEO, dedupe_threshold=1.5)

    assert fs.stage == "raw"
    assert len(fs.frames) == N_TOTAL
    for i, fr in enumerate(fs.frames):
        assert fr.index == i
        assert fr.file == frame_filename(i)
        assert fr.duration_ms == 100  # round(1000 / 10)
        png = out / fr.file
        assert png.exists()
        with Image.open(png) as img:
            assert img.mode == "RGBA"
            assert img.size == (SIZE, SIZE)


def test_dedupe_discards_consecutive_duplicates(video_path: Path, tmp_path: Path):
    out = tmp_path / "raw"
    fs = extract_frames(video_path, out, fps=FPS_VIDEO, dedupe_threshold=0.995)

    # Los 3 duplicados consecutivos desaparecen; los 12 únicos quedan.
    assert len(fs.frames) == N_UNIQUE

    # Los cuadros conservados son consecutivamente distintos según la métrica.
    arrays = [
        np.asarray(Image.open(out / fr.file).convert("RGB"), dtype=np.float32)
        for fr in fs.frames
    ]
    for a, b in zip(arrays, arrays[1:]):
        similarity = 1.0 - float(np.mean(np.abs(a - b))) / 255.0
        assert similarity <= 0.995


def test_manifest_roundtrip_and_meta_merge(video_path: Path, tmp_path: Path):
    out = tmp_path / "raw"
    meta_in = {"provider": "mock", "action": "idle"}
    fs = extract_frames(video_path, out, fps=FPS_VIDEO, dedupe_threshold=0.995, meta=meta_in)

    # El dict de entrada no se muta (se copia antes de fusionar).
    assert meta_in == {"provider": "mock", "action": "idle"}

    loaded = FrameSet.load(out)
    assert loaded.stage == "raw"
    assert loaded.meta["fps"] == FPS_VIDEO
    assert loaded.meta["video"] == str(video_path)
    assert loaded.meta["provider"] == "mock"
    assert loaded.meta["action"] == "idle"
    assert len(loaded.frames) == len(fs.frames)
    assert [f.file for f in loaded.frames] == [f.file for f in fs.frames]
    assert all((out / f.file).exists() for f in loaded.frames)


def test_subsampling_to_half_fps(video_path: Path, tmp_path: Path):
    out = tmp_path / "raw"
    fs = extract_frames(video_path, out, fps=5, dedupe_threshold=1.5)

    # Índices round(k * 10 / 5) = 0, 2, 4, ..., 14 -> 8 cuadros.
    assert len(fs.frames) == 8
    assert all(f.duration_ms == 200 for f in fs.frames)  # round(1000 / 5)
    assert [f.file for f in fs.frames] == [frame_filename(i) for i in range(8)]


def test_fps_above_video_fps_takes_all_frames_once(video_path: Path, tmp_path: Path):
    out = tmp_path / "raw"
    fs = extract_frames(video_path, out, fps=12, dedupe_threshold=1.5)

    # Índices round(k * 10 / 12) únicos cubren los 15 cuadros sin repetirlos.
    assert len(fs.frames) == N_TOTAL
    # El timing refleja el fps EFECTIVO de muestreo min(12, 10) = 10, no el
    # pedido: no se pueden muestrear más cuadros de los que el video tiene.
    assert all(f.duration_ms == 100 for f in fs.frames)  # round(1000 / 10)
    assert fs.meta["fps"] == FPS_VIDEO


@pytest.fixture(scope="module")
def video_8fps_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("video8") / "synthetic_8fps.mp4"
    imageio.mimwrite(str(path), _video_frames(), fps=8, macro_block_size=1, quality=8)
    return path


def test_fps_24_over_8fps_video_keeps_real_timing(video_8fps_path: Path, tmp_path: Path):
    """Pedir fps=24 sobre un video de 8 fps toma todos los cuadros una vez y
    el timing usa el fps efectivo min(24, 8) = 8: duration_ms = 125.

    Regresión: antes duration_ms = round(1000/24) = 42, reproduciendo la
    animación 3x más rápida que el video fuente."""
    out = tmp_path / "raw"
    fs = extract_frames(video_8fps_path, out, fps=24, dedupe_threshold=1.5)

    assert len(fs.frames) == N_TOTAL  # todos los cuadros, una sola vez
    assert all(f.duration_ms == 125 for f in fs.frames)  # round(1000 / 8)
    assert fs.meta["fps"] == 8


def test_no_fps_metadata_takes_all_frames_with_requested_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Sin metadata válida de fps: se toman TODOS los cuadros y el timing usa
    el fps pedido como suposición (documentado en extract_frames)."""
    rgba_frames = [
        np.dstack([_unique_frame(i), np.full((SIZE, SIZE, 1), 255, np.uint8)])
        for i in range(6)
    ]
    # El lector reporta fps_video = 0.0 (contenedor sin metadata de fps).
    monkeypatch.setattr(
        "sprite_pipeline.extract._read_with_cv2", lambda path: (rgba_frames, 0.0)
    )
    video = tmp_path / "no_meta.mp4"
    video.write_bytes(b"stub")

    out = tmp_path / "raw"
    fs = extract_frames(video, out, fps=12, dedupe_threshold=1.5)

    assert len(fs.frames) == 6  # todos los cuadros del video
    assert all(f.duration_ms == 83 for f in fs.frames)  # round(1000 / 12)
    assert fs.meta["fps"] == 12
