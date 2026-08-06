"""Tests de sprite_pipeline.export (export_animation)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sprite_pipeline.export import export_animation
from sprite_pipeline.models import Frame, FrameSet, frame_filename
from sprite_pipeline.stages.base import save_frame_rgba

SIZE = 64
N = 5
DURATIONS = [80, 120, 100, 150, 200]  # múltiplos de 10 ms: redondean exacto en GIF
COLORS = [
    (255, 0, 0, 255),
    (0, 255, 0, 255),
    (0, 0, 255, 255),
    (255, 255, 0, 255),
    (255, 0, 255, 255),
]
BLOB = (16, 48)  # rango [16, 48) del blob opaco dentro del cuadro 64x64

# Layout esperado con defaults: columns=ceil(sqrt(5))=3, padding=2
COLS = 3
ROWS = 2
CELL = SIZE + 2  # 66
SHEET_W = COLS * CELL  # 198
SHEET_H = ROWS * CELL  # 132


def _blob_frame(i: int) -> np.ndarray:
    arr = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    arr[BLOB[0] : BLOB[1], BLOB[0] : BLOB[1]] = COLORS[i]
    return arr


@pytest.fixture()
def frameset_dir(tmp_path: Path) -> Path:
    """FrameSet sintético: 5 blobs 64x64 de colores y duraciones distintos."""
    d = tmp_path / "frameset"
    d.mkdir()
    frames = []
    for i, dur in enumerate(DURATIONS):
        save_frame_rgba(d, frame_filename(i), _blob_frame(i))
        frames.append(Frame(index=i, file=frame_filename(i), duration_ms=dur))
    FrameSet(stage="smooth", frames=frames, meta={"action": "idle"}).save(d)
    return d


# ------------------------------------------------------------------ retorno

def test_default_export_files_and_result(frameset_dir: Path, tmp_path: Path):
    out = tmp_path / "out"
    res = export_animation(frameset_dir, out)

    assert res["sheet"] == {"columns": COLS, "cell": [CELL, CELL], "count": N}
    expected = {"sheet.png", "sheet.json", "preview.gif"} | {
        f"frames/frame_{i:04d}.png" for i in range(N)
    }
    assert set(res["files"]) == expected  # default sin spriteframes.tres
    for rel in res["files"]:
        assert (out / rel).is_file(), f"falta {rel}"


def test_invalid_args(frameset_dir: Path, tmp_path: Path):
    with pytest.raises(ValueError):
        export_animation(frameset_dir, tmp_path / "a", formats=["webm"])
    with pytest.raises(ValueError):
        export_animation(frameset_dir, tmp_path / "b", scale=0)
    with pytest.raises(ValueError):
        export_animation(frameset_dir, tmp_path / "c", scale=1.5)  # type: ignore[arg-type]


def test_out_of_range_params_raise_valueerror(frameset_dir: Path, tmp_path: Path):
    """Cotas superiores anti-OOM: columns/scale/padding fuera de rango -> ValueError."""
    # columns: máximo max(64, nº de cuadros) = 64 con N=5
    with pytest.raises(ValueError, match="columns"):
        export_animation(frameset_dir, tmp_path / "a", formats=["sheet"], columns=100000)
    with pytest.raises(ValueError, match="columns"):
        export_animation(frameset_dir, tmp_path / "b", formats=["sheet"], columns=65)
    with pytest.raises(ValueError, match="columns"):
        export_animation(frameset_dir, tmp_path / "c", formats=["sheet"], columns=0)
    # scale: 1..8
    with pytest.raises(ValueError, match="scale"):
        export_animation(frameset_dir, tmp_path / "d", formats=["sheet"], scale=9)
    with pytest.raises(ValueError, match="scale"):
        export_animation(frameset_dir, tmp_path / "e", formats=["sheet"], scale=99)
    # padding: 0..64 (y entero)
    with pytest.raises(ValueError, match="padding"):
        export_animation(frameset_dir, tmp_path / "f", formats=["sheet"], padding=65)
    with pytest.raises(ValueError, match="padding"):
        export_animation(frameset_dir, tmp_path / "g", formats=["sheet"], padding=10**9)
    with pytest.raises(ValueError, match="padding"):
        export_animation(frameset_dir, tmp_path / "h", formats=["sheet"], padding=-1)
    # nada fuera de rango escribió artefactos
    for sub in "abcdefgh":
        assert not (tmp_path / sub).exists()


def test_boundary_params_ok(frameset_dir: Path, tmp_path: Path):
    """Los valores frontera (columns=64, padding=64, scale=8) sí se aceptan."""
    out = tmp_path / "cols64"
    res = export_animation(frameset_dir, out, formats=["sheet"], columns=64, padding=0)
    assert res["sheet"]["columns"] == 64
    assert (out / "sheet.png").is_file()

    out2 = tmp_path / "pad-scale"
    res2 = export_animation(frameset_dir, out2, formats=["sheet"], padding=64, scale=8)
    cell = SIZE * 8 + 64
    assert res2["sheet"] == {"columns": COLS, "cell": [cell, cell], "count": N}


# -------------------------------------------------------------------- sheet

def test_sheet_png_dimensions_and_cells(frameset_dir: Path, tmp_path: Path):
    out = tmp_path / "out"
    export_animation(frameset_dir, out, formats=["sheet"])

    img = Image.open(out / "sheet.png")
    assert img.mode == "RGBA"
    assert img.size == (SHEET_W, SHEET_H)

    for i in range(N):
        r, c = divmod(i, COLS)
        # centro del blob del cuadro i, en su celda
        px = img.getpixel((c * CELL + SIZE // 2, r * CELL + SIZE // 2))
        assert px == COLORS[i], f"cuadro {i} fuera de lugar"
        # esquina de la celda (fuera del blob): transparente
        assert img.getpixel((c * CELL + 2, r * CELL + 2))[3] == 0
    # sexta celda (fila 1, col 2) vacía
    assert img.getpixel((2 * CELL + SIZE // 2, CELL + SIZE // 2))[3] == 0


def test_sheet_columns_override(frameset_dir: Path, tmp_path: Path):
    out = tmp_path / "out"
    res = export_animation(frameset_dir, out, formats=["sheet"], columns=5)
    assert res["sheet"]["columns"] == 5
    img = Image.open(out / "sheet.png")
    assert img.size == (5 * CELL, 1 * CELL)


def test_sheet_json_complete(frameset_dir: Path, tmp_path: Path):
    out = tmp_path / "out"
    export_animation(frameset_dir, out, formats=["sheet"])
    data = json.loads((out / "sheet.json").read_text())

    assert set(data) == {"frames", "meta"}
    assert list(data["frames"]) == [frame_filename(i) for i in range(N)]
    for i in range(N):
        entry = data["frames"][frame_filename(i)]
        r, c = divmod(i, COLS)
        assert entry["frame"] == {"x": c * CELL, "y": r * CELL, "w": SIZE, "h": SIZE}
        assert entry["rotated"] is False
        assert entry["trimmed"] is False
        assert entry["spriteSourceSize"] == {"x": 0, "y": 0, "w": SIZE, "h": SIZE}
        assert entry["sourceSize"] == {"w": SIZE, "h": SIZE}
        assert entry["duration"] == DURATIONS[i]

    meta = data["meta"]
    assert meta["app"] == "ai-sprite-pipeline"
    assert meta["image"] == "sheet.png"
    assert meta["size"] == {"w": SHEET_W, "h": SHEET_H}
    assert meta["scale"] == "1"
    assert meta["frameTags"] == [
        {"name": "idle", "from": 0, "to": N - 1, "direction": "forward"}
    ]


# ------------------------------------------------------------------- pngseq

def test_pngseq_renumbered_and_identical(frameset_dir: Path, tmp_path: Path):
    out = tmp_path / "out"
    export_animation(frameset_dir, out, formats=["pngseq"])
    for i in range(N):
        p = out / "frames" / f"frame_{i:04d}.png"
        assert p.is_file()
        arr = np.asarray(Image.open(p).convert("RGBA"))
        assert np.array_equal(arr, _blob_frame(i))
    assert not (out / "frames" / f"frame_{N:04d}.png").exists()


# ---------------------------------------------------------------------- gif

def test_gif_frames_durations_and_transparency(frameset_dir: Path, tmp_path: Path):
    out = tmp_path / "out"
    export_animation(frameset_dir, out, formats=["gif"])

    gif = Image.open(out / "preview.gif")
    assert gif.n_frames == N
    assert gif.info.get("loop", None) == 0
    for i in range(N):
        gif.seek(i)
        assert gif.info["duration"] == DURATIONS[i]
        rgba = gif.convert("RGBA")
        r, g, b, a = rgba.getpixel((SIZE // 2, SIZE // 2))
        assert a == 255
        exp = COLORS[i]
        assert abs(r - exp[0]) <= 8 and abs(g - exp[1]) <= 8 and abs(b - exp[2]) <= 8
        # fuera del blob: transparente (o, en el fallback documentado, #2b2b2b)
        cr, cg, cb, ca = rgba.getpixel((2, 2))
        assert ca == 0 or (cr, cg, cb) == (0x2B, 0x2B, 0x2B)


# -------------------------------------------------------------------- godot

def test_godot_tres_includes_sheet_and_regions(frameset_dir: Path, tmp_path: Path):
    out = tmp_path / "out"
    res = export_animation(frameset_dir, out, formats=["godot"])

    # godot incluye el sheet aunque "sheet" no esté en formats
    assert (out / "sheet.png").is_file()
    assert (out / "sheet.json").is_file()
    assert "sheet.png" in res["files"]
    assert "spriteframes.tres" in res["files"]
    assert res["sheet"]["columns"] == COLS

    content = (out / "spriteframes.tres").read_text()
    assert content.startswith("[gd_resource ")
    assert 'type="SpriteFrames"' in content
    assert f"load_steps={N + 2}" in content
    assert "format=3" in content
    assert "sheet.png" in content
    assert content.count('[sub_resource type="AtlasTexture"') == N
    assert content.count("SubResource(") == N
    for i in range(N):
        r, c = divmod(i, COLS)
        assert f"Rect2({c * CELL}, {r * CELL}, {SIZE}, {SIZE})" in content
    assert '&"idle"' in content
    assert '"loop": true' in content


# -------------------------------------------------------------------- scale

def test_scale_2_doubles_dimensions(frameset_dir: Path, tmp_path: Path):
    out = tmp_path / "out"
    res = export_animation(
        frameset_dir, out, formats=["sheet", "pngseq", "gif"], scale=2
    )

    cell2 = SIZE * 2 + 2  # 130: cuadro duplicado + padding
    assert res["sheet"] == {"columns": COLS, "cell": [cell2, cell2], "count": N}

    sheet = Image.open(out / "sheet.png")
    assert sheet.size == (COLS * cell2, ROWS * cell2)
    # centro del blob del cuadro 0, ahora a escala 2 (NEAREST: color exacto)
    assert sheet.getpixel((SIZE, SIZE)) == COLORS[0]

    frame0 = Image.open(out / "frames" / "frame_0000.png")
    assert frame0.size == (SIZE * 2, SIZE * 2)
    arr = np.asarray(frame0.convert("RGBA"))
    # NEAREST preserva colores exactos y bordes duros
    assert tuple(arr[SIZE, SIZE]) == COLORS[0]
    assert arr[BLOB[0] * 2 - 1, BLOB[0] * 2 - 1, 3] == 0
    assert arr[BLOB[0] * 2, BLOB[0] * 2, 3] == 255

    data = json.loads((out / "sheet.json").read_text())
    entry = data["frames"]["frame_0000.png"]
    assert entry["sourceSize"] == {"w": SIZE * 2, "h": SIZE * 2}
    assert data["meta"]["size"] == {"w": COLS * cell2, "h": ROWS * cell2}

    gif = Image.open(out / "preview.gif")
    assert gif.size == (SIZE * 2, SIZE * 2)
    assert gif.n_frames == N
