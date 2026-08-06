"""Exportación de un FrameSet a formatos consumibles por motores de juego.

``export_animation`` carga el FrameSet de ``frameset_dir`` y escribe en
``out_dir`` los artefactos pedidos en ``formats``:

- ``sheet``: ``sheet.png`` (grilla RGBA) + ``sheet.json`` en dialecto
  TexturePacker "frames hash" (compatible Aseprite/Phaser/PixiJS).
- ``pngseq``: ``frames/frame_0000.png`` ... (renumerados según el manifiesto).
- ``gif``: ``preview.gif`` respetando ``duration_ms`` por cuadro, con
  transparencia sobre paleta (disposal=2). Los píxeles semitransparentes se
  componen sobre ``#2b2b2b`` para evitar halos; si la codificación con
  transparencia falla, el fallback documentado compone TODO el cuadro sobre
  ``#2b2b2b`` y emite un GIF opaco.
- ``godot``: ``spriteframes.tres`` (recurso SpriteFrames de Godot 4 con un
  AtlasTexture por región sobre ``sheet.png``). Pedir ``godot`` genera el
  sheet aunque ``sheet`` no esté en ``formats``.

``scale`` (entero >= 1) reescala los cuadros con ``PIL.Image.NEAREST`` para
mantener los sprites nítidos. Las celdas del sheet miden
``max(ancho), max(alto)`` de los cuadros (ya escalados) más ``padding``.

Devuelve ``{"files": [rutas relativas a out_dir],
"sheet": {"columns": ..., "cell": [w, h], "count": n}}`` (``sheet`` es
``None`` si ningún formato pedido necesitó el sheet).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from sprite_pipeline.models import FrameSet, frame_filename
from sprite_pipeline.stages.base import load_frame_rgba, save_frame_rgba

DEFAULT_FORMATS = ("sheet", "pngseq", "gif")
KNOWN_FORMATS = frozenset({"sheet", "pngseq", "gif", "godot"})

#: Color de fondo del fallback del GIF (y base de composición de semialfas).
GIF_FALLBACK_BG = (0x2B, 0x2B, 0x2B)

#: Índice de paleta reservado para el color transparente del GIF.
_GIF_TRANSPARENT_INDEX = 255


def export_animation(
    frameset_dir: Path,
    out_dir: Path,
    formats: list[str] | None = None,
    columns: int | None = None,
    padding: int = 2,
    scale: int = 1,
) -> dict:
    """Exporta el FrameSet de ``frameset_dir`` a ``out_dir``. Ver módulo."""
    frameset_dir = Path(frameset_dir)
    out_dir = Path(out_dir)
    formats = list(formats) if formats is not None else list(DEFAULT_FORMATS)
    unknown = sorted(set(formats) - KNOWN_FORMATS)
    if unknown:
        raise ValueError(
            f"Formatos desconocidos: {unknown}. Soportados: {sorted(KNOWN_FORMATS)}"
        )
    if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
        raise ValueError(f"scale debe ser un entero >= 1; llegó {scale!r}")
    if padding < 0:
        raise ValueError(f"padding debe ser >= 0; llegó {padding!r}")
    if columns is not None and columns < 1:
        raise ValueError(f"columns debe ser >= 1; llegó {columns!r}")

    fs = FrameSet.load(frameset_dir)
    if not fs.frames:
        raise ValueError(f"FrameSet sin cuadros en {frameset_dir}")

    arrays = [load_frame_rgba(frameset_dir, f) for f in fs.frames]
    if scale > 1:
        arrays = [_scale_nearest(a, scale) for a in arrays]
    durations = [int(f.duration_ms) for f in fs.frames]

    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    sheet_info: dict | None = None
    layout: dict | None = None

    # El sheet se necesita también para godot, aunque no esté en formats.
    if "sheet" in formats or "godot" in formats:
        layout = _build_sheet(arrays, columns, padding)
        save_frame_rgba(out_dir, "sheet.png", layout["image"])
        files.append("sheet.png")
        manifest = _sheet_manifest(fs, layout, durations)
        (out_dir / "sheet.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False)
        )
        files.append("sheet.json")
        sheet_info = {
            "columns": layout["columns"],
            "cell": [layout["cell_w"], layout["cell_h"]],
            "count": len(arrays),
        }

    if "pngseq" in formats:
        seq_dir = out_dir / "frames"
        seq_dir.mkdir(parents=True, exist_ok=True)
        for i, arr in enumerate(arrays):
            name = frame_filename(i)  # renumerado según el orden del manifiesto
            save_frame_rgba(seq_dir, name, arr)
            files.append(f"frames/{name}")

    if "gif" in formats:
        _write_gif(out_dir / "preview.gif", arrays, durations)
        files.append("preview.gif")

    if "godot" in formats:
        assert layout is not None
        (out_dir / "spriteframes.tres").write_text(
            _godot_spriteframes(fs, layout, durations)
        )
        files.append("spriteframes.tres")

    return {"files": files, "sheet": sheet_info}


# ----------------------------------------------------------------- helpers

def _scale_nearest(arr: np.ndarray, scale: int) -> np.ndarray:
    """Reescala un cuadro RGBA con vecino más cercano (sprites nítidos)."""
    h, w = arr.shape[:2]
    img = Image.fromarray(arr, mode="RGBA").resize((w * scale, h * scale), Image.NEAREST)
    return np.asarray(img, dtype=np.uint8).copy()


def _build_sheet(arrays: list[np.ndarray], columns: int | None, padding: int) -> dict:
    """Compone la grilla del sprite sheet.

    Cada celda mide ``max(w) + padding`` x ``max(h) + padding``; el cuadro se
    apoya en la esquina superior izquierda de su celda, de modo que el padding
    queda a la derecha/abajo de cada sprite.
    """
    n = len(arrays)
    cols = int(columns) if columns else math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    max_w = max(a.shape[1] for a in arrays)
    max_h = max(a.shape[0] for a in arrays)
    cell_w = max_w + padding
    cell_h = max_h + padding
    sheet = np.zeros((rows * cell_h, cols * cell_w, 4), dtype=np.uint8)
    regions: list[tuple[int, int, int, int]] = []
    for i, arr in enumerate(arrays):
        r, c = divmod(i, cols)
        x, y = c * cell_w, r * cell_h
        h, w = arr.shape[:2]
        sheet[y : y + h, x : x + w] = arr
        regions.append((x, y, w, h))
    return {
        "image": sheet,
        "columns": cols,
        "rows": rows,
        "cell_w": cell_w,
        "cell_h": cell_h,
        "width": sheet.shape[1],
        "height": sheet.shape[0],
        "regions": regions,
    }


def _animation_name(fs: FrameSet) -> str:
    return str(fs.meta.get("action") or "animation")


def _sheet_manifest(fs: FrameSet, layout: dict, durations: list[int]) -> dict:
    """``sheet.json`` en dialecto TexturePacker "frames hash" (Aseprite/Phaser)."""
    frames_hash: dict[str, dict] = {}
    for i, (x, y, w, h) in enumerate(layout["regions"]):
        frames_hash[frame_filename(i)] = {
            "frame": {"x": x, "y": y, "w": w, "h": h},
            "rotated": False,
            "trimmed": False,
            "spriteSourceSize": {"x": 0, "y": 0, "w": w, "h": h},
            "sourceSize": {"w": w, "h": h},
            "duration": durations[i],
        }
    return {
        "frames": frames_hash,
        "meta": {
            "app": "ai-sprite-pipeline",
            "image": "sheet.png",
            "size": {"w": layout["width"], "h": layout["height"]},
            "scale": "1",
            "frameTags": [
                {
                    "name": _animation_name(fs),
                    "from": 0,
                    "to": len(durations) - 1,
                    "direction": "forward",
                }
            ],
        },
    }


# --------------------------------------------------------------------- GIF

def _gif_frame_transparent(arr: np.ndarray) -> Image.Image:
    """Cuadro paletizado con transparencia binaria en el índice reservado.

    Los píxeles con alfa parcial se componen sobre ``GIF_FALLBACK_BG`` (el GIF
    no soporta semitransparencia); los píxeles con alfa < 128 quedan en el
    índice transparente.
    """
    img = Image.fromarray(arr, mode="RGBA")
    alpha = img.getchannel("A")
    bg = Image.new("RGBA", img.size, (*GIF_FALLBACK_BG, 255))
    rgb = Image.alpha_composite(bg, img).convert("RGB")
    # colors=255 deja libre el índice 255 para la transparencia.
    pal = rgb.convert("P", palette=Image.ADAPTIVE, colors=_GIF_TRANSPARENT_INDEX)
    mask = alpha.point(lambda a: 255 if a < 128 else 0)
    pal.paste(_GIF_TRANSPARENT_INDEX, mask=mask)
    return pal


def _gif_frame_opaque(arr: np.ndarray) -> Image.Image:
    """Fallback documentado: compone todo el cuadro sobre ``#2b2b2b``."""
    img = Image.fromarray(arr, mode="RGBA")
    bg = Image.new("RGBA", img.size, (*GIF_FALLBACK_BG, 255))
    rgb = Image.alpha_composite(bg, img).convert("RGB")
    return rgb.convert("P", palette=Image.ADAPTIVE, colors=256)


def _write_gif(path: Path, arrays: list[np.ndarray], durations: list[int]) -> None:
    """Escribe ``preview.gif`` con duración por cuadro y loop infinito."""
    try:
        frames = [_gif_frame_transparent(a) for a in arrays]
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            disposal=2,
            transparency=_GIF_TRANSPARENT_INDEX,
            optimize=False,
        )
    except Exception:
        # Fallback documentado: si el alfa da problemas al codificar,
        # componer sobre #2b2b2b y emitir un GIF opaco.
        frames = [_gif_frame_opaque(a) for a in arrays]
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            disposal=2,
            optimize=False,
        )


# ------------------------------------------------------------------- Godot

def _godot_spriteframes(fs: FrameSet, layout: dict, durations: list[int]) -> str:
    """Recurso ``SpriteFrames`` (texto, Godot 4) con AtlasTexture por región.

    ``speed`` se fija en 1000/duración media y cada cuadro lleva su duración
    relativa, de modo que el tiempo real por cuadro respeta ``duration_ms``.
    """
    n = len(durations)
    avg = (sum(durations) / n) or 100.0
    speed = round(1000.0 / avg, 4)
    name = _animation_name(fs)

    lines = [
        f'[gd_resource type="SpriteFrames" load_steps={n + 2} format=3]',
        "",
        '[ext_resource type="Texture2D" path="res://sheet.png" id="1_sheet"]',
        "",
    ]
    for i, (x, y, w, h) in enumerate(layout["regions"]):
        lines += [
            f'[sub_resource type="AtlasTexture" id="AtlasTexture_{i:04d}"]',
            'atlas = ExtResource("1_sheet")',
            f"region = Rect2({x}, {y}, {w}, {h})",
            "",
        ]
    entries = []
    for i, dur in enumerate(durations):
        rel = round(dur / avg, 4)
        entries.append(
            "{\n"
            f'"duration": {rel},\n'
            f'"texture": SubResource("AtlasTexture_{i:04d}")\n'
            "}"
        )
    lines += [
        "[resource]",
        "animations = [{",
        '"frames": [' + ", ".join(entries) + "],",
        '"loop": true,',
        f'"name": &"{name}",',
        f'"speed": {speed}',
        "}]",
        "",
    ]
    return "\n".join(lines)
