"""Gestión de versiones de edición sobre ``Paths.edits_dir``.

Cada operación del editor crea una versión nueva ``edit_NNNN`` (copia física
completa de la anterior) y aplica la mutación ahí, de modo que ``undo`` es
simplemente borrar la última versión. Tras cada operación los archivos se
renombran a la convención ``frame_NNNN.png`` consistente con el orden de la
lista y se persiste el manifiesto.
"""

from __future__ import annotations

import base64
import io
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from sprite_pipeline.models import MANIFEST_NAME, Frame, FrameSet, frame_filename
from sprite_pipeline.ops.interpolate import interpolate_frames
from sprite_pipeline.ops.retouch import retouch_region
from sprite_pipeline.stages.base import load_frame_rgba, save_frame_rgba
from sprite_pipeline.storage import Paths

VALID_OPS = ("delete", "duplicate", "reorder", "set_duration", "interpolate", "retouch")

#: Un lock por animación (keyed por ``str(paths.animation_dir)``): serializa
#: ensure_edit_session / apply_edit / undo entre hilos (los endpoints FastAPI
#: síncronos corren en threadpool, así que hay concurrencia real).
_ANIM_LOCKS: dict[str, threading.RLock] = {}
_ANIM_LOCKS_GUARD = threading.Lock()


def _animation_lock(paths: Paths) -> threading.RLock:
    key = str(paths.animation_dir)
    with _ANIM_LOCKS_GUARD:
        lock = _ANIM_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _ANIM_LOCKS[key] = lock
        return lock


# --------------------------------------------------------------------- helpers

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _check_index(value, n: int, name: str = "index") -> int:
    try:
        idx = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} inválido: {value!r}") from None
    if not 0 <= idx < n:
        raise ValueError(f"{name} fuera de rango: {idx} (hay {n} cuadros)")
    return idx


def _free_name(directory: Path, prefix: str) -> str:
    """Nombre de archivo libre ``{prefix}NNNN.png`` en ``directory``."""
    k = 0
    while True:
        name = f"{prefix}{k:04d}.png"
        if not (directory / name).exists():
            return name
        k += 1


def _renumber_files(fs: FrameSet, directory: Path) -> None:
    """Renombra los PNGs a ``frame_NNNN.png`` según el orden de la lista.

    Dos fases (via nombres temporales) para evitar colisiones al reordenar.
    """
    tmp_names: list[str] = []
    for k, fr in enumerate(fs.frames):
        tmp = f"__renumber_{k:04d}.png"
        (directory / fr.file).rename(directory / tmp)
        tmp_names.append(tmp)
    for k, (fr, tmp) in enumerate(zip(fs.frames, tmp_names)):
        final = frame_filename(k)
        (directory / tmp).rename(directory / final)
        fr.file = final
    fs.reindex()


# ------------------------------------------------------------------ sesión/undo

def ensure_edit_session(paths: Paths) -> Path:
    """Devuelve el dir de la versión de edición actual, creándola si no existe.

    Sin ediciones previas (válidas) copia el último stage dir COMPLETO a una
    versión nueva. La copia se hace a un nombre temporal y se renombra de forma
    atómica, de modo que nunca queda una versión parcial sin manifiesto.
    """
    with _animation_lock(paths):
        versions = paths.list_edit_versions()
        if versions:
            return paths.edit_dir(versions[-1])
        src = paths.latest_stage_dir()
        if src is None:
            raise FileNotFoundError(
                "No hay etapas procesadas para esta animación: nada que editar"
            )
        if not (src / MANIFEST_NAME).exists():
            raise FileNotFoundError(
                "La animación aún se está procesando (etapa sin manifiesto): "
                "no se puede editar todavía"
            )
        dst = paths.edit_dir(paths.next_edit_version())
        tmp = dst.with_name(dst.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        try:
            shutil.copytree(src, tmp)
            os.replace(tmp, dst)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return dst


def undo(paths: Paths) -> FrameSet | None:
    """Borra la última versión de edición si hay más de una; devuelve la anterior.

    Solo cuenta versiones válidas (con manifiesto): las copias parciales que
    hubiera dejado un fallo previo se ignoran.
    """
    with _animation_lock(paths):
        versions = paths.list_edit_versions()
        if len(versions) <= 1:
            return None
        shutil.rmtree(paths.edit_dir(versions[-1]))
        return FrameSet.load(paths.edit_dir(versions[-2]))


# -------------------------------------------------------------------- apply_edit

def apply_edit(paths: Paths, op: dict) -> FrameSet:
    """Crea ``edit_{n+1}`` (copia física de la versión actual) y aplica ``op``.

    La versión nueva se construye en un dir temporal y solo se renombra al
    nombre final tras guardar el manifiesto: ante cualquier fallo el temporal
    se limpia y el historial de versiones queda intacto.
    """
    if not isinstance(op, dict) or "op" not in op:
        raise ValueError("La operación debe ser un dict con clave 'op'")
    name = op["op"]
    if name not in VALID_OPS:
        raise ValueError(f"Operación desconocida: {name!r}. Válidas: {list(VALID_OPS)}")

    with _animation_lock(paths):
        current = ensure_edit_session(paths)
        new_dir = paths.edit_dir(paths.next_edit_version())
        tmp_dir = new_dir.with_name(new_dir.name + ".tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        try:
            shutil.copytree(current, tmp_dir)
            fs = FrameSet.load(tmp_dir)
            _OP_HANDLERS[name](fs, tmp_dir, op)
            _renumber_files(fs, tmp_dir)
            params = {k: v for k, v in op.items() if k != "mask_png_base64"}
            fs.history = [*fs.history, {"stage": "edit", "params": params, "at": _now()}]
            fs.stage = "edit"
            fs.save(tmp_dir)
            os.replace(tmp_dir, new_dir)
            return fs
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise


# ------------------------------------------------------------------ operaciones

def _op_delete(fs: FrameSet, directory: Path, op: dict) -> None:
    idx = _check_index(op.get("index"), len(fs.frames))
    if len(fs.frames) <= 1:
        raise ValueError("No se puede borrar el último cuadro de la animación")
    frame = fs.frames.pop(idx)
    (directory / frame.file).unlink(missing_ok=True)


def _op_duplicate(fs: FrameSet, directory: Path, op: dict) -> None:
    idx = _check_index(op.get("index"), len(fs.frames))
    src = fs.frames[idx]
    name = _free_name(directory, "frame_dup")
    shutil.copy2(directory / src.file, directory / name)
    dup = Frame(
        index=idx + 1,
        file=name,
        duration_ms=src.duration_ms,
        flags=list(src.flags),
        scores=dict(src.scores),
        transform=dict(src.transform) if src.transform else None,
        source="duplicated",
    )
    fs.frames.insert(idx + 1, dup)


def _op_reorder(fs: FrameSet, directory: Path, op: dict) -> None:
    order = op.get("order")
    n = len(fs.frames)
    if not isinstance(order, (list, tuple)) or sorted(int(i) for i in order) != list(range(n)):
        raise ValueError(f"'order' debe ser una permutación de 0..{n - 1}; llegó {order!r}")
    fs.frames = [fs.frames[int(i)] for i in order]


def _op_set_duration(fs: FrameSet, directory: Path, op: dict) -> None:
    idx = _check_index(op.get("index"), len(fs.frames))
    try:
        duration = int(op.get("duration_ms"))
    except (TypeError, ValueError):
        raise ValueError(f"duration_ms inválido: {op.get('duration_ms')!r}") from None
    if duration <= 0:
        raise ValueError(f"duration_ms debe ser positivo; llegó {duration}")
    fs.frames[idx].duration_ms = duration


def _op_interpolate(fs: FrameSet, directory: Path, op: dict) -> None:
    i = _check_index(op.get("after_index"), len(fs.frames), name="after_index")
    j = (i + 1) % len(fs.frames)  # envolvente: tras el último, contra el primero
    a = load_frame_rgba(directory, fs.frames[i])
    b = load_frame_rgba(directory, fs.frames[j])
    mid = interpolate_frames(a, b, 0.5)
    name = _free_name(directory, "frame_interp")
    save_frame_rgba(directory, name, mid)
    frame = Frame(
        index=i + 1,
        file=name,
        duration_ms=fs.frames[i].duration_ms,
        source="interpolated",
    )
    fs.frames.insert(i + 1, frame)


def _op_retouch(fs: FrameSet, directory: Path, op: dict) -> None:
    idx = _check_index(op.get("index"), len(fs.frames))
    b64 = op.get("mask_png_base64")
    if not b64:
        raise ValueError("retouch requiere 'mask_png_base64' (PNG en escala de grises)")
    try:
        raw = base64.b64decode(b64)
        mask_img = Image.open(io.BytesIO(raw)).convert("L")
    except Exception:
        raise ValueError("mask_png_base64 no es un PNG base64 válido") from None

    img = load_frame_rgba(directory, fs.frames[idx])
    h, w = img.shape[:2]
    if mask_img.size != (w, h):
        mask_img = mask_img.resize((w, h), Image.NEAREST)
    mask = np.asarray(mask_img, dtype=np.uint8)

    n = len(fs.frames)
    neighbors: list[np.ndarray] = []
    if n > 1:
        prev_i, next_i = (idx - 1) % n, (idx + 1) % n
        neighbor_ids = [prev_i] if prev_i == next_i else [prev_i, next_i]
        neighbors = [load_frame_rgba(directory, fs.frames[k]) for k in neighbor_ids]

    out = retouch_region(img, mask, neighbors)
    save_frame_rgba(directory, fs.frames[idx].file, out)
    fs.frames[idx].source = "retouched"


_OP_HANDLERS = {
    "delete": _op_delete,
    "duplicate": _op_duplicate,
    "reorder": _op_reorder,
    "set_duration": _op_set_duration,
    "interpolate": _op_interpolate,
    "retouch": _op_retouch,
}
