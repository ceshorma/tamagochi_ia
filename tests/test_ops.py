"""Tests de sprite_pipeline.ops: interpolate, retouch y edits (end-to-end)."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from sprite_pipeline.models import Frame, FrameSet, frame_filename
from sprite_pipeline.ops import interpolate as interp_mod
from sprite_pipeline.ops.edits import apply_edit, ensure_edit_session, undo
from sprite_pipeline.ops.interpolate import interpolate_frames
from sprite_pipeline.ops.retouch import retouch_region
from sprite_pipeline.stages.base import load_frame_rgba, save_frame_rgba
from sprite_pipeline.storage import Paths

SIZE = 96
# Blobs distinguibles por color y posición (uno por cuadro del FrameSet sintético).
COLORS = [(220, 60, 60, 255), (60, 220, 60, 255), (60, 60, 220, 255), (220, 220, 60, 255)]
CENTERS = [30, 42, 54, 66]


# ------------------------------------------------------------------- helpers

def circle_frame(cx: int, cy: int = 48, r: int = 12,
                 color: tuple = (200, 80, 60, 255), size: int = SIZE) -> np.ndarray:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    return np.asarray(img, dtype=np.uint8).copy()


def centroid_x(rgba: np.ndarray) -> float:
    alpha = rgba[..., 3].astype(np.float64)
    xs = np.arange(rgba.shape[1], dtype=np.float64)
    return float((alpha.sum(axis=0) * xs).sum() / alpha.sum())


def frame_color(directory: Path, filename: str) -> np.ndarray:
    """RGB medio de los píxeles opacos del PNG (identifica el blob)."""
    arr = np.asarray(Image.open(Path(directory) / filename).convert("RGBA"))
    mask = arr[..., 3] > 200
    return arr[..., :3][mask].mean(axis=0)


def assert_color(directory: Path, filename: str, rgba: tuple) -> None:
    got = frame_color(directory, filename)
    assert np.allclose(got, rgba[:3], atol=8), f"{filename}: {got} != {rgba[:3]}"


def png_names(directory: Path) -> list[str]:
    return sorted(p.name for p in Path(directory).glob("frame_*.png"))


def mask_b64(size: int = SIZE, box: tuple = (40, 40, 64, 64)) -> str:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rectangle(list(box), fill=255)
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture()
def anim(data_dir: Path) -> Paths:
    """Stage dir sintético 05_smooth con 4 blobs + manifiesto bajo un Paths."""
    paths = Paths("p", "a", data_dir=data_dir)
    stage_dir = paths.stages_dir / "05_smooth"
    stage_dir.mkdir(parents=True)
    frames = []
    for i in range(4):
        fname = frame_filename(i)
        save_frame_rgba(stage_dir, fname, circle_frame(CENTERS[i], color=COLORS[i]))
        frames.append(Frame(index=i, file=fname, duration_ms=100))
    fs = FrameSet(stage="smooth", frames=frames, meta={"size": [SIZE, SIZE]},
                  history=[{"stage": "smooth", "params": {}}])
    fs.save(stage_dir)
    return paths


# ------------------------------------------------------- interpolate_frames

def test_interpolate_frames_moving_circle():
    a = circle_frame(40, r=14)
    b = circle_frame(56, r=14)
    mid = interpolate_frames(a, b, 0.5)

    assert mid.shape == a.shape and mid.dtype == np.uint8
    cx = centroid_x(mid)
    assert 44.0 < cx < 52.0, f"centroide {cx} no está entre ambos círculos"
    # El flujo produce UN blob coherente en el medio; un crossfade dejaría dos
    # fantasmas semitransparentes con muy poca área opaca.
    area_mid = int((mid[..., 3] > 128).sum())
    area_a = int((a[..., 3] > 128).sum())
    assert 0.7 * area_a < area_mid < 1.5 * area_a


def test_interpolate_frames_t_asimetrico():
    a = circle_frame(40, r=14)
    b = circle_frame(56, r=14)
    quarter = centroid_x(interpolate_frames(a, b, 0.25))
    three_q = centroid_x(interpolate_frames(a, b, 0.75))
    assert 41.0 < quarter < 47.5
    assert 48.5 < three_q < 55.0
    assert quarter < three_q


def test_interpolate_fallback_si_flujo_lanza(monkeypatch: pytest.MonkeyPatch):
    a = circle_frame(40)
    b = circle_frame(56)

    def boom(*args, **kwargs):
        raise interp_mod.cv2.error("flujo roto")

    monkeypatch.setattr(interp_mod.cv2, "calcOpticalFlowFarneback", boom)
    out = interpolate_frames(a, b, 0.5)
    expected = np.clip(0.5 * a.astype(np.float32) + 0.5 * b.astype(np.float32), 0, 255).astype(np.uint8)
    assert np.allclose(out.astype(int), expected.astype(int), atol=1)


def test_interpolate_fallback_si_flujo_nan(monkeypatch: pytest.MonkeyPatch):
    a = circle_frame(40)
    b = circle_frame(56)
    h, w = a.shape[:2]

    monkeypatch.setattr(
        interp_mod.cv2, "calcOpticalFlowFarneback",
        lambda *args, **kwargs: np.full((h, w, 2), np.nan, dtype=np.float32),
    )
    out = interpolate_frames(a, b, 0.5)
    expected = np.clip(0.5 * a.astype(np.float32) + 0.5 * b.astype(np.float32), 0, 255).astype(np.uint8)
    assert np.allclose(out.astype(int), expected.astype(int), atol=1)


def test_interpolate_valida_entrada():
    with pytest.raises(ValueError):
        interpolate_frames(np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8, 4), np.uint8))


# ----------------------------------------------------------- retouch_region

def _noisy_over_base():
    base = circle_frame(48, r=30, color=(90, 180, 120, 255))
    rng = np.random.default_rng(7)
    noisy = base.copy()
    noisy[36:60, 36:60, :3] = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    mask[36:60, 36:60] = 255
    return base, noisy, mask


def test_retouch_con_vecinos_elimina_ruido():
    base, noisy, mask = _noisy_over_base()
    out = retouch_region(noisy, mask, [base.copy(), base.copy()])

    assert out.shape == noisy.shape and out.dtype == np.uint8
    region = np.s_[36:60, 36:60, :3]
    err_before = np.abs(noisy[region].astype(float) - base[region].astype(float)).mean()
    err_after = np.abs(out[region].astype(float) - base[region].astype(float)).mean()
    assert err_after < 0.3 * err_before, f"antes={err_before:.1f} después={err_after:.1f}"
    # Lejos de la región (fuera del alcance del blur) no cambia nada.
    assert np.allclose(out[:20, :20].astype(int), noisy[:20, :20].astype(int), atol=1)


def test_retouch_sin_vecinos_inpaint():
    base, noisy, mask = _noisy_over_base()
    out = retouch_region(noisy, mask, [])

    assert out.shape == noisy.shape and out.dtype == np.uint8
    region = np.s_[36:60, 36:60, :3]
    err_before = np.abs(noisy[region].astype(float) - base[region].astype(float)).mean()
    err_after = np.abs(out[region].astype(float) - base[region].astype(float)).mean()
    assert err_after < err_before  # el inpaint reconstruye desde el entorno
    # Alfa: intacto lejos de la región, desvanecido en el centro de la región.
    assert np.array_equal(out[:20, :20, 3], noisy[:20, :20, 3])
    assert out[48, 48, 3] < 30


def test_retouch_mascara_vacia_no_cambia():
    _, noisy, _ = _noisy_over_base()
    out = retouch_region(noisy, np.zeros((SIZE, SIZE), np.uint8), [])
    assert np.array_equal(out, noisy)


# ------------------------------------------------------------------- edits

def test_ensure_edit_session_copia_ultimo_stage(anim: Paths):
    d1 = ensure_edit_session(anim)
    assert d1 == anim.edit_dir(1)
    assert (d1 / "frameset.json").exists()
    assert png_names(d1) == [frame_filename(i) for i in range(4)]
    # Llamar de nuevo devuelve la misma versión sin crear otra.
    assert ensure_edit_session(anim) == d1
    assert anim.list_edit_versions() == [1]
    # El stage dir original queda intacto.
    stage_dir = anim.stages_dir / "05_smooth"
    assert png_names(stage_dir) == [frame_filename(i) for i in range(4)]
    assert FrameSet.load(stage_dir).stage == "smooth"


def test_ensure_edit_session_sin_stages(data_dir: Path):
    with pytest.raises(FileNotFoundError):
        ensure_edit_session(Paths("p", "sin_stages", data_dir=data_dir))


def test_apply_edit_delete(anim: Paths):
    fs = apply_edit(anim, {"op": "delete", "index": 1})

    d = anim.edit_dir(2)
    assert len(fs.frames) == 3
    assert [f.file for f in fs.frames] == [frame_filename(i) for i in range(3)]
    assert [f.index for f in fs.frames] == [0, 1, 2]
    assert png_names(d) == [frame_filename(i) for i in range(3)]
    # El verde (índice 1) desapareció: 0=rojo, 1=azul, 2=amarillo.
    assert_color(d, "frame_0000.png", COLORS[0])
    assert_color(d, "frame_0001.png", COLORS[2])
    assert_color(d, "frame_0002.png", COLORS[3])
    # Manifiesto persistido, con historial de edición y sin tocar la versión previa.
    saved = FrameSet.load(d)
    assert saved.stage == "edit"
    assert saved.history[-1] == fs.history[-1]
    assert saved.history[-1]["stage"] == "edit"
    assert saved.history[-1]["params"]["op"] == "delete"
    assert png_names(anim.edit_dir(1)) == [frame_filename(i) for i in range(4)]


def test_apply_edit_delete_no_deja_frameset_vacio(anim: Paths):
    for _ in range(3):
        fs = apply_edit(anim, {"op": "delete", "index": 0})
    assert len(fs.frames) == 1
    versions = anim.list_edit_versions()
    with pytest.raises(ValueError):
        apply_edit(anim, {"op": "delete", "index": 0})
    assert anim.list_edit_versions() == versions  # la versión fallida se limpió


def test_apply_edit_duplicate(anim: Paths):
    fs = apply_edit(anim, {"op": "duplicate", "index": 1})

    d = anim.edit_dir(2)
    assert len(fs.frames) == 5
    assert [f.file for f in fs.frames] == [frame_filename(i) for i in range(5)]
    assert png_names(d) == [frame_filename(i) for i in range(5)]
    assert fs.frames[2].source == "duplicated"
    assert fs.frames[1].source != "duplicated"
    # 0=rojo, 1=verde, 2=verde (copia), 3=azul, 4=amarillo.
    assert_color(d, "frame_0001.png", COLORS[1])
    assert_color(d, "frame_0002.png", COLORS[1])
    assert_color(d, "frame_0003.png", COLORS[2])


def test_apply_edit_reorder(anim: Paths):
    fs = apply_edit(anim, {"op": "reorder", "order": [3, 2, 1, 0]})

    d = anim.edit_dir(2)
    assert [f.file for f in fs.frames] == [frame_filename(i) for i in range(4)]
    assert [f.index for f in fs.frames] == [0, 1, 2, 3]
    assert png_names(d) == [frame_filename(i) for i in range(4)]
    # Orden invertido: 0=amarillo ... 3=rojo.
    assert_color(d, "frame_0000.png", COLORS[3])
    assert_color(d, "frame_0001.png", COLORS[2])
    assert_color(d, "frame_0002.png", COLORS[1])
    assert_color(d, "frame_0003.png", COLORS[0])


def test_apply_edit_reorder_invalido_no_crea_version(anim: Paths):
    ensure_edit_session(anim)
    with pytest.raises(ValueError):
        apply_edit(anim, {"op": "reorder", "order": [0, 0, 1, 2]})
    assert anim.list_edit_versions() == [1]


def test_apply_edit_set_duration(anim: Paths):
    fs = apply_edit(anim, {"op": "set_duration", "index": 2, "duration_ms": 160})
    assert fs.frames[2].duration_ms == 160
    assert [f.duration_ms for f in fs.frames] == [100, 100, 160, 100]
    assert FrameSet.load(anim.edit_dir(2)).frames[2].duration_ms == 160


def test_apply_edit_interpolate(anim: Paths):
    fs = apply_edit(anim, {"op": "interpolate", "after_index": 1})

    d = anim.edit_dir(2)
    assert len(fs.frames) == 5
    assert [f.file for f in fs.frames] == [frame_filename(i) for i in range(5)]
    assert png_names(d) == [frame_filename(i) for i in range(5)]
    assert fs.frames[2].source == "interpolated"
    # El cuadro sintetizado queda entre los centros de los cuadros 1 y 2.
    mid = load_frame_rgba(d, fs.frames[2])
    assert CENTERS[1] < centroid_x(mid) < CENTERS[2]
    assert FrameSet.load(d).frames[2].source == "interpolated"


def test_apply_edit_interpolate_envuelve_al_final(anim: Paths):
    fs = apply_edit(anim, {"op": "interpolate", "after_index": 3})
    assert len(fs.frames) == 5
    assert fs.frames[4].source == "interpolated"
    assert [f.file for f in fs.frames] == [frame_filename(i) for i in range(5)]
    # Interpolado entre el último (cx=66) y el primero (cx=30).
    mid = load_frame_rgba(anim.edit_dir(2), fs.frames[4])
    assert CENTERS[0] < centroid_x(mid) < CENTERS[3]


def test_apply_edit_retouch(anim: Paths):
    before = load_frame_rgba(ensure_edit_session(anim), Frame(index=1, file=frame_filename(1)))
    fs = apply_edit(anim, {"op": "retouch", "index": 1, "mask_png_base64": mask_b64()})

    d = anim.edit_dir(2)
    assert fs.frames[1].source == "retouched"
    after = load_frame_rgba(d, fs.frames[1])
    # La región enmascarada cambió (mezcla de los vecinos rojo/azul sobre el verde).
    assert not np.array_equal(after[44:60, 44:60], before[44:60, 44:60])
    # Fuera del alcance de la máscara no cambia.
    assert np.array_equal(after[:20, :20], before[:20, :20])
    # El historial no arrastra el base64.
    saved = FrameSet.load(d)
    assert saved.history[-1]["params"] == {"op": "retouch", "index": 1}
    assert "mask_png_base64" not in saved.history[-1]["params"]


def test_apply_edit_retouch_mascara_de_otro_tamano(anim: Paths):
    b64 = mask_b64(size=48, box=(0, 0, 47, 47))  # se reescala al tamaño del cuadro
    fs = apply_edit(anim, {"op": "retouch", "index": 0, "mask_png_base64": b64})
    assert fs.frames[0].source == "retouched"
    assert png_names(anim.edit_dir(2)) == [frame_filename(i) for i in range(4)]


def test_apply_edit_op_desconocida(anim: Paths):
    with pytest.raises(ValueError):
        apply_edit(anim, {"op": "explode"})
    assert anim.list_edit_versions() == []  # ni siquiera se creó la sesión


def test_apply_edit_indice_fuera_de_rango(anim: Paths):
    ensure_edit_session(anim)
    with pytest.raises(ValueError):
        apply_edit(anim, {"op": "delete", "index": 9})
    assert anim.list_edit_versions() == [1]


def test_ensure_edit_session_fuente_sin_manifiesto_no_crea_version(data_dir: Path):
    """Con solo un stage dir parcial (sin frameset.json) el error es claro y
    NO queda ninguna versión basura en edits/."""
    paths = Paths("p", "parcial", data_dir=data_dir)
    partial = paths.stages_dir / "03_align"
    partial.mkdir(parents=True)
    save_frame_rgba(partial, frame_filename(0), circle_frame(30))  # PNG sin manifiesto

    with pytest.raises(FileNotFoundError):
        ensure_edit_session(paths)

    assert paths.list_edit_versions() == []
    assert not paths.edits_dir.exists() or not any(paths.edits_dir.iterdir())


def test_ensure_edit_session_salta_stage_parcial(anim: Paths):
    """Un stage dir más avanzado pero parcial se salta: la sesión se crea desde
    la última etapa COMPLETA."""
    partial = anim.stages_dir / "06_extra"
    partial.mkdir(parents=True)
    (partial / "frame_0000.png").write_bytes(b"parcial, sin manifiesto")

    d1 = ensure_edit_session(anim)

    assert d1 == anim.edit_dir(1)
    assert (d1 / "frameset.json").exists()
    assert FrameSet.load(d1).stage == "smooth"  # copiado de 05_smooth, no del parcial
    assert png_names(d1) == [frame_filename(i) for i in range(4)]


def test_apply_edit_concurrente_historial_consistente(anim: Paths):
    """4 hilos aplicando ediciones a la vez sobre la misma animación: sin
    excepciones y con historial secuencial de versiones, todas válidas."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    n_threads = 4
    barrier = threading.Barrier(n_threads)

    def worker(_i: int):
        barrier.wait()
        return apply_edit(anim, {"op": "duplicate", "index": 0})

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        results = list(ex.map(worker, range(n_threads)))  # re-lanza excepciones

    assert len(results) == n_threads
    # Sesión (edit_0001) + una versión por edición, sin huecos ni colisiones.
    assert anim.list_edit_versions() == [1, 2, 3, 4, 5]
    # Todos los manifiestos cargan y cada versión añade exactamente un cuadro.
    counts = [len(FrameSet.load(anim.edit_dir(v)).frames) for v in [1, 2, 3, 4, 5]]
    assert counts == [4, 5, 6, 7, 8]
    # No quedaron dirs temporales ni basura en edits/.
    names = sorted(d.name for d in anim.edits_dir.iterdir())
    assert names == [f"edit_{v:04d}" for v in range(1, 6)]


def test_undo_ignora_versiones_invalidas_y_no_reutiliza_numeros(anim: Paths):
    apply_edit(anim, {"op": "delete", "index": 0})  # crea edit_0001 y edit_0002
    garbage = anim.edits_dir / "edit_0007"
    garbage.mkdir()
    (garbage / "frame_0000.png").write_bytes(b"basura sin manifiesto")

    fs = undo(anim)  # deshace edit_0002; la basura no cuenta como versión
    assert fs is not None and len(fs.frames) == 4
    assert anim.list_edit_versions() == [1]
    assert garbage.exists()

    # La siguiente edición NO colisiona con el número de la basura.
    apply_edit(anim, {"op": "duplicate", "index": 0})
    assert anim.list_edit_versions() == [1, 8]
    assert (anim.edit_dir(8) / "frameset.json").exists()


def test_undo(anim: Paths):
    assert undo(anim) is None  # sin ediciones no hay nada que deshacer
    ensure_edit_session(anim)
    assert undo(anim) is None  # una sola versión: no se borra
    assert anim.list_edit_versions() == [1]

    apply_edit(anim, {"op": "delete", "index": 0})
    apply_edit(anim, {"op": "duplicate", "index": 0})
    assert anim.list_edit_versions() == [1, 2, 3]

    fs = undo(anim)  # vuelve a la versión 2 (tras el delete)
    assert fs is not None and len(fs.frames) == 3
    assert anim.list_edit_versions() == [1, 2]
    assert not anim.edit_dir(3).exists()

    fs = undo(anim)  # vuelve a la versión 1 (copia del stage: 4 cuadros)
    assert fs is not None and len(fs.frames) == 4
    assert anim.list_edit_versions() == [1]

    assert undo(anim) is None
    assert anim.list_edit_versions() == [1]
    assert (anim.edit_dir(1) / "frameset.json").exists()
