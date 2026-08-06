"""Proveedor mock: video sintético determinista para desarrollo y tests.

Anima la criatura de :mod:`sprite_pipeline.sample` cuando la acción tiene una
pose definida; para cualquier otra acción aplica transformaciones genéricas
(bob/squash) a la imagen de entrada. El video simula a propósito los defectos
que el pipeline corrige después:

- fondo casi uniforme gris claro con gradiente vertical suave y ruido leve,
- deriva de posición (±2–4 px) y de escala (±3 %) entre cuadros, determinista
  a partir de la semilla (``numpy.random.RandomState(seed or 0)``),
- con ``"inject_anomaly"`` en ``req.prompt_extra``, exactamente un cuadro
  intermedio corrupto (ruido fuerte + shift de canal).

Todo es determinista: misma request (misma semilla) -> mismo video.
"""

from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from sprite_pipeline.providers.base import (
    GenerationRequest,
    GenerationResult,
    VideoGenerationProvider,
    build_prompt,
    register_provider,
)
from sprite_pipeline.sample import creature_pose, draw_creature

# Acciones con pose definida en sample.creature_pose (rama explícita).
POSE_ACTIONS = frozenset({"idle", "reposo", "eat", "comer", "sleep", "dormir", "jump", "saltar"})

# Fondo: gris claro con gradiente vertical suave y ruido leve.
_BG_TOP = 233.0
_BG_BOTTOM = 214.0
_BG_NOISE_STD = 2.0

# Deriva por cuadro: posición ±2–4 px, escala ±3 %.
_DRIFT_PX_MIN = 2.0
_DRIFT_PX_MAX = 4.0
_DRIFT_SCALE = 0.03


def _round16(value: int, minimum: int = 64) -> int:
    """Redondea hacia arriba a múltiplo de 16 (dimensiones amigables para h264)."""
    value = max(int(minimum), int(value))
    return int(math.ceil(value / 16) * 16)


def _canvas_size(req: GenerationRequest) -> tuple[int, int]:
    if req.size:
        return _round16(req.size[0]), _round16(req.size[1])
    try:
        with Image.open(req.image_path) as im:
            side = max(im.size)
    except Exception:
        side = 256
    side = min(max(side, 64), 512)
    side = _round16(side)
    return side, side


def _background(w: int, h: int, rng: np.random.RandomState) -> np.ndarray:
    """Fondo RGB uint8 (h, w, 3): gris claro, gradiente vertical + ruido leve."""
    grad = np.linspace(_BG_TOP, _BG_BOTTOM, h, dtype=np.float32)[:, None]
    base = np.broadcast_to(grad, (h, w)).copy()
    base += rng.normal(0.0, _BG_NOISE_STD, size=(h, w))
    gray = np.clip(base, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _generic_sprite(base: Image.Image, t: float, target_px: int) -> tuple[Image.Image, float]:
    """Bob/squash genéricos sobre una imagen arbitraria. Devuelve (sprite, bob)."""
    w = 2.0 * math.pi * t
    bob = math.sin(w)
    squash = 0.2 * math.sin(2.0 * w)
    scale = target_px / max(base.size)
    new_w = max(1, int(round(base.width * scale * (1.0 + 0.10 * squash))))
    new_h = max(1, int(round(base.height * scale * (1.0 - 0.10 * squash))))
    return base.resize((new_w, new_h), Image.LANCZOS), bob


def _compose(bg_rgb: np.ndarray, sprite: Image.Image, cx: float, cy: float) -> np.ndarray:
    """Pega el sprite RGBA (con recorte) sobre el fondo; devuelve RGB uint8."""
    canvas = Image.fromarray(bg_rgb, mode="RGB")
    x = int(round(cx - sprite.width / 2.0))
    y = int(round(cy - sprite.height / 2.0))
    canvas.paste(sprite, (x, y), sprite)
    return np.asarray(canvas, dtype=np.uint8).copy()


def _corrupt(frame: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Corrupción fuerte: ruido intenso (fino + en bloques) y shift de canal.

    El ruido en bloques (baja frecuencia) sobrevive a la compresión del mp4,
    para que el cuadro corrupto siga siendo un outlier claro tras recodificar.
    """
    h, w = frame.shape[:2]
    blocky = rng.normal(0.0, 70.0, size=((h + 7) // 8, (w + 7) // 8, 3))
    blocky = np.kron(blocky, np.ones((8, 8, 1)))[:h, :w, :]
    fine = rng.normal(0.0, 40.0, size=frame.shape)
    noisy = frame.astype(np.float32) + blocky + fine
    out = np.clip(noisy, 0, 255).astype(np.uint8)
    out = np.roll(out, 1, axis=2)  # cicla los canales RGB
    out[:, :, 0] = np.roll(out[:, :, 0], max(4, w // 6), axis=1)  # desplaza un canal
    return out


@register_provider
class MockProvider(VideoGenerationProvider):
    """Proveedor local sin red: escribe ``workdir/video.mp4`` con imageio/ffmpeg."""

    name = "mock"

    def generate(self, req: GenerationRequest, workdir: Path) -> GenerationResult:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        w, h = _canvas_size(req)
        n_frames = max(1, int(round(req.duration_s * req.fps)))
        rng = np.random.RandomState(req.seed or 0)

        action = (req.action or "").strip().lower()
        use_creature = action in POSE_ACTIONS
        base_img: Image.Image | None = None
        if not use_creature:
            base_img = Image.open(req.image_path).convert("RGBA")

        sprite_px = max(8, int(min(w, h) * 0.78))
        frames: list[np.ndarray] = []
        for i in range(n_frames):
            t = i / n_frames  # ciclo 0..1 (bucle perfecto)
            bg = _background(w, h, rng)
            # Deriva determinista de posición (±2–4 px) y escala (±3 %).
            dx = float(rng.choice([-1.0, 1.0]) * rng.uniform(_DRIFT_PX_MIN, _DRIFT_PX_MAX))
            dy = float(rng.choice([-1.0, 1.0]) * rng.uniform(_DRIFT_PX_MIN, _DRIFT_PX_MAX))
            scale = 1.0 + float(rng.uniform(-_DRIFT_SCALE, _DRIFT_SCALE))
            px = max(8, int(round(sprite_px * scale)))
            if use_creature:
                sprite = draw_creature(size=px, **creature_pose(action, t))
                bob_offset = 0.0  # draw_creature ya aplica el bob internamente
            else:
                sprite, bob = _generic_sprite(base_img, t, px)
                bob_offset = bob * 0.05 * h
            cx = w / 2.0 + dx
            cy = h * 0.55 + dy - bob_offset
            frames.append(_compose(bg, sprite, cx, cy))

        anomaly_index: int | None = None
        if "inject_anomaly" in (req.prompt_extra or "") and n_frames >= 3:
            anomaly_index = n_frames // 2  # siempre un cuadro intermedio
            frames[anomaly_index] = _corrupt(frames[anomaly_index], rng)

        video_path = workdir / "video.mp4"
        writer = imageio.get_writer(str(video_path), fps=req.fps, macro_block_size=1)
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()

        return GenerationResult(
            video_path=video_path,
            provider=self.name,
            prompt=build_prompt(req),
            raw={
                "frames": n_frames,
                "size": [w, h],
                "fps": req.fps,
                "seed": req.seed or 0,
                "mode": "creature" if use_creature else "generic",
                "anomaly_index": anomaly_index,
            },
        )
