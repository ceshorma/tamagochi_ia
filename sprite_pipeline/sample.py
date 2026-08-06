"""Criatura de ejemplo del tamagotchi, dibujada de forma paramétrica con PIL.

La usan el proveedor mock (para animar poses), los tests y el script de demo.
Todo es determinista: misma semilla/parámetros -> misma imagen.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

BODY = (120, 200, 130, 255)      # verde suave
BODY_DARK = (90, 165, 100, 255)
BELLY = (215, 240, 205, 255)
EYE = (40, 40, 45, 255)
CHEEK = (250, 160, 150, 255)


def draw_creature(
    size: int = 220,
    bob: float = 0.0,        # -1..1 desplazamiento vertical (rebote)
    squash: float = 0.0,     # -1..1 aplastar(-)/estirar(+) el cuerpo
    eye_open: float = 1.0,   # 0 cerrado .. 1 abierto
    mouth_open: float = 0.0, # 0 cerrado .. 1 abierto (comer)
    lean: float = 0.0,       # -1..1 inclinación lateral en grados (dormir)
) -> Image.Image:
    """Devuelve la criatura RGBA centrada sobre fondo transparente."""
    canvas = size * 2  # dibujar a 2x y reducir = bordes suaves
    img = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    cx = canvas / 2
    body_w = canvas * 0.62 * (1 + 0.10 * squash)
    body_h = canvas * 0.66 * (1 - 0.10 * squash)
    ground = canvas * 0.92
    cy = ground - body_h / 2 - bob * canvas * 0.04

    # pies
    foot_r = canvas * 0.09
    for sx in (-1, 1):
        fx = cx + sx * body_w * 0.28
        d.ellipse([fx - foot_r, ground - foot_r * 1.2, fx + foot_r, ground + foot_r * 0.4], fill=BODY_DARK)

    # cuerpo
    d.ellipse([cx - body_w / 2, cy - body_h / 2, cx + body_w / 2, cy + body_h / 2], fill=BODY)
    # panza
    belly_w, belly_h = body_w * 0.55, body_h * 0.45
    d.ellipse(
        [cx - belly_w / 2, cy + body_h * 0.08 - belly_h / 2, cx + belly_w / 2, cy + body_h * 0.08 + belly_h / 2],
        fill=BELLY,
    )

    # antena
    ax = cx + lean * canvas * 0.05
    a_base_y = cy - body_h / 2
    d.line([cx, a_base_y, ax, a_base_y - canvas * 0.10], fill=BODY_DARK, width=int(canvas * 0.025))
    d.ellipse(
        [ax - canvas * 0.035, a_base_y - canvas * 0.135, ax + canvas * 0.035, a_base_y - canvas * 0.065],
        fill=CHEEK,
    )

    # ojos
    eye_y = cy - body_h * 0.12
    eye_h = max(canvas * 0.055 * eye_open, canvas * 0.008)
    eye_w = canvas * 0.05
    for sx in (-1, 1):
        ex = cx + sx * body_w * 0.18
        if eye_open > 0.15:
            d.ellipse([ex - eye_w, eye_y - eye_h, ex + eye_w, eye_y + eye_h], fill=EYE)
        else:  # ojo cerrado: arco
            d.arc([ex - eye_w, eye_y - eye_w, ex + eye_w, eye_y + eye_w], 20, 160, fill=EYE, width=int(canvas * 0.015))

    # cachetes
    cheek_r = canvas * 0.035
    for sx in (-1, 1):
        chx = cx + sx * body_w * 0.30
        d.ellipse([chx - cheek_r, eye_y + canvas * 0.05 - cheek_r, chx + cheek_r, eye_y + canvas * 0.05 + cheek_r], fill=CHEEK)

    # boca
    mouth_y = cy + body_h * 0.10
    mw = canvas * 0.06
    if mouth_open > 0.1:
        mh = canvas * 0.05 * mouth_open
        d.ellipse([cx - mw, mouth_y - mh, cx + mw, mouth_y + mh], fill=(120, 60, 60, 255))
    else:
        d.arc([cx - mw, mouth_y - mw, cx + mw, mouth_y + mw], 20, 160, fill=EYE, width=int(canvas * 0.015))

    if lean:
        img = img.rotate(lean * 8, resample=Image.BICUBIC, center=(cx, cy))

    return img.resize((size, size), Image.LANCZOS)


def creature_pose(action: str, t: float) -> dict:
    """Parámetros de pose para una acción en el instante t (0..1 del ciclo)."""
    w = 2 * math.pi * t
    if action in ("idle", "reposo"):
        return {"bob": math.sin(w), "squash": 0.15 * math.sin(w), "eye_open": 0.0 if 0.46 < t < 0.54 else 1.0}
    if action in ("eat", "comer"):
        return {"bob": 0.3 * math.sin(2 * w), "squash": 0.25 * math.sin(2 * w), "mouth_open": max(0.0, math.sin(2 * w))}
    if action in ("sleep", "dormir"):
        return {"bob": 0.4 * math.sin(w * 0.5), "squash": -0.3, "eye_open": 0.0, "lean": 0.6}
    if action in ("jump", "saltar"):
        arc = math.sin(w)
        return {"bob": 3.0 * max(0.0, arc), "squash": 0.5 * arc if arc > 0 else -0.4, "eye_open": 1.0}
    # default: idle suave
    return {"bob": 0.5 * math.sin(w), "eye_open": 1.0}


def save_sample_image(path: Path, size: int = 220) -> Path:
    """Imagen de entrada de referencia (pose neutra) para alimentar el pipeline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    draw_creature(size=size).save(path)
    return path
