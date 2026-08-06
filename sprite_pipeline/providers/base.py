"""Interfaz adaptadora de proveedores de generación de video (imagen -> video).

El resto del pipeline SOLO conoce esta interfaz. Cambiar de proveedor es
configuración (``SPRITE_PROVIDER``), nunca código de otros módulos.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GenerationRequest:
    image_path: Path
    action: str  # descripción de la acción: "idle", "walk right", "eat", ...
    prompt_extra: str = ""
    seed: int | None = None
    duration_s: float = 2.0
    fps: int = 12
    size: tuple[int, int] | None = None  # (ancho, alto) sugerido del video


@dataclass
class GenerationResult:
    video_path: Path
    provider: str
    prompt: str
    raw: dict = field(default_factory=dict)  # metadata cruda del proveedor


class VideoGenerationProvider(ABC):
    """Un proveedor genera un video a partir de una imagen + acción.

    ``generate`` es bloqueante: los proveedores reales hacen submit + polling
    internamente. La asincronía vive en el orquestador, no aquí.
    """

    name: str = "base"

    @abstractmethod
    def generate(self, req: GenerationRequest, workdir: Path) -> GenerationResult:
        ...


def build_prompt(req: GenerationRequest) -> str:
    """Prompt enriquecido común: fuerza condiciones que facilitan el pipeline
    (cámara fija, fondo uniforme, personaje completo, animación en bucle)."""
    parts = [
        f"2D game character sprite animation: {req.action}",
        "full body, character centered",
        "fixed camera, no camera movement, no zoom",
        "plain uniform solid light-gray background, no shadows on background",
        "consistent character design and colors in every frame",
        "seamless looping animation",
    ]
    if req.prompt_extra:
        parts.append(req.prompt_extra)
    return ", ".join(parts)


PROVIDER_REGISTRY: dict[str, type[VideoGenerationProvider]] = {}


def register_provider(cls: type[VideoGenerationProvider]) -> type[VideoGenerationProvider]:
    PROVIDER_REGISTRY[cls.name] = cls
    return cls


def get_provider(name: str, **kwargs) -> VideoGenerationProvider:
    try:
        cls = PROVIDER_REGISTRY[name]
    except KeyError:
        raise KeyError(f"Proveedor desconocido: {name!r}. Registrados: {sorted(PROVIDER_REGISTRY)}") from None
    return cls(**kwargs)
