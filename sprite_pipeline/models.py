"""Modelos núcleo del pipeline: Frame y FrameSet con su manifiesto JSON.

Convenciones que TODOS los módulos deben respetar:

- Un *FrameSet* vive en un directorio: PNGs de los cuadros + ``frameset.json``.
- Los archivos de cuadro se llaman ``frame_0000.png``, ``frame_0001.png``, ...
  y se renumeran al guardar según el orden de la lista ``frames``.
- Las imágenes en memoria son arrays numpy **RGBA uint8** (alto, ancho, 4).
  La conversión desde/hacia disco se hace con los helpers de
  ``sprite_pipeline.stages.base`` (via PIL), nunca con cv2.imread directo
  (que devuelve BGR y rompería los canales).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = "1.0"
MANIFEST_NAME = "frameset.json"
FRAME_PATTERN = "frame_{:04d}.png"


@dataclass
class Frame:
    index: int
    file: str
    duration_ms: int = 100
    flags: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    transform: dict | None = None
    source: str = "generated"  # generated|interpolated|duplicated|retouched

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Frame":
        return cls(
            index=d["index"],
            file=d["file"],
            duration_ms=d.get("duration_ms", 100),
            flags=list(d.get("flags", [])),
            scores=dict(d.get("scores", {})),
            transform=d.get("transform"),
            source=d.get("source", "generated"),
        )


@dataclass
class FrameSet:
    stage: str  # etiqueta de la última etapa aplicada: raw|preprocess|background|align|anomaly|smooth|edit
    frames: list[Frame]
    meta: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    # ------------------------------------------------------------------ I/O
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "meta": self.meta,
            "history": self.history,
            "frames": [f.to_dict() for f in self.frames],
        }
        (directory / MANIFEST_NAME).write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, directory: Path) -> "FrameSet":
        directory = Path(directory)
        payload = json.loads((directory / MANIFEST_NAME).read_text())
        return cls(
            stage=payload["stage"],
            frames=[Frame.from_dict(f) for f in payload["frames"]],
            meta=payload.get("meta", {}),
            history=payload.get("history", []),
            schema_version=payload.get("schema_version", SCHEMA_VERSION),
        )

    # ------------------------------------------------------------- helpers
    def with_stage(self, stage: str, params: dict | None = None) -> "FrameSet":
        """Copia superficial con etiqueta de etapa nueva y entrada de historial."""
        entry = {"stage": stage, "params": params or {}, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        return FrameSet(
            stage=stage,
            frames=self.frames,
            meta=dict(self.meta),
            history=[*self.history, entry],
            schema_version=self.schema_version,
        )

    def reindex(self) -> None:
        """Renumera ``index`` según el orden actual de la lista (los archivos no se renombran)."""
        for i, f in enumerate(self.frames):
            f.index = i

    def total_duration_ms(self) -> int:
        return sum(f.duration_ms for f in self.frames)

    def flagged(self) -> list[Frame]:
        return [f for f in self.frames if f.flags]


def frame_filename(i: int) -> str:
    return FRAME_PATTERN.format(i)
