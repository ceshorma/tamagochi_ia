"""Proveedor real Runway (image-to-video, modelo ``gen4_turbo``).

Flujo bloqueante: submit HTTP -> polling hasta estado terminal -> descarga del
mp4 a ``workdir/video.mp4``. Todo el tráfico pasa por un ``httpx.Client``
construido con el ``transport`` inyectado en el constructor, lo que permite
testear el ciclo completo con ``httpx.MockTransport`` sin tocar la red.

ADVERTENCIA — llamadas reales NO validadas: este entorno de desarrollo no
tiene claves de Runway ni acceso a la red, así que las peticiones contra el
servicio real NUNCA fueron ejecutadas. Las rutas, cabeceras y formas de
payload/respuesta siguen la documentación pública de la API de Runway
(versión ``2024-11-06``) a fecha de escritura; verificar contra el servicio
antes de usar en producción.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import httpx

from sprite_pipeline.config import get_settings
from sprite_pipeline.providers.base import (
    GenerationRequest,
    GenerationResult,
    VideoGenerationProvider,
    build_prompt,
    register_provider,
)

RUNWAY_API_BASE = "https://api.dev.runwayml.com/v1"
RUNWAY_VERSION = "2024-11-06"

#: Estados terminales según la documentación pública de la API de tareas.
_TERMINAL_OK = {"SUCCEEDED"}
_TERMINAL_ERR = {"FAILED", "CANCELLED"}


class ProviderConfigError(RuntimeError):
    """Configuración de proveedor incompleta (p. ej. falta la API key)."""


_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def image_to_data_uri(path: Path) -> str:
    """Codifica una imagen de disco como data URI base64 (``data:<mime>;base64,...``)."""
    path = Path(path)
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


@register_provider
class RunwayProvider(VideoGenerationProvider):
    """Adaptador de la API image-to-video de Runway.

    Las llamadas reales al servicio NO están validadas en este entorno (sin
    claves ni red); los tests cubren el flujo solo con ``httpx.MockTransport``.
    """

    name = "runway"

    def __init__(
        self,
        api_key: str | None = None,
        poll_interval: float = 2.0,
        max_polls: int = 150,
        transport: httpx.BaseTransport | None = None,
    ):
        if api_key is None:
            api_key = get_settings().runway_api_key
        self.api_key = api_key or ""
        self.poll_interval = poll_interval
        self.max_polls = max_polls
        self.transport = transport

    # ------------------------------------------------------------- helpers
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-Runway-Version": RUNWAY_VERSION,
        }

    @staticmethod
    def _duration(req: GenerationRequest) -> int:
        """gen4_turbo solo acepta duraciones de 5 o 10 segundos."""
        return 5 if req.duration_s <= 5 else 10

    @staticmethod
    def _ratio(req: GenerationRequest) -> str:
        """Ratio soportado por gen4_turbo más cercano a la petición."""
        if req.size:
            w, h = req.size
            if w > h:
                return "1280:720"
            if h > w:
                return "720:1280"
        return "960:960"

    # ------------------------------------------------------------ generate
    def generate(self, req: GenerationRequest, workdir: Path) -> GenerationResult:
        if not self.api_key:
            raise ProviderConfigError(
                "Falta la API key de Runway: pásala al constructor o define RUNWAY_API_KEY."
            )
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        prompt = build_prompt(req)
        payload = {
            "model": "gen4_turbo",
            "promptImage": image_to_data_uri(req.image_path),
            "promptText": prompt,
            "duration": self._duration(req),
            "ratio": self._ratio(req),
        }
        if req.seed is not None:
            payload["seed"] = int(req.seed)

        with httpx.Client(transport=self.transport, timeout=60.0) as client:
            resp = client.post(
                f"{RUNWAY_API_BASE}/image_to_video", json=payload, headers=self._headers()
            )
            resp.raise_for_status()
            task_id = resp.json()["id"]

            task = self._poll_until_done(client, task_id)

            output = task.get("output") or []
            if not output:
                raise RuntimeError(f"Runway: la tarea {task_id} terminó sin URL de video: {task}")
            video_path = workdir / "video.mp4"
            download = client.get(output[0])
            download.raise_for_status()
            video_path.write_bytes(download.content)

        return GenerationResult(video_path=video_path, provider=self.name, prompt=prompt, raw=task)

    def _poll_until_done(self, client: httpx.Client, task_id: str) -> dict:
        url = f"{RUNWAY_API_BASE}/tasks/{task_id}"
        for _ in range(self.max_polls):
            resp = client.get(url, headers=self._headers())
            resp.raise_for_status()
            task = resp.json()
            status = str(task.get("status", "")).upper()
            if status in _TERMINAL_OK:
                return task
            if status in _TERMINAL_ERR:
                detail = task.get("failure") or task.get("failureCode") or "error desconocido"
                raise RuntimeError(f"Runway: la tarea {task_id} terminó en {status}: {detail}")
            time.sleep(self.poll_interval)
        raise RuntimeError(
            f"Runway: la tarea {task_id} no llegó a estado terminal tras {self.max_polls} sondeos"
        )
