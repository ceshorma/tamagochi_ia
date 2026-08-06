"""Proveedor real Luma Dream Machine (modelo ``ray-2``).

Flujo bloqueante: submit HTTP -> polling hasta estado terminal -> descarga del
mp4 a ``workdir/video.mp4``. Todo el tráfico pasa por un ``httpx.Client``
construido con el ``transport`` inyectado en el constructor, lo que permite
testear el ciclo completo con ``httpx.MockTransport`` sin tocar la red.

ADVERTENCIA — llamadas reales NO validadas: este entorno de desarrollo no
tiene claves de Luma ni acceso a la red, así que las peticiones contra el
servicio real NUNCA fueron ejecutadas. Las rutas y formas de payload/respuesta
siguen la documentación pública de la API Dream Machine a fecha de escritura;
verificar contra el servicio antes de usar en producción.
"""

from __future__ import annotations

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
from sprite_pipeline.providers.runway import ProviderConfigError, image_to_data_uri

LUMA_API_BASE = "https://api.lumalabs.ai/dream-machine/v1"

#: Estados terminales según la documentación pública de Dream Machine.
_TERMINAL_OK = {"completed"}
_TERMINAL_ERR = {"failed"}


@register_provider
class LumaProvider(VideoGenerationProvider):
    """Adaptador de la API Dream Machine de Luma (imagen -> video).

    Las llamadas reales al servicio NO están validadas en este entorno (sin
    claves ni red); los tests cubren el flujo solo con ``httpx.MockTransport``.
    """

    name = "luma"

    def __init__(
        self,
        api_key: str | None = None,
        poll_interval: float = 2.0,
        max_polls: int = 150,
        transport: httpx.BaseTransport | None = None,
    ):
        if api_key is None:
            api_key = get_settings().luma_api_key
        self.api_key = api_key or ""
        self.poll_interval = poll_interval
        self.max_polls = max_polls
        self.transport = transport

    # ------------------------------------------------------------- helpers
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    # ------------------------------------------------------------ generate
    def generate(self, req: GenerationRequest, workdir: Path) -> GenerationResult:
        if not self.api_key:
            raise ProviderConfigError(
                "Falta la API key de Luma: pásala al constructor o define LUMA_API_KEY."
            )
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        prompt = build_prompt(req)
        payload = {
            "model": "ray-2",
            "prompt": prompt,
            "keyframes": {
                "frame0": {"type": "image", "url": image_to_data_uri(req.image_path)},
            },
        }

        with httpx.Client(transport=self.transport, timeout=60.0) as client:
            resp = client.post(
                f"{LUMA_API_BASE}/generations", json=payload, headers=self._headers()
            )
            resp.raise_for_status()
            generation_id = resp.json()["id"]

            generation = self._poll_until_done(client, generation_id)

            video_url = (generation.get("assets") or {}).get("video")
            if not video_url:
                raise RuntimeError(
                    f"Luma: la generación {generation_id} terminó sin URL de video: {generation}"
                )
            video_path = workdir / "video.mp4"
            download = client.get(video_url)
            download.raise_for_status()
            video_path.write_bytes(download.content)

        return GenerationResult(
            video_path=video_path, provider=self.name, prompt=prompt, raw=generation
        )

    def _poll_until_done(self, client: httpx.Client, generation_id: str) -> dict:
        url = f"{LUMA_API_BASE}/generations/{generation_id}"
        for _ in range(self.max_polls):
            resp = client.get(url, headers=self._headers())
            resp.raise_for_status()
            generation = resp.json()
            state = str(generation.get("state", "")).lower()
            if state in _TERMINAL_OK:
                return generation
            if state in _TERMINAL_ERR:
                detail = generation.get("failure_reason") or "error desconocido"
                raise RuntimeError(
                    f"Luma: la generación {generation_id} terminó en {state}: {detail}"
                )
            time.sleep(self.poll_interval)
        raise RuntimeError(
            f"Luma: la generación {generation_id} no llegó a estado terminal tras "
            f"{self.max_polls} sondeos"
        )
