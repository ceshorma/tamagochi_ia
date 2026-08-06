"""Tests de los proveedores reales (Runway / Luma).

Todo el tráfico HTTP se simula con ``httpx.MockTransport``: jamás se toca la
red. Se ejercita el ciclo completo crear -> pending -> completed -> descarga,
el fallo reportado por el proveedor y la ausencia de API key.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from sprite_pipeline.config import reset_settings_cache
from sprite_pipeline.providers.base import PROVIDER_REGISTRY, GenerationRequest, build_prompt
from sprite_pipeline.providers.luma import LumaProvider
from sprite_pipeline.providers.runway import ProviderConfigError, RunwayProvider

FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42FAKE-VIDEO-BYTES"
RUNWAY_VIDEO_URL = "https://assets.example.test/runway/task-abc/video.mp4"
LUMA_VIDEO_URL = "https://assets.example.test/luma/gen-123/video.mp4"


def make_request(image: Path) -> GenerationRequest:
    return GenerationRequest(image_path=image, action="idle", duration_s=2.0, fps=10, seed=7)


# ------------------------------------------------------------ transports mock

def runway_transport(state: dict, *, fail: bool = False) -> httpx.MockTransport:
    """Simula la API de Runway: creación, polling con 2 pendings y descarga."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.method == "POST" and url == "https://api.dev.runwayml.com/v1/image_to_video":
            state["create_headers"] = dict(request.headers)
            state["create_json"] = json.loads(request.content.decode())
            return httpx.Response(200, json={"id": "task-abc", "status": "PENDING"})
        if request.method == "GET" and url == "https://api.dev.runwayml.com/v1/tasks/task-abc":
            state["polls"] = state.get("polls", 0) + 1
            if fail:
                return httpx.Response(
                    200,
                    json={"id": "task-abc", "status": "FAILED", "failure": "moderation: prompt rechazado"},
                )
            if state["polls"] < 3:
                return httpx.Response(200, json={"id": "task-abc", "status": "PENDING"})
            return httpx.Response(
                200, json={"id": "task-abc", "status": "SUCCEEDED", "output": [RUNWAY_VIDEO_URL]}
            )
        if request.method == "GET" and url == RUNWAY_VIDEO_URL:
            state["downloaded"] = True
            return httpx.Response(200, content=FAKE_MP4)
        return httpx.Response(404, json={"detail": f"ruta inesperada: {request.method} {url}"})

    return httpx.MockTransport(handler)


def luma_transport(state: dict, *, fail: bool = False) -> httpx.MockTransport:
    """Simula la API Dream Machine de Luma: creación, polling y descarga."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.method == "POST" and url == "https://api.lumalabs.ai/dream-machine/v1/generations":
            state["create_headers"] = dict(request.headers)
            state["create_json"] = json.loads(request.content.decode())
            return httpx.Response(201, json={"id": "gen-123", "state": "queued"})
        if request.method == "GET" and url == "https://api.lumalabs.ai/dream-machine/v1/generations/gen-123":
            state["polls"] = state.get("polls", 0) + 1
            if fail:
                return httpx.Response(
                    200,
                    json={"id": "gen-123", "state": "failed", "failure_reason": "flagged by moderation"},
                )
            if state["polls"] < 3:
                return httpx.Response(200, json={"id": "gen-123", "state": "dreaming"})
            return httpx.Response(
                200,
                json={"id": "gen-123", "state": "completed", "assets": {"video": LUMA_VIDEO_URL}},
            )
        if request.method == "GET" and url == LUMA_VIDEO_URL:
            state["downloaded"] = True
            return httpx.Response(200, content=FAKE_MP4)
        return httpx.Response(404, json={"detail": f"ruta inesperada: {request.method} {url}"})

    return httpx.MockTransport(handler)


def no_traffic_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no debería haber tráfico HTTP sin API key: {request.url}")

    return httpx.MockTransport(handler)


# ------------------------------------------------------------------- registro

def test_both_providers_registered():
    assert PROVIDER_REGISTRY["runway"] is RunwayProvider
    assert PROVIDER_REGISTRY["luma"] is LumaProvider


# --------------------------------------------------------------------- runway

def test_runway_full_cycle(sample_image: Path, tmp_path: Path):
    state: dict = {}
    provider = RunwayProvider(api_key="rw-key", poll_interval=0.0, transport=runway_transport(state))
    req = make_request(sample_image)
    workdir = tmp_path / "work"

    result = provider.generate(req, workdir)

    assert result.provider == "runway"
    assert result.prompt == build_prompt(req)
    assert result.video_path == workdir / "video.mp4"
    assert result.video_path.read_bytes() == FAKE_MP4
    assert result.raw.get("status") == "SUCCEEDED"

    # Petición de creación: modelo, prompt enriquecido, imagen como data URI.
    body = state["create_json"]
    assert body["model"] == "gen4_turbo"
    assert body["promptText"] == build_prompt(req)
    assert body["promptImage"].startswith("data:image/png;base64,")
    assert body["duration"] in (5, 10)
    assert "ratio" in body
    # Cabeceras obligatorias (httpx normaliza a minúsculas).
    assert state["create_headers"]["x-runway-version"] == "2024-11-06"
    assert state["create_headers"]["authorization"] == "Bearer rw-key"
    # Hubo polling real (pending -> pending -> succeeded) y descarga.
    assert state["polls"] == 3
    assert state.get("downloaded") is True


def test_runway_failed_task_raises_with_provider_message(sample_image: Path, tmp_path: Path):
    state: dict = {}
    provider = RunwayProvider(api_key="rw-key", poll_interval=0.0, transport=runway_transport(state, fail=True))
    with pytest.raises(RuntimeError, match="moderation: prompt rechazado"):
        provider.generate(make_request(sample_image), tmp_path / "work")
    assert not (tmp_path / "work" / "video.mp4").exists()


def test_runway_without_api_key_raises_config_error(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, sample_image: Path, tmp_path: Path
):
    monkeypatch.delenv("RUNWAY_API_KEY", raising=False)
    reset_settings_cache()
    provider = RunwayProvider(transport=no_traffic_transport())
    assert isinstance(ProviderConfigError("x"), RuntimeError)
    with pytest.raises(ProviderConfigError):
        provider.generate(make_request(sample_image), tmp_path / "work")


# ----------------------------------------------------------------------- luma

def test_luma_full_cycle(sample_image: Path, tmp_path: Path):
    state: dict = {}
    provider = LumaProvider(api_key="luma-key", poll_interval=0.0, transport=luma_transport(state))
    req = make_request(sample_image)
    workdir = tmp_path / "work"

    result = provider.generate(req, workdir)

    assert result.provider == "luma"
    assert result.prompt == build_prompt(req)
    assert result.video_path == workdir / "video.mp4"
    assert result.video_path.read_bytes() == FAKE_MP4
    assert result.raw.get("state") == "completed"

    body = state["create_json"]
    assert body["model"] == "ray-2"
    assert body["prompt"] == build_prompt(req)
    assert body["keyframes"]["frame0"]["type"] == "image"
    assert body["keyframes"]["frame0"]["url"].startswith("data:image/png;base64,")
    assert state["create_headers"]["authorization"] == "Bearer luma-key"
    assert state["polls"] == 3
    assert state.get("downloaded") is True


def test_luma_failed_generation_raises_with_provider_message(sample_image: Path, tmp_path: Path):
    state: dict = {}
    provider = LumaProvider(api_key="luma-key", poll_interval=0.0, transport=luma_transport(state, fail=True))
    with pytest.raises(RuntimeError, match="flagged by moderation"):
        provider.generate(make_request(sample_image), tmp_path / "work")
    assert not (tmp_path / "work" / "video.mp4").exists()


def test_luma_without_api_key_raises_config_error(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, sample_image: Path, tmp_path: Path
):
    monkeypatch.delenv("LUMA_API_KEY", raising=False)
    reset_settings_cache()
    provider = LumaProvider(transport=no_traffic_transport())
    with pytest.raises(ProviderConfigError):
        provider.generate(make_request(sample_image), tmp_path / "work")
