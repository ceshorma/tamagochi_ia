"""Tests end-to-end del orquestador y del CLI (proveedor mock, sin red)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from sprite_pipeline import cli
from sprite_pipeline.config import reset_settings_cache
from sprite_pipeline.models import FrameSet
from sprite_pipeline.orchestrator import (
    PIPELINE_ORDER,
    get_orchestrator,
    run_animation_pipeline,
)
from sprite_pipeline.stages.base import load_frame_rgba
from sprite_pipeline.storage import Paths, Store

FAST_PARAMS = {"fps": 6, "duration_s": 1.0, "work_size": 128}


def _make_animation(store: Store, provider: str = "mock") -> tuple[dict, dict]:
    project = store.create_project("test")
    anim = store.create_animation(project["id"], "idle", "idle", provider)
    return project, anim


# --------------------------------------------------------- run_animation_pipeline

def test_run_animation_pipeline_ready(data_dir: Path, sample_image: Path):
    store = Store()
    project, anim = _make_animation(store)
    params = dict(FAST_PARAMS, seed=1, raise_errors=True)

    result = run_animation_pipeline(store, anim["id"], sample_image, params)

    assert result["status"] == "ready"
    assert result["current_stage"] == "smooth"
    assert not result["error"]

    paths = Paths(project["id"], anim["id"])
    assert paths.source_image.exists()
    assert paths.raw_video.exists()

    # Directorios 00_raw .. 05_smooth, cada uno con su manifiesto.
    for i, name in enumerate(["raw", *PIPELINE_ORDER]):
        stage_dir = paths.stage_dir(i, name)
        assert (stage_dir / "frameset.json").exists(), f"falta {stage_dir}"

    final_dir = paths.stage_dir(len(PIPELINE_ORDER), "smooth")
    fs = FrameSet.load(final_dir)
    assert fs.stage == "smooth"
    assert fs.frames
    assert fs.meta.get("source_image") == str(paths.source_image)

    # Cuadro final: RGBA al work_size, con alfa real (esquinas transparentes).
    rgba = load_frame_rgba(final_dir, fs.frames[0])
    assert rgba.shape == (128, 128, 4)
    assert rgba.dtype.name == "uint8"
    for corner in (rgba[0, 0], rgba[0, -1], rgba[-1, 0], rgba[-1, -1]):
        assert corner[3] == 0
    assert int(rgba[..., 3].max()) == 255  # el sujeto sigue opaco


def test_run_animation_pipeline_provider_failure(data_dir: Path, sample_image: Path):
    store = Store()
    _, anim = _make_animation(store, provider="no-such-provider")

    result = run_animation_pipeline(store, anim["id"], sample_image, dict(FAST_PARAMS))

    assert result["status"] == "failed"
    assert result["error"]
    assert "no-such-provider" in result["error"]


def test_run_animation_pipeline_failure_reraises(data_dir: Path, sample_image: Path):
    store = Store()
    _, anim = _make_animation(store, provider="no-such-provider")
    with pytest.raises(KeyError):
        run_animation_pipeline(
            store, anim["id"], sample_image, dict(FAST_PARAMS, raise_errors=True)
        )
    assert store.get_animation(anim["id"])["status"] == "failed"


# ----------------------------------------------------------------- Orchestrator

def test_submit_generation_job_succeeds(data_dir: Path, sample_image: Path):
    store = Store()
    _, anim = _make_animation(store)
    orch = get_orchestrator(store)
    assert get_orchestrator(store) is orch  # singleton por store

    job = orch.submit_generation(
        anim["id"], sample_image, dict(FAST_PARAMS, seed=3, work_size=96)
    )
    assert job["kind"] == "pipeline"
    assert job["animation_id"] == anim["id"]

    deadline = time.monotonic() + 60
    current = job
    while time.monotonic() < deadline:
        current = store.get_job(job["id"])
        if current["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.2)

    assert current["status"] == "succeeded", f"job: {current}"
    assert current["progress"] == pytest.approx(1.0)
    assert store.get_animation(anim["id"])["status"] == "ready"


# ------------------------------------------------------------------------ CLI

def test_cli_generate_and_export(
    data_dir: Path,
    sample_image: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    monkeypatch.setenv("SPRITE_WORK_SIZE", "128")
    reset_settings_cache()

    rc = cli.main(
        [
            "generate",
            "--image", str(sample_image),
            "--action", "idle",
            "--provider", "mock",
            "--fps", "6",
            "--duration-s", "1.0",
            "--seed", "2",
            "--data-dir", str(data_dir),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    lines = {
        line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
        for line in out.splitlines()
        if ":" in line
    }
    aid = lines["animation"]
    assert lines["status"] == "ready"
    assert Path(lines["frames"]).name == "05_smooth"

    store = Store()
    anim = store.get_animation(aid)
    assert anim is not None and anim["status"] == "ready"

    rc = cli.main(
        [
            "export",
            "--animation", aid,
            "--formats", "sheet,gif",
            "--data-dir", str(data_dir),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "export:" in out

    paths = Paths(anim["project_id"], aid)
    exports = [d for d in paths.exports_dir.iterdir() if d.is_dir()]
    assert len(exports) == 1
    export_dir = exports[0]
    assert (export_dir / "sheet.png").exists()
    assert (export_dir / "sheet.json").exists()
    assert (export_dir / "preview.gif").exists()
    assert not (export_dir / "frames").exists()  # pngseq no pedido


def test_cli_serve_parser_only():
    args = cli.build_parser().parse_args(["serve", "--port", "9001"])
    assert args.func is cli.cmd_serve
    assert args.host == "127.0.0.1"
    assert args.port == 9001
