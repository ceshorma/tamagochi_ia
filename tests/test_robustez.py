"""Tests de robustez: dirs de etapa parciales, atomicidad de ``run_stage`` y
reconciliación de jobs huérfanos tras un reinicio."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from sprite_pipeline.models import Frame, FrameSet, frame_filename
from sprite_pipeline.orchestrator import STALE_JOB_ERROR, reconcile_stale_jobs
from sprite_pipeline.stages.base import Stage, run_stage, save_frame_rgba
from sprite_pipeline.storage import Paths, Store


# ------------------------------------------------------------------- helpers

def _make_frameset(directory: Path, n: int = 2, stage: str = "raw") -> FrameSet:
    """FrameSet mínimo (PNGs + frameset.json) en ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    rgba = np.zeros((16, 16, 4), np.uint8)
    rgba[4:12, 4:12] = (200, 40, 40, 255)
    frames = []
    for i in range(n):
        fname = frame_filename(i)
        save_frame_rgba(directory, fname, rgba)
        frames.append(Frame(index=i, file=fname, duration_ms=100))
    fs = FrameSet(stage=stage, frames=frames, meta={"size": [16, 16]})
    fs.save(directory)
    return fs


class _BoomStage(Stage):
    """Etapa que escribe un cuadro y explota a mitad, como una etapa real fallida."""

    name = "boom"

    def process(self, fs, in_dir, out_dir, params):
        save_frame_rgba(out_dir, frame_filename(0), np.zeros((16, 16, 4), np.uint8))
        raise RuntimeError("boom a mitad de etapa")


class _CopyStage(Stage):
    name = "copystage"

    def process(self, fs, in_dir, out_dir, params):
        for fr in fs.frames:
            shutil.copy2(Path(in_dir) / fr.file, Path(out_dir) / fr.file)
        return fs


# -------------------------------------------- latest_stage_dir / working_dir

def test_latest_stage_dir_salta_dir_parcial(data_dir: Path):
    paths = Paths("p", "a", data_dir=data_dir)
    _make_frameset(paths.stage_dir(2, "background"), stage="background")
    partial = paths.stage_dir(3, "align")
    partial.mkdir(parents=True)
    (partial / "frame_0000.png").write_bytes(b"parcial, sin frameset.json")

    assert paths.latest_stage_dir() == paths.stage_dir(2, "background")
    # working_dir se beneficia: sin ediciones usa la etapa completa anterior.
    assert paths.working_dir() == paths.stage_dir(2, "background")


def test_latest_stage_dir_sin_etapas_completas(data_dir: Path):
    paths = Paths("p", "b", data_dir=data_dir)
    partial = paths.stage_dir(0, "raw")
    partial.mkdir(parents=True)

    assert paths.latest_stage_dir() is None
    assert paths.working_dir() is None


def test_working_dir_ignora_ediciones_sin_manifiesto(data_dir: Path):
    paths = Paths("p", "c", data_dir=data_dir)
    _make_frameset(paths.stage_dir(5, "smooth"), stage="smooth")
    poisoned = paths.edit_dir(1)
    poisoned.mkdir(parents=True)  # edición envenenada: dir sin frameset.json

    assert paths.list_edit_versions() == []
    assert paths.working_dir() == paths.stage_dir(5, "smooth")
    # Pero el número de la versión basura no se reutiliza (evita colisiones).
    assert paths.next_edit_version() == 2


# ----------------------------------------------------------------- run_stage

def test_run_stage_fallido_no_deja_dir_parcial_y_permite_reintento(data_dir: Path):
    paths = Paths("p", "d", data_dir=data_dir)
    in_dir = paths.stage_dir(0, "raw")
    _make_frameset(in_dir)
    out_dir = paths.stage_dir(1, "boom")

    with pytest.raises(RuntimeError):
        run_stage(_BoomStage(), in_dir, out_dir)

    assert not out_dir.exists()  # ni dir parcial sin manifiesto...
    assert not out_dir.with_name(out_dir.name + ".tmp").exists()  # ...ni temporal
    assert paths.latest_stage_dir() == in_dir  # la etapa previa sigue siendo la última

    # Reintento sobre el MISMO out_dir con una etapa sana funciona.
    fs = run_stage(_CopyStage(), in_dir, out_dir)
    assert (out_dir / "frameset.json").exists()
    assert FrameSet.load(out_dir).stage == "copystage"
    assert len(fs.frames) == 2


def test_run_stage_reemplaza_out_dir_previo(data_dir: Path):
    paths = Paths("p", "e", data_dir=data_dir)
    in_dir = paths.stage_dir(0, "raw")
    _make_frameset(in_dir)
    out_dir = paths.stage_dir(1, "copystage")
    out_dir.mkdir(parents=True)
    (out_dir / "restos.png").write_bytes(b"restos de una corrida anterior")

    fs = run_stage(_CopyStage(), in_dir, out_dir)

    assert (out_dir / "frameset.json").exists()
    assert not (out_dir / "restos.png").exists()  # el dir viejo fue reemplazado
    assert [f.file for f in fs.frames] == [frame_filename(i) for i in range(2)]


# ------------------------------------------------------- reconcile_stale_jobs

def test_reconcile_stale_jobs_marca_colgados(data_dir: Path):
    store = Store()
    project = store.create_project("p")
    a1 = store.create_animation(project["id"], "a1", "idle", "mock")
    a2 = store.create_animation(project["id"], "a2", "idle", "mock")
    a3 = store.create_animation(project["id"], "a3", "idle", "mock")

    j_queued = store.create_job(a1["id"], kind="pipeline")
    j_running = store.create_job(a2["id"], kind="pipeline")
    store.update_job(j_running["id"], status="running", progress=0.4)
    store.update_animation(a2["id"], status="processing")
    j_done = store.create_job(a3["id"], kind="pipeline")
    store.update_job(j_done["id"], status="succeeded", progress=1.0)
    store.update_animation(a3["id"], status="ready")

    assert reconcile_stale_jobs(store) == 2

    assert store.get_job(j_queued["id"])["status"] == "failed"
    assert store.get_job(j_queued["id"])["error"] == STALE_JOB_ERROR
    assert store.get_job(j_running["id"])["status"] == "failed"
    assert store.get_job(j_running["id"])["error"] == STALE_JOB_ERROR
    # El job terminado y su animación no se tocan.
    assert store.get_job(j_done["id"])["status"] == "succeeded"
    assert store.get_animation(a3["id"])["status"] == "ready"
    # La animación colgada en processing también queda failed.
    assert store.get_animation(a2["id"])["status"] == "failed"
    assert store.get_animation(a2["id"])["error"] == STALE_JOB_ERROR

    # Idempotente: segunda pasada no encuentra nada.
    assert reconcile_stale_jobs(store) == 0
