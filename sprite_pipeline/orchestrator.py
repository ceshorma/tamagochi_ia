"""Orquestación del pipeline: imagen -> video -> etapas -> sprite listo.

- :func:`run_animation_pipeline` corre TODO el pipeline de forma síncrona y
  actualiza ``animations.status/current_stage`` (y el job, si se pasa
  ``params["job_id"]``) en cada paso.
- :class:`Orchestrator` encola ejecuciones en un ``ThreadPoolExecutor``
  (max_workers=2) y crea el job ``kind="pipeline"`` asociado.
- :func:`get_orchestrator` devuelve un singleton por proceso y por ``Store``.

Estados de animación: created -> generating -> processing -> ready | failed.
Estados de job: queued -> running -> succeeded | failed.
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sprite_pipeline.config import get_settings
from sprite_pipeline.extract import extract_frames
from sprite_pipeline.providers.base import GenerationRequest, get_provider
from sprite_pipeline.stages.base import get_stage, run_stage
from sprite_pipeline.storage import Paths, Store

PIPELINE_ORDER = ["preprocess", "background", "align", "anomaly", "smooth"]

#: Claves de ``params`` que son del orquestador/proveedor, no de las etapas.
_NON_STAGE_KEYS = frozenset(
    {"job_id", "raise_errors", "fps", "duration_s", "seed", "prompt_extra", "provider"}
)


def run_animation_pipeline(
    store: Store,
    animation_id: str,
    image_path: Path,
    params: dict | None = None,
) -> dict:
    """Corre el pipeline completo de forma síncrona. Devuelve la fila final.

    Ante cualquier excepción marca la animación (y el job) como ``failed`` con
    el mensaje en ``error`` y re-lanza solo si ``params.get("raise_errors")``.
    """
    params = dict(params or {})
    anim = store.get_animation(animation_id)
    if anim is None:
        raise KeyError(f"Animación desconocida: {animation_id!r}")

    settings = get_settings()
    paths = Paths(anim["project_id"], animation_id)
    job_id = params.get("job_id")
    total_steps = len(PIPELINE_ORDER) + 2  # generación+extracción, etapas, cierre

    def _job(**fields) -> None:
        if job_id:
            store.update_job(job_id, **fields)

    try:
        if job_id:
            _job(status="running", progress=0.0)

        # 1. Imagen fuente al layout de la animación.
        image_path = Path(image_path)
        paths.source_image.parent.mkdir(parents=True, exist_ok=True)
        if image_path.resolve() != paths.source_image.resolve():
            shutil.copyfile(image_path, paths.source_image)

        # 2. Generación de video con el proveedor configurado.
        store.update_animation(animation_id, status="generating")
        provider = get_provider(anim["provider"] or settings.provider)
        fps = int(params.get("fps") or settings.default_fps)
        req = GenerationRequest(
            image_path=paths.source_image,
            action=anim["action"],
            prompt_extra=str(params.get("prompt_extra") or ""),
            seed=params.get("seed"),
            duration_s=float(params.get("duration_s") or settings.default_duration_s),
            fps=fps,
        )
        paths.raw_video.parent.mkdir(parents=True, exist_ok=True)
        result = provider.generate(req, paths.raw_video.parent)
        if Path(result.video_path).resolve() != paths.raw_video.resolve():
            shutil.copy2(result.video_path, paths.raw_video)

        # 3. Extracción de fotogramas -> FrameSet raw (stage dir 00_raw).
        store.update_animation(animation_id, status="processing", current_stage="raw")
        extract_frames(
            paths.raw_video,
            paths.stage_dir(0, "raw"),
            fps=fps,
            meta={
                "provider": result.provider,
                "prompt": result.prompt,
                "action": anim["action"],
                "source_image": str(paths.source_image),
            },
        )
        _job(stage="raw", progress=1.0 / total_steps)

        # 4. Etapas en orden, cada una lee el directorio de la anterior.
        stage_params = {k: v for k, v in params.items() if k not in _NON_STAGE_KEYS}
        prev_dir = paths.stage_dir(0, "raw")
        for i, stage_name in enumerate(PIPELINE_ORDER, 1):
            out_dir = paths.stage_dir(i, stage_name)
            run_stage(get_stage(stage_name), prev_dir, out_dir, stage_params)
            store.update_animation(animation_id, current_stage=stage_name)
            _job(stage=stage_name, progress=(i + 1) / total_steps)
            prev_dir = out_dir

        store.update_animation(animation_id, status="ready", error=None)
        _job(status="succeeded", progress=1.0)
    except Exception as exc:  # noqa: BLE001 - el estado failed debe registrarse siempre
        store.update_animation(animation_id, status="failed", error=str(exc))
        _job(status="failed", error=str(exc))
        if params.get("raise_errors"):
            raise
    return store.get_animation(animation_id)


class Orchestrator:
    """Encola ejecuciones del pipeline en hilos de fondo (máx. 2 en paralelo)."""

    def __init__(self, store: Store):
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sprite-pipeline")

    def submit_generation(self, animation_id: str, image_path: Path, params: dict | None = None) -> dict:
        """Crea el job ``kind="pipeline"``, encola la ejecución y devuelve el job."""
        params = dict(params or {})
        job = self.store.create_job(animation_id, kind="pipeline")
        params["job_id"] = job["id"]
        self.executor.submit(run_animation_pipeline, self.store, animation_id, Path(image_path), params)
        return job


#: Singleton por proceso: una instancia de Orchestrator por Store.
_ORCHESTRATORS: dict[int, Orchestrator] = {}


def get_orchestrator(store: Store) -> Orchestrator:
    """Orchestrator singleton para ``store`` (nueva instancia si el store cambia)."""
    key = id(store)
    inst = _ORCHESTRATORS.get(key)
    if inst is None:
        inst = Orchestrator(store)
        _ORCHESTRATORS[key] = inst
    return inst
