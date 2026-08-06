"""API REST del pipeline (FastAPI).

Contrato en ``docs/API_SPEC.md``: factory :func:`create_app` (acepta un
``Store`` inyectado para tests) e instancia ``app`` a nivel de módulo.

El ``Store`` por defecto se crea de forma perezosa en el primer request y se
re-crea si ``settings.data_dir`` cambia entre requests (los tests aíslan
``SPRITE_DATA_DIR`` por test con ``reset_settings_cache``). Los ``Paths`` se
construyen por request; nunca se cachean globalmente.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from starlette.background import BackgroundTask

from sprite_pipeline import orchestrator as orchestrator_module
from sprite_pipeline.config import get_settings
from sprite_pipeline.export import export_animation
from sprite_pipeline.models import MANIFEST_NAME, FrameSet
from sprite_pipeline.ops import edits as edit_ops
from sprite_pipeline.orchestrator import get_orchestrator, run_animation_pipeline
from sprite_pipeline.storage import Paths, Store, new_id

#: Directorio de assets del editor dentro del paquete.
STATIC_DIR = Path(__file__).resolve().parents[1] / "editor" / "static"

#: Tope de tamaño de la imagen subida (bytes). Excedido -> HTTP 413.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
#: Lado máximo (px) de la imagen subida. Excedido -> HTTP 400.
MAX_IMAGE_SIDE = 4096
#: Tamaño de chunk al leer la subida (nunca se materializa entera en memoria).
UPLOAD_CHUNK_BYTES = 1024 * 1024

#: Estados de animación con el pipeline en curso: las mutaciones (edits/undo/
#: export) se rechazan con HTTP 409 mientras dure.
BUSY_STATUSES = ("generating", "processing")


# ------------------------------------------------------------------ helpers

def _check_filename(name: str) -> str:
    """Nombre simple de archivo: sin separadores ni ``..`` (anti-traversal)."""
    if not name or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail=f"Nombre de archivo inválido: {name!r}")
    return name


def _check_relpath(path: str) -> str:
    """Ruta relativa segura (puede tener subdirectorios, nunca ``..``)."""
    if not path or path.startswith(("/", "\\")) or "\\" in path:
        raise HTTPException(status_code=400, detail=f"Ruta inválida: {path!r}")
    if any(part in ("", ".", "..") for part in path.split("/")):
        raise HTTPException(status_code=400, detail=f"Ruta inválida: {path!r}")
    return path


async def _receive_upload(file: UploadFile, dest: Path) -> None:
    """Escribe la subida en ``dest`` por chunks y la valida como imagen.

    - Más de ``MAX_UPLOAD_BYTES`` -> 413 (se aborta sin leer el resto y sin
      cargar el archivo completo en memoria).
    - No decodificable por PIL -> 400.
    - Lado mayor que ``MAX_IMAGE_SIDE`` px -> 400.

    Ante cualquier fallo se elimina ``dest``.
    """
    size = 0
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Archivo demasiado grande "
                            f"(máximo {MAX_UPLOAD_BYTES} bytes)"
                        ),
                    )
                out.write(chunk)
        try:
            # verify() detecta archivos truncados/corruptos pero invalida el
            # objeto: se re-abre para leer las dimensiones.
            with Image.open(dest) as img:
                img.verify()
            with Image.open(dest) as img:
                width, height = img.size
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="El archivo subido no es una imagen válida",
            ) from exc
        if max(width, height) > MAX_IMAGE_SIDE:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Imagen demasiado grande: {width}x{height} px "
                    f"(lado máximo {MAX_IMAGE_SIDE} px)"
                ),
            )
    except BaseException:
        dest.unlink(missing_ok=True)
        raise


def _load_manifest(directory: Path) -> dict:
    return json.loads((directory / MANIFEST_NAME).read_text())


def _fs_to_dict(fs: FrameSet) -> dict:
    """Manifiesto de un FrameSet como dict (mismo formato que ``frameset.json``)."""
    return {
        "schema_version": fs.schema_version,
        "stage": fs.stage,
        "meta": fs.meta,
        "history": fs.history,
        "frames": [f.to_dict() for f in fs.frames],
    }


def _resolve_frameset_dir(paths: Paths, version: str) -> Path:
    """Directorio del FrameSet pedido por ``version``.

    - ``working``: última edición o, si no hay, última etapa.
    - ``auto``: última etapa procesada.
    - nombre de etapa (``raw``, ``preprocess``, ...): busca ``NN_<stage>``
      por sufijo dentro de ``stages/``.
    """
    version = (version or "working").strip() or "working"
    directory: Path | None
    if version == "working":
        directory = paths.working_dir()
    elif version == "auto":
        directory = paths.latest_stage_dir()
    else:
        directory = None
        if paths.stages_dir.exists():
            for cand in sorted(paths.stages_dir.iterdir()):
                if cand.is_dir() and cand.name.endswith(f"_{version}"):
                    directory = cand
    if directory is None or not (directory / MANIFEST_NAME).exists():
        raise HTTPException(
            status_code=404, detail=f"No hay FrameSet para version={version!r}"
        )
    return directory


# ---------------------------------------------------------------------- app

def create_app(store: Store | None = None) -> FastAPI:
    """Crea la aplicación FastAPI. ``store`` inyectable para tests."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Jobs/animaciones que quedaron 'queued'/'running'/'processing' por un
        # reinicio: el executor es solo memoria, así que se reconcilian al
        # arrancar. getattr tolerante: no-op si la función aún no existe.
        reconcile = getattr(orchestrator_module, "reconcile_stale_jobs", None)
        if reconcile is not None:
            reconcile(get_store())
        yield

    app = FastAPI(title="AI Sprite Pipeline API", lifespan=lifespan)
    app.state.injected_store = store
    app.state.default_store = None

    # ------------------------------------------------------------ store/paths

    def get_store() -> Store:
        if app.state.injected_store is not None:
            return app.state.injected_store
        # Store por defecto perezoso; se re-crea si cambió settings.data_dir.
        expected_db = get_settings().data_dir / "sprite.db"
        cached: Store | None = app.state.default_store
        if cached is None or Path(cached.db_path).resolve() != expected_db.resolve():
            cached = Store()
            app.state.default_store = cached
        return cached

    def _anim_or_404(store: Store, aid: str) -> dict:
        anim = store.get_animation(aid)
        if anim is None:
            raise HTTPException(status_code=404, detail=f"Animación no encontrada: {aid}")
        return anim

    def _paths_for(anim: dict) -> Paths:
        return Paths(anim["project_id"], anim["id"])

    def _reject_if_busy(anim: dict) -> None:
        """409 si el pipeline de la animación sigue corriendo (ver BUSY_STATUSES)."""
        status = anim.get("status")
        if status in BUSY_STATUSES:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"La animación {anim['id']} está '{status}': el pipeline "
                    "sigue corriendo; espera a que termine (status 'ready') "
                    "antes de editar, deshacer o exportar"
                ),
            )

    # ----------------------------------------------------------------- health

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    # --------------------------------------------------------------- projects

    @app.post("/api/projects")
    def create_project(payload: dict = Body(...), store: Store = Depends(get_store)) -> dict:
        name = (payload or {}).get("name")
        if not name or not isinstance(name, str):
            raise HTTPException(status_code=400, detail="Falta 'name' (string no vacío)")
        return store.create_project(name)

    @app.get("/api/projects")
    def list_projects(store: Store = Depends(get_store)) -> list[dict]:
        return store.list_projects()

    @app.get("/api/projects/{pid}")
    def get_project(pid: str, store: Store = Depends(get_store)) -> dict:
        project = store.get_project(pid)
        if project is None:
            raise HTTPException(status_code=404, detail=f"Proyecto no encontrado: {pid}")
        return project

    @app.get("/api/projects/{pid}/animations")
    def list_animations(pid: str, store: Store = Depends(get_store)) -> list[dict]:
        if store.get_project(pid) is None:
            raise HTTPException(status_code=404, detail=f"Proyecto no encontrado: {pid}")
        return store.list_animations(pid)

    # ------------------------------------------------------------- animations

    @app.post("/api/projects/{pid}/animations")
    async def create_animation(
        pid: str,
        file: UploadFile = File(...),
        name: str = Form(...),
        action: str = Form(...),
        provider: str | None = Form(None),
        fps: int | None = Form(None),
        duration_s: float | None = Form(None),
        seed: int | None = Form(None),
        sync: bool = Query(False),
        store: Store = Depends(get_store),
    ) -> dict:
        if store.get_project(pid) is None:
            raise HTTPException(status_code=404, detail=f"Proyecto no encontrado: {pid}")
        settings = get_settings()
        gen_params: dict = {}
        if fps is not None:
            gen_params["fps"] = fps
        if duration_s is not None:
            gen_params["duration_s"] = duration_s
        if seed is not None:
            gen_params["seed"] = seed

        # La subida se recibe y valida (tope de bytes -> 413, imagen PIL
        # decodificable y lado <= MAX_IMAGE_SIDE -> 400) ANTES de crear la
        # animación, en un dir temporal propio del request.
        suffix = Path(file.filename or "upload.png").suffix or ".png"
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        upload_dir = Path(tempfile.mkdtemp(prefix="upload-", dir=settings.data_dir))
        try:
            upload_tmp = upload_dir / f"upload{suffix}"
            await _receive_upload(file, upload_tmp)

            anim = store.create_animation(
                pid, name, action, provider or settings.provider, params=json.dumps(gen_params)
            )
            paths = Paths(pid, anim["id"])

            # Imagen subida a un tmp dentro del dir de la animación; el
            # orquestador la copia luego a source.png.
            tmp_dir = paths.animation_dir / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / f"upload{suffix}"
            shutil.move(str(upload_tmp), tmp_path)
        finally:
            shutil.rmtree(upload_dir, ignore_errors=True)

        if sync:
            job = store.create_job(anim["id"], kind="pipeline")
            run_params = {**gen_params, "job_id": job["id"], "raise_errors": False}
            animation = run_animation_pipeline(store, anim["id"], tmp_path, run_params)
            return {"animation": animation, "job": store.get_job(job["id"])}

        job = get_orchestrator(store).submit_generation(anim["id"], tmp_path, gen_params)
        return {"animation": store.get_animation(anim["id"]), "job": job}

    @app.get("/api/animations/{aid}")
    def get_animation(aid: str, store: Store = Depends(get_store)) -> dict:
        anim = _anim_or_404(store, aid)
        paths = _paths_for(anim)
        stages: list[str] = []
        if paths.stages_dir.exists():
            stages = sorted(d.name for d in paths.stages_dir.iterdir() if d.is_dir())
        detail = dict(anim)
        detail["stages"] = stages

        flags_summary = None
        working = paths.working_dir()
        if working is not None and (working / MANIFEST_NAME).exists():
            fs = FrameSet.load(working)
            counts: dict[str, int] = {}
            for frame in fs.frames:
                for flag in frame.flags:
                    counts[flag] = counts.get(flag, 0) + 1
            flags_summary = {
                "frames": len(fs.frames),
                "flagged": len(fs.flagged()),
                "by_flag": counts,
            }
        detail["flags"] = flags_summary
        return detail

    @app.get("/api/animations/{aid}/frameset")
    def get_frameset(
        aid: str,
        version: str = Query("working"),
        store: Store = Depends(get_store),
    ) -> dict:
        anim = _anim_or_404(store, aid)
        directory = _resolve_frameset_dir(_paths_for(anim), version)
        return _load_manifest(directory)

    @app.get("/api/animations/{aid}/frames/{file}")
    def get_frame(
        aid: str,
        file: str,
        version: str = Query("working"),
        store: Store = Depends(get_store),
    ) -> FileResponse:
        anim = _anim_or_404(store, aid)
        _check_filename(file)
        directory = _resolve_frameset_dir(_paths_for(anim), version)
        path = directory / file
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Cuadro no encontrado: {file}")
        return FileResponse(path, media_type="image/png")

    # ------------------------------------------------------------------ edits

    @app.post("/api/animations/{aid}/edits")
    def post_edit(
        aid: str,
        op: dict = Body(...),
        store: Store = Depends(get_store),
    ) -> dict:
        anim = _anim_or_404(store, aid)
        _reject_if_busy(anim)
        paths = _paths_for(anim)
        try:
            edit_ops.ensure_edit_session(paths)
            fs = edit_ops.apply_edit(paths, op)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _fs_to_dict(fs)

    @app.post("/api/animations/{aid}/undo")
    def post_undo(aid: str, store: Store = Depends(get_store)) -> dict:
        anim = _anim_or_404(store, aid)
        _reject_if_busy(anim)
        paths = _paths_for(anim)
        fs = edit_ops.undo(paths)
        if fs is not None:
            return _fs_to_dict(fs)
        # Nada que deshacer: devolver el frameset de trabajo actual.
        working = paths.working_dir()
        if working is None or not (working / MANIFEST_NAME).exists():
            raise HTTPException(
                status_code=404, detail="No hay ediciones ni etapas para esta animación"
            )
        return _load_manifest(working)

    # -------------------------------------------------------- preview / export

    @app.get("/api/animations/{aid}/preview.gif")
    def preview_gif(
        aid: str,
        version: str = Query("working"),
        store: Store = Depends(get_store),
    ) -> FileResponse:
        anim = _anim_or_404(store, aid)
        paths = _paths_for(anim)
        directory = _resolve_frameset_dir(paths, version)
        # Subdir único por request: dos requests concurrentes (o de versiones
        # distintas) nunca escriben/sirven el mismo archivo. Se limpia tras
        # enviar la respuesta (BackgroundTask del FileResponse).
        cache_dir = paths.animation_dir / "preview_cache" / uuid.uuid4().hex
        try:
            export_animation(directory, cache_dir, formats=["gif"])
        except ValueError as exc:
            shutil.rmtree(cache_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return FileResponse(
            cache_dir / "preview.gif",
            media_type="image/gif",
            background=BackgroundTask(shutil.rmtree, cache_dir, ignore_errors=True),
        )

    @app.post("/api/animations/{aid}/export")
    def post_export(
        aid: str,
        payload: dict | None = Body(None),
        store: Store = Depends(get_store),
    ) -> dict:
        anim = _anim_or_404(store, aid)
        _reject_if_busy(anim)
        paths = _paths_for(anim)
        payload = payload or {}
        working = paths.working_dir()
        if working is None or not (working / MANIFEST_NAME).exists():
            raise HTTPException(
                status_code=404, detail="No hay FrameSet procesado para exportar"
            )
        export_id = new_id()
        out_dir = paths.export_dir(export_id)
        try:
            result = export_animation(
                working,
                out_dir,
                formats=payload.get("formats"),
                columns=payload.get("columns"),
                padding=payload.get("padding", 2),
                scale=payload.get("scale", 1),
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"export_id": export_id, "files": result["files"], "sheet": result["sheet"]}

    @app.get("/api/animations/{aid}/exports/{export_id}/{path:path}")
    def download_export(
        aid: str,
        export_id: str,
        path: str,
        store: Store = Depends(get_store),
    ) -> FileResponse:
        anim = _anim_or_404(store, aid)
        _check_filename(export_id)
        _check_relpath(path)
        base = _paths_for(anim).export_dir(export_id)
        target = base / path
        try:
            target.resolve().relative_to(base.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Ruta inválida: {path!r}") from None
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"Artefacto no encontrado: {path}")
        return FileResponse(target)

    # ------------------------------------------------------------------- jobs

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str, store: Store = Depends(get_store)) -> dict:
        job = store.get_job(jid)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job no encontrado: {jid}")
        return job

    # ----------------------------------------------------------------- editor

    @app.get("/editor/static/{file}")
    def editor_static(file: str) -> FileResponse:
        _check_filename(file)
        path = STATIC_DIR / file
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Asset no encontrado: {file}")
        return FileResponse(path)

    @app.get("/editor/{aid}")
    def editor_page(aid: str) -> FileResponse:
        index = STATIC_DIR / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404, detail="Editor no disponible")
        return FileResponse(index, media_type="text/html")

    return app


app = create_app()
