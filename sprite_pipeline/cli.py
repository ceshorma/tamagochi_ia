"""CLI del pipeline (``sprite-pipeline``): generate / export / serve / demo.

Todos los subcomandos aceptan ``--data-dir``: si se pasa, se exporta
``SPRITE_DATA_DIR`` y se resetea la caché de settings ANTES de crear el Store,
de modo que datos y artefactos queden bajo ese directorio.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from sprite_pipeline.config import get_settings, reset_settings_cache
from sprite_pipeline.export import export_animation
from sprite_pipeline.orchestrator import run_animation_pipeline
from sprite_pipeline.sample import save_sample_image
from sprite_pipeline.storage import Paths, Store, new_id

DEMO_ACTIONS = [("idle", 1), ("eat", 2), ("sleep", 3)]


# ------------------------------------------------------------------ helpers

def _apply_data_dir(data_dir: str | None) -> None:
    """Aplica ``--data-dir`` al entorno ANTES de instanciar Store/Paths."""
    if data_dir:
        os.environ["SPRITE_DATA_DIR"] = str(data_dir)
        reset_settings_cache()


def _get_or_create_project(store: Store, name: str) -> dict:
    """Reusa el primer proyecto con ese nombre o lo crea."""
    for project in store.list_projects():
        if project["name"] == name:
            return project
    return store.create_project(name)


def _generate_one(
    store: Store,
    project: dict,
    image: Path,
    action: str,
    name: str,
    provider: str | None,
    fps: int | None,
    duration_s: float | None,
    seed: int | None,
) -> dict:
    """Crea la animación y corre el pipeline síncrono (errores re-lanzados)."""
    params: dict = {}
    if fps is not None:
        params["fps"] = fps
    if duration_s is not None:
        params["duration_s"] = duration_s
    if seed is not None:
        params["seed"] = seed
    anim = store.create_animation(
        project["id"], name, action, provider, params=json.dumps(params)
    )
    params["raise_errors"] = True
    run_animation_pipeline(store, anim["id"], Path(image), params)
    return store.get_animation(anim["id"])


# ---------------------------------------------------------------- comandos

def cmd_generate(args: argparse.Namespace) -> int:
    _apply_data_dir(args.data_dir)
    store = Store()
    project = _get_or_create_project(store, args.project)
    anim = _generate_one(
        store,
        project,
        image=Path(args.image),
        action=args.action,
        name=args.name or args.action,
        provider=args.provider,
        fps=args.fps,
        duration_s=args.duration_s,
        seed=args.seed,
    )
    paths = Paths(anim["project_id"], anim["id"])
    print(f"animation: {anim['id']}")
    print(f"status: {anim['status']}")
    print(f"frames: {paths.latest_stage_dir()}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    _apply_data_dir(args.data_dir)
    store = Store()
    anim = store.get_animation(args.animation)
    if anim is None:
        raise SystemExit(f"Animación no encontrada: {args.animation!r}")
    paths = Paths(anim["project_id"], anim["id"])
    src = paths.working_dir()
    if src is None:
        raise SystemExit(f"La animación {args.animation!r} no tiene cuadros procesados aún")
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    out_dir = paths.export_dir(new_id())
    result = export_animation(
        src, out_dir, formats=formats, columns=args.columns, scale=args.scale
    )
    print(f"export: {out_dir}")
    for rel in result["files"]:
        print(f"  {out_dir / rel}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    _apply_data_dir(args.data_dir)
    import uvicorn

    uvicorn.run("sprite_pipeline.api.app:app", host=args.host, port=args.port)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    _apply_data_dir(args.data_dir)
    get_settings().ensure_dirs()
    store = Store()
    project = _get_or_create_project(store, "demo")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    summary: list[tuple[str, str, int, Path]] = []
    with tempfile.TemporaryDirectory(prefix="sprite-demo-") as tmp:
        image = save_sample_image(Path(tmp) / "creature.png", size=args.size)
        for action, seed in DEMO_ACTIONS:
            anim = _generate_one(
                store,
                project,
                image=image,
                action=action,
                name=action,
                provider="mock",
                fps=args.fps,
                duration_s=args.duration_s,
                seed=seed,
            )
            paths = Paths(anim["project_id"], anim["id"])
            export_dir = paths.export_dir(new_id())
            result = export_animation(
                paths.working_dir(), export_dir, formats=["sheet", "gif", "godot"]
            )
            dest = out_root / action
            shutil.copytree(export_dir, dest, dirs_exist_ok=True)
            summary.append((action, anim["id"], len(result["files"]), dest))

    print(f"demo: {len(summary)} animaciones exportadas a {out_root}")
    for action, aid, n_files, dest in summary:
        print(f"  {action}: animation {aid}, {n_files} archivos -> {dest}")
    return 0


# ------------------------------------------------------------------ parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sprite-pipeline",
        description="AI Sprite Pipeline: imagen -> video generado -> sprite animado.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Genera una animación end-to-end (síncrono)")
    g.add_argument("--image", required=True, help="Imagen fuente del personaje")
    g.add_argument("--action", required=True, help='Acción a animar ("idle", "eat", ...)')
    g.add_argument("--name", default=None, help="Nombre de la animación (default: la acción)")
    g.add_argument("--project", default="default", help="Nombre de proyecto (crea/reusa)")
    g.add_argument("--provider", default=None, help="Proveedor de video (default: settings)")
    g.add_argument("--fps", type=int, default=None)
    g.add_argument("--duration-s", dest="duration_s", type=float, default=None)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--data-dir", default=None)
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("export", help="Exporta el FrameSet de trabajo de una animación")
    e.add_argument("--animation", required=True, help="ID de la animación")
    e.add_argument("--formats", default="sheet,pngseq,gif", help="CSV: sheet,pngseq,gif,godot")
    e.add_argument("--columns", type=int, default=None)
    e.add_argument("--scale", type=int, default=1)
    e.add_argument("--data-dir", default=None)
    e.set_defaults(func=cmd_export)

    s = sub.add_parser("serve", help="Levanta el API REST con uvicorn")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--data-dir", default=None)
    s.set_defaults(func=cmd_serve)

    d = sub.add_parser("demo", help="Genera y exporta idle/eat/sleep de la criatura de muestra")
    d.add_argument("--out", default="examples")
    d.add_argument("--data-dir", default=None)
    d.add_argument("--size", type=int, default=200)
    d.add_argument("--fps", type=int, default=10)
    d.add_argument("--duration-s", dest="duration_s", type=float, default=2.0)
    d.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
