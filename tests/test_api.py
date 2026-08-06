"""Tests de la API REST (FastAPI + TestClient) contra docs/API_SPEC.md.

La app se crea DESPUÉS de activar el fixture ``data_dir`` (``create_app(Store())``)
para que cada test viva en su propio ``SPRITE_DATA_DIR``. Los tests de edición,
export y preview usan FrameSets sintéticos (rápidos); el flujo completo
proveedor mock -> etapas se ejerce una sola vez con ``?sync=1``.
"""

from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from sprite_pipeline.config import get_settings, reset_settings_cache
from sprite_pipeline.models import Frame, FrameSet, frame_filename
from sprite_pipeline.storage import Paths, Store


# ------------------------------------------------------------------ fixtures

@pytest.fixture()
def store(data_dir: Path) -> Store:
    """Store creado DESPUÉS de aislar SPRITE_DATA_DIR."""
    return Store()


@pytest.fixture()
def client(store: Store, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Trabajo a 128 px para velocidad. El default del dataclass Settings se
    # evaluó al importar el módulo, así que además de la variable de entorno
    # se fija el atributo en la instancia cacheada.
    monkeypatch.setenv("SPRITE_WORK_SIZE", "128")
    reset_settings_cache()
    get_settings().work_size = 128

    from sprite_pipeline.api.app import create_app

    with TestClient(create_app(store)) as c:
        yield c


# ------------------------------------------------------------------- helpers

def _synthetic_animation(store: Store, n: int = 4, size: int = 64) -> str:
    """Animación 'ready' con un FrameSet sintético en stages/01_preprocess."""
    project = store.create_project("proyecto-sintetico")
    anim = store.create_animation(project["id"], "bloques", "idle", "mock")
    store.update_animation(anim["id"], status="ready", current_stage="preprocess")
    paths = Paths(project["id"], anim["id"])
    stage_dir = paths.stage_dir(1, "preprocess")
    stage_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Frame] = []
    for i in range(n):
        rgba = np.zeros((size, size, 4), dtype=np.uint8)
        rgba[8:-8, 8:-8] = (30 + 50 * i, 80, 200 - 40 * i, 255)  # color por cuadro
        Image.fromarray(rgba, mode="RGBA").save(stage_dir / frame_filename(i))
        frames.append(Frame(index=i, file=frame_filename(i), duration_ms=100))
    FrameSet(stage="preprocess", frames=frames, meta={"action": "idle"}).save(stage_dir)
    return anim["id"]


def _get_frameset(client: TestClient, aid: str, version: str = "working") -> dict:
    r = client.get(f"/api/animations/{aid}/frameset", params={"version": version})
    assert r.status_code == 200, r.text
    return r.json()


# --------------------------------------------------------------------- health

def test_health(client: TestClient):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ------------------------------------------------------------------- projects

def test_projects_crud_lite(client: TestClient):
    r = client.post("/api/projects", json={"name": "tamagotchi"})
    assert r.status_code == 200, r.text
    project = r.json()
    assert project["name"] == "tamagotchi"
    assert project["id"]

    r = client.get("/api/projects")
    assert r.status_code == 200
    assert any(p["id"] == project["id"] for p in r.json())

    r = client.get(f"/api/projects/{project['id']}")
    assert r.status_code == 200
    assert r.json()["name"] == "tamagotchi"

    assert client.get("/api/projects/no-existe").status_code == 404
    assert client.get("/api/projects/no-existe/animations").status_code == 404
    # body sin 'name' -> 400
    assert client.post("/api/projects", json={}).status_code == 400


# ------------------------------------------- flujo completo síncrono (mock)

def test_pipeline_sync_full_flow(client: TestClient, sample_image: Path):
    pid = client.post("/api/projects", json={"name": "flujo"}).json()["id"]

    with open(sample_image, "rb") as fh:
        r = client.post(
            f"/api/projects/{pid}/animations",
            params={"sync": 1},
            files={"file": ("creature.png", fh, "image/png")},
            data={
                "name": "idle",
                "action": "idle",
                "provider": "mock",
                "fps": "6",
                "duration_s": "1.0",
                "seed": "3",
            },
        )
    assert r.status_code == 200, r.text
    payload = r.json()
    anim, job = payload["animation"], payload["job"]
    assert anim["status"] == "ready", anim
    aid = anim["id"]

    # job del run síncrono
    r = client.get(f"/api/jobs/{job['id']}")
    assert r.status_code == 200
    job_row = r.json()
    assert job_row["status"] == "succeeded"
    assert job_row["progress"] == pytest.approx(1.0)

    # aparece en la lista de animaciones del proyecto
    r = client.get(f"/api/projects/{pid}/animations")
    assert r.status_code == 200
    assert any(a["id"] == aid for a in r.json())

    # detalle: stages presentes + resumen de flags del working frameset
    r = client.get(f"/api/animations/{aid}")
    assert r.status_code == 200
    detail = r.json()
    assert detail["status"] == "ready"
    assert detail["stages"][0] == "00_raw"
    assert detail["stages"][-1] == "05_smooth"
    assert len(detail["stages"]) == 6
    assert detail["flags"] is not None
    assert detail["flags"]["frames"] >= 2
    assert "by_flag" in detail["flags"]

    # frameset: working (sin ediciones == última etapa), auto y por etapa
    working = _get_frameset(client, aid, "working")
    assert working["stage"] == "smooth"
    assert len(working["frames"]) >= 2
    assert _get_frameset(client, aid, "auto")["stage"] == "smooth"
    assert _get_frameset(client, aid, "raw")["stage"] == "raw"
    assert _get_frameset(client, aid, "preprocess")["stage"] == "preprocess"
    r = client.get(f"/api/animations/{aid}/frameset", params={"version": "inexistente"})
    assert r.status_code == 404

    # PNG de un cuadro por nombre de archivo
    first = working["frames"][0]["file"]
    r = client.get(f"/api/animations/{aid}/frames/{first}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content.startswith(b"\x89PNG")

    # 404 cuadro inexistente; 400 nombre con '..'
    assert client.get(f"/api/animations/{aid}/frames/nope.png").status_code == 404
    assert client.get(f"/api/animations/{aid}/frames/raro..nombre.png").status_code == 400


# ------------------------------------------------------- validación de subidas

def _post_animation(client: TestClient, pid: str, content: bytes, name: str = "up.png"):
    return client.post(
        f"/api/projects/{pid}/animations",
        files={"file": (name, io.BytesIO(content), "image/png")},
        data={"name": "x", "action": "idle", "provider": "mock"},
    )


def test_upload_no_imagen_400(client: TestClient):
    """Un archivo que PIL no decodifica se rechaza con 400 y no crea nada."""
    pid = client.post("/api/projects", json={"name": "subidas"}).json()["id"]
    r = _post_animation(client, pid, b"esto no es una imagen valida" * 64)
    assert r.status_code == 400
    assert "imagen" in r.json()["detail"]
    # no quedó ninguna animación huérfana
    assert client.get(f"/api/projects/{pid}/animations").json() == []


def test_upload_gigante_413(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """Subida que excede el tope de bytes -> 413 (tope bajado por monkeypatch)."""
    import sprite_pipeline.api.app as app_module

    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 1024)
    pid = client.post("/api/projects", json={"name": "subidas"}).json()["id"]
    payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8192  # > 1024 bytes
    r = _post_animation(client, pid, payload, name="grande.png")
    assert r.status_code == 413
    assert client.get(f"/api/projects/{pid}/animations").json() == []


def test_upload_dimensiones_excesivas_400(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    """Imagen válida pero con lado mayor al máximo -> 400 (límite por monkeypatch)."""
    import sprite_pipeline.api.app as app_module

    monkeypatch.setattr(app_module, "MAX_IMAGE_SIDE", 64)
    pid = client.post("/api/projects", json={"name": "subidas"}).json()["id"]
    buf = io.BytesIO()
    Image.new("RGBA", (128, 128), (255, 0, 0, 255)).save(buf, format="PNG")
    r = _post_animation(client, pid, buf.getvalue())
    assert r.status_code == 400
    assert "px" in r.json()["detail"]
    assert client.get(f"/api/projects/{pid}/animations").json() == []


# ---------------------------------------------------------------------- edits

def test_edits_y_undo(client: TestClient, store: Store):
    aid = _synthetic_animation(store, n=4)

    def edit(op: dict) -> dict:
        r = client.post(f"/api/animations/{aid}/edits", json=op)
        assert r.status_code == 200, r.text
        return r.json()

    # delete: 4 -> 3 cuadros, archivos renumerados
    fs = edit({"op": "delete", "index": 0})
    assert len(fs["frames"]) == 3
    assert [f["file"] for f in fs["frames"]] == [frame_filename(i) for i in range(3)]

    # duplicate: 3 -> 4, el nuevo va tras el original con source=duplicated
    fs = edit({"op": "duplicate", "index": 1})
    assert len(fs["frames"]) == 4
    assert fs["frames"][2]["source"] == "duplicated"

    # set_duration
    fs = edit({"op": "set_duration", "index": 0, "duration_ms": 160})
    assert fs["frames"][0]["duration_ms"] == 160

    # interpolate: inserta cuadro sintetizado tras after_index
    fs = edit({"op": "interpolate", "after_index": 1})
    assert len(fs["frames"]) == 5
    assert fs["frames"][2]["source"] == "interpolated"
    assert [f["source"] for f in fs["frames"]] == [
        "generated", "generated", "interpolated", "duplicated", "generated",
    ]

    # el working frameset refleja las ediciones y todos los PNG se sirven
    working = _get_frameset(client, aid, "working")
    assert working["stage"] == "edit"
    assert len(working["frames"]) == 5
    for frame in working["frames"]:
        r = client.get(f"/api/animations/{aid}/frames/{frame['file']}")
        assert r.status_code == 200, frame["file"]

    # undo: quita la interpolación
    r = client.post(f"/api/animations/{aid}/undo")
    assert r.status_code == 200
    fs = r.json()
    assert len(fs["frames"]) == 4
    assert fs["frames"][0]["duration_ms"] == 160
    assert [f["source"] for f in fs["frames"]] == [
        "generated", "generated", "duplicated", "generated",
    ]

    # undo otra vez: revierte set_duration
    fs = client.post(f"/api/animations/{aid}/undo").json()
    assert fs["frames"][0]["duration_ms"] == 100
    assert _get_frameset(client, aid, "working")["frames"][0]["duration_ms"] == 100

    # 'auto' sigue apuntando a la última etapa, sin ediciones
    assert _get_frameset(client, aid, "auto")["stage"] == "preprocess"


def test_edits_errores(client: TestClient, store: Store):
    aid = _synthetic_animation(store, n=3)

    # op desconocida -> 400 con detail
    r = client.post(f"/api/animations/{aid}/edits", json={"op": "explotar"})
    assert r.status_code == 400
    assert "explotar" in r.json()["detail"]

    # índice fuera de rango -> 400
    r = client.post(f"/api/animations/{aid}/edits", json={"op": "delete", "index": 99})
    assert r.status_code == 400

    # body sin 'op' -> 400
    assert client.post(f"/api/animations/{aid}/edits", json={}).status_code == 400

    # animación inexistente -> 404
    r = client.post("/api/animations/zzz/edits", json={"op": "delete", "index": 0})
    assert r.status_code == 404
    assert client.post("/api/animations/zzz/undo").status_code == 404

    # undo sin ediciones -> devuelve el frameset de trabajo actual
    aid2 = _synthetic_animation(store, n=3)
    r = client.post(f"/api/animations/{aid2}/undo")
    assert r.status_code == 200
    assert len(r.json()["frames"]) == 3


def test_mutaciones_409_mientras_corre_el_pipeline(client: TestClient, store: Store):
    """edits/undo/export -> 409 con detail claro si el pipeline sigue corriendo."""
    aid = _synthetic_animation(store, n=3)
    for status in ("generating", "processing"):
        store.update_animation(aid, status=status)
        r = client.post(f"/api/animations/{aid}/edits", json={"op": "delete", "index": 0})
        assert r.status_code == 409, r.text
        assert status in r.json()["detail"]
        r = client.post(f"/api/animations/{aid}/undo")
        assert r.status_code == 409
        assert status in r.json()["detail"]
        r = client.post(f"/api/animations/{aid}/export", json={"formats": ["sheet"]})
        assert r.status_code == 409
        assert status in r.json()["detail"]
    # nada mutó mientras estaba ocupado: siguen los 3 cuadros originales
    assert len(_get_frameset(client, aid, "working")["frames"]) == 3
    # al quedar 'ready' las mutaciones vuelven a aceptarse
    store.update_animation(aid, status="ready")
    r = client.post(f"/api/animations/{aid}/edits", json={"op": "delete", "index": 0})
    assert r.status_code == 200, r.text
    assert len(r.json()["frames"]) == 2


# ------------------------------------------------------------ preview / export

def test_preview_gif(client: TestClient, store: Store):
    aid = _synthetic_animation(store, n=3)
    r = client.get(f"/api/animations/{aid}/preview.gif")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/gif"
    assert r.content.startswith(b"GIF8")
    assert client.get("/api/animations/zzz/preview.gif").status_code == 404


def test_preview_gif_concurrente(client: TestClient, store: Store):
    """Dos GET preview.gif a la vez: ambos 200, GIFs válidos y de SU versión.

    Además el cache por request se limpia tras servir: sin el fix (ruta
    compartida ``preview_cache/preview.gif``) quedaría un archivo persistente
    mutable entre requests.
    """
    aid = _synthetic_animation(store, n=4)
    # una edición para que 'working' (3 cuadros) difiera de 'preprocess' (4)
    r = client.post(f"/api/animations/{aid}/edits", json={"op": "delete", "index": 0})
    assert r.status_code == 200, r.text

    def fetch(version: str):
        return version, client.get(
            f"/api/animations/{aid}/preview.gif", params={"version": version}
        )

    with ThreadPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(fetch, ["working", "preprocess"]))

    expected_frames = {"working": 3, "preprocess": 4}
    for version, resp in results:
        assert resp.status_code == 200, (version, resp.text)
        assert resp.headers["content-type"] == "image/gif"
        gif = Image.open(io.BytesIO(resp.content))  # PIL lo abre: GIF no corrupto
        assert gif.n_frames == expected_frames[version], version

    # nada compartido persiste: el subdir por request se limpió tras responder
    anim = store.get_animation(aid)
    cache_root = Paths(anim["project_id"], aid).animation_dir / "preview_cache"
    leftovers = sorted(str(p) for p in cache_root.rglob("*")) if cache_root.exists() else []
    assert leftovers == []


def test_export_y_descarga(client: TestClient, store: Store):
    aid = _synthetic_animation(store, n=4)

    r = client.post(
        f"/api/animations/{aid}/export",
        json={"formats": ["sheet", "gif"], "columns": 2, "scale": 1},
    )
    assert r.status_code == 200, r.text
    payload = r.json()
    export_id = payload["export_id"]
    assert export_id
    assert set(payload["files"]) == {"sheet.png", "sheet.json", "preview.gif"}
    assert payload["sheet"]["columns"] == 2

    base = f"/api/animations/{aid}/exports/{export_id}"

    r = client.get(f"{base}/sheet.png")
    assert r.status_code == 200
    assert "image/png" in r.headers["content-type"]
    assert r.content.startswith(b"\x89PNG")

    r = client.get(f"{base}/sheet.json")
    assert r.status_code == 200
    assert "json" in r.headers["content-type"]
    manifest = json.loads(r.content)
    assert len(manifest["frames"]) == 4

    r = client.get(f"{base}/preview.gif")
    assert r.status_code == 200
    assert "image/gif" in r.headers["content-type"]
    assert r.content.startswith(b"GIF8")

    # artefacto inexistente -> 404; traversal -> rechazado
    assert client.get(f"{base}/no-esta.png").status_code == 404
    r = client.get(f"{base}/%2e%2e/%2e%2e/source.png")
    assert r.status_code in (400, 404)

    # formato desconocido -> 400; export de animación inexistente -> 404
    r = client.post(f"/api/animations/{aid}/export", json={"formats": ["exe"]})
    assert r.status_code == 400
    assert client.post("/api/animations/zzz/export", json={}).status_code == 404


def test_export_parametros_fuera_de_rango_400(client: TestClient, store: Store):
    """columns/scale/padding desmesurados -> 400 (nunca OOM ni 500)."""
    aid = _synthetic_animation(store, n=4)
    base = f"/api/animations/{aid}/export"

    r = client.post(base, json={"formats": ["sheet"], "columns": 100000})
    assert r.status_code == 400
    assert "columns" in r.json()["detail"]

    r = client.post(base, json={"formats": ["sheet"], "scale": 99})
    assert r.status_code == 400
    assert "scale" in r.json()["detail"]

    r = client.post(base, json={"formats": ["sheet"], "padding": 10**9})
    assert r.status_code == 400
    assert "padding" in r.json()["detail"]


# -------------------------------------------------------------------- startup

def test_startup_llama_reconcile_stale_jobs(
    store: Store, monkeypatch: pytest.MonkeyPatch
):
    """El arranque de la app reconcilia jobs huérfanos vía el orquestador."""
    import sprite_pipeline.orchestrator as orch

    calls: list = []
    monkeypatch.setattr(orch, "reconcile_stale_jobs", calls.append, raising=False)
    from sprite_pipeline.api.app import create_app

    with TestClient(create_app(store)):
        pass
    assert calls == [store]


# --------------------------------------------------------------------- editor

def test_editor_html_y_estaticos(client: TestClient, store: Store):
    aid = _synthetic_animation(store, n=2)
    r = client.get(f"/editor/{aid}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "app.js" in r.text

    r = client.get("/editor/static/app.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]

    r = client.get("/editor/static/style.css")
    assert r.status_code == 200
    assert "css" in r.headers["content-type"]

    assert client.get("/editor/static/no-existe.js").status_code == 404


# ----------------------------------------------------------------------- 404s

def test_404_animacion_y_job(client: TestClient):
    assert client.get("/api/animations/no-existe").status_code == 404
    assert client.get("/api/animations/no-existe/frameset").status_code == 404
    assert client.get("/api/animations/no-existe/frames/frame_0000.png").status_code == 404
    assert client.get("/api/jobs/no-existe").status_code == 404
