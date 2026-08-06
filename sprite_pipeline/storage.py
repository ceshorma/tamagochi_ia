"""Persistencia: metadata en SQLite + artefactos en disco.

Layout de datos bajo ``settings.data_dir``::

    data/
      sprite.db
      projects/<pid>/animations/<aid>/
        source.png                  # imagen de entrada
        raw/video.mp4               # video crudo del proveedor
        stages/00_raw/              # FrameSet por etapa (frameset.json + PNGs)
        stages/01_preprocess/
        ...
        edits/edit_0001/            # versiones de edición (historial de deshacer)
        exports/<export_id>/        # artefactos exportados
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from sprite_pipeline.config import get_settings
from sprite_pipeline.models import MANIFEST_NAME

#: Nombre de dir de versión de edición: ``edit_NNNN`` (exacto para versiones
#: válidas; como prefijo para detectar basura/temporales y no reutilizar números).
_EDIT_DIR_RE = re.compile(r"^edit_(\d+)$")
_EDIT_DIR_PREFIX_RE = re.compile(r"^edit_(\d+)")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS animations (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    name TEXT NOT NULL,
    action TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'created',
    provider TEXT,
    current_stage TEXT,
    error TEXT,
    params TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    animation_id TEXT NOT NULL REFERENCES animations(id),
    kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    stage TEXT,
    progress REAL NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

# Estados de animación: created -> generating -> processing -> ready | failed
# Estados de job: queued -> running -> succeeded | failed


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class Store:
    """Acceso a SQLite thread-safe (una conexión por hilo)."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path) if db_path else get_settings().data_dir / "sprite.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    # ------------------------------------------------------------- projects
    def create_project(self, name: str) -> dict:
        pid = new_id()
        with self._conn() as c:
            c.execute(
                "INSERT INTO projects (id, name, created_at) VALUES (?,?,?)",
                (pid, name, _now()),
            )
        return self.get_project(pid)

    def get_project(self, pid: str) -> dict | None:
        row = self._conn().execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
        return dict(row) if row else None

    def list_projects(self) -> list[dict]:
        rows = self._conn().execute("SELECT * FROM projects ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    # ----------------------------------------------------------- animations
    def create_animation(self, project_id: str, name: str, action: str, provider: str, params: str = "{}") -> dict:
        aid = new_id()
        now = _now()
        with self._conn() as c:
            c.execute(
                "INSERT INTO animations (id, project_id, name, action, status, provider, params, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (aid, project_id, name, action, "created", provider, params, now, now),
            )
        return self.get_animation(aid)

    def get_animation(self, aid: str) -> dict | None:
        row = self._conn().execute("SELECT * FROM animations WHERE id=?", (aid,)).fetchone()
        return dict(row) if row else None

    def list_animations(self, project_id: str | None = None) -> list[dict]:
        if project_id:
            rows = self._conn().execute(
                "SELECT * FROM animations WHERE project_id=? ORDER BY created_at", (project_id,)
            ).fetchall()
        else:
            rows = self._conn().execute("SELECT * FROM animations ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    def update_animation(self, aid: str, **fields) -> None:
        if not fields:
            return
        fields["updated_at"] = _now()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE animations SET {cols} WHERE id=?", (*fields.values(), aid))

    # ----------------------------------------------------------------- jobs
    def create_job(self, animation_id: str, kind: str) -> dict:
        jid = new_id()
        now = _now()
        with self._conn() as c:
            c.execute(
                "INSERT INTO jobs (id, animation_id, kind, status, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                (jid, animation_id, kind, "queued", now, now),
            )
        return self.get_job(jid)

    def get_job(self, jid: str) -> dict | None:
        row = self._conn().execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        return dict(row) if row else None

    def list_jobs(self, animation_id: str) -> list[dict]:
        rows = self._conn().execute(
            "SELECT * FROM jobs WHERE animation_id=? ORDER BY created_at", (animation_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def update_job(self, jid: str, **fields) -> None:
        if not fields:
            return
        fields["updated_at"] = _now()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), jid))

    def list_unfinished_jobs(self) -> list[dict]:
        """Jobs que no llegaron a un estado terminal (``queued``/``running``)."""
        rows = self._conn().execute(
            "SELECT * FROM jobs WHERE status IN ('queued','running') ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]


class Paths:
    """Rutas de artefactos de una animación."""

    def __init__(self, project_id: str, animation_id: str, data_dir: Path | None = None):
        base = data_dir if data_dir is not None else get_settings().data_dir
        self.animation_dir = Path(base) / "projects" / project_id / "animations" / animation_id

    @property
    def source_image(self) -> Path:
        return self.animation_dir / "source.png"

    @property
    def raw_video(self) -> Path:
        return self.animation_dir / "raw" / "video.mp4"

    @property
    def stages_dir(self) -> Path:
        return self.animation_dir / "stages"

    def stage_dir(self, order: int, stage: str) -> Path:
        return self.stages_dir / f"{order:02d}_{stage}"

    @property
    def edits_dir(self) -> Path:
        return self.animation_dir / "edits"

    def edit_dir(self, version: int) -> Path:
        return self.edits_dir / f"edit_{version:04d}"

    def list_edit_versions(self) -> list[int]:
        """Versiones de edición VÁLIDAS: dirs ``edit_NNNN`` con ``frameset.json``.

        Las copias parciales (sin manifiesto) y los dirs temporales se ignoran
        para que nunca se elijan como versión de trabajo.
        """
        if not self.edits_dir.exists():
            return []
        versions = []
        for d in self.edits_dir.iterdir():
            if not d.is_dir():
                continue
            m = _EDIT_DIR_RE.match(d.name)
            if m and (d / MANIFEST_NAME).exists():
                versions.append(int(m.group(1)))
        return sorted(versions)

    def next_edit_version(self) -> int:
        """Siguiente número de versión libre.

        Considera TODOS los dirs ``edit_*`` (válidos, parciales o temporales)
        para no colisionar con basura dejada por fallos previos.
        """
        if not self.edits_dir.exists():
            return 1
        highest = 0
        for d in self.edits_dir.iterdir():
            if not d.is_dir():
                continue
            m = _EDIT_DIR_PREFIX_RE.match(d.name)
            if m:
                highest = max(highest, int(m.group(1)))
        return highest + 1

    @property
    def exports_dir(self) -> Path:
        return self.animation_dir / "exports"

    def export_dir(self, export_id: str) -> Path:
        return self.exports_dir / export_id

    def latest_stage_dir(self) -> Path | None:
        """Directorio de etapa COMPLETO más avanzado (ordenado por prefijo NN_).

        Solo cuentan los dirs con ``frameset.json``: un stage dir parcial (de
        una etapa en curso o fallida) se salta en favor de la etapa completa
        anterior. Los dirs temporales ``*.tmp`` se ignoran siempre.
        """
        if not self.stages_dir.exists():
            return None
        dirs = sorted(
            d for d in self.stages_dir.iterdir()
            if d.is_dir() and not d.name.endswith(".tmp")
        )
        for d in reversed(dirs):
            if (d / MANIFEST_NAME).exists():
                return d
        return None

    def working_dir(self) -> Path | None:
        """FrameSet de trabajo actual: última edición si existe, si no la última etapa."""
        versions = self.list_edit_versions()
        if versions:
            return self.edit_dir(versions[-1])
        return self.latest_stage_dir()
