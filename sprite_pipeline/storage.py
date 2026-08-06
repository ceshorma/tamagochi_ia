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

import sqlite3
import threading
import time
import uuid
from pathlib import Path

from sprite_pipeline.config import get_settings

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
        if not self.edits_dir.exists():
            return []
        versions = []
        for d in self.edits_dir.iterdir():
            if d.is_dir() and d.name.startswith("edit_"):
                try:
                    versions.append(int(d.name.split("_")[1]))
                except (IndexError, ValueError):
                    continue
        return sorted(versions)

    @property
    def exports_dir(self) -> Path:
        return self.animation_dir / "exports"

    def export_dir(self, export_id: str) -> Path:
        return self.exports_dir / export_id

    def latest_stage_dir(self) -> Path | None:
        """Directorio de etapa más avanzado que exista (ordenado por prefijo NN_)."""
        if not self.stages_dir.exists():
            return None
        dirs = sorted(d for d in self.stages_dir.iterdir() if d.is_dir())
        return dirs[-1] if dirs else None

    def working_dir(self) -> Path | None:
        """FrameSet de trabajo actual: última edición si existe, si no la última etapa."""
        versions = self.list_edit_versions()
        if versions:
            return self.edit_dir(versions[-1])
        return self.latest_stage_dir()
