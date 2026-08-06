"""El editor estático: archivos presentes, autocontenidos y contra la API del spec."""

from __future__ import annotations

import re
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parents[1] / "sprite_pipeline" / "editor" / "static"
FILES = ("index.html", "app.js", "style.css")


def _read(name: str) -> str:
    path = STATIC_DIR / name
    assert path.is_file(), f"Falta el archivo estático {path}"
    text = path.read_text(encoding="utf-8")
    assert text.strip(), f"{name} está vacío"
    return text


def test_archivos_existen_y_no_estan_vacios():
    for name in FILES:
        _read(name)


def test_index_referencia_assets_como_editor_static():
    html = _read("index.html")
    assert "/editor/static/style.css" in html, "index.html no enlaza style.css vía /editor/static/"
    assert "/editor/static/app.js" in html, "index.html no enlaza app.js vía /editor/static/"


def test_app_js_usa_los_endpoints_clave():
    js = _read("app.js")
    for endpoint in ("/frameset", "/edits", "/undo", "/export"):
        assert endpoint in js, f"app.js no usa el endpoint {endpoint}"


def test_app_js_contiene_todas_las_ops_de_edicion():
    js = _read("app.js")
    for op in ("delete", "duplicate", "reorder", "set_duration", "interpolate", "retouch"):
        pattern = rf"op\s*:\s*[\"']{op}[\"']"
        assert re.search(pattern, js), f"app.js no envía la op {op!r}"


def test_sin_urls_externas_en_ningun_archivo():
    external = re.compile(r"https?://", re.IGNORECASE)
    for name in FILES:
        match = external.search(_read(name))
        assert match is None, f"{name} contiene una URL externa ({match.group(0)}...)"
