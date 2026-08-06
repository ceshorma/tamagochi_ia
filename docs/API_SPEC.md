# Especificación de contratos internos y API REST

Este documento es el contrato de coordinación entre módulos. Todo módulo se
implementa **contra lo que dice aquí y contra los contratos ya escritos en**
`sprite_pipeline/models.py`, `sprite_pipeline/stages/base.py`,
`sprite_pipeline/providers/base.py`, `sprite_pipeline/storage.py`,
`sprite_pipeline/config.py` y `sprite_pipeline/sample.py`.

## Convenciones globales

- Imágenes en memoria: numpy **RGBA uint8** `(h, w, 4)`. I/O con
  `load_frame_rgba` / `save_frame_rgba` de `stages/base.py`.
- Un FrameSet vive en un directorio: `frameset.json` + `frame_NNNN.png`.
- Ninguna etapa muta su directorio de entrada.
- Todo determinista con la misma semilla; tests sin red y sin GPU.

## Módulos y sus contratos

### `sprite_pipeline/providers/mock.py` — proveedor mock
- `class MockProvider(VideoGenerationProvider)`, `name = "mock"`, registrado con `@register_provider`.
- `generate(req, workdir)` produce `workdir/video.mp4` animando la criatura de
  `sample.py` (si `req.image_path` no es la criatura, igual usa la imagen dada
  compuesta con transformaciones de pose genéricas: bob/squash por acción).
- El video DEBE simular los defectos reales que el pipeline corrige:
  fondo casi-uniforme con gradiente y ruido leve, deriva de posición/escala
  entre cuadros (determinista por semilla), y con `req.prompt_extra`
  conteniendo `"inject_anomaly"` un cuadro corrupto (ruido/deformación).
- Duración/fps del video según `req.duration_s` / `req.fps`.
- Escritura de video: `imageio.mimwrite(..., fps=...)` (plugin ffmpeg) o cv2.VideoWriter.

### `sprite_pipeline/providers/runway.py` y `luma.py` — proveedores reales
- `RunwayProvider` (`name="runway"`), `LumaProvider` (`name="luma"`).
- Constructor acepta `api_key: str | None = None` (default: settings).
- `generate()` = submit + polling HTTP con `httpx` + descarga del mp4 a `workdir`.
- Sin clave → `ProviderConfigError` (definir en el mismo módulo, heredando de RuntimeError).
- Tests con `httpx.MockTransport`; jamás red real.

### `sprite_pipeline/extract.py` — extracción de fotogramas
- `extract_frames(video_path: Path, out_dir: Path, fps: int = 10, dedupe_threshold: float = 0.995, meta: dict | None = None) -> FrameSet`
- Lee el video (cv2.VideoCapture; fallback imageio), muestrea a `fps` efectivo,
  descarta cuadros consecutivos casi idénticos (similitud > threshold, medida
  simple tipo correlación/diff normalizado), guarda PNGs RGBA y el manifiesto
  con `stage="raw"`, `duration_ms=1000/fps`, `meta` fusionado (provider, prompt,
  action, source_image, video original).

### `sprite_pipeline/stages/preprocess.py` — etapa `preprocess`
- Normaliza: canvas cuadrado de `params.get("work_size", settings.work_size)`,
  centra el contenido detectando bounding box del sujeto (diferencia contra el
  color de fondo dominante estimado en los bordes), reescala manteniendo aspecto
  con margen configurable (`params.get("margin", 0.08)`).
- Anota en `meta`: `size: [w, h]`, `bg_color_estimate: [r,g,b]`.

### `sprite_pipeline/stages/background.py` — etapa `background`
- Segmenta al personaje y produce alfa limpio. Método por defecto `"auto"`:
  color de fondo estimado (de `meta.bg_color_estimate` o bordes) + flood fill
  desde los bordes + operaciones morfológicas + suavizado del borde alfa.
- Consistencia temporal: la máscara del cuadro anterior se usa como prior
  (mezcla/estabilización) para evitar parpadeo de silueta.
- `params["method"]` reservado para métodos futuros ("rembg", "sam2") — si se
  pide uno no disponible, lanzar `ValueError` con mensaje claro.

### `sprite_pipeline/stages/align.py` — etapa `align`
- Registra cada cuadro contra un ancla común para eliminar deriva de
  posición/escala. `params.get("anchor", "feet")`: `"feet"` (centro-x del bbox
  del alfa, borde inferior fijo) o `"center"` (centroide del alfa).
- Escala: normaliza el alto del bbox al alto mediano de la secuencia
  (tolerancia `params.get("scale_tolerance", 0.05)`: si la desviación es menor, no reescala).
- Guarda en `Frame.transform` el `{dx, dy, scale}` aplicado.

### `sprite_pipeline/stages/anomaly.py` — etapa `anomaly`
- NO modifica píxeles: solo puntúa y marca. Scores por cuadro en `Frame.scores`:
  - `identity`: similitud con la imagen fuente (`meta.source_image` si existe;
    si no, contra el cuadro mediano de la secuencia). Histograma HSV + tamaño de silueta.
  - `coherence`: similitud con vecinos (diff RGBA normalizado sobre la unión de alfas).
  - `quality`: score compuesto 0..1.
- Cuadros con `quality < params.get("threshold", 0.55)` reciben flag `"anomaly"`
  y `"review"`. Nunca eliminar cuadros.

### `sprite_pipeline/stages/smooth.py` — etapa `smooth`
- Estabilización temporal sin destruir detalle:
  - Normaliza brillo/color global por cuadro hacia la mediana de la secuencia
    (ganancia limitada, p. ej. ±10 %).
  - Suaviza el borde del alfa (feather leve, `params.get("feather", 1.5)` px).
  - Uniformiza `duration_ms` (media) salvo `params.get("keep_timing", False)`.

### `sprite_pipeline/ops/` — operaciones del editor
- `interpolate.py`: `interpolate_frames(a: np.ndarray, b: np.ndarray, t: float = 0.5) -> np.ndarray`
  RGBA→RGBA. Flujo óptico Farneback sobre luma con warp de ambos extremos y
  mezcla; el alfa se interpola igual. Fallback a crossfade si el flujo falla.
- `retouch.py`: `retouch_region(img: np.ndarray, mask: np.ndarray, neighbors: list[np.ndarray]) -> np.ndarray`
  `mask` uint8 (255 = regenerar). Estrategia: si hay vecinos, parche desde el
  promedio alineado de vecinos con blending de bordes; fallback cv2.inpaint
  sobre RGB + reconstrucción de alfa.
- `edits.py`: gestión de versiones de edición sobre `Paths.edits_dir`:
  - `ensure_edit_session(paths: Paths) -> Path` — si no hay ediciones, copia el
    último stage a `edit_0001`; devuelve el dir de la versión actual.
  - `apply_edit(paths: Paths, op: dict) -> FrameSet` — crea `edit_{n+1}` con el
    resultado de la operación (ver payloads de `/edits` abajo) y lo devuelve.
  - `undo(paths: Paths) -> FrameSet | None` — elimina la última versión si hay más de una.

### `sprite_pipeline/export.py` — exportación
- `export_animation(frameset_dir: Path, out_dir: Path, formats: list[str] | None = None, columns: int | None = None, padding: int = 2, scale: int = 1) -> dict`
- `formats` default `["sheet", "pngseq", "gif"]`; también soporta `"godot"`.
  - `sheet`: `sheet.png` (grid, `columns` default ≈ cuadrado) + `sheet.json`
    dialecto TexturePacker "frames hash" con `frame`, `sourceSize`, `duration`
    (compatible Phaser/PixiJS).
  - `pngseq`: `frames/frame_0000.png`, ...
  - `gif`: `preview.gif` respetando `duration_ms` por cuadro (fondo transparente→checkerboard NO: usar disposal correcto y transparencia).
  - `godot`: `spriteframes.tres` (recurso SpriteFrames texto de Godot 4 refiriendo `sheet.png` por regiones).
- Devuelve `{"files": [rutas relativas], "sheet": {...meta...}}`.

### `sprite_pipeline/orchestrator.py` — orquestación
- `PIPELINE_ORDER = ["preprocess", "background", "align", "anomaly", "smooth"]`
- `run_animation_pipeline(store: Store, animation_id: str, image_path: Path, params: dict | None = None) -> dict`
  Síncrono: copia imagen fuente → provider.generate → extract_frames (stage dir
  `00_raw`) → etapas en orden (`01_preprocess`...) → status final. Actualiza
  `animations.status/current_stage` y el job (`progress` 0..1) en cada paso.
  Ante excepción: status `failed`, `error` con mensaje, re-lanza solo si `params.get("raise_errors")`.
- `class Orchestrator`: ThreadPoolExecutor(max_workers=2);
  `submit_generation(animation_id, image_path, params) -> job dict`;
  `get_orchestrator(store) -> Orchestrator` singleton por proceso.

### `sprite_pipeline/cli.py` — CLI
- Comandos argparse (`main(argv=None)`):
  - `generate --image PATH --action TXT [--name] [--project] [--provider] [--fps] [--duration-s] [--seed] [--data-dir]` → corre síncrono, imprime rutas.
  - `export --animation AID [--formats sheet,pngseq,gif,godot] [--columns] [--scale]`
  - `serve [--host 127.0.0.1] [--port 8000]` → uvicorn del API.
  - `demo [--out examples/] [--data-dir]` → genera idle/comer/dormir de la criatura de muestra end-to-end y exporta.

## API REST (`sprite_pipeline/api/app.py`)

FastAPI con factory `create_app(store: Store | None = None) -> FastAPI` y
`app = create_app()` a nivel de módulo. Prefijo `/api`. Errores → JSON
`{"detail": ...}` con códigos 404/400/409 razonables.

| Método y ruta | Descripción |
|---|---|
| `GET /api/health` | `{"status":"ok"}` |
| `POST /api/projects` | body JSON `{name}` → proyecto |
| `GET /api/projects` | lista |
| `GET /api/projects/{pid}` | detalle |
| `GET /api/projects/{pid}/animations` | lista de animaciones del proyecto |
| `POST /api/projects/{pid}/animations` | multipart: `file` (imagen), `name`, `action`, opc. `provider`, `fps`, `duration_s`, `seed` → `{animation, job}`; encola en el Orchestrator |
| `GET /api/animations/{aid}` | fila + `stages` presentes + resumen de flags |
| `GET /api/animations/{aid}/frameset?version=working` | manifiesto del FrameSet. `version`: `working` (default, últimas ediciones o última etapa), `auto` (última etapa), o nombre de etapa (`raw`, `preprocess`, ...) |
| `GET /api/animations/{aid}/frames/{file}?version=working` | PNG del cuadro por nombre de archivo del manifiesto |
| `POST /api/animations/{aid}/edits` | body JSON con la operación (abajo) → frameset resultante |
| `POST /api/animations/{aid}/undo` | deshace la última edición → frameset resultante |
| `GET /api/animations/{aid}/preview.gif?version=working` | GIF generado al vuelo |
| `POST /api/animations/{aid}/export` | body `{formats?, columns?, padding?, scale?}` → `{export_id, files}` |
| `GET /api/animations/{aid}/exports/{export_id}/{path}` | descarga de artefacto exportado |
| `GET /api/jobs/{jid}` | estado del job |
| `GET /editor/{aid}` | página HTML del editor (inyecta el `aid`) |
| `GET /editor/static/{file}` | assets estáticos del editor |

### Payloads de `/edits` (`op` obligatorio)

```json
{"op": "delete",       "index": 3}
{"op": "duplicate",    "index": 3}
{"op": "reorder",      "order": [0, 2, 1, 3]}
{"op": "set_duration", "index": 3, "duration_ms": 160}
{"op": "interpolate",  "after_index": 3}
{"op": "retouch",      "index": 3, "mask_png_base64": "..."}
```

- `interpolate` inserta el cuadro sintetizado entre `after_index` y el
  siguiente (envolvente: si es el último, entre último y primero), con
  `source="interpolated"` y pasa el detector de anomalías sobre ese cuadro si
  es barato (opcional).
- `retouch` recibe la máscara como PNG base64 en escala de grises del mismo
  tamaño del cuadro (blanco = regenerar), aplica `ops.retouch.retouch_region`
  con los cuadros vecinos, `source="retouched"`.

## Editor web (`sprite_pipeline/editor/static/`)

SPA sin build ni CDNs (vanilla JS + CSS, autocontenida): `index.html`,
`app.js`, `style.css`. Servida por el API en `/editor/{aid}`.

Funcionalidad mínima obligatoria:
- Preview en bucle (canvas) con play/pausa y control de velocidad (0.25x–2x).
- Timeline de miniaturas; cuadros con flag `anomaly`/`review` resaltados.
- Selección de cuadro: eliminar, duplicar, duración (input ms), interpolar
  después del cuadro, deshacer global.
- Reordenar por drag & drop en el timeline.
- Retoque: vista ampliada del cuadro con pincel para pintar máscara y botón
  aplicar (envía `retouch` con la máscara en base64).
- Botón exportar (formats por checkboxes) que muestra los enlaces de descarga.
- Todo contra la API REST de arriba; refrescar el frameset tras cada operación.
