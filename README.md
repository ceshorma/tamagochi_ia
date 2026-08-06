# AI Sprite Pipeline

Convierte una imagen de un personaje en un sprite animado listo para videojuegos, minimizando la edición manual: generación de video con IA (proveedor desacoplado) → extracción de fotogramas → limpieza y alineación automáticas → editor inteligente cuadro a cuadro → exportación a formatos de motor de juego.

Primer caso de uso: animar la criatura del proyecto **tamagochi_ia**.

## Estado

Fases 1 y 2 del [roadmap](docs/PLAN.md#5-roadmap-por-fases) implementadas: pipeline automático end-to-end + editor web + API REST + CLI. El proveedor `mock` permite desarrollar y probar todo el pipeline sin claves de API; los adaptadores de Runway y Luma están implementados pero **no validados contra los servicios reales** (requieren `RUNWAY_API_KEY` / `LUMA_API_KEY`).

## Instalación

```bash
pip install -e ".[dev]"
```

## Uso rápido

```bash
# Demo completa: genera idle/eat/sleep de la criatura de muestra y exporta
sprite-pipeline demo --out examples

# Pipeline con tu propia imagen
sprite-pipeline generate --image mi_personaje.png --action "idle" --provider mock

# Exportar una animación (sprite sheet + PNG seq + GIF; también godot)
sprite-pipeline export --animation <ID> --formats sheet,pngseq,gif,godot

# Servidor API + editor web
sprite-pipeline serve --port 8000
# → editor en http://localhost:8000/editor/<animation_id>
```

Variables de entorno: `SPRITE_DATA_DIR` (datos, default `./data`), `SPRITE_PROVIDER` (`mock`|`runway`|`luma`), `SPRITE_WORK_SIZE` (canvas de trabajo, default 256), `RUNWAY_API_KEY`, `LUMA_API_KEY`.

## Qué hace el pipeline

1. **Generación** de video desde la imagen + acción vía proveedor desacoplado (`providers/`).
2. **Extracción** de fotogramas con deduplicación (`extract.py`).
3. **Preprocesado**: normalización de canvas y encuadre estable por unión de bboxes (`stages/preprocess.py`).
4. **Eliminación de fondo** con consistencia temporal de silueta (`stages/background.py`).
5. **Alineación** contra ancla común (pies/centro) eliminando deriva de posición y escala (`stages/align.py`).
6. **Detección de anomalías**: scoring de identidad y coherencia por cuadro; marca, nunca borra (`stages/anomaly.py`).
7. **Suavizado temporal**: estabilización de brillo, feather del alfa, timing uniforme (`stages/smooth.py`).
8. **Editor inteligente** (web): eliminar, duplicar, reordenar, duración por cuadro, interpolación por flujo óptico, retoque por pincel con vecinos como referencia, deshacer (`editor/`, `ops/`).
9. **Exportación**: sprite sheet + JSON dialecto TexturePacker/Aseprite (compatible Phaser/PixiJS), secuencia PNG, GIF y SpriteFrames de Godot 4 (`export.py`).

## Documentación

- 📋 [Plan fundacional del producto](docs/PLAN.md) — visión, arquitectura, historias de usuario, roadmap, riesgos.
- 🔌 [Especificación de contratos y API REST](docs/API_SPEC.md).

## Tests

```bash
python -m pytest tests/ -q
```
