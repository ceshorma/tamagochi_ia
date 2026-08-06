"""Etapas de procesamiento del pipeline.

Importar este paquete registra todas las etapas disponibles en STAGE_REGISTRY.
"""

from sprite_pipeline.stages.base import (  # noqa: F401
    STAGE_REGISTRY,
    Stage,
    get_stage,
    load_frame_rgba,
    register_stage,
    run_stage,
    save_frame_rgba,
)

# Importa los módulos de etapas para que se registren.
from sprite_pipeline.stages import (  # noqa: F401,E402
    preprocess,
    background,
    align,
    anomaly,
    smooth,
)
