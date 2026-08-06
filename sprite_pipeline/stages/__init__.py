"""Etapas de procesamiento del pipeline.

Las etapas se registran de forma perezosa: ``get_stage(name)`` importa
``sprite_pipeline.stages.<name>`` bajo demanda (ver ``base.get_stage``).
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
