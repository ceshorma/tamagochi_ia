"""Proveedores de generación de video (interfaz adaptadora desacoplada).

Los proveedores se registran de forma perezosa: ``get_provider(name)`` importa
``sprite_pipeline.providers.<name>`` bajo demanda (ver ``base.get_provider``).
"""

from sprite_pipeline.providers.base import (  # noqa: F401
    PROVIDER_REGISTRY,
    GenerationRequest,
    GenerationResult,
    VideoGenerationProvider,
    build_prompt,
    get_provider,
    register_provider,
)
