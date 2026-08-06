"""Proveedores de generación de video (interfaz adaptadora desacoplada).

Importar este paquete registra los proveedores disponibles en PROVIDER_REGISTRY.
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

from sprite_pipeline.providers import mock, runway, luma  # noqa: F401,E402
