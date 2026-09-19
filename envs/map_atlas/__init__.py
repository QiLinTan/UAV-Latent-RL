"""Versioned map specifications and PyBullet runtime adapters.

The atlas deliberately sits below the navigation policy: map selection changes
geometry and task endpoints, but does not change the upper-policy action
interface or the ReferencePacket -> DSLPID control chain.
"""

from .runtime import MapRuntime
from .spec import (
    MapCatalog,
    MapSpec,
    MapValidationReport,
    StaticObstacleSpec,
    coerce_map_spec,
)

__all__ = [
    "MapCatalog",
    "MapRuntime",
    "MapSpec",
    "MapValidationReport",
    "StaticObstacleSpec",
    "coerce_map_spec",
]
