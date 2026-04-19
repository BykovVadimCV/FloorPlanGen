"""FloorPlanGen — synthetic floor-plan generator (see DESIGN.md)."""
from .api import generate_sample
from .config import GeneratorConfig
from .types import (
    GeneratorOutput,
    Footprint,
    Room,
    WallSegment,
    WallGraph,
    Opening,
    PlacedIcon,
    Annotation,
    PIXEL_VALUES,
    CLASS_PIXEL_MAP,
)

__all__ = [
    "generate_sample", "GeneratorConfig", "GeneratorOutput",
    "Footprint", "Room", "WallSegment", "WallGraph",
    "Opening", "PlacedIcon", "Annotation",
    "PIXEL_VALUES", "CLASS_PIXEL_MAP",
]
__version__ = "0.1.0"
