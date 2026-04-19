"""Mask write pipeline — 7-level priority stack (DESIGN §11.2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from shapely.geometry import Polygon

from ..types import (
    Annotation,
    Footprint,
    Opening,
    PIXEL_BACKGROUND,
    PIXEL_DOOR,
    PIXEL_FURNITURE,
    PIXEL_WALL,
    PIXEL_WINDOW,
    PlacedIcon,
    Room,
    WallGraph,
    CLASS_PIXEL_MAP,
)
from .fill import fill_polygon


@dataclass
class MaskInputs:
    footprint: Footprint
    rooms: list[Room]
    walls: WallGraph
    openings: list[Opening]
    icons: list[PlacedIcon]
    annotations: list[Annotation]


def write_mask(image_size: tuple[int, int], inputs: MaskInputs) -> np.ndarray:
    """Return a (H, W) uint8 mask in {0,64,128,192,255} per §11.2.

    Priority stack:
        1. Background fill                        (class 0, pixel 0)
        2. Footprint interior                     (class 4, pixel 255)
        3. Room fills                             (class 4, pixel 255)
        4. Wall bands (with opening carve-out)    (class 1, pixel 64)
        5. Openings (windows / doors)             (class 2 / 3, pixel 128 / 192)
        6. Icons and furniture                    (class 4, pixel 255)
        7. Dimension annotations                  (class 0, pixel 0)
    """
    h, w = image_size
    mask = np.full((h, w), PIXEL_BACKGROUND, dtype=np.uint8)

    # Priority 2 — footprint interior (background inside building, not furniture)
    # Room interiors are class 0 (background); only walls/openings/icons differ.
    # (Footprint fill removed — entire image starts as background already)

    # Priority 3 — room fills (no-op: interior stays class 0)

    # Priority 4 — walls (with opening carve-out, §11.5)
    opening_union = None
    if inputs.openings:
        from shapely.ops import unary_union
        opening_union = unary_union([op.class_polygon for op in inputs.openings])
    for seg in inputs.walls.segments:
        band = seg.ensure_band()
        if opening_union is not None and not opening_union.is_empty:
            try:
                band = band.difference(opening_union)
            except Exception:
                pass
        fill_polygon(mask, band, PIXEL_WALL)

    # Priority 5 — openings
    for op in inputs.openings:
        value = PIXEL_WINDOW if op.kind == "window" else PIXEL_DOOR
        fill_polygon(mask, op.class_polygon, value)

    # Priority 6 — icons / furniture
    for icon in inputs.icons:
        fill_polygon(mask, icon.footprint_polygon, PIXEL_FURNITURE)

    # Priority 7 — dimension annotations (back to class 0)
    for ann in inputs.annotations:
        for p in ann.polygons:
            fill_polygon(mask, p, PIXEL_BACKGROUND)

    return mask


def encode_class_mask(mask: np.ndarray) -> np.ndarray:
    """§11.8 — compress {0,64,128,192,255} into contiguous {0..4}."""
    out = np.zeros_like(mask)
    for pixel_val, class_id in CLASS_PIXEL_MAP.items():
        out[mask == pixel_val] = class_id
    return out
