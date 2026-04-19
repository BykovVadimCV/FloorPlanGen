"""Mask validation invariants (DESIGN §11.9)."""
from __future__ import annotations

import numpy as np

from ..types import (
    PIXEL_BACKGROUND,
    PIXEL_DOOR,
    PIXEL_FURNITURE,
    PIXEL_WALL,
    PIXEL_WINDOW,
    PIXEL_VALUES,
    Footprint,
)
from .fill import rasterize_polygon


def validate_mask(mask: np.ndarray, footprint: Footprint | None = None,
                  require_wall_fraction: bool = True) -> list[str]:
    errors: list[str] = []

    # 1. Legal pixel values only
    unique = set(int(v) for v in np.unique(mask))
    illegal = unique - PIXEL_VALUES
    if illegal:
        errors.append(f"Illegal pixel values: {sorted(illegal)}")

    # 2. No wall/window/door outside footprint
    if footprint is not None:
        interior = rasterize_polygon(footprint.polygon, mask.shape)
        for val, name in [(PIXEL_WALL, "wall"),
                          (PIXEL_WINDOW, "window"),
                          (PIXEL_DOOR, "door")]:
            outside = int(np.sum((mask == val) & ~interior))
            if outside > 0:
                errors.append(f"{outside} {name} pixels outside footprint")

    # 3. Zero overlap wall vs. window/door (cannot happen by construction; sanity)
    wall_pixels = mask == PIXEL_WALL
    for val, name in [(PIXEL_WINDOW, "window"), (PIXEL_DOOR, "door")]:
        overlap = int(np.sum(wall_pixels & (mask == val)))
        if overlap > 0:
            errors.append(f"{overlap} pixels simultaneously wall and {name}")

    # 4. Wall coverage sanity
    if require_wall_fraction:
        frac = float(np.sum(mask == PIXEL_WALL)) / float(mask.size)
        if frac < 0.005:
            errors.append("Wall class covers < 0.5% of image — suspicious")

    return errors
