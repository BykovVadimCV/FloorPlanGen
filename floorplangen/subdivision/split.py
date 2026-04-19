"""Half-plane intersection/difference splitter (DESIGN §3.4)."""
from __future__ import annotations

import math

from shapely.geometry import MultiPolygon, Polygon, box


def half_plane_split(poly: Polygon,
                     origin: tuple[float, float],
                     angle_rad: float) -> tuple[Polygon | None, Polygon | None]:
    """Split `poly` by the line through origin at angle_rad.

    Returns (left_side, right_side) as Polygons. Each side may be None if empty.
    The "left" side is where the half-plane normal (rotate angle by +90°) points.
    """
    minx, miny, maxx, maxy = poly.bounds
    diag = math.hypot(maxx - minx, maxy - miny) * 4.0 + 32.0
    cx, cy = origin

    # Axis direction (along the split line) and normal
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    nx = -dy
    ny = dx

    # Build a very large rectangle on the "left" side of the split line
    far = diag
    corners = [
        (cx + dx * far + nx * 0.0,        cy + dy * far + ny * 0.0),
        (cx + dx * far + nx * far,        cy + dy * far + ny * far),
        (cx - dx * far + nx * far,        cy - dy * far + ny * far),
        (cx - dx * far + nx * 0.0,        cy - dy * far + ny * 0.0),
    ]
    left_half = Polygon(corners)
    if not left_half.is_valid:
        left_half = left_half.buffer(0)

    try:
        left_side = poly.intersection(left_half)
        right_side = poly.difference(left_half)
    except Exception:
        return None, None

    def _largest(g):
        if g is None or g.is_empty:
            return None
        if isinstance(g, Polygon):
            return g
        if isinstance(g, MultiPolygon):
            polys = [p for p in g.geoms if not p.is_empty]
            return max(polys, key=lambda p: p.area) if polys else None
        try:
            from shapely import get_parts
            polys = [p for p in get_parts(g) if isinstance(p, Polygon) and not p.is_empty]
            return max(polys, key=lambda p: p.area) if polys else None
        except Exception:
            return None

    return _largest(left_side), _largest(right_side)
