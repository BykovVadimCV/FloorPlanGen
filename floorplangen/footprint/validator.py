"""Footprint validator (DESIGN §2.5)."""
from __future__ import annotations

import math

from shapely.geometry import Polygon

MIN_INTERIOR_WIDTH_PX_AT_512 = 80
MIN_ANGLE_DEG = 55.0
MIN_AREA_FRACTION = 0.28  # at least 28% of the bounding box area


def _scale_min_width(image_size: tuple[int, int]) -> float:
    # Scale min-interior-width threshold to canvas size (§9.7.3 scaling convention)
    return MIN_INTERIOR_WIDTH_PX_AT_512 * min(image_size) / 512.0


def _interior_angles(poly: Polygon) -> list[float]:
    coords = list(poly.exterior.coords)
    # polygon ring is closed — first == last; drop last
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    n = len(coords)
    angles: list[float] = []
    for i in range(n):
        p = coords[(i - 1) % n]
        v = coords[i]
        q = coords[(i + 1) % n]
        ax, ay = p[0] - v[0], p[1] - v[1]
        bx, by = q[0] - v[0], q[1] - v[1]
        na = math.hypot(ax, ay)
        nb = math.hypot(bx, by)
        if na == 0 or nb == 0:
            continue
        cos_ = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
        angles.append(math.degrees(math.acos(cos_)))
    return angles


def validate(candidate, image_size: tuple[int, int]) -> bool:
    """Return True if candidate polygon passes all five checks."""
    if candidate is None or candidate.is_empty:
        return False
    if not isinstance(candidate, Polygon):
        return False
    if not candidate.is_valid:
        return False
    if len(candidate.interiors) != 0:
        return False

    # Check 3 — min interior width via minimum rotated rectangle side length
    try:
        mrr = candidate.minimum_rotated_rectangle
        if not isinstance(mrr, Polygon):
            return False
        mrr_coords = list(mrr.exterior.coords)[:4]
        side_lens = []
        for i in range(4):
            x1, y1 = mrr_coords[i]
            x2, y2 = mrr_coords[(i + 1) % 4]
            side_lens.append(math.hypot(x2 - x1, y2 - y1))
        min_width = min(side_lens)
        if min_width < _scale_min_width(image_size):
            return False
    except Exception:
        return False

    # Check 4 — angle guard (avoid razor-thin re-entrants)
    angles = _interior_angles(candidate)
    if angles and min(angles) < MIN_ANGLE_DEG:
        return False

    # Check 5 — minimum area
    minx, miny, maxx, maxy = candidate.bounds
    bbox_area = max(1.0, (maxx - minx) * (maxy - miny))
    if candidate.area / bbox_area < MIN_AREA_FRACTION:
        return False

    return True
