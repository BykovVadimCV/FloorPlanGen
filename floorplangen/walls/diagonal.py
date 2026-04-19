"""Diagonal wall insertion (DESIGN §4).

Conservative implementation: at most 1–2 diagonal walls per image, attached
externally to convex corners (bevelled vertices are ideal). All diagonal walls
receive no_annotate=True so the opening placement in Ch. 6 skips them (§4.7).
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString

from ..config import GeneratorConfig
from ..types import Footprint, Room, WallGraph, WallSegment


DIAG_PER_IMAGE_PROB = 0.04  # ~4% of images get a diagonal wall


def inject_diagonal_walls(graph: WallGraph, footprint: Footprint,
                          rooms: list[Room], cfg: GeneratorConfig,
                          rng: np.random.Generator, era: str,
                          interior_thk: float, exterior_thk: float,
                          image_style: str) -> None:
    if rng.random() > DIAG_PER_IMAGE_PROB:
        return
    # Attach to the longest external edge — re-use it with an angular jitter
    coords = list(footprint.polygon.exterior.coords)[:-1]
    if len(coords) < 3:
        return

    n = len(coords)
    # Find the longest edge
    lengths = []
    for i in range(n):
        a = coords[i]
        b = coords[(i + 1) % n]
        lengths.append((math.hypot(b[0] - a[0], b[1] - a[1]), i))
    lengths.sort(reverse=True)
    _, idx = lengths[0]
    a = coords[idx]
    b = coords[(idx + 1) % n]
    # Internal diagonal: pick a starting point along this edge,
    # angled into the interior by 20–50 degrees
    t0 = float(rng.uniform(0.25, 0.55))
    t1 = float(rng.uniform(0.70, 0.92))
    p0 = (a[0] + t0 * (b[0] - a[0]), a[1] + t0 * (b[1] - a[1]))
    # Inward normal
    ex, ey = b[0] - a[0], b[1] - a[1]
    elen = math.hypot(ex, ey) or 1.0
    nx, ny = -ey / elen, ex / elen
    # Test interior side
    cx, cy = footprint.polygon.centroid.x, footprint.polygon.centroid.y
    if (cx - p0[0]) * nx + (cy - p0[1]) * ny < 0:
        nx, ny = -nx, -ny

    length = float(rng.uniform(0.15, 0.28)) * elen
    angle_deg = float(rng.choice([30, 45, 60, 120, 135, 150]))
    theta = math.radians(angle_deg)
    # Ray direction: rotate edge direction by theta, into the interior
    dx = ex / elen * math.cos(theta) - ey / elen * math.sin(theta)
    dy = ex / elen * math.sin(theta) + ey / elen * math.cos(theta)
    # Ensure the ray points inward
    if dx * nx + dy * ny < 0:
        dx, dy = -dx, -dy
    p1 = (p0[0] + dx * length, p0[1] + dy * length)

    # Clip to footprint
    line = LineString([p0, p1])
    clipped = line.intersection(footprint.polygon)
    if clipped.is_empty or clipped.geom_type != "LineString":
        return
    if clipped.length < 20.0:
        return

    seg = WallSegment(
        centreline=clipped,
        thickness_px=interior_thk,
        style=image_style,
        is_exterior=False,
        is_diagonal=True,
        angle_deg=angle_deg,
        no_annotate=True,
    )
    graph.segments.append(seg)
