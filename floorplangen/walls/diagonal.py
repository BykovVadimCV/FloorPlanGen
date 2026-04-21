"""Diagonal wall insertion.

Conservative: at most one diagonal wall per image, attached to a long exterior
edge and angled inward. Diagonal walls are flagged `no_annotate=True` so the
opening placement (Chapter 6) skips them per the spec — openings sit on
axis-aligned walls only.
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString

from ..config import GeneratorConfig
from ..linework import LineworkSpec
from ..types import Footprint, Room, WallGraph, WallSegment
from .thickness import sample_wall_thickness_px


DIAG_PER_IMAGE_PROB = 0.04


def inject_diagonal_walls(graph: WallGraph, footprint: Footprint,
                          rooms: list[Room], cfg: GeneratorConfig,
                          rng: np.random.Generator,
                          image_style: str,
                          spec: LineworkSpec) -> None:
    if rng.random() > DIAG_PER_IMAGE_PROB:
        return
    coords = list(footprint.polygon.exterior.coords)[:-1]
    if len(coords) < 3:
        return

    n = len(coords)
    lengths = []
    for i in range(n):
        a = coords[i]
        b = coords[(i + 1) % n]
        lengths.append((math.hypot(b[0] - a[0], b[1] - a[1]), i))
    lengths.sort(reverse=True)
    _, idx = lengths[0]
    a = coords[idx]
    b = coords[(idx + 1) % n]

    t0 = float(rng.uniform(0.25, 0.55))
    p0 = (a[0] + t0 * (b[0] - a[0]), a[1] + t0 * (b[1] - a[1]))
    ex, ey = b[0] - a[0], b[1] - a[1]
    elen = math.hypot(ex, ey) or 1.0
    nx, ny = -ey / elen, ex / elen
    cx, cy = footprint.polygon.centroid.x, footprint.polygon.centroid.y
    if (cx - p0[0]) * nx + (cy - p0[1]) * ny < 0:
        nx, ny = -nx, -ny

    length = float(rng.uniform(0.15, 0.28)) * elen
    angle_deg = float(rng.choice([30, 45, 60, 120, 135, 150]))
    theta = math.radians(angle_deg)
    dx = ex / elen * math.cos(theta) - ey / elen * math.sin(theta)
    dy = ex / elen * math.sin(theta) + ey / elen * math.cos(theta)
    if dx * nx + dy * ny < 0:
        dx, dy = -dx, -dy
    p1 = (p0[0] + dx * length, p0[1] + dy * length)

    line = LineString([p0, p1])
    clipped = line.intersection(footprint.polygon)
    if clipped.is_empty or clipped.geom_type != "LineString":
        return
    if clipped.length < 20.0:
        return

    thickness = sample_wall_thickness_px(rng, cfg.image_size, spec, is_exterior=False)
    seg = WallSegment(
        centreline=clipped,
        thickness_px=thickness,
        style=image_style,
        is_exterior=False,
        is_diagonal=True,
        angle_deg=angle_deg,
        no_annotate=True,
    )
    graph.segments.append(seg)
